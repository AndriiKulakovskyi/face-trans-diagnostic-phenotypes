"""OOP prognosis engine on the Gaussian-copula M2 object (M4, reworked).

A clean, config-first OOP engine that reruns the FACE M4 prognosis layer on the **copula** M2 stratification
(the continuum: continuous coordinates + a stable A=4 archetype simplex + a nested K-family tessellation),
parallel to the imperative ``scripts/40-48`` exactly as ``strata_model_oop.py`` sits beside ``scripts/20-26``
and ``measurement_model_oop.py`` beside ``continuous_core``.

Design stance (deliberate):
  * **Wrap, do not reimplement.** Every statistical kernel is reused verbatim from the proven native modules
    — ``glm.fit_glm`` (the errors-in-variables Bayesian GLM), ``compare.delta_elpd`` (held-out LOO-ELPD model
    comparison), and the generic ``reference.py`` design builders (``design_for_rung``, ``coord_eiv_block``,
    ``fixed_block``, ``armB_block``, ``foundation_design``, ``arm_block``, ``modeling_frame``,
    ``outcome_vector``, ``site_index``). This module is orchestration + the copula-source frame + the
    **operative-K selection** + caching/visualization. **No edits to the native M3/M4 modules.**
  * **The operative-K question lives here.** On a continuum K is a granularity convention, not a discovered
    kind-count; *which* encoding — continuous durable coords, A=4 archetypes, or the tessellation at K=2/3/4 —
    adds incremental predictive value beyond DSM-5 + severity + baseline outcome is an *outcome* question,
    answered by ``IncrementalValidator`` (the only generalization vs the native engine: it discovers the
    archetype/K-family columns dynamically and loops the family, instead of the hard-coded A=8/K=4 constants).
  * **Consumer of fixed objects; no re-scoring, no imputation.** V0 predictors are read directly from the
    copula M2 hand-off (``results/face/strata_oop/``); outcomes are the native-scale follow-up scales
    (``data/processed/baseline_v{0,1,2}.parquet``); attrition IPW is reused from the native M3
    (strata-independent, retention-on-V0-covariates). M4 needs no V1/V2 coordinate re-scoring.

Layers (mirroring ``strata_model_oop``):
  * ``PrognosisConfig``     — frozen config + ``with_smoke_defaults`` + ``_config_sig`` cache key + paths.
  * ``PrognosisStage``      — one rung of the deterministic plan (dispatch ``kind``).
  * ``PrognosisData``       — build (``prepare``) + load the copula-sourced V0 analysis frame.
  * ``ReferenceLadder``     — R0->R1->R2->R3y reference bar (wraps ``reference`` + ``glm``).      [stage: reference]
  * ``IncrementalValidator``— R3y + {durable, archetypesA/B, tess K-family, specifics} -> operative-K. [incremental]
  * ``TransdiagnosticH2H``  — foundation / +DSM-5 / +map / +both head-to-head.                    [transdiagnostic]
  * ``EndpointAtlas``       — A=4 archetype prognostic atlas (endpoint rates per archetype x cohort). [endpoints]
  * ``ClinicalValue``       — 5-fold CV AUC, foundation vs +map.                                  [clinical_value]
  * ``RobustnessSweep``     — IPW / reliability / leave-one-cohort-out / permutation ΔELPD.        [robustness]
  * ``PrognosisProjector``  — consolidate: prognosis_summary + per-patient risk (M5 hand-off).    [consolidate]
  * ``PrognosisRunner``     — staged, cached orchestration (manifest = model_version+stage_spec+config_sig).
  * ``PrognosisVisualizer`` — figures.

Engineering note: we do NOT call ``numpyro.set_host_device_count`` (the project gotcha — forcing the host
platform device count slows numpyro markedly); 4-chain NUTS runs sequentially on one host device, which is
fine for these small GLMs. Run heavy plans detached (``scripts/run_job.py``) with ``PYTHONPATH=$PWD/src``.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from face.prognosis import CANON, DURABLE
from face.prognosis import reference as ref
from face.prognosis.compare import coefficient_table, delta_elpd
from face.prognosis.frame import (
    derive_endpoints,
    extract_outcomes,
    load_outcome_config,
)
from face.prognosis.glm import fit_glm

REPO = Path(__file__).resolve().parents[3]
COPULA_STRATA = REPO / "results" / "m2_strata"          # the copula M2 hand-off (input)
COPULA_M3_IPW = REPO / "results" / "m3_temporal" / "attrition"   # copula OOP M3 strata-independent IPW
RESULTS = REPO / "results" / "m4_prognosis"            # parallel output (never touches results/face/m4)
FIGURES = REPO / "docs" / "figures" / "m4_prognosis"
CONFIG = REPO / "configs" / "prognosis_outcomes.yaml"
PROC = REPO / "data" / "processed"
MODEL_VERSION = "m4_prognosis"   # regenerated on the copula A=5 8-factor strata (2026-06-27)
CGI_BASELINE = "cgi_s__V0"                                        # the manifest baseline-CGI-S column
_COORD_METRICS = ("mean", "sd", "hdi_lo", "hdi_hi", "n_obs", "reliability")


# ----------------------------------------------------------------------------------------------------------
# Dynamic encoding discovery — the ONE generalization vs the native engine's hard-coded A=8 / K=4 constants.
# ----------------------------------------------------------------------------------------------------------
def arch_cols(strata: pd.DataFrame) -> list[str]:
    """Arm-A archetype weight columns present in the hand-off (drop-one reference handled at fit time)."""
    return sorted((c for c in strata.columns if c.startswith("arch_w") and not c.endswith("_sd")),
                  key=lambda c: int(c.split("arch_w")[1]))


def archB_cols(strata: pd.DataFrame) -> list[str]:
    """Arm-B (G-residualized) archetype weight columns, if exported."""
    return sorted((c for c in strata.columns if c.startswith("archB_w")),
                  key=lambda c: int(c.split("archB_w")[1]))


def tess_family(strata: pd.DataFrame) -> dict[int, list[str]]:
    """The exported nested K-family: ``{K: [tessfam_k{K}_r0, ...]}`` for every K present. The operative K is
    selected downstream by incremental validity, never by an internal rule."""
    fam: dict[int, list[str]] = {}
    for c in strata.columns:
        if c.startswith("tessfam_k") and "_r" in c:
            K = int(c.split("tessfam_k")[1].split("_r")[0])
            fam.setdefault(K, []).append(c)
    return {K: sorted(cols, key=lambda c: int(c.split("_r")[1])) for K, cols in sorted(fam.items())}


def drop_one(cols: list[str]) -> list[str]:
    """Drop-one-reference for a soft-membership block (simplex sums to 1 -> last column is redundant)."""
    return list(cols[:-1])


def encoding_block(sub: pd.DataFrame, name: str, strata: pd.DataFrame, *, profiles_path):
    """Build one candidate map encoding as ``(glm_kwargs, extra_design | None)`` from a modelling subframe.
    Single source of the encoding logic, shared by the incremental and robustness stages (so the headline is
    stressed with the *identical* block it was scored on). Returns ``None`` if the encoding is unavailable.
      +durable      -> EIV(cognition, metabolic, inflammatory)        (kwargs carry eiv_obs/eiv_sd)
      +specifics8   -> EIV(8 ⊥G specific axes, ceiling)
      +archetypesA  -> z-scored A=A archetype weights (drop-one), full phenotype
      +archetypesB  -> ⊥G Arm-B archetype projection (or exported archB_w*, drop-one)
      +tess_k{K}    -> z-scored K-region tessellation responsibilities (drop-one)
    """
    if name == "+durable":
        dob, dsd, _ = ref.coord_eiv_block(sub, DURABLE)
        return dict(eiv_obs=dob, eiv_sd=dsd), None
    if name == "+specifics8":
        sob, ssd, _ = ref.coord_eiv_block(sub, ref.SPECIFICS)
        return dict(eiv_obs=sob, eiv_sd=ssd), None
    if name == "+archetypesA":
        a = arch_cols(strata)
        if len(a) < 2:
            return None
        m, _ = ref.fixed_block(sub, drop_one(a))
        return {}, m
    if name == "+archetypesB":
        try:
            ab, _ = ref.armB_block(sub, profiles_path=profiles_path)
            return {}, ab
        except (FileNotFoundError, KeyError, ValueError):
            b = archB_cols(strata)
            if len(b) < 2:
                return None
            bb, _ = ref.fixed_block(sub, drop_one(b))
            return {}, bb
    if name.startswith("+tess_k"):
        K = int(name.split("+tess_k")[1])
        fam = tess_family(strata)
        if K not in fam:
            return None
        tk, _ = ref.fixed_block(sub, drop_one(fam[K]))
        return {}, tk
    return None


def expand_encodings(requested: tuple[str, ...], strata: pd.DataFrame) -> list[str]:
    """Expand the configured encoding list, turning ``+tessfamily`` into one ``+tess_k{K}`` per discovered K."""
    out: list[str] = []
    for enc in requested:
        if enc == "+tessfamily":
            out += [f"+tess_k{K}" for K in tess_family(strata)]
        else:
            out.append(enc)
    return out


# ----------------------------------------------------------------------------------------------------------
# Config + stage
# ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PrognosisConfig:
    """Central config: input paths to the copula M2 object + the outcome registry, the parallel output dir,
    and the MCMC knobs. Model-affecting fields enter ``_config_sig`` (cache key)."""
    strata_dir: Path = COPULA_STRATA          # copula M2 hand-off: coordinates/ + consolidate/
    ipw_dir: Path = COPULA_M3_IPW              # copula OOP M3 strata-independent attrition weights
    config_path: Path = CONFIG
    proc_dir: Path = PROC
    output_dir: Path = RESULTS
    figure_dir: Path = FIGURES
    # model-affecting
    horizon: str = "V2"
    encodings: tuple[str, ...] = ("+durable", "+archetypesA", "+archetypesB", "+tessfamily", "+specifics8")
    # the headline encoding(s) the robustness sweep stresses — the operative/predictive winners on the copula
    # map are the archetypes (the durable-trio-alone EIV is not robust here), so stress those, not +durable.
    robust_encodings: tuple[str, ...] = ("+archetypesA", "+archetypesB")
    seed: int = 20260610
    draws: int = 800
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.95
    smoke: bool = False

    def with_smoke_defaults(self) -> PrognosisConfig:
        """Fast wiring config: tiny draws / 2 chains — validates the path, not the science."""
        return replace(self, draws=120, tune=120, chains=2, target_accept=0.9, smoke=True)

    @property
    def coords_dir(self) -> Path:
        return self.strata_dir / "coordinates"

    @property
    def consolidate_dir(self) -> Path:
        return self.strata_dir / "consolidate"

    @property
    def profiles_path(self) -> Path:
        return self.consolidate_dir / "archetype_profiles.csv"

    def fit_kw(self) -> dict:
        return dict(draws=self.draws, tune=self.tune, chains=self.chains, seed=self.seed,
                    target_accept=self.target_accept)

    @property
    def stage_plan(self) -> list[PrognosisStage]:
        return [
            PrognosisStage("frame", "frame"),
            PrognosisStage("reference", "reference"),
            PrognosisStage("incremental", "incremental"),
            PrognosisStage("transdiagnostic", "transdiagnostic"),
            PrognosisStage("endpoints", "endpoints"),
            PrognosisStage("clinical_value", "clinical_value"),
            PrognosisStage("robustness", "robustness"),
            PrognosisStage("consolidate", "consolidate"),
        ]


@dataclass(frozen=True)
class PrognosisStage:
    name: str
    kind: str


def _config_sig(c: PrognosisConfig) -> dict:
    return {"horizon": c.horizon, "encodings": list(c.encodings), "seed": int(c.seed),
            "draws": int(c.draws), "tune": int(c.tune), "chains": int(c.chains),
            "target_accept": float(c.target_accept), "smoke": bool(c.smoke),
            "strata_dir": str(c.strata_dir)}


def _stage_spec(s: PrognosisStage) -> dict:
    return {"name": s.name, "kind": s.kind}


# ----------------------------------------------------------------------------------------------------------
# Data — the copula-sourced V0 analysis frame (no M3 panel; no imputation)
# ----------------------------------------------------------------------------------------------------------
class PrognosisData:
    """Assemble (``prepare``) + load the one-row-per-patient V0 analysis frame **directly from the copula M2
    hand-off** — bypassing the native M3 panel (schemas are identical, so this is a clean repoint, not a
    reformat): baseline coordinate mean/SD (+ reliability) from ``coordinates/coordinates_full.parquet``, the
    strata representations (``arch_*`` / ``tess_*`` / ``tessfam_*`` / ``archB_*``) from
    ``consolidate/patient_strata.parquet``, covariates from ``coordinates/validation_table.parquet``,
    native-scale outcomes (reused ``extract_outcomes``) + derived endpoints, and the reused M3 IPW weights."""

    def __init__(self, config: PrognosisConfig | None = None):
        self.config = config or PrognosisConfig()

    @property
    def _frame(self) -> Path:
        return self.config.output_dir / "frame" / "analysis_frame.parquet"

    def specs(self):
        return load_outcome_config(self.config.config_path).outcomes

    def prepare(self, *, overwrite: bool = False) -> pd.DataFrame:
        if self._frame.exists() and not overwrite:
            return self.load()
        frame = self._assemble()
        self._frame.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self._frame)
        return frame

    def load(self) -> pd.DataFrame:
        if not self._frame.exists():
            raise FileNotFoundError(f"analysis frame not prepared — run PrognosisData.prepare() ({self._frame})")
        return pd.read_parquet(self._frame)

    def _assemble(self) -> pd.DataFrame:
        cfg = self.config
        specs = self.specs()
        coords = pd.read_parquet(cfg.coords_dir / "coordinates_full.parquet").set_index(["cohort", "patient_id"])
        coord_cols = [f"{ax}__{m}" for ax in CANON for m in _COORD_METRICS
                      if f"{ax}__{m}" in coords.columns]
        frame = coords[coord_cols].copy()

        strata = pd.read_parquet(cfg.consolidate_dir / "patient_strata.parquet")
        if not isinstance(strata.index, pd.MultiIndex):
            strata = strata.set_index(["cohort", "patient_id"])
        rep_cols = [c for c in strata.columns
                    if c.startswith(("arch_", "archB_", "tess_", "tessfam_")) and c != "arm"]
        frame = frame.join(strata[rep_cols], how="left")

        vt = pd.read_parquet(cfg.coords_dir / "validation_table.parquet").set_index(["cohort", "patient_id"])
        for c in ("age", "sex", "education_years", "siteid_city", "arm"):
            if c in vt.columns:
                frame[c] = vt[c].reindex(frame.index)
        if "sex" in frame.columns:                       # design_for_rung needs a numeric sex
            frame["sex"] = pd.to_numeric(frame["sex"], errors="coerce")

        outc = extract_outcomes(specs, visits=("V0", "V1", "V2"), proc_dir=cfg.proc_dir)
        frame = frame.join(outc.reindex(frame.index), how="left")
        frame = derive_endpoints(frame, specs, horizon=cfg.horizon)

        ipw_path = cfg.ipw_dir / "ipw_weights.parquet"
        if ipw_path.exists():
            ipw = pd.read_parquet(ipw_path).set_index(["cohort", "patient_id"])
            for c in ("p_retained_V1", "w_retained_V1", "p_retained_V2", "w_retained_V2"):
                if c in ipw.columns:
                    frame[c] = ipw[c].reindex(frame.index)
        else:
            warnings.warn(f"M3 IPW weights absent ({ipw_path}) — robustness IPW check will be skipped",
                          stacklevel=2)
        return frame.reset_index()


# ----------------------------------------------------------------------------------------------------------
# Reference ladder (stage: reference) — the diagnosis + severity + baseline-outcome bar
# ----------------------------------------------------------------------------------------------------------
class ReferenceLadder:
    """R0 (age+sex+site) -> R1 (+DSM-5 arm) -> R2 (+severity) -> R3y (+baseline outcome), per primary outcome,
    on the complete-case sample. Wraps ``reference`` + ``glm.fit_glm``; ΔELPD vs R0 via ``compare.delta_elpd``.
    R3y is the bar the map must beat (stage incremental)."""

    def __init__(self, config: PrognosisConfig | None = None):
        self.config = config or PrognosisConfig()

    def fit_one(self, frame: pd.DataFrame, spec, *, horizon: str, fit_kw: dict) -> dict:
        sev = ref.severity_column(spec, cgi_baseline_col=CGI_BASELINE)
        sub = ref.modeling_frame(frame, spec, horizon=horizon, severity_col=sev)
        y, fam, n_cat = ref.outcome_vector(sub, spec, horizon=horizon)
        grp, ng = ref.site_index(sub)
        fits, designs = {}, {}
        for rung in ref.RUNGS:
            X, names = ref.design_for_rung(sub, spec, rung, severity_col=sev, horizon=horizon)
            fits[rung] = fit_glm(y, X, family=fam, group=grp, n_groups=ng, n_cat=n_cat, **fit_kw)
            designs[rung] = names
        cmp = delta_elpd(fits, reference="R0").assign(outcome=spec.name, n=len(sub), severity=sev)
        coef = coefficient_table(fits["R3y"], names=designs["R3y"]).assign(outcome=spec.name)
        return {"cmp": cmp, "coef": coef, "sev": sev, "n": len(sub)}


# ----------------------------------------------------------------------------------------------------------
# Incremental validity (stage: incremental) — THE operative-K selector
# ----------------------------------------------------------------------------------------------------------
class IncrementalValidator:
    """On top of R3y, add each candidate encoding and rank by held-out ΔELPD: continuous durable coords (EIV),
    A=4 archetypes (Arm A and the ⊥G Arm B), the **whole tessellation K-family** (K=2/3/4), and the
    8-specifics ceiling. The operative K is whichever tessellation granularity (if any) is predictive and best
    — or the verdict that the continuous/archetype representation wins and no hard K is needed. Wraps
    ``reference`` builders + ``glm.fit_glm`` + ``compare.delta_elpd``; the family loop + dynamic discovery are
    the only generalization vs the native engine."""

    def __init__(self, config: PrognosisConfig | None = None):
        self.config = config or PrognosisConfig()

    def _encodings(self, sub, strata) -> list[tuple[str, dict, np.ndarray | None]]:
        """(name, glm-kwargs, extra-design) for each configured encoding present in the hand-off, via the
        shared ``encoding_block`` builder (``+tessfamily`` expands to one entry per discovered K)."""
        out: list[tuple[str, dict, np.ndarray | None]] = []
        for name in expand_encodings(self.config.encodings, strata):
            blk = encoding_block(sub, name, strata, profiles_path=self.config.profiles_path)
            if blk is not None:
                out.append((name, blk[0], blk[1]))
        return out

    def fit_one(self, frame: pd.DataFrame, spec, *, horizon: str, fit_kw: dict) -> dict:
        sev = ref.severity_column(spec, cgi_baseline_col=CGI_BASELINE)
        sub = ref.modeling_frame(frame, spec, horizon=horizon, severity_col=sev)
        y, fam, n_cat = ref.outcome_vector(sub, spec, horizon=horizon)
        grp, ng = ref.site_index(sub)
        Xr, _ = ref.design_for_rung(sub, spec, "R3y", severity_col=sev, horizon=horizon)
        base = dict(family=fam, group=grp, n_groups=ng, n_cat=n_cat, **fit_kw)
        fits = {"R3y": fit_glm(y, Xr, **base)}
        durable_fit = None
        for name, kw, extra in self._encodings(sub, frame):
            X = Xr if extra is None else np.column_stack([Xr, extra])
            fits[name] = fit_glm(y, X, **base, **kw)
            if name == "+durable":
                durable_fit = fits[name]
        cmp = delta_elpd(fits, reference="R3y").assign(outcome=spec.name, n=len(sub))
        coef = None
        if durable_fit is not None:
            c = durable_fit["coef"]
            rows = c[c.term.str.startswith("beta_eiv")].reset_index(drop=True)
            if len(rows) == len(DURABLE):
                coef = rows.assign(outcome=spec.name, axis=list(DURABLE), severity=sev)[
                    ["outcome", "axis", "severity", "mean", "sd", "eti_lo", "eti_hi", "p_direction"]]
        return {"cmp": cmp, "coef": coef, "n": len(sub)}

    @staticmethod
    def operative_k(comparison: pd.DataFrame) -> dict:
        """Pick the operative tessellation K from the cross-outcome incremental table: the K whose mean ΔELPD
        across primary outcomes is highest AND predictive — but only if it beats the best continuous/archetype
        encoding; otherwise the honest verdict is that no hard K is needed (the continuum wins)."""
        m = comparison.copy()
        agg = (m.groupby("model")
               .agg(mean_delta=("d_elpd_vs_ref", "mean"),
                    any_predictive=("verdict", lambda v: (v == "predictive").any()))
               .reset_index())
        tess = agg[agg.model.str.startswith("+tess_k")]
        cont = agg[agg.model.isin(["+durable", "+archetypesA", "+archetypesB", "+specifics8"])]
        best_tess = tess.sort_values("mean_delta", ascending=False).head(1)
        best_cont = cont.sort_values("mean_delta", ascending=False).head(1)
        out = {"family_K": sorted({int(x.split("+tess_k")[1]) for x in tess.model}),
               "best_tessellation": None, "best_continuous": None, "operative_K": None,
               "verdict": "no encoding predictive"}
        if len(best_tess):
            r = best_tess.iloc[0]
            out["best_tessellation"] = {"model": r.model, "mean_delta": round(float(r.mean_delta), 2),
                                        "predictive": bool(r.any_predictive)}
        if len(best_cont):
            r = best_cont.iloc[0]
            out["best_continuous"] = {"model": r.model, "mean_delta": round(float(r.mean_delta), 2),
                                      "predictive": bool(r.any_predictive)}
        bt, bc = out["best_tessellation"], out["best_continuous"]
        if bt and bt["predictive"] and (not bc or bt["mean_delta"] >= bc["mean_delta"]):
            out["operative_K"] = int(bt["model"].split("+tess_k")[1])
            out["verdict"] = (f"operative K = {out['operative_K']} (tessellation predictive and best)")
        elif bc and bc["predictive"]:
            out["operative_K"] = None
            out["verdict"] = (f"no hard K — the continuous/archetype encoding ({bc['model']}) wins; "
                              "the tessellation adds nothing beyond it")
        return out


# ----------------------------------------------------------------------------------------------------------
# Transdiagnostic head-to-head vs DSM-5 (stage: transdiagnostic)
# ----------------------------------------------------------------------------------------------------------
class TransdiagnosticH2H:
    """foundation (age+sex+severity+baseline) / +DSM-5 arm / +map / +both — does the map add beyond DSM-5 and
    vice-versa? Wraps ``foundation_design`` + ``arm_block`` + the operative map block (default: A=4 Arm-B
    archetypes; falls back to durable coords)."""

    def __init__(self, config: PrognosisConfig | None = None):
        self.config = config or PrognosisConfig()

    def _map_block(self, sub, frame):
        try:
            archB, _ = ref.armB_block(sub, profiles_path=self.config.profiles_path)
            return archB
        except (FileNotFoundError, KeyError, ValueError):
            dob, dsd, _ = ref.coord_eiv_block(sub, DURABLE)
            return dob   # point-estimate fallback (kept simple for the H2H descriptive comparison)

    def fit_one(self, frame, spec, *, horizon, fit_kw):
        sev = ref.severity_column(spec, cgi_baseline_col=CGI_BASELINE)
        sub = ref.modeling_frame(frame, spec, horizon=horizon, severity_col=sev)
        y, fam, n_cat = ref.outcome_vector(sub, spec, horizon=horizon)
        grp, ng = ref.site_index(sub)
        Xf, _ = ref.foundation_design(sub, spec, severity_col=sev, horizon=horizon)
        arm, _ = ref.arm_block(sub)
        mp = self._map_block(sub, frame)
        base = dict(family=fam, group=grp, n_groups=ng, n_cat=n_cat, **fit_kw)
        fits = {
            "foundation": fit_glm(y, Xf, **base),
            "+dsm5": fit_glm(y, np.column_stack([Xf, arm]), **base),
            "+map": fit_glm(y, np.column_stack([Xf, mp]), **base),
            "+both": fit_glm(y, np.column_stack([Xf, arm, mp]), **base),
        }
        return delta_elpd(fits, reference="foundation").assign(outcome=spec.name, n=len(sub))


# ----------------------------------------------------------------------------------------------------------
# Archetype prognostic atlas (stage: endpoints) — descriptive, A=4, no model
# ----------------------------------------------------------------------------------------------------------
class EndpointAtlas:
    """Per dominant archetype (A discovered) x cohort: endpoint rate (Wilson CI) and mean horizon outcome —
    the clinician-facing prognostic atlas. Descriptive only (no imputation)."""

    def __init__(self, config: PrognosisConfig | None = None):
        self.config = config or PrognosisConfig()

    @staticmethod
    def _wilson(k, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan, np.nan)
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (p, max(0.0, c - h), min(1.0, c + h))

    def build(self, frame, specs, *, horizon):
        rows = []
        for spec in specs:
            rem = f"{spec.name}__remission_{horizon}"
            yt = f"{spec.name}__{horizon}"
            if "arch_dominant" not in frame.columns:
                continue
            for (arch, cohort), g in frame.groupby(["arch_dominant", "cohort"]):
                nm = g["arch_dominant_name"].iloc[0] if "arch_dominant_name" in g else str(arch)
                row = {"outcome": spec.name, "archetype": int(arch), "archetype_name": nm,
                       "cohort": cohort, "n": int(len(g))}
                if rem in g.columns and g[rem].notna().any():
                    v = g[rem].dropna()
                    p, lo, hi = self._wilson(int(v.sum()), int(len(v)))
                    row.update({"remission_rate": round(p, 3), "rem_lo": round(lo, 3), "rem_hi": round(hi, 3),
                                "n_rem": int(len(v))})
                if yt in g.columns and g[yt].notna().any():
                    row["mean_outcome"] = round(float(g[yt].mean()), 2)
                rows.append(row)
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# Clinical value (stage: clinical_value) — 5-fold CV AUC, foundation vs +map
# ----------------------------------------------------------------------------------------------------------
class ClinicalValue:
    """Decision-relevant currency: for each binary remission endpoint, 5-fold cross-validated AUC of a
    foundation logistic (age+sex+severity+baseline) vs +map (A=4 Arm-A archetype weights), with the ΔAUC. A
    frequentist CV check (not Bayesian) — the deployable-classifier read."""

    def __init__(self, config: PrognosisConfig | None = None):
        self.config = config or PrognosisConfig()

    def build(self, frame, specs, *, horizon):

        rows = []
        a_cols = arch_cols(frame)
        for spec in specs:
            rem = f"{spec.name}__remission_{horizon}"
            sevc = ref.severity_column(spec, cgi_baseline_col=CGI_BASELINE)
            base_cols = [c for c in ("age", "sex", sevc, f"{spec.name}__V0") if c in frame.columns]
            if rem not in frame.columns or not base_cols:
                continue
            sub = frame.dropna(subset=[rem, *base_cols]).copy()
            y = sub[rem].to_numpy(int)
            if len(np.unique(y)) < 2 or len(sub) < 60:
                continue
            Xb = sub[base_cols].to_numpy("float64")
            Xm = np.column_stack([Xb, sub[drop_one(a_cols)].to_numpy("float64")]) if len(a_cols) >= 2 else Xb
            auc_b, auc_m = self._cv_auc(Xb, y), self._cv_auc(Xm, y)
            rows.append({"outcome": spec.name, "endpoint": f"remission_{horizon}", "n": int(len(sub)),
                         "prevalence": round(float(y.mean()), 3), "auc_foundation": round(auc_b, 3),
                         "auc_plus_map": round(auc_m, 3), "delta_auc": round(auc_m - auc_b, 3)})
        return pd.DataFrame(rows)

    @staticmethod
    def _cv_auc(X, y, *, seed=20260610):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        preds = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            preds[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        return float(roc_auc_score(y, preds))


# ----------------------------------------------------------------------------------------------------------
# Robustness (stage: robustness) — IPW / leave-one-cohort-out / permutation on the headline +durable model
# ----------------------------------------------------------------------------------------------------------
class RobustnessSweep:
    """Stress the **headline (operative) encoding(s)** per primary outcome — by default the archetypes (the
    predictive winners on the copula map; the durable-trio-alone EIV is not robust here) — under IPW-weighted
    refit (attrition), leave-one-cohort-out, and a label-permutation null. Reports ΔELPD vs R3y for each.
    Wraps the same kernels + the shared ``encoding_block`` so each encoding is stressed with the identical
    block it was scored on."""

    def __init__(self, config: PrognosisConfig | None = None):
        self.config = config or PrognosisConfig()

    def _encoding_delta(self, sub, spec, name, *, sev, horizon, fit_kw, weights=None, permute=False, seed=0):
        if weights is not None:                         # IPW: keep only positively-weighted (retained) rows —
            w = np.asarray(weights, dtype="float64")    # the stabilized attrition weight is 0 for non-retained
            keep = w > 0                                # (a row with V0&V2 outcome but <3 visits gets w=0)
            sub, weights = sub[keep], w[keep]
        blk = encoding_block(sub, name, sub, profiles_path=self.config.profiles_path)
        if blk is None:
            return None
        kw, extra = blk
        y, fam, n_cat = ref.outcome_vector(sub, spec, horizon=horizon)
        if permute:
            y = np.random.default_rng(seed).permutation(y)
        grp, ng = ref.site_index(sub)
        Xr, _ = ref.design_for_rung(sub, spec, "R3y", severity_col=sev, horizon=horizon)
        Xfull = Xr if extra is None else np.column_stack([Xr, extra])
        base = dict(family=fam, group=grp, n_groups=ng, n_cat=n_cat, **fit_kw)
        wk = {} if weights is None else dict(weights=weights)
        fits = {"R3y": fit_glm(y, Xr, **base, **wk), name: fit_glm(y, Xfull, **base, **kw, **wk)}
        d = delta_elpd(fits, reference="R3y")
        return d[d.model == name].iloc[0]

    def build(self, frame, specs, *, horizon, fit_kw):
        rows = []
        for spec in specs:
            sev = ref.severity_column(spec, cgi_baseline_col=CGI_BASELINE)
            sub = ref.modeling_frame(frame, spec, horizon=horizon, severity_col=sev)
            pseed = self.config.seed + sum(ord(c) for c in spec.name)      # deterministic null seed
            wcol = f"w_retained_{horizon}"
            for name in self.config.robust_encodings:
                if encoding_block(sub, name, sub, profiles_path=self.config.profiles_path) is None:
                    continue
                checks = [("base", {}), ("permutation", dict(permute=True, seed=pseed))]
                if wcol in sub.columns and sub[wcol].notna().any():
                    checks.append(("ipw", dict(weights=sub[wcol].fillna(1.0).to_numpy("float64"))))
                for label, kw in checks:
                    r = self._encoding_delta(sub, spec, name, sev=sev, horizon=horizon, fit_kw=fit_kw, **kw)
                    rows.append({"outcome": spec.name, "encoding": name, "check": label, "n": int(len(sub)),
                                 "d_elpd_vs_ref": r["d_elpd_vs_ref"], "se_d_elpd": r["se_d_elpd"],
                                 "verdict": r["verdict"]})
                for cohort in sorted(sub["cohort"].unique()):
                    loco = sub[sub.cohort != cohort]
                    if len(loco) < 80 or loco[f"{spec.name}__{horizon}"].notna().sum() < 60:
                        continue
                    r = self._encoding_delta(loco, spec, name, sev=sev, horizon=horizon, fit_kw=fit_kw)
                    rows.append({"outcome": spec.name, "encoding": name, "check": f"drop_{cohort}",
                                 "n": int(len(loco)), "d_elpd_vs_ref": r["d_elpd_vs_ref"],
                                 "se_d_elpd": r["se_d_elpd"], "verdict": r["verdict"]})
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# Consolidate (stage: consolidate) — the M5 hand-off + the operative-K verdict
# ----------------------------------------------------------------------------------------------------------
class PrognosisProjector:
    """Lock the verdicts: ``prognosis_summary.csv`` (per-outcome incremental ΔELPD + the operative-K verdict)
    and ``prognosis_patient_risk.parquet`` (per-patient archetype membership + the binary endpoints), the
    M5 hand-off."""

    def __init__(self, config: PrognosisConfig | None = None):
        self.config = config or PrognosisConfig()

    def summary(self, incremental: pd.DataFrame, operative: dict) -> pd.DataFrame:
        s = incremental[incremental.model != "R3y"][
            ["outcome", "model", "d_elpd_vs_ref", "se_d_elpd", "verdict", "n"]].copy()
        s["operative_K"] = json.dumps(operative.get("operative_K"))
        s["operative_verdict"] = operative.get("verdict")
        return s

    def patient_risk(self, frame: pd.DataFrame, specs, *, horizon: str) -> pd.DataFrame:
        keep = ["cohort", "patient_id", "arm"]
        keep += [c for c in frame.columns if c.startswith(("arch_w", "arch_dominant", "archB_w"))]
        keep += [c for c in frame.columns
                 if c.endswith((f"__remission_{horizon}", f"__response_{horizon}", f"__{horizon}"))]
        keep = [c for c in dict.fromkeys(keep) if c in frame.columns]
        return frame[keep].copy()


# ----------------------------------------------------------------------------------------------------------
# Runner — staged + cached
# ----------------------------------------------------------------------------------------------------------
class PrognosisRunner:
    """Walk the deterministic plan (frame -> reference -> incremental -> transdiagnostic -> endpoints ->
    clinical_value -> robustness -> consolidate), caching each stage to ``output_dir/<stage>/`` and reusing it
    when ``MODEL_VERSION`` + ``stage_spec`` + ``config_sig`` all match. The accumulated ``state`` carries live
    objects through one ``run_plan`` call."""

    def __init__(self, config: PrognosisConfig | None = None):
        self.config = config or PrognosisConfig()
        self.data = PrognosisData(self.config)
        self.reference = ReferenceLadder(self.config)
        self.incremental = IncrementalValidator(self.config)
        self.h2h = TransdiagnosticH2H(self.config)
        self.atlas = EndpointAtlas(self.config)
        self.clinical = ClinicalValue(self.config)
        self.robust = RobustnessSweep(self.config)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    # -- cache plumbing --
    def _cache_ok(self, out: Path, stage: PrognosisStage) -> bool:
        mf = out / "manifest.json"
        if not mf.exists():
            return False
        m = json.loads(mf.read_text())
        return (m.get("model_version") == MODEL_VERSION and m.get("stage_spec") == _stage_spec(stage)
                and m.get("config_sig") == _config_sig(self.config))

    def _manifest(self, out: Path, stage: PrognosisStage, summary: dict) -> None:
        (out / "manifest.json").write_text(json.dumps(
            {"model_version": MODEL_VERSION, "stage": stage.name, "stage_spec": _stage_spec(stage),
             "config_sig": _config_sig(self.config), "summary": summary}, indent=2, default=str))

    def run_stage(self, stage: PrognosisStage, state: dict, *, overwrite: bool = False) -> dict:
        cfg = self.config
        out = cfg.output_dir / stage.name
        out.mkdir(parents=True, exist_ok=True)
        cached = self._cache_ok(out, stage) and not overwrite
        horizon, fit_kw = cfg.horizon, cfg.fit_kw()
        specs_all = self.data.specs()
        primary = [s for s in specs_all if s.role == "primary"]

        if stage.kind == "frame":
            frame = self.data.prepare(overwrite=overwrite)
            state["frame"] = frame
            self._manifest(out, stage, {"rows": int(len(frame)), "cols": int(frame.shape[1])})
            return state

        frame = state.get("frame")
        if frame is None:
            frame = self.data.load(); state["frame"] = frame

        if stage.kind == "reference":
            if cached:
                state["reference"] = pd.read_csv(out / "elpd_reference.csv")
            else:
                res = [self.reference.fit_one(frame, s, horizon=horizon, fit_kw=fit_kw) for s in primary]
                comp = pd.concat([r["cmp"] for r in res], ignore_index=True)
                comp.to_csv(out / "elpd_reference.csv", index=False)
                pd.concat([r["coef"] for r in res], ignore_index=True).to_csv(out / "coef_reference.csv", index=False)
                state["reference"] = comp
                self._manifest(out, stage, {"outcomes": [s.name for s in primary]})
            return state

        if stage.kind == "incremental":
            if cached:
                state["incremental"] = pd.read_csv(out / "incremental_comparison.csv")
                state["operative_k"] = json.loads((out / "operative_k.json").read_text())
            else:
                res = [self.incremental.fit_one(frame, s, horizon=horizon, fit_kw=fit_kw) for s in primary]
                comp = pd.concat([r["cmp"] for r in res], ignore_index=True)
                comp.to_csv(out / "incremental_comparison.csv", index=False)
                coefs = [r["coef"] for r in res if r["coef"] is not None]
                if coefs:
                    pd.concat(coefs, ignore_index=True).to_csv(out / "coef_durable.csv", index=False)
                operative = IncrementalValidator.operative_k(comp)
                (out / "operative_k.json").write_text(json.dumps(operative, indent=2, default=str))
                state["incremental"], state["operative_k"] = comp, operative
                self._manifest(out, stage, {"operative_K": operative.get("operative_K"),
                                            "verdict": operative.get("verdict")})
            return state

        if stage.kind == "transdiagnostic":
            if cached:
                state["transdiagnostic"] = pd.read_csv(out / "h2h_dsm5.csv")
            else:
                comp = pd.concat([self.h2h.fit_one(frame, s, horizon=horizon, fit_kw=fit_kw) for s in primary],
                                 ignore_index=True)
                comp.to_csv(out / "h2h_dsm5.csv", index=False)
                state["transdiagnostic"] = comp
                self._manifest(out, stage, {"outcomes": [s.name for s in primary]})
            return state

        if stage.kind == "endpoints":
            if cached:
                state["endpoints"] = pd.read_csv(out / "archetype_atlas.csv")
            else:
                atlas = self.atlas.build(frame, primary, horizon=horizon)
                atlas.to_csv(out / "archetype_atlas.csv", index=False)
                state["endpoints"] = atlas
                self._manifest(out, stage, {"rows": int(len(atlas))})
            return state

        if stage.kind == "clinical_value":
            if cached:
                state["clinical_value"] = pd.read_csv(out / "clinical_value.csv")
            else:
                cv = self.clinical.build(frame, primary, horizon=horizon)
                cv.to_csv(out / "clinical_value.csv", index=False)
                state["clinical_value"] = cv
                self._manifest(out, stage, {"rows": int(len(cv))})
            return state

        if stage.kind == "robustness":
            if cached:
                state["robustness"] = pd.read_csv(out / "robustness.csv")
            else:
                rob = self.robust.build(frame, primary, horizon=horizon, fit_kw=fit_kw)
                rob.to_csv(out / "robustness.csv", index=False)
                state["robustness"] = rob
                self._manifest(out, stage, {"rows": int(len(rob))})
            return state

        if stage.kind == "consolidate":
            proj = PrognosisProjector(cfg)
            inc = state.get("incremental")
            if inc is None:
                inc = pd.read_csv(cfg.output_dir / "incremental" / "incremental_comparison.csv")
            operative = state.get("operative_k") or json.loads(
                (cfg.output_dir / "incremental" / "operative_k.json").read_text())
            summ = proj.summary(inc, operative)
            summ.to_csv(out / "prognosis_summary.csv", index=False)
            risk = proj.patient_risk(frame, primary, horizon=horizon)
            risk.to_parquet(out / "prognosis_patient_risk.parquet")
            state["prognosis_summary"], state["operative_k"] = summ, operative
            self._manifest(out, stage, {"operative_K": operative.get("operative_K"),
                                        "verdict": operative.get("verdict"), "risk_rows": int(len(risk))})
            return state

        raise ValueError(f"unknown stage kind: {stage.kind}")

    def run_plan(self, *, stop_after: str | None = None, overwrite: bool = False) -> dict:
        state: dict = {}
        for stage in self.config.stage_plan:
            state = self.run_stage(stage, state, overwrite=overwrite)
            if stop_after and stage.name == stop_after:
                break
        return state

    def load_state(self) -> dict:
        """Reconstruct state from cached stage artifacts (no recompute) — for the figures script / notebook."""
        out = self.config.output_dir
        state: dict = {}
        fp = out / "frame" / "analysis_frame.parquet"
        if fp.exists():
            state["frame"] = pd.read_parquet(fp)
        for stage, fname in [("reference", "elpd_reference.csv"), ("incremental", "incremental_comparison.csv"),
                             ("transdiagnostic", "h2h_dsm5.csv"), ("endpoints", "archetype_atlas.csv"),
                             ("clinical_value", "clinical_value.csv"), ("robustness", "robustness.csv")]:
            p = out / stage / fname
            if p.exists():
                state[stage] = pd.read_csv(p)
        opk = out / "incremental" / "operative_k.json"
        if opk.exists():
            state["operative_k"] = json.loads(opk.read_text())
        ps = out / "consolidate" / "prognosis_summary.csv"
        if ps.exists():
            state["prognosis_summary"] = pd.read_csv(ps)
        return state


# ----------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------
class PrognosisVisualizer:
    """The headline figures: the reference ladder, the incremental added-value bars (with the K-family), and
    the archetype atlas. Reads cached stage CSVs (load mode) or live state."""

    def __init__(self, config: PrognosisConfig | None = None):
        self.config = config or PrognosisConfig()
        self.config.figure_dir.mkdir(parents=True, exist_ok=True)

    def _mpl(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt

    def incremental_bars(self, comparison: pd.DataFrame, filename: str = "incremental_added_value.png") -> Path:
        plt = self._mpl()
        outcomes = list(dict.fromkeys(comparison["outcome"]))
        fig, ax = plt.subplots(1, len(outcomes), figsize=(6.2 * len(outcomes), 4.6), squeeze=False)
        for j, name in enumerate(outcomes):
            sub = comparison[(comparison.outcome == name) & (comparison.model != "R3y")]
            a = ax[0][j]
            colors = ["#2c7fb8" if (v - 2 * s) > 0 else ("#888" if (v + 2 * s) > 0 else "#d73027")
                      for v, s in zip(sub["d_elpd_vs_ref"], sub["se_d_elpd"], strict=False)]
            a.bar(range(len(sub)), sub["d_elpd_vs_ref"], color=colors)
            a.errorbar(range(len(sub)), sub["d_elpd_vs_ref"], yerr=2 * sub["se_d_elpd"], fmt="none",
                       ecolor="#222", capsize=3)
            a.axhline(0, color="k", lw=0.8)
            a.set_xticks(range(len(sub)))
            a.set_xticklabels(sub["model"], rotation=25, ha="right", fontsize=8)
            a.set_title(f"{name}: ΔELPD vs R3y (N={int(sub['n'].iloc[0])})")
            a.set_ylabel("ΔELPD (held-out, ↑ better)")
            a.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = self.config.figure_dir / filename
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path
