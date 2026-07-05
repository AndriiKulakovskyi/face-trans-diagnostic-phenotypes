"""OOP treatment-moderation engine on the Gaussian-copula objects (M5, reworked).

Parallel OOP engine that reruns the FACE M5 treatment causal pipeline on the copula map + the A=5 copula
archetypes, mirroring `strata.engine.py` / `prognosis.engine.py` / `temporal.engine.py` and
**wrapping the proven kernels** (`treatment.{medications,endpoints,propensity,moderation}`,
`prognosis.{glm,reference,compare}`) — **no edits to native M5** (`scripts/50-57`).

The causal question is the strongest "actionable" test: does the map **moderate** treatment response? The
pipeline is the identification-first arc: overlap gate (propensity common support) → doubly-robust EIV
moderation (treat × map-axis interaction) + E-value → confounder-survival (does the copula-M4 carrier survive
treatment adjustment?) → tolerability (side-effects × map). Per the build decision, moderation interacts
treatment with **both** the durable trio (native parity) **and** the A=5 archetypes (the copula-M4 carrier).

Nothing here is map-specific except the *inputs*: the analysis frame's predictor side is the copula
m4_prognosis frame (copula coords + A=5 archetypes + covariates + outcomes + IPW); treatment exposures are
the map-independent harmonized drug-class flags (reused `build_treatment_exposures`). Honest expectation: the
observational-TAU boundary replays (same data/confounding); the contribution is parity on the better map + the
archetype-moderation view + the copula-carrier survival check. Output under `results/m5_treatment/`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from face.prognosis import DURABLE
from face.prognosis.compare import delta_elpd
from face.prognosis.frame import OutcomeSpec
from face.prognosis.glm import fit_glm
from face.prognosis.reference import arm_block, coord_eiv_block, foundation_design, site_index
from face.treatment.endpoints import build_endpoints
from face.treatment.medications import build_treatment_exposures
from face.treatment.moderation import e_value, mde, sd_from_eti
from face.treatment.propensity import (
    QUESTIONS,
    confounder_matrix,
    define_exposure,
    overlap,
    propensity_score,
    smd,
    stabilized_iptw,
)

REPO = Path(__file__).resolve().parents[3]
PROGNOSIS_DIR = REPO / "results" / "m4_prognosis"     # the copula "M4 predictor side"
RESULTS = REPO / "results" / "m5_treatment"
FIGURES = REPO / "docs" / "figures" / "m5_treatment"
DATA = REPO / "data"
MODEL_VERSION = "m5_treatment"   # bounds-and-defends; regenerated on copula A=5 8-factor strata (2026-06-27)
SEV = "overall_severity__mean"
CGI_BASELINE = "cgi_s__V0"
MODES = ("active_comparator", "on_off")
# the native M5 primary contrasts (active-comparator; clozapine on/off since active-comparator is channeled)
PRIMARY_RUNS = (("lithium_bp", "active_comparator"), ("antipsychotic_bp", "active_comparator"),
                ("clozapine_sz", "on_off"))
TREAT_COLS = ["on_antipsychotic", "on_antidepressant", "on_mood_stabilizer", "on_lithium", "on_anxiolytic"]
LOGIT_TO_D = 0.5513                                             # logistic coef -> standardized d (E-value)
SPEC_EGF = OutcomeSpec(name="egf", label="egf", source_var="egf", family="gaussian",
                       direction="higher_better", cohort_scope=("bp", "sz", "dr"), severity_anchor="G",
                       role="primary")
SPEC_CGI = OutcomeSpec(name="cgi_s", label="cgi", source_var="cgi01", family="gaussian",
                       direction="lower_better", cohort_scope=("bp", "sz"), severity_anchor="G", role="primary")
OUTCOMES = [("functioning", "egf__V2", "gaussian", SPEC_EGF),
            ("cgi_response", "ep_response", "bernoulli", SPEC_CGI)]


# ----------------------------------------------------------------------------------------------------------
# The archetype EIV block (the A=5 analogue of reference.coord_eiv_block) — the only new kernel
# ----------------------------------------------------------------------------------------------------------
def arch_cols(frame: pd.DataFrame) -> list[str]:
    return sorted((c for c in frame.columns if c.startswith("arch_w") and not c.endswith("_sd")),
                  key=lambda c: int(c.split("arch_w")[1]))


def arch_eiv_block(sub: pd.DataFrame, *, drop_one: bool = True):
    """Errors-in-variables block for the A archetype weights (the copula-M4 carrier), so the treat×archetype
    moderation runs through the same ``fit_glm(eiv_obs=, eiv_sd=, eiv_interact=treat)`` machinery as the
    durable-axis route. Standardize each weight by its population SD; carry ``arch_w{k}_sd`` (the membership
    uncertainty) on the same scale. Drop one (the simplex sums to 1) to avoid collinearity. Returns
    (obs [N, A-1], sd [N, A-1], names)."""
    cols = arch_cols(sub)
    cols = cols[:-1] if drop_one and len(cols) > 1 else cols
    obs, sd = [], []
    for c in cols:
        m = sub[c].to_numpy("float64")
        s = (sub[f"{c}_sd"].to_numpy("float64") if f"{c}_sd" in sub.columns else np.zeros_like(m))
        psd = m.std() or 1.0
        obs.append((m - m.mean()) / psd)
        sd.append(s / psd)
    return np.column_stack(obs), np.column_stack(sd), list(cols)


def _arch_fixed_block(sub: pd.DataFrame, *, drop_one: bool = True):
    """Standardized archetype-weight columns as FIXED predictors (drop-one). Archetype memberships are
    deterministic point values (their `arch_w_sd` is ≈0 for confident patients), so they enter moderation as
    a fixed `treat × arch_w` interaction, NOT through the EIV machinery (which degenerates on ~0 SD)."""
    cols = arch_cols(sub)
    cols = cols[:-1] if drop_one and len(cols) > 1 else cols
    mat = np.column_stack([(sub[c].to_numpy("float64") - sub[c].mean()) / (sub[c].std() or 1.0) for c in cols])
    return mat, list(cols)


def _map_block(sub: pd.DataFrame, rep: str):
    """The moderator map block for a representation. Returns a tagged tuple:
      durable    -> ("eiv", obs, sd>0, names)   — genuine coordinate posterior SD (errors-in-variables)
      archetypes -> ("fixed", mat, names)       — point memberships (fixed interaction, no EIV)
    """
    if rep == "durable":
        obs, sd, names = coord_eiv_block(sub, DURABLE)
        return ("eiv", obs, np.maximum(np.asarray(sd, "float64"), 1e-6), names)
    mat, names = _arch_fixed_block(sub)
    return ("fixed", mat, names)


def _safe_delta_elpd(fit0, fit1) -> tuple[float, float]:
    """Held-out ΔELPD (moderation vs no-interaction), best-effort. PSIS-LOO degenerates on the IPTW-weighted
    log-likelihood (the weight-scaled likelihood gives some obs a constant importance tail), so on failure we
    return NaN and let the verdict rest on the per-axis interaction HDI + the E-value."""
    try:
        cmp = delta_elpd({"no_interaction": fit0, "moderation": fit1}, reference="no_interaction")
        m = cmp[cmp.model == "moderation"].iloc[0]
        return float(m["d_elpd_vs_ref"]), float(m["se_d_elpd"])
    except (ValueError, KeyError, ZeroDivisionError, FloatingPointError):
        return float("nan"), float("nan")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate k/n (the atlas CIs; stable at the cell extremes)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


def _verdict(diag: dict, smd_after: float) -> str:
    """The 55 overlap gate: estimable / channeled / non-estimable."""
    if min(diag["n_treated"], diag["n_control"]) < 30:
        return "non-estimable (arm < 30)"
    if diag["frac_in_support"] < 0.5:
        return "channeled (poor overlap)"
    if smd_after > 0.25:
        return "estimable — residual imbalance (caution)"
    return "estimable"


# ----------------------------------------------------------------------------------------------------------
# Config + stage
# ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TreatmentConfig:
    prognosis_dir: Path = PROGNOSIS_DIR
    data_dir: Path = DATA
    output_dir: Path = RESULTS
    figure_dir: Path = FIGURES
    horizon: str = "V2"
    moderation_reps: tuple[str, ...] = ("durable", "archetypes")
    seed: int = 20260611
    draws: int = 700
    tune: int = 700
    chains: int = 4
    smoke: bool = False

    def with_smoke_defaults(self) -> TreatmentConfig:
        return replace(self, draws=120, tune=120, chains=2, smoke=True)

    def fit_kw(self) -> dict:
        return dict(draws=self.draws, tune=self.tune, chains=self.chains, seed=self.seed)

    @property
    def frame_path(self) -> Path:
        return self.output_dir / "frame" / "analysis_frame.parquet"

    @property
    def exposures_path(self) -> Path:
        return self.output_dir / "exposures" / "treatment_exposures.parquet"

    @property
    def stage_plan(self) -> list[TreatmentStage]:
        return [TreatmentStage("exposures", "exposures"), TreatmentStage("frame", "frame"),
                TreatmentStage("propensity", "propensity"), TreatmentStage("moderation", "moderation"),
                TreatmentStage("confounder", "confounder"), TreatmentStage("tolerability", "tolerability"),
                TreatmentStage("heterogeneity", "heterogeneity"), TreatmentStage("atlas", "atlas"),
                TreatmentStage("consolidate", "consolidate")]


@dataclass(frozen=True)
class TreatmentStage:
    name: str
    kind: str


def _config_sig(c: TreatmentConfig) -> dict:
    return {"horizon": c.horizon, "moderation_reps": list(c.moderation_reps), "seed": int(c.seed),
            "draws": int(c.draws), "tune": int(c.tune), "chains": int(c.chains), "smoke": bool(c.smoke),
            "prognosis_dir": str(c.prognosis_dir)}


def _stage_spec(s: TreatmentStage) -> dict:
    return {"name": s.name, "kind": s.kind}


# ----------------------------------------------------------------------------------------------------------
# Data — the copula M5 frame (m4_prognosis predictor side + treatment-response endpoints) + exposures
# ----------------------------------------------------------------------------------------------------------
class TreatmentData:
    """Assemble the copula M5 analysis frame (the m4_prognosis predictor side + the `ep_*` treatment-response
    endpoints) and the map-independent harmonized treatment exposures. No re-scoring, no imputation."""

    def __init__(self, config: TreatmentConfig | None = None):
        self.config = config or TreatmentConfig()

    def build_exposures(self) -> pd.DataFrame:
        exp = build_treatment_exposures(self.config.data_dir, visit="V0")     # map-independent (raw cohort CSVs)
        self.config.exposures_path.parent.mkdir(parents=True, exist_ok=True)
        exp.to_parquet(self.config.exposures_path, index=False)
        return exp

    def build_frame(self) -> pd.DataFrame:
        from face.treatment.frame import _response_signals  # reuse the raw-signal extractor
        pred = pd.read_parquet(self.config.prognosis_dir / "frame" / "analysis_frame.parquet")
        pred["patient_id"] = pred["patient_id"].astype(str)
        sig = _response_signals(self.config.horizon)                          # (cohort, patient_id)-indexed
        ep = build_endpoints(sig)
        ep_cols = [c for c in ep.columns if c.startswith("ep_")]
        frame = pred.set_index(["cohort", "patient_id"]).join(ep[ep_cols], how="left").reset_index()
        frame.loc[frame["cohort"] == "dr", "ep_low_adherence"] = float("nan")  # MARS mis-scaled (M5.0 QC)
        self.config.frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self.config.frame_path, index=False)
        return frame

    def load_frame(self) -> pd.DataFrame:
        return pd.read_parquet(self.config.frame_path)

    def load_exposures(self) -> pd.DataFrame:
        return pd.read_parquet(self.config.exposures_path)

    def merged(self) -> pd.DataFrame:
        frame, exp = self.load_frame(), self.load_exposures()
        frame["patient_id"] = frame["patient_id"].astype(str)
        exp["patient_id"] = exp["patient_id"].astype(str)
        return frame.merge(exp.drop(columns=["temporality"]), on=["cohort", "patient_id"], how="left")


# ----------------------------------------------------------------------------------------------------------
# Overlap gate (propensity) — wraps the propensity kernels
# ----------------------------------------------------------------------------------------------------------
class OverlapGate:
    def __init__(self, config: TreatmentConfig | None = None):
        self.config = config or TreatmentConfig()

    def run(self, merged: pd.DataFrame, out: Path) -> pd.DataFrame:
        rows = []
        for q in QUESTIONS:
            for mode in MODES:
                sub, treat = define_exposure(merged, q, mode)
                if treat.sum() < 5 or (treat == 0).sum() < 5:
                    rows.append({"question": q, "mode": mode, "n_treated": int(treat.sum()),
                                 "n_control": int((treat == 0).sum()), "verdict": "no contrast"})
                    continue
                X, _names, row_ok = confounder_matrix(sub)
                tr = treat[row_ok]
                if tr.sum() < 5 or (tr == 0).sum() < 5:
                    rows.append({"question": q, "mode": mode, "verdict": "no contrast (after NaN drop)"})
                    continue
                ps = propensity_score(X, tr, seed=self.config.seed)
                diag = overlap(ps, tr)
                w, keep = stabilized_iptw(ps, tr)
                smd_before, smd_after = float(smd(X, tr).max()), float(smd(X[keep], tr[keep], w[keep]).max())
                verdict = _verdict(diag, smd_after)
                rows.append({"question": q, "mode": mode,
                             **{k: diag[k] for k in ("n_treated", "n_control", "frac_in_support")},
                             "max_smd_before": round(smd_before, 3), "max_smd_after": round(smd_after, 3),
                             "verdict": verdict})
                sub_ok = sub[row_ok]
                pd.DataFrame({"cohort": sub_ok["cohort"].to_numpy(), "patient_id": sub_ok["patient_id"].to_numpy(),
                              "treat": tr, "ps": ps, "iptw": w, "in_support": keep}
                             ).to_parquet(out / f"propensity_{q}_{mode}.parquet", index=False)
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# Moderation — wraps fit_glm (treat×axis EIV interaction) + delta_elpd + e_value, over BOTH representations
# ----------------------------------------------------------------------------------------------------------
class ModerationModel:
    def __init__(self, config: TreatmentConfig | None = None):
        self.config = config or TreatmentConfig()

    def _sample(self, q: str, mode: str, frame: pd.DataFrame, prop_dir: Path) -> pd.DataFrame | None:
        p = prop_dir / f"propensity_{q}_{mode}.parquet"
        if not p.exists():
            return None
        ps = pd.read_parquet(p)
        ps["patient_id"] = ps["patient_id"].astype(str)
        ps = ps[ps["in_support"]]
        return frame.merge(ps[["cohort", "patient_id", "treat", "iptw"]], on=["cohort", "patient_id"], how="inner")

    def _fit_pair(self, sub, y, family, spec, treat, w, rep, fit_kw):
        found, _ = foundation_design(sub, spec, severity_col=SEV, horizon=self.config.horizon)
        arm, _ = arm_block(sub)
        grp, ng = site_index(sub)
        treat_idx = found.shape[1] + arm.shape[1]                             # treat column position in X
        block = _map_block(sub, rep)
        gbase = dict(family=family, group=grp, n_groups=ng, weights=w, **fit_kw)
        if block[0] == "eiv":                                                 # durable trio (errors-in-variables)
            _, ob, sd, axes = block
            X = np.column_stack([found, arm, treat[:, None]])
            fit0 = fit_glm(y, X, eiv_obs=ob, eiv_sd=sd, **gbase)
            fit1 = fit_glm(y, X, eiv_obs=ob, eiv_sd=sd, eiv_interact=treat, **gbase)
            int_terms = [f"beta_eiv_int[{i}]" for i in range(len(axes))]
        else:                                                                # archetypes (fixed interaction)
            _, mat, axes = block
            X0 = np.column_stack([found, arm, treat[:, None], mat])
            X1 = np.column_stack([X0, treat[:, None] * mat])                  # interaction = last len(axes) cols
            fit0 = fit_glm(y, X0, **gbase)
            fit1 = fit_glm(y, X1, **gbase)
            int_terms = [f"beta[{X0.shape[1] + i}]" for i in range(len(axes))]
        return fit0, fit1, treat_idx, int_terms, axes

    def _row(self, q, mode, oname, rep, family, fit0, fit1, treat_idx, int_terms, axes, n):
        d_elpd, se_elpd = _safe_delta_elpd(fit0, fit1)
        c0 = fit0["coef"].set_index("term")
        ate = float(c0.loc[f"beta[{treat_idx}]", "mean"])
        ate_lo, ate_hi = float(c0.loc[f"beta[{treat_idx}]", "eti_lo"]), float(c0.loc[f"beta[{treat_idx}]", "eti_hi"])
        ate_se = float(c0.loc[f"beta[{treat_idx}]", "sd"])                     # for the MDE / power guard
        d = ate if family == "gaussian" else ate * LOGIT_TO_D
        c1 = fit1["coef"].set_index("term")
        inter = {ax: (float(c1.loc[t, "mean"]), float(c1.loc[t, "eti_lo"]), float(c1.loc[t, "eti_hi"]),
                      float(c1.loc[t, "sd"])) for ax, t in zip(axes, int_terms, strict=False)}
        any_mod = any((lo > 0 or hi < 0) for _, lo, hi, _ in inter.values())
        return {"question": q, "mode": mode, "outcome": oname, "representation": rep, "n": int(n),
                "ate": round(ate, 3), "ate_lo": round(ate_lo, 3), "ate_hi": round(ate_hi, 3),
                "ate_se": round(ate_se, 4), "ate_excludes0": bool(ate_lo > 0 or ate_hi < 0),
                "e_value": round(e_value(d), 2),
                "moderation_d_elpd": round(d_elpd, 2) if np.isfinite(d_elpd) else np.nan,
                "moderation_se": round(se_elpd, 2) if np.isfinite(se_elpd) else np.nan,
                "moderation_any_axis": bool(any_mod),
                "axes": ";".join(axes),
                "int_means": ";".join(f"{inter[a][0]:+.3f}" for a in axes),
                "int_his": ";".join(f"[{inter[a][1]:+.3f},{inter[a][2]:+.3f}]" for a in axes),
                "int_ses": ";".join(f"{inter[a][3]:.4f}" for a in axes)}

    def run(self, frame: pd.DataFrame, estimable: list[tuple[str, str]], prop_dir: Path) -> pd.DataFrame:
        fit_kw = self.config.fit_kw()
        rows = []
        for q, mode in estimable:
            sub_all = self._sample(q, mode, frame, prop_dir)
            if sub_all is None:
                continue
            for oname, ycol, family, spec in OUTCOMES:
                for rep in self.config.moderation_reps:
                    axes_cols = ([f"{a}__mean" for a in DURABLE] + [f"{a}__sd" for a in DURABLE]
                                 if rep == "durable" else arch_cols(sub_all))
                    need = [ycol, SEV, f"{spec.name}__V0", "age", "sex", "siteid_city", "arm",
                            *axes_cols, "treat", "iptw"]
                    sub = sub_all.dropna(subset=[c for c in need if c in sub_all.columns]).copy()
                    if len(sub) < 60 or sub["treat"].nunique() < 2:
                        continue
                    y = sub[ycol].to_numpy(float)
                    y = (y - y.mean()) / (y.std() or 1.0) if family == "gaussian" else y.astype("int64")
                    treat = sub["treat"].to_numpy(float)
                    w = sub["iptw"].to_numpy(float); w = w / w.mean()
                    fit0, fit1, ti, it, axes = self._fit_pair(sub, y, family, spec, treat, w, rep, fit_kw)
                    rows.append(self._row(q, mode, oname, rep, family, fit0, fit1, ti, it, axes, len(sub)))
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# Confounder-survival — does the copula-M4 carrier survive treatment adjustment? (durable + archetypes)
# ----------------------------------------------------------------------------------------------------------
class ConfounderSensitivity:
    def __init__(self, config: TreatmentConfig | None = None):
        self.config = config or TreatmentConfig()

    def _fit_rows(self, sub: pd.DataFrame, rep: str, weighting: str, w, fit_kw) -> list[dict]:
        """One confounder-survival pass on `sub` (a rep × weighting cell): re-fit the copula-M4 functioning
        prognosis with vs without the harmonized drug-class exposures and read each carrier's β / HDI /
        attenuation. `w` is None (unweighted) or the renormalized M3 attrition IPW (weighted-likelihood fit;
        only the coefficient HDIs are read, never a weighted LOO)."""
        y = sub["egf__V2"].to_numpy(float); y = (y - y.mean()) / (y.std() or 1.0)
        found, _ = foundation_design(sub, SPEC_EGF, severity_col=SEV, horizon=self.config.horizon)
        arm, _ = arm_block(sub)
        grp, ng = site_index(sub)
        treat = sub[TREAT_COLS].to_numpy(float)
        block = _map_block(sub, rep)
        gbase = dict(family="gaussian", group=grp, n_groups=ng, weights=w, **fit_kw)
        if block[0] == "eiv":                                                # durable: the carrier is beta_eiv
            _, ob, sd, axes = block
            eb = dict(eiv_obs=ob, eiv_sd=sd)
            fit_no = fit_glm(y, np.column_stack([found, arm]), **eb, **gbase)
            fit_tx = fit_glm(y, np.column_stack([found, arm, treat]), **eb, **gbase)
            terms = {a: f"beta_eiv[{i}]" for i, a in enumerate(axes)}
        else:                                                                # archetypes: carrier is fixed beta
            _, mat, axes = block
            base_no, base_tx = np.column_stack([found, arm, mat]), np.column_stack([found, arm, mat, treat])
            fit_no = fit_glm(y, base_no, **gbase)
            fit_tx = fit_glm(y, base_tx, **gbase)
            start = found.shape[1] + arm.shape[1]
            terms = {a: f"beta[{start + i}]" for i, a in enumerate(axes)}

        def betas(fit, terms=terms):
            c = fit["coef"].set_index("term")
            return {a: (float(c.loc[t, "mean"]), float(c.loc[t, "eti_lo"]), float(c.loc[t, "eti_hi"]))
                    for a, t in terms.items()}
        b_no, b_tx = betas(fit_no), betas(fit_tx)
        return [{"representation": rep, "weighting": weighting, "axis": a, "n": int(len(sub)),
                 "beta_no_treat": round(b_no[a][0], 3),
                 "hdi_no": f"[{b_no[a][1]:+.3f},{b_no[a][2]:+.3f}]",
                 "beta_with_treat": round(b_tx[a][0], 3),
                 "hdi_with": f"[{b_tx[a][1]:+.3f},{b_tx[a][2]:+.3f}]",
                 "survives": bool(b_tx[a][1] > 0 or b_tx[a][2] < 0),
                 "attenuation_pct": round(100 * (1 - abs(b_tx[a][0]) / (abs(b_no[a][0]) or 1e-9)), 1)}
                for a in axes]

    def run(self, merged: pd.DataFrame) -> pd.DataFrame:
        """Confounder-survival, reported both **unweighted** and under the M3 strata-independent attrition
        IPW (`w_retained_V2`, the V0→V2 completer weight already on the frame). Exposures are baseline/
        lifetime (BP) or current (SZ/DR), V0 only — this adjusts for baseline drug-class exposure, not a
        marginal structural model over time-varying treatment (stated as an honest caveat)."""
        fit_kw = self.config.fit_kw()
        sub0 = merged[merged["on_antipsychotic"].notna()].copy()             # treatment-data subset
        for c in TREAT_COLS:
            sub0[c] = sub0[c].fillna(0.0)
        rows = []
        for rep in self.config.moderation_reps:
            axes_cols = ([f"{a}__mean" for a in DURABLE] + [f"{a}__sd" for a in DURABLE] if rep == "durable"
                         else arch_cols(sub0))
            need = ["egf__V2", "egf__V0", SEV, "age", "sex", "siteid_city", "arm", *axes_cols]
            base = sub0.dropna(subset=[c for c in need if c in sub0.columns]).copy()
            if len(base) < 60:
                continue
            rows += self._fit_rows(base, rep, "none", None, fit_kw)          # native parity (unweighted)
            if "w_retained_V2" in base.columns:                             # + M3 attrition IPW
                ww = base["w_retained_V2"].fillna(0.0).to_numpy(float)
                keep = ww > 0                                                # M3 convention: w=0 → not retained
                sub_ipw = base.loc[keep].copy()
                if len(sub_ipw) >= 60:
                    w = ww[keep]; w = w / w.mean()
                    rows += self._fit_rows(sub_ipw, rep, "ipw_v2", w, fit_kw)
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# Tolerability — does the map predict treatment-response endpoints beyond diagnosis+severity? (novel test)
# ----------------------------------------------------------------------------------------------------------
class Tolerability:
    ENDPOINTS = ["ep_side_effects", "ep_response", "ep_resistance"]

    def __init__(self, config: TreatmentConfig | None = None):
        self.config = config or TreatmentConfig()

    def run(self, frame: pd.DataFrame) -> pd.DataFrame:
        fit_kw = self.config.fit_kw()
        rows = []
        for ep in self.ENDPOINTS:
            if ep not in frame.columns:
                continue
            for rep in self.config.moderation_reps:
                axes_cols = ([f"{a}__mean" for a in DURABLE] + [f"{a}__sd" for a in DURABLE] if rep == "durable"
                             else arch_cols(frame))
                need = [ep, SEV, CGI_BASELINE, "age", "sex", "siteid_city", "arm", *axes_cols]
                sub = frame.dropna(subset=[c for c in need if c in frame.columns]).copy()
                if len(sub) < 80 or sub[ep].nunique() < 2:
                    continue
                y = sub[ep].to_numpy("int64")
                found = np.column_stack([
                    (sub["age"] - sub["age"].mean()).to_numpy() / (sub["age"].std() or 1),
                    sub["sex"].to_numpy("float64"),
                    (sub[CGI_BASELINE] - sub[CGI_BASELINE].mean()).to_numpy() / (sub[CGI_BASELINE].std() or 1)])
                arm, _ = arm_block(sub)
                grp, ng = site_index(sub)
                base = dict(family="bernoulli", group=grp, n_groups=ng, **fit_kw)
                f0 = fit_glm(y, np.column_stack([found, arm]), **base)
                block = _map_block(sub, rep)
                if block[0] == "eiv":                                        # durable: +map via EIV
                    f1 = fit_glm(y, np.column_stack([found, arm]), eiv_obs=block[1], eiv_sd=block[2], **base)
                else:                                                        # archetypes: +map fixed
                    f1 = fit_glm(y, np.column_stack([found, arm, block[1]]), **base)
                d_elpd, se_elpd = _safe_delta_elpd(f0, f1)        # +map vs foundation (best-effort LOO)
                rows.append({"endpoint": ep, "representation": rep, "n": int(len(sub)),
                             "prevalence": round(float(y.mean()), 3),
                             "d_elpd_vs_foundation": round(d_elpd, 2) if np.isfinite(d_elpd) else np.nan,
                             "se_d_elpd": round(se_elpd, 2) if np.isfinite(se_elpd) else np.nan})
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# Heterogeneity — degeneracy-free held-out ΔAUC for the treatment-response endpoints (the descriptive pillar)
# ----------------------------------------------------------------------------------------------------------
class Heterogeneity:
    """The clinical-currency, **degeneracy-free** complement to `Tolerability`'s ΔELPD: does +map improve
    held-out *discrimination* (5-fold CV AUC) of the treatment-response endpoints beyond age + sex +
    baseline CGI-S + DSM-5 arm? Frequentist logistic CV (`prognosis.clinical_value`) — no MCMC, no
    IPTW-LOO degeneracy — so it answers "is the descriptive signal real, in AUC?" without the weighted-LOO
    caveat. Reports both representations (archetypes carry it; the durable trio is flat)."""

    ENDPOINTS = ["ep_resistance", "ep_response", "ep_side_effects"]

    def __init__(self, config: TreatmentConfig | None = None):
        self.config = config or TreatmentConfig()

    def run(self, frame: pd.DataFrame) -> pd.DataFrame:
        from face.prognosis.clinical_value import auc, cv_predict, paired_auc_delta
        rows = []
        for ep in self.ENDPOINTS:
            if ep not in frame.columns:
                continue
            for rep in self.config.moderation_reps:
                axes_cols = ([f"{a}__mean" for a in DURABLE] if rep == "durable" else arch_cols(frame))
                need = [ep, CGI_BASELINE, "age", "sex", "arm", *axes_cols]
                sub = frame.dropna(subset=[c for c in need if c in frame.columns]).copy()
                if len(sub) < 80 or sub[ep].nunique() < 2:
                    continue
                y = sub[ep].to_numpy("int64")
                found = np.column_stack([
                    (sub["age"] - sub["age"].mean()).to_numpy() / (sub["age"].std() or 1),
                    sub["sex"].to_numpy("float64"),
                    (sub[CGI_BASELINE] - sub[CGI_BASELINE].mean()).to_numpy() / (sub[CGI_BASELINE].std() or 1)])
                arm, _ = arm_block(sub)
                if rep == "durable":                                         # durable means, z-scored fixed
                    mp = np.column_stack([(sub[f"{a}__mean"] - sub[f"{a}__mean"].mean()).to_numpy()
                                          / (sub[f"{a}__mean"].std() or 1) for a in DURABLE])
                else:                                                        # archetype simplex (drop-one)
                    mp, _ = _arch_fixed_block(sub)
                Xf = np.column_stack([found, arm])
                Xm = np.column_stack([Xf, mp])
                pf = cv_predict(Xf, y, seed=self.config.seed)
                pm = cv_predict(Xm, y, seed=self.config.seed)
                d_auc, lo, hi, pgt = paired_auc_delta(y, pf, pm, seed=self.config.seed)
                rows.append({"endpoint": ep, "representation": rep, "n": int(len(sub)),
                             "prevalence": round(float(y.mean()), 3),
                             "auc_foundation": round(auc(y, pf), 3), "auc_plus_map": round(auc(y, pm), 3),
                             "delta_auc": round(d_auc, 3), "delta_auc_lo": round(lo, 3),
                             "delta_auc_hi": round(hi, 3), "delta_auc_p_gt0": round(pgt, 3)})
        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------
# Treatment-course atlas — the clinician-legible monitoring artifact + the proof gates (the M5 co-headline)
# ----------------------------------------------------------------------------------------------------------
class TreatmentAtlas:
    """Per A=5 archetype corner, the 2-year rate (Wilson CI) of each treatment-response endpoint
    (resistance / response / side-effects), pooled and within cohort — the monitoring artifact (the M4-atlas
    analogue). Plus the gates that make it *proven* rather than chance: **specificity** (the corner adds beyond
    baseline severity + substance comorbidity + demographics; and how much substance alone carries),
    **de-confounding** (composition share of the worst→best corner gradient + the corner×cohort interaction), and a
    held-out **ΔAUC permutation null**. Descriptive (monitoring), never prescriptive."""

    ENDPOINTS = ["ep_resistance", "ep_response", "ep_side_effects"]
    # A=5 archetype corners on the 8-factor map (the biology corner is A2 immunometabolic — see M2 atlas)
    CORNER = {0: "A0 activation", 1: "A1 severe·clean-bio", 2: "A2 immunometabolic",
              3: "A3 trauma", 4: "A4 well"}

    def __init__(self, config: TreatmentConfig | None = None):
        self.config = config or TreatmentConfig()

    def _delta_auc_perm(self, d: pd.DataFrame, n_perm: int = 200) -> tuple[float, float]:
        """Real held-out ΔAUC (foundation = age+sex+baseline-CGI-S+arm vs +archetypes) and its permutation-null
        p-value (fraction of label-shuffled ΔAUCs ≥ real). The honest discrimination gate."""
        from face.prognosis.clinical_value import auc, cv_predict
        cols = arch_cols(d)
        sub = d.dropna(subset=["cgi_s__V0", "age", "sex", "arm", *cols]).copy()
        if len(sub) < 80 or sub["y"].nunique() < 2:
            return float("nan"), float("nan")
        y = sub["y"].to_numpy("int64")
        found = np.column_stack([
            (sub["age"] - sub["age"].mean()).to_numpy() / (sub["age"].std() or 1),
            sub["sex"].to_numpy("float64"),
            (sub["cgi_s__V0"] - sub["cgi_s__V0"].mean()).to_numpy() / (sub["cgi_s__V0"].std() or 1)])
        arm, _ = arm_block(sub)
        mp, _ = _arch_fixed_block(sub)
        Xf, Xm = np.column_stack([found, arm]), np.column_stack([found, arm, mp])
        real = auc(y, cv_predict(Xm, y, seed=self.config.seed)) - auc(y, cv_predict(Xf, y, seed=self.config.seed))
        rng = np.random.default_rng(self.config.seed)
        nd = np.empty(n_perm)
        for i in range(n_perm):
            yp = rng.permutation(y)
            nd[i] = (auc(yp, cv_predict(Xm, yp, seed=self.config.seed))
                     - auc(yp, cv_predict(Xf, yp, seed=self.config.seed)))
        return float(real), float((nd >= real).mean())

    def run(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        import statsmodels.formula.api as smf
        from scipy.stats import chi2

        fr = frame.copy()
        fr["corner_id"] = fr["arch_dominant"].astype("int64")
        nA = int(fr["corner_id"].max()) + 1                                  # A=5 on the 8-factor map (dynamic)
        atlas_rows, gate_rows = [], []
        for ep in self.ENDPOINTS:
            if ep not in fr.columns:
                continue
            d = fr.dropna(subset=[ep]).copy(); d["y"] = d[ep].astype("int64")
            present = [c for c in ("bp", "sz", "dr") if (d["cohort"] == c).sum() >= 30]
            for a in range(nA):                                              # per-corner rates + Wilson CIs
                for coh in ["pooled", *present]:
                    sub = d[d.corner_id == a] if coh == "pooled" else d[(d.corner_id == a) & (d["cohort"] == coh)]
                    n, k = len(sub), int(sub["y"].sum())
                    lo, hi = _wilson(k, n)
                    atlas_rows.append({"endpoint": ep, "archetype": a, "corner": self.CORNER[a], "cohort": coh,
                                       "n": n, "n_pos": k, "rate": round(k / n, 3) if n else np.nan,
                                       "lo": round(lo, 3) if n else np.nan, "hi": round(hi, 3) if n else np.nan})
            # specificity gate — corner beyond baseline severity + substance + demographics (and substance alone)
            spec_p = subst_p = float("nan")
            dg = d.dropna(subset=["cgi_s__V0", "substance__mean", "age", "sex"]).copy()
            for raw, z in (("cgi_s__V0", "cgi_z"), ("substance__mean", "sub_z"), ("age", "age_z")):
                dg[z] = (dg[raw] - dg[raw].mean()) / (dg[raw].std() or 1)
            try:
                b = smf.logit("y ~ cgi_z + sub_z + age_z + sex", dg).fit(disp=0)
                f_ = smf.logit("y ~ cgi_z + sub_z + age_z + sex + C(corner_id)", dg).fit(disp=0)
                spec_p = float(chi2.sf(2 * (f_.llf - b.llf), f_.df_model - b.df_model))
                s0 = smf.logit("y ~ cgi_z + age_z + sex", dg).fit(disp=0)
                s1 = smf.logit("y ~ cgi_z + age_z + sex + sub_z", dg).fit(disp=0)
                subst_p = float(chi2.sf(2 * (s1.llf - s0.llf), 1))
            except Exception:                                                # noqa: BLE001 (degenerate cell)
                pass
            # de-confounding — composition share of the worst→best corner gradient + corner×cohort interaction
            comp_share = inter_p = float("nan")
            if len(present) >= 2:
                dc = d[d["cohort"].isin(present)]
                mix = dc["cohort"].value_counts(normalize=True)
                cell = dc.groupby(["corner_id", "cohort"])["y"].mean().unstack().reindex(columns=mix.index)
                pooled = dc.groupby("corner_id")["y"].mean()
                std = (cell * mix).sum(1)
                if len(pooled) >= 2:                            # data-driven worst→best corner spread (A-agnostic)
                    WORST, BEST = int(pooled.idxmin()), int(pooled.idxmax())
                    raw = float(pooled[BEST] - pooled[WORST])
                    comp_share = (round(1 - float(std.get(BEST, np.nan) - std.get(WORST, np.nan)) / raw, 3)
                                  if raw else np.nan)
                try:
                    ma = smf.logit("y ~ C(corner_id) + C(cohort)", dc).fit(disp=0)
                    mi = smf.logit("y ~ C(corner_id) * C(cohort)", dc).fit(disp=0)
                    inter_p = round(float(chi2.sf(2 * (mi.llf - ma.llf), mi.df_model - ma.df_model)), 3)
                except Exception:                                            # noqa: BLE001
                    pass
            d_auc, perm_p = self._delta_auc_perm(d)                          # held-out ΔAUC permutation null
            gate_rows.append({"endpoint": ep, "n": int(len(d)), "prevalence": round(float(d["y"].mean()), 3),
                              "corner_beyond_sev_subst_demo_p": round(spec_p, 4) if np.isfinite(spec_p) else np.nan,
                              "substance_alone_p": round(subst_p, 4) if np.isfinite(subst_p) else np.nan,
                              "composition_share": comp_share, "cohort_interaction_p": inter_p,
                              "delta_auc": round(d_auc, 3) if np.isfinite(d_auc) else np.nan,
                              "delta_auc_perm_p": round(perm_p, 3) if np.isfinite(perm_p) else np.nan})
        return pd.DataFrame(atlas_rows), pd.DataFrame(gate_rows)


# ----------------------------------------------------------------------------------------------------------
# Runner + projector + figures
# ----------------------------------------------------------------------------------------------------------
class TreatmentRunner:
    """Walk the deterministic plan (exposures -> frame -> propensity -> moderation -> confounder ->
    tolerability -> consolidate), caching each stage under ``output_dir/<stage>/`` (manifest = model_version +
    stage_spec + config_sig). Wraps the proven M5 + M4 kernels; native M5 untouched."""

    def __init__(self, config: TreatmentConfig | None = None):
        self.config = config or TreatmentConfig()
        self.data = TreatmentData(self.config)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _cache_ok(self, out: Path, stage: TreatmentStage) -> bool:
        mf = out / "manifest.json"
        if not mf.exists():
            return False
        m = json.loads(mf.read_text())
        return (m.get("model_version") == MODEL_VERSION and m.get("stage_spec") == _stage_spec(stage)
                and m.get("config_sig") == _config_sig(self.config))

    def _manifest(self, out: Path, stage: TreatmentStage, summary: dict) -> None:
        (out / "manifest.json").write_text(json.dumps(
            {"model_version": MODEL_VERSION, "stage": stage.name, "stage_spec": _stage_spec(stage),
             "config_sig": _config_sig(self.config), "summary": summary}, indent=2, default=str))

    def run_stage(self, stage: TreatmentStage, state: dict, *, overwrite: bool = False) -> dict:
        out = self.config.output_dir / stage.name
        out.mkdir(parents=True, exist_ok=True)
        cached = self._cache_ok(out, stage) and not overwrite

        if stage.kind == "exposures":
            state["exposures"] = self.data.load_exposures() if cached else self.data.build_exposures()
            if not cached:
                self._manifest(out, stage, {"rows": int(len(state["exposures"]))})
            return state
        if stage.kind == "frame":
            state["frame"] = self.data.load_frame() if cached else self.data.build_frame()
            if not cached:
                self._manifest(out, stage, {"rows": int(len(state["frame"]))})
            return state

        if stage.kind == "propensity":
            if cached:
                state["propensity_summary"] = pd.read_csv(out / "propensity_summary.csv")
            else:
                summ = OverlapGate(self.config).run(self.data.merged(), out)
                summ.to_csv(out / "propensity_summary.csv", index=False)
                state["propensity_summary"] = summ
                self._manifest(out, stage, {"verdicts": summ.get("verdict", pd.Series()).tolist()})
            return state

        if stage.kind == "moderation":
            summ = state.get("propensity_summary")
            if summ is None:
                summ = pd.read_csv(self.config.output_dir / "propensity" / "propensity_summary.csv")
            est = {(r.question, r.mode) for r in summ.itertuples()
                   if isinstance(getattr(r, "verdict", None), str) and r.verdict.startswith("estimable")}
            estimable = [qm for qm in PRIMARY_RUNS if qm in est]   # native primary contrasts, gated by overlap
            if cached:
                state["moderation"] = pd.read_csv(out / "moderation.csv")
            else:
                mod = ModerationModel(self.config).run(self.data.load_frame(), estimable,
                                                       self.config.output_dir / "propensity")
                mod.to_csv(out / "moderation.csv", index=False)
                state["moderation"] = mod
                self._manifest(out, stage, {"estimable": estimable, "rows": int(len(mod))})
            return state

        if stage.kind == "confounder":
            if cached:
                state["confounder"] = pd.read_csv(out / "confounder.csv")
            else:
                conf = ConfounderSensitivity(self.config).run(self.data.merged())
                conf.to_csv(out / "confounder.csv", index=False)
                state["confounder"] = conf
                self._manifest(out, stage, {"rows": int(len(conf))})
            return state

        if stage.kind == "tolerability":
            if cached:
                state["tolerability"] = pd.read_csv(out / "tolerability.csv")
            else:
                tol = Tolerability(self.config).run(self.data.load_frame())
                tol.to_csv(out / "tolerability.csv", index=False)
                state["tolerability"] = tol
                self._manifest(out, stage, {"rows": int(len(tol))})
            return state

        if stage.kind == "heterogeneity":
            if cached:
                state["heterogeneity"] = pd.read_csv(out / "heterogeneity.csv")
            else:
                het = Heterogeneity(self.config).run(self.data.load_frame())
                het.to_csv(out / "heterogeneity.csv", index=False)
                state["heterogeneity"] = het
                self._manifest(out, stage, {"rows": int(len(het))})
            return state

        if stage.kind == "atlas":
            if cached:
                state["atlas"] = pd.read_csv(out / "treatment_course_atlas.csv")
                state["atlas_gates"] = pd.read_csv(out / "atlas_gates.csv")
            else:
                atlas, gates = TreatmentAtlas(self.config).run(self.data.load_frame())
                atlas.to_csv(out / "treatment_course_atlas.csv", index=False)
                gates.to_csv(out / "atlas_gates.csv", index=False)
                state["atlas"], state["atlas_gates"] = atlas, gates
                if not self.config.smoke:
                    TreatmentVisualizer(self.config).atlas_bars(atlas, gates)
                self._manifest(out, stage, {"rows": int(len(atlas)), "gates": int(len(gates))})
            return state

        if stage.kind == "consolidate":
            summ = TreatmentProjector(self.config).summary(state)
            summ.to_csv(out / "treatment_summary.csv", index=False)
            state["treatment_summary"] = summ
            self._manifest(out, stage, {"rows": int(len(summ))})
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
        out, state = self.config.output_dir, {}
        for stage, fn in [("propensity", "propensity_summary.csv"), ("moderation", "moderation.csv"),
                          ("confounder", "confounder.csv"), ("tolerability", "tolerability.csv"),
                          ("heterogeneity", "heterogeneity.csv"), ("atlas", "treatment_course_atlas.csv"),
                          ("consolidate", "treatment_summary.csv")]:
            p = out / stage / fn
            if p.exists():
                state[stage if stage != "propensity" else "propensity_summary"] = pd.read_csv(p)
        return state


class TreatmentProjector:
    """Lock the M5 verdict table: per estimable question × outcome × representation, the moderation verdict
    (ΔELPD + any-axis HDI), the ATE E-value, and the confounder-survival summary — the M5 hand-off."""

    def __init__(self, config: TreatmentConfig | None = None):
        self.config = config or TreatmentConfig()

    @staticmethod
    def _parse_eti(token: str) -> tuple[float, float]:
        lo, hi = token.strip().lstrip("[").rstrip("]").split(",")
        return float(lo), float(hi)

    def _mde(self, r) -> dict:
        """MDE/power columns for a moderation row — the smallest ATE / per-axis interaction the design
        resolves at 80% power. Exact from the posterior SD (`ate_se`/`int_ses`) when present, else from the
        serialized 94% ETI (no refit). A small MDE ⇒ a **bounded** null; a large one ⇒ underpowered."""
        ate_se = (float(r["ate_se"]) if "ate_se" in r and pd.notna(r.get("ate_se"))
                  else sd_from_eti(float(r["ate_lo"]), float(r["ate_hi"])))
        if isinstance(r.get("int_ses"), str) and r["int_ses"]:
            ses = [float(x) for x in r["int_ses"].split(";")]
        else:
            ses = [sd_from_eti(*self._parse_eti(b)) for b in str(r["int_his"]).split(";")]
        int_mdes = [mde(s) for s in ses]
        return {"ate_mde": round(mde(ate_se), 3), "int_mde_min": round(min(int_mdes), 3),
                "int_mde_max": round(max(int_mdes), 3)}

    def summary(self, state: dict) -> pd.DataFrame:
        mod = state.get("moderation")
        if mod is None or mod.empty:
            return pd.DataFrame([{"note": "no estimable moderation cells"}])
        keep = ["question", "mode", "outcome", "representation", "n", "ate", "e_value",
                "moderation_d_elpd", "moderation_se", "moderation_any_axis"]
        s = mod[keep].copy()
        s = pd.concat([s, mod.apply(lambda r: pd.Series(self._mde(r)), axis=1)], axis=1)

        def _v(r):
            elpd_ok = np.isfinite(r["moderation_d_elpd"]) and np.isfinite(r["moderation_se"])
            held_out = elpd_ok and (r["moderation_d_elpd"] - 2 * r["moderation_se"] > 0)
            if held_out and r["moderation_any_axis"]:
                return "moderates (held-out + HDI)"
            if r["moderation_any_axis"]:                       # interaction HDI excludes 0 but ΔELPD weak/NA
                return "suggestive (HDI only)" if not elpd_ok else "suggestive (HDI, ΔELPD weak)"
            # no credible interaction — the MDE distinguishes a bounded null from an underpowered one
            # (≤0.30 SD ⇒ the design could have resolved a small–moderate interaction and didn't)
            bounded = np.isfinite(r.get("int_mde_max", np.nan)) and r["int_mde_max"] <= 0.30
            return "no moderation (bounded null)" if bounded else "no moderation (underpowered)"
        s["moderation_verdict"] = s.apply(_v, axis=1)
        return s


class TreatmentVisualizer:
    def __init__(self, config: TreatmentConfig | None = None):
        self.config = config or TreatmentConfig()
        self.config.figure_dir.mkdir(parents=True, exist_ok=True)

    def moderation_bars(self, mod: pd.DataFrame, filename: str = "moderation.png") -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if mod.empty:
            mod = pd.DataFrame([{"question": "none", "outcome": "-", "representation": "-",
                                 "moderation_d_elpd": 0.0, "moderation_se": 0.0}])
        lab = mod["question"] + "·" + mod["outcome"] + "·" + mod["representation"]
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(mod) + 1.5))
        ax.barh(range(len(mod)), mod["moderation_d_elpd"],
                xerr=2 * mod["moderation_se"], color="#1a9850", capsize=3, error_kw={"lw": 0.8})
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(mod))); ax.set_yticklabels(lab, fontsize=7); ax.invert_yaxis()
        ax.set_xlabel("moderation ΔELPD vs no-interaction (±2·SE)")
        ax.set_title("Does the map moderate treatment response? (copula)")
        path = self.config.figure_dir / filename
        fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
        return path

    def atlas_bars(self, atlas: pd.DataFrame, gates: pd.DataFrame, filename: str = "treatment_course_atlas.png") -> Path:
        """The treatment-course atlas figure: per-corner 2-year rate (pooled, Wilson CIs) for each endpoint —
        the monitoring artifact. Written to the docs figure dir and (for the report) to report/figures/."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eps = [("ep_resistance", "treatment resistance"), ("ep_response", "CGI response"),
               ("ep_side_effects", "significant side-effects")]
        # A=5 archetype corners on the 8-factor map (the biology corner is A2 immunometabolic — see M2 atlas)
        corners = ["A0 activation", "A1 severe·clean-bio", "A2 immunometabolic", "A3 trauma", "A4 well"]
        colors = ["#6B4FA1", "#B7791F", "#B42318", "#0F766E", "#1a9850"]
        nA = len(corners)
        g = atlas[atlas.cohort == "pooled"]
        gate = gates.set_index("endpoint") if gates is not None and len(gates) else None
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
        for ax, (ep, label) in zip(axes, eps, strict=False):
            sub = g[g.endpoint == ep].set_index("archetype").reindex(range(nA))
            rates = sub["rate"].to_numpy()
            yerr = np.vstack([rates - sub["lo"].to_numpy(), sub["hi"].to_numpy() - rates])
            ax.bar(range(nA), rates, color=colors, yerr=yerr, capsize=3, error_kw={"lw": 0.8})
            ax.set_xticks(range(nA)); ax.set_xticklabels([c.split()[0] for c in corners], fontsize=9)
            ax.set_ylim(0, min(1.0, np.nanmax(sub["hi"].to_numpy()) + 0.12))
            ax.set_title(label, fontsize=10.5, fontweight="bold")
            if ax is axes[0]:
                ax.set_ylabel("2-year rate")
            if gate is not None and ep in gate.index:
                pp = gate.loc[ep, "delta_auc_perm_p"]; sp = gate.loc[ep, "corner_beyond_sev_subst_demo_p"]
                ax.annotate(f"beyond sev+subst p={sp:.0e}\nΔAUC perm p={pp:.2f}", (0.5, 0.97),
                            xycoords="axes fraction", ha="center", va="top", fontsize=7.5, color="#5B6573")
        fig.suptitle("Treatment-course atlas — the immunometabolic corner (A2) carries the difficult course "
                     "(monitoring, not prescribing)", y=1.02, fontsize=11.5, fontweight="bold", color="#1E366B")
        fig.tight_layout()
        out = self.config.figure_dir / filename
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
        report_fig = REPO / "report" / "figures" / "m5_treatment_atlas.png"
        if report_fig.parent.exists():
            fig.savefig(report_fig, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return out
