"""OOP temporal-coherence engine on the Gaussian-copula M1/M2 objects (M3, reworked).

Parallel OOP engine that reruns the FACE M3 temporal-coherence layer on the **copula** map + the A=5
copula archetypes, mirroring `strata.engine.py` / `prognosis.engine.py` and **wrapping the proven
temporal kernels** (`standardize` / `invariance` / `variance` / `persistence` / `membership` / `dropout`)
and the strata scoring kernels (`scoring.conditional_gaussian_draws`, `scoring.project_explicit_full_n`) —
**no edits to native M3** (`scripts/30-37`).

The one genuinely new component is **scoring V1/V2 coordinates under the FIXED copula M1**. The native M3
freezes the V0 *parametric* standardization (`apply_spec`: mean/sd/sign/logmin); the copula M1 instead uses
the frozen rank-INT map `z = Φ⁻¹(F_j(y))` AND residualizes covariates (age-spline+sex+edu+site, FWL). So a
faithful follow-up score is: (1) orient + `copula_forward` each gaussianized cell onto the V0 z-scale via the
frozen `CoreData.copula[item]` map; (2) apply the **frozen-V0 covariate residualization** with the visit's
covariates (per-visit age); (3) project onto the fixed copula Λ/Φ/σ (continuous: `conditional_gaussian_draws`;
explicit: `project_explicit_full_n`); (4) project onto the A=5 copula archetypes. This module is that scorer +
the staged orchestration; the gates (G1/G3/G4) wrap the established kernels with `A=5`.

This file is built incrementally (incremental-QC): the **copula follow-up scorer foundation** (this module's
`copula_forward`, `FrozenCovariateDesign`, `TemporalData`, `CopulaPanelScorer.score_continuous`) is validated
first; the explicit-axis projection, the G1/G3/G4 gates, and the staged runner are layered on once the scorer
is proven.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from face.temporal import CANON, VISITS
from face.temporal.standardize import V0StdSpec, capture_v0_spec, load_spec, save_spec

DURABLE: tuple[str, ...] = ("cognition", "immunometabolic")             # the trait axes (G3/G4 headline)
SPINE: str = "overall_severity"                                          # the general/severity axis

REPO = Path(__file__).resolve().parents[3]
COPULA_MAP = REPO / "results" / "m1_measurement"
MAP_STAGE = "primary"                                            # the weighted 8-factor operational fit
FOLDED_MATRIX = REPO / "configs" / "loading_matrix.immunometabolic_crossload.csv"
STRATA_DIR = REPO / "results" / "m2_strata"
RESULTS = REPO / "results" / "m3_temporal"
FIGURES = REPO / "docs" / "figures" / "m3_temporal"
PROC = REPO / "data" / "processed"
MODEL_VERSION = "m3_temporal"
# Fit order (matches the idata's Lam/Phi/f_e columns); CANON is the presentation order from the package.
F8_FIT = ["overall_severity", "cognition", "immunometabolic", "sleep", "suicidality",
          "developmental_risk", "mania_activation", "substance"]
CONT5 = ["overall_severity", "cognition", "immunometabolic", "sleep", "mania_activation"]
EXPL3 = ["suicidality", "developmental_risk", "substance"]


# ----------------------------------------------------------------------------------------------------------
# The frozen copula forward transform (the sibling of measurement.engine.copula_invert)
# ----------------------------------------------------------------------------------------------------------
def copula_forward(raw_oriented: np.ndarray, sorted_values: np.ndarray, sorted_z: np.ndarray) -> np.ndarray:
    """Gaussian-copula FORWARD map: a follow-up visit's ORIENTED raw values -> the V0 latent z-scale, via the
    frozen empirical map ``CoreData.copula[item] = (sorted_oriented_values, sorted_z)`` (the same object whose
    inverse is ``measurement.engine.copula_invert``). Monotone-interp, clamped to the V0 support; NaN in ->
    NaN out (never imputed). This holds M1 fixed — F_j is NOT re-estimated on the follow-up sample."""
    y = np.asarray(raw_oriented, dtype="float64")
    out = np.full(y.shape, np.nan)
    obs = np.isfinite(y)
    if obs.any() and len(sorted_values) >= 2:
        out[obs] = np.interp(y[obs], sorted_values, sorted_z)   # np.interp clamps to the endpoints (V0 support)
    return out


# ----------------------------------------------------------------------------------------------------------
# Frozen-V0 covariate residualization (FWL), applied per-visit (age at visit)
# ----------------------------------------------------------------------------------------------------------
@dataclass
class FrozenCovariateDesign:
    """The V0 covariate design frozen for out-of-sample (follow-up) application — the faithful analog of the
    copula fit's in-sample FWL residualization (``MeasurementDataset._residualize_on_covariates`` with
    ``covariate_mode='residualize'``, age-spline(4)+sex+edu+site, no cohort). We freeze the fitted age-spline
    basis + the block standardization (mean/sd) + the site dummy columns from V0, plus the per-item OLS
    residualization coefficients, and reapply them with each visit's covariates (per-visit age)."""
    spline: object                       # the fitted sklearn SplineTransformer (V0 age)
    age_basis_mean: np.ndarray
    age_basis_sd: np.ndarray
    edu_mean: float
    edu_sd: float
    edu_name: str
    site_columns: list                   # the V0 site-dummy columns (drop-first), as object labels
    names: list[str]
    betas: dict[str, np.ndarray]         # item -> OLS beta on [1, design] (frozen on V0 observed cells)

    def design(self, cov: pd.DataFrame) -> np.ndarray:
        """Build the covariate design [N, P] for a visit's covariate frame (columns age/sex/edu + site),
        on the FROZEN V0 basis/standardization. Missing numerics -> 0 after centring (mean-impute on the
        frozen mean is implicit since we standardize by V0 moments)."""
        n = len(cov)

        def col(name, default_mean):
            v = (pd.to_numeric(cov[name], errors="coerce").to_numpy("float64")
                 if name in cov.columns else np.full(n, np.nan))
            return np.nan_to_num(v, nan=default_mean).reshape(-1, 1)

        age = col("age", float(self.spline.bsplines_[0].t.mean()) if hasattr(self.spline, "bsplines_") else 0.0)
        sex = col("sex", 0.0)
        edu = col(self.edu_name, self.edu_mean)
        age_basis = self.spline.transform(age)
        age_basis = (age_basis - self.age_basis_mean) / np.where(self.age_basis_sd > 0, self.age_basis_sd, 1.0)
        edu = (edu - self.edu_mean) / (self.edu_sd if self.edu_sd > 0 else 1.0)
        blocks = [age_basis, sex, edu, age_basis * sex]
        if self.site_columns:
            site = (cov["siteid_city"].reindex(cov.index) if "siteid_city" in cov.columns
                    else pd.Series(np.nan, index=cov.index)).round().astype("Int64")
            dum = pd.get_dummies(site.astype("object"), prefix="site", dummy_na=False)
            dum = dum.reindex(columns=self.site_columns, fill_value=0)
            blocks.append(dum.to_numpy("float64"))
        return np.column_stack(blocks).astype("float64")

    def residualize(self, z: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
        """Subtract the frozen-V0 covariate fit from each item's z (FWL), using the visit's covariates."""
        A = np.column_stack([np.ones((len(z), 1)), self.design(cov.reindex(z.index))])
        out = z.copy()
        for item in out.columns:
            if item not in self.betas:
                continue
            y = out[item].to_numpy("float64").copy()
            obs = np.isfinite(y)
            y[obs] = y[obs] - A[obs] @ self.betas[item]
            out[item] = y
        return out


# ----------------------------------------------------------------------------------------------------------
# Config + stage
# ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TemporalConfig:
    map_dir: Path = COPULA_MAP
    strata_dir: Path = STRATA_DIR
    output_dir: Path = RESULTS
    figure_dir: Path = FIGURES
    proc_dir: Path = PROC
    A: int = 5                                   # copula archetype granularity (stability-gated in M2: A=5 on the 8-factor map)
    hdi_prob: float = 0.94
    seed: int = 20260622
    n_keep_draws: int = 200
    # MCMC knobs for the explicit-axis projection / G1 backbone / G3 trait-state
    proj_draws: int = 500
    proj_tune: int = 600
    proj_chains: int = 2
    smoke: bool = False

    def with_smoke_defaults(self) -> TemporalConfig:
        return replace(self, proj_draws=80, proj_tune=80, proj_chains=2, n_keep_draws=40, smoke=True)

    @property
    def coords_dir(self) -> Path:
        return self.strata_dir / "coordinates"

    @property
    def profiles_path(self) -> Path:
        return self.strata_dir / "consolidate" / "archetype_profiles.csv"

    @property
    def spec_path(self) -> Path:
        return self.proc_dir / "v0_standardization_spec.json"


def _config_sig(c: TemporalConfig) -> dict:
    return {"A": int(c.A), "seed": int(c.seed), "n_keep_draws": int(c.n_keep_draws),
            "proj_draws": int(c.proj_draws), "proj_tune": int(c.proj_tune), "proj_chains": int(c.proj_chains),
            "smoke": bool(c.smoke), "map_dir": str(c.map_dir)}


# ----------------------------------------------------------------------------------------------------------
# Data — V0 spec freeze + copula F_j rebuild + per-visit baselines + the frozen covariate design
# ----------------------------------------------------------------------------------------------------------
class TemporalData:
    """Freeze the V0 standardization spec, rebuild the frozen copula F_j maps + signs + the certified mixed
    structure from the copula M1 (the exact call ``StrataData.prepare`` uses), and build the frozen-V0
    covariate design. No re-fit, no imputation."""

    def __init__(self, config: TemporalConfig | None = None):
        self.config = config or TemporalConfig()
        self._mp = None
        self._idata = None
        self._frozen_cov = None

    def spec(self) -> V0StdSpec:
        if self.config.spec_path.exists():
            return load_spec(self.config.spec_path)
        spec = capture_v0_spec()
        self.config.spec_path.parent.mkdir(parents=True, exist_ok=True)
        save_spec(spec, self.config.spec_path)
        return spec

    def baseline(self, visit: str) -> pd.DataFrame:
        return pd.read_parquet(self.config.proc_dir / f"baseline_{visit.lower()}.parquet")

    def copula_mixed(self):
        """Rebuild the copula MixedData (carries `.base.copula` F_j maps, `.base.signs`, the item/factor
        structure) + load the copula idata — both frozen M1 objects."""
        if self._mp is not None:
            return self._mp, self._idata
        import arviz as az

        from face.measurement.engine import (
            DEFAULT_EXPLICIT_FACTORS,
            MeasurementConfig,
            MeasurementDataset,
        )
        # Match the 8-factor operational map's config exactly (folded matrix + substance pinned orthogonal)
        # so the rebuilt MixedData is row/column-aligned to the frozen idata's Lam/Phi/f_e.
        mcfg = MeasurementConfig(likelihood_mode="gaussian_copula", cohort_weighted=True,
                                 prior_matrix=FOLDED_MATRIX,
                                 output_dir=self.config.map_dir).with_substance_orthogonal()
        dataset = MeasurementDataset(mcfg)
        self._dataset = dataset
        self._mp = dataset.mixed(F8_FIT, explicit_factors=DEFAULT_EXPLICIT_FACTORS, min_cohorts=2,
                                 balanced=False, n_subsample=None)
        self._idata = az.from_netcdf(str(self.config.map_dir / MAP_STAGE / "idata.nc"))
        return self._mp, self._idata

    def covariate_frame(self, index, visit: str) -> pd.DataFrame:
        """Per-visit covariate frame (age at visit; sex/edu/site time-invariant from V0). Age at a follow-up
        visit = V0 age + nominal offset (V1:+1, V2:+2yr) when a per-visit age is not separately harmonized."""
        cov0 = pd.read_parquet(self.config.proc_dir / "covariates_v0.parquet").reindex(index)
        site = (pd.read_parquet(self.config.proc_dir / "site_v0.parquet")["siteid_city"].reindex(index)
                if (self.config.proc_dir / "site_v0.parquet").exists() else pd.Series(np.nan, index=index))
        offset = {"V0": 0.0, "V1": 1.0, "V2": 2.0}.get(visit, 0.0)
        out = cov0.copy()
        if "age" in out.columns:
            out["age"] = pd.to_numeric(out["age"], errors="coerce") + offset
        out["siteid_city"] = site
        return out

    def frozen_covariates(self, z0: pd.DataFrame) -> FrozenCovariateDesign:
        """Freeze the V0 covariate design (fitted age-spline basis + block standardization + site columns) and
        the per-item OLS residualization betas, from the V0 copula-z block ``z0`` (index = V0 patients)."""
        if self._frozen_cov is not None:
            return self._frozen_cov
        from sklearn.preprocessing import SplineTransformer
        cov0 = self.covariate_frame(z0.index, "V0")
        age = np.nan_to_num(pd.to_numeric(cov0.get("age"), errors="coerce").to_numpy("float64"),
                            nan=float(np.nanmean(pd.to_numeric(cov0.get("age"), errors="coerce")))).reshape(-1, 1)
        sex = np.nan_to_num(pd.to_numeric(cov0.get("sex"), errors="coerce").to_numpy("float64"), nan=0.0).reshape(-1, 1)
        edu_name = "edulevel" if "edulevel" in cov0.columns else "education_years"
        edu_raw = pd.to_numeric(cov0.get(edu_name), errors="coerce").to_numpy("float64")
        edu_mean = float(np.nanmean(edu_raw)) if np.isfinite(np.nanmean(edu_raw)) else 0.0
        edu = np.nan_to_num(edu_raw, nan=edu_mean).reshape(-1, 1)
        spline = SplineTransformer(n_knots=4, degree=3, include_bias=False).fit(age)
        ab = spline.transform(age)
        ab_mean, ab_sd = ab.mean(0), ab.std(0)
        ab_std = (ab - ab_mean) / np.where(ab_sd > 0, ab_sd, 1.0)
        edu_sd = float(edu.std()) or 1.0
        edu_std = (edu - edu_mean) / edu_sd
        site = cov0["siteid_city"].round().astype("Int64") if "siteid_city" in cov0.columns else pd.Series(dtype="Int64")
        dum = pd.get_dummies(site.astype("object"), prefix="site", dummy_na=False, drop_first=True)
        blocks = [ab_std, sex, edu_std, ab_std * sex]
        names = [f"age_spline_{i}" for i in range(ab.shape[1])] + ["sex", edu_name] + \
                [f"age_spline_{i}:sex" for i in range(ab.shape[1])]
        site_columns = list(dum.columns)
        if site_columns:
            blocks.append(dum.to_numpy("float64")); names.extend(site_columns)
        X0 = np.column_stack(blocks).astype("float64")
        A0 = np.column_stack([np.ones((len(z0), 1)), X0])
        betas: dict[str, np.ndarray] = {}
        min_obs = A0.shape[1] + 2
        for item in z0.columns:
            y = z0[item].to_numpy("float64"); obs = np.isfinite(y)
            if int(obs.sum()) >= min_obs:
                betas[item], *_ = np.linalg.lstsq(A0[obs], y[obs], rcond=None)
        self._frozen_cov = FrozenCovariateDesign(
            spline=spline, age_basis_mean=ab_mean, age_basis_sd=ab_sd, edu_mean=edu_mean, edu_sd=edu_sd,
            edu_name=edu_name, site_columns=site_columns, names=names, betas=betas)
        return self._frozen_cov


# ----------------------------------------------------------------------------------------------------------
# The copula follow-up scorer (the load-bearing new piece) — continuous axes first (validated slice)
# ----------------------------------------------------------------------------------------------------------
class CopulaPanelScorer:
    """Score a follow-up visit's coordinates under the FIXED copula M1. The continuous-axis path (this slice):
    orient + ``copula_forward`` each gaussianized cell onto the V0 z-scale, apply the frozen-V0 covariate
    residualization with the visit's covariates, then ``scoring.conditional_gaussian_draws`` under the copula
    posterior Λ/Φ/σ. The explicit-axis projection (``project_explicit_full_n``) is layered on next."""

    def __init__(self, config: TemporalConfig | None = None, data: TemporalData | None = None):
        self.config = config or TemporalConfig()
        self.data = data or TemporalData(self.config)

    def _copula_z_block(self, mp, B_visit: pd.DataFrame, items: list[str]) -> pd.DataFrame:
        """Apply the frozen copula F_j (orient by sign + interp) to a visit's raw cells for `items`."""
        copula = mp.base.copula
        signs = mp.base.signs
        out = {}
        for it in items:
            raw = (pd.to_numeric(B_visit[it], errors="coerce").to_numpy("float64")
                   if it in B_visit.columns else np.full(len(B_visit), np.nan))
            if it in copula:
                oriented = signs.get(it, 1) * raw
                out[it] = copula_forward(oriented, copula[it][0], copula[it][1])
            else:
                out[it] = np.full(len(B_visit), np.nan)
        return pd.DataFrame(out, index=B_visit.index)

    def score_continuous(self, visit: str) -> dict:
        """Continuous 6-axis coordinates for a follow-up visit under the fixed copula M1."""
        from face.strata.scoring import conditional_gaussian_draws
        mp, idata = self.data.copula_mixed()
        B = self.data.baseline(visit)
        items = list(mp.base.items)                                  # the continuous-block items (copula-z)
        # V0 copula-z block to freeze the covariate residualization on
        z0 = self._copula_z_block(mp, self.data.baseline("V0"), items)
        frozen = self.data.frozen_covariates(z0)
        # this visit's copula-z, residualized on the frozen-V0 covariates with the visit's covariates
        zv = self._copula_z_block(mp, B, items)
        cov_v = self.data.covariate_frame(B.index, visit)
        zv = frozen.residualize(zv, cov_v)
        Mv = zv[items].to_numpy("float64")
        post = idata.posterior
        cg = conditional_gaussian_draws(Mv, post, list(mp.base.factor_cols),
                                        n_draws=self.config.n_keep_draws, seed=self.config.seed)
        cidx = {f: i for i, f in enumerate(mp.base.factor_cols)}
        return {"index": B.index, "factor_cols": list(mp.base.factor_cols), "cidx": cidx,
                "mean": cg["mean"], "sd": cg["sd"], "draws": cg["draws"]}

    def _copula_prep_visit_mixed(self, mp_v0, B_visit, visit, *, cert_index, B_v0):
        """A MixedPrep with the certified copula V0 STRUCTURE but the visit's data: continuous block on the
        frozen copula z-scale (``copula_forward`` + frozen-V0 covariate residualization), native non-Gaussian
        cells read raw, ordinals re-coded to the certified V0 categories. The copula analog of
        ``standardize.prep_visit_mixed`` — feeds ``project_explicit_full_n`` unchanged."""
        items = mp_v0.base.items
        z0 = self._copula_z_block(mp_v0, self.data.baseline("V0"), items)
        frozen = self.data.frozen_covariates(z0)
        zc = self._copula_z_block(mp_v0, B_visit, items)
        zc = frozen.residualize(zc, self.data.covariate_frame(B_visit.index, visit))
        Mvis = zc[items].to_numpy("float64")
        cohort = np.asarray(B_visit.index.get_level_values("cohort"))
        base_vis = replace(mp_v0.base, M=Mvis, index=B_visit.index, cohort=cohort)

        def grab(cols):
            return pd.DataFrame({c: (pd.to_numeric(B_visit[c], errors="coerce") if c in B_visit.columns
                                     else np.nan) for c in cols}, index=B_visit.index).to_numpy().astype(float)

        mp_vis = replace(mp_v0, base=base_vis, Bin=grab(mp_v0.bin_items), Cnt=grab(mp_v0.cnt_items),
                         Ord=grab(mp_v0.ord_items), ord_K=list(mp_v0.ord_K))
        for k, it in enumerate(mp_v0.ord_items):                 # re-code ordinals to certified V0 categories
            uniq = np.sort(pd.to_numeric(B_v0.loc[cert_index][it], errors="coerce").dropna().unique())
            remap = {float(v): i for i, v in enumerate(uniq)}
            K = len(uniq)
            raw = mp_vis.Ord[:, k]
            code = np.full(len(raw), np.nan)
            for i, v in enumerate(raw):
                if np.isnan(v):
                    continue
                code[i] = (remap[v] if v in remap else (K - 1 if v > uniq[-1] else
                           (0 if v < uniq[0] else remap[float(uniq[uniq <= v].max())])))
            mp_vis.Ord[:, k] = code
            mp_vis.ord_K[k] = K
        return mp_vis

    def score_explicit(self, visit: str) -> dict:
        """Explicit 3-axis coordinates for a follow-up visit: project the per-patient explicit latents under
        the fixed copula M1 (``project_explicit_full_n``), cached to ``proj_{visit}.npz``."""
        from face.strata.scoring import project_explicit_full_n
        mp, idata = self.data.copula_mixed()
        # cache keyed by draw count so a smoke projection is never reused by a full run (and vice-versa)
        cache = self.config.output_dir / "panel" / f"proj_{visit}_d{self.config.proj_draws}.npz"
        if cache.exists():
            d = np.load(cache, allow_pickle=True)
            return {"mean": d["mean"], "sd": d["sd"], "draws": d["draws"], "fcols": list(d["fcols"])}
        B = self.data.baseline(visit)
        mp_vis = self._copula_prep_visit_mixed(mp, B, visit, cert_index=mp.base.index,
                                               B_v0=self.data.baseline("V0"))
        res = project_explicit_full_n(mp_vis, idata, draws=self.config.proj_draws, tune=self.config.proj_tune,
                                      chains=self.config.proj_chains, seed=self.config.seed)
        keep = max(1, res["draws"].shape[0] // self.config.n_keep_draws)
        res["draws"] = res["draws"][::keep][:self.config.n_keep_draws]
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, mean=res["mean"], sd=res["sd"], draws=res["draws"],
                            fcols=np.array(res["fcols"]))
        return {"mean": res["mean"], "sd": res["sd"], "draws": res["draws"], "fcols": res["fcols"]}

    def score_visit(self, visit: str) -> tuple[pd.DataFrame, np.ndarray]:
        """Assemble the full 8-dim coordinates (+ [n_keep, N, 8] draws) for a follow-up visit: continuous 5
        axes (conditional-Gaussian) + explicit 3 axes (projection), on the copula scale."""
        from scipy.stats import norm

        from face.strata.scoring import explicit_nobs
        mp, _ = self.data.copula_mixed()
        cont = self.score_continuous(visit)
        expl = self.score_explicit(visit)
        index = cont["index"]
        nk = min(cont["draws"].shape[0], expl["draws"].shape[0])
        z = float(norm.ppf(1 - (1 - self.config.hdi_prob) / 2))
        en = explicit_nobs(mp)                                   # explicit-axis observed counts (reliability)
        en_df = pd.DataFrame(en["n_obs"], index=mp.base.index, columns=en["fcols"]).reindex(index)
        df = pd.DataFrame(index=index)
        draws = np.full((nk, len(index), len(CANON)), np.nan, dtype="float32")
        for di, f in enumerate(CANON):
            if f in CONT5:
                ci = cont["cidx"][f]
                m, s = cont["mean"][:, ci], cont["sd"][:, ci]
                draws[:, :, di] = cont["draws"][:nk, :, ci]
                n = np.full(len(index), 3); rel = np.full(len(index), "well")     # continuous coverage tier
            else:
                k = expl["fcols"].index(f)
                m, s = expl["mean"][:, k], expl["sd"][:, k]
                draws[:, :, di] = expl["draws"][:nk, :, k]
                n = en_df[f].to_numpy() if f in en_df.columns else np.full(len(index), 0)
                rel = np.where(n >= 3, "well", np.where(n >= 1, "partial", "prior-dominated"))
            df[f"{f}__mean"] = np.round(m, 3)
            df[f"{f}__sd"] = np.round(s, 3)
            df[f"{f}__hdi_lo"] = np.round(m - z * s, 3)
            df[f"{f}__hdi_hi"] = np.round(m + z * s, 3)
            df[f"{f}__n_obs"] = np.asarray(n).astype(int)
            df[f"{f}__reliability"] = rel
        return df, draws


# ----------------------------------------------------------------------------------------------------------
# Panel assembly — V0 reused from strata_oop; V1/V2 scored; A=5 archetype memberships per visit
# ----------------------------------------------------------------------------------------------------------
def _uid(index: pd.MultiIndex) -> np.ndarray:
    return np.array([f"{c}|{p}" for c, p in zip(index.get_level_values("cohort"),
                                                index.get_level_values("patient_id"), strict=False)])


class PanelBuilder:
    """Assemble the long temporal panel (one row per patient×visit): V0 reused from the copula M2 coords,
    V1/V2 scored under the fixed copula M1, A=5 archetype memberships projected onto the frozen profiles per
    visit (arms A all-9 + B ⊥G), retention, and the G1 license (attached by the runner)."""

    def __init__(self, config: TemporalConfig | None = None, scorer: CopulaPanelScorer | None = None):
        self.config = config or TemporalConfig()
        self.scorer = scorer or CopulaPanelScorer(self.config)
        self.data = self.scorer.data

    def _profiles(self):
        prof = pd.read_csv(self.config.profiles_path)
        ZA = prof[prof.arm == "A_all9"][list(CANON)].to_numpy("float64")
        ZB = prof[prof.arm == "B_specifics"][[c for c in CANON if c != SPINE]].to_numpy("float64")
        nA = prof[prof.arm == "A_all9"]["name"].tolist()
        nB = prof[prof.arm == "B_specifics"]["name"].tolist()
        return ZA, ZB, nA, nB

    def _v0(self):
        coords = pd.read_parquet(self.config.coords_dir / "coordinates_full.parquet").set_index(
            ["cohort", "patient_id"])
        dz = np.load(self.config.coords_dir / "coordinates_draws.npz", allow_pickle=True)
        cols = [f"{ax}__{m}" for ax in CANON for m in
                ("mean", "sd", "hdi_lo", "hdi_hi", "n_obs", "reliability") if f"{ax}__{m}" in coords.columns]
        return coords[cols], dz["draws"]

    def build(self) -> tuple[pd.DataFrame, np.ndarray]:
        from face.temporal.membership import archetype_membership
        ZA, ZB, nA, nB = self._profiles()
        b_cols = [CANON.index(c) for c in CANON if c != SPINE]
        v0_df, v0_draws = self._v0()
        coords = {"V0": v0_df}
        draws = {"V0": v0_draws}
        for v in [x for x in VISITS if x != "V0"]:
            cdf, dr = self.scorer.score_visit(v)
            coords[v], draws[v] = cdf, dr
        parts, draw_stack = [], []
        for v in VISITS:
            cdf, dr = coords[v], draws[v]
            mA = archetype_membership(cdf[[f"{f}__mean" for f in CANON]].to_numpy(), dr, list(range(len(CANON))),
                                      ZA, nA, prefix="archA", index=cdf.index,
                                      n_draw=min(40, dr.shape[0]), seed=self.config.seed)
            mB = archetype_membership(cdf[[f"{f}__mean" for f in CANON if f != SPINE]].to_numpy(), dr, b_cols,
                                      ZB, nB, prefix="archB", index=cdf.index,
                                      n_draw=min(40, dr.shape[0]), seed=self.config.seed)
            vp = pd.concat([cdf, mA, mB], axis=1).reset_index()
            vp["visit"] = v
            vp["patient_uid"] = _uid(cdf.index)
            parts.append(vp)
            draw_stack.append(dr.astype("float32"))
        panel = pd.concat(parts, ignore_index=True)
        counts = panel.groupby("patient_uid")["visit"].agg(["nunique", "max"])
        panel["n_visits"] = panel["patient_uid"].map(counts["nunique"])
        panel["last_visit"] = panel["patient_uid"].map(counts["max"])
        nk = min(d.shape[0] for d in draw_stack)          # V0 (M2, 200) vs V1/V2 (n_keep) may differ -> align
        draws_all = np.concatenate([d[:nk] for d in draw_stack], axis=1)
        return panel, draws_all


# ----------------------------------------------------------------------------------------------------------
# Gates — G1 invariance, G3 trait/state, G4 persistence (wrap the kernels with A=5)
# ----------------------------------------------------------------------------------------------------------
class InvarianceGate:
    """G1 longitudinal measurement invariance: re-fit the simple-structure backbone per visit (scale-invariant
    by design — map-agnostic), Tucker-φ the primary loadings vs V0, license each axis. Self-contained re-run."""

    def __init__(self, config: TemporalConfig | None = None):
        self.config = config or TemporalConfig()

    def run(self, *, seeds=(20260605, 20260606)) -> dict:
        import face.measurement.kernel as cc
        from face.temporal.invariance import (
            axis_license,
            congruence_over_visits,
            fit_visit_backbone,
        )
        # 8-factor invariance backbone: the S1 continuous backbone with the two biology factors merged
        # (G + cognition + immunometabolic + sleep). Point prepare() at the folded matrix so it builds the
        # immunometabolic axis (the module-global MATRIX is the only knob; restored in finally).
        INV_FACTORS = ["overall_severity", "cognition", "immunometabolic", "sleep"]
        d = dict(draws=self.config.proj_draws, tune=self.config.proj_tune, chains=self.config.proj_chains)
        fits, converged = {}, set()
        saved_matrix = cc.MATRIX
        cc.MATRIX = FOLDED_MATRIX
        try:
            for v in VISITS:
                for s in seeds:
                    rec, diag = fit_visit_backbone(v, seed=s, factors=INV_FACTORS, label=f"inv-{v}", step="G1", **d)
                    fits[(v, s)] = rec
                    if diag["converged"]:
                        converged.add((v, s))
        finally:
            cc.MATRIX = saved_matrix
        # converged-only φ; if NONE converged (e.g. a smoke run with tiny draws) fall back to all fits so the
        # pipeline still produces a (caveated) license rather than crashing on an empty congruence table.
        cong = congruence_over_visits(fits, list(INV_FACTORS), list(VISITS), list(seeds),
                                      converged=(converged or None))
        lic = (axis_license(cong) if not cong.empty
               else pd.DataFrame(columns=["axis", "min_phi", "license"]))
        return {"congruence": cong, "license": lic, "n_converged": len(converged)}


class TraitStateModel:
    """G3 trait/state: per-axis measurement-error variance decomposition (ICC) over the panel."""

    def __init__(self, config: TemporalConfig | None = None):
        self.config = config or TemporalConfig()

    def run(self, panel: pd.DataFrame) -> pd.DataFrame:
        from face.temporal.variance import decompose, patient_patterns
        patterns = patient_patterns(panel)
        return decompose(panel, list(CANON), patterns, draws=self.config.proj_draws,
                         tune=self.config.proj_tune, chains=self.config.proj_chains, seed=self.config.seed)


class PersistenceModel:
    """G4 persistence + spine-vs-corner geometry (A=5 archetypes), and the G3⟷G4 synthesis."""

    def __init__(self, config: TemporalConfig | None = None):
        self.config = config or TemporalConfig()

    def run(self, panel: pd.DataFrame, trait_state: pd.DataFrame | None = None) -> dict:
        from scipy.stats import spearmanr

        from face.temporal.persistence import (
            membership_persistence,
            reliable_change_rate,
            spine_corner,
            trajectory_types,
        )
        rc = reliable_change_rate(panel, list(CANON), s="V0", t="V2")
        sc = spine_corner(panel, s="V0", t="V2")
        mp = membership_persistence(panel, arm="archB", A=self.config.A, s="V0", t="V2")
        tt = trajectory_types(panel, axis=SPINE)
        synthesis = None
        if trait_state is not None:
            m = rc.merge(trait_state[["axis", "icc"]], on="axis", how="inner")
            if len(m) >= 3:
                rho, p = spearmanr(m["frac_reliable"], m["icc"])
                synthesis = {"spearman_rho": round(float(rho), 3), "p": round(float(p), 4),
                             "reads_as": "trait axes hold, state axes move" if rho < 0 else "unexpected"}
        return {"reliable_change": rc, "spine_corner": sc, "membership_persistence": mp,
                "trajectory_types": tt, "g3_g4_synthesis": synthesis}


class AttritionGate:
    """G6 attrition + IPW recompute on the copula V0 coordinates (retention model is map-independent, but we
    recompute here for self-containment). Stabilized inverse-probability-of-retention weights."""

    def __init__(self, config: TemporalConfig | None = None):
        self.config = config or TemporalConfig()

    def run(self, panel: pd.DataFrame) -> pd.DataFrame:
        from sklearn.linear_model import LogisticRegression
        v0 = panel[panel.visit == "V0"].copy()
        X = v0[[f"{f}__mean" for f in CANON]].to_numpy("float64")
        Xz = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-9)
        Xz = np.nan_to_num(Xz)
        coh = pd.get_dummies(v0["cohort"], drop_first=True).to_numpy("float64")
        Xd = np.column_stack([Xz, coh])
        out = v0[["cohort", "patient_id"]].copy()
        for t in ("V1", "V2"):
            y = (v0["n_visits"].to_numpy() >= ({"V1": 2, "V2": 3}[t])).astype(int)
            if len(np.unique(y)) < 2:
                continue
            p = LogisticRegression(max_iter=2000).fit(Xd, y).predict_proba(Xd)[:, 1].clip(1e-3, 1 - 1e-3)
            out[f"p_retained_{t}"] = np.round(p, 4)
            out[f"w_retained_{t}"] = np.round(np.where(y == 1, y.mean() / p, 0.0), 4)
        return out


# ----------------------------------------------------------------------------------------------------------
# Staged runner + projector + figures
# ----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TemporalStage:
    name: str
    kind: str


def _stage_spec(s: TemporalStage) -> dict:
    return {"name": s.name, "kind": s.kind}


class TemporalRunner:
    """Walk the deterministic plan (invariance -> panel -> attrition -> trait_state -> persistence ->
    consolidate), caching each stage under ``output_dir/<stage>/`` (manifest = model_version+stage_spec+
    config_sig). The panel stage runs the copula V1/V2 scoring (the explicit projection is cached per visit)."""

    PLAN = [TemporalStage("invariance", "invariance"), TemporalStage("panel", "panel"),
            TemporalStage("attrition", "attrition"), TemporalStage("trait_state", "trait_state"),
            TemporalStage("persistence", "persistence"), TemporalStage("consolidate", "consolidate")]

    def __init__(self, config: TemporalConfig | None = None):
        self.config = config or TemporalConfig()
        self.data = TemporalData(self.config)
        self.scorer = CopulaPanelScorer(self.config, self.data)
        self.builder = PanelBuilder(self.config, self.scorer)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _cache_ok(self, out: Path, stage: TemporalStage) -> bool:
        mf = out / "manifest.json"
        if not mf.exists():
            return False
        m = json.loads(mf.read_text())
        return (m.get("model_version") == MODEL_VERSION and m.get("stage_spec") == _stage_spec(stage)
                and m.get("config_sig") == _config_sig(self.config))

    def _manifest(self, out: Path, stage: TemporalStage, summary: dict) -> None:
        (out / "manifest.json").write_text(json.dumps(
            {"model_version": MODEL_VERSION, "stage": stage.name, "stage_spec": _stage_spec(stage),
             "config_sig": _config_sig(self.config), "summary": summary}, indent=2, default=str))

    def run_stage(self, stage: TemporalStage, state: dict, *, overwrite: bool = False) -> dict:
        out = self.config.output_dir / stage.name
        out.mkdir(parents=True, exist_ok=True)
        cached = self._cache_ok(out, stage) and not overwrite

        if stage.kind == "invariance":
            if cached:
                state["invariance"] = {"license": pd.read_csv(out / "license.csv"),
                                       "congruence": pd.read_csv(out / "congruence.csv")}
            else:
                res = InvarianceGate(self.config).run()
                res["license"].to_csv(out / "license.csv", index=False)
                res["congruence"].to_csv(out / "congruence.csv", index=False)
                state["invariance"] = res
                self._manifest(out, stage, {"axes": int(len(res["license"]))})
            return state

        if stage.kind == "panel":
            if cached:
                state["panel"] = pd.read_parquet(out / "panel_coords.parquet")
                state["panel_draws"] = np.load(out / "panel_draws.npz", allow_pickle=True)["draws"]
            else:
                panel, draws = self.builder.build()
                lic = (state.get("invariance") or {}).get("license")
                if lic is not None:
                    lmap = dict(zip(lic["axis"], lic["license"], strict=False))
                    for ax in CANON:
                        panel[f"{ax}__license"] = lmap.get(ax, "not-tested")
                panel.to_parquet(out / "panel_coords.parquet", index=False)
                np.savez_compressed(out / "panel_draws.npz", draws=draws,
                                    patient_uid=panel["patient_uid"].to_numpy(),
                                    visit=panel["visit"].to_numpy(), dims=np.array(CANON))
                state["panel"], state["panel_draws"] = panel, draws
                self._manifest(out, stage, {"rows": int(len(panel)), "visits": sorted(panel["visit"].unique())})
            return state

        panel = state.get("panel")
        if panel is None:
            panel = pd.read_parquet(self.config.output_dir / "panel" / "panel_coords.parquet")
            state["panel"] = panel

        if stage.kind == "attrition":
            if cached:
                state["ipw"] = pd.read_parquet(out / "ipw_weights.parquet")
            else:
                ipw = AttritionGate(self.config).run(panel)
                ipw.to_parquet(out / "ipw_weights.parquet", index=False)
                state["ipw"] = ipw
                self._manifest(out, stage, {"rows": int(len(ipw))})
            return state

        if stage.kind == "trait_state":
            if cached:
                state["trait_state"] = pd.read_csv(out / "trait_state.csv")
            else:
                ts = TraitStateModel(self.config).run(panel)
                ts.to_csv(out / "trait_state.csv", index=False)
                state["trait_state"] = ts
                self._manifest(out, stage, {"axes": int(len(ts))})
            return state

        if stage.kind == "persistence":
            if cached:
                state["persistence"] = json.loads((out / "persistence.json").read_text())
                state["reliable_change"] = pd.read_csv(out / "reliable_change.csv")
            else:
                res = PersistenceModel(self.config).run(panel, trait_state=state.get("trait_state"))
                res["reliable_change"].to_csv(out / "reliable_change.csv", index=False)
                payload = {"spine_corner": res["spine_corner"],
                           "membership_persistence": {k: v for k, v in res["membership_persistence"].items()
                                                      if k != "cos" and k != "transition"},
                           "transition": res["membership_persistence"]["transition"].tolist(),
                           "trajectory_types": res["trajectory_types"], "g3_g4_synthesis": res["g3_g4_synthesis"]}
                (out / "persistence.json").write_text(json.dumps(payload, indent=2, default=str))
                state["persistence"], state["reliable_change"] = payload, res["reliable_change"]
                self._manifest(out, stage, {"synthesis": res["g3_g4_synthesis"]})
            return state

        if stage.kind == "consolidate":
            panel = state["panel"].copy()
            if "ipw" in state:
                ipw = state["ipw"].set_index(["cohort", "patient_id"])
                for c in ipw.columns:
                    panel[c] = panel.set_index(["cohort", "patient_id"]).index.map(ipw[c]).to_numpy() \
                        if c not in ("cohort", "patient_id") else panel[c]
            panel.to_parquet(out / "patient_panel.parquet", index=False)
            if "panel_draws" in state:
                np.savez_compressed(out / "panel_draws.npz", draws=state["panel_draws"],
                                    patient_uid=panel["patient_uid"].to_numpy(),
                                    visit=panel["visit"].to_numpy(), dims=np.array(CANON))
            self._manifest(out, stage, {"rows": int(len(panel))})
            state["patient_panel"] = panel
            return state

        raise ValueError(f"unknown stage kind: {stage.kind}")

    def run_plan(self, *, stop_after: str | None = None, overwrite: bool = False) -> dict:
        state: dict = {}
        for stage in self.PLAN:
            state = self.run_stage(stage, state, overwrite=overwrite)
            if stop_after and stage.name == stop_after:
                break
        return state

    def load_state(self) -> dict:
        out = self.config.output_dir
        state: dict = {}
        for stage, loader in [
            ("invariance", lambda d: {"license": pd.read_csv(d / "license.csv"),
                                      "congruence": pd.read_csv(d / "congruence.csv")}),
            ("panel", lambda d: pd.read_parquet(d / "panel_coords.parquet")),
            ("attrition", lambda d: pd.read_parquet(d / "ipw_weights.parquet")),
            ("trait_state", lambda d: pd.read_csv(d / "trait_state.csv")),
        ]:
            p = out / stage
            if (p / "manifest.json").exists():
                key = "ipw" if stage == "attrition" else stage
                try:
                    state[key] = loader(p)
                except FileNotFoundError:
                    pass
        pj = out / "persistence" / "persistence.json"
        if pj.exists():
            state["persistence"] = json.loads(pj.read_text())
        return state


class TemporalProjector:
    """The M4-contract hand-off: ``patient_panel.parquet`` (panel coords + A=5 memberships + license +
    retention/IPW) — produced by the consolidate stage above; this thin wrapper exposes it for reuse."""

    def __init__(self, config: TemporalConfig | None = None):
        self.config = config or TemporalConfig()

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.config.output_dir / "consolidate" / "patient_panel.parquet")


class TemporalVisualizer:
    """Headline figures: the G3 trait/state ICC forest and the G4 spine-vs-corner reliable-change bars."""

    def __init__(self, config: TemporalConfig | None = None):
        self.config = config or TemporalConfig()
        self.config.figure_dir.mkdir(parents=True, exist_ok=True)

    def _mpl(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt

    def trait_state(self, ts: pd.DataFrame, filename: str = "trait_state_icc.png") -> Path:
        plt = self._mpl()
        ts = ts.sort_values("icc")
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(ts) + 1.5))
        ax.errorbar(ts["icc"], range(len(ts)),
                    xerr=[ts["icc"] - ts["icc_lo"], ts["icc_hi"] - ts["icc"]], fmt="o", color="#2c7fb8", capsize=3)
        ax.axvline(0.5, color="k", lw=0.8, ls="--")
        ax.set_yticks(range(len(ts))); ax.set_yticklabels(ts["axis"], fontsize=8)
        ax.set_xlabel("ICC (trait fraction)  —  >0.5 trait, <0.5 state"); ax.set_xlim(0, 1)
        ax.set_title("G3 trait/state (copula map)")
        path = self.config.figure_dir / filename
        fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
        return path
