"""Data wiring for the representation benchmark — all features come from the Gaussian-copula vertical.

Three feature *representations* of the same patients, plus the outcomes, on the ``(cohort, patient_id)`` key:

* **RAW** — the exact 143 indicators M1 ingested (``data/processed/baseline_v0.parquet``), native clinical
  scales, NaN preserved + an explicit observed-mask. No imputation; the missingness *is* information.
* **LATENT** — the copula M1/M2 hand-off: 9 coordinate means (+ sds for the uncertainty arm) and the A=4
  archetype weights, read from ``results/face/strata_oop/`` (the certified copula objects).
* **REFERENCE** — the clinician bar: DSM-5 arm + baseline severity + baseline GAF.

Outcomes are the V0->V2 functional endpoints (``endpoints.build_endpoints``): recovery (impaired -> GAF>=71)
and deterioration (GAF drop >=10), plus the continuous GAF@V2 backbone. Targets never enter a feature block.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..endpoints import build_endpoints
from . import ARCH, CANON

REPO = Path(__file__).resolve().parents[4]
PROC = REPO / "data" / "processed"
COPULA_COORDS = REPO / "results" / "face" / "strata_oop" / "coordinates"
COPULA_CONSOLIDATE = REPO / "results" / "face" / "strata_oop" / "consolidate"
IPW_PATH = REPO / "results" / "face" / "m3" / "ipw_weights.parquet"

KEY = ["cohort", "patient_id"]


def _keyed(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` indexed by ``(cohort, patient_id)`` whether the key is columns or already the index."""
    if isinstance(df.index, pd.MultiIndex) and list(df.index.names) == KEY:
        return df
    return df.set_index(KEY)


# --------------------------------------------------------------------------- RAW (143 indicators + mask)
def load_raw() -> pd.DataFrame:
    """The 143 raw V0 indicators M1 ingested, native scales, NaN = missing (never imputed)."""
    return _keyed(pd.read_parquet(PROC / "baseline_v0.parquet"))


def raw_mask(raw: pd.DataFrame) -> pd.DataFrame:
    """Observed-cell mask (1 = observed, 0 = missing), aligned to ``raw``."""
    return raw.notna().astype("int8")


# --------------------------------------------------------------------------- LATENT (copula coords + arch)
def load_latent() -> pd.DataFrame:
    """Copula M1/M2 latent features per patient: ``{ax}__mean`` and ``{ax}__sd`` for the 9 axes, plus the
    A=4 archetype weights. Source = the certified Gaussian-copula objects under ``results/face/strata_oop``."""
    coords = _keyed(pd.read_parquet(COPULA_COORDS / "coordinates_full.parquet"))
    strata = _keyed(pd.read_parquet(COPULA_CONSOLIDATE / "patient_strata.parquet"))
    mean_cols = [f"{ax}__mean" for ax in CANON]
    sd_cols = [f"{ax}__sd" for ax in CANON]
    out = coords[mean_cols + sd_cols].join(strata[list(ARCH)], how="left")
    return out


# feature-block column groups (the arms, latent side)
def latent_blocks() -> dict[str, list[str]]:
    mean_cols = [f"{ax}__mean" for ax in CANON]
    sd_cols = [f"{ax}__sd" for ax in CANON]
    return {
        "LAT-mu": mean_cols,
        "LAT-sigma": mean_cols + sd_cols,
        "LAT-A": mean_cols + sd_cols + list(ARCH),
    }


# --------------------------------------------------------------------------- OUTCOMES (V0->V2 functional)
def _visit_scale(visit: str, src_var: str, index: pd.MultiIndex) -> pd.Series:
    """Read one native-scale variable at a visit, reindexed onto the V0 roster (absent visit -> NaN)."""
    df = _keyed(pd.read_parquet(PROC / f"baseline_{visit}.parquet"))
    s = pd.to_numeric(df[src_var], errors="coerce") if src_var in df.columns else pd.Series(np.nan, index=df.index)
    return s.reindex(index)


def load_outcomes() -> pd.DataFrame:
    """Build the functional-outcome panel (egf/cgi_s at V0/V1/V2) and the derived endpoints. Keeps the
    continuous ``egf__V{0,1,2}`` (backbone + autoregression) and the binary endpoints at BOTH horizons:
    ``ep_egf_recovery__V{1,2}`` (impaired GAF<61 -> GAF>=71) and ``ep_egf_deterioration__V{1,2}`` (drop >=10).
    The unsuffixed ``ep_egf_*`` (V2) from ``build_endpoints`` are kept for back-compat."""
    idx = load_raw().index
    panel = pd.DataFrame(index=idx)
    for visit, tag in (("v0", "V0"), ("v1", "V1"), ("v2", "V2")):
        panel[f"egf__{tag}"] = _visit_scale(visit, "egf", idx)
        panel[f"cgi_s__{tag}"] = _visit_scale(visit, "cgi01", idx)
    panel = build_endpoints(panel)
    e0 = panel["egf__V0"]
    for h in ("V1", "V2"):                                     # horizon-explicit endpoints (1-yr and 2-yr)
        eh = panel[f"egf__{h}"]
        have = e0.notna() & eh.notna()
        panel[f"ep_egf_recovery__{h}"] = ((e0 < 61) & (eh >= 71)).where(have & (e0 < 61)).astype(float)
        panel[f"ep_egf_deterioration__{h}"] = (eh <= e0 - 10).where(have).astype(float)
    return panel


# --------------------------------------------------------------------------- REFERENCE + IPW
def load_reference() -> pd.DataFrame:
    """The clinician bar building blocks: DSM-5 ``arm`` (validation-only label, used here as a model
    covariate), demographics where available, and the latent G coordinate (error-aware baseline severity).
    Baseline GAF (the autoregression term) lives on the outcome panel as ``egf__V0``."""
    coords = _keyed(pd.read_parquet(COPULA_COORDS / "coordinates_full.parquet"))
    ref = pd.DataFrame(index=coords.index)
    ref["G_mean"] = coords["overall_severity__mean"]            # error-aware latent severity
    vt_path = COPULA_COORDS / "validation_table.parquet"
    if vt_path.exists():
        vt = _keyed(pd.read_parquet(vt_path))
        for c in ("arm", "age", "sex", "education_years", "siteid_city"):
            if c in vt.columns:
                ref[c] = vt[c]
    if "arm" not in ref.columns:                                # fall back to the strata hand-off
        strata = _keyed(pd.read_parquet(COPULA_CONSOLIDATE / "patient_strata.parquet"))
        if "arm" in strata.columns:
            ref["arm"] = strata["arm"]
    return ref


def load_ipw() -> pd.Series:
    """Stabilised inverse-probability-of-retention weight at V2 (attrition sensitivity; 0 for dropouts)."""
    ipw = _keyed(pd.read_parquet(IPW_PATH))
    return ipw["w_retained_V2"]


# --------------------------------------------------------------------------- assemble
def assemble(cohorts: tuple[str, ...] | None = None) -> pd.DataFrame:
    """The analysis frame keyed by ``(cohort, patient_id)``: outcomes + endpoints + reference + IPW + the
    latent feature blocks (RAW is loaded separately via :func:`load_raw`, aligned on the shared index).

    ``cohorts`` filters the roster (e.g. ``("bp", "dr")`` for the episodic headline). Diagnosis/cohort are
    carried for *stratification and validation only*; never pass them as features.
    """
    frame = load_outcomes()
    frame = frame.join(load_reference(), how="left")
    frame["w_retained_V2"] = load_ipw().reindex(frame.index)
    frame = frame.join(load_latent(), how="left")
    if cohorts is not None:                                     # filter on the index level (cohort is NOT a
        lvl = frame.index.get_level_values("cohort")           # column — avoids groupby('cohort') ambiguity)
        frame = frame[lvl.isin(cohorts)]
    return frame


def cohort_of(frame: pd.DataFrame) -> np.ndarray:
    """Cohort labels as an array (read from the ``(cohort, patient_id)`` index, never a column)."""
    return frame.index.get_level_values("cohort").to_numpy()


def eligible(frame: pd.DataFrame, target: str, horizon: str = "V2") -> np.ndarray:
    """Boolean mask of rows with a defined (non-missing) target at the given horizon — recovery is
    auto-restricted to the baseline-impaired (GAF<61); deterioration to those with V0&V{horizon} GAF."""
    col = f"ep_{target}__{horizon}"
    if col not in frame.columns:
        col = f"ep_{target}"
    return frame[col].notna().to_numpy()
