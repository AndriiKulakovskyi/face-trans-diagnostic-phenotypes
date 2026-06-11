"""The M5 analysis frame — the fixed M4 predictor side + the treatment-response endpoints.

Reuses the M4 analysis frame wholesale for the predictor side (the 9 coordinates + per-patient SD, the
8 archetypes + 4-region tessellation, the reference covariates age/sex/site/arm/cohort, the baseline
CGI-S `cgi_s__V0`, the general factor `overall_severity__mean`, and the IPW weights), and joins the
treatment-response endpoints built from the raw harmonized CGI signals (`cgi02/03a/03b`, `mars`,
`cgi01`) at the horizon. Response signals are BP/SZ only (DR has no CGI efficacy index); adherence
excludes DR (mis-scaled). Nothing re-scored, nothing imputed. Methods: docs/TREATMENT_MODEL.md.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from face.treatment import RESPONSE_SIGNALS
from face.treatment.endpoints import build_endpoints

_REPO = Path(__file__).resolve().parents[3]
_RESULTS = _REPO / "results" / "face"
_XLSX = _REPO / "data" / "face-common-vars.xlsx"


def _response_signals(visit: str):
    """Native-scale response signals at one visit, indexed (cohort, patient_id)."""
    from face.data import build_unified_dataframe, load_variables, to_harmonized_dataset

    variables = load_variables(str(_XLSX))
    df = build_unified_dataframe("data", str(_XLSX), readiness=["READY", "PARTIAL"], format="long")
    ds = to_harmonized_dataset(df, variables, visit=visit, normalize=False, apply_skip_logic=True)
    cols = [c for c in RESPONSE_SIGNALS if c in ds.X.columns]
    return ds.X[cols].apply(pd.to_numeric, errors="coerce")


def build_m5_frame(*, horizon: str = "V2", mars_low: float = 5, resistance_cgis: float = 4,
                   results_dir: str | Path = _RESULTS) -> pd.DataFrame:
    """One row per V0-roster patient: the M4 predictor side + the `ep_{name}` treatment-response
    endpoints at `horizon`. DR is set NaN on `ep_low_adherence` (MARS mis-scaled, M5.0 data-QC)."""
    results_dir = Path(results_dir)
    m4 = pd.read_parquet(results_dir / "m4" / "analysis_frame.parquet").set_index(["cohort", "patient_id"])
    sig = _response_signals(horizon)
    ep = build_endpoints(sig, mars_low=mars_low, resistance_cgis=resistance_cgis)
    ep_cols = [c for c in ep.columns if c.startswith("ep_")]
    frame = m4.join(ep[ep_cols], how="left").reset_index()
    # DR adherence is a harmonization artefact (MARS mis-scaled) — exclude it (never imputed elsewhere)
    frame.loc[frame["cohort"] == "dr", "ep_low_adherence"] = float("nan")
    return frame
