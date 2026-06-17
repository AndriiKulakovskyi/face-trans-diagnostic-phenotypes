#!/usr/bin/env python3
"""02 — persist V0 covariates for the covariate-adjusted measurement sensitivity arm (issue P0-04).

The published measurement equation adjusts each item for age/sex/education/site/cohort/year, but the
primary engine never implemented covariates and the processed baseline carries none. This builds a
side-table of the demographic covariates (age, sex, education_years), aligned to the SAME
``(cohort, patient_id)`` index as ``data/processed/baseline_v0.parquet`` via the same harmonization
pipeline as ``01_build_data`` (it does NOT touch the baseline). Site lives in ``site_v0.parquet``;
cohort is the index. The covariate arm (``scripts/10_covariate_sensitivity.py``) then residualizes each
continuous indicator on age(spline)+sex+education+site before the factor model — Frisch–Waugh–Lovell-
equivalent to the published ``β_jᵀ c_i`` for Gaussian items — and re-derives biology⊥G under adjustment.

    python3 scripts/02_build_covariates.py

Writes ``data/processed/covariates_v0.parquet`` (per-patient -> gitignored).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.data import build_unified_dataframe, load_variables, to_harmonized_dataset  # noqa: E402

XLSX = REPO / "data" / "face-common-vars.xlsx"
PROC = REPO / "data" / "processed"
# Demographic covariates for the adjustment arm (site comes from site_v0.parquet; cohort is the index).
CANDIDATES = ["age", "sex", "education_years", "edulevel"]


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    variables = load_variables(str(XLSX))
    df = build_unified_dataframe("data", str(XLSX), readiness=["READY", "PARTIAL"], format="long")
    ds = to_harmonized_dataset(df, variables, visit="V0", normalize=False, apply_skip_logic=True)
    X = ds.X

    present = [c for c in CANDIDATES if c in X.columns]
    absent = [c for c in CANDIDATES if c not in X.columns]
    if not present:
        raise SystemExit(f"no covariates found in harmonized matrix; looked for {CANDIDATES}")
    cov = X[present].apply(pd.to_numeric, errors="coerce")
    cov.to_parquet(PROC / "covariates_v0.parquet")

    coverage = {c: round(float(cov[c].notna().mean()), 3) for c in present}
    print(f"covariates present: {present} | absent: {absent}")
    print(f"shape {cov.shape} | coverage {coverage}")
    print(f"wrote {PROC / 'covariates_v0.parquet'}")


if __name__ == "__main__":
    main()
