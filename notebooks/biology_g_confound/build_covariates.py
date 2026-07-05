#!/usr/bin/env python3
"""Build extended V0 covariates for the biology⊥G confound-sensitivity arm.

Augments ``data/processed/covariates_v0.parquet`` (age/sex/edu — built by
``scripts/02_build_covariates.py``) with two confounders the published demographic arm omits, so the
biology⊥G headline can be re-derived adjusting for medication and adiposity (not just age/sex/site):

  - ``on_antipsychotic`` : harmonized antipsychotic exposure
                           (``results/m5_treatment/exposures/treatment_exposures.parquet``).
                           BP = lifetime, SZ/DR = current; coverage ~54 % (NaN where no treatment data,
                           mean-imputed for the design only — a stated limitation).
  - ``bmi``              : body-mass index from the V0 baseline. NB: BMI is itself a *metabolic indicator*,
                           so the BMI arm is **exploratory / partly circular** (see run_confound_sensitivity.py).

Non-destructive to the existing columns (only adds the two). The default ``_covariate_design`` still pulls
only age/sex/edu, so the primary engine is byte-for-byte unchanged; the new columns are used only when an
arm passes ``covariate_extra_cols=(...)``. Idempotent — re-run after ``scripts/02_build_covariates.py``.

    PYTHONPATH=$PWD/src python notebooks/biology_g_confound/build_covariates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
PROC = REPO / "data" / "processed"
COV = PROC / "covariates_v0.parquet"
EXPO = REPO / "results" / "face" / "treatment_oop" / "exposures" / "treatment_exposures.parquet"


def _pid(p) -> str:
    """Canonicalize a patient id to a string ('123.0' -> '123') so keys join across tables."""
    s = str(p)
    return s[:-2] if s.endswith(".0") else s


def _key(cohort, pid) -> tuple[str, str]:
    return (str(cohort).lower(), _pid(pid))


def main() -> None:
    if not COV.exists():
        raise SystemExit(f"{COV} missing — run scripts/02_build_covariates.py first")
    cov = pd.read_parquet(COV)
    keys = [_key(c, p) for c, p in cov.index]                      # normalized join keys, cov order

    # --- antipsychotic exposure (map-independent harmonized drug-class flag) ---
    expo = pd.read_parquet(EXPO)
    ap = {_key(c, p): v for c, p, v in
          zip(expo["cohort"], expo["patient_id"], expo["on_antipsychotic"])}
    cov["on_antipsychotic"] = [ap.get(k, float("nan")) for k in keys]

    # --- BMI from the V0 baseline (a metabolic indicator; exploratory arm) ---
    base = pd.read_parquet(PROC / "baseline_v0.parquet")
    bmi = {_key(c, p): v for (c, p), v in base["bmi"].items()}
    cov["bmi"] = [bmi.get(k, float("nan")) for k in keys]

    cov.to_parquet(COV)
    cover = {c: f"{int(cov[c].notna().sum())}/{len(cov)} ({cov[c].notna().mean():.0%})"
             for c in ["on_antipsychotic", "bmi"]}
    print(f"augmented {COV}")
    print(f"  columns now: {list(cov.columns)}")
    print(f"  coverage: on_antipsychotic {cover['on_antipsychotic']} · bmi {cover['bmi']}")
    if cov["on_antipsychotic"].notna().sum() < 1000:
        print("  ⚠ low antipsychotic coverage — check the (cohort, patient_id) join keys")


if __name__ == "__main__":
    main()
