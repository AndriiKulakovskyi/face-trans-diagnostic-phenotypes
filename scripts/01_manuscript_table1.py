"""Table 1 — cohort composition & demographics for the manuscript.

Reproducible from the harmonized data:
  - N per cohort (V0) and DSM subtype (arm) counts
  - age mean (SD), sex distribution per cohort
  - number of distinct sites
  - per-visit retention (V0..V4)

Writes results/manuscript_table1.csv (+ prints a Markdown table).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # so Python can find src/trans_diag without pip install

from trans_diag import (  # noqa: E402
    build_unified_dataframe,  # reads + merges the three cohort CSVs into one table
    load_variables,           # reads the variable dictionary (which columns exist + how to recode them)
    to_harmonized_dataset,    # cuts to one visit and returns a numeric patient × feature matrix
)
from trans_diag.adapter import ADMINISTRATIVE_FEATURES  # noqa: E402
# ADMINISTRATIVE_FEATURES = patient IDs, site codes, arm labels.
# Excluded from the feature matrix so they never accidentally drive any analysis.

RESULTS = REPO_ROOT / "results"
COHORTS = ["bp", "sz", "dr"]
COHORT_LABEL = {"bp": "Bipolar (BP)", "sz": "Schizophrenia (SZ)", "dr": "Depression (DR)"}


def fmt_msd(x: np.ndarray) -> str:
    # Medical papers report continuous variables as "mean (SD)" in Table 1.
    # This helper centralises that format and silently skips NaNs,
    # because not every patient has every measurement.
    x = x[~np.isnan(x)]
    return f"{np.mean(x):.1f} ({np.std(x, ddof=1):.1f})" if x.size else "—"


def main() -> int:
    # The variable dictionary maps each canonical name (e.g. "age") to the correct
    # CSV column in each cohort. Without it the loader doesn't know which columns to read.
    variables = load_variables(REPO_ROOT / "data" / "face-common-vars.xlsx")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Merge all three CSVs into one long table (one row per patient × visit).
        # readiness=["READY","PARTIAL"] skips variables not yet cleared for analysis.
        long = build_unified_dataframe(
            REPO_ROOT / "data", REPO_ROOT / "data" / "face-common-vars.xlsx",
            readiness=["READY", "PARTIAL"], format="long")

        # Keep only baseline (V0) rows and produce a numeric feature matrix.
        # full.X is indexed by (cohort, patient_id) — this MultiIndex prevents ID
        # collisions because the same numeric ID can appear in multiple cohort CSVs.
        full = to_harmonized_dataset(long, variables, visit="V0",
                                     exclude=ADMINISTRATIVE_FEATURES)

    X, meta = full.X, full.metadata
    cohort = X.index.get_level_values("cohort").astype(str)  # "bp" / "sz" / "dr" per row
    age    = X["age"].to_numpy(float)
    sex    = X["sex"].to_numpy(float)
    dsm    = meta["dsm_diagnosis"].astype(str)  # DSM subtype label (e.g. "BD-I", "SZ")

    # Detect which integer codes the three cohorts use for sex (0/1 vs 1/2 etc.)
    # and report raw counts so a reader can verify the mapping against the dictionary.
    sex_vals = pd.Series(sex).dropna().astype(int)
    codes = sorted(sex_vals.unique())

    rows = []
    for c in COHORTS:
        m = cohort == c  # boolean mask for this cohort's patients
        sx = pd.Series(sex[m]).dropna().astype(int)
        sex_str = "; ".join(f"{k}:{(sx == k).sum()}" for k in codes) if len(sx) else "—"
        rows.append({
            "cohort":      COHORT_LABEL[c],
            "n_V0":        int(m.sum()),         # patients present at baseline
            "age_mean_sd": fmt_msd(age[m]),
            "sex_counts":  sex_str,
            "n_subtypes":  dsm[m].nunique(),     # how many distinct DSM labels in this cohort
        })

    # "All" summary row spanning all three cohorts.
    allm = np.ones(len(cohort), bool)
    sx = pd.Series(sex).dropna().astype(int)
    rows.append({
        "cohort":      "All",
        "n_V0":        int(allm.sum()),
        "age_mean_sd": fmt_msd(age),
        "sex_counts":  "; ".join(f"{k}:{(sx == k).sum()}" for k in codes),
        "n_subtypes":  dsm.nunique(),
    })
    table = pd.DataFrame(rows)

    # DSM subtype breakdown — how many patients per (cohort, DSM label).
    # Published so reviewers can see the exact subtype mix (e.g. BD-I vs BD-II within BP).
    subtype = (pd.DataFrame({"cohort": cohort, "dsm": dsm.to_numpy()})
               .value_counts().rename("n").reset_index()
               .sort_values(["cohort", "n"], ascending=[True, False]))

    # Site count — siteid is excluded from the feature matrix (it's administrative),
    # so we read it directly from the raw long table.
    # Needed for reporting multi-site generalisability and for the ComBat step (script 13).
    site_col = next((v.source_col("BP") for v in variables
                     if "site" in v.canonical_name.lower()), None)
    n_sites = "—"
    if "siteid_city" in long.columns:
        n_sites = int(long["siteid_city"].nunique())
    elif site_col:
        n_sites = int(long.get(site_col, pd.Series(dtype=object)).nunique())

    # Per-visit retention: how many patients came back at V1, V2, V3, V4.
    # Required by CONSORT reporting guidelines and lets readers judge longitudinal power.
    # We count by patient_uid (globally unique key) not usubjid_patients (which collides across cohorts).
    retention = (long.groupby("visit")["patient_uid"].nunique()
                 .reindex([f"V{i}" for i in range(5)]).fillna(0).astype(int))
    ret_by_cohort = (long.groupby(["cohort", "visit"])["patient_uid"].nunique()
                     .unstack("visit").reindex(columns=[f"V{i}" for i in range(5)])
                     .reindex([c.upper() for c in COHORTS]).fillna(0).astype(int))

    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "manuscript_table1.csv", index=False)
    subtype.to_csv(RESULTS / "manuscript_table1_subtypes.csv", index=False)
    ret_by_cohort.to_csv(RESULTS / "manuscript_table1_retention.csv")

    print("=== Table 1: cohort composition ===")
    print(table.to_string(index=False))
    print(f"\nsex codes present: {codes} (verify mapping vs dictionary value set)")
    print(f"distinct sites: {n_sites}")
    print("\n=== DSM subtypes (cohort × dsm_diagnosis) ===")
    print(subtype.to_string(index=False))
    print("\n=== per-visit retention (distinct patients) ===")
    print("overall:", retention.to_dict())
    print(ret_by_cohort.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
