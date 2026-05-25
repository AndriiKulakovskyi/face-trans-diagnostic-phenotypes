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
sys.path.insert(0, str(REPO_ROOT / "src"))

from trans_diag import (  # noqa: E402
    build_unified_dataframe,
    load_variables,
    to_harmonized_dataset,
)
from trans_diag.adapter import ADMINISTRATIVE_FEATURES  # noqa: E402

RESULTS = REPO_ROOT / "results"
COHORTS = ["bp", "sz", "dr"]
COHORT_LABEL = {"bp": "Bipolar (BP)", "sz": "Schizophrenia (SZ)", "dr": "Depression (DR)"}


def fmt_msd(x: np.ndarray) -> str:
    x = x[~np.isnan(x)]
    return f"{np.mean(x):.1f} ({np.std(x, ddof=1):.1f})" if x.size else "—"


def main() -> int:
    variables = load_variables(REPO_ROOT / "data" / "face-common-vars.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        long = build_unified_dataframe(
            REPO_ROOT / "data", REPO_ROOT / "data" / "face-common-vars.xlsx",
            readiness=["READY", "PARTIAL"], format="long")
        full = to_harmonized_dataset(long, variables, visit="V0",
                                     exclude=ADMINISTRATIVE_FEATURES)

    X, meta = full.X, full.metadata
    cohort = X.index.get_level_values("cohort").astype(str)
    age = X["age"].to_numpy(float)
    sex = X["sex"].to_numpy(float)
    dsm = meta["dsm_diagnosis"].astype(str)

    # ---- sex coding: report the two most common codes as a ratio ----
    sex_vals = pd.Series(sex).dropna().astype(int)
    codes = sorted(sex_vals.unique())

    rows = []
    for c in COHORTS:
        m = cohort == c
        sx = pd.Series(sex[m]).dropna().astype(int)
        sex_str = "; ".join(f"{k}:{(sx == k).sum()}" for k in codes) if len(sx) else "—"
        rows.append({
            "cohort": COHORT_LABEL[c],
            "n_V0": int(m.sum()),
            "age_mean_sd": fmt_msd(age[m]),
            "sex_counts": sex_str,
            "n_subtypes": dsm[m].nunique(),
        })
    allm = np.ones(len(cohort), bool)
    sx = pd.Series(sex).dropna().astype(int)
    rows.append({
        "cohort": "All",
        "n_V0": int(allm.sum()),
        "age_mean_sd": fmt_msd(age),
        "sex_counts": "; ".join(f"{k}:{(sx == k).sum()}" for k in codes),
        "n_subtypes": dsm.nunique(),
    })
    table = pd.DataFrame(rows)

    # ---- DSM subtype breakdown ----
    subtype = (pd.DataFrame({"cohort": cohort, "dsm": dsm.to_numpy()})
               .value_counts().rename("n").reset_index().sort_values(["cohort", "n"],
                                                                      ascending=[True, False]))

    # ---- sites (from raw CSV col, since siteid is administrative/excluded) ----
    site_col = next((v.source_col("BP") for v in variables
                     if "site" in v.canonical_name.lower()), None)
    n_sites = "—"
    if "siteid_city" in long.columns:
        n_sites = int(long["siteid_city"].nunique())
    elif site_col:
        n_sites = int(long.get(site_col, pd.Series(dtype=object)).nunique())

    # ---- per-visit retention (distinct patients per recoded yearly visit) ----
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
