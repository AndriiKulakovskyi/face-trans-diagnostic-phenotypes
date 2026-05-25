"""End-to-end smoke test for trans_diag.

Run from the repo root:

    python3 scripts/verify.py

Exits non-zero on any assertion failure. Prints a per-canonical rule status
table at the end.
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

# Allow running as a plain script: prepend repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from trans_diag import (  # noqa: E402
    RULES,
    YEARLY_VISIT_MAP,
    build_unified_dataframe,
    load_variables,
)

DATA_DIR = REPO_ROOT / "data"
DICT_PATH = REPO_ROOT / "data" / "face-common-vars.xlsx"

EXPECTED_PATIENTS = {"BP": 6252, "SZ": 2209, "DR": 552}
EXPECTED_ARMS = {
    "BP": {"Bipolaire de type 1", "Bipolaire de type 2", "Bipolaire non spécifié"},
    "SZ": {"Schizophrénie", "Trouble schizo-affectif", "Trouble schizophréniforme"},
    "DR": {"Trouble dépressif majeur"},
}


def header(text: str) -> None:
    bar = "=" * len(text)
    print(f"\n{bar}\n{text}\n{bar}")


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "OK " if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(label + (f" — {detail}" if detail else ""))


def run_test_1_ready_long() -> pd.DataFrame:
    header("Test 1: readiness=['READY'], format='long'")
    df = build_unified_dataframe(DATA_DIR, DICT_PATH, readiness=["READY"], format="long")
    print(f"  shape: {df.shape}")
    print(f"  visit unique: {sorted(df['visit'].dropna().unique())}")
    print(f"  cohort counts (visit-level): {df['cohort'].value_counts().to_dict()}")

    check("'cohort' column present", "cohort" in df.columns)
    check("'arm' column present", "arm" in df.columns)
    check("'visit' column present", "visit" in df.columns)
    check("'usubjid_patients' column present", "usubjid_patients" in df.columns)

    valid_visits = set(YEARLY_VISIT_MAP.values())
    visits_seen = set(df["visit"].dropna().unique())
    check("all visits are yearly V0/V1/...", visits_seen.issubset(valid_visits),
          f"seen={visits_seen}")

    patients_per_cohort = (
        df.groupby("cohort")["usubjid_patients"].nunique().to_dict()
    )
    print(f"  unique patients per cohort: {patients_per_cohort}")
    for cohort, expected in EXPECTED_PATIENTS.items():
        observed = patients_per_cohort.get(cohort, 0)
        check(f"{cohort} patients ≤ {expected}",
              observed <= expected,
              f"observed={observed}")
        check(f"{cohort} patients ≥ 50% of {expected}",
              observed >= expected * 0.5,
              f"observed={observed}")

    for cohort, expected_arms in EXPECTED_ARMS.items():
        seen_arms = set(df.loc[df["cohort"] == cohort, "arm"].dropna().unique())
        check(f"{cohort} arm labels ⊆ expected",
              seen_arms.issubset(expected_arms),
              f"unexpected={seen_arms - expected_arms}")

    ready_vars = [v for v in load_variables(DICT_PATH)
                  if v.cluster_readiness.startswith("READY")]
    ready_canonicals = {v.canonical_name for v in ready_vars}
    feature_columns = set(df.columns) - {"usubjid_patients", "cohort", "arm",
                                          "visit", "visitnum"}
    missing = ready_canonicals - feature_columns - {"usubjid_patients", "fondacode",
                                                     "arm", "armcd", "visitnum",
                                                     "visit"}
    check("every READY canonical_name appears (modulo identifiers)",
          not missing, f"missing={sorted(missing)[:5]}")

    return df


def run_test_2_ready_wide() -> pd.DataFrame:
    header("Test 2: readiness=['READY'], format='wide'")
    df = build_unified_dataframe(DATA_DIR, DICT_PATH, readiness=["READY"], format="wide")
    print(f"  shape: {df.shape}")
    # One row per patient_uid (globally unique). usubjid_patients is NOT
    # unique — it collides across cohorts (970 shared ids).
    check("one row per patient_uid", df["patient_uid"].is_unique)
    check("usubjid_patients NOT globally unique (collisions expected)",
          not df["usubjid_patients"].is_unique)
    check("'cohort' column unsuffixed", "cohort" in df.columns)
    check("'arm' column unsuffixed", "arm" in df.columns)

    suffix_cols = [c for c in df.columns if c.endswith("_V0")]
    check("at least 10 _V0 suffixed columns", len(suffix_cols) >= 10,
          f"count={len(suffix_cols)}")

    has_v1 = any(c.endswith("_V1") for c in df.columns)
    check("at least one _V1 suffixed column exists", has_v1)
    return df


def run_test_3_partial_long() -> pd.DataFrame:
    header("Test 3: readiness=['READY','PARTIAL'], format='long'")
    df = build_unified_dataframe(DATA_DIR, DICT_PATH,
                                 readiness=["READY", "PARTIAL"], format="long")
    print(f"  shape: {df.shape}")

    df_ready = build_unified_dataframe(DATA_DIR, DICT_PATH,
                                       readiness=["READY"], format="long")
    ready_cols = set(df_ready.columns)
    partial_cols = set(df.columns)
    grew_by = len(partial_cols - ready_cols)
    print(f"  feature columns grew by {grew_by} when adding PARTIAL")
    check("PARTIAL adds at least 100 new feature columns", grew_by >= 100,
          f"grew_by={grew_by}")
    return df


def run_test_4_handpicked(df_partial: pd.DataFrame) -> None:
    header("Test 4: hand-picked canonical column properties")
    picks = ["sex", "siteid_city", "ppartpremier_episode"]
    available = [p for p in picks if p in df_partial.columns]
    print(f"  picks available in df: {available}")

    for canonical in available:
        per_cohort_nan = (
            df_partial.groupby("cohort")[canonical]
            .apply(lambda s: s.isna().mean()).to_dict()
        )
        print(f"  {canonical}: per-cohort NaN rate = "
              f"{ {k: round(v, 3) for k, v in per_cohort_nan.items()} }")

    if "sex" in df_partial.columns:
        sex_vals = set(df_partial["sex"].dropna().unique())
        check("sex values ⊆ {0,1}", sex_vals.issubset({0, 1}),
              f"observed={sex_vals}")

    if "ppartpremier_episode" in df_partial.columns:
        ppe = df_partial["ppartpremier_episode"].dropna()
        ppe_vals = set(ppe.unique())
        check("ppartpremier_episode values ⊆ {0,1}",
              ppe_vals.issubset({0, 1}), f"observed={ppe_vals}")

    if "age" in df_partial.columns:
        age_vals = pd.to_numeric(df_partial["age"], errors="coerce").dropna()
        if len(age_vals):
            check("age within plausible 0-120 range",
                  age_vals.between(0, 120).mean() > 0.95,
                  f"in-range fraction={age_vals.between(0,120).mean():.3f}")


def print_rule_status_table(df_partial: pd.DataFrame) -> None:
    header("Rule status table")
    variables = load_variables(DICT_PATH)
    useful = [v for v in variables
              if v.cluster_readiness.startswith(("READY", "PARTIAL"))]

    status_counts: Counter[str] = Counter()
    registered: list[str] = []
    identity_clean: list[str] = []
    identity_warn: list[str] = []
    absent: list[str] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        present_cols = set(df_partial.columns)
        for v in useful:
            if v.canonical_name in {"usubjid_patients", "fondacode", "arm",
                                    "armcd", "visitnum", "visit"}:
                continue
            if v.canonical_name not in present_cols:
                absent.append(v.canonical_name)
                continue
            if v.canonical_name in RULES:
                registered.append(v.canonical_name)
            else:
                identity_clean.append(v.canonical_name)

    print(f"  registered transformers (used): {len(registered)}")
    print(f"    examples: {registered[:5]}")
    print(f"  identity-cast: {len(identity_clean)}")
    print(f"    examples: {identity_clean[:5]}")
    print(f"  absent from output (no cohort had a source column?): {len(absent)}")
    print(f"    examples: {absent[:5]}")


def main() -> int:
    df1 = run_test_1_ready_long()
    df2 = run_test_2_ready_wide()
    df3 = run_test_3_partial_long()
    run_test_4_handpicked(df3)
    print_rule_status_table(df3)

    header("All tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
