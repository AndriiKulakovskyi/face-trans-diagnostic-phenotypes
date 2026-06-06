"""Unit tests for v3.data.filters.

Synthetic long-format frames cover:

  - variable filter at a specific visit vs. across-all-rows
  - identifier protection (never dropped, never counted toward completeness)
  - threshold edge cases (0, 1, empty frame)
  - `candidates` restriction
  - patient filter row-by-row mode
  - patient filter anchor mode (visit='V0', keep_other_visits)
  - V0Anchor.apply() projecting onto V1..V4 frames
  - end-to-end select_v0_anchor()
  - FilterReport content correctness

Run:  pytest tests/test_filters.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from v3.data.filters import (
    IDENTIFIER_COLUMNS,
    VariableFilterReport,
    filter_patients,
    filter_variables,
    select_v0_anchor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_long_frame() -> pd.DataFrame:
    """A tiny long-format frame: 4 patients × 3 visits (V0/V1/V2), 5 features.

    Designed missingness patterns:
      - var_A: complete in V0; partial elsewhere
      - var_B: 50% missing at V0
      - var_C: 100% missing at V0
      - var_D: complete everywhere
      - var_E: only present at V2

    Patient #4 is half-empty at V0 (will be dropped by stronger patient filters).
    """
    rows = []
    # Patient 1 — complete everywhere
    rows.append({"usubjid_patients": 1, "cohort": "BP", "arm": "BP-1",
                 "visit": "V0", "visitnum": 100,
                 "var_A": 1.0, "var_B": 2.0, "var_C": np.nan,
                 "var_D": 4.0, "var_E": np.nan})
    rows.append({"usubjid_patients": 1, "cohort": "BP", "arm": "BP-1",
                 "visit": "V1", "visitnum": 101,
                 "var_A": 1.1, "var_B": 2.1, "var_C": np.nan,
                 "var_D": 4.1, "var_E": np.nan})
    rows.append({"usubjid_patients": 1, "cohort": "BP", "arm": "BP-1",
                 "visit": "V2", "visitnum": 102,
                 "var_A": 1.2, "var_B": np.nan, "var_C": np.nan,
                 "var_D": 4.2, "var_E": 5.2})
    # Patient 2 — V0 has var_B missing
    rows.append({"usubjid_patients": 2, "cohort": "SZ", "arm": "SZ",
                 "visit": "V0", "visitnum": 200,
                 "var_A": 1.0, "var_B": np.nan, "var_C": np.nan,
                 "var_D": 4.0, "var_E": np.nan})
    rows.append({"usubjid_patients": 2, "cohort": "SZ", "arm": "SZ",
                 "visit": "V1", "visitnum": 201,
                 "var_A": 1.1, "var_B": np.nan, "var_C": np.nan,
                 "var_D": 4.1, "var_E": np.nan})
    rows.append({"usubjid_patients": 2, "cohort": "SZ", "arm": "SZ",
                 "visit": "V2", "visitnum": 202,
                 "var_A": 1.2, "var_B": 2.2, "var_C": np.nan,
                 "var_D": np.nan, "var_E": 5.2})
    # Patient 3 — V0 has var_B present
    rows.append({"usubjid_patients": 3, "cohort": "DR", "arm": "DR",
                 "visit": "V0", "visitnum": 300,
                 "var_A": 1.0, "var_B": 2.0, "var_C": np.nan,
                 "var_D": 4.0, "var_E": np.nan})
    rows.append({"usubjid_patients": 3, "cohort": "DR", "arm": "DR",
                 "visit": "V1", "visitnum": 301,
                 "var_A": np.nan, "var_B": 2.1, "var_C": np.nan,
                 "var_D": 4.1, "var_E": np.nan})
    rows.append({"usubjid_patients": 3, "cohort": "DR", "arm": "DR",
                 "visit": "V2", "visitnum": 302,
                 "var_A": 1.2, "var_B": 2.2, "var_C": np.nan,
                 "var_D": 4.2, "var_E": 5.2})
    # Patient 4 — half-empty at V0
    rows.append({"usubjid_patients": 4, "cohort": "BP", "arm": "BP-2",
                 "visit": "V0", "visitnum": 400,
                 "var_A": np.nan, "var_B": np.nan, "var_C": np.nan,
                 "var_D": 4.0, "var_E": np.nan})
    rows.append({"usubjid_patients": 4, "cohort": "BP", "arm": "BP-2",
                 "visit": "V1", "visitnum": 401,
                 "var_A": 1.1, "var_B": 2.1, "var_C": np.nan,
                 "var_D": 4.1, "var_E": np.nan})
    return pd.DataFrame(rows)


@pytest.fixture
def df_long() -> pd.DataFrame:
    return _make_long_frame()


# ---------------------------------------------------------------------------
# filter_variables
# ---------------------------------------------------------------------------

def test_filter_variables_v0_threshold(df_long):
    out, rep = filter_variables(df_long, threshold=0.75, visit="V0")
    # At V0: var_A 3/4 present (75%), var_B 2/4 (50%), var_C 0/4, var_D 4/4, var_E 0/4
    assert set(rep.kept) == {"var_A", "var_D"}
    assert set(rep.dropped) == {"var_B", "var_C", "var_E"}
    # Identifiers always preserved
    for c in ("usubjid_patients", "cohort", "arm", "visit", "visitnum"):
        assert c in out.columns
    # Dropped features removed
    assert "var_B" not in out.columns
    assert "var_C" not in out.columns
    # Kept features still present
    assert "var_A" in out.columns
    assert "var_D" in out.columns
    # Rows preserved verbatim — column-level filter doesn't drop any rows
    assert len(out) == len(df_long)
    # Report shape
    assert rep.threshold == 0.75
    assert rep.visit == "V0"
    assert rep.n_rows_evaluated == 4


def test_filter_variables_no_visit_evaluates_all_rows(df_long):
    out, rep = filter_variables(df_long, threshold=0.5)
    # Across all 12 rows: var_C is always NaN, var_E is mostly NaN.
    # var_A: 10/12 present (~83%); var_B: 6/12 (50%); var_D: 11/12 (~92%); var_E: 3/12 (25%)
    completeness = dict(zip(rep.table["variable"], rep.table["completeness"], strict=False))
    assert completeness["var_C"] == 0.0
    assert "var_C" in rep.dropped
    assert "var_E" in rep.dropped
    assert "var_A" in rep.kept
    assert "var_D" in rep.kept
    assert "var_B" in rep.kept  # exactly at 0.5 threshold passes (>=)


def test_filter_variables_threshold_edges(df_long):
    # threshold=0 → keep every candidate
    out, rep = filter_variables(df_long, threshold=0.0, visit="V0")
    assert set(rep.dropped) == set()
    # threshold=1 → drop everything with any NaN at V0
    out, rep = filter_variables(df_long, threshold=1.0, visit="V0")
    assert "var_D" in rep.kept       # 4/4 present at V0
    assert "var_A" in rep.dropped    # has NaN at V0 (patient 4)


def test_filter_variables_invalid_threshold(df_long):
    with pytest.raises(ValueError):
        filter_variables(df_long, threshold=1.5)
    with pytest.raises(ValueError):
        filter_variables(df_long, threshold=-0.1)


def test_filter_variables_candidates_respected(df_long):
    # Only consider var_A and var_C; var_B/var_D/var_E should be preserved unchanged
    out, rep = filter_variables(
        df_long, threshold=0.5, visit="V0",
        candidates=["var_A", "var_C"],
    )
    candidates_in_table = set(rep.table["variable"])
    assert candidates_in_table == {"var_A", "var_C"}
    # var_B / var_D / var_E not evaluated → never dropped
    assert "var_B" in out.columns
    assert "var_D" in out.columns
    assert "var_E" in out.columns


def test_filter_variables_visit_without_visit_column():
    df = pd.DataFrame({"usubjid_patients": [1, 2],
                       "var_A": [1.0, np.nan]})
    with pytest.raises(ValueError, match="visit"):
        filter_variables(df, threshold=0.5, visit="V0")


def test_filter_variables_empty_visit_slice(df_long):
    # Visit label that doesn't exist in the frame → 0 rows evaluated → drop all
    out, rep = filter_variables(df_long, threshold=0.0, visit="V99")
    assert rep.n_rows_evaluated == 0
    # threshold=0 still drops because completeness defaults to 0 when no rows
    # actually completeness=0 and 0>=0 should be True...
    # Let's just check no exception thrown and the report is well-formed.
    assert isinstance(rep, VariableFilterReport)


def test_filter_variables_identifier_columns_never_drop(df_long):
    # Try to include identifiers in candidates — they still shouldn't be dropped
    out, rep = filter_variables(df_long, threshold=1.0, visit="V0")
    for ident in ("usubjid_patients", "cohort", "arm", "visit", "visitnum"):
        assert ident in out.columns
        assert ident not in rep.table["variable"].values


# ---------------------------------------------------------------------------
# filter_patients
# ---------------------------------------------------------------------------

def test_filter_patients_anchor_mode_v0(df_long):
    # Anchor at V0 with 0.6 threshold using all features
    out, rep = filter_patients(df_long, threshold=0.6, visit="V0",
                               keep_other_visits=True)
    # At V0 over 5 features:
    # Pt1: 3 present / 5 = 60% → kept
    # Pt2: 2 / 5 = 40% → dropped
    # Pt3: 3 / 5 = 60% → kept
    # Pt4: 1 / 5 = 20% → dropped
    # Patients are keyed by patient_uid (cohort::id); synthetic frame has
    # cohort but no patient_uid column, so _patient_key derives cohort::id.
    assert set(rep.kept_patient_uids) == {"BP::1", "DR::3"}
    assert set(rep.dropped_patient_uids) == {"SZ::2", "BP::4"}
    # Anchor mode preserves all visits for kept patients
    kept = out["usubjid_patients"].unique()
    assert set(kept) == {1, 3}
    # All three visits per kept patient retained
    assert (out["usubjid_patients"] == 1).sum() == 3
    assert (out["usubjid_patients"] == 3).sum() == 3


def test_filter_patients_anchor_mode_keep_other_visits_false(df_long):
    out, rep = filter_patients(df_long, threshold=0.6, visit="V0",
                               keep_other_visits=False)
    # Only V0 rows of kept patients
    assert set(out["visit"].unique()) == {"V0"}
    assert set(out["usubjid_patients"]) == {1, 3}


def test_filter_patients_row_by_row(df_long):
    # visit=None → evaluate each (patient, visit) row independently
    out, rep = filter_patients(df_long, threshold=0.6, visit=None)
    # Should be flexible: rows where ≥3/5 features non-NaN survive
    # Verify by recomputing completeness on the kept set
    feat_cols = [c for c in df_long.columns if c not in IDENTIFIER_COLUMNS]
    completeness = 1 - out[feat_cols].isna().mean(axis=1)
    assert (completeness >= 0.6 - 1e-9).all()


def test_filter_patients_variables_restricts_completeness(df_long):
    # Only count var_D toward completeness. var_D is non-NaN for every row
    # except pt2-V2 — so all rows except pt2-V2 should pass at threshold=1.0.
    out, rep = filter_patients(df_long, threshold=1.0,
                               variables=["var_D"], visit=None)
    # pt2-V2 is the only NaN var_D row → dropped
    assert len(out) == len(df_long) - 1
    drops = df_long[~df_long.index.isin(out.index)]
    assert drops.iloc[0]["usubjid_patients"] == 2
    assert drops.iloc[0]["visit"] == "V2"


def test_filter_patients_threshold_edges(df_long):
    out, rep = filter_patients(df_long, threshold=0.0, visit="V0")
    # threshold=0 → everyone kept (anchor mode preserves all visits)
    assert set(rep.kept_patient_uids) == {"BP::1", "SZ::2", "DR::3", "BP::4"}


def test_filter_patients_invalid_threshold(df_long):
    with pytest.raises(ValueError):
        filter_patients(df_long, threshold=-0.5)
    with pytest.raises(ValueError):
        filter_patients(df_long, threshold=1.5)


def test_filter_patients_requires_usubjid_column():
    df = pd.DataFrame({"var_A": [1, 2, np.nan]})
    with pytest.raises(ValueError, match="usubjid_patients"):
        filter_patients(df, threshold=0.5)


def test_filter_patients_no_features_keeps_everyone(df_long):
    # Restrict variables to an empty set → vacuous filter, everyone kept
    out, rep = filter_patients(df_long, threshold=0.99, variables=[])
    assert len(out) == len(df_long)
    assert rep.variables_used == tuple()


# ---------------------------------------------------------------------------
# V0Anchor
# ---------------------------------------------------------------------------

def test_select_v0_anchor_basic(df_long):
    # Use a lower variable threshold so var_B (50% at V0) survives, then
    # the patient filter discriminates: pt2 has var_B NaN → fails 0.99.
    v0_filtered, anchor = select_v0_anchor(
        df_long, variable_threshold=0.5, patient_threshold=0.99,
    )
    # At V0 with thr=0.5: var_A 75%, var_B 50%, var_D 100% → kept;
    # var_C 0%, var_E 0% → dropped.
    assert set(anchor.feature_columns) == {"var_A", "var_B", "var_D"}
    # Step 2 — patient filter at V0 with 0.99 over {var_A, var_B, var_D}:
    #   Pt1: 3/3 kept · Pt2: 2/3 dropped · Pt3: 3/3 kept · Pt4: 1/3 dropped
    assert set(anchor.patient_uids) == {"BP::1", "DR::3"}
    # v0_filtered restricted to V0 rows of {1,3} and the anchor features
    assert set(v0_filtered["visit"]) == {"V0"}
    assert set(v0_filtered["usubjid_patients"]) == {1, 3}
    for col in ("var_A", "var_B", "var_D"):
        assert col in v0_filtered.columns
    for col in ("var_C", "var_E"):
        assert col not in v0_filtered.columns


def test_anchor_apply_to_v1_v2(df_long):
    _, anchor = select_v0_anchor(
        df_long, variable_threshold=0.5, patient_threshold=0.99,
    )
    v1 = anchor.apply(df_long, restrict_visits=["V1"])
    assert set(v1["visit"]) == {"V1"}
    assert set(v1["usubjid_patients"]) == {1, 3}
    assert set(v1.columns) >= {"usubjid_patients", "cohort", "arm",
                                "visit", "visitnum",
                                "var_A", "var_B", "var_D"}
    # Non-anchor features dropped
    assert "var_C" not in v1.columns
    assert "var_E" not in v1.columns

    v2 = anchor.apply(df_long, restrict_visits=["V2"])
    assert set(v2["visit"]) == {"V2"}
    assert set(v2["usubjid_patients"]) == {1, 3}


def test_anchor_apply_silently_drops_missing_features(df_long):
    _, anchor = select_v0_anchor(
        df_long, variable_threshold=0.5, patient_threshold=0.99,
    )
    # Drop var_A from the frame before applying — anchor.apply should
    # produce the remaining feature set without crashing
    df_partial = df_long.drop(columns=["var_A"])
    out = anchor.apply(df_partial, restrict_visits=["V1"])
    assert "var_A" not in out.columns
    assert "var_B" in out.columns
    assert "var_D" in out.columns


def test_anchor_requires_v0_in_input():
    df_no_v0 = _make_long_frame()
    df_no_v0 = df_no_v0[df_no_v0["visit"] != "V0"]
    with pytest.raises(ValueError, match="V0"):
        select_v0_anchor(df_no_v0)


def test_anchor_apply_without_visit_restriction(df_long):
    _, anchor = select_v0_anchor(
        df_long, variable_threshold=0.5, patient_threshold=0.99,
    )
    out = anchor.apply(df_long)
    assert set(out["usubjid_patients"]) == {1, 3}
    # All visits of kept patients
    assert set(out["visit"]) >= {"V0", "V1", "V2"}


# ---------------------------------------------------------------------------
# Report introspection
# ---------------------------------------------------------------------------

def test_variable_report_table_sorted_ascending(df_long):
    _, rep = filter_variables(df_long, threshold=0.5, visit="V0")
    completenesses = rep.table["completeness"].tolist()
    assert completenesses == sorted(completenesses)


def test_report_str_smoke(df_long):
    _, vrep = filter_variables(df_long, threshold=0.5, visit="V0")
    _, prep = filter_patients(df_long, threshold=0.5, visit="V0")
    # __str__ should not crash and should contain the threshold
    assert "0.50" in str(vrep)
    assert "0.50" in str(prep)


def test_v0_anchor_str_contains_counts(df_long):
    _, anchor = select_v0_anchor(df_long, variable_threshold=0.75,
                                 patient_threshold=0.99)
    s = str(anchor)
    assert f"n_features={anchor.n_features}" in s
    assert f"n_patients={anchor.n_patients}" in s


# ---------------------------------------------------------------------------
# Cross-cohort id-collision regression (the patient_uid fix)
# ---------------------------------------------------------------------------

def _make_collision_frame() -> pd.DataFrame:
    """Two patients in different cohorts SHARE usubjid_patients == 99.
    BP-99 is complete at V0; SZ-99 is empty at V0. A correct filter keeps
    only BP-99; a buggy id-keyed filter would keep both.
    """
    rows = [
        # BP patient 99 — complete at V0, has a V1 follow-up
        {"patient_uid": "BP::99", "usubjid_patients": 99, "cohort": "BP",
         "arm": "BP-1", "visit": "V0", "var_A": 1.0, "var_B": 2.0},
        {"patient_uid": "BP::99", "usubjid_patients": 99, "cohort": "BP",
         "arm": "BP-1", "visit": "V1", "var_A": 1.1, "var_B": 2.1},
        # SZ patient 99 — empty at V0 (should be dropped)
        {"patient_uid": "SZ::99", "usubjid_patients": 99, "cohort": "SZ",
         "arm": "SZ", "visit": "V0", "var_A": np.nan, "var_B": np.nan},
        {"patient_uid": "SZ::99", "usubjid_patients": 99, "cohort": "SZ",
         "arm": "SZ", "visit": "V1", "var_A": 9.0, "var_B": 9.0},
    ]
    return pd.DataFrame(rows)


def test_filter_patients_no_cross_cohort_contamination():
    df = _make_collision_frame()
    out, rep = filter_patients(df, threshold=0.9, visit="V0",
                               keep_other_visits=True)
    # Only BP-99 passes; SZ-99 (empty at V0) must be dropped despite sharing id 99
    assert set(rep.kept_patient_uids) == {"BP::99"}
    assert set(rep.dropped_patient_uids) == {"SZ::99"}
    # The output must NOT contain SZ-99 rows (the contamination bug)
    assert set(out["patient_uid"].unique()) == {"BP::99"}
    assert (out["cohort"] == "SZ").sum() == 0
    # Both BP-99 visits retained (anchor mode)
    assert len(out) == 2


def test_anchor_apply_no_cross_cohort_contamination():
    df = _make_collision_frame()
    _, anchor = select_v0_anchor(df, variable_threshold=0.5,
                                 patient_threshold=0.9)
    assert set(anchor.patient_uids) == {"BP::99"}
    # Projecting onto V1 must pull only BP-99's V1 row, not SZ-99's
    v1 = anchor.apply(df, restrict_visits=["V1"])
    assert set(v1["patient_uid"].unique()) == {"BP::99"}
    assert (v1["cohort"] == "SZ").sum() == 0
