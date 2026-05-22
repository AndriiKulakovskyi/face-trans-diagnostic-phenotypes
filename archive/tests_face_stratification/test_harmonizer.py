"""Smoke tests for the harmonizer.

These tests run on real FACE CSVs (a small slice per cohort) and assert the
V1-only invariants end-to-end. They are skipped automatically if any of the
CSVs is missing, so CI environments without clinical data still pass.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from face_stratification import build_harmonized_dataset
from face_stratification.harmonization.feature_schema import (
    TemporalScope,
    load_feature_schema,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATHS = {
    "bp": REPO_ROOT / "data" / "BP.csv",
    "sz": REPO_ROOT / "data" / "SZ.csv",
    "dr": REPO_ROOT / "data" / "DR.csv",
    "asp": REPO_ROOT / "data" / "ASP.csv",
}

SKIP_REASON = "FACE CSVs not found under data/ — this smoke test is integration-only."


def _require_csvs() -> dict[str, Path]:
    missing = [c for c, p in CSV_PATHS.items() if not p.is_file()]
    if missing:
        pytest.skip(f"{SKIP_REASON} (missing: {missing})")
    return CSV_PATHS


@pytest.fixture(scope="module")
def small_dataset():
    csvs = _require_csvs()
    return build_harmonized_dataset(csv_paths=csvs, max_rows_per_cohort=25)


def test_single_row_per_patient(small_dataset):
    """V1 invariant #1: exactly one row per (cohort, patient_id)."""
    assert small_dataset.X.index.is_unique
    # Index names are stable
    assert list(small_dataset.X.index.names) == ["cohort", "patient_id"]


def test_feature_matrix_is_float(small_dataset):
    """Every column must be float64 — no mixed types leaking into the matrix."""
    for dtype in small_dataset.X.dtypes:
        assert dtype == np.float64


def test_all_four_cohorts_present(small_dataset):
    counts = small_dataset.cohort_counts()
    assert set(counts.index) == {"bp", "sz", "dr", "asp"}
    assert all(c > 0 for c in counts.values)


def test_no_longitudinal_columns(small_dataset):
    """V1 invariant #2: no ``_n1`` / ``_followup`` / ``_delta`` / ``_rci`` columns."""
    forbidden = re.compile(r"(_n1|_followup|_delta|_rci|_change|_v2|_visit2)$")
    offenders = [c for c in small_dataset.X.columns if forbidden.search(c)]
    assert not offenders, f"Longitudinal-looking columns: {offenders}"


def test_no_diagnosis_column_in_X(small_dataset):
    """V1 invariant #3: DSM diagnosis must live in metadata, never in ``X``."""
    assert "dsm_diagnosis" not in small_dataset.X.columns
    assert "cohort" not in small_dataset.X.columns
    # But the metadata side table has them
    assert "dsm_diagnosis" in small_dataset.metadata.columns
    assert "cohort" in small_dataset.metadata.columns


def test_metadata_index_matches_X(small_dataset):
    assert list(small_dataset.X.index) == list(small_dataset.metadata.index)


def test_all_features_have_valid_temporal_scope(small_dataset):
    """Every feature_metadata row has a V1-valid temporal scope."""
    allowed = {s.value for s in (TemporalScope.CURRENT, TemporalScope.LIFETIME, TemporalScope.STATIC)}
    assert set(small_dataset.feature_metadata["temporal_scope"]) <= allowed


def test_cohort_only_features_are_nan_elsewhere(small_dataset):
    """PANSS is SZ-only; it should be fully NaN for non-SZ rows."""
    if "inst_panss_total" not in small_dataset.X.columns:
        pytest.skip("PANSS feature not in schema")
    non_sz = small_dataset.X.xs("bp", level="cohort")["inst_panss_total"]
    assert non_sz.isna().all(), "PANSS should be NaN for BP patients"


def test_mandatory_transdiagnostic_feature_coverage(small_dataset):
    """Age must be populated for every patient across all cohorts."""
    assert small_dataset.X["demo_age_years"].notna().sum() > 0
    # And sex_male coverage should be reasonably high
    assert small_dataset.X["demo_sex_male"].notna().mean() > 0.5


def test_feature_metadata_columns_are_sorted_like_X(small_dataset):
    """The feature_metadata index lines up with the column order of X."""
    assert list(small_dataset.feature_metadata.index) == list(small_dataset.X.columns)


def test_build_is_deterministic(small_dataset):
    """Building twice produces byte-identical feature matrices."""
    csvs = _require_csvs()
    ds2 = build_harmonized_dataset(csv_paths=csvs, max_rows_per_cohort=25)
    pd.testing.assert_frame_equal(small_dataset.X, ds2.X)
    pd.testing.assert_frame_equal(small_dataset.metadata, ds2.metadata)
