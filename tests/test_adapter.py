"""Unit tests for the engine bridge: schema_gen + adapter.

Synthetic dictionary + long frame cover:

  - dictionary section → FeatureBlock (slug), row → UnifiedFeature
  - dtype → FeatureType mapping (binary/ordinal/categorical/continuous)
  - per-cohort source column → cohorts; no-source features are skipped
  - HarmonizedDataset invariants (unique MultiIndex, X==metadata index)
  - cohort lowercasing + cross-cohort usubjid collision safety
  - non-numeric features dropped from the float matrix
  - visit restriction + min_coverage floor

Run:  pytest tests/test_adapter.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face_common.adapter import COHORT_TO_CODE, to_harmonized_dataset
from face_common.schema_gen import _slug, build_feature_schema, feature_cohorts
from face_common.variable import Variable


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _var(name, dtype, section, *, bp=True, sz=True, dr=True, label=""):
    return Variable(
        canonical_name=name,
        bp_csv_col=f"{name}_bp" if bp else None,
        sz_csv_col=f"{name}_sz" if sz else None,
        dr_csv_col=f"{name}_dr" if dr else None,
        dtype=dtype,
        unit_or_value_set="",
        cluster_readiness="READY",
        clinical_rationale="clinical reason",
        rule="",
        section=section,
        label=label or name,
        findings="",
    )


def _variables():
    return [
        _var("sex", "int8 binary", "PATIENT"),
        _var("age", "float", "PATIENT"),
        _var("madrs", "float", "AUTO-QUESTIONNAIRES", sz=False),   # bp+dr only
        _var("panss", "float", "HETERO-QUESTIONNAIRES", bp=False, dr=False),  # sz only
        _var("ord1", "int8 ordinal", "SUICIDE"),
        _var("cat1", "int8 categorical", "SOCIAL"),
        _var("note", "string", "PATIENT"),          # non-numeric
        _var("novar", "float", "PATIENT", bp=False, sz=False, dr=False),  # no source
    ]


def _row(usubjid, cohort, arm, visit, **feats):
    base = {"usubjid_patients": usubjid, "cohort": cohort, "arm": arm,
            "visit": visit, "visitnum": 100,
            "sex": np.nan, "age": np.nan, "madrs": np.nan, "panss": np.nan,
            "ord1": np.nan, "cat1": np.nan, "note": None}
    base.update(feats)
    return base


def _make_frame():
    """4 patients (incl. a BP/SZ usubjid collision) across V0/V1."""
    rows = [
        _row(1, "BP", "Bipolaire de type 1", "V0",
             sex=1, age=40.0, madrs=12.0, ord1=2, cat1=3, note="x"),
        _row(1, "BP", "Bipolaire de type 1", "V1",
             sex=1, age=41.0, madrs=10.0, ord1=1, cat1=3, note="y"),
        _row(2, "BP", "Bipolaire de type 2", "V0",
             sex=0, age=55.0, madrs=20.0, ord1=3, cat1=1, note="z"),
        _row(1, "SZ", "Schizophrénie", "V0",          # usubjid 1 collides with BP
             sex=0, age=33.0, panss=80.0, ord1=2, cat1=2, note="a"),
        _row(1, "DR", "Trouble dépressif majeur", "V0",
             sex=1, age=29.0, madrs=25.0, ord1=1, cat1=4, note="b"),
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# schema_gen
# ---------------------------------------------------------------------------

def test_slug_handles_accents_and_spaces():
    assert _slug("BILAN BIOLOGIQUE") == "bilan_biologique"
    assert _slug("PÉRINATALITÉ") == "perinatalite"
    assert _slug("SOIN SUIVI HOSP ARRET TRAVAIL") == "soin_suivi_hosp_arret_travail"


def test_feature_cohorts_from_source_columns():
    madrs = next(v for v in _variables() if v.canonical_name == "madrs")
    assert feature_cohorts(madrs) == ("bp", "dr")
    panss = next(v for v in _variables() if v.canonical_name == "panss")
    assert feature_cohorts(panss) == ("sz",)


def test_build_schema_valid_and_typed():
    vs = _variables()
    ids = [v.canonical_name for v in vs]
    schema = build_feature_schema(vs, ids)

    by_id = schema.features_by_id()
    # 'novar' (no source column) is skipped; the other 7 are kept.
    assert set(by_id) == {"sex", "age", "madrs", "panss", "ord1", "cat1", "note"}
    assert by_id["sex"].type.value == "binary"
    assert by_id["age"].type.value == "continuous"
    assert by_id["ord1"].type.value == "ordinal"
    assert by_id["cat1"].type.value == "categorical"
    assert by_id["note"].type.value == "categorical"   # string → categorical
    assert by_id["madrs"].cohorts == ("bp", "dr")
    assert by_id["panss"].cohorts == ("sz",)
    # every feature's temporal scope is the V0 default
    assert all(f.temporal_scope.value == "current" for f in schema.features)


def test_build_schema_blocks_from_sections():
    vs = _variables()
    schema = build_feature_schema(vs, [v.canonical_name for v in vs])
    block_ids = set(schema.block_ids())
    assert {"patient", "auto_questionnaires", "hetero_questionnaires",
            "suicide", "social"} <= block_ids
    # every feature references a declared block
    assert all(f.block in block_ids for f in schema.features)


def test_build_schema_empty_raises():
    with pytest.raises(ValueError):
        build_feature_schema(_variables(), ["does_not_exist"])


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------

def test_cohort_code_map():
    assert COHORT_TO_CODE == {"BP": "bp", "SZ": "sz", "DR": "dr"}


def test_harmonized_dataset_shape_and_index():
    ds = to_harmonized_dataset(_make_frame(), _variables(), visit="V0")
    # 4 distinct patients at V0 (BP1, BP2, SZ1, DR1)
    assert ds.n_patients == 4
    assert ds.X.index.is_unique
    assert ds.X.index.names == ["cohort", "patient_id"]
    # cohorts lowercased
    assert set(ds.X.index.get_level_values("cohort")) == {"bp", "sz", "dr"}
    # X and metadata share the exact index
    assert list(ds.X.index) == list(ds.metadata.index)


def test_cross_cohort_usubjid_collision_safe():
    ds = to_harmonized_dataset(_make_frame(), _variables(), visit="V0")
    # usubjid 1 exists in BP, SZ and DR — disambiguated by cohort level
    assert ("bp", "1") in ds.X.index
    assert ("sz", "1") in ds.X.index
    assert ("dr", "1") in ds.X.index


def test_non_numeric_features_dropped():
    ds = to_harmonized_dataset(_make_frame(), _variables(), visit="V0")
    assert "note" not in ds.X.columns          # string feature dropped
    assert "novar" not in ds.X.columns          # no source column
    assert (ds.X.dtypes == "float64").all()
    # numeric features retained
    assert {"sex", "age", "madrs", "panss", "ord1", "cat1"} == set(ds.X.columns)


def test_feature_metadata_matches_columns():
    ds = to_harmonized_dataset(_make_frame(), _variables(), visit="V0")
    assert list(ds.feature_metadata.index) == list(ds.X.columns)
    assert set(ds.schema.feature_ids()) == set(ds.X.columns)


def test_metadata_dsm_from_arm():
    ds = to_harmonized_dataset(_make_frame(), _variables(), visit="V0")
    assert ds.metadata.loc[("bp", "1"), "dsm_diagnosis"] == "Bipolaire de type 1"
    assert ds.metadata.loc[("sz", "1"), "dsm_diagnosis"] == "Schizophrénie"


def test_visit_restriction():
    frame = _make_frame()
    ds_v0 = to_harmonized_dataset(frame, _variables(), visit="V0")
    ds_v1 = to_harmonized_dataset(frame, _variables(), visit="V1")
    assert ds_v0.n_patients == 4
    assert ds_v1.n_patients == 1     # only BP1 has a V1 row


def test_min_coverage_drops_sparse_columns():
    # panss is observed in only 1 of 4 V0 patients (25%) → dropped at floor 0.5
    ds = to_harmonized_dataset(_make_frame(), _variables(), visit="V0", min_coverage=0.5)
    assert "panss" not in ds.X.columns
    assert "sex" in ds.X.columns      # observed in all 4


def test_duplicate_patient_rows_warn_and_dedupe():
    frame = _make_frame()
    dup = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)  # repeat BP1 V0
    with pytest.warns(UserWarning, match="duplicate"):
        ds = to_harmonized_dataset(dup, _variables(), visit="V0")
    assert ds.X.index.is_unique
    assert ds.n_patients == 4
