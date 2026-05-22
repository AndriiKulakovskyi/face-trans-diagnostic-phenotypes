"""Schema-level tests for face_stratification.

These tests check the V1-only invariants declared in
``config/face_stratification/feature_schema.yaml`` at the Pydantic layer —
they run without touching any FACE CSVs.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from face_stratification.harmonization.feature_schema import (
    FeatureSchema,
    FeatureType,
    TemporalScope,
    UnifiedFeature,
    load_feature_schema,
)


def test_schema_loads_from_yaml():
    schema = load_feature_schema()
    assert schema.version
    assert len(schema.blocks) > 0
    assert len(schema.features) > 0


def test_feature_ids_are_unique():
    schema = load_feature_schema()
    ids = [f.id for f in schema.features]
    assert len(ids) == len(set(ids))


def test_every_feature_block_is_known():
    schema = load_feature_schema()
    block_ids = {b.id for b in schema.blocks}
    for f in schema.features:
        assert f.block in block_ids, f"{f.id} references unknown block {f.block}"


def test_no_trajectory_temporal_scope():
    """The sub-project is V1-only — trajectory features are forbidden."""
    schema = load_feature_schema()
    for f in schema.features:
        assert f.temporal_scope in {
            TemporalScope.CURRENT,
            TemporalScope.LIFETIME,
            TemporalScope.STATIC,
        }


def test_no_longitudinal_feature_ids():
    """Guard against ``_n1`` / ``_followup`` / ``_delta`` / ``_rci`` ids."""
    schema = load_feature_schema()
    forbidden = ("_n1", "_followup", "_delta", "_rci", "_change", "_v2", "_visit2")
    for f in schema.features:
        assert not any(f.id.endswith(s) for s in forbidden), f.id


def test_all_cohorts_valid():
    schema = load_feature_schema()
    valid = {"bp", "sz", "dr", "asp"}
    for f in schema.features:
        for c in f.cohorts:
            assert c in valid


def test_pydantic_rejects_bad_temporal_scope():
    """Direct Pydantic-level assertion: 'trajectory' is not a valid scope."""
    with pytest.raises(ValidationError):
        UnifiedFeature(
            id="foo",
            label_fr="Foo",
            block="demographics",
            type=FeatureType.CONTINUOUS,
            temporal_scope="trajectory",  # type: ignore[arg-type]
            direction="none",
            cohorts=("bp",),
        )


def test_pydantic_rejects_longitudinal_suffix():
    with pytest.raises(ValidationError):
        UnifiedFeature(
            id="inst_madrs_total_n1",
            label_fr="MADRS follow-up",
            block="mood",
            type=FeatureType.CONTINUOUS,
            temporal_scope=TemporalScope.CURRENT,
            direction="higher_is_worse",
            cohorts=("bp",),
        )


def test_pydantic_rejects_unknown_cohort():
    with pytest.raises(ValidationError):
        UnifiedFeature(
            id="inst_foo_total",
            label_fr="Foo",
            block="mood",
            type=FeatureType.CONTINUOUS,
            temporal_scope=TemporalScope.CURRENT,
            direction="none",
            cohorts=("klingon",),
        )


def test_feature_schema_rejects_duplicate_ids():
    """Global invariant: ids must be unique across the whole schema."""
    raw = {
        "version": "0.0.1",
        "blocks": [
            {"id": "x", "label_fr": "X", "description": "x", "metric": "cosine"}
        ],
        "features": [
            {
                "id": "foo",
                "label_fr": "Foo",
                "block": "x",
                "type": "continuous",
                "temporal_scope": "current",
                "direction": "none",
                "cohorts": ["bp"],
            },
            {
                "id": "foo",
                "label_fr": "Foo bis",
                "block": "x",
                "type": "continuous",
                "temporal_scope": "current",
                "direction": "none",
                "cohorts": ["sz"],
            },
        ],
    }
    with pytest.raises(ValidationError):
        FeatureSchema.model_validate(raw)


def test_cohort_specific_query():
    schema = load_feature_schema()
    bp = schema.features_for_cohort("bp")
    sz = schema.features_for_cohort("sz")
    # Sanity: BP should have MADRS, SZ should have PANSS.
    bp_ids = {f.id for f in bp}
    sz_ids = {f.id for f in sz}
    assert "inst_madrs_total" in bp_ids
    assert "inst_panss_total" in sz_ids
    assert "inst_panss_total" not in bp_ids
    assert "inst_madrs_total" not in sz_ids
