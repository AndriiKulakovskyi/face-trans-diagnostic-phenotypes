"""Tests for the data-driven transdiagnostic feature selector + graph builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face_stratification.graph.transdiagnostic import (
    build_transdiagnostic_graph,
    compute_per_cohort_coverage,
    select_transdiagnostic_features,
)
from face_stratification.harmonization.feature_schema import load_feature_schema


def _synthetic_dataset(schema, *, n_per_cohort: int = 40):
    """Build a small synthetic harmonized matrix with known per-cohort coverage."""
    rng = np.random.default_rng(0)
    cohorts = ("bp", "sz", "dr", "asp")
    n = n_per_cohort * len(cohorts)
    idx = pd.MultiIndex.from_tuples(
        [(c, f"p{i}") for c in cohorts for i in range(n_per_cohort)],
        names=("cohort", "patient_id"),
    )

    data = {f.id: rng.normal(size=n).astype(np.float32) for f in schema.features}
    X = pd.DataFrame(data, index=idx, dtype="float64")

    # Simulate realistic structural missingness:
    # - ASP has no biology (all biology features NaN)
    # - SZ has no MADRS (BP+DR only)
    # - DR has no PANSS (SZ only)
    asp_mask = X.index.get_level_values("cohort") == "asp"
    sz_mask = X.index.get_level_values("cohort") == "sz"
    non_sz_mask = X.index.get_level_values("cohort") != "sz"

    for col in X.columns:
        feat = schema.features_by_id()[col]
        if feat.block == "biology":
            X.loc[asp_mask, col] = np.nan
        if col == "inst_madrs_total":
            X.loc[sz_mask, col] = np.nan
            X.loc[asp_mask, col] = np.nan
        if feat.block == "psychosis":
            X.loc[non_sz_mask, col] = np.nan

    metadata = pd.DataFrame(
        {
            "cohort": [c for c, _ in idx],
            "patient_id": [p for _, p in idx],
            "dsm_diagnosis": [c.upper() for c, _ in idx],
        },
        index=idx,
    )
    return X, metadata


def test_compute_per_cohort_coverage_shape():
    schema = load_feature_schema()
    X, metadata = _synthetic_dataset(schema)
    coverage = compute_per_cohort_coverage(X, metadata)
    assert coverage.shape == (len(schema.features), 4)
    assert set(coverage.columns) == {"bp", "sz", "dr", "asp"}


def test_per_cohort_coverage_reflects_simulated_missingness():
    schema = load_feature_schema()
    X, metadata = _synthetic_dataset(schema)
    coverage = compute_per_cohort_coverage(X, metadata)

    # ASP biology coverage = 0
    assert coverage.loc["bio_bmi", "asp"] == 0.0
    # SZ + ASP have no MADRS
    assert coverage.loc["inst_madrs_total", "sz"] == 0.0
    assert coverage.loc["inst_madrs_total", "asp"] == 0.0
    # BP + DR have MADRS
    assert coverage.loc["inst_madrs_total", "bp"] == 1.0
    assert coverage.loc["inst_madrs_total", "dr"] == 1.0
    # Non-SZ have no PANSS
    assert coverage.loc["inst_panss_total", "bp"] == 0.0
    assert coverage.loc["inst_panss_total", "sz"] == 1.0


def test_transdiagnostic_selection_excludes_sparse_features():
    schema = load_feature_schema()
    X, metadata = _synthetic_dataset(schema)
    fs = select_transdiagnostic_features(X, metadata, schema)

    # MADRS is absent in SZ+ASP (≥0 coverage) → must be excluded
    assert "inst_madrs_total" not in fs.feature_ids
    # PANSS is SZ-only → excluded
    assert "inst_panss_total" not in fs.feature_ids
    # Biology is ASP-missing → excluded
    assert "bio_bmi" not in fs.feature_ids

    # Age / sex have full coverage in all cohorts → admitted
    assert "demo_age_years" in fs.feature_ids
    assert "demo_sex_male" in fs.feature_ids


def test_transdiagnostic_graph_has_non_negative_edges_and_valid_overlap():
    schema = load_feature_schema()
    X, metadata = _synthetic_dataset(schema, n_per_cohort=20)
    result = build_transdiagnostic_graph(X, metadata, schema, k=5)

    assert result.feature_set.n_selected > 0
    # Every edge must satisfy the overlap constraint
    for src, dst, sim, overlap, dist in result.edges:
        assert overlap >= result.min_shared_features
        assert np.isfinite(sim)
        assert np.isfinite(dist)
        assert src != dst


def test_transdiagnostic_min_shared_respects_config_fraction():
    schema = load_feature_schema()
    X, metadata = _synthetic_dataset(schema)
    fs = select_transdiagnostic_features(X, metadata, schema)
    result = build_transdiagnostic_graph(X, metadata, schema, feature_set=fs)
    cfg = schema.transdiagnostic_selection
    expected = max(1, int(np.ceil(cfg.min_shared_features_fraction * fs.n_selected)))
    assert result.min_shared_features == expected


def test_transdiagnostic_excluded_by_config_respected():
    schema = load_feature_schema()
    X, metadata = _synthetic_dataset(schema)
    # Force-exclude a feature that would otherwise pass
    from face_stratification.harmonization.feature_schema import (
        FeatureSchema,
        TransdiagnosticSelectionConfig,
    )
    tight_cfg = TransdiagnosticSelectionConfig(
        min_cohort_coverage=0.5,
        min_shared_features_fraction=0.75,
        metric="cosine",
        excluded_feature_ids=("demo_age_years",),
    )
    # Build a shallow copy of the schema with the tighter config
    tight_schema = FeatureSchema(
        version=schema.version,
        blocks=schema.blocks,
        features=schema.features,
        transdiagnostic_selection=tight_cfg,
    )
    fs = select_transdiagnostic_features(X, metadata, tight_schema)
    assert "demo_age_years" not in fs.feature_ids
    assert "demo_age_years" in fs.excluded_by_config
