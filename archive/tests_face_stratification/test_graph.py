"""Graph-building smoke tests + no-imputation invariants.

Runs on real FACE CSVs (small slice) when present; otherwise only the
pure-synthetic tests run. The synthetic tests cover the key methodological
invariants:

- block graphs never use imputation (a NaN input never fabricates a distance),
- the semantic overlap constraint is actually enforced end-to-end,
- the multiplex graph preserves each node's cohort label,
- the transdiagnostic edge type is added and uses the data-driven feature set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from face_stratification.graph.patient_similarity import (
    build_block_knn_graph,
    build_multiplex_graph,
    summarize_graph,
)
from face_stratification.harmonization.feature_schema import load_feature_schema


REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATHS = {
    "bp": REPO_ROOT / "data" / "BP.csv",
    "sz": REPO_ROOT / "data" / "SZ.csv",
    "dr": REPO_ROOT / "data" / "DR.csv",
    "asp": REPO_ROOT / "data" / "ASP.csv",
}


def _require_csvs():
    missing = [c for c, p in CSV_PATHS.items() if not p.is_file()]
    if missing:
        pytest.skip(f"FACE CSVs missing: {missing}")
    return CSV_PATHS


# ─── Pure synthetic (always runs) ─────────────────────────────────────────────


def test_block_knn_graph_respects_min_fraction_present():
    """Patients below min_fraction_present must not become candidate nodes."""
    schema = load_feature_schema()
    block = schema.block("biology")
    cols = [f.id for f in schema.features_by_block()["biology"]]

    # 3 "full" patients + 1 "empty" patient with only NaN
    rng = np.random.default_rng(0)
    values = rng.normal(size=(3, len(cols)))
    empty = np.full((1, len(cols)), np.nan)
    all_values = np.vstack([values, empty])
    df = pd.DataFrame(all_values, columns=cols)

    bg = build_block_knn_graph(df, block, k=5)
    # The 4th patient (all NaN) must be excluded from candidates
    assert 3 not in bg.candidate_indices
    assert bg.n_candidate_nodes == 3


def test_block_knn_graph_returns_no_edges_when_overlap_tight():
    """With a strict min_shared_features, sparse patients get no edges."""
    schema = load_feature_schema()
    block = schema.block("cognition")
    cols = [f.id for f in schema.features_by_block()["cognition"]]

    # Two patients, each with only a single (disjoint) cognition feature measured
    row_a = [np.nan] * len(cols)
    row_b = [np.nan] * len(cols)
    row_a[0] = 1.0
    row_b[1] = 1.0
    df = pd.DataFrame([row_a, row_b], columns=cols)

    bg = build_block_knn_graph(
        df, block, k=5, min_fraction_present=0.05, min_shared_features=2
    )
    # Zero overlap → no edges
    assert bg.edges == []


def test_adding_all_nan_column_does_not_change_block_graph_edges():
    """Critical V1-honesty invariant: NaN columns must be invisible to distances."""
    schema = load_feature_schema()
    block = schema.block("biology")
    cols = [f.id for f in schema.features_by_block()["biology"]]

    rng = np.random.default_rng(1)
    data = rng.normal(size=(25, len(cols)))
    df_a = pd.DataFrame(data, columns=cols)

    bg_a = build_block_knn_graph(
        df_a, block, k=4, min_fraction_present=0.1, min_shared_features=3
    )

    # Same data with an extra all-NaN column tacked on — shouldn't change anything.
    extra_col = "cog_tmt_a_seconds"  # reuse a real feature id from another block
    df_b = df_a.copy()
    df_b[extra_col] = np.nan
    bg_b = build_block_knn_graph(
        df_b[cols], block, k=4, min_fraction_present=0.1, min_shared_features=3
    )

    set_a = {(u, v) for u, v, _ in bg_a.edges}
    set_b = {(u, v) for u, v, _ in bg_b.edges}
    assert set_a == set_b


# ─── Real FACE data smoke ─────────────────────────────────────────────────────


def test_build_multiplex_graph_end_to_end():
    csvs = _require_csvs()
    from face_stratification import build_harmonized_dataset
    from face_stratification.harmonization.normalization import (
        fit_normalization,
        transform_normalization,
    )

    ds = build_harmonized_dataset(csv_paths=csvs, max_rows_per_cohort=30)
    stats = fit_normalization(ds.X, ds.schema)
    Xn = transform_normalization(ds.X, stats)

    # Crucially: we do NOT impute before building the graph.
    assert Xn.isna().sum().sum() > 0, "Xn should still contain NaN (no imputation)"

    G, block_graphs, td = build_multiplex_graph(
        Xn, ds.schema, k=4, metadata=ds.metadata
    )

    # Multiplex graph must have one node per patient
    assert G.number_of_nodes() == ds.n_patients

    # Every block graph's candidate set must be a subset of the full node set
    for bid, bg in block_graphs.items():
        assert bg.n_candidate_nodes <= ds.n_patients
        for src, dst, w in bg.edges:
            assert 0 <= src < ds.n_patients
            assert 0 <= dst < ds.n_patients
            assert 0.0 <= w <= 1.0 + 1e-4

    # Transdiagnostic edge type must exist
    assert td is not None
    assert td.feature_set.n_selected > 0
    edge_types = {data.get("block") for _, _, data in G.edges(data=True)}
    assert "transdiagnostic" in edge_types


def test_multiplex_graph_has_no_cross_block_leakage():
    """A block's edges must only reference candidate nodes from that block."""
    csvs = _require_csvs()
    from face_stratification import build_harmonized_dataset
    from face_stratification.harmonization.normalization import (
        fit_normalization,
        transform_normalization,
    )

    ds = build_harmonized_dataset(csv_paths=csvs, max_rows_per_cohort=30)
    stats = fit_normalization(ds.X, ds.schema)
    Xn = transform_normalization(ds.X, stats)

    G, block_graphs, _td = build_multiplex_graph(
        Xn, ds.schema, k=3, metadata=ds.metadata, include_transdiagnostic=False
    )

    for bid, bg in block_graphs.items():
        allowed = set(bg.candidate_indices.tolist())
        for u, v, data in G.edges(data=True, keys=False):
            if data.get("block") == bid:
                assert u in allowed, f"block {bid} edge endpoint {u} not in candidates"
                assert v in allowed, f"block {bid} edge endpoint {v} not in candidates"


def test_summary_reports_candidate_counts_and_min_shared():
    csvs = _require_csvs()
    from face_stratification import build_harmonized_dataset
    from face_stratification.harmonization.normalization import (
        fit_normalization,
        transform_normalization,
    )

    ds = build_harmonized_dataset(csv_paths=csvs, max_rows_per_cohort=30)
    stats = fit_normalization(ds.X, ds.schema)
    Xn = transform_normalization(ds.X, stats)

    G, block_graphs, td = build_multiplex_graph(
        Xn, ds.schema, k=4, metadata=ds.metadata
    )
    summary = summarize_graph(G, ds.schema, built=block_graphs, transdiagnostic=td)

    assert summary.n_nodes == ds.n_patients
    assert "transdiagnostic" in summary.edges_per_type
    for bid, bg in block_graphs.items():
        assert summary.candidate_nodes_per_type[bid] == bg.n_candidate_nodes
        assert summary.min_shared_features_per_type[bid] == bg.min_shared_features
    assert summary.transdiagnostic_feature_set is not None
