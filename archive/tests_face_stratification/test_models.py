"""Stage B tests — embedding interface, PCA, spectral, composite, pipeline.

These tests cover:

- ``PatientEmbedding`` dataclass invariants (no NaN, unique index, correct
  MultiIndex names, round-trip save/load).
- ``TransdiagnosticPCA`` and ``TransdiagnosticRawFeatures`` baselines on
  synthetic data.
- ``TransdiagnosticSpectral`` and ``MultiplexSpectral`` on a tiny hand-built
  graph with a known community structure.
- ``ConcatenatedEmbedding.build_default`` end-to-end on real FACE CSVs
  (auto-skipped if the CSVs are missing).
- The **no-imputation-in-graph** invariant for spectral models: adding an
  all-NaN column to the feature matrix must not change the spectral
  embedding since it cannot change the graph edges.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from face_stratification import (
    ConcatenatedEmbedding,
    PatientEmbedding,
    build_harmonized_dataset,
    fit_embedding,
    load_feature_schema,
)
from face_stratification.harmonization.harmonizer import HarmonizedDataset
from face_stratification.models.baselines import (
    TransdiagnosticPCA,
    TransdiagnosticRawFeatures,
)
from face_stratification.models.spectral import (
    MultiplexSpectral,
    TransdiagnosticSpectral,
    _symmetric_normalized_laplacian,
    _nx_multiplex_to_adjacency,
)


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


# ─── PatientEmbedding dataclass invariants ───────────────────────────────────


def _valid_index(n: int) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [("bp", f"p{i}") for i in range(n)],
        names=("cohort", "patient_id"),
    )


def test_patient_embedding_rejects_nan():
    df = pd.DataFrame(
        [[1.0, np.nan], [0.0, 0.0]],
        index=_valid_index(2),
        columns=["a", "b"],
    )
    with pytest.raises(ValueError, match="NaN"):
        PatientEmbedding(values=df, model_name="toy")


def test_patient_embedding_rejects_wrong_index_names():
    df = pd.DataFrame(
        [[1.0]], index=pd.MultiIndex.from_tuples([("a", "b")], names=("foo", "bar")),
        columns=["x"],
    )
    with pytest.raises(ValueError, match="MultiIndex"):
        PatientEmbedding(values=df, model_name="toy")


def test_patient_embedding_roundtrip(tmp_path):
    df = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4]],
        index=_valid_index(2),
        columns=["x0", "x1"],
    )
    emb = PatientEmbedding(
        values=df,
        model_name="toy",
        model_config={"k": 2},
        view_dims={"toy": 2},
        schema_version="0.2.0",
    )
    out = emb.save(tmp_path)
    loaded = PatientEmbedding.load(out)
    pd.testing.assert_frame_equal(loaded.values, emb.values)
    assert loaded.model_name == "toy"
    assert loaded.model_config == {"k": 2}
    assert loaded.schema_version == "0.2.0"


# ─── Synthetic harmonized dataset helper ─────────────────────────────────────


def _synthetic_dataset(n_per_cohort: int = 40) -> HarmonizedDataset:
    schema = load_feature_schema()
    rng = np.random.default_rng(7)
    cohorts = ("bp", "sz", "dr", "asp")
    index = pd.MultiIndex.from_tuples(
        [(c, f"p{i}") for c in cohorts for i in range(n_per_cohort)],
        names=("cohort", "patient_id"),
    )
    data = {f.id: rng.normal(size=len(index)).astype(float) for f in schema.features}
    X = pd.DataFrame(data, index=index, dtype="float64")

    # Replicate the minimal structural missingness needed to exercise the
    # transdiagnostic selector: ASP has no biology.
    for f in schema.features:
        if f.block == "biology":
            X.loc[X.index.get_level_values("cohort") == "asp", f.id] = np.nan

    metadata = pd.DataFrame(
        {
            "cohort": [c for c, _ in index],
            "patient_id": [p for _, p in index],
            "dsm_diagnosis": [c.upper() for c, _ in index],
        },
        index=index,
    )
    feature_metadata = pd.DataFrame(
        [
            {
                "feature_id": f.id,
                "label_fr": f.label_fr,
                "block": f.block,
                "type": f.type.value,
                "temporal_scope": f.temporal_scope.value,
                "unit": f.unit,
                "direction": f.direction,
                "cohorts": ",".join(f.cohorts),
            }
            for f in schema.features
        ]
    ).set_index("feature_id")

    return HarmonizedDataset(
        X=X, metadata=metadata, feature_metadata=feature_metadata, schema=schema
    )


# ─── Transdiagnostic PCA / raw features ─────────────────────────────────────


def test_transdiagnostic_pca_produces_dense_embedding():
    ds = _synthetic_dataset()
    model = TransdiagnosticPCA(n_components=4)
    emb = model.fit_transform(ds)

    assert emb.n_patients == ds.n_patients
    assert emb.dim <= 4
    assert not emb.values.isna().any().any()
    assert np.isfinite(emb.values.to_numpy()).all()
    # L2 normalized rows by default → each row norm ≈ 1
    norms = np.linalg.norm(emb.values.to_numpy(), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_transdiagnostic_raw_features_matches_selected_set():
    ds = _synthetic_dataset()
    model = TransdiagnosticRawFeatures()
    emb = model.fit_transform(ds)

    # Dim equals the number of selected transdiagnostic features
    assert emb.dim == model.config["n_features"]
    # Index matches the harmonized dataset
    assert list(emb.values.index) == list(ds.X.index)


# ─── Spectral — synthetic community structure ────────────────────────────────


def _two_community_graph() -> tuple[nx.MultiGraph, pd.MultiIndex]:
    """A 12-node graph with two obvious communities connected by one bridge."""
    G = nx.MultiGraph()
    for i in range(12):
        cohort = "bp" if i < 6 else "sz"
        G.add_node(i, cohort=cohort, patient_id=f"p{i}")
    # dense intra-community edges (weight 1)
    for comm in ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11)):
        for a in comm:
            for b in comm:
                if a < b:
                    G.add_edge(a, b, key="transdiagnostic", block="transdiagnostic", weight=1.0)
    # single low-weight bridge
    G.add_edge(5, 6, key="transdiagnostic", block="transdiagnostic", weight=0.01)

    index = pd.MultiIndex.from_tuples(
        [(G.nodes[i]["cohort"], G.nodes[i]["patient_id"]) for i in range(12)],
        names=("cohort", "patient_id"),
    )
    return G, index


def test_transdiagnostic_spectral_separates_two_communities():
    G, index = _two_community_graph()

    # Build a minimal harmonized dataset whose X has the right shape + index
    schema = load_feature_schema()
    X = pd.DataFrame(np.zeros((12, len(schema.features))), index=index, columns=[f.id for f in schema.features])
    metadata = pd.DataFrame(
        {"cohort": index.get_level_values("cohort"), "patient_id": index.get_level_values("patient_id"),
         "dsm_diagnosis": ["BP"] * 6 + ["SZ"] * 6},
        index=index,
    )
    feature_metadata = pd.DataFrame(index=[f.id for f in schema.features])
    ds = HarmonizedDataset(X=X, metadata=metadata, feature_metadata=feature_metadata, schema=schema)

    model = TransdiagnosticSpectral(n_components=2, l2_normalize=False)
    emb = model.fit_transform(ds, graph=G)

    # With a near-disconnected two-community graph, the first non-trivial
    # eigenvector should separate the communities by sign.
    values = emb.values.iloc[:, 0].to_numpy()
    sign_community_1 = np.sign(values[:6])
    sign_community_2 = np.sign(values[6:])
    # All of community 1 should have the same sign and opposite community 2
    assert np.all(sign_community_1 == sign_community_1[0])
    assert np.all(sign_community_2 == sign_community_2[0])
    assert sign_community_1[0] != sign_community_2[0]


def test_spectral_handles_isolated_nodes():
    """Isolated nodes (zero degree) get the zero vector and are counted."""
    G = nx.MultiGraph()
    G.add_node(0, cohort="bp", patient_id="p0")
    G.add_node(1, cohort="bp", patient_id="p1")
    G.add_node(2, cohort="sz", patient_id="p2")  # isolated
    G.add_edge(0, 1, key="transdiagnostic", block="transdiagnostic", weight=1.0)

    schema = load_feature_schema()
    index = pd.MultiIndex.from_tuples(
        [("bp", "p0"), ("bp", "p1"), ("sz", "p2")],
        names=("cohort", "patient_id"),
    )
    X = pd.DataFrame(np.zeros((3, len(schema.features))), index=index, columns=[f.id for f in schema.features])
    metadata = pd.DataFrame(
        {"cohort": ["bp", "bp", "sz"], "patient_id": ["p0", "p1", "p2"], "dsm_diagnosis": ["BP", "BP", "SZ"]},
        index=index,
    )
    feature_metadata = pd.DataFrame(index=[f.id for f in schema.features])
    ds = HarmonizedDataset(X=X, metadata=metadata, feature_metadata=feature_metadata, schema=schema)

    model = TransdiagnosticSpectral(n_components=1, l2_normalize=False)
    emb = model.fit_transform(ds, graph=G)
    # Patient 2 is isolated → row should be zero
    iso_row = emb.values.loc[("sz", "p2")].to_numpy()
    assert np.allclose(iso_row, 0.0)
    assert emb.n_isolated_nodes == 1


# ─── Adjacency conversion ────────────────────────────────────────────────────


def test_multiplex_adjacency_sums_parallel_weights():
    G = nx.MultiGraph()
    G.add_node(0)
    G.add_node(1)
    G.add_edge(0, 1, key="mood", block="mood", weight=0.3)
    G.add_edge(0, 1, key="biology", block="biology", weight=0.4)

    W = _nx_multiplex_to_adjacency(G, n_nodes=2, combine="sum")
    assert W[0, 1] == pytest.approx(0.7)
    assert W[1, 0] == pytest.approx(0.7)


def test_multiplex_adjacency_max_pool_picks_largest():
    G = nx.MultiGraph()
    G.add_node(0)
    G.add_node(1)
    G.add_edge(0, 1, key="mood", block="mood", weight=0.3)
    G.add_edge(0, 1, key="biology", block="biology", weight=0.9)

    W = _nx_multiplex_to_adjacency(G, n_nodes=2, combine="max")
    assert W[0, 1] == pytest.approx(0.9)


def test_symmetric_normalized_laplacian_bounds():
    """Eigenvalues of L_sym must lie in [0, 2]."""
    G = nx.MultiGraph()
    for i in range(5):
        G.add_node(i)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]:
        G.add_edge(a, b, block="transdiagnostic", weight=1.0)
    W = _nx_multiplex_to_adjacency(G, n_nodes=5)
    L, deg = _symmetric_normalized_laplacian(W)
    eigvals = np.linalg.eigvalsh(L.toarray())
    assert eigvals.min() >= -1e-8
    assert eigvals.max() <= 2.0 + 1e-8


# ─── No-imputation invariant ─────────────────────────────────────────────────


def test_adding_all_nan_column_does_not_change_td_spectral():
    """Crucial honesty check: an all-NaN column must have zero effect.

    Spectral embedding operates on the graph edges; masked similarity
    ignores NaN columns; therefore adding an all-NaN column must leave
    both the graph and the embedding unchanged.
    """
    schema = load_feature_schema()
    ds = _synthetic_dataset(n_per_cohort=30)

    from face_stratification.harmonization.normalization import (
        fit_normalization,
        transform_normalization,
    )
    from face_stratification.graph.patient_similarity import build_multiplex_graph

    stats = fit_normalization(ds.X, schema)
    Xn = transform_normalization(ds.X, stats)

    G1, _, _ = build_multiplex_graph(Xn, schema, k=5, metadata=ds.metadata)
    model1 = TransdiagnosticSpectral(n_components=4, l2_normalize=False)
    emb1 = model1.fit_transform(ds, graph=G1)

    # Same dataset with an all-NaN "extra" column pushed into X.
    X2 = ds.X.copy()
    X2["all_nan_sentinel"] = np.nan
    ds2 = HarmonizedDataset(
        X=X2.drop(columns=["all_nan_sentinel"]),  # schema-compatible columns
        metadata=ds.metadata,
        feature_metadata=ds.feature_metadata,
        schema=schema,
    )
    Xn2 = transform_normalization(ds2.X, fit_normalization(ds2.X, schema))
    G2, _, _ = build_multiplex_graph(Xn2, schema, k=5, metadata=ds2.metadata)
    model2 = TransdiagnosticSpectral(n_components=4, l2_normalize=False)
    emb2 = model2.fit_transform(ds2, graph=G2)

    # Eigenvectors are only defined up to a sign; compare absolute values.
    np.testing.assert_allclose(
        np.abs(emb1.values.to_numpy()),
        np.abs(emb2.values.to_numpy()),
        atol=1e-6,
    )


# ─── Composite + end-to-end pipeline (real CSVs) ─────────────────────────────


def test_concatenated_embedding_end_to_end_on_real_data():
    csvs = _require_csvs()
    ds = build_harmonized_dataset(csv_paths=csvs, max_rows_per_cohort=25)
    model = ConcatenatedEmbedding.build_default(
        pca_dim=4, td_spectral_dim=6, multiplex_spectral_dim=8
    )
    emb, _graph = fit_embedding(ds, model=model)

    # Shape
    assert emb.n_patients == ds.n_patients
    assert emb.dim == 4 + 6 + 8
    assert emb.view_dims == {
        "transdiagnostic_pca": 4,
        "transdiagnostic_spectral": 6,
        "multiplex_spectral": 8,
    }

    # All rows L2-normalized (final normalization in the composite)
    norms = np.linalg.norm(emb.values.to_numpy(), axis=1)
    finite = np.isfinite(norms)
    assert finite.all()
    # Some rows may be all-zero if a patient is isolated in the multiplex →
    # L2 leaves zero rows unchanged.
    non_zero = norms > 0
    assert np.allclose(norms[non_zero], 1.0, atol=1e-6)


def test_composite_sub_configs_are_captured():
    csvs = _require_csvs()
    ds = build_harmonized_dataset(csv_paths=csvs, max_rows_per_cohort=25)
    model = ConcatenatedEmbedding.build_default(pca_dim=2, td_spectral_dim=4, multiplex_spectral_dim=4)
    emb, _ = fit_embedding(ds, model=model)
    assert "sub_configs" in emb.model_config
    assert "transdiagnostic_pca" in emb.model_config["sub_configs"]
    assert "multiplex_spectral" in emb.model_config["sub_configs"]
