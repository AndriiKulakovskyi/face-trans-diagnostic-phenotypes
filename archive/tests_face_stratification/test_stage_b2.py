"""Tests for Stage B2 (GCN primitives + GAE + GraphContrastive) + biomarker validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from face_stratification.stage_b2.gcn import (
    GCNEncoder,
    SparseGCNLayer,
    build_multiplex_adjacency_from_nx,
    normalize_adjacency,
)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not available")


# ─── Sparse GCN primitives ───────────────────────────────────────────────────


def test_normalize_adjacency_produces_valid_sparse_tensor():
    A = sp.csr_matrix(np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float64))
    norm = normalize_adjacency(A)
    assert norm.is_sparse
    dense = norm.to_dense()
    # Diagonal should be non-zero (self-loops added)
    assert (torch.diag(dense) > 0).all()
    # Symmetric
    assert torch.allclose(dense, dense.T, atol=1e-6)
    # All values finite
    assert torch.isfinite(dense).all()


def test_sparse_gcn_layer_forward_pass():
    n, in_dim, out_dim = 5, 4, 3
    A = sp.csr_matrix(np.ones((n, n)) - np.eye(n))
    norm_adj = normalize_adjacency(A)
    layer = SparseGCNLayer(in_dim=in_dim, out_dim=out_dim)
    H = torch.randn(n, in_dim)
    out = layer(H, norm_adj)
    assert out.shape == (n, out_dim)
    assert torch.isfinite(out).all()


def test_gcn_encoder_outputs_l2_normalized_rows():
    n, in_dim = 10, 5
    A = sp.csr_matrix(np.ones((n, n)) - np.eye(n))
    norm_adj = normalize_adjacency(A)
    encoder = GCNEncoder(in_dim=in_dim, hidden_dim=8, out_dim=4, l2_normalize=True)
    encoder.eval()
    H = torch.randn(n, in_dim)
    with torch.no_grad():
        Z = encoder(H, norm_adj)
    assert Z.shape == (n, 4)
    row_norms = torch.linalg.norm(Z, dim=1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5)


def test_build_adjacency_from_networkx_multigraph():
    import networkx as nx
    G = nx.MultiGraph()
    for i in range(5):
        G.add_node(i)
    G.add_edge(0, 1, block="mood", weight=0.5)
    G.add_edge(0, 1, block="biology", weight=0.5)  # parallel edge
    G.add_edge(1, 2, block="mood", weight=0.3)

    A = build_multiplex_adjacency_from_nx(G, n_nodes=5, combine="sum")
    assert A.shape == (5, 5)
    # (0,1) should have weight 0.5 + 0.5 = 1.0
    assert A[0, 1] == pytest.approx(1.0)
    assert A[1, 0] == pytest.approx(1.0)
    # (1,2) has weight 0.3
    assert A[1, 2] == pytest.approx(0.3)


def test_build_adjacency_max_pool():
    import networkx as nx
    G = nx.MultiGraph()
    G.add_node(0)
    G.add_node(1)
    G.add_edge(0, 1, block="mood", weight=0.3)
    G.add_edge(0, 1, block="biology", weight=0.9)
    A = build_multiplex_adjacency_from_nx(G, n_nodes=2, combine="max")
    assert A[0, 1] == pytest.approx(0.9)


# ─── Contrastive augmentations ───────────────────────────────────────────────


def test_drop_edges_preserves_symmetry():
    from face_stratification.stage_b2.contrastive import _drop_edges
    rng = np.random.default_rng(0)
    A = sp.csr_matrix(np.ones((10, 10)) - np.eye(10))
    A_aug = _drop_edges(A, p_edge=0.3, rng=rng)
    dense = A_aug.toarray()
    assert np.array_equal(dense, dense.T), "Edge-drop must preserve symmetry"
    # Some edges should be removed
    assert (dense.sum() < A.toarray().sum())


def test_mask_features_zeros_columns_only():
    from face_stratification.stage_b2.contrastive import _mask_features
    rng = np.random.default_rng(0)
    X = torch.ones(10, 20)
    X_aug = _mask_features(X, p_feat=0.5, rng=rng)
    # Columns should be either fully 1 or fully 0
    col_sums = X_aug.sum(dim=0)
    assert ((col_sums == 10) | (col_sums == 0)).all()


def test_nt_xent_loss_finite_and_positive():
    from face_stratification.stage_b2.contrastive import _nt_xent_loss
    torch.manual_seed(0)
    z1 = torch.randn(8, 4)
    z2 = torch.randn(8, 4)
    loss = _nt_xent_loss(z1, z2, temperature=0.5)
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_nt_xent_loss_low_for_aligned_views():
    """If the two views are identical, NT-Xent should be relatively low."""
    from face_stratification.stage_b2.contrastive import _nt_xent_loss
    torch.manual_seed(0)
    z1 = torch.randn(16, 4)
    aligned_loss = _nt_xent_loss(z1, z1, temperature=0.5).item()
    random_loss = _nt_xent_loss(z1, torch.randn(16, 4), temperature=0.5).item()
    assert aligned_loss < random_loss


# ─── Clinical-feature panel validation ─────────────────────────────────────
# (legacy name: "biomarker panel" — see stage_c.clinical_panels for why the
# terminology was retired)


def test_clinical_feature_panel_evaluate_auc_on_new_data():
    from face_stratification.stage_c.clinical_panels import (
        EMBEDDING_INPUT_FEATURES,
        discover_clinical_feature_panel,
    )
    from face_stratification import load_feature_schema

    schema = load_feature_schema()
    rng = np.random.default_rng(0)
    n = 200
    index = pd.MultiIndex.from_tuples(
        [("bp", f"p{i}") for i in range(n)], names=("cohort", "patient_id")
    )
    data = {f.id: rng.normal(size=n) for f in schema.features}
    X = pd.DataFrame(data, index=index)
    cluster = pd.Series([0] * 100 + [1] * 100, index=index, dtype=int)
    cohort = pd.Series(["bp"] * n, index=index)

    # Plant a strong cluster effect on BMI (which is NOT in
    # EMBEDDING_INPUT_FEATURES, so the sanitised default whitelist still
    # sees it).
    assert "bio_bmi" not in EMBEDDING_INPUT_FEATURES
    X.loc[X.index[:100], "bio_bmi"] += 10

    panel = discover_clinical_feature_panel(
        X, cluster, cohort, target_cluster=0,
        max_panel_size=3,
        min_univariate_auc=0.55,
    )
    # Leakage-safe default
    assert panel.whitelist_excludes_embedding_inputs is True
    # predict_proba must run on new data
    scores = panel.predict_proba(X)
    assert scores.shape == (n,)
    assert (scores >= 0).all() and (scores <= 1).all()
    # Held-out AUC should be reasonable
    y_true = (cluster.to_numpy() == 0).astype(int)
    auc = panel.evaluate_auc(X, y_true)
    assert 0.5 <= auc <= 1.0


def test_clinical_feature_panel_cv_runs_end_to_end():
    from face_stratification.stage_c.clinical_panels import (
        validate_clinical_feature_panel_cv,
    )
    from face_stratification import load_feature_schema

    schema = load_feature_schema()
    rng = np.random.default_rng(0)
    n = 300
    index = pd.MultiIndex.from_tuples(
        [("bp", f"p{i}") for i in range(n)], names=("cohort", "patient_id")
    )
    data = {f.id: rng.normal(size=n) for f in schema.features}
    X = pd.DataFrame(data, index=index)
    # Plant effect on BMI (still in the sanitised whitelist)
    X.loc[X.index[:150], "bio_bmi"] += 10

    cluster = pd.Series([0] * 150 + [1] * 150, index=index, dtype=int)
    cohort = pd.Series(["bp"] * n, index=index)

    result = validate_clinical_feature_panel_cv(
        X, cluster, cohort, target_cluster=0,
        n_splits=3, test_fraction=0.3,
        max_panel_size=3,
    )
    assert result.n_splits >= 1
    assert 0.5 <= result.train_auc_mean <= 1.0
    assert 0.5 <= result.test_auc_mean <= 1.0
    # Train and test AUC shouldn't diverge wildly for this synthetic data
    assert abs(result.train_auc_mean - result.test_auc_mean) < 0.2
    assert result.whitelist_excludes_embedding_inputs is True


def test_legacy_biomarker_aliases_still_work_with_deprecation():
    import warnings

    from face_stratification.stage_c.biomarkers import (
        BiomarkerPanel,
        discover_biomarker_panel,
    )
    from face_stratification.stage_c.clinical_panels import (
        ClinicalFeaturePanel,
    )

    # BiomarkerPanel IS ClinicalFeaturePanel (same class, just an alias).
    assert BiomarkerPanel is ClinicalFeaturePanel

    # discover_biomarker_panel is a deprecated wrapper that warns then
    # delegates. We only test that the warning fires.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            discover_biomarker_panel(
                pd.DataFrame({"x": [0.0, 1.0]}),
                pd.Series([0, 1]),
                pd.Series(["bp", "bp"]),
                target_cluster=0,
            )
        except Exception:
            pass  # the tiny frame will raise; we only care about the warning
        assert any(
            issubclass(wi.category, DeprecationWarning) for wi in w
        ), "discover_biomarker_panel should emit a DeprecationWarning"


# ─── Stage B2.5 sweep infrastructure ────────────────────────────────────────


def test_gcn_encoder_variable_depth_matches_expected_params():
    """Each additional layer must add the right number of parameters."""
    params = {}
    for depth in (1, 2, 3, 4):
        enc = GCNEncoder(in_dim=5, hidden_dim=8, out_dim=4, n_layers=depth)
        params[depth] = sum(p.numel() for p in enc.parameters())
    # Deeper encoder must always have more parameters
    assert params[1] < params[2] < params[3] < params[4]
    # Sanity: depth 1 is just the final layer (5→4: W=20 + b=4 = 24)
    assert params[1] == 24


def test_filter_adjacency_by_include_edge_types():
    import networkx as nx
    G = nx.MultiGraph()
    for i in range(4):
        G.add_node(i)
    G.add_edge(0, 1, block="mood", weight=1.0)
    G.add_edge(0, 2, block="transdiagnostic", weight=1.0)
    G.add_edge(1, 2, block="psychosis", weight=1.0)

    A_all = build_multiplex_adjacency_from_nx(G, n_nodes=4)
    A_trans = build_multiplex_adjacency_from_nx(
        G, n_nodes=4, include_edge_types=("transdiagnostic",)
    )
    A_no_psych = build_multiplex_adjacency_from_nx(
        G, n_nodes=4, exclude_edge_types=("psychosis",)
    )
    assert A_all.nnz // 2 == 3
    assert A_trans.nnz // 2 == 1
    assert A_no_psych.nnz // 2 == 2


def test_sweep_config_id_is_descriptive():
    from face_stratification.stage_b2 import SweepConfig
    c = SweepConfig(
        model="contrastive", n_layers=3, hidden_dim=32, out_dim=16,
        temperature=0.1, p_edge=0.3, include_edge_types=("transdiagnostic",),
    )
    cid = c.config_id()
    # All the important hyperparameters should appear in the id
    assert "contrastive" in cid
    assert "L=3" in cid
    assert "h=32" in cid
    assert "T=0.1" in cid
    assert "transdiagnostic" in cid


def test_compute_transdiagnostic_score_rewards_entropy_over_dsm():
    """A config with higher entropy and lower V should score higher."""
    from face_stratification.stage_b2 import compute_transdiagnostic_score

    high_trans = compute_transdiagnostic_score(
        silhouette=0.5, davies_bouldin=1.0,
        transdiagnostic_score=0.8, cramers_v=0.2,
    )
    low_trans = compute_transdiagnostic_score(
        silhouette=0.5, davies_bouldin=1.0,
        transdiagnostic_score=0.3, cramers_v=0.7,
    )
    assert high_trans > low_trans


def test_pick_best_transdiagnostic_config_returns_highest_score():
    from face_stratification.stage_b2 import pick_best_transdiagnostic_config
    df = pd.DataFrame([
        {"config_id": "a", "optimization_score": 1.5, "silhouette": 0.4},
        {"config_id": "b", "optimization_score": 2.1, "silhouette": 0.5},
        {"config_id": "c", "optimization_score": 1.9, "silhouette": 0.45},
    ])
    best = pick_best_transdiagnostic_config(df)
    assert best["config_id"] == "b"
    assert best["optimization_score"] == 2.1
