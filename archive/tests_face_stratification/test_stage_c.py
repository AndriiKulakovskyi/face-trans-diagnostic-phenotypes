"""Tests for Stage C: algorithms, consensus, comparison, ablation, narrative.

The pure-synthetic tests cover the mathematical primitives (Hungarian
alignment, co-association matrix, consensus partition, per-patient
confidence, chi-square, Cramér's V, BH-FDR enrichment passthrough). The
end-to-end tests run on a small real-data slice when the FACE CSVs are
available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from face_stratification.stage_c.algorithms import (
    ALGORITHMS,
    run_algorithm,
    run_gmm,
    run_kmeans,
    run_spectral,
    run_ward,
)
from face_stratification.stage_c.comparison import (
    chi_square_independence,
    cramers_v,
    full_dsm_comparison,
    per_cluster_cohort_entropy,
    per_cohort_purity,
)
from face_stratification.stage_c.consensus import (
    align_labels_to_reference,
    build_coassociation_matrix,
    compute_per_patient_confidence,
    consensus_partition,
    run_consensus_clustering,
)
from face_stratification.stage_c.ablation import (
    compute_optimization_score,
    pick_best_configuration,
    run_algorithm_k_grid,
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


def _toy_embedding(n_per: int = 30, k: int = 4, noise: float = 0.3, seed: int = 0) -> pd.DataFrame:
    """A clearly-clustered synthetic embedding for clustering tests."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 5, size=(k, 6))
    rows = []
    index_tuples = []
    for cid in range(k):
        for i in range(n_per):
            rows.append(centers[cid] + rng.normal(0, noise, size=6))
            index_tuples.append(("bp", f"c{cid}_p{i}"))
    arr = np.asarray(rows)
    idx = pd.MultiIndex.from_tuples(index_tuples, names=("cohort", "patient_id"))
    return pd.DataFrame(arr, index=idx, columns=[f"x{i}" for i in range(6)])


# ─── Algorithm wrappers ──────────────────────────────────────────────────────


def test_run_kmeans_returns_valid_assignment():
    emb = _toy_embedding()
    a = run_kmeans(emb, n_clusters=4, random_state=0)
    assert a.n_clusters == 4
    assert len(a.labels) == len(emb)
    assert set(a.labels.unique()) == set(range(4))


def test_run_gmm_returns_valid_assignment():
    emb = _toy_embedding()
    a = run_gmm(emb, n_clusters=4, random_state=0, n_init=2)
    assert a.n_clusters == 4
    assert "log_likelihood" in a.config
    assert "bic" in a.config


def test_run_ward_is_deterministic():
    emb = _toy_embedding()
    a1 = run_ward(emb, n_clusters=4)
    a2 = run_ward(emb, n_clusters=4)
    np.testing.assert_array_equal(a1.labels.values, a2.labels.values)


def test_run_spectral_returns_valid_assignment():
    emb = _toy_embedding(n_per=20)
    a = run_spectral(emb, n_clusters=4, random_state=0, n_neighbors=10)
    assert a.n_clusters == 4
    assert len(a.labels) == len(emb)


def test_run_algorithm_dispatch_works_for_each():
    emb = _toy_embedding(n_per=20)
    for name in ("kmeans", "gmm", "ward"):
        a = run_algorithm(name, emb, n_clusters=4, random_state=0)
        assert a.n_clusters == 4
        assert "runtime_seconds" in a.config


# ─── Hungarian label alignment ──────────────────────────────────────────────


def test_align_labels_to_reference_roundtrip():
    """Align an arbitrarily-relabelled clustering and verify perfect agreement."""
    rng = np.random.default_rng(0)
    n = 100
    n_clusters = 5
    reference = rng.integers(0, n_clusters, size=n)
    permutation = np.array([2, 4, 0, 3, 1])
    relabelled = permutation[reference]
    aligned = align_labels_to_reference(relabelled, reference, n_clusters=n_clusters)
    np.testing.assert_array_equal(aligned, reference)


def test_align_labels_handles_imperfect_match():
    """When there's noise, the alignment should still be the best possible."""
    rng = np.random.default_rng(0)
    n = 200
    n_clusters = 4
    reference = rng.integers(0, n_clusters, size=n)
    permutation = np.array([3, 2, 0, 1])
    relabelled = permutation[reference]
    # Add 10% noise
    flip = rng.choice(n, size=20, replace=False)
    relabelled[flip] = (relabelled[flip] + 1) % n_clusters
    aligned = align_labels_to_reference(relabelled, reference, n_clusters=n_clusters)
    # At least 80% should still match
    assert (aligned == reference).mean() >= 0.85


# ─── Co-association matrix ──────────────────────────────────────────────────


def test_coassociation_matrix_diagonal_is_one():
    n = 30
    df = pd.DataFrame({
        "a": np.array([0] * 15 + [1] * 15),
        "b": np.array([0] * 10 + [1] * 5 + [1] * 15),
    })
    M = build_coassociation_matrix(df)
    assert M.shape == (n, n)
    assert np.allclose(np.diag(M), 1.0)


def test_coassociation_matrix_perfect_agreement_yields_block_structure():
    """Two perfectly-agreeing clusterings → block-diagonal co-association."""
    labels = np.array([0] * 5 + [1] * 5)
    df = pd.DataFrame({"a": labels, "b": labels})
    M = build_coassociation_matrix(df)
    # Cluster 0 block
    assert (M[:5, :5] == 1.0).all()
    # Cluster 1 block
    assert (M[5:, 5:] == 1.0).all()
    # Off-diagonal blocks
    assert (M[:5, 5:] == 0.0).all()
    assert (M[5:, :5] == 0.0).all()


def test_coassociation_matrix_disagreeing_clusterings_have_intermediate_values():
    labels_a = np.array([0, 0, 1, 1])
    labels_b = np.array([0, 1, 0, 1])  # 50% agreement
    df = pd.DataFrame({"a": labels_a, "b": labels_b})
    M = build_coassociation_matrix(df)
    # M[0,1] = 1/2 (same in a, different in b)
    assert M[0, 1] == pytest.approx(0.5)
    # M[0,2] = 1/2
    assert M[0, 2] == pytest.approx(0.5)


def test_consensus_partition_recovers_obvious_clusters():
    # Build a co-association where two clear clusters exist
    n = 10
    M = np.eye(n, dtype=np.float32)
    M[:5, :5] = 1.0
    M[5:, 5:] = 1.0
    labels = consensus_partition(M, n_clusters=2)
    # The two halves should be in different clusters
    assert labels[0] == labels[1] == labels[2] == labels[3] == labels[4]
    assert labels[5] == labels[6] == labels[7] == labels[8] == labels[9]
    assert labels[0] != labels[5]


# ─── Per-patient confidence ──────────────────────────────────────────────────


def test_per_patient_confidence_high_for_pure_clusters():
    """A perfect 2-cluster co-association → confidence ≈ +1 for everyone."""
    n = 10
    M = np.eye(n, dtype=np.float32)
    M[:5, :5] = 1.0
    M[5:, 5:] = 1.0
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    conf = compute_per_patient_confidence(M, labels)
    assert (conf >= 0.99).all()


def test_per_patient_confidence_low_for_random_labels():
    """Random labels on a perfect M → low or negative confidence."""
    n = 10
    M = np.eye(n, dtype=np.float32)
    M[:5, :5] = 1.0
    M[5:, 5:] = 1.0
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # alternating
    conf = compute_per_patient_confidence(M, labels)
    # Should be close to zero or negative
    assert conf.mean() < 0.2


# ─── End-to-end consensus ────────────────────────────────────────────────────


def test_run_consensus_clustering_end_to_end():
    n = 20
    index = pd.MultiIndex.from_tuples(
        [("bp", f"p{i}") for i in range(n)], names=("cohort", "patient_id")
    )
    # Three base clusterings, all agreeing on 2 clusters
    base = {
        "a": np.array([0] * 10 + [1] * 10),
        "b": np.array([1] * 10 + [0] * 10),  # permuted but same partition
        "c": np.array([0] * 10 + [1] * 10),
    }
    result = run_consensus_clustering(
        base, n_clusters=2, embedding_index=index, keep_matrix=True
    )
    assert result.n_base_clusterings == 3
    # All three pairwise ARIs should be 1
    assert np.allclose(result.algorithm_pairwise_ari.values, 1.0)
    # Consensus should perfectly separate the two halves
    labels = result.labels.values
    assert labels[0] == labels[1] and labels[10] != labels[0]


# ─── Comparison statistics ───────────────────────────────────────────────────


def test_chi_square_detects_strong_association():
    # Cluster perfectly aligned with cohort → very high chi²
    cluster = np.array([0] * 50 + [1] * 50)
    cohort = np.array(["bp"] * 50 + ["sz"] * 50)
    chi2, dof, p, _exp = chi_square_independence(cluster, cohort)
    assert chi2 > 50
    assert p < 0.001
    assert dof == 1


def test_chi_square_detects_independence():
    rng = np.random.default_rng(0)
    cluster = rng.integers(0, 4, size=200)
    cohort = rng.choice(["bp", "sz", "dr", "asp"], size=200)
    chi2, dof, p, _exp = chi_square_independence(cluster, cohort)
    assert p > 0.01  # Random → not significant
    assert dof == 9


def test_cramers_v_perfect_alignment():
    cluster = np.array([0] * 25 + [1] * 25)
    cohort = np.array(["bp"] * 25 + ["sz"] * 25)
    v = cramers_v(cluster, cohort)
    assert v == pytest.approx(1.0, abs=1e-6)


def test_cramers_v_independence():
    rng = np.random.default_rng(0)
    cluster = rng.integers(0, 4, size=400)
    cohort = rng.choice(["bp", "sz", "dr", "asp"], size=400)
    v = cramers_v(cluster, cohort)
    assert v < 0.15  # Should be small under independence


def test_per_cluster_cohort_entropy_zero_for_pure():
    cluster = np.array([0, 0, 0, 0])
    cohort = np.array(["bp", "bp", "bp", "bp"])
    entropy, td_score = per_cluster_cohort_entropy(cluster, cohort)
    assert entropy[0] == 0.0
    assert td_score[0] == 0.0


def test_per_cluster_cohort_entropy_max_for_balanced_pair():
    cluster = np.array([0, 0, 0, 0])
    cohort = np.array(["bp", "sz", "bp", "sz"])
    entropy, td_score = per_cluster_cohort_entropy(cluster, cohort)
    # log2(2) = 1.0
    assert entropy[0] == pytest.approx(1.0, abs=1e-6)
    # max entropy for 2 cohorts is log2(2) = 1, so td_score = 1
    assert td_score[0] == pytest.approx(1.0, abs=1e-6)


def test_full_dsm_comparison_returns_consistent_metrics():
    cluster = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    cohort = np.array(["bp", "bp", "sz", "bp", "sz", "sz", "dr", "dr", "asp"])
    cmp = full_dsm_comparison(cluster, cohort)
    assert cmp.n_clusters == 3
    assert cmp.n_cohorts == 4
    assert 0.0 <= cmp.cramers_v <= 1.0
    assert cmp.contingency.shape == (3, 4)
    # Sum of contingency = total patients
    assert cmp.contingency.values.sum() == 9


# ─── Optimization score ──────────────────────────────────────────────────────


def test_optimization_score_prefers_high_silhouette():
    row_good = pd.Series({
        "silhouette": 0.6, "davies_bouldin": 1.0,
        "mean_transdiagnostic_score": 0.5, "ari": 0.1,
    })
    row_bad = pd.Series({
        "silhouette": 0.2, "davies_bouldin": 2.0,
        "mean_transdiagnostic_score": 0.5, "ari": 0.1,
    })
    assert compute_optimization_score(row_good) > compute_optimization_score(row_bad)


def test_pick_best_configuration_returns_top_row():
    df = pd.DataFrame([
        {"algorithm": "a", "k": 4, "silhouette": 0.3, "davies_bouldin": 1.5, "mean_transdiagnostic_score": 0.5, "ari": 0.1},
        {"algorithm": "b", "k": 5, "silhouette": 0.7, "davies_bouldin": 1.0, "mean_transdiagnostic_score": 0.6, "ari": 0.1},
        {"algorithm": "c", "k": 6, "silhouette": 0.4, "davies_bouldin": 1.2, "mean_transdiagnostic_score": 0.4, "ari": 0.2},
    ])
    best = pick_best_configuration(df)
    assert best["algorithm"] == "b"
    assert "optimization_score" in best


# ─── End-to-end ablation grid (real data) ───────────────────────────────────


def test_algorithm_k_grid_runs_on_real_data():
    csvs = _require_csvs()
    from face_stratification import build_harmonized_dataset, ConcatenatedEmbedding, fit_embedding

    ds = build_harmonized_dataset(csv_paths=csvs, max_rows_per_cohort=25)
    model = ConcatenatedEmbedding.build_default(
        pca_dim=4, td_spectral_dim=6, multiplex_spectral_dim=8
    )
    emb, _ = fit_embedding(ds, model=model)
    grid = run_algorithm_k_grid(
        emb.values,
        reference_labels=ds.metadata["cohort"].values,
        k_values=(4, 6),
        algorithms=("kmeans", "ward"),
    )
    # 2 algorithms × 2 ks × 1 seed = 4 rows
    assert len(grid) == 4
    assert {"silhouette", "ari", "davies_bouldin", "mean_cluster_entropy_bits"}.issubset(grid.columns)
