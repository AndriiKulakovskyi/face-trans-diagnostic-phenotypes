"""Unit tests for the pairwise-complete masked similarity kernels.

These are pure synthetic tests — no FACE CSVs required. They cover the
mathematical correctness of cosine / Euclidean / Gower on NaN-bearing
matrices, the overlap-count output, and the semantic overlap constraint
enforced by :func:`masked_knn_edges`.
"""

from __future__ import annotations

import numpy as np
import pytest

from face_stratification.graph.masked_similarity import (
    masked_cosine,
    masked_euclidean,
    masked_gower,
    masked_knn_edges,
    masked_manhattan,
    masked_similarity,
)


def test_masked_cosine_identical_rows():
    """Two identical rows with a NaN in the same position have cosine = 1."""
    X = np.array([
        [1.0, 2.0, 3.0, np.nan],
        [1.0, 2.0, 3.0, np.nan],
    ], dtype=np.float32)
    r = masked_cosine(X)
    assert r.overlap[0, 1] == 3
    assert abs(r.similarity[0, 1] - 1.0) < 1e-5
    assert abs(r.distance[0, 1]) < 1e-5


def test_masked_cosine_orthogonal_rows():
    X = np.array([
        [1.0, 0.0, np.nan],
        [0.0, 1.0, np.nan],
    ], dtype=np.float32)
    r = masked_cosine(X)
    assert r.overlap[0, 1] == 2
    assert abs(r.similarity[0, 1]) < 1e-5


def test_masked_cosine_no_overlap_returns_minus_inf():
    """A pair with zero shared features must be excluded."""
    X = np.array([
        [1.0, np.nan, np.nan],
        [np.nan, 1.0, np.nan],
    ], dtype=np.float32)
    r = masked_cosine(X)
    assert r.overlap[0, 1] == 0
    assert r.similarity[0, 1] == -np.inf
    assert r.distance[0, 1] == np.inf


def test_masked_cosine_partial_overlap_uses_only_shared_features():
    """Patient A vs B shared on [0,1], not [2]. Cosine should ignore col 2."""
    X = np.array([
        [1.0, 2.0, 99.0],  # extreme value at col 2 should be ignored vs row 1
        [1.0, 2.0, np.nan],
    ], dtype=np.float32)
    r = masked_cosine(X)
    # Row 0 vs Row 1 overlap = {0, 1}
    assert r.overlap[0, 1] == 2
    # Cosine on cols {0, 1} is identical → 1.0
    assert abs(r.similarity[0, 1] - 1.0) < 1e-5


def test_masked_euclidean_distance_is_mean_rescaled():
    """Mean-rescaled distance: ||x - y|| over shared features / sqrt(n_shared)."""
    X = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0, 2.0],
    ], dtype=np.float32)
    r = masked_euclidean(X)
    # Σ diff² = 16; mean = 4; sqrt = 2
    assert abs(r.distance[0, 1] - 2.0) < 1e-5


def test_masked_euclidean_no_overlap():
    X = np.array([
        [1.0, np.nan],
        [np.nan, 2.0],
    ], dtype=np.float32)
    r = masked_euclidean(X)
    assert r.distance[0, 1] == np.inf
    assert r.similarity[0, 1] == -np.inf


def test_masked_gower_with_feature_ranges():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ], dtype=np.float32)
    # ranges = [2, 2] → per-feature contribution = 0.5 + 0.5, averaged = 0.5
    r = masked_gower(X, feature_ranges=np.array([2.0, 2.0], dtype=np.float32))
    assert abs(r.distance[0, 1] - 0.5) < 1e-5


def test_masked_similarity_dispatch_rejects_unknown_metric():
    X = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        masked_similarity(X, "klingon")  # type: ignore[arg-type]


def test_masked_similarity_dispatch_gower_requires_ranges():
    X = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        masked_similarity(X, "gower")


# ─── kNN + overlap constraint ─────────────────────────────────────────────────


def test_knn_respects_semantic_overlap_constraint():
    """A pair below the overlap threshold must never appear as an edge."""
    X = np.array([
        [1.0, 2.0, 3.0, 4.0],  # 4 observed
        [1.0, 2.0, 3.0, 4.0],  # 4 observed, identical
        [1.0, 2.0, np.nan, np.nan],  # 2 observed — overlap with 0,1 = 2
        [np.nan, np.nan, 3.0, 4.0],  # 2 observed — overlap with 2 = 0
    ], dtype=np.float32)

    # Threshold = 3 → pair (0,2) and (1,2) have overlap 2 < 3 → must NOT exist
    edges = masked_knn_edges(X, metric="cosine", k=4, min_shared_features=3)
    pairs = {(u, v) for u, v, *_ in edges}
    assert (0, 1) in pairs  # 4 shared → OK
    assert (0, 2) not in pairs
    assert (1, 2) not in pairs
    # Pair (2, 3) has overlap 0 → excluded
    assert (2, 3) not in pairs


def test_knn_returns_fewer_than_k_when_constraint_is_tight():
    """With a strict constraint some nodes legitimately have few neighbours."""
    X = np.array([
        [1.0, 2.0, np.nan, np.nan],
        [1.0, 2.0, np.nan, np.nan],
        [np.nan, np.nan, 3.0, 4.0],
        [np.nan, np.nan, 3.0, 4.0],
    ], dtype=np.float32)
    edges = masked_knn_edges(X, metric="cosine", k=3, min_shared_features=2)
    pairs = {(u, v) for u, v, *_ in edges}
    # Only (0,1) and (2,3) share enough features
    assert pairs == {(0, 1), (2, 3)}


def test_knn_edges_contain_positive_overlap_and_distance():
    X = np.random.default_rng(0).normal(size=(20, 5)).astype(np.float32)
    edges = masked_knn_edges(X, metric="cosine", k=3, min_shared_features=2)
    assert edges
    for u, v, sim, overlap, dist in edges:
        assert overlap >= 2
        assert -1.0 - 1e-4 <= sim <= 1.0 + 1e-4
        assert 0.0 <= dist <= 2.0 + 1e-4
        assert u != v


def test_knn_never_uses_imputation():
    """Crucial V1-honesty invariant: a NaN column must not affect distances.

    We build two matrices that differ only in a column that is NaN for both
    patients. The masked kNN result must be identical.
    """
    rng = np.random.default_rng(42)
    base = rng.normal(size=(30, 6)).astype(np.float32)

    # Add a column of real numbers present for everyone — baseline
    X_full = base.copy()
    # Add a column of NaN for everyone — should be ignored completely
    X_with_nan = np.concatenate([base, np.full((30, 1), np.nan, dtype=np.float32)], axis=1)

    e_full = masked_knn_edges(X_full, metric="cosine", k=5, min_shared_features=3)
    e_nan = masked_knn_edges(X_with_nan, metric="cosine", k=5, min_shared_features=3)

    # Edge sets (u, v) must match; the extra NaN column had no effect.
    assert {(u, v) for u, v, *_ in e_full} == {(u, v) for u, v, *_ in e_nan}
