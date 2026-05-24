"""NaN-tolerant pairwise-complete similarity kernels (no imputation).

This module implements the mathematical core of the "no imputation" design:
every pairwise similarity is computed strictly on the subset of features where
both patients have an *observed* value. Missing values contribute nothing to
either the numerator or the denominator of the similarity — they are not
filled in, not replaced by zero, not replaced by the column median.

Supported kernels
-----------------
- :func:`masked_cosine` — cosine of the angle between the two patients
  restricted to their shared-observed subspace.
- :func:`masked_euclidean` — mean squared difference over shared features,
  re-scaled to a distance (so it is comparable across pairs with different
  overlap counts).
- :func:`masked_manhattan` — analogous, using absolute differences.
- :func:`masked_gower` — Gower distance for mixed continuous / ordinal /
  binary features, with an observed-only denominator.

Every kernel also returns the **overlap count** (number of shared observed
features per pair) so the graph builder can enforce the semantic overlap
constraint downstream.

Batching
--------
The N × N pairwise matrices can be large (for the full 11 k cohort, 484 MB at
float32). All functions therefore support a ``query_indices`` argument that
computes similarities for a subset of "query" rows against every "reference"
row, so the caller can stream top-k neighbours without materializing the full
matrix.

Vectorization
-------------
The core trick is to replace NaN with 0 in the value matrix ``X`` and keep a
separate boolean mask ``M``. Then:

    dot[i, j]      = Σ_f  X₀[i,f] * X₀[j,f]               =   X₀ @ X₀.T
    overlap[i, j]  = Σ_f  M[i,f]  * M[j,f]                =   M  @ M.T
    norm_sq[i, j]  = Σ_f  X₀[i,f]² * M[j,f]               = X₀² @ M.T      (asymmetric!)

Cosine then reduces to dot / sqrt(norm_sq[i,j] * norm_sq[j,i]). Euclidean and
Manhattan use analogous mask-aware accumulators. No Python loops inside the
hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

# ─── Public types ─────────────────────────────────────────────────────────────


@dataclass
class MaskedSimilarityResult:
    """Result of a masked pairwise similarity computation.

    Attributes
    ----------
    similarity:
        ``(Q, N)`` matrix where ``Q = len(query_indices)`` (or ``N`` if the
        query set is the whole reference set). Higher values mean more similar.
        Entries with ``overlap < 1`` are set to ``-inf`` so they are naturally
        excluded by downstream top-k selection.
    overlap:
        ``(Q, N)`` integer matrix — number of features both patients observed.
    distance:
        ``(Q, N)`` matrix — a non-negative distance interpretation of the
        similarity, suitable for Gaussian edge weighting. For cosine, this is
        ``1 - similarity``; for Euclidean / Manhattan / Gower it is the raw
        distance. Entries with ``overlap < 1`` are ``+inf``.
    """

    similarity: np.ndarray
    overlap: np.ndarray
    distance: np.ndarray


Metric = Literal["cosine", "euclidean", "manhattan", "gower"]


# ─── Low-level helpers ────────────────────────────────────────────────────────


def _prepare(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (X0, M) where X0 has NaN replaced by 0 and M is the observed mask.

    Both arrays are contiguous float32 for matmul efficiency.
    """
    arr = np.asarray(X, dtype=np.float32)
    mask = np.isfinite(arr).astype(np.float32)
    vals = np.where(mask == 1.0, arr, 0.0).astype(np.float32)
    return vals, mask


def _select_rows(
    X0: np.ndarray, M: np.ndarray, query_indices: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    if query_indices is None:
        return X0, M
    return X0[query_indices], M[query_indices]


def _safe_divide(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Elementwise ``num / denom`` with 0/0 → 0 and x/0 → 0 (never nan/inf)."""
    out = np.zeros_like(num, dtype=np.float32)
    valid = denom > 0
    out[valid] = num[valid] / denom[valid]
    return out


# ─── Public kernels ───────────────────────────────────────────────────────────


def masked_cosine(
    X: np.ndarray,
    *,
    query_indices: np.ndarray | None = None,
) -> MaskedSimilarityResult:
    """Pairwise-complete cosine similarity (no imputation).

    The cosine is computed only on the subset of features observed by both
    patients. Formally:

        cos(i, j) = Σ_{f ∈ shared} x_if x_jf
                    ─────────────────────────────────────────────
                    √(Σ_{f ∈ shared} x_if²) · √(Σ_{f ∈ shared} x_jf²)

    where shared(i, j) = { f : both x_if and x_jf are observed }.

    Pairs with zero shared features are assigned similarity ``-inf`` and
    distance ``+inf`` so they are excluded by any kNN filter.
    """
    X0, M = _prepare(X)
    X0q, Mq = _select_rows(X0, M, query_indices)

    # Dot product — NaN-corrupted features contribute 0 on either side.
    dot = X0q @ X0.T  # (Q, N)

    # Norms restricted to shared features.
    X0_sq = X0 * X0
    X0q_sq = X0q * X0q
    norm_sq_i = X0q_sq @ M.T  # (Q, N)  = Σ_f x_if² * m_jf
    norm_sq_j = Mq @ X0_sq.T  # (Q, N)  = Σ_f m_if * x_jf²

    norm_i = np.sqrt(np.maximum(norm_sq_i, 0.0))
    norm_j = np.sqrt(np.maximum(norm_sq_j, 0.0))

    overlap = (Mq @ M.T).astype(np.int32)

    sim = _safe_divide(dot, norm_i * norm_j)
    sim[overlap < 1] = -np.inf

    distance = 1.0 - sim
    distance[overlap < 1] = np.inf
    # Guard: clamp tiny numerical overshoot (cosine stays in [-1, 1]).
    np.clip(sim, -1.0, 1.0, out=sim, where=np.isfinite(sim))

    return MaskedSimilarityResult(similarity=sim, overlap=overlap, distance=distance)


def masked_euclidean(
    X: np.ndarray,
    *,
    query_indices: np.ndarray | None = None,
) -> MaskedSimilarityResult:
    """Pairwise-complete mean-normalized Euclidean distance (no imputation).

    The distance is
        d(i, j) = √( (1 / |shared|) * Σ_{f ∈ shared} (x_if - x_jf)² )

    i.e. the root-mean-square feature gap over shared features. Dividing by
    ``|shared|`` makes distances comparable across pairs that share different
    numbers of features. Similarity is exposed as ``-distance`` so top-k
    largest similarity still means "most similar".
    """
    X0, M = _prepare(X)
    X0q, Mq = _select_rows(X0, M, query_indices)

    overlap = (Mq @ M.T).astype(np.int32)
    with np.errstate(invalid="ignore"):
        # (x - y)² summed over shared features = Σ (x²m + my² - 2xy)
        x_sq = X0q * X0q
        y_sq = X0 * X0
        s = (x_sq @ M.T) + (Mq @ y_sq.T) - 2.0 * (X0q @ X0.T)
        s = np.maximum(s, 0.0)
        mean_sq = _safe_divide(s, overlap.astype(np.float32))
        distance = np.sqrt(mean_sq)

    distance[overlap < 1] = np.inf
    similarity = -distance.copy()
    similarity[overlap < 1] = -np.inf

    return MaskedSimilarityResult(
        similarity=similarity, overlap=overlap, distance=distance
    )


def masked_manhattan(
    X: np.ndarray,
    *,
    query_indices: np.ndarray | None = None,
) -> MaskedSimilarityResult:
    """Pairwise-complete mean absolute-difference distance (no imputation).

    Used as the core of the Gower kernel for purely numeric blocks. Distance
    is divided by the overlap count so it is scale-stable across pairs.

    Implementation: we can't vectorize |x_i - x_j| directly with matmuls, but
    we can still batch by looping over the query rows in chunks. For the block
    sizes we care about (N ≤ 11 000, d ≤ 16), this is fast enough.
    """
    X0, M = _prepare(X)
    X0q, Mq = _select_rows(X0, M, query_indices)

    q = X0q.shape[0]
    n = X0.shape[0]
    overlap = (Mq @ M.T).astype(np.int32)
    distance = np.full((q, n), np.inf, dtype=np.float32)

    # Chunked loop over query rows to keep memory bounded.
    chunk = 256
    for start in range(0, q, chunk):
        end = min(q, start + chunk)
        xq = X0q[start:end, None, :]  # (c, 1, d)
        mq = Mq[start:end, None, :]   # (c, 1, d)
        xr = X0[None, :, :]           # (1, n, d)
        mr = M[None, :, :]            # (1, n, d)
        joint = mq * mr               # (c, n, d)
        diff = np.abs(xq - xr) * joint  # zero where either is missing
        summed = diff.sum(axis=-1)    # (c, n)
        count = overlap[start:end].astype(np.float32)
        mean = _safe_divide(summed, count)
        distance[start:end] = np.where(overlap[start:end] >= 1, mean, np.inf)

    similarity = -distance.copy()
    similarity[overlap < 1] = -np.inf
    return MaskedSimilarityResult(
        similarity=similarity, overlap=overlap, distance=distance
    )


def masked_gower(
    X: np.ndarray,
    feature_ranges: np.ndarray,
    *,
    query_indices: np.ndarray | None = None,
) -> MaskedSimilarityResult:
    """Pairwise-complete Gower distance (no imputation).

    For each shared feature ``f``, the per-feature contribution is
    ``|x_if - x_jf| / range_f`` (clipped to [0, 1]). The final distance is the
    mean of these contributions over shared features only. This makes it well
    defined for mixed continuous / ordinal / binary data where each column is
    pre-normalized to its observed range.

    Parameters
    ----------
    X:
        ``(N, d)`` float array, possibly containing NaN.
    feature_ranges:
        ``(d,)`` float array giving each feature's empirical range (max − min).
        A value of 0 is replaced with 1 so degenerate constant columns
        contribute 0 / 1 = 0 to the distance.
    """
    ranges = np.asarray(feature_ranges, dtype=np.float32)
    ranges = np.where(ranges > 0, ranges, 1.0)
    X_scaled = np.asarray(X, dtype=np.float32) / ranges[None, :]
    return masked_manhattan(X_scaled, query_indices=query_indices)


# ─── Dispatch ─────────────────────────────────────────────────────────────────


def masked_similarity(
    X: np.ndarray,
    metric: Metric,
    *,
    query_indices: np.ndarray | None = None,
    feature_ranges: np.ndarray | None = None,
) -> MaskedSimilarityResult:
    """Dispatch helper used by the graph builder."""
    if metric == "cosine":
        return masked_cosine(X, query_indices=query_indices)
    if metric == "euclidean":
        return masked_euclidean(X, query_indices=query_indices)
    if metric == "manhattan":
        return masked_manhattan(X, query_indices=query_indices)
    if metric == "gower":
        if feature_ranges is None:
            raise ValueError("masked_gower requires `feature_ranges`.")
        return masked_gower(X, feature_ranges, query_indices=query_indices)
    raise ValueError(f"Unknown metric: {metric!r}")


# ─── kNN on top of masked similarity ──────────────────────────────────────────


def masked_knn_edges(
    X: np.ndarray,
    *,
    metric: Metric,
    k: int,
    min_shared_features: int,
    feature_ranges: np.ndarray | None = None,
    batch_size: int = 512,
) -> list[tuple[int, int, float, int, float]]:
    """Return an undirected edge list from a masked k-nearest-neighbours search.

    An edge ``(src, dst)`` is created only if the pair satisfies the semantic
    overlap constraint ``overlap(src, dst) >= min_shared_features`` AND ``dst``
    is among the ``k`` nearest neighbours of ``src`` (or vice-versa; duplicates
    are deduped).

    Returns
    -------
    list of tuples
        ``(src_idx, dst_idx, similarity, overlap, distance)``. Indices refer to
        rows of ``X``.
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if n < 2:
        return []

    # Deduped, undirected edge set keyed by (min, max).
    edges: dict[tuple[int, int], tuple[float, int, float]] = {}

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        q_idx = np.arange(start, end)
        res = masked_similarity(
            X, metric, query_indices=q_idx, feature_ranges=feature_ranges
        )
        sim = res.similarity
        ovl = res.overlap
        dist = res.distance

        # Mask out self-loops and pairs that fail the overlap constraint.
        for local_i, global_i in enumerate(q_idx):
            sim[local_i, global_i] = -np.inf
        sim[ovl < min_shared_features] = -np.inf

        # For each query row, pick the k largest similarities.
        # If fewer than k candidates pass the constraint, we pick as many as exist.
        k_eff = min(k, n - 1)
        top_idx = np.argpartition(-sim, kth=min(k_eff, sim.shape[1] - 1), axis=1)[:, :k_eff]
        for local_i, global_i in enumerate(q_idx):
            picked = top_idx[local_i]
            for dst in picked:
                dst = int(dst)
                s = float(sim[local_i, dst])
                if not math.isfinite(s) or s == -np.inf:
                    continue
                ov = int(ovl[local_i, dst])
                if ov < min_shared_features:
                    continue
                d = float(dist[local_i, dst])
                key = (min(global_i, dst), max(global_i, dst))
                prev = edges.get(key)
                # Keep the "closest" representation (largest similarity).
                if prev is None or s > prev[0]:
                    edges[key] = (s, ov, d)

    return [
        (u, v, sim, ov, dist_val)
        for (u, v), (sim, ov, dist_val) in edges.items()
    ]
