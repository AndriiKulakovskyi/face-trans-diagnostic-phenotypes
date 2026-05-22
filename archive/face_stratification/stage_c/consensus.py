"""Consensus clustering via co-association matrix + hierarchical aggregation.

This is the mathematical core of Stage C. Given ``B`` base clusterings of the
same ``N`` patients (e.g. KMeans × 5 seeds + GMM × 5 seeds + Ward + Spectral
× 5 seeds = 16 base clusterings), we:

1. **Align** the arbitrary cluster ids across base clusterings via
   Hungarian matching on the confusion matrix against a reference partition.
   The reference is the base clustering with the highest silhouette, so
   aligned labels in different algorithms refer to roughly "the same" cluster.

2. **Build the co-association matrix**
   $$M_{ij} = \\frac{1}{B}\\sum_{b=1}^{B} \\mathbb{1}[c_b(i) = c_b(j)]$$
   which is a symmetric ``N × N`` float32 matrix in ``[0, 1]``.
   ``M_{ij} = 1`` means patients ``i`` and ``j`` co-cluster in every base
   run; ``M_{ij} = 0`` means they never do.

3. **Compute the consensus partition** by running Ward agglomerative
   clustering on the distance matrix ``1 - M``. This is the CSPA
   (Cluster-based Similarity Partitioning Algorithm) of Strehl & Ghosh
   (2002).

4. **Score per-patient confidence**: for each patient ``i`` in consensus
   cluster ``c``,
   $$\\text{conf}(i) = \\text{mean}_{j \\in c, j \\neq i} M_{ij}
                     - \\max_{c' \\neq c} \\text{mean}_{j \\in c'} M_{ij}$$
   A value near ``+1`` means the patient co-clusters with its cluster
   members almost always and with other clusters almost never (high
   confidence); a value near ``0`` is a boundary patient; negative means
   the consensus has placed them in the "wrong" cluster according to
   the co-association votes.

Memory
------
The co-association matrix is stored as float32. For N=11,014 this is
11,014² × 4 bytes ≈ 485 MB — manageable on any modern machine. If
RAM becomes a concern in the future, this can be computed in chunks
and stored sparsely (only entries above a threshold kept).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Public types ─────────────────────────────────────────────────────────────


@dataclass
class ConsensusResult:
    """Result of a consensus clustering run.

    Attributes
    ----------
    labels:
        Final consensus cluster assignment, indexed by
        ``MultiIndex[cohort, patient_id]`` matching the embedding.
    coassociation_matrix:
        ``(N, N)`` float32 matrix; ``M[i, j]`` is the fraction of base
        clusterings where patients ``i`` and ``j`` are in the same
        cluster. Can be ``None`` if the caller chose to discard it
        to save RAM.
    confidence:
        Per-patient confidence score (see module docstring), same index
        as ``labels``.
    n_base_clusterings:
        Number of base clusterings aggregated.
    aligned_base_labels:
        ``(N, B)`` DataFrame of the aligned cluster labels for every
        base clustering, column-named ``<algorithm>_s<seed>``.
    algorithm_pairwise_ari:
        Pairwise ARI between the aligned base clusterings (``B × B``
        DataFrame). Diagonal is 1.
    """

    labels: pd.Series
    coassociation_matrix: np.ndarray | None
    confidence: pd.Series
    n_base_clusterings: int
    aligned_base_labels: pd.DataFrame
    algorithm_pairwise_ari: pd.DataFrame


# ─── Label alignment ──────────────────────────────────────────────────────────


def align_labels_to_reference(
    labels: np.ndarray,
    reference: np.ndarray,
    *,
    n_clusters: int,
) -> np.ndarray:
    """Hungarian-match ``labels`` to ``reference`` so both share cluster ids.

    Both arrays must have integer labels in ``{0, ..., n_clusters - 1}``.
    Returns a new array where each label in the original ``labels`` has
    been re-mapped so that the total agreement with ``reference`` is
    maximized. The re-mapping is a bijection: every original label becomes
    a unique new label.

    Implementation: build the confusion matrix, negate, feed to
    ``scipy.optimize.linear_sum_assignment``.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError("align_labels_to_reference requires scipy.") from exc

    labels = np.asarray(labels, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.int64)
    assert labels.shape == reference.shape, "labels and reference must be the same length"

    # Confusion matrix: [i, j] = # of patients where labels=i and reference=j
    conf = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for src, dst in zip(labels, reference, strict=True):
        if 0 <= src < n_clusters and 0 <= dst < n_clusters:
            conf[src, dst] += 1

    # Maximize by minimizing the negative
    row_ind, col_ind = linear_sum_assignment(-conf)
    mapping = {int(src): int(dst) for src, dst in zip(row_ind, col_ind, strict=True)}

    aligned = np.array([mapping.get(int(label), int(label)) for label in labels], dtype=np.int64)
    return aligned


# ─── Co-association matrix ────────────────────────────────────────────────────


def build_coassociation_matrix(
    aligned_labels: pd.DataFrame,
    *,
    dtype: np.dtype = np.float32,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Build the ``N × N`` co-association matrix from multiple base clusterings.

    Parameters
    ----------
    aligned_labels:
        ``(N, B)`` DataFrame where each column is a base clustering's
        aligned label vector.
    dtype:
        Output dtype; float32 is enough (entries are fractions of B).
    chunk_size:
        Patients per chunk when computing the matrix. The chunked version
        uses ``O(chunk * N * B)`` temporary memory — the default of 2048
        uses < 100 MB of scratch space for N=11 k, B=16.
    """
    if not isinstance(aligned_labels, pd.DataFrame):
        raise TypeError("aligned_labels must be a DataFrame")

    L = aligned_labels.to_numpy(dtype=np.int32)  # (N, B)
    n, b = L.shape
    if b == 0:
        raise ValueError("Need at least one base clustering")
    if n < 2:
        raise ValueError("Need at least two patients")

    M = np.zeros((n, n), dtype=dtype)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        # Broadcasting: (chunk, 1, B) == (1, N, B) → (chunk, N, B) boolean
        chunk_labels = L[start:end, None, :]  # (chunk, 1, B)
        ref_labels = L[None, :, :]            # (1, N, B)
        matches = (chunk_labels == ref_labels).sum(axis=-1, dtype=np.int32)
        M[start:end] = (matches / float(b)).astype(dtype)

    # Symmetrize just in case
    M = 0.5 * (M + M.T)
    np.fill_diagonal(M, 1.0)
    return M


# ─── Consensus partition ──────────────────────────────────────────────────────


def consensus_partition(
    M: np.ndarray,
    *,
    n_clusters: int,
    linkage: str = "average",
) -> np.ndarray:
    """Hierarchical clustering on the distance ``1 - M``.

    Parameters
    ----------
    M:
        ``(N, N)`` co-association matrix.
    n_clusters:
        Target number of consensus clusters.
    linkage:
        ``"average"`` (default) treats each candidate pair as similar if
        their average co-association exceeds the threshold. ``"complete"``
        is stricter (all members must co-cluster often); ``"single"`` is
        more permissive.
    """
    try:
        from scipy.cluster.hierarchy import fcluster, linkage as scipy_linkage
        from scipy.spatial.distance import squareform
    except ImportError as exc:
        raise ImportError("consensus_partition requires scipy.") from exc

    n = M.shape[0]
    dist = 1.0 - M
    # Symmetry and zero-diagonal enforcement
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)

    condensed = squareform(dist, checks=False)
    Z = scipy_linkage(condensed, method=linkage)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1  # 0-indexed
    return labels.astype(np.int64)


# ─── Per-patient confidence ───────────────────────────────────────────────────


def compute_per_patient_confidence(
    M: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Per-patient confidence from the co-association matrix.

    For each patient ``i`` in cluster ``c``, the confidence is

        conf(i) = mean_{j in c, j != i} M[i, j]
                  - max_{c' != c} mean_{j in c'} M[i, j]

    Positive → i's co-association with its cluster exceeds that with any
    other cluster. Values near 0 are boundary patients. Negative means
    the co-association votes put i in a different cluster than the
    hierarchical consensus did.
    """
    labels = np.asarray(labels, dtype=np.int64)
    n = M.shape[0]
    unique = sorted(np.unique(labels))

    # Pre-compute per-cluster masks for vectorized mean-over-cluster
    cluster_masks = {c: (labels == c) for c in unique}

    confidence = np.zeros(n, dtype=np.float32)
    for i in range(n):
        my_c = int(labels[i])
        my_mask = cluster_masks[my_c].copy()
        my_mask[i] = False
        if my_mask.any():
            intra = float(M[i, my_mask].mean())
        else:
            intra = 0.0

        # max extra-cluster mean
        max_extra = 0.0
        for c, mask in cluster_masks.items():
            if c == my_c:
                continue
            if mask.any():
                extra = float(M[i, mask].mean())
                if extra > max_extra:
                    max_extra = extra
        confidence[i] = intra - max_extra
    return confidence


# ─── Full consensus pipeline ──────────────────────────────────────────────────


def run_consensus_clustering(
    base_labels: dict[str, np.ndarray],
    *,
    n_clusters: int,
    embedding_index: pd.MultiIndex,
    reference_key: str | None = None,
    linkage_method: str = "average",
    keep_matrix: bool = True,
) -> ConsensusResult:
    """End-to-end consensus clustering from a dict of base clusterings.

    Parameters
    ----------
    base_labels:
        Mapping from base clustering name (e.g. ``"kmeans_s0"``) to a
        ``(N,)`` numpy array of integer cluster labels.
    n_clusters:
        Target number of consensus clusters (also used for label
        alignment).
    embedding_index:
        The ``MultiIndex[cohort, patient_id]`` of the embedding, used to
        index the output labels.
    reference_key:
        If given, use this base clustering as the alignment reference.
        If ``None``, use the lexicographically first key.
    linkage_method:
        ``"average"`` / ``"complete"`` / ``"single"`` for the hierarchical
        consensus step.
    keep_matrix:
        If False, discard the (N, N) co-association matrix after computing
        the consensus partition (saves ~500 MB).
    """
    if not base_labels:
        raise ValueError("Need at least one base clustering")

    keys = sorted(base_labels.keys())
    if reference_key is None:
        reference_key = keys[0]
    reference = np.asarray(base_labels[reference_key], dtype=np.int64)

    # Align all base clusterings to the reference
    aligned: dict[str, np.ndarray] = {}
    for key in keys:
        aligned[key] = align_labels_to_reference(
            base_labels[key], reference, n_clusters=n_clusters
        )

    aligned_df = pd.DataFrame(aligned, index=embedding_index)

    logger.info(
        "Building co-association matrix (%d patients × %d base clusterings)...",
        len(embedding_index),
        len(keys),
    )
    M = build_coassociation_matrix(aligned_df)

    logger.info("Running hierarchical consensus (%s linkage)...", linkage_method)
    consensus_labels = consensus_partition(
        M, n_clusters=n_clusters, linkage=linkage_method
    )

    logger.info("Computing per-patient confidence scores...")
    confidence = compute_per_patient_confidence(M, consensus_labels)

    # Pairwise ARI between base clusterings (measures algorithmic agreement)
    from sklearn.metrics import adjusted_rand_score
    pairwise_ari = pd.DataFrame(
        np.zeros((len(keys), len(keys))), index=keys, columns=keys
    )
    for i, a in enumerate(keys):
        pairwise_ari.loc[a, a] = 1.0
        for b in keys[i + 1 :]:
            ari = float(adjusted_rand_score(aligned[a], aligned[b]))
            pairwise_ari.loc[a, b] = ari
            pairwise_ari.loc[b, a] = ari

    labels_series = pd.Series(
        consensus_labels, index=embedding_index, name="cluster", dtype="int64"
    )
    confidence_series = pd.Series(
        confidence, index=embedding_index, name="confidence", dtype="float32"
    )

    return ConsensusResult(
        labels=labels_series,
        coassociation_matrix=M if keep_matrix else None,
        confidence=confidence_series,
        n_base_clusterings=len(keys),
        aligned_base_labels=aligned_df,
        algorithm_pairwise_ari=pairwise_ari,
    )
