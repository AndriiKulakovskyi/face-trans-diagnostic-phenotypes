"""Deep analysis of Stage C consensus clusters.

Companion module to :mod:`face_stratification.stage_c.pipeline` — takes the
Stage C outputs and produces per-cluster diagnostic analytics:

- :func:`analyze_boundary_patients` — identifies the negative-confidence
  patients, computes their "second-best cluster" (where they would have
  gone), and builds the boundary migration matrix.
- :func:`compute_cluster_compactness` — mean and std of distance-to-centroid
  for each cluster in the embedding space.
- :func:`cluster_feature_profile` — per-cluster standardized profile of the
  Stage A features: z-score of each feature's cluster-mean relative to the
  global distribution, plus raw median.
- :func:`cohort_stratified_profile` — for a transdiagnostic cluster, compute
  the per-feature mean separately for each cohort represented in the cluster
  and report the cross-cohort spread. Small spread → genuine transdiagnostic
  phenotype; large spread → forced grouping.
- :func:`sub_cluster` — hierarchical or k-means sub-clustering inside one
  cluster to detect hidden subgroups.
- :func:`rebuild_coassociation_from_base` — reconstruct the 485 MB
  co-association matrix from the saved base clusterings so we can compute
  per-patient, per-cluster co-association for the boundary analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Boundary analysis ───────────────────────────────────────────────────────


@dataclass
class BoundaryAnalysis:
    """Output of :func:`analyze_boundary_patients`."""

    n_total: int
    n_negative_conf: int
    negative_fraction: float
    assigned_to_second_best: pd.DataFrame  # per-patient: (cohort, patient_id, assigned_cluster, second_best, diff)
    migration_matrix: pd.DataFrame  # (assigned_cluster, second_best) → count
    by_cohort: pd.DataFrame  # cohort × assigned_cluster counts


def rebuild_coassociation_from_base(
    base_labels: pd.DataFrame,
    *,
    dtype: np.dtype = np.float32,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Rebuild the N×N co-association matrix from saved aligned base labels.

    Identical math to :func:`build_coassociation_matrix` in
    :mod:`face_stratification.stage_c.consensus`, but kept here as a
    standalone helper so the deep-analysis script can re-load the Stage C
    pipeline state without re-running consensus.
    """
    L = base_labels.to_numpy(dtype=np.int32)
    n, b = L.shape
    M = np.zeros((n, n), dtype=dtype)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        chunk = L[start:end, None, :]
        ref = L[None, :, :]
        matches = (chunk == ref).sum(axis=-1, dtype=np.int32)
        M[start:end] = (matches / float(b)).astype(dtype)
    M = 0.5 * (M + M.T)
    np.fill_diagonal(M, 1.0)
    return M


def analyze_boundary_patients(
    M: np.ndarray,
    cluster_labels: pd.Series,
    confidence: pd.Series,
    metadata: pd.DataFrame,
) -> BoundaryAnalysis:
    """Identify boundary (negative-confidence) patients and their flow direction.

    For every patient with negative confidence, compute:
        - assigned cluster (from the hierarchical consensus)
        - second-best cluster (argmax over c' ≠ c of mean M[i, j in c'])
        - confidence gap (how much better the second-best cluster is)

    Then aggregate into a migration matrix (assigned × second-best).
    """
    if M.shape[0] != len(cluster_labels):
        raise ValueError("M and cluster_labels must have the same length")

    labels = cluster_labels.to_numpy()
    conf = confidence.to_numpy()
    unique_clusters = sorted(np.unique(labels))

    # Pre-compute per-cluster masks
    masks = {c: (labels == c) for c in unique_clusters}

    records = []
    n = len(labels)
    for i in range(n):
        if conf[i] >= 0:
            continue
        my_c = int(labels[i])
        # Compute mean co-association with every cluster
        best_c: int | None = None
        best_score = -np.inf
        for c, mask in masks.items():
            if c == my_c:
                continue
            if not mask.any():
                continue
            score = float(M[i, mask].mean())
            if score > best_score:
                best_score = score
                best_c = c
        my_mask = masks[my_c].copy()
        my_mask[i] = False
        my_score = float(M[i, my_mask].mean()) if my_mask.any() else 0.0
        records.append({
            "cohort": metadata.iloc[i]["cohort"],
            "patient_id": metadata.iloc[i]["patient_id"],
            "assigned_cluster": my_c,
            "second_best_cluster": best_c,
            "my_coassoc": my_score,
            "second_best_coassoc": best_score,
            "gap": best_score - my_score,
            "confidence": conf[i],
        })

    neg_df = pd.DataFrame(records)

    # Migration matrix
    if not neg_df.empty:
        migration = pd.crosstab(
            neg_df["assigned_cluster"], neg_df["second_best_cluster"]
        )
        by_cohort = pd.crosstab(neg_df["cohort"], neg_df["assigned_cluster"])
    else:
        migration = pd.DataFrame()
        by_cohort = pd.DataFrame()

    return BoundaryAnalysis(
        n_total=n,
        n_negative_conf=len(neg_df),
        negative_fraction=len(neg_df) / n if n > 0 else 0.0,
        assigned_to_second_best=neg_df,
        migration_matrix=migration,
        by_cohort=by_cohort,
    )


# ─── Cluster compactness ─────────────────────────────────────────────────────


@dataclass
class ClusterCompactness:
    """Per-cluster distance-to-centroid statistics in the embedding space."""

    cluster_id: int
    n_patients: int
    centroid: np.ndarray
    mean_radius: float
    median_radius: float
    std_radius: float
    max_radius: float
    density: float  # n_patients / volume proxy (1 / mean_radius^d)


def compute_cluster_compactness(
    embedding: pd.DataFrame,
    cluster_labels: pd.Series,
) -> dict[int, ClusterCompactness]:
    """Compactness analysis of each cluster in the embedding space."""
    arr = embedding.to_numpy(dtype=np.float64)
    labels = cluster_labels.to_numpy()
    out: dict[int, ClusterCompactness] = {}
    for c in sorted(np.unique(labels)):
        if c < 0:
            continue
        mask = labels == c
        sub = arr[mask]
        centroid = sub.mean(axis=0)
        dists = np.linalg.norm(sub - centroid[None, :], axis=1)
        d = sub.shape[1]
        density = float(mask.sum()) / (max(dists.mean(), 1e-6) ** min(d, 3))
        out[int(c)] = ClusterCompactness(
            cluster_id=int(c),
            n_patients=int(mask.sum()),
            centroid=centroid,
            mean_radius=float(dists.mean()),
            median_radius=float(np.median(dists)),
            std_radius=float(dists.std()),
            max_radius=float(dists.max()),
            density=density,
        )
    return out


# ─── Cluster feature profile ─────────────────────────────────────────────────


def cluster_feature_profile(
    X: pd.DataFrame,
    cluster_labels: pd.Series,
    *,
    feature_subset: Iterable[str] | None = None,
    min_samples: int = 5,
) -> pd.DataFrame:
    """Per-cluster standardized profile of Stage A features.

    For every (cluster, feature) pair, compute:
    - ``mean_inside`` / ``median_inside`` — raw statistics inside the cluster
    - ``mean_outside`` / ``median_outside`` — raw statistics outside
    - ``z_cluster_mean`` — the cluster's mean in z-units of the global
      distribution (mean 0, std 1 across the whole cohort)
    - ``n_inside`` — how many non-NaN values contributed
    - ``effect`` — same rank-biserial as the enrichment module, for ranking

    The z-score `z_cluster_mean` is the most useful statistic here: it
    says "this cluster's mean on MADRS is +1.3 standard deviations above
    the pooled mean", which is directly interpretable.
    """
    if feature_subset is None:
        feature_subset = list(X.columns)

    labels = cluster_labels.to_numpy()
    global_stats = X.agg(["mean", "std"])

    rows = []
    for cid in sorted(np.unique(labels)):
        if cid < 0:
            continue
        mask = labels == cid
        inside = X.loc[mask]
        outside = X.loc[~mask]
        for feat in feature_subset:
            ins = inside[feat].dropna()
            outs = outside[feat].dropna()
            if len(ins) < min_samples:
                continue
            mu_ins = float(ins.mean())
            med_ins = float(ins.median())
            mu_outs = float(outs.mean()) if len(outs) else np.nan
            med_outs = float(outs.median()) if len(outs) else np.nan
            g_mean = float(global_stats.loc["mean", feat])
            g_std = float(global_stats.loc["std", feat])
            z_mean = (mu_ins - g_mean) / g_std if g_std > 0 else 0.0
            rows.append({
                "cluster": int(cid),
                "feature": feat,
                "n_inside": int(len(ins)),
                "mean_inside": mu_ins,
                "median_inside": med_ins,
                "mean_outside": mu_outs,
                "median_outside": med_outs,
                "z_cluster_mean": float(z_mean),
                "global_mean": g_mean,
                "global_std": g_std,
            })
    return pd.DataFrame(rows)


# ─── Cohort-stratified profile (within-cluster homogeneity) ─────────────────


def cohort_stratified_profile(
    X: pd.DataFrame,
    cluster_labels: pd.Series,
    cohort_labels: pd.Series,
    target_cluster: int,
    *,
    feature_subset: Iterable[str] | None = None,
    min_cohort_size: int = 30,
) -> pd.DataFrame:
    """Per-feature cross-cohort spread *within* one cluster.

    For every feature, compute the mean separately for each cohort present
    in the target cluster. Small cross-cohort variance → the patients from
    different DSM cohorts in this cluster share the same clinical profile
    (genuine transdiagnostic phenotype). Large variance → the cluster is
    a forced grouping of cohort-specific subgroups.

    Returns a DataFrame with one row per feature:
    ``feature, global_mean, global_std, bp_mean, sz_mean, dr_mean, asp_mean,
    cross_cohort_std_z, n_cohorts_present``.
    """
    if feature_subset is None:
        feature_subset = list(X.columns)

    labels = cluster_labels.to_numpy()
    cohorts = cohort_labels.to_numpy()
    mask = labels == target_cluster
    sub_X = X.loc[mask]
    sub_cohorts = cohorts[mask]

    # Global stats for z-scoring
    global_stats = X.agg(["mean", "std"])

    rows = []
    cohort_names = sorted(set(sub_cohorts))
    for feat in feature_subset:
        row: dict[str, Any] = {"feature": feat}
        means: list[float] = []
        row["global_mean"] = float(global_stats.loc["mean", feat])
        row["global_std"] = float(global_stats.loc["std", feat])
        for c in ("bp", "sz", "dr", "asp"):
            c_mask = sub_cohorts == c
            if c_mask.sum() < min_cohort_size:
                row[f"{c}_mean"] = np.nan
                row[f"{c}_n"] = int(c_mask.sum())
                continue
            vals = sub_X.loc[c_mask, feat].dropna()
            if len(vals) < min_cohort_size:
                row[f"{c}_mean"] = np.nan
                row[f"{c}_n"] = int(len(vals))
                continue
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_n"] = int(len(vals))
            means.append(float(vals.mean()))
        if len(means) >= 2 and row["global_std"] > 0:
            # Spread across cohorts, in global-z units
            row["cross_cohort_std_z"] = float(np.std(means) / row["global_std"])
        else:
            row["cross_cohort_std_z"] = np.nan
        row["n_cohorts_present"] = int(sum(1 for c in ("bp", "sz", "dr", "asp")
                                           if not pd.isna(row.get(f"{c}_mean"))))
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Sub-clustering within a cluster ─────────────────────────────────────────


@dataclass
class SubClusteringResult:
    """Output of :func:`sub_cluster`."""

    parent_cluster: int
    n_parent: int
    n_sub_clusters: int
    sub_labels: pd.Series
    sub_sizes: dict[int, int]
    sub_centroid_distances: pd.DataFrame  # pairwise centroid distances


def sub_cluster(
    embedding: pd.DataFrame,
    cluster_labels: pd.Series,
    target_cluster: int,
    *,
    n_sub_clusters: int = 3,
    method: str = "kmeans",
    random_state: int = 0,
) -> SubClusteringResult:
    """Run sub-clustering inside one cluster to detect hidden subgroups."""
    try:
        from sklearn.cluster import AgglomerativeClustering, KMeans
    except ImportError as exc:
        raise ImportError("sub_cluster requires scikit-learn.") from exc

    mask = (cluster_labels == target_cluster).to_numpy()
    sub_arr = embedding.to_numpy(dtype=np.float64)[mask]
    if sub_arr.shape[0] < n_sub_clusters * 2:
        raise ValueError(
            f"Cluster {target_cluster} has too few members ({sub_arr.shape[0]}) "
            f"for {n_sub_clusters} sub-clusters."
        )

    if method == "kmeans":
        model = KMeans(n_clusters=n_sub_clusters, random_state=random_state, n_init=10)
    elif method == "ward":
        model = AgglomerativeClustering(n_clusters=n_sub_clusters, linkage="ward")
    else:
        raise ValueError(f"Unknown sub-clustering method: {method!r}")

    sub_labels_arr = model.fit_predict(sub_arr)
    sub_index = embedding.index[mask]
    sub_labels = pd.Series(sub_labels_arr, index=sub_index, name="sub_cluster", dtype="int64")

    # Per-sub-cluster centroids
    centroids = {}
    for sc in sorted(np.unique(sub_labels_arr)):
        centroids[int(sc)] = sub_arr[sub_labels_arr == sc].mean(axis=0)

    # Pairwise centroid distances
    scs = sorted(centroids.keys())
    dists = pd.DataFrame(np.zeros((len(scs), len(scs))), index=scs, columns=scs)
    for i, a in enumerate(scs):
        for b in scs[i + 1:]:
            d = float(np.linalg.norm(centroids[a] - centroids[b]))
            dists.loc[a, b] = d
            dists.loc[b, a] = d

    sub_sizes = {int(k): int((sub_labels_arr == k).sum()) for k in scs}

    return SubClusteringResult(
        parent_cluster=int(target_cluster),
        n_parent=int(sub_arr.shape[0]),
        n_sub_clusters=n_sub_clusters,
        sub_labels=sub_labels,
        sub_sizes=sub_sizes,
        sub_centroid_distances=dists,
    )
