"""Comprehensive clustering validation: internal, external, and information-theoretic.

Three families of metrics, each answering a different question:

- **Internal:** Are the clusters geometrically well-separated in embedding space?
  (silhouette, Calinski-Harabasz, Davies-Bouldin, Dunn index)
- **External:** How do clusters align with the DSM nosological labels?
  (ARI, NMI, AMI, V-measure, Cramer's V, chi-square, homogeneity, completeness)
- **Information-theoretic:** How much information do clusters carry about cohorts,
  and vice versa?  (Shannon entropy, conditional entropy, mutual information,
  variation of information)

All functions return plain dicts of ``{metric_name: float}`` for easy DataFrame
aggregation. Array inputs can be numpy arrays, pandas Series, or lists.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Internal validation ────────────────────────────────────────────────────


def compute_internal_validation(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    silhouette_sample_size: int | None = 5000,
    random_state: int = 0,
) -> dict[str, float]:
    """Compute internal clustering quality metrics.

    Returns silhouette (cosine), Calinski-Harabasz, Davies-Bouldin, and Dunn index.
    """
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    labels = np.asarray(labels)
    valid = labels >= 0
    emb_v = embedding[valid]
    lab_v = labels[valid]
    n = len(lab_v)
    n_clusters = len(np.unique(lab_v))

    if n < 2 or n_clusters < 2:
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
            "dunn_index": float("nan"),
        }

    # Silhouette (cosine, subsampled)
    if silhouette_sample_size and n > silhouette_sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, silhouette_sample_size, replace=False)
        sil = float(silhouette_score(emb_v[idx], lab_v[idx], metric="cosine"))
    else:
        sil = float(silhouette_score(emb_v, lab_v, metric="cosine"))

    ch = float(calinski_harabasz_score(emb_v, lab_v))
    db = float(davies_bouldin_score(emb_v, lab_v))
    dunn = _dunn_index(emb_v, lab_v)

    return {
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "dunn_index": dunn,
    }


def _dunn_index(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int = 500,
) -> float:
    """Approximate Dunn index: min(inter-cluster) / max(intra-cluster).

    Exact Dunn is O(N^2); we subsample each cluster to ``sample_size``
    for tractability on 9K+ patients.
    """
    from scipy.spatial.distance import cdist

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return float("nan")

    rng = np.random.default_rng(0)

    # Per-cluster subsampled points
    clusters: dict[int, np.ndarray] = {}
    for lab in unique_labels:
        mask = labels == lab
        pts = X[mask]
        if len(pts) > sample_size:
            idx = rng.choice(len(pts), sample_size, replace=False)
            pts = pts[idx]
        clusters[int(lab)] = pts

    # Max intra-cluster diameter
    max_intra = 0.0
    for pts in clusters.values():
        if len(pts) < 2:
            continue
        dists = cdist(pts, pts, metric="euclidean")
        max_intra = max(max_intra, float(dists.max()))

    if max_intra == 0:
        return float("nan")

    # Min inter-cluster distance
    min_inter = float("inf")
    labs = list(clusters.keys())
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            dists = cdist(clusters[labs[i]], clusters[labs[j]], metric="euclidean")
            min_inter = min(min_inter, float(dists.min()))

    return float(min_inter / max_intra) if max_intra > 0 else float("nan")


# ─── External validation ────────────────────────────────────────────────────


def compute_external_validation(
    labels: np.ndarray,
    reference: np.ndarray,
    *,
    dsm_subtypes: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute external validation against DSM cohort labels (and optionally subtypes).

    Returns ARI, NMI, AMI, V-measure, Cramer's V, chi-square p-value,
    homogeneity, completeness — against both coarse cohort and fine subtypes.
    """
    from scipy.stats import chi2_contingency
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        completeness_score,
        homogeneity_score,
        normalized_mutual_info_score,
        v_measure_score,
    )

    labels = np.asarray(labels)
    reference = np.asarray(reference)

    # Filter valid
    valid = labels >= 0
    lab_v = labels[valid]
    ref_v = reference[valid]

    if len(lab_v) < 2 or len(np.unique(lab_v)) < 2:
        return {k: float("nan") for k in [
            "ari", "nmi", "ami", "v_measure", "homogeneity", "completeness",
            "cramers_v", "chi2_pvalue",
        ]}

    result: dict[str, float] = {
        "ari": float(adjusted_rand_score(ref_v, lab_v)),
        "nmi": float(normalized_mutual_info_score(ref_v, lab_v)),
        "ami": float(adjusted_mutual_info_score(ref_v, lab_v)),
        "v_measure": float(v_measure_score(ref_v, lab_v)),
        "homogeneity": float(homogeneity_score(ref_v, lab_v)),
        "completeness": float(completeness_score(ref_v, lab_v)),
    }

    # Cramer's V and chi-square
    ct = pd.crosstab(lab_v, ref_v)
    try:
        chi2, p, _, _ = chi2_contingency(ct)
        n = ct.sum().sum()
        k = min(ct.shape) - 1
        result["cramers_v"] = float(np.sqrt(chi2 / (n * k))) if k > 0 else float("nan")
        result["chi2_pvalue"] = float(p)
    except ValueError:
        result["cramers_v"] = float("nan")
        result["chi2_pvalue"] = float("nan")

    # If DSM subtypes available, compute metrics against those too
    if dsm_subtypes is not None:
        dsm_v = np.asarray(dsm_subtypes)[valid]
        result["ari_vs_subtypes"] = float(adjusted_rand_score(dsm_v, lab_v))
        result["nmi_vs_subtypes"] = float(normalized_mutual_info_score(dsm_v, lab_v))
        result["v_measure_vs_subtypes"] = float(v_measure_score(dsm_v, lab_v))

    return result


# ─── Information-theoretic ──────────────────────────────────────────────────


def compute_information_theoretic_validation(
    labels: np.ndarray,
    cohort_labels: np.ndarray,
    *,
    feature_matrix: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Comprehensive information-theoretic analysis.

    Returns Shannon entropy, conditional entropies, mutual information,
    variation of information, and per-feature information gain.
    """
    labels = np.asarray(labels)
    cohort_labels = np.asarray(cohort_labels)

    valid = labels >= 0
    lab_v = labels[valid]
    coh_v = cohort_labels[valid]

    h_c = _entropy(lab_v)
    h_r = _entropy(coh_v)
    h_c_given_r = _conditional_entropy(lab_v, coh_v)
    h_r_given_c = _conditional_entropy(coh_v, lab_v)
    mi = h_c - h_c_given_r
    nmi = 2 * mi / (h_c + h_r) if (h_c + h_r) > 0 else 0.0
    vi = h_c_given_r + h_r_given_c

    # Transdiagnostic score: normalized cohort entropy per cluster
    n_cohorts = len(np.unique(coh_v))
    max_entropy = np.log2(n_cohorts) if n_cohorts > 1 else 1.0
    per_cluster_entropy = _per_cluster_cohort_entropy(lab_v, coh_v)
    mean_cluster_entropy = float(np.mean(per_cluster_entropy))
    transdiagnostic_score = mean_cluster_entropy / max_entropy

    result: dict[str, Any] = {
        "cluster_entropy": h_c,
        "cohort_entropy": h_r,
        "h_cluster_given_cohort": h_c_given_r,
        "h_cohort_given_cluster": h_r_given_c,
        "mutual_information": mi,
        "normalized_mi": nmi,
        "variation_of_information": vi,
        "mean_cluster_cohort_entropy": mean_cluster_entropy,
        "transdiagnostic_score": transdiagnostic_score,
        "per_cluster_entropy": per_cluster_entropy.tolist(),
    }

    # Per-feature information gain (how much each feature's variance is
    # explained by cluster membership)
    if feature_matrix is not None:
        feat_info = _feature_information_gain(lab_v, feature_matrix[valid], feature_names)
        result["feature_information_gain"] = feat_info

    return result


def _entropy(x: np.ndarray) -> float:
    """Shannon entropy in bits."""
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-15)))


def _conditional_entropy(x: np.ndarray, given: np.ndarray) -> float:
    """H(X|Y) = sum_y P(y) * H(X|Y=y)."""
    unique_y, counts_y = np.unique(given, return_counts=True)
    p_y = counts_y / counts_y.sum()
    h = 0.0
    for y, py in zip(unique_y, p_y):
        mask = given == y
        h += py * _entropy(x[mask])
    return float(h)


def _per_cluster_cohort_entropy(
    labels: np.ndarray,
    cohort_labels: np.ndarray,
) -> np.ndarray:
    """Per-cluster Shannon entropy of cohort distribution (bits)."""
    unique_clusters = np.unique(labels)
    entropies = np.zeros(len(unique_clusters))
    for i, c in enumerate(unique_clusters):
        mask = labels == c
        entropies[i] = _entropy(cohort_labels[mask])
    return entropies


def _feature_information_gain(
    labels: np.ndarray,
    features: np.ndarray,
    feature_names: list[str] | None = None,
) -> list[dict[str, float]]:
    """Per-feature information gain from cluster assignment.

    Uses ANOVA F-statistic as a proxy for information gain (higher F = more
    variance explained by clusters).
    """
    from scipy.stats import f_oneway

    n_features = features.shape[1]
    names = feature_names or [f"feat_{i}" for i in range(n_features)]
    results = []

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return results

    for j in range(n_features):
        col = features[:, j]
        valid_mask = np.isfinite(col)
        if valid_mask.sum() < 10:
            results.append({"feature": names[j], "f_statistic": float("nan"), "p_value": float("nan")})
            continue

        groups = [col[valid_mask & (labels == lab)] for lab in unique_labels]
        groups = [g for g in groups if len(g) >= 2]

        if len(groups) < 2:
            results.append({"feature": names[j], "f_statistic": float("nan"), "p_value": float("nan")})
            continue

        try:
            f_stat, p_val = f_oneway(*groups)
            results.append({
                "feature": names[j],
                "f_statistic": float(f_stat) if np.isfinite(f_stat) else float("nan"),
                "p_value": float(p_val) if np.isfinite(p_val) else float("nan"),
            })
        except Exception:
            results.append({"feature": names[j], "f_statistic": float("nan"), "p_value": float("nan")})

    # Sort by F-statistic descending
    results.sort(key=lambda x: x.get("f_statistic", 0) if np.isfinite(x.get("f_statistic", 0)) else 0, reverse=True)
    return results
