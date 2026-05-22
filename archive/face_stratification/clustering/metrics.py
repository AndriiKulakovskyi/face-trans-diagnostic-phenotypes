"""Clustering quality metrics — one stable API for ARI/NMI/V/silhouette/etc.

All metrics return plain Python floats. Array inputs can be numpy arrays,
pandas Series, or lists. Missing labels (``-1``) are excluded from
supervised metrics so HDBSCAN noise points don't break the scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ClusterMetrics:
    """Container for every scalar metric we report on a clustering."""

    n_clusters: int
    n_noise: int
    n_patients: int
    silhouette: float
    ari_vs_reference: float
    ami_vs_reference: float
    nmi_vs_reference: float
    v_measure: float
    homogeneity: float
    completeness: float
    cohort_entropy_mean: float
    per_cohort_purity: dict[str, float]
    calinski_harabasz: float = float("nan")
    davies_bouldin: float = float("nan")
    cramers_v: float = float("nan")


def _clean_arrays(
    predicted: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows where either label is missing (``-1`` or NaN)."""
    p = np.asarray(predicted)
    r = np.asarray(reference)
    mask = (p >= 0) & (r != "" if r.dtype.kind == "U" else np.isfinite(r.astype(float, errors="ignore")) if r.dtype.kind == "f" else np.ones_like(p, dtype=bool))
    if p.dtype.kind == "f":
        mask &= np.isfinite(p)
    return p[mask], r[mask]


def compute_cluster_metrics(
    embedding: np.ndarray,
    labels: np.ndarray,
    reference: np.ndarray,
    *,
    silhouette_sample_size: int | None = 5000,
    random_state: int = 0,
) -> ClusterMetrics:
    """Compute every scalar metric for a clustering against a reference label.

    Parameters
    ----------
    embedding:
        ``(N, d)`` embedding on which to compute the silhouette score.
    labels:
        ``(N,)`` cluster assignments. Use ``-1`` for noise (HDBSCAN).
    reference:
        ``(N,)`` reference categorical labels — typically the DSM cohort.
    silhouette_sample_size:
        Subsample size for silhouette (the full O(N² d) computation is
        expensive on 11 k patients). ``None`` uses the full dataset.
    """
    try:
        from sklearn.metrics import (
            adjusted_mutual_info_score,
            adjusted_rand_score,
            completeness_score,
            homogeneity_score,
            normalized_mutual_info_score,
            silhouette_score,
            v_measure_score,
        )
    except ImportError as exc:
        raise ImportError(
            "Stage B clustering metrics require scikit-learn. "
            "Install the 'stratification' extra."
        ) from exc

    labels = np.asarray(labels)
    reference = np.asarray(reference)

    # Drop noise for supervised metrics (HDBSCAN emits -1 for outliers)
    valid = labels >= 0
    labels_valid = labels[valid]
    reference_valid = reference[valid]
    emb_valid = embedding[valid]

    n_noise = int((~valid).sum())
    n = int(labels_valid.size)
    n_clusters = int(pd.unique(labels_valid).size) if n > 0 else 0

    if n < 2 or n_clusters < 2:
        return ClusterMetrics(
            n_clusters=n_clusters,
            n_noise=n_noise,
            n_patients=len(labels),
            silhouette=float("nan"),
            ari_vs_reference=float("nan"),
            ami_vs_reference=float("nan"),
            nmi_vs_reference=float("nan"),
            v_measure=float("nan"),
            homogeneity=float("nan"),
            completeness=float("nan"),
            cohort_entropy_mean=float("nan"),
            per_cohort_purity={},
        )

    # Silhouette on a subsample to keep this affordable.
    if silhouette_sample_size is not None and n > silhouette_sample_size:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(n, silhouette_sample_size, replace=False)
        sil = float(
            silhouette_score(
                emb_valid[sample_idx], labels_valid[sample_idx], metric="cosine"
            )
        )
    else:
        sil = float(silhouette_score(emb_valid, labels_valid, metric="cosine"))

    ari = float(adjusted_rand_score(reference_valid, labels_valid))
    ami = float(adjusted_mutual_info_score(reference_valid, labels_valid))
    nmi = float(normalized_mutual_info_score(reference_valid, labels_valid))
    v = float(v_measure_score(reference_valid, labels_valid))
    h = float(homogeneity_score(reference_valid, labels_valid))
    c = float(completeness_score(reference_valid, labels_valid))

    # Per-cluster cohort entropy: H(cohort | cluster)
    # Low → each cluster is cohort-pure (DSM-aligned)
    # High → each cluster mixes cohorts (transdiagnostic)
    ct = pd.crosstab(labels_valid, reference_valid, normalize="index")
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(ct * np.log(ct.replace(0, np.nan))).sum(axis=1)
    cohort_entropy_mean = float(np.nanmean(ent.values))

    # Per-cohort purity: for each cohort, the fraction concentrated in its
    # single most-dominant cluster. Low → the cohort is split; high → the
    # cohort is clustered together (DSM-aligned).
    ct2 = pd.crosstab(reference_valid, labels_valid, normalize="index")
    per_cohort_purity = {
        str(cohort): float(ct2.loc[cohort].max()) for cohort in ct2.index
    }

    # Calinski-Harabasz and Davies-Bouldin
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

    ch = float(calinski_harabasz_score(emb_valid, labels_valid)) if n_clusters >= 2 else float("nan")
    db = float(davies_bouldin_score(emb_valid, labels_valid)) if n_clusters >= 2 else float("nan")

    # Cramér's V (strength of association between clusters and reference)
    cramers_v = _compute_cramers_v(labels_valid, reference_valid)

    return ClusterMetrics(
        n_clusters=n_clusters,
        n_noise=n_noise,
        n_patients=len(labels),
        silhouette=sil,
        ari_vs_reference=ari,
        ami_vs_reference=ami,
        nmi_vs_reference=nmi,
        v_measure=v,
        homogeneity=h,
        completeness=c,
        cohort_entropy_mean=cohort_entropy_mean,
        per_cohort_purity=per_cohort_purity,
        calinski_harabasz=ch,
        davies_bouldin=db,
        cramers_v=cramers_v,
    )


def _compute_cramers_v(
    labels: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Compute Cramér's V between cluster labels and reference labels."""
    ct = pd.crosstab(labels, reference)
    n = ct.sum().sum()
    if n == 0:
        return float("nan")
    from scipy.stats import chi2_contingency
    try:
        chi2, _, _, _ = chi2_contingency(ct)
        k = min(ct.shape) - 1
        if k == 0 or n == 0:
            return float("nan")
        return float(np.sqrt(chi2 / (n * k)))
    except ValueError:
        return float("nan")


def compute_information_theoretic(
    labels: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    """Compute information-theoretic metrics between clusters and reference.

    Returns
    -------
    dict with:
        - cluster_entropy: H(clusters)
        - reference_entropy: H(reference)
        - conditional_entropy_cluster_given_ref: H(cluster|reference)
        - conditional_entropy_ref_given_cluster: H(reference|cluster)
        - mutual_information: I(cluster; reference)
        - normalized_mi: NMI (arithmetic average)
        - variation_of_information: VI = H(cluster|ref) + H(ref|cluster)
    """
    labels = np.asarray(labels)
    reference = np.asarray(reference)

    def _entropy(x: np.ndarray) -> float:
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

    h_c = _entropy(labels)
    h_r = _entropy(reference)
    h_c_given_r = _conditional_entropy(labels, reference)
    h_r_given_c = _conditional_entropy(reference, labels)
    mi = h_c - h_c_given_r
    nmi = 2 * mi / (h_c + h_r) if (h_c + h_r) > 0 else 0.0
    vi = h_c_given_r + h_r_given_c

    return {
        "cluster_entropy": h_c,
        "reference_entropy": h_r,
        "conditional_entropy_cluster_given_ref": h_c_given_r,
        "conditional_entropy_ref_given_cluster": h_r_given_c,
        "mutual_information": mi,
        "normalized_mi": nmi,
        "variation_of_information": vi,
    }
