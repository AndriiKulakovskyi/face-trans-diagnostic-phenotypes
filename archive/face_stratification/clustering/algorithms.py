"""Clustering algorithms + k sweep + bootstrap stability.

Only k-means is in this first pass — Stage D will add GMM / HDBSCAN /
Leiden. k-means is enough to drive the review pass: it is deterministic,
cheap on 11k × 56-dim embeddings, and directly comparable across
normalization variants and bootstrap resamples.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from face_stratification.clustering.metrics import (
    ClusterMetrics,
    compute_cluster_metrics,
)

logger = logging.getLogger(__name__)


# ─── Public types ─────────────────────────────────────────────────────────────


@dataclass
class ClusterAssignment:
    """A labelled clustering of patients indexed by ``(cohort, patient_id)``."""

    labels: pd.Series           # MultiIndex[cohort, patient_id] → int
    model_name: str             # e.g. "kmeans"
    n_clusters: int
    random_state: int
    config: dict                # model hyperparameters
    metrics: ClusterMetrics | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.labels, pd.Series):
            raise TypeError("ClusterAssignment.labels must be a pandas Series")
        if self.labels.index.nlevels != 2 or tuple(self.labels.index.names) != (
            "cohort",
            "patient_id",
        ):
            raise ValueError(
                "ClusterAssignment.labels must be indexed by MultiIndex['cohort', 'patient_id']"
            )

    def counts(self) -> pd.Series:
        return self.labels.value_counts().sort_index()


# ─── k-means ──────────────────────────────────────────────────────────────────


def run_kmeans(
    embedding: pd.DataFrame,
    *,
    n_clusters: int,
    random_state: int = 0,
    n_init: int = 10,
    reference_labels: pd.Series | np.ndarray | None = None,
    silhouette_sample_size: int | None = 5000,
) -> ClusterAssignment:
    """Fit a deterministic k-means on an embedding and wrap the result.

    Parameters
    ----------
    embedding:
        ``(N, d)`` DataFrame indexed by ``MultiIndex[cohort, patient_id]``
        (exactly the output of :class:`PatientEmbedding.values`).
    n_clusters:
        ``k`` for k-means.
    reference_labels:
        Optional reference labels (e.g. DSM cohort) to compute metrics
        against. Aligned by position with ``embedding``.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError("k-means requires scikit-learn.") from exc

    if embedding.index.nlevels != 2 or tuple(embedding.index.names) != (
        "cohort",
        "patient_id",
    ):
        raise ValueError(
            "embedding must be indexed by MultiIndex['cohort', 'patient_id']"
        )

    arr = embedding.to_numpy(dtype=np.float64)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(arr)

    series = pd.Series(labels, index=embedding.index, name="cluster", dtype="int64")
    metrics = None
    if reference_labels is not None:
        ref = np.asarray(reference_labels)
        metrics = compute_cluster_metrics(
            arr, labels, ref, silhouette_sample_size=silhouette_sample_size
        )

    return ClusterAssignment(
        labels=series,
        model_name="kmeans",
        n_clusters=n_clusters,
        random_state=random_state,
        config={
            "n_clusters": n_clusters,
            "n_init": n_init,
            "inertia": float(km.inertia_),
        },
        metrics=metrics,
    )


def kmeans_sweep(
    embedding: pd.DataFrame,
    *,
    k_values: Iterable[int] = (2, 3, 4, 5, 6, 7, 8, 9, 10),
    reference_labels: pd.Series | np.ndarray | None = None,
    random_state: int = 0,
    silhouette_sample_size: int | None = 5000,
) -> pd.DataFrame:
    """Run k-means for a grid of ``k`` and return a tidy metric DataFrame.

    Columns:
        k, silhouette, ari, ami, nmi, v_measure, homogeneity, completeness,
        cohort_entropy_mean, inertia
    """
    rows = []
    for k in k_values:
        assignment = run_kmeans(
            embedding,
            n_clusters=k,
            random_state=random_state,
            reference_labels=reference_labels,
            silhouette_sample_size=silhouette_sample_size,
        )
        m = assignment.metrics
        row = {
            "k": k,
            "inertia": assignment.config["inertia"],
        }
        if m is not None:
            row.update(
                {
                    "silhouette": m.silhouette,
                    "ari": m.ari_vs_reference,
                    "ami": m.ami_vs_reference,
                    "nmi": m.nmi_vs_reference,
                    "v_measure": m.v_measure,
                    "homogeneity": m.homogeneity,
                    "completeness": m.completeness,
                    "cohort_entropy_mean": m.cohort_entropy_mean,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ─── Bootstrap stability ──────────────────────────────────────────────────────


def bootstrap_stability(
    embedding: pd.DataFrame,
    *,
    n_clusters: int,
    n_bootstraps: int = 25,
    subsample_fraction: float = 0.8,
    random_state: int = 0,
) -> dict:
    """Compute mean pairwise ARI between clusterings fit on bootstrap resamples.

    Each bootstrap samples ``subsample_fraction * N`` patients without
    replacement, runs k-means, and compares the overlapping labels to every
    other bootstrap clustering. Returns the mean and standard deviation of
    the ARI across all (N_bootstraps choose 2) pairs.

    High mean ARI means the clustering is **reproducible** — the same
    patients tend to land in the same cluster regardless of which subset
    the algorithm saw. This is the standard bootstrap-stability criterion
    used in consensus clustering (Monti et al. 2003).
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
    except ImportError as exc:
        raise ImportError("bootstrap_stability requires scikit-learn.") from exc

    n = embedding.shape[0]
    sample_size = int(round(subsample_fraction * n))
    rng = np.random.default_rng(random_state)
    arr = embedding.to_numpy(dtype=np.float64)

    # For each bootstrap, store labels at the subsampled positions.
    all_labels: list[tuple[np.ndarray, np.ndarray]] = []  # (indices, labels)
    for b in range(n_bootstraps):
        idx = rng.choice(n, size=sample_size, replace=False)
        idx.sort()
        sub = arr[idx]
        km = KMeans(n_clusters=n_clusters, random_state=b, n_init=10)
        labels = km.fit_predict(sub)
        all_labels.append((idx, labels))

    # Pairwise ARI on the intersection of each pair's sampled indices.
    pairwise_ari: list[float] = []
    for i in range(n_bootstraps):
        for j in range(i + 1, n_bootstraps):
            idx_i, lab_i = all_labels[i]
            idx_j, lab_j = all_labels[j]
            common = np.intersect1d(idx_i, idx_j, assume_unique=True)
            if common.size < n_clusters + 1:
                continue
            pos_i = np.searchsorted(idx_i, common)
            pos_j = np.searchsorted(idx_j, common)
            ari = adjusted_rand_score(lab_i[pos_i], lab_j[pos_j])
            pairwise_ari.append(float(ari))

    if not pairwise_ari:
        return {
            "n_clusters": n_clusters,
            "n_bootstraps": n_bootstraps,
            "mean_ari": float("nan"),
            "std_ari": float("nan"),
            "n_pairs": 0,
        }

    arr_ari = np.asarray(pairwise_ari)
    return {
        "n_clusters": n_clusters,
        "n_bootstraps": n_bootstraps,
        "subsample_fraction": subsample_fraction,
        "mean_ari": float(arr_ari.mean()),
        "std_ari": float(arr_ari.std()),
        "min_ari": float(arr_ari.min()),
        "max_ari": float(arr_ari.max()),
        "n_pairs": int(arr_ari.size),
    }


# ─── Soft clustering ─────────────────────────────────────────────────────────


def run_gmm_soft(
    embeddings: np.ndarray,
    k: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Gaussian Mixture Model and return soft assignments.

    Returns
    -------
    labels : (n,) hard cluster assignments
    posteriors : (n, k) posterior probabilities per cluster
    """
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=k,
        covariance_type='full',
        random_state=random_state,
        n_init=3,
    )
    labels = gmm.fit_predict(embeddings)
    posteriors = gmm.predict_proba(embeddings)
    return labels, posteriors


def compute_assignment_entropy(posteriors: np.ndarray) -> np.ndarray:
    """Compute per-patient Shannon entropy from soft assignment probabilities.

    Parameters
    ----------
    posteriors : (n, k) probability matrix (rows sum to 1)

    Returns
    -------
    entropy : (n,) array of entropy values in bits
    """
    p = np.clip(posteriors, 1e-10, 1.0)
    return -np.sum(p * np.log2(p), axis=1)


def run_kmedoids(
    embedding: pd.DataFrame,
    *,
    n_clusters: int,
    random_state: int = 0,
    n_init: int = 5,
    reference_labels: pd.Series | np.ndarray | None = None,
    silhouette_sample_size: int | None = 5000,
) -> ClusterAssignment:
    """k-medoids (PAM algorithm) — produces actual patient prototypes.

    Medoid-based clustering produces representative actual patients (not
    centroids in embedding space), directly useful for French clinical
    vignette retrieval.
    """
    try:
        from sklearn_extra.cluster import KMedoids
    except ImportError:
        try:
            from sklearn.cluster import KMeans
            logger.warning("sklearn_extra not installed; falling back to k-means")
            return run_kmeans(
                embedding, n_clusters=n_clusters, random_state=random_state,
                reference_labels=reference_labels,
            )
        except ImportError as exc:
            raise ImportError("k-medoids requires sklearn-extra.") from exc

    arr = embedding.to_numpy(dtype=np.float64)
    km = KMedoids(n_clusters=n_clusters, random_state=random_state, init="k-medoids++")
    labels = km.fit_predict(arr)

    series = pd.Series(labels, index=embedding.index, name="cluster", dtype="int64")
    metrics = None
    if reference_labels is not None:
        ref = np.asarray(reference_labels)
        metrics = compute_cluster_metrics(
            arr, labels, ref, silhouette_sample_size=silhouette_sample_size
        )

    return ClusterAssignment(
        labels=series, model_name="kmedoids", n_clusters=n_clusters,
        random_state=random_state,
        config={"n_clusters": n_clusters, "medoid_indices": km.medoid_indices_.tolist()},
        metrics=metrics,
    )


def run_minibatch_kmeans(
    embedding: pd.DataFrame,
    *,
    n_clusters: int,
    random_state: int = 0,
    batch_size: int = 1024,
    reference_labels: pd.Series | np.ndarray | None = None,
    silhouette_sample_size: int | None = 5000,
) -> ClusterAssignment:
    """Mini-batch k-means — fast variant for permutation tests (1000+ runs)."""
    from sklearn.cluster import MiniBatchKMeans

    arr = embedding.to_numpy(dtype=np.float64)
    km = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=random_state,
        batch_size=batch_size, n_init=3,
    )
    labels = km.fit_predict(arr)

    series = pd.Series(labels, index=embedding.index, name="cluster", dtype="int64")
    metrics = None
    if reference_labels is not None:
        ref = np.asarray(reference_labels)
        metrics = compute_cluster_metrics(
            arr, labels, ref, silhouette_sample_size=silhouette_sample_size
        )

    return ClusterAssignment(
        labels=series, model_name="minibatch_kmeans", n_clusters=n_clusters,
        random_state=random_state,
        config={"n_clusters": n_clusters, "batch_size": batch_size,
                "inertia": float(km.inertia_)},
        metrics=metrics,
    )


def run_gmm_variants(
    embedding: pd.DataFrame,
    *,
    n_clusters: int,
    covariance_type: str = "full",
    random_state: int = 0,
    reference_labels: pd.Series | np.ndarray | None = None,
    silhouette_sample_size: int | None = 5000,
) -> ClusterAssignment:
    """GMM with configurable covariance type (full, tied, diag)."""
    from sklearn.mixture import GaussianMixture

    arr = embedding.to_numpy(dtype=np.float64)
    gmm = GaussianMixture(
        n_components=n_clusters, covariance_type=covariance_type,
        random_state=random_state, n_init=3,
    )
    labels = gmm.fit_predict(arr)

    series = pd.Series(labels, index=embedding.index, name="cluster", dtype="int64")
    metrics = None
    if reference_labels is not None:
        ref = np.asarray(reference_labels)
        metrics = compute_cluster_metrics(
            arr, labels, ref, silhouette_sample_size=silhouette_sample_size
        )

    return ClusterAssignment(
        labels=series, model_name=f"gmm_{covariance_type}", n_clusters=n_clusters,
        random_state=random_state,
        config={"n_clusters": n_clusters, "covariance_type": covariance_type,
                "bic": float(gmm.bic(arr)), "aic": float(gmm.aic(arr))},
        metrics=metrics,
    )


def run_bayesian_gmm(
    embedding: pd.DataFrame,
    *,
    max_clusters: int = 20,
    random_state: int = 0,
    reference_labels: pd.Series | np.ndarray | None = None,
    silhouette_sample_size: int | None = 5000,
) -> ClusterAssignment:
    """Bayesian GMM with Dirichlet process prior — automatic k selection."""
    from sklearn.mixture import BayesianGaussianMixture

    arr = embedding.to_numpy(dtype=np.float64)
    bgm = BayesianGaussianMixture(
        n_components=max_clusters,
        weight_concentration_prior_type="dirichlet_process",
        random_state=random_state,
        n_init=3,
        max_iter=300,
    )
    labels = bgm.fit_predict(arr)

    # Effective number of clusters (components with >1% of patients)
    unique, counts = np.unique(labels, return_counts=True)
    effective_k = int((counts / counts.sum() > 0.01).sum())

    series = pd.Series(labels, index=embedding.index, name="cluster", dtype="int64")
    metrics = None
    if reference_labels is not None:
        ref = np.asarray(reference_labels)
        metrics = compute_cluster_metrics(
            arr, labels, ref, silhouette_sample_size=silhouette_sample_size
        )

    return ClusterAssignment(
        labels=series, model_name="bayesian_gmm",
        n_clusters=effective_k, random_state=random_state,
        config={"max_clusters": max_clusters, "effective_k": effective_k,
                "weight_concentration": bgm.weight_concentration_.tolist()},
        metrics=metrics,
    )


def run_hdbscan(
    embedding: pd.DataFrame,
    *,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    reference_labels: pd.Series | np.ndarray | None = None,
    silhouette_sample_size: int | None = 5000,
) -> ClusterAssignment:
    """HDBSCAN — density-based, discovers arbitrary shapes, identifies noise.

    Noise labels (-1) are handled by ClusterMetrics (excluded from
    supervised metrics).
    """
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError as exc:
            raise ImportError("HDBSCAN requires hdbscan or sklearn >= 1.3.") from exc

    arr = embedding.to_numpy(dtype=np.float64)
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = hdb.fit_predict(arr)

    n_clusters = int(len(set(labels) - {-1}))
    n_noise = int((labels == -1).sum())
    logger.info("HDBSCAN: %d clusters, %d noise points (%.1f%%)",
                n_clusters, n_noise, 100 * n_noise / len(labels))

    series = pd.Series(labels, index=embedding.index, name="cluster", dtype="int64")
    metrics = None
    if reference_labels is not None and n_clusters >= 2:
        ref = np.asarray(reference_labels)
        metrics = compute_cluster_metrics(
            arr, labels, ref, silhouette_sample_size=silhouette_sample_size
        )

    return ClusterAssignment(
        labels=series, model_name="hdbscan", n_clusters=n_clusters,
        random_state=0,
        config={"min_cluster_size": min_cluster_size, "min_samples": min_samples,
                "n_noise": n_noise},
        metrics=metrics,
    )


def run_spectral_clustering(
    embedding: pd.DataFrame,
    *,
    n_clusters: int,
    affinity: str = "rbf",
    random_state: int = 0,
    reference_labels: pd.Series | np.ndarray | None = None,
    silhouette_sample_size: int | None = 5000,
) -> ClusterAssignment:
    """Spectral clustering in embedding space with kernel affinity."""
    from sklearn.cluster import SpectralClustering

    arr = embedding.to_numpy(dtype=np.float64)
    sc = SpectralClustering(
        n_clusters=n_clusters, affinity=affinity,
        random_state=random_state, n_init=10,
    )
    labels = sc.fit_predict(arr)

    series = pd.Series(labels, index=embedding.index, name="cluster", dtype="int64")
    metrics = None
    if reference_labels is not None:
        ref = np.asarray(reference_labels)
        metrics = compute_cluster_metrics(
            arr, labels, ref, silhouette_sample_size=silhouette_sample_size
        )

    return ClusterAssignment(
        labels=series, model_name="spectral_clustering", n_clusters=n_clusters,
        random_state=random_state,
        config={"n_clusters": n_clusters, "affinity": affinity},
        metrics=metrics,
    )


def run_hierarchical(
    embedding: pd.DataFrame,
    *,
    n_clusters: int,
    linkage: str = "ward",
    reference_labels: pd.Series | np.ndarray | None = None,
    silhouette_sample_size: int | None = 5000,
) -> ClusterAssignment:
    """Hierarchical (agglomerative) clustering with configurable linkage."""
    from sklearn.cluster import AgglomerativeClustering

    arr = embedding.to_numpy(dtype=np.float64)
    ac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = ac.fit_predict(arr)

    series = pd.Series(labels, index=embedding.index, name="cluster", dtype="int64")
    metrics = None
    if reference_labels is not None:
        ref = np.asarray(reference_labels)
        metrics = compute_cluster_metrics(
            arr, labels, ref, silhouette_sample_size=silhouette_sample_size
        )

    return ClusterAssignment(
        labels=series, model_name=f"hierarchical_{linkage}", n_clusters=n_clusters,
        random_state=0,
        config={"n_clusters": n_clusters, "linkage": linkage},
        metrics=metrics,
    )


def identify_boundary_patients(
    posteriors: np.ndarray,
    entropy_threshold: float = 1.5,
) -> dict[str, Any]:
    """Identify boundary patients from soft clustering.

    Parameters
    ----------
    posteriors : (n, k) probability matrix
    entropy_threshold : entropy above this marks a boundary patient

    Returns
    -------
    dict with:
    - 'entropy': (n,) per-patient entropy
    - 'is_boundary': (n,) boolean mask
    - 'n_boundary': count of boundary patients
    - 'boundary_fraction': fraction of patients that are boundary
    - 'primary_cluster': (n,) most likely cluster
    - 'secondary_cluster': (n,) second most likely cluster
    - 'primary_confidence': (n,) max posterior probability
    """
    entropy = compute_assignment_entropy(posteriors)
    is_boundary = entropy > entropy_threshold

    primary = np.argmax(posteriors, axis=1)
    post_copy = posteriors.copy()
    post_copy[np.arange(len(primary)), primary] = -1
    secondary = np.argmax(post_copy, axis=1)

    return {
        'entropy': entropy,
        'is_boundary': is_boundary,
        'n_boundary': int(is_boundary.sum()),
        'boundary_fraction': float(is_boundary.mean()),
        'primary_cluster': primary,
        'secondary_cluster': secondary,
        'primary_confidence': posteriors[np.arange(len(primary)), primary],
    }
