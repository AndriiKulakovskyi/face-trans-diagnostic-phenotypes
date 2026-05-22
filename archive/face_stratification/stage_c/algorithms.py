"""Clustering algorithm wrappers used by Stage C consensus clustering.

This module exposes four scikit-learn algorithms behind a single signature so
that the consensus and ablation machinery can treat them interchangeably:

- **KMeans** — Lloyd's algorithm. Deterministic given a seed, fast, the
  Stage B baseline.
- **Gaussian Mixture Model (GMM)** — probabilistic, fits an ellipsoidal
  Gaussian per cluster and assigns via maximum a posteriori. Sensitive to
  initialization; we average over seeds.
- **Ward hierarchical** — agglomerative, minimizes within-cluster variance
  at every merge. Deterministic (no random_state).
- **Spectral clustering** — builds a kNN-affinity graph, computes the bottom
  eigenvectors of the normalized Laplacian, and runs k-means in the spectral
  space. Good for non-convex cluster shapes.

Each wrapper returns a :class:`ClusterAssignment` compatible with the
Stage B clustering infrastructure. All algorithms produce integer cluster
labels in ``{0, ..., n_clusters - 1}``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from face_stratification.clustering.algorithms import ClusterAssignment
from face_stratification.clustering.metrics import (
    ClusterMetrics,
    compute_cluster_metrics,
)

logger = logging.getLogger(__name__)


# ─── Internal helper ──────────────────────────────────────────────────────────


def _wrap_assignment(
    labels: np.ndarray,
    *,
    embedding: pd.DataFrame,
    reference_labels: np.ndarray | None,
    model_name: str,
    random_state: int,
    config: dict[str, Any],
    silhouette_sample_size: int | None = 5000,
) -> ClusterAssignment:
    labels = np.asarray(labels, dtype=np.int64)
    series = pd.Series(labels, index=embedding.index, name="cluster", dtype="int64")
    n_clusters = int(np.unique(labels[labels >= 0]).size)

    metrics = None
    if reference_labels is not None:
        metrics = compute_cluster_metrics(
            embedding.to_numpy(dtype=np.float64),
            labels,
            np.asarray(reference_labels),
            silhouette_sample_size=silhouette_sample_size,
        )
    return ClusterAssignment(
        labels=series,
        model_name=model_name,
        n_clusters=n_clusters,
        random_state=random_state,
        config=config,
        metrics=metrics,
    )


# ─── KMeans ───────────────────────────────────────────────────────────────────


def run_kmeans(
    embedding: pd.DataFrame,
    *,
    n_clusters: int,
    random_state: int = 0,
    n_init: int = 10,
    reference_labels: pd.Series | np.ndarray | None = None,
    silhouette_sample_size: int | None = 5000,
) -> ClusterAssignment:
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError("KMeans requires scikit-learn.") from exc

    arr = embedding.to_numpy(dtype=np.float64)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(arr)
    return _wrap_assignment(
        labels,
        embedding=embedding,
        reference_labels=(
            reference_labels if reference_labels is None else np.asarray(reference_labels)
        ),
        model_name="kmeans",
        random_state=random_state,
        config={"n_clusters": n_clusters, "n_init": n_init, "inertia": float(km.inertia_)},
        silhouette_sample_size=silhouette_sample_size,
    )


# ─── GMM ──────────────────────────────────────────────────────────────────────


def run_gmm(
    embedding: pd.DataFrame,
    *,
    n_clusters: int,
    random_state: int = 0,
    covariance_type: str = "full",
    reg_covar: float = 1e-4,
    n_init: int = 3,
    reference_labels: pd.Series | np.ndarray | None = None,
    silhouette_sample_size: int | None = 5000,
) -> ClusterAssignment:
    """Gaussian Mixture Model with full covariance by default.

    ``reg_covar`` adds a small diagonal regularizer to each component's
    covariance; critical when the embedding has some collinear dimensions
    (common for our 56-dim normalized spectral + PCA composite).
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise ImportError("GMM requires scikit-learn.") from exc

    arr = embedding.to_numpy(dtype=np.float64)
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        n_init=n_init,
        random_state=random_state,
        max_iter=200,
    )
    labels = gmm.fit_predict(arr)
    return _wrap_assignment(
        labels,
        embedding=embedding,
        reference_labels=(
            reference_labels if reference_labels is None else np.asarray(reference_labels)
        ),
        model_name="gmm",
        random_state=random_state,
        config={
            "n_clusters": n_clusters,
            "covariance_type": covariance_type,
            "reg_covar": reg_covar,
            "n_init": n_init,
            "log_likelihood": float(gmm.score(arr)),
            "bic": float(gmm.bic(arr)),
            "aic": float(gmm.aic(arr)),
        },
        silhouette_sample_size=silhouette_sample_size,
    )


# ─── Ward hierarchical ────────────────────────────────────────────────────────


def run_ward(
    embedding: pd.DataFrame,
    *,
    n_clusters: int,
    reference_labels: pd.Series | np.ndarray | None = None,
    silhouette_sample_size: int | None = 5000,
) -> ClusterAssignment:
    """Agglomerative clustering with Ward linkage.

    Ward linkage minimizes the total within-cluster variance at every merge,
    which in practice produces balanced, globular clusters. It is the
    classic deterministic baseline to pair with k-means.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:
        raise ImportError("Ward hierarchical requires scikit-learn.") from exc

    arr = embedding.to_numpy(dtype=np.float64)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(arr)
    return _wrap_assignment(
        labels,
        embedding=embedding,
        reference_labels=(
            reference_labels if reference_labels is None else np.asarray(reference_labels)
        ),
        model_name="ward",
        random_state=-1,  # deterministic
        config={"n_clusters": n_clusters, "linkage": "ward"},
        silhouette_sample_size=silhouette_sample_size,
    )


# ─── Spectral clustering ──────────────────────────────────────────────────────


def run_spectral(
    embedding: pd.DataFrame,
    *,
    n_clusters: int,
    random_state: int = 0,
    n_neighbors: int = 20,
    affinity: str = "nearest_neighbors",
    assign_labels: str = "kmeans",
    reference_labels: pd.Series | np.ndarray | None = None,
    silhouette_sample_size: int | None = 5000,
) -> ClusterAssignment:
    """Spectral clustering on a kNN affinity graph.

    Uses the normalized Laplacian's bottom eigenvectors and then runs k-means
    in the spectral space. The kNN graph (default k=20) gives scale-free
    connectivity, which is the standard choice for large N.
    """
    try:
        from sklearn.cluster import SpectralClustering
    except ImportError as exc:
        raise ImportError("Spectral clustering requires scikit-learn.") from exc

    arr = embedding.to_numpy(dtype=np.float64)
    model = SpectralClustering(
        n_clusters=n_clusters,
        random_state=random_state,
        n_neighbors=n_neighbors,
        affinity=affinity,
        assign_labels=assign_labels,
        n_jobs=-1,
    )
    labels = model.fit_predict(arr)
    return _wrap_assignment(
        labels,
        embedding=embedding,
        reference_labels=(
            reference_labels if reference_labels is None else np.asarray(reference_labels)
        ),
        model_name="spectral",
        random_state=random_state,
        config={
            "n_clusters": n_clusters,
            "n_neighbors": n_neighbors,
            "affinity": affinity,
            "assign_labels": assign_labels,
        },
        silhouette_sample_size=silhouette_sample_size,
    )


# ─── Dispatch ─────────────────────────────────────────────────────────────────


ALGORITHMS: dict[str, dict[str, Any]] = {
    "kmeans":   {"fn": run_kmeans,   "stochastic": True,  "default_seeds": (0, 1, 2, 3, 4)},
    "gmm":      {"fn": run_gmm,      "stochastic": True,  "default_seeds": (0, 1, 2, 3, 4)},
    "ward":     {"fn": run_ward,     "stochastic": False, "default_seeds": (0,)},
    "spectral": {"fn": run_spectral, "stochastic": True,  "default_seeds": (0, 1, 2, 3, 4)},
}


def run_algorithm(
    name: str,
    embedding: pd.DataFrame,
    *,
    n_clusters: int,
    random_state: int = 0,
    reference_labels: pd.Series | np.ndarray | None = None,
    silhouette_sample_size: int | None = 5000,
    extra_kwargs: dict[str, Any] | None = None,
) -> ClusterAssignment:
    """Dispatch a single algorithm by name.

    Extra keyword arguments are forwarded to the underlying run_* helper
    (e.g. ``covariance_type="diag"`` for GMM, ``n_neighbors=30`` for
    spectral).
    """
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {name!r}. Available: {sorted(ALGORITHMS)}")
    fn = ALGORITHMS[name]["fn"]
    kwargs: dict[str, Any] = {"n_clusters": n_clusters}
    if ALGORITHMS[name]["stochastic"]:
        kwargs["random_state"] = random_state
    kwargs["reference_labels"] = reference_labels
    kwargs["silhouette_sample_size"] = silhouette_sample_size
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    t0 = time.time()
    assignment = fn(embedding, **kwargs)
    elapsed = time.time() - t0
    if assignment.config is None:
        assignment.config = {}
    assignment.config["runtime_seconds"] = elapsed
    return assignment
