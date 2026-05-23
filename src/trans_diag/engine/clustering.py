"""k-means assignment + bootstrap-stability, extracted from the sister engine.

Only the two routines the FACE pipeline calls (``run_kmeans``,
``bootstrap_stability``) and their return type (``ClusterAssignment``) are
carried over, verbatim except that the optional reference-label metrics branch
(never used here — ``reference_labels`` is always ``None``) is omitted so we do
not need to vendor ``clustering/metrics.py``. scikit-learn is imported lazily.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ClusterAssignment:
    """A labelled clustering of patients indexed by ``(cohort, patient_id)``."""

    labels: pd.Series           # MultiIndex[cohort, patient_id] → int
    model_name: str             # e.g. "kmeans"
    n_clusters: int
    random_state: int
    config: dict                # model hyperparameters
    metrics: Any | None = None  # reference-label metrics omitted in this extract

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


def run_kmeans(
    embedding: pd.DataFrame,
    *,
    n_clusters: int,
    random_state: int = 0,
    n_init: int = 10,
    reference_labels: pd.Series | np.ndarray | None = None,
    silhouette_sample_size: int | None = 5000,
) -> ClusterAssignment:
    """Fit a deterministic k-means on an embedding and wrap the result."""
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
    if reference_labels is not None:
        raise NotImplementedError(
            "reference-label metrics are omitted in trans_diag.engine.clustering; "
            "the FACE pipeline never passes reference_labels."
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
        metrics=None,
    )


def bootstrap_stability(
    embedding: pd.DataFrame,
    *,
    n_clusters: int,
    n_bootstraps: int = 25,
    subsample_fraction: float = 0.8,
    random_state: int = 0,
) -> dict:
    """Mean pairwise ARI between clusterings fit on bootstrap resamples.

    Each bootstrap samples ``subsample_fraction * N`` patients without
    replacement, runs k-means, and compares the overlapping labels to every
    other bootstrap clustering (Monti et al. 2003 stability criterion).
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

    all_labels: list[tuple[np.ndarray, np.ndarray]] = []  # (indices, labels)
    for b in range(n_bootstraps):
        idx = rng.choice(n, size=sample_size, replace=False)
        idx.sort()
        sub = arr[idx]
        km = KMeans(n_clusters=n_clusters, random_state=b, n_init=10)
        labels = km.fit_predict(sub)
        all_labels.append((idx, labels))

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
