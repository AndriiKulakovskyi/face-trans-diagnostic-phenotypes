"""Clustering + clustering-quality metrics for Stage B embeddings.

This package wraps scikit-learn's clustering algorithms and provides the
metric functions Stage D will consume, in one stable public API:

- :func:`run_kmeans` — deterministic k-means with a fixed seed and the
  cluster assignments returned as a ``(cohort, patient_id)`` Series.
- :func:`kmeans_sweep` — run k-means for a grid of ``k`` values and return
  a DataFrame of silhouette + ARI + NMI + V-measure.
- :func:`bootstrap_stability` — compute the mean ARI between clusterings
  fit on bootstrap resamples of an embedding. High values mean the
  clustering is reproducible under resampling.
- :func:`compute_cluster_metrics` — ARI, NMI, AMI, V-measure, homogeneity,
  completeness, silhouette, and per-cohort entropy for a given
  clustering.
"""

from face_stratification.clustering.algorithms import (
    ClusterAssignment,
    bootstrap_stability,
    compute_assignment_entropy,
    identify_boundary_patients,
    kmeans_sweep,
    run_gmm_soft,
    run_kmeans,
)
from face_stratification.clustering.metrics import (
    ClusterMetrics,
    compute_cluster_metrics,
)
from face_stratification.clustering.k_selection import (
    KSelectionResult,
    compute_clinical_metrics,
    compute_gap_statistic,
    run_dual_criterion_k_selection,
)

__all__ = [
    "ClusterAssignment",
    "ClusterMetrics",
    "KSelectionResult",
    "bootstrap_stability",
    "compute_assignment_entropy",
    "compute_clinical_metrics",
    "compute_cluster_metrics",
    "compute_gap_statistic",
    "identify_boundary_patients",
    "kmeans_sweep",
    "run_dual_criterion_k_selection",
    "run_gmm_soft",
    "run_kmeans",
]
