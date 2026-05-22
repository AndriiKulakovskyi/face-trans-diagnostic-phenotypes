"""Cohort-fair clustering evaluation.

Ensures no cohort is systematically disadvantaged by the clustering.
DR (N=350) is particularly vulnerable — a 90% DR cluster merely
recovers the DSM label, which is not transdiagnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cohort_fairness_metrics(
    labels: np.ndarray,
    cohort_labels: np.ndarray,
) -> dict[str, float]:
    """Compute cohort-fairness metrics for the clustering.

    Returns
    -------
    dict with:
        - entropy_ratio: H(cohort|cluster) / H(cohort). Close to 1 = transdiagnostic.
        - max_cluster_imbalance: for the most imbalanced cluster, how much the
          dominant cohort exceeds expected proportions.
        - per_cohort_silhouette_gap: (reserved for future extension)
    """
    labels = np.asarray(labels)
    cohort_labels = np.asarray(cohort_labels)
    valid = labels >= 0
    lab_v = labels[valid]
    coh_v = cohort_labels[valid]

    # H(cohort)
    _, counts = np.unique(coh_v, return_counts=True)
    p = counts / counts.sum()
    h_cohort = float(-np.sum(p * np.log2(p + 1e-15)))

    # H(cohort|cluster)
    unique_clusters = np.unique(lab_v)
    h_coh_given_cl = 0.0
    cluster_sizes = []
    for c in unique_clusters:
        mask = lab_v == c
        n_c = mask.sum()
        cluster_sizes.append(n_c)
        _, c_counts = np.unique(coh_v[mask], return_counts=True)
        c_p = c_counts / c_counts.sum()
        h_c = float(-np.sum(c_p * np.log2(c_p + 1e-15)))
        h_coh_given_cl += (n_c / len(lab_v)) * h_c

    entropy_ratio = h_coh_given_cl / h_cohort if h_cohort > 0 else 0.0

    # Max cluster imbalance
    expected_proportions = dict(zip(*np.unique(coh_v, return_counts=True)))
    total = len(coh_v)
    expected_props = {k: v / total for k, v in expected_proportions.items()}

    max_imbalance = 0.0
    for c in unique_clusters:
        mask = lab_v == c
        cluster_coh = coh_v[mask]
        _, c_counts = np.unique(cluster_coh, return_counts=True)
        c_n = mask.sum()
        if c_n == 0:
            continue
        for cohort, count in zip(*np.unique(cluster_coh, return_counts=True)):
            observed_prop = count / c_n
            expected_prop = expected_props.get(cohort, 0)
            if expected_prop > 0:
                ratio = observed_prop / expected_prop
                max_imbalance = max(max_imbalance, ratio)

    return {
        "entropy_ratio": float(entropy_ratio),
        "max_cluster_imbalance": float(max_imbalance),
        "h_cohort": float(h_cohort),
        "h_cohort_given_cluster": float(h_coh_given_cl),
        "n_clusters": len(unique_clusters),
    }
