"""Meta-stability scoring across embedding and clustering variants.

Computes per-patient stability of cluster assignments across multiple
embedding methods and clustering algorithms.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MetaStabilityResult:
    """Per-patient meta-stability across methods."""
    patient_ids: np.ndarray
    method_labels: dict[str, np.ndarray] = field(default_factory=dict)
    mode_cluster: np.ndarray | None = None
    agreement_fraction: np.ndarray | None = None
    is_core: np.ndarray | None = None
    is_boundary: np.ndarray | None = None

    @property
    def n_patients(self) -> int:
        return len(self.patient_ids)

    @property
    def n_methods(self) -> int:
        return len(self.method_labels)

    def summary(self) -> dict[str, Any]:
        if self.agreement_fraction is None:
            return {}
        return {
            'n_patients': self.n_patients,
            'n_methods': self.n_methods,
            'n_core': int(self.is_core.sum()) if self.is_core is not None else 0,
            'n_boundary': int(self.is_boundary.sum()) if self.is_boundary is not None else 0,
            'mean_agreement': float(self.agreement_fraction.mean()),
            'median_agreement': float(np.median(self.agreement_fraction)),
        }


def compute_meta_stability(
    patient_ids: np.ndarray,
    method_labels: dict[str, np.ndarray],
    core_threshold: float = 0.8,
    boundary_threshold: float = 0.5,
) -> MetaStabilityResult:
    """Compute meta-stability for each patient across methods.

    For each patient, we need to align cluster labels across methods
    (since cluster IDs are arbitrary). We use pairwise Hungarian alignment
    against the first method as reference.

    Parameters
    ----------
    patient_ids : (n,) patient identifiers
    method_labels : dict mapping method_name -> (n,) integer cluster labels
    core_threshold : agreement fraction above this = core member
    boundary_threshold : agreement fraction below this = boundary

    Returns
    -------
    MetaStabilityResult with per-patient stability metrics
    """
    from scipy.optimize import linear_sum_assignment  # noqa: F401

    result = MetaStabilityResult(
        patient_ids=patient_ids,
        method_labels=method_labels,
    )

    if not method_labels:
        return result

    methods = list(method_labels.keys())
    n = len(patient_ids)

    ref_labels = method_labels[methods[0]]
    aligned: dict[str, np.ndarray] = {methods[0]: ref_labels}

    for name in methods[1:]:
        labels = method_labels[name]
        aligned[name] = _align_labels(ref_labels, labels)

    label_matrix = np.column_stack([aligned[m] for m in methods])  # (n, n_methods)

    mode_cluster = np.zeros(n, dtype=int)
    agreement = np.zeros(n)

    for i in range(n):
        row = label_matrix[i]
        values, counts = np.unique(row, return_counts=True)
        best_idx = np.argmax(counts)
        mode_cluster[i] = values[best_idx]
        agreement[i] = counts[best_idx] / len(methods)

    result.mode_cluster = mode_cluster
    result.agreement_fraction = agreement
    result.is_core = agreement >= core_threshold
    result.is_boundary = agreement < boundary_threshold

    return result


def _align_labels(ref: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Align cluster labels to a reference via Hungarian algorithm."""
    from scipy.optimize import linear_sum_assignment

    ref_unique = np.unique(ref)
    lab_unique = np.unique(labels)

    n_ref = len(ref_unique)
    n_lab = len(lab_unique)
    size = max(n_ref, n_lab)

    cost = np.zeros((size, size))
    for i, r in enumerate(ref_unique):
        for j, l in enumerate(lab_unique):
            cost[i, j] = -np.sum((ref == r) & (labels == l))

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {}
    for i, j in zip(row_ind, col_ind):
        if j < n_lab:
            if i < n_ref:
                mapping[lab_unique[j]] = ref_unique[i]
            else:
                mapping[lab_unique[j]] = lab_unique[j] + 1000


    aligned = np.array([mapping.get(l, l) for l in labels])
    return aligned
