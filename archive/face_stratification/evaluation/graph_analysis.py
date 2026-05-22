"""Graph feature propagation analysis via Moran's I.

For each feature, compute the spatial autocorrelation (Moran's I) on the
patient similarity graph. Features with high Moran's I are those where
similar patients (by graph structure) have similar values — they are the
features the graph is "seeing."

This explains why graph methods do or do not outperform feature methods.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def compute_morans_i(
    feature_values: np.ndarray,
    adjacency: sp.csr_matrix,
) -> float:
    """Compute Moran's I spatial autocorrelation for a single feature.

    I = (N / W) * (sum_ij w_ij (x_i - x_bar)(x_j - x_bar)) / (sum_i (x_i - x_bar)^2)

    Range: -1 (perfect dispersion) to +1 (perfect clustering). 0 = random.
    """
    x = np.asarray(feature_values, dtype=float)
    valid = np.isfinite(x)
    if valid.sum() < 10:
        return float("nan")

    # Restrict to valid observations
    x_valid = x[valid]
    n = len(x_valid)
    x_bar = x_valid.mean()
    x_dev = x_valid - x_bar

    ss = np.sum(x_dev ** 2)
    if ss == 0:
        return float("nan")

    # Restrict adjacency to valid nodes
    valid_idx = np.where(valid)[0]
    A_sub = adjacency[valid_idx][:, valid_idx]

    # Sum of weights
    W = A_sub.sum()
    if W == 0:
        return float("nan")

    # Numerator: sum_ij w_ij * (x_i - xbar) * (x_j - xbar)
    # Efficiently: x_dev.T @ A @ x_dev
    numerator = float(x_dev @ A_sub @ x_dev)

    return float((n / W) * (numerator / ss))


def graph_feature_propagation(
    feature_matrix: pd.DataFrame,
    adjacency: sp.csr_matrix,
    *,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Compute Moran's I for each feature on the patient graph.

    Returns DataFrame sorted by Moran's I descending. Features with
    high I are "visible" to graph methods; low I features are invisible.
    """
    results = []
    for col in feature_matrix.columns:
        values = feature_matrix[col].to_numpy()
        mi = compute_morans_i(values, adjacency)
        n_valid = np.isfinite(values).sum()
        results.append({
            "feature": col,
            "morans_i": mi,
            "n_valid": int(n_valid),
            "coverage": float(n_valid / len(values)),
        })

    df = pd.DataFrame(results)
    df = df.sort_values("morans_i", ascending=False).reset_index(drop=True)

    if top_n:
        df = df.head(top_n)

    return df
