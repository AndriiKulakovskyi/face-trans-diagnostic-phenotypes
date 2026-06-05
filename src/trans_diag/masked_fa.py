"""Imputation-free factor analysis from a pairwise-complete (masked) correlation matrix.

This estimator underlies the whole v2 hierarchical/bifactor pipeline (scripts 01–06): every
factor model — within-construct, second-order, per-visit, split-half — NEVER fills a missing cell.
Mean-filling would reweight each correlation by co-observation (``corr_fill ≈ O · corr_masked``,
``O = n_AB/√(n_A n_B)``) and so partially re-import the cohort-by-missingness confound at the
weakest factor (derivation: docs/legacy_v2/AGGREGATION_RATIONALE.md / docs/legacy_v2/PIPELINE.md §3–4).

This module provides that estimator:
  - ``masked_correlation``  : pairwise-complete (masked) correlation → nearest-PD; no cell filled.
  - ``paf_loadings``        : principal-axis factoring of a correlation matrix.
  - ``varimax``             : Kaiser varimax rotation (orthogonal, simple structure).
  - ``masked_loadings``     : convenience = varimax(paf_loadings(masked_correlation(sc))).
  - ``masked_scores``       : FA posterior-mean factor scores on each patient's OBSERVED
                              support only (regression/Thomson scores), so no imputed value
                              ever enters a score:
                              ``f_i = (I + Lₒ' Ψₒ⁻¹ Lₒ)⁻¹ Lₒ' Ψₒ⁻¹ z_{i,o}``, ``Ψ = 1 − communality``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["nearest_pd", "masked_correlation", "paf_loadings", "varimax",
           "masked_loadings", "masked_scores"]

DEFAULT_MIN_PAIR = 100
DEFAULT_PSI_FLOOR = 0.05


def nearest_pd(A: np.ndarray) -> np.ndarray:
    """Nearest positive-definite correlation matrix (clip eigenvalues, renormalize diagonal)."""
    A = (A + A.T) / 2.0
    w, V = np.linalg.eigh(A)
    P = (V * np.clip(w, 1e-8, None)) @ V.T
    d = np.sqrt(np.clip(np.diag(P), 1e-12, None))
    P = P / np.outer(d, d)
    P = (P + P.T) / 2.0
    np.fill_diagonal(P, 1.0)
    return P


def masked_correlation(sc: pd.DataFrame, min_pair: int = DEFAULT_MIN_PAIR) -> np.ndarray:
    """Pairwise-complete (masked) correlation matrix → nearest-PD. No cell is ever filled.

    Each entry uses only the patients observed on both domains; pairs with fewer than
    ``min_pair`` co-observed patients (or undefined) are set to 0 (treated as uncorrelated —
    a covariance-matrix choice, not the imputation of any data value)."""
    R = sc.corr(min_periods=min_pair).to_numpy(float).copy()  # writable (newer numpy returns read-only)
    R[~np.isfinite(R)] = 0.0
    np.fill_diagonal(R, 1.0)
    return nearest_pd(R)


def paf_loadings(R: np.ndarray, k: int, n_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Principal-axis factoring of a correlation matrix R → ``p × k`` unrotated loadings.

    Iterated communality estimation: put communalities on the diagonal, take the top-k
    eigenpairs, recompute communalities, repeat. Initialized at the squared multiple
    correlations (SMC)."""
    p = R.shape[0]
    try:
        h2 = np.clip(1.0 - 1.0 / np.clip(np.diag(np.linalg.pinv(R)), 1e-6, None), 0.0, 1.0)
    except np.linalg.LinAlgError:
        h2 = np.full(p, 0.5)
    L = np.zeros((p, k))
    prev = h2.copy()
    for _ in range(n_iter):
        Rr = R.copy()
        np.fill_diagonal(Rr, h2)
        w, V = np.linalg.eigh(Rr)
        idx = np.argsort(w)[::-1][:k]
        L = V[:, idx] * np.sqrt(np.clip(w[idx], 0.0, None))
        h2 = np.clip(np.sum(L ** 2, axis=1), 0.0, 1.0)
        if np.max(np.abs(h2 - prev)) < tol:
            break
        prev = h2.copy()
    return L


def varimax(L: np.ndarray, gamma: float = 1.0, n_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
    """Kaiser varimax rotation of a ``p × k`` loading matrix (orthogonal, simple structure)."""
    p, k = L.shape
    Rot = np.eye(k)
    d = 0.0
    for _ in range(n_iter):
        Lam = L @ Rot
        G = L.T @ (Lam ** 3 - (gamma / p) * Lam @ np.diag(np.diag(Lam.T @ Lam)))
        u, s, vt = np.linalg.svd(G)
        Rot = u @ vt
        dn = float(np.sum(s))
        if d != 0.0 and dn / d < 1 + tol:
            break
        d = dn
    return L @ Rot


def masked_loadings(sc: pd.DataFrame, k: int, min_pair: int = DEFAULT_MIN_PAIR) -> np.ndarray:
    """Imputation-free varimax loadings (``p × k``) from the masked correlation matrix."""
    return varimax(paf_loadings(masked_correlation(sc, min_pair), k))


def masked_scores(z, L: np.ndarray, psi_floor: float = DEFAULT_PSI_FLOOR) -> np.ndarray:
    """FA posterior-mean factor scores on each row's OBSERVED support only (no imputation).

    ``z`` is an ``N × p`` standardized matrix (DataFrame or ndarray) with NaN for missing
    entries; ``L`` is the ``p × k`` loading matrix. Uniquenesses ``Ψ = 1 − communality`` are
    derived from ``L`` (floored at ``psi_floor`` to guard Heywood cases). Rows with fewer than
    ``k`` observed entries are left NaN."""
    Zn = np.asarray(z, dtype=float)
    n, p = Zn.shape
    k = L.shape[1]
    psi = np.clip(1.0 - np.sum(L ** 2, axis=1), psi_floor, 1.0)
    WL = L / psi[:, None]                       # Ψ⁻¹ L
    out = np.full((n, k), np.nan)
    eye = np.eye(k)
    for i in range(n):
        o = np.isfinite(Zn[i])
        if int(o.sum()) < k:
            continue
        Lo, Wo = L[o], WL[o]
        out[i] = np.linalg.solve(eye + Lo.T @ Wo, Wo.T @ Zn[i, o])
    return out
