"""M4.7 robustness — does the headline survive the obvious threats?

The headline (M4.3/4.6): the durable ⊥G biology (metabolic / inflammatory) and the archetype map add
prognostic value for *functioning* beyond diagnosis + severity + baseline. Four stress tests:
  IPW           — reweight V2-completers to the full V0 roster (attrition; M3 weights)
  reliability   — restrict to well-measured patients (not a prior-dominated-coordinate artefact)
  leave-one-cohort-out — drop BP (the dominant cohort; tests course-dependence from M4.4)
  permutation null — is the map's incremental ΔR² beyond chance given the foundation?

The IPW / reliability / LOCO refits are done in the script (subset/weight + re-fit the EIV GLM / CV
model). This module provides the permutation-null helper. (The measurement-error-in-baseline / Lord-RTM
concern is additionally addressed by the M4.3 Q2 result — the effect survives the *error-corrected* G
severity.)
"""
from __future__ import annotations

import numpy as np


def _ols_r2(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return 1.0 - np.sum((y - X1 @ beta) ** 2) / np.sum((y - y.mean()) ** 2)


def permutation_null(y, X_found, X_block, *, n_sim: int = 1000, seed: int = 20260610):
    """Is the `X_block` incremental ΔR² over `X_found` beyond chance? Permute the block rows (breaking
    its link to the outcome while preserving the foundation), recompute the incremental ΔR² under each
    permutation, and compare. Returns real ΔR², null 95th pct, and a one-sided p-value."""
    y = np.asarray(y, dtype="float64")
    X_found = np.asarray(X_found, dtype="float64")
    X_block = np.asarray(X_block, dtype="float64")
    real = _ols_r2(y, np.column_stack([X_found, X_block])) - _ols_r2(y, X_found)
    rng = np.random.default_rng(seed)
    n = len(y)
    nulls = np.empty(n_sim)
    for b in range(n_sim):
        perm = rng.permutation(n)
        nulls[b] = _ols_r2(y, np.column_stack([X_found, X_block[perm]])) - _ols_r2(y, X_found)
    return {"real_dR2": float(real), "null_p95": float(np.percentile(nulls, 95)),
            "null_mean": float(nulls.mean()), "p_value": float((nulls >= real).mean() if real == real else np.nan)}
