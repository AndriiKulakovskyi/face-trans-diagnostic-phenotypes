"""M4.7 — permutation-null robustness helper (pure)."""
from __future__ import annotations

import numpy as np

from face.prognosis.robustness import permutation_null


def test_permutation_null_flags_real_signal():
    rng = np.random.default_rng(0)
    n = 400
    found = rng.normal(size=(n, 2))
    block = rng.normal(size=(n, 1))
    y = found @ np.array([0.5, -0.3]) + 0.8 * block[:, 0] + rng.normal(0, 0.5, n)   # block matters
    r = permutation_null(y, found, block, n_sim=400, seed=0)
    assert r["real_dR2"] > r["null_p95"]            # real gain beyond the permutation null
    assert r["p_value"] < 0.05


def test_permutation_null_nulls_noise():
    rng = np.random.default_rng(1)
    n = 400
    found = rng.normal(size=(n, 2))
    block = rng.normal(size=(n, 1))                  # unrelated to y
    y = found @ np.array([0.5, -0.3]) + rng.normal(0, 0.5, n)
    r = permutation_null(y, found, block, n_sim=400, seed=0)
    assert r["p_value"] > 0.05                        # no real incremental signal
