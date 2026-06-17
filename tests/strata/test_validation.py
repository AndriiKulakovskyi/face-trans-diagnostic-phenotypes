"""Guard for the imbalance-robust missingness-artefact metrics (P3-06).

``coverage_artifact`` must (a) expose balanced accuracy / macro-F1 / log-loss / a permutation p-value,
(b) report NO significant skill when membership is independent of coverage, and (c) DETECT skill when
coverage genuinely encodes the label.
"""
from __future__ import annotations

import numpy as np

from face.strata.validation import coverage_artifact


def test_no_skill_gives_high_permutation_pvalue():
    rng = np.random.default_rng(0)
    nobs = rng.integers(0, 4, size=(300, 5))
    labels = rng.integers(0, 3, size=300)                 # independent of coverage
    res = coverage_artifact(nobs, labels, seed=0, n_perm=20)
    assert {"balanced_acc", "balanced_chance", "macro_f1", "log_loss", "perm_p_value"} <= set(res)
    assert 0.0 <= res["perm_p_value"] <= 1.0
    assert res["perm_p_value"] > 0.2                       # no genuine skill -> not significant


def test_real_skill_is_detected():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, size=300)
    nobs = np.zeros((300, 3))
    nobs[np.arange(300), labels] = 5.0                     # a coverage column encodes the label
    nobs += rng.normal(0, 0.1, nobs.shape)
    res = coverage_artifact(nobs, labels, seed=0, n_perm=20)
    assert res["balanced_acc"] > 0.8                       # recovers the planted skill
    assert res["perm_p_value"] < 0.1                       # significant vs permuted nulls
