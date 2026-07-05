"""M4.6 — clinical-value metrics (CV-AUC, net benefit, paired ΔAUC)."""
from __future__ import annotations

import numpy as np

from face.prognosis.clinical_value import auc, cv_predict, net_benefit, paired_auc_delta


def _signal(n=400, beta=1.6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-beta * x))).astype(int)
    return x, y, rng


def test_cv_auc_detects_signal_and_nulls_noise():
    x, y, rng = _signal()
    assert auc(y, cv_predict(x[:, None], y, seed=0)) > 0.65        # real predictor discriminates
    noise = rng.normal(size=len(y))
    assert 0.40 < auc(y, cv_predict(noise[:, None], y, seed=0)) < 0.60   # pure noise ~ chance


def test_net_benefit_matches_definition():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    nb = net_benefit(y, p, [0.5])
    assert abs(nb["model"][0] - 0.5) < 1e-9          # 2 TP / 4, 0 FP -> 0.5
    assert abs(nb["treat_all"][0] - 0.0) < 1e-9      # prev .5 at threshold .5 -> 0
    assert nb["treat_none"][0] == 0.0


def test_paired_auc_delta_prefers_the_informative_model():
    x, y, rng = _signal()
    p_good = cv_predict(x[:, None], y, seed=0)
    p_ref = cv_predict(rng.normal(size=len(y))[:, None], y, seed=0)
    d, lo, hi, pgt = paired_auc_delta(y, p_ref, p_good, n_boot=500, seed=0)
    assert d > 0 and pgt > 0.9                       # the informative model wins
