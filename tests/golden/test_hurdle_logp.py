"""Golden test (issue P7-02 / P1-03): the hand-written hurdle-NB log-likelihood == a scipy reference.

``_hurdle_nb_logp`` replaces ``pm.HurdleNegativeBinomial`` (whose NB.logcdf → betainc gradient JAX does
not support) with differentiable ops (gammaln + log1mexp). This checks it equals the closed-form hurdle:
zeros contribute ``log(1-psi)``; positives contribute ``log(psi) + NB.logpmf(y) - log(1-NB.pmf(0))``,
with NB parameterized by mean ``mu`` and concentration ``alpha`` (scipy ``nbinom(n=alpha, p=alpha/(alpha+mu))``).
"""
from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
from scipy.stats import nbinom

from face.measurement.kernel import _hurdle_nb_logp


def _ref_hurdle_logp(y, psi, mu, alpha):
    p = alpha / (alpha + mu)
    nb = nbinom(n=alpha, p=p)
    log_pos = np.log(psi) + nb.logpmf(y) - np.log1p(-nb.pmf(0))     # zero-truncated NB, scaled by psi
    return np.where(y == 0, np.log1p(-psi), log_pos)


def test_hurdle_logp_matches_scipy_reference():
    rng = np.random.default_rng(0)
    y = np.array([0, 0, 1, 2, 0, 5, 0, 3, 10, 0], dtype="float64")
    psi = rng.uniform(0.1, 0.9, size=y.size)
    mu = rng.uniform(0.5, 4.0, size=y.size)
    alpha = 1.7
    got = np.asarray(_hurdle_nb_logp(pt, pt.as_tensor(y), pt.as_tensor(psi),
                                     pt.as_tensor(mu), pt.as_tensor(alpha)).eval())
    ref = _ref_hurdle_logp(y, psi, mu, alpha)
    assert np.allclose(got, ref, atol=1e-9), f"max |Δ| = {np.max(np.abs(got - ref)):.2e}"


def test_hurdle_zero_branch_is_log1m_psi():
    y = np.zeros(5)
    psi = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
    got = np.asarray(_hurdle_nb_logp(pt, pt.as_tensor(y), pt.as_tensor(psi),
                                     pt.as_tensor(np.full(5, 2.0)), pt.as_tensor(1.0)).eval())
    assert np.allclose(got, np.log1p(-psi), atol=1e-12)


def test_hurdle_positive_mass_normalizes():
    # For a fixed (psi, mu, alpha), Σ_{y>0} exp(logp) should equal psi (the non-zero probability mass).
    psi, mu, alpha = 0.6, 2.5, 1.3
    ys = np.arange(1, 2000, dtype="float64")
    lp = np.asarray(_hurdle_nb_logp(pt, pt.as_tensor(ys), pt.as_tensor(np.full(ys.size, psi)),
                                    pt.as_tensor(np.full(ys.size, mu)), pt.as_tensor(alpha)).eval())
    assert abs(np.exp(lp).sum() - psi) < 1e-6
