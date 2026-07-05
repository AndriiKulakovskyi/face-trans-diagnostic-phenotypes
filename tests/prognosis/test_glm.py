"""M4.2 — the outcome GLM: synthetic recovery, incl. the EIV de-attenuation (small fits)."""
from __future__ import annotations

import numpy as np
import pytest

from face.prognosis.glm import fit_glm

# fit_glm runs the NumPyro/JAX NUTS backend (the `bayesian` extra). Light CI installs only `.[dev]`,
# so skip these sampling tests when the backend is absent — matching how the engine tests guard pymc.
pytest.importorskip("numpyro", reason="needs the NumPyro/JAX backend (pip install -e '.[bayesian]')")

FIT = dict(draws=300, tune=300, chains=2, seed=0)


def _beta(coef, i):
    row = coef.loc[coef.term == f"beta[{i}]"]
    return float(row["mean"].iloc[0])


def test_gaussian_recovers_signed_slopes():
    rng = np.random.default_rng(1)
    n = 250
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 0.8 * x1 - 0.6 * x2 + rng.normal(0, 0.5, n)
    y = (y - y.mean()) / y.std()
    r = fit_glm(y, np.column_stack([x1, x2]), family="gaussian", **FIT)
    assert r["rhat"] < 1.1 and r["divergences"] == 0
    assert _beta(r["coef"], 0) > 0.2          # positive slope recovered
    assert _beta(r["coef"], 1) < -0.2         # negative slope recovered


def test_eiv_corrects_attenuation():
    # true predictor xi, observed with large known error (sd=1.0) -> naive slope is attenuated to ~0.5;
    # the EIV model, given the known sd, should de-attenuate toward the true slope (~1.0).
    rng = np.random.default_rng(2)
    n = 400
    xi = rng.normal(size=n)
    sd = np.full(n, 1.0)
    z = xi + rng.normal(0, 1.0, n)
    y = 1.0 * xi + rng.normal(0, 0.5, n)
    y = (y - y.mean()) / y.std()
    naive = float(np.cov(y, z)[0, 1] / np.var(z))            # attenuated OLS slope of y on z
    r = fit_glm(y, np.empty((n, 0)), family="gaussian",
                eiv_obs=z[:, None], eiv_sd=sd[:, None], draws=500, tune=500, chains=2, seed=0)
    b_eiv = float(r["coef"].loc[r["coef"].term == "beta_eiv[0]", "mean"].iloc[0])
    assert r["rhat"] < 1.15
    assert b_eiv > naive + 0.1                                # de-attenuated above the naive slope
    assert 0.5 < b_eiv < 1.6                                  # in the neighbourhood of the true 1.0


def test_bernoulli_recovers_sign():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(1.3 * x)))
    y = rng.binomial(1, p)
    r = fit_glm(y, x[:, None], family="bernoulli", **FIT)
    assert r["rhat"] < 1.1
    assert _beta(r["coef"], 0) > 0.3
