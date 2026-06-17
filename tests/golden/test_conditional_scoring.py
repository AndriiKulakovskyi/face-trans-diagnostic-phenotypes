"""Golden test (issue P7-02): conditional-Gaussian factor scoring == the analytic posterior.

``conditional_gaussian_scores`` returns each patient's factor-score posterior for the marginalized fit,
observed cells only:  f_i | x_O ~ N( Φ Λ_Oᵀ Σ_OO⁻¹ x_O ,  Φ − Φ Λ_Oᵀ Σ_OO⁻¹ Λ_O Φ ),  Σ = ΛΦΛᵀ+diag(σ²).
This checks the implementation reproduces that closed form (mean + marginal SD) under ragged missingness.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from face.scoring import conditional_gaussian_scores


def _post(Lam, Phi, sigma):
    """Wrap point parameters as a 1-chain/1-draw posterior (the function takes posterior means)."""
    return {
        "Lam": xr.DataArray(Lam[None, None], dims=("chain", "draw", "item", "factor")),
        "Phi": xr.DataArray(Phi[None, None], dims=("chain", "draw", "fi", "fj")),
        "sigma": xr.DataArray(sigma[None, None], dims=("chain", "draw", "item")),
    }


def _ref(M, Lam, Phi, sigma):
    Sigma = Lam @ Phi @ Lam.T + np.diag(sigma ** 2)
    N, F = M.shape[0], Lam.shape[1]
    mean = np.full((N, F), np.nan)
    sd = np.full((N, F), np.nan)
    for i, row in enumerate(M):
        o = np.flatnonzero(~np.isnan(row))
        if o.size == 0:
            continue
        LamO, So = Lam[o], Sigma[np.ix_(o, o)]
        B = Phi @ LamO.T @ np.linalg.inv(So)               # [F, |o|]
        mean[i] = B @ row[o]
        cov = Phi - B @ LamO @ Phi
        sd[i] = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return mean, sd


def _toy(seed=1, N=30, J=6, F=2, miss=0.3):
    rng = np.random.default_rng(seed)
    Lam = rng.normal(0.6, 0.25, size=(J, F))
    A = rng.normal(size=(F, F)); Phi = A @ A.T + np.eye(F)
    d = np.sqrt(np.diag(Phi)); Phi = Phi / np.outer(d, d)
    sigma = rng.uniform(0.4, 0.8, size=J)
    Sigma = Lam @ Phi @ Lam.T + np.diag(sigma ** 2)
    X = rng.multivariate_normal(np.zeros(J), Sigma, size=N)
    X[rng.random(X.shape) < miss] = np.nan
    return X, Lam, Phi, sigma


def test_conditional_mean_and_sd_match_analytic():
    X, Lam, Phi, sigma = _toy()
    out = conditional_gaussian_scores(X, _post(Lam, Phi, sigma), ["g", "s"], psi_floor=0.0)
    m_ref, sd_ref = _ref(X, Lam, Phi, sigma)
    obs = ~np.isnan(m_ref)
    assert np.allclose(out["mean"][obs], m_ref[obs], atol=1e-8)
    assert np.allclose(out["sd"][obs], sd_ref[obs], atol=1e-8)


def test_hdi_is_gaussian_quantile_of_sd():
    X, Lam, Phi, sigma = _toy(seed=4)
    out = conditional_gaussian_scores(X, _post(Lam, Phi, sigma), ["g", "s"], psi_floor=0.0, hdi_prob=0.94)
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - 0.94) / 2)
    obs = ~np.isnan(out["mean"])
    assert np.allclose((out["hdi_high"] - out["mean"])[obs], (z * out["sd"])[obs], atol=1e-8)
