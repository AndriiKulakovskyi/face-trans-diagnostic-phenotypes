"""Golden test (issue P7-02): the marginalized Woodbury log-likelihood == dense MVN, on observed cells.

``_woodbury_potential`` integrates the latent factors out and evaluates each patient's observed-cell
marginal log-density via the matrix-determinant lemma + Woodbury identity, with per-pattern Cholesky
reuse. This checks it reproduces the brute-force ``multivariate_normal.logpdf`` over each patient's
observed cells (Σ = ΛΦΛᵀ + diag(σ²)), including ragged missingness — the engine's load-bearing kernel.
"""
from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
from scipy.stats import multivariate_normal

from face.models.bayesian.continuous_core import _patterns, _woodbury_potential


def _woodbury_eval(M, Lam, Phi, sigma):
    """Per-patient Woodbury log-lik [N], as the engine computes it inside build_marginalized."""
    N, J = M.shape
    F = Lam.shape[1]
    mask = (~np.isnan(M)).astype("float64")
    x = np.nan_to_num(M, nan=0.0)
    kobs = mask.sum(1)
    log2pi = float(np.log(2.0 * np.pi))
    pat_mask, pat_inv = _patterns(mask)
    R = np.linalg.cholesky(Phi)
    Lt = pt.as_tensor(Lam @ R)
    psi = pt.as_tensor(sigma ** 2)
    ll = _woodbury_potential(pt, pt.as_tensor(x), mask, Lt, psi, pat_mask, pat_inv, kobs, F, log2pi)
    return np.asarray(ll.eval())


def _dense_logpdf(M, Lam, Phi, sigma):
    """Brute-force reference: per-patient observed-cell MVN logpdf under Σ = ΛΦΛᵀ + diag(σ²)."""
    Sigma = Lam @ Phi @ Lam.T + np.diag(sigma ** 2)
    out = np.empty(M.shape[0])
    for i, row in enumerate(M):
        o = np.flatnonzero(~np.isnan(row))
        out[i] = multivariate_normal.logpdf(row[o], mean=np.zeros(o.size), cov=Sigma[np.ix_(o, o)])
    return out


def _toy(seed=0, N=40, J=6, F=2, miss=0.25):
    rng = np.random.default_rng(seed)
    Lam = rng.normal(0.6, 0.3, size=(J, F))
    A = rng.normal(size=(F, F))
    Phi = A @ A.T + np.eye(F)
    d = np.sqrt(np.diag(Phi))
    Phi = Phi / np.outer(d, d)                       # valid correlation matrix (unit diagonal)
    sigma = rng.uniform(0.4, 0.9, size=J)
    Sigma = Lam @ Phi @ Lam.T + np.diag(sigma ** 2)
    X = rng.multivariate_normal(np.zeros(J), Sigma, size=N)
    X[rng.random(X.shape) < miss] = np.nan           # ragged MCAR missingness
    return X, Lam, Phi, sigma


def test_woodbury_matches_dense_mvn_per_patient():
    X, Lam, Phi, sigma = _toy()
    got = _woodbury_eval(X, Lam, Phi, sigma)
    ref = _dense_logpdf(X, Lam, Phi, sigma)
    assert np.allclose(got, ref, atol=1e-6), f"max |Δ| = {np.max(np.abs(got - ref)):.2e}"


def test_woodbury_handles_all_observed_and_single_cell_rows():
    X, Lam, Phi, sigma = _toy(seed=3, miss=0.0)
    X[0, 1:] = np.nan                                # a 1-observed-cell patient
    X[1] = X[1]                                       # a fully-observed patient
    got = _woodbury_eval(X, Lam, Phi, sigma)
    ref = _dense_logpdf(X, Lam, Phi, sigma)
    assert np.allclose(got, ref, atol=1e-6)
