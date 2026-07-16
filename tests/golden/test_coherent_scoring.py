"""Golden test (issue P2-02): the coherent ``f_m | f_e`` conditional is correct and cross-block-coupled.

``conditional_fm_given_fe`` scores the marginalized specifics conditioned on the SAME explicit-latent
draw under the shared Φ (the fix for the old incoherent dimension-wise assembly). Two checks, no MCMC
(it takes f_e draws as input): (a) with no observed continuous cells it returns the prior conditional
mean ``M·f_e`` (= cross-block coupling via Φ); (b) with observed data it recovers a planted ``f_m`` and
shrinks the posterior variance below the prior.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from face.strata.scoring import conditional_fm_given_fe


def _setup(seed=0, F=4, Jc=8, N=80):
    rng = np.random.default_rng(seed)
    e_cols, m_cols = [0, 1], [2, 3]
    Km, Ke = len(m_cols), len(e_cols)
    A = rng.normal(size=(F, F))
    Phi = A @ A.T + np.eye(F)
    d = np.sqrt(np.diag(Phi))
    Phi = Phi / np.outer(d, d)                       # valid correlation (cross-block coupling ≠ 0)
    Lam = np.zeros((Jc, F))
    for j in range(Jc):                              # each continuous item loads on one m-factor only
        Lam[j, m_cols[j % Km]] = 0.7
    sigma = np.full(Jc, 0.5)
    P = {"Phi": Phi, "Lam": Lam, "sigma": sigma}
    Mmat = Phi[np.ix_(m_cols, e_cols)] @ np.linalg.inv(Phi[np.ix_(e_cols, e_cols)])
    return rng, e_cols, m_cols, Km, Ke, Jc, N, P, Mmat, Lam, sigma


def _mp(M, e_cols, m_cols, covariates=None):
    return SimpleNamespace(
        base=SimpleNamespace(
            M=M,
            factor_cols=["g", "e1", "m1", "m2"],
            covariates=covariates,
        ),
        e_cols=e_cols,
        m_cols=m_cols,
    )


def test_no_data_returns_prior_conditional_mean():
    rng, e_cols, m_cols, Km, Ke, Jc, N, P, Mmat, _, _ = _setup()
    fe0 = rng.normal(size=(N, Ke))
    fe = fe0[None].repeat(600, 0)                    # fix f_e across draws so δ averages out cleanly
    M = np.full((N, Jc), np.nan)                     # nothing observed -> f_m | f_e ~ N(M·f_e, S_res)
    fm = conditional_fm_given_fe(_mp(M, e_cols, m_cols), P, fe, seed=1)
    assert np.allclose(fm.mean(0), fe0 @ Mmat.T, atol=0.12), "prior conditional mean (M·f_e) off"
    Sres = P["Phi"][np.ix_(m_cols, m_cols)] - Mmat @ P["Phi"][np.ix_(m_cols, e_cols)].T
    assert np.allclose(fm.std(0).mean(0), np.sqrt(np.diag(Sres)), atol=0.15), "prior conditional SD off"


def test_observed_data_recovers_fm_and_shrinks_variance():
    rng, e_cols, m_cols, Km, Ke, Jc, N, P, Mmat, Lam, sigma = _setup(seed=3)
    fe = rng.normal(size=(1, N, Ke)).repeat(300, 0)  # fix f_e across draws to isolate the data conditional
    fe0 = fe[0]
    delta_true = rng.normal(size=(N, Km)) @ np.linalg.cholesky(
        P["Phi"][np.ix_(m_cols, m_cols)] - Mmat @ P["Phi"][np.ix_(m_cols, e_cols)].T + 1e-9 * np.eye(Km)).T
    fm_true = fe0 @ Mmat.T + delta_true
    X = fm_true @ Lam[:, m_cols].T + rng.normal(0, sigma, size=(N, Jc))   # observed continuous data

    fm_obs = conditional_fm_given_fe(_mp(X, e_cols, m_cols), P, fe, seed=2)
    fm_prior = conditional_fm_given_fe(_mp(np.full((N, Jc), np.nan), e_cols, m_cols), P, fe, seed=2)

    # recovery: posterior mean correlates with the planted f_m
    r = np.corrcoef(fm_obs.mean(0).ravel(), fm_true.ravel())[0, 1]
    assert r > 0.8, f"f_m recovery corr = {r:.3f}"
    # shrinkage: observing data reduces posterior SD below the prior conditional SD
    assert fm_obs.std(0).mean() < fm_prior.std(0).mean(), "observing data did not shrink variance"


def test_conditional_fm_removes_frozen_covariate_mean():
    rng, e_cols, m_cols, _km, ke, jc, n, P, mmat, lam, sigma = _setup(seed=8)
    fe = rng.normal(size=(30, n, ke))
    fm = fe[0] @ mmat.T + rng.normal(size=(n, len(m_cols))) * 0.2
    x = fm @ lam[:, m_cols].T + rng.normal(0, sigma, size=(n, jc))
    covariates = rng.normal(size=(n, 2))
    alpha = rng.normal(0, 0.3, size=jc)
    beta = rng.normal(0, 0.2, size=(jc, 2))
    shifted = x + alpha[None, :] + covariates @ beta.T
    adjusted = dict(P, alpha=alpha, beta=beta)
    got = conditional_fm_given_fe(
        _mp(shifted, e_cols, m_cols, covariates), adjusted, fe, seed=21
    )
    ref = conditional_fm_given_fe(_mp(x, e_cols, m_cols), P, fe, seed=21)
    assert np.allclose(got, ref, atol=1e-6)


def test_matched_parameter_draws_equal_fixed_path_when_states_are_identical():
    rng, e_cols, m_cols, _km, ke, jc, n, P, _mmat, _lam, _sigma = _setup(
        seed=13
    )
    fe = rng.normal(size=(5, n, ke))
    x = rng.normal(size=(n, jc))
    parameter_draws = {
        name: np.repeat(np.asarray(P[name])[None, ...], len(fe), axis=0)
        for name in ("Lam", "Phi", "sigma")
    }
    fixed = conditional_fm_given_fe(
        _mp(x, e_cols, m_cols), P, fe, seed=44
    )
    matched = conditional_fm_given_fe(
        _mp(x, e_cols, m_cols),
        P,
        fe,
        parameter_draws=parameter_draws,
        seed=44,
    )
    assert np.allclose(matched, fixed, atol=1e-6)
