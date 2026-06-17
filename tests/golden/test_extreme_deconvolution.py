"""Golden test (issue P7-02): Extreme-Deconvolution EM recovers known means AND deconvolves the noise.

``xd_em`` fits a K-component mixture where each patient carries known diagonal measurement noise ``S_i``.
On data generated as (true latent ~ component) + N(0, diag(S)), it should (a) recover the component means,
and (b) recover the NOISE-FREE latent covariance — i.e. the fitted component covariance ≈ the true latent
covariance, strictly below the inflated observed covariance (latent + S). That deconvolution is the whole
point of using XD over a plain GMM on point coordinates.
"""
from __future__ import annotations

import numpy as np

from face.strata.mixture import xd_em


def _match_means(mu_hat, mu_true):
    """Nearest-permutation alignment for K=2 (label switching)."""
    import itertools
    best = None
    for perm in itertools.permutations(range(len(mu_true))):
        err = np.sum((mu_hat[list(perm)] - mu_true) ** 2)
        if best is None or err < best[0]:
            best = (err, list(perm))
    return mu_hat[best[1]]


def test_xd_recovers_means_and_deconvolves_noise():
    rng = np.random.default_rng(0)
    N, D = 600, 2
    mu_true = np.array([[-3.0, -3.0], [3.0, 3.0]])
    latent_cov = 0.5 * np.eye(D)                      # true within-component (noise-free) spread
    noise_var = 1.0                                   # known per-patient measurement variance
    lab = rng.integers(0, 2, size=N)
    latent = np.array([rng.multivariate_normal(mu_true[k], latent_cov) for k in lab])
    S = np.full((N, D), noise_var)                    # diagonal measurement noise, per patient
    X = latent + rng.normal(0.0, np.sqrt(noise_var), size=(N, D))

    res = xd_em(X, S, K=2, seed=0)
    mu_hat = _match_means(res["mu"], mu_true)
    assert np.allclose(mu_hat, mu_true, atol=0.4), f"means off: {mu_hat} vs {mu_true}"

    # Deconvolution: fitted component variance ≈ latent 0.5, well below observed (0.5 + 1.0 = 1.5).
    fitted_var = np.mean([np.mean(np.diag(res["V"][k])) for k in range(2)])
    assert fitted_var < 1.0, f"V not deconvolved: {fitted_var:.3f} (should approach 0.5, not 1.5)"
    assert abs(fitted_var - 0.5) < 0.35, f"deconvolved var {fitted_var:.3f} far from latent 0.5"


def test_xd_bic_is_finite_and_loglik_increases_with_K_separation():
    rng = np.random.default_rng(2)
    N, D = 400, 2
    X = np.vstack([rng.normal(-2, 0.4, (N // 2, D)), rng.normal(2, 0.4, (N // 2, D))])
    S = np.full((N, D), 0.2)
    one = xd_em(X, S, K=1, seed=0)
    two = xd_em(X, S, K=2, seed=0)
    assert np.isfinite(one["bic"]) and np.isfinite(two["bic"])
    assert two["loglik"] > one["loglik"]             # 2 components fit the two-cluster data better
