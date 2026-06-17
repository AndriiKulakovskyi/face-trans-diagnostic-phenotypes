"""Per-patient dimension scoring (§7) — analytic conditional-Gaussian factor scores.

The marginalized model integrates the latent factors out, so there are no per-patient latent draws.
For the continuous factors we recover the Bayesian factor-score posterior analytically: for patient i
with observed continuous cells O_i,

    f_i | x_{i,O_i}  ~  Normal( Φ Λ_Oᵀ Σ_OO⁻¹ (x_O − μ_O),   Φ − Φ Λ_Oᵀ Σ_OO⁻¹ Λ_O Φ ),
    Σ = Λ Φ Λᵀ + diag(σ²) .

**Uncertainty scope (honest, issue P2-01).** ``conditional_gaussian_scores`` evaluates this at the
posterior-MEAN (Λ, Φ, σ), so it propagates the per-patient *conditional* uncertainty only — NOT the
measurement-parameter (loading / correlation / residual) uncertainty. For well-identified loadings the
parameter uncertainty is small relative to the conditional uncertainty, but the two are not the same and
this scorer does not claim to capture both. The coherent draw-wise joint scorer that assembles all 9
axes from one model state (incl. the explicit block's own G) and exports the full per-patient covariance
lives in ``face.strata.scoring.coherent_joint_coords`` (P2-02 / P2-04). Conditional cov + regression
depend on the row only through its observed pattern, so they are computed once per unique pattern.
"""
from __future__ import annotations

import numpy as np


def conditional_gaussian_scores(M: np.ndarray, post, factor_cols: list[str], *,
                                psi_floor: float = 0.05, hdi_prob: float = 0.94):
    """Per-patient conditional factor-score posterior for a marginalized Gaussian fit (observed cells
    only), evaluated at the posterior-mean loadings. For patient i with observed cols O:

        f_i | x_O ~ Normal( Φ Λ_Oᵀ Σ_OO⁻¹ x_O ,  Φ − Φ Λ_Oᵀ Σ_OO⁻¹ Λ_O Φ ),   μ = 0 (z-scored).

    The conditional cov depends only on the observed PATTERN (coverage), not the values, so one
    F×F regression + cov is computed per unique pattern. SD = sqrt(diag(cov)); HDI from the Gaussian.
    Returns dict of [N, F] arrays: mean, sd, hdi_low, hdi_high. (For well-identified loadings the
    parameter uncertainty is small vs this conditional uncertainty — the standard factor-score posterior.)
    `post` has `Lam` [c,d,J,F], `Phi` [c,d,F,F], `sigma` [c,d,J]."""
    from scipy.stats import norm
    Lam = np.asarray(post["Lam"].mean(("chain", "draw")).values)              # [J, F]
    Phi = np.asarray(post["Phi"].mean(("chain", "draw")).values)             # [F, F]
    sig = psi_floor + np.asarray(post["sigma"].mean(("chain", "draw")).values)  # [J]
    N, J = M.shape
    F = len(factor_cols)
    mask = ~np.isnan(M)
    X = np.nan_to_num(M, nan=0.0)
    pats, inv = np.unique(mask, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    mean = np.full((N, F), np.nan)
    sd = np.full((N, F), np.nan)
    for p in range(pats.shape[0]):
        cols = np.flatnonzero(pats[p])
        if cols.size == 0:
            continue
        rows = np.flatnonzero(inv == p)
        LamO = Lam[cols]                                                      # [k, F]
        So = LamO @ Phi @ LamO.T + np.diag(sig[cols] ** 2)                    # [k, k]
        try:
            Soi = np.linalg.inv(So)
        except np.linalg.LinAlgError:
            Soi = np.linalg.pinv(So)
        B = Phi @ LamO.T @ Soi                                               # [F, k]
        mean[rows] = X[np.ix_(rows, cols)] @ B.T
        cov = Phi - B @ LamO @ Phi
        sd[rows] = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    z = float(norm.ppf(1 - (1 - hdi_prob) / 2))
    return dict(mean=mean, sd=sd, hdi_low=mean - z * sd, hdi_high=mean + z * sd)


def reliability_flags(M: np.ndarray, items: list[str], home: list[str], factor_cols: list[str]):
    """Per-(patient, factor) count of observed HOME indicators + a reliability tier.

    well-characterized (≥3 observed home indicators) · partial (1–2) · prior-dominated (0, the score
    is driven by cross-factor info + prior, not that factor's own indicators)."""
    N = M.shape[0]
    F = len(factor_cols)
    col = {f: i for i, f in enumerate(factor_cols)}
    n_obs = np.zeros((N, F), dtype=int)
    obs = ~np.isnan(M)
    for j, h in enumerate(home):
        if h in col:
            n_obs[:, col[h]] += obs[:, j].astype(int)
    tier = np.where(n_obs >= 3, "well", np.where(n_obs >= 1, "partial", "prior-dominated"))
    return n_obs, tier
