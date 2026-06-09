"""Estimator/prior-robustness confirmation for the measurement map (§5, reframed).

Replaces the standalone FIML confirmation. Per the methods doc §3.5 the marginalized Bayesian model
and FIML optimize the *same* observed-data likelihood, so a separate SEM estimator adds little new
evidence; instead we answer §5's actual questions in the existing engine:

  (A) prior-free refit       — flat loading priors → Λ, Φ ≈ the soft-prior fit ⇒ not a prior artefact
                               (a flat-prior MAP = MLE = FIML, the §3.5 marginal).
  (B) posterior-predictive   — model-implied vs observed pairwise correlations → Bayesian SRMR +
                               residual-correlation matrix (absolute fit, no χ² asymptotics).
  (C) WAIC model comparison  — bifactor vs unidimensional vs correlated-factors (incremental fit).

This module is pure-NumPy post-hoc analysis over a fitted posterior (Λ, Φ, σ draws) — it never
re-reads raw data or imputes. The per-patient marginal log-lik mirrors the engine's Woodbury kernel
(`continuous_core._woodbury_potential`) but in NumPy, so PPC/WAIC reuse one likelihood.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

REPO = Path(__file__).resolve().parents[2]
LOG2PI = float(np.log(2.0 * np.pi))


# ----------------------------- (C) structural comparison variants -----------------------------
# Built from one bifactor base prep (prepare(S1_FACTORS, correlated=True, windows=False)) so all
# three models share the SAME items, M, and subsample — a clean WAIC comparison on identical data.

def unidim_prep(base):
    """Unidimensional null: ONE general factor, every item loads on it (no specific factors).
    Tests whether multidimensionality is needed at all."""
    g = base.factor_cols[base.g_col]
    pos = [(j, 0, 0.4, 0.5) for j in range(len(base.items))]
    kind = {(j, 0): "primary" for j in range(len(base.items))}
    return replace(base, factor_cols=[g], spec_factors=[], g_col=0,
                   pos_cells=pos, sgn_cells=[], correlated=False, g_correlated=False, kind=kind)


def corr_no_g_prep(base):
    """Correlated-factors alternative: simple structure (each item on its home factor only) — the
    specific-domain items do NOT load on the general factor. Drops the bifactor-G cross-loadings;
    the 4 specifics stay correlated (Φ, LKJ) and severity stays its own factor orthogonal to them
    (matching the project's G⊥specifics identification, and using the working n=4 LKJ — the engine's
    LKJ-over-all-factors `g_correlated` path breaks PyTensor jitter-init for n≥5). This is the direct
    'is the general factor's loading on the specific-domain items needed?' test vs the bifactor."""
    return replace(base, sgn_cells=[],
                   kind={k: v for k, v in base.kind.items() if v != "bifactor_G"})


def load_posterior(idata_path: str | Path):
    """Return the posterior xarray group from a stored idata (.nc), arviz-version agnostic."""
    import arviz as az
    idata = az.from_netcdf(str(idata_path))
    return idata.posterior


def _draws(post, name: str) -> np.ndarray:
    """Stack [chain, draw, ...] → [S, ...] for a posterior variable."""
    a = np.asarray(post[name].values)
    return a.reshape((-1,) + a.shape[2:])


def implied_cov(Lam: np.ndarray, Phi: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Σ = Λ Φ Λᵀ + diag(σ²)."""
    return Lam @ Phi @ Lam.T + np.diag(sigma ** 2)


def cov_to_corr(S: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    return S / np.outer(d, d)


def pairwise_corr(M: np.ndarray) -> np.ndarray:
    """Observed pairwise-complete correlation matrix (NaN = missing, never imputed)."""
    return pd.DataFrame(M).corr().to_numpy()


def marginal_loglik(M: np.ndarray, Lam: np.ndarray, Phi: np.ndarray, sigma: np.ndarray,
                    mu: np.ndarray | None = None) -> np.ndarray:
    """Per-patient observed-data Gaussian log-lik under Σ = ΛΦΛᵀ+diag(σ²); pattern-grouped.

    This is exactly the FIML per-patient term (§3.5): log N(x_{i,Oi} | μ_Oi, Σ_{Oi,Oi}) over each
    patient's *observed* cells; missing cells contribute no term (no imputation). Rows with zero
    observed cells get ll = 0."""
    N, J = M.shape
    Sigma = implied_cov(Lam, Phi, sigma)
    mask = ~np.isnan(M)
    X = np.nan_to_num(M, nan=0.0)
    if mu is None:
        mu = np.zeros(J)
    pats, inv = np.unique(mask, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    ll = np.zeros(N)
    for p in range(pats.shape[0]):
        cols = np.flatnonzero(pats[p])
        if cols.size == 0:
            continue
        rows = np.flatnonzero(inv == p)
        So = Sigma[np.ix_(cols, cols)]
        L = np.linalg.cholesky(So)
        logdet = 2.0 * np.log(np.diag(L)).sum()
        d = (X[np.ix_(rows, cols)] - mu[cols]).T                 # [k, n_rows]
        z = solve_triangular(L, d, lower=True)
        quad = (z ** 2).sum(0)                                   # [n_rows]
        ll[rows] = -0.5 * (cols.size * LOG2PI + logdet + quad)
    return ll


def ppc_residual_correlations(M: np.ndarray, post, max_draws: int = 400, psi_floor: float = 0.05):
    """(B) Absolute fit. Compare each posterior draw's model-implied correlation to the observed
    pairwise correlation; return the Bayesian SRMR posterior (RMS off-diagonal residual) and the
    posterior-mean residual-correlation matrix. `psi_floor` matches the engine's residual-SD floor
    (the stored `sigma` is the HalfNormal; the model uses `psi_floor + sigma`)."""
    Lam, Phi, sig = _draws(post, "Lam"), _draws(post, "Phi"), psi_floor + _draws(post, "sigma")
    S = Lam.shape[0]
    idx = np.linspace(0, S - 1, min(S, max_draws)).astype(int)
    S_obs = pairwise_corr(M)
    J = S_obs.shape[0]
    iu = np.triu_indices(J, 1)
    obs_off = S_obs[iu]
    srmr = np.empty(len(idx))
    resid_sum = np.zeros((J, J))
    for t, s in enumerate(idx):
        R = cov_to_corr(implied_cov(Lam[s], Phi[s], sig[s]))
        resid = S_obs - R
        srmr[t] = np.sqrt(np.nanmean((obs_off - R[iu]) ** 2))
        resid_sum += np.nan_to_num(resid)
    return srmr, resid_sum / len(idx), S_obs


def pointwise_loglik(M: np.ndarray, post, max_draws: int = 400, psi_floor: float = 0.05) -> np.ndarray:
    """[S, N] per-patient observed-data log-lik across (thinned) posterior draws — for WAIC/LOO."""
    Lam, Phi, sig = _draws(post, "Lam"), _draws(post, "Phi"), psi_floor + _draws(post, "sigma")
    S = Lam.shape[0]
    idx = np.linspace(0, S - 1, min(S, max_draws)).astype(int)
    out = np.empty((len(idx), M.shape[0]))
    for t, s in enumerate(idx):
        out[t] = marginal_loglik(M, Lam[s], Phi[s], sig[s])
    return out


def waic(ll: np.ndarray) -> dict:
    """WAIC from an [S, N] pointwise log-lik matrix (per-patient observations).
    lppd = Σ_i log mean_s exp(ll_si); p_waic = Σ_i var_s(ll_si); WAIC = -2(lppd − p_waic)."""
    from scipy.special import logsumexp
    Sd = ll.shape[0]
    lppd_i = logsumexp(ll, axis=0) - np.log(Sd)
    p_i = ll.var(axis=0, ddof=1)
    elpd_i = lppd_i - p_i
    return dict(elpd_waic=float(elpd_i.sum()), p_waic=float(p_i.sum()),
                waic=float(-2.0 * elpd_i.sum()), se=float(np.sqrt(len(elpd_i)) * elpd_i.std(ddof=1)),
                elpd_i=elpd_i)


def waic_compare(models: dict[str, np.ndarray]) -> pd.DataFrame:
    """Rank models by WAIC (lower better); ΔWAIC and SE of the difference vs the best."""
    w = {k: waic(ll) for k, ll in models.items()}
    best = min(w, key=lambda k: w[k]["waic"])
    rows = []
    for k, wk in sorted(w.items(), key=lambda kv: kv[1]["waic"]):
        dse = float(np.sqrt(len(wk["elpd_i"])) * (wk["elpd_i"] - w[best]["elpd_i"]).std(ddof=1)) \
            if k != best else 0.0
        rows.append(dict(model=k, waic=round(wk["waic"], 1), elpd_waic=round(wk["elpd_waic"], 1),
                         p_waic=round(wk["p_waic"], 1), d_waic=round(wk["waic"] - w[best]["waic"], 1),
                         se_diff=round(dse, 1)))
    return pd.DataFrame(rows)


def tucker_phi(a: np.ndarray, b: np.ndarray) -> float:
    """Tucker's congruence coefficient between two loading vectors/columns."""
    num = float((a * b).sum())
    den = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return num / den if den > 0 else float("nan")


def loadings_long(post, items: list[str], factor_cols: list[str]) -> pd.DataFrame:
    """Posterior-mean Λ as a long (item, factor, loading) frame."""
    Lam = np.asarray(post["Lam"].mean(("chain", "draw")).values)
    return pd.DataFrame([dict(item=items[j], factor=factor_cols[c], loading=float(Lam[j, c]))
                         for j in range(len(items)) for c in range(len(factor_cols))])
