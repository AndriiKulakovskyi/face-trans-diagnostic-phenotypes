"""Proper scores for the representation benchmark.

The continuous-GAF backbone is scored with **CRPS** (a strictly proper score for the whole predictive
distribution) so a richer/sharper forecast is rewarded even when a binary threshold barely moves — the exact
blind spot of AUC that made ``ΔELPD +59`` look like ``ΔAUC +0.011``. Binary endpoints derived from the
forecast are scored with Brier / log-loss / calibration slope and ranked for *decisions* by net benefit
(decision-curve analysis, reused from :mod:`face.prognosis.clinical_value`).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# reuse the certified decision-curve + Brier/AUC kernels (do not reimplement)
from ..clinical_value import auc, brier, net_benefit  # noqa: F401


def crps_gaussian(y, mu, sigma) -> float:
    """Mean CRPS of Gaussian predictive ``N(mu, sigma)`` against observations ``y`` (Gneiting & Raftery 2007,
    closed form). Lower is better; units = the outcome scale. ``sigma`` is clipped away from 0."""
    y = np.asarray(y, dtype="float64")
    mu = np.asarray(mu, dtype="float64")
    sigma = np.clip(np.asarray(sigma, dtype="float64"), 1e-9, None)
    z = (y - mu) / sigma
    crps = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def crps_ensemble(y, samples) -> float:
    """Mean CRPS from an ensemble of predictive draws (for non-Gaussian forecasts, e.g. XGBoost quantiles or
    posterior draws). ``samples`` is ``[n_draw, N]``. Energy-form estimator
    ``CRPS = E|X - y| - 0.5 E|X - X'|`` (unbiased pairwise term)."""
    y = np.asarray(y, dtype="float64")
    s = np.asarray(samples, dtype="float64")
    if s.ndim != 2:
        raise ValueError("samples must be [n_draw, N]")
    m = s.shape[0]
    term1 = np.mean(np.abs(s - y[None, :]), axis=0)                      # E|X - y|, per patient
    # E|X - X'| via mean of pairwise abs diffs (sort trick keeps it O(m log m) per patient)
    ss = np.sort(s, axis=0)
    weights = (2 * np.arange(1, m + 1) - m - 1).astype("float64")[:, None]
    term2 = (2.0 / (m * m)) * np.sum(weights * ss, axis=0)               # = mean_{i,j}|x_i - x_j|
    return float(np.mean(term1 - 0.5 * term2))


def log_loss(y, p, *, eps: float = 1e-7) -> float:
    """Binary cross-entropy with clipped probabilities (a proper score; lower is better)."""
    y = np.asarray(y, dtype="float64")
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_slope(y, p, *, eps: float = 1e-6) -> float:
    """Logistic recalibration slope: fit ``logit(y) ~ a + b·logit(p)`` and return ``b``. ``b = 1`` = perfectly
    calibrated; ``b < 1`` = over-confident (typical of overfit raw models). Returns NaN if ``y`` is one class."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype="int64")
    if len(np.unique(y)) < 2:
        return float("nan")
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1 - eps)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=2000).fit(lp, y)             # ~unpenalised → recalibration fit
    return float(m.coef_[0, 0])


def net_benefit_band(y, p, lo: float = 0.05, hi: float = 0.50, step: float = 0.01) -> dict:
    """Decision-curve net benefit over the pre-registered threshold band ``[lo, hi]``."""
    thr = np.round(np.arange(lo, hi + 1e-9, step), 4)
    return net_benefit(y, p, thr)
