"""Clinical-value metrics — does the map change *decisions*, in the clinician's currency?

The M4.2–4.4 Bayesian EIV ladder established uncertainty-aware incremental validity (ΔELPD). This
module translates that into the metrics a clinical-prediction paper (TRIPOD) reports: cross-validated
**discrimination** (AUC), **calibration** (Brier + reliability), and **net benefit** (decision-curve
analysis) — "is acting on the map worth it across plausible decision thresholds?". Discrimination is
estimated by **patient-level K-fold cross-validation** of a logistic model (proper internal
validation, no in-sample optimism); the paired AUC gain (map vs the clinician's reference) gets a
bootstrap CI. Frequentist CV is the field standard here and is fast (no MCMC).
"""
from __future__ import annotations

import numpy as np


def cv_predict(X, y, *, n_splits: int = 5, seed: int = 20260610, C: float = 1.0):
    """Out-of-fold predicted probabilities from stratified K-fold logistic regression (standardized
    inputs assumed). Returns p_hat [N] aligned to the rows of X."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="int64")
    p = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = LogisticRegression(max_iter=2000, C=C).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def brier(y, p) -> float:
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(np.asarray(y), np.asarray(p)))


def paired_auc_delta(y, p_ref, p_new, *, n_boot: int = 2000, seed: int = 20260610):
    """Bootstrap the paired AUC gain (new − reference) over patients. Returns (delta, lo, hi, p_gt0)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    base = auc(y, p_new) - auc(y, p_ref)
    n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            deltas[b] = np.nan
            continue
        deltas[b] = auc(yb, p_new[idx]) - auc(yb, p_ref[idx])
    deltas = deltas[np.isfinite(deltas)]
    lo, hi = np.percentile(deltas, [3, 97])
    return float(base), float(lo), float(hi), float((deltas > 0).mean())


def net_benefit(y, p, thresholds):
    """Decision-curve net benefit at each threshold probability `pt`:
        NB(pt) = TP/N − FP/N · pt/(1−pt),  treating p ≥ pt as 'act'.
    Plus the treat-all and treat-none reference strategies."""
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    prev = y.mean()
    nb_model, nb_all = [], []
    for pt in thresholds:
        flag = p >= pt
        tp = np.sum(flag & (y == 1))
        fp = np.sum(flag & (y == 0))
        w = pt / (1 - pt)
        nb_model.append(tp / n - fp / n * w)
        nb_all.append(prev - (1 - prev) * w)            # treat-all
    return {"thresholds": np.asarray(thresholds), "model": np.asarray(nb_model),
            "treat_all": np.asarray(nb_all), "treat_none": np.zeros(len(thresholds))}
