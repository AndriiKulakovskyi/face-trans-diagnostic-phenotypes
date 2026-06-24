"""XGBoost arms for the representation benchmark.

One **fixed, regularised** config is used for *every* arm, so the contrast isolates the representation, not
the tuning (a per-arm hyper-search would let the modeller, not the data, decide the winner). ``tree_method="hist"``
handles missing values natively, so the RAW arm keeps its NaNs — no imputation, ever.
"""
from __future__ import annotations

import numpy as np


def xgb_params(seed: int) -> dict:
    """A single regularised tabular config shared by all arms (strong but not over-fit on small N)."""
    return dict(
        n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
        reg_lambda=2.0, min_child_weight=5.0, random_state=seed, n_jobs=0, tree_method="hist",
    )


def oof_regress(X, y, folds, *, seed: int) -> np.ndarray:
    """Out-of-fold continuous predictions (GAF@V2 backbone), **averaged across CV repeats** (each row is in
    one test fold per repeat). NaNs in ``X`` are handled natively."""
    from xgboost import XGBRegressor

    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="float64")
    acc = np.zeros(len(y), dtype="float64")
    cnt = np.zeros(len(y), dtype="float64")
    for tr, te in folds:
        m = XGBRegressor(objective="reg:squarederror", **xgb_params(seed))
        m.fit(X[tr], y[tr])
        acc[te] += m.predict(X[te])
        cnt[te] += 1.0
    return acc / np.maximum(cnt, 1.0)


def oof_classify(X, y, folds, *, seed: int) -> np.ndarray:
    """Out-of-fold probabilities from a direct binary classifier (the cross-check on the derived path),
    averaged across CV repeats."""
    from xgboost import XGBClassifier

    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="int64")
    acc = np.zeros(len(y), dtype="float64")
    cnt = np.zeros(len(y), dtype="float64")
    for tr, te in folds:
        ytr = y[tr]
        if len(np.unique(ytr)) < 2:                     # degenerate train fold → predict prevalence
            acc[te] += float(ytr.mean())
            cnt[te] += 1.0
            continue
        m = XGBClassifier(objective="binary:logistic", eval_metric="logloss", **xgb_params(seed))
        m.fit(X[tr], ytr)
        acc[te] += m.predict_proba(X[te])[:, 1]
        cnt[te] += 1.0
    return acc / np.maximum(cnt, 1.0)
