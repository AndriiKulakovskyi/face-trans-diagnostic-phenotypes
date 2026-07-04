"""Efficiency (learning curves) and transportability (LOCO) for the representation arms.

* **learning_curve** — AUC vs training-set size N. H2: the 9-dim map (LAT-A) should generalise better than raw
  (143 sparse features) at small N, converging as N grows.
* **loco** — leave-one-cohort-out out-of-sample AUC. H4: a compact, meaningful representation should *transfer*
  across cohorts better (smaller OOS drop) than a raw black box that can latch onto cohort-specific quirks.

Both reuse the harness blocks and the same fixed XGBoost config; binary endpoint via the direct classifier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import SEED, cv, data, metrics, models
from .harness import ARM_SPECS, _blocks


def _design(blocks: dict, arm: str) -> np.ndarray:
    return np.hstack([blocks[k] for k in ARM_SPECS[arm]])


def learning_curve(*, target: str = "egf_recovery", horizon: str = "V2", scope: str = "pooled",
                   arms: tuple[str, ...] = ("REF", "REF+RAW", "REF+LAT-A"),
                   grid: tuple = (150, 300, 500, 800, None), seeds: tuple = (1, 2, 3),
                   n_splits: int = 5, seed: int = SEED) -> pd.DataFrame:
    cohorts = None if scope == "pooled" else ("bp", "dr")
    frame = data.assemble(cohorts=cohorts)
    E0 = frame[data.eligible(frame, target, horizon)]
    raw = data.load_raw()
    rows = []
    for N in grid:
        for s in seeds:
            E = E0 if (N is None or N >= len(E0)) else E0.sample(N, random_state=seed + s)
            y = E[f"ep_{target}__{horizon}"].to_numpy("int64")
            if len(np.unique(y)) < 2:
                continue
            folds = cv.make_folds(y, data.cohort_of(E), n_splits=n_splits, n_repeats=1, seed=seed + s)
            blocks = _blocks(E, raw.reindex(E.index))
            for arm in arms:
                p = models.oof_classify(_design(blocks, arm), y, folds, seed=seed + s)
                rows.append({"arm": arm, "N": int(len(y)), "seed": s, "auc": metrics.auc(y, p)})
    df = pd.DataFrame(rows)
    return df.groupby(["arm", "N"]).auc.agg(["mean", "std", "count"]).reset_index()


def loco(*, target: str = "egf_recovery", horizon: str = "V2",
         arms: tuple[str, ...] = ("REF", "REF+RAW", "REF+LAT-A"), seed: int = SEED) -> pd.DataFrame:
    from xgboost import XGBClassifier

    frame = data.assemble(cohorts=None)
    E = frame[data.eligible(frame, target, horizon)]
    coh = data.cohort_of(E)
    y = E[f"ep_{target}__{horizon}"].to_numpy("int64")
    blocks = _blocks(E, data.load_raw().reindex(E.index))     # built on full eligible → aligned dx columns
    designs = {arm: _design(blocks, arm) for arm in arms}
    rows = []
    for held in np.unique(coh):
        tr, te = coh != held, coh == held
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        for arm in arms:
            X = designs[arm]
            m = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                              **models.xgb_params(seed)).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            rows.append({"target": target, "held_out": held, "arm": arm, "n_test": int(te.sum()),
                         "events_test": int(y[te].sum()), "auc_oos": metrics.auc(y[te], p)})
    return pd.DataFrame(rows)
