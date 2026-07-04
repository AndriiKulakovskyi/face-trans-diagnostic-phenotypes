"""Cross-validation folds that are identical across every feature arm.

The whole benchmark is a *paired* comparison of representations, so the contrast must isolate the
representation, not the resampling. ``make_folds`` builds the fold assignment from the outcome and cohort
alone (never from the features), so RAW / LATENT / REF all train and test on exactly the same patients —
each arm just indexes its own design matrix by the shared row positions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import SEED


def make_folds(y, cohort, *, n_splits: int = 5, n_repeats: int = 1, seed: int = SEED):
    """Repeated stratified K-fold over row *positions*, stratified by ``cohort × y`` so each fold preserves
    the (sparse) event rate within each cohort. Returns a list of ``(train_pos, test_pos)`` integer-position
    arrays — feature-independent, hence identical across arms.

    ``y`` is the (binary) target on the *eligible* rows; ``cohort`` the matching cohort labels.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    y = np.asarray(y)
    cohort = np.asarray(cohort).astype(str)
    if len(y) != len(cohort):
        raise ValueError("y and cohort must align")
    strata = pd.Series(cohort).str.cat(pd.Series(y.astype(int)).astype(str), sep="|").to_numpy()
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    dummy = np.zeros((len(y), 1))
    return [(tr, te) for tr, te in rskf.split(dummy, strata)]


def oof_indices(folds) -> np.ndarray:
    """Sanity helper: the concatenated test positions of one repeat cover every row exactly once."""
    return np.sort(np.concatenate([te for _, te in folds]))
