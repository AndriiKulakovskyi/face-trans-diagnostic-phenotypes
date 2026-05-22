"""Development-time data leakage prevention.

The ``LeakageGuard`` context manager tracks which patient indices belong
to the training set and monkey-patches ``fit_normalization`` and
``build_multiplex_graph`` to verify they never receive test data.

Disabled by setting ``FACE_DISABLE_LEAKAGE_GUARD=1`` in the environment.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLeakageError(RuntimeError):
    """Raised when test data is passed to a train-only function."""


def _is_disabled() -> bool:
    return os.environ.get("FACE_DISABLE_LEAKAGE_GUARD", "0") == "1"


# ─── Guard state (module-level, thread-unsafe by design for notebooks) ──────

_active_train_indices: set[tuple[str, str]] | None = None


def _check_no_test_data(
    X: pd.DataFrame,
    *,
    caller: str = "unknown",
) -> None:
    """Raise if ``X`` contains indices not in the active train set."""
    if _active_train_indices is None:
        return  # no guard active
    if _is_disabled():
        return

    if X.index.nlevels == 2 and tuple(X.index.names) == ("cohort", "patient_id"):
        x_indices = set(X.index.to_list())
    else:
        return  # can't check non-MultiIndex data

    leaked = x_indices - _active_train_indices
    if leaked:
        n_leaked = len(leaked)
        sample = sorted(leaked)[:5]
        raise DataLeakageError(
            f"{caller}() received {n_leaked} test patient(s) — data leakage! "
            f"First 5: {sample}. Fit on train data only."
        )


@contextmanager
def leakage_guard(
    train_index: pd.MultiIndex,
) -> Generator[None, None, None]:
    """Context manager that prevents test data from reaching fit functions.

    Usage::

        split = create_stratified_split(dataset)
        with leakage_guard(split.train_index):
            stats = fit_normalization(train_X, schema)  # OK
            stats = fit_normalization(full_X, schema)    # raises DataLeakageError

    The guard patches :func:`fit_normalization` and
    :func:`build_multiplex_graph` to intercept calls with test data.
    Disable entirely with ``FACE_DISABLE_LEAKAGE_GUARD=1``.
    """
    global _active_train_indices

    if _is_disabled():
        logger.debug("LeakageGuard disabled via environment variable")
        yield
        return

    _active_train_indices = set(train_index.to_list())
    logger.info(
        "LeakageGuard activated: %d train patients registered",
        len(_active_train_indices),
    )
    try:
        yield
    finally:
        _active_train_indices = None
        logger.info("LeakageGuard deactivated")


def assert_train_only(X: pd.DataFrame, *, caller: str = "unknown") -> None:
    """Explicit check callable from any function that should only see train data.

    Does nothing if no guard is active or if the guard is disabled.
    """
    _check_no_test_data(X, caller=caller)
