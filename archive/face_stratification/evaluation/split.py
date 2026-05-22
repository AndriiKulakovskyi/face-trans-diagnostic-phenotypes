"""Leak-proof train/test splitting and cross-validation for Stage B.

All splits are stratified by both cohort AND DSM subtype to ensure
proportional representation across the highly unbalanced FACE cohorts
(DR=350 vs BP=5400). The smallest cohort (DR) retains ~70 test patients
at the default 20% split — the minimum for stable silhouette estimation.

Data leakage prevention protocol:
    1. ``fit_normalization()`` on train split only
    2. ``transform_normalization(test, train_stats)`` for test
    3. Graph built on train patients only
    4. GNN inductive inference: frozen encoder + test-to-train edges
    5. Clustering fitted on train; test assigned to nearest centroid
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from face_stratification.harmonization.harmonizer import HarmonizedDataset

logger = logging.getLogger(__name__)


@dataclass
class StratifiedCohortSplit:
    """A single train/test (or train/val) split of the harmonized dataset.

    Indices are aligned with the ``HarmonizedDataset.X`` MultiIndex.
    """

    train_idx: np.ndarray       # integer positional indices into X
    test_idx: np.ndarray        # integer positional indices into X
    train_index: pd.MultiIndex  # (cohort, patient_id) labels
    test_index: pd.MultiIndex   # (cohort, patient_id) labels
    split_name: str = "default"
    split_seed: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)

    def train_dataset(self, dataset: HarmonizedDataset) -> HarmonizedDataset:
        """Slice the dataset to train patients only."""
        return HarmonizedDataset(
            X=dataset.X.iloc[self.train_idx],
            metadata=dataset.metadata.iloc[self.train_idx],
            feature_metadata=dataset.feature_metadata,
            schema=dataset.schema,
        )

    def test_dataset(self, dataset: HarmonizedDataset) -> HarmonizedDataset:
        """Slice the dataset to test patients only."""
        return HarmonizedDataset(
            X=dataset.X.iloc[self.test_idx],
            metadata=dataset.metadata.iloc[self.test_idx],
            feature_metadata=dataset.feature_metadata,
            schema=dataset.schema,
        )

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for logging and diagnostics."""
        return {
            "split_name": self.split_name,
            "seed": self.split_seed,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_fraction": self.n_train / (self.n_train + self.n_test),
            **self.metadata,
        }


def _build_stratification_key(dataset: HarmonizedDataset) -> np.ndarray:
    """Build a combined cohort+DSM stratification key per patient.

    Uses ``"{cohort}_{dsm_diagnosis}"`` for fine-grained stratification.
    Rare subtype bins (<5 patients) are collapsed to ``"{cohort}_other"``
    to avoid StratifiedShuffleSplit failures on tiny bins.
    """
    cohort = dataset.metadata["cohort"].values
    dsm = dataset.metadata["dsm_diagnosis"].values
    raw_keys = np.array([f"{c}_{d}" for c, d in zip(cohort, dsm)])

    # Collapse rare bins
    unique, counts = np.unique(raw_keys, return_counts=True)
    rare = set(unique[counts < 5])
    if rare:
        logger.info(
            "Collapsing %d rare DSM subtype bins to '{cohort}_other': %s",
            len(rare),
            sorted(rare),
        )
        cohort_for_rare = {k: k.split("_", 1)[0] for k in rare}
        collapsed = np.array([
            f"{cohort_for_rare[k]}_other" if k in rare else k
            for k in raw_keys
        ])
        return collapsed
    return raw_keys


def create_stratified_split(
    dataset: HarmonizedDataset,
    *,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> StratifiedCohortSplit:
    """Create a single stratified train/test split.

    Stratified by cohort AND DSM subtype to preserve proportional
    representation in both splits. DR (N~350) gets ~70 test patients.

    Parameters
    ----------
    dataset:
        The harmonized dataset to split.
    test_fraction:
        Fraction of patients held out for testing.
    seed:
        Random seed for reproducibility.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    strat_key = _build_stratification_key(dataset)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_fraction,
        random_state=seed,
    )
    train_pos, test_pos = next(splitter.split(dataset.X, strat_key))

    train_index = dataset.X.index[train_pos]
    test_index = dataset.X.index[test_pos]

    # Compute per-cohort counts for logging
    train_cohorts = pd.Series(train_index.get_level_values("cohort")).value_counts().to_dict()
    test_cohorts = pd.Series(test_index.get_level_values("cohort")).value_counts().to_dict()

    split = StratifiedCohortSplit(
        train_idx=train_pos,
        test_idx=test_pos,
        train_index=train_index,
        test_index=test_index,
        split_name="stratified_train_test",
        split_seed=seed,
        metadata={
            "test_fraction": test_fraction,
            "train_cohort_counts": train_cohorts,
            "test_cohort_counts": test_cohorts,
        },
    )

    logger.info(
        "Created stratified split: %d train / %d test (seed=%d)",
        split.n_train,
        split.n_test,
        seed,
    )
    for cohort in sorted(set(train_cohorts) | set(test_cohorts)):
        n_tr = train_cohorts.get(cohort, 0)
        n_te = test_cohorts.get(cohort, 0)
        logger.info("  %s: %d train / %d test (%.1f%%)", cohort, n_tr, n_te,
                     100 * n_te / (n_tr + n_te) if (n_tr + n_te) else 0)

    return split


def create_loco_splits(
    dataset: HarmonizedDataset,
) -> list[StratifiedCohortSplit]:
    """Create leave-one-cohort-out (LOCO) splits.

    Returns 4 splits, each holding out one entire cohort. These are NOT
    for training — they measure stability: does the clustering structure
    survive removal of an entire diagnostic category?

    The held-out cohort goes into ``test_idx``; the remaining 3 cohorts
    go into ``train_idx``.
    """
    cohort_labels = dataset.metadata["cohort"].values
    unique_cohorts = sorted(pd.unique(cohort_labels))
    all_idx = np.arange(len(dataset.X))

    splits = []
    for held_out in unique_cohorts:
        test_mask = cohort_labels == held_out
        train_pos = all_idx[~test_mask]
        test_pos = all_idx[test_mask]

        splits.append(StratifiedCohortSplit(
            train_idx=train_pos,
            test_idx=test_pos,
            train_index=dataset.X.index[train_pos],
            test_index=dataset.X.index[test_pos],
            split_name=f"loco_{held_out}",
            split_seed=0,
            metadata={
                "held_out_cohort": held_out,
                "n_held_out": int(test_mask.sum()),
            },
        ))
        logger.info(
            "LOCO split: held out %s (%d patients), train=%d",
            held_out,
            int(test_mask.sum()),
            int((~test_mask).sum()),
        )

    return splits


def create_repeated_stratified_kfold(
    dataset: HarmonizedDataset,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> list[StratifiedCohortSplit]:
    """Create repeated stratified k-fold splits for hyperparameter selection.

    Returns ``n_splits * n_repeats`` splits. Each fold is stratified by
    cohort to ensure proportional representation.

    These should only be used on the TRAIN portion of the main split
    (never on test data) to avoid double-dipping.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    strat_key = _build_stratification_key(dataset)
    rkf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )

    splits = []
    for fold_i, (train_pos, val_pos) in enumerate(rkf.split(dataset.X, strat_key)):
        repeat_num = fold_i // n_splits
        fold_num = fold_i % n_splits
        splits.append(StratifiedCohortSplit(
            train_idx=train_pos,
            test_idx=val_pos,
            train_index=dataset.X.index[train_pos],
            test_index=dataset.X.index[val_pos],
            split_name=f"cv_r{repeat_num}_f{fold_num}",
            split_seed=seed,
            metadata={
                "repeat": repeat_num,
                "fold": fold_num,
                "n_splits": n_splits,
                "n_repeats": n_repeats,
            },
        ))

    logger.info(
        "Created %d CV splits (%d-fold × %d repeats, seed=%d)",
        len(splits),
        n_splits,
        n_repeats,
        seed,
    )
    return splits
