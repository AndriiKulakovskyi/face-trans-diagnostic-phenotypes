"""Evaluation infrastructure for Stage B: split, validation, stability, interpretability.

Provides leak-proof train/test splitting, cross-validation, permutation testing,
and comprehensive clustering quality metrics.
"""

from __future__ import annotations

from face_stratification.evaluation.split import (
    StratifiedCohortSplit,
    create_loco_splits,
    create_repeated_stratified_kfold,
    create_stratified_split,
)

__all__ = [
    "StratifiedCohortSplit",
    "create_stratified_split",
    "create_loco_splits",
    "create_repeated_stratified_kfold",
]
