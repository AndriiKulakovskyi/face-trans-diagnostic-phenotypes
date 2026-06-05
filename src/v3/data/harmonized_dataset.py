"""``HarmonizedDataset`` — the matrix contract consumed by the embedding engine.

Extracted verbatim from the vendored sister engine
(``face_stratification.harmonization.harmonizer.HarmonizedDataset``). The
cohort-specific builder functions and their ``face_rlvr`` / ``cohort_adapters``
dependencies are intentionally **not** carried over — ``v3.data`` constructs
this dataclass directly (see :func:`v3.data.adapter.to_harmonized_dataset`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .feature_schema import FeatureSchema


@dataclass
class HarmonizedDataset:
    """A unified patient × feature matrix plus its schema and metadata.

    Attributes
    ----------
    X:
        Unified feature matrix. Index is a ``MultiIndex[cohort, patient_id]``.
        Columns are feature ids; values are ``float`` (``NaN`` = missing).
    metadata:
        Side table indexed identically to ``X`` (``cohort``, ``dsm_diagnosis``).
        Never used during similarity / model training — kept for downstream
        comparison only.
    feature_metadata:
        One row per column of ``X`` describing its schema entry.
    schema:
        The :class:`FeatureSchema` used to build this dataset.
    """

    X: pd.DataFrame
    metadata: pd.DataFrame
    feature_metadata: pd.DataFrame
    schema: FeatureSchema

    def __post_init__(self) -> None:
        # Strict cross-sectional invariant: one row per (cohort, patient_id).
        if not self.X.index.is_unique:
            raise ValueError(
                "Harmonized matrix has duplicate (cohort, patient_id) rows — "
                "single-visit builds must contain exactly one row per patient."
            )
        if list(self.X.index) != list(self.metadata.index):
            raise ValueError("X and metadata indices do not match.")

    @property
    def n_patients(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def cohort_counts(self) -> pd.Series:
        return self.metadata["cohort"].value_counts().sort_index()

    def feature_availability(self) -> pd.DataFrame:
        """For each feature, return coverage (% non-null) per cohort and overall."""
        rows: list[dict[str, Any]] = []
        for feat_id in self.X.columns:
            row: dict[str, Any] = {"feature_id": feat_id}
            for cohort, sub in self.X.groupby(level="cohort"):
                row[f"coverage_{cohort}"] = float(sub[feat_id].notna().mean())
            row["coverage_total"] = float(self.X[feat_id].notna().mean())
            rows.append(row)
        return pd.DataFrame(rows).set_index("feature_id")
