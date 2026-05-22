"""Raw feature baseline for stratification comparison.

The baseline = k-means on robust-normalized features (Xn), no graph,
no embedding reduction. All other methods must report metrics relative
to this baseline.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseEmbeddingModel, PatientEmbedding
from face_stratification.harmonization.harmonizer import HarmonizedDataset

logger = logging.getLogger(__name__)


class RawFeatureBaseline(BaseEmbeddingModel):
    """Baseline: use normalized features directly (no dimensionality reduction).

    Serves as the reference point for all other embedding methods.
    If a method doesn't beat this baseline, its added complexity is not justified.
    """

    name = "raw_baseline"

    def __init__(self, *, nan_fill: float = 0.0):
        super().__init__(nan_fill=nan_fill)
        self.nan_fill = nan_fill
        self._embedding: PatientEmbedding | None = None

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> RawFeatureBaseline:
        X = dataset.X.to_numpy(dtype=np.float64, na_value=np.nan)
        X = np.where(np.isnan(X), self.nan_fill, X)

        columns = [f"raw_{i}" for i in range(X.shape[1])]
        values = pd.DataFrame(X, index=dataset.X.index, columns=columns)

        self._embedding = PatientEmbedding(
            values=values,
            model_name=self.name,
            model_config=self.config,
            view_dims={self.name: X.shape[1]},
        )
        self._fitted = True
        return self

    def transform(self) -> PatientEmbedding:
        self._ensure_fitted()
        assert self._embedding is not None
        return self._embedding
