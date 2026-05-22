"""Baseline Stage B embedding models — feature-based, no graph required.

The :class:`TransdiagnosticPCA` baseline runs a standard PCA on the 8-feature
data-driven transdiagnostic subset selected in Stage A. Because every
selected feature has 100% per-cohort coverage in practice, the PCA sees a
fully-observed matrix and does not need any imputation shim. This is the
most conservative baseline in the Stage B suite: every patient is embedded
strictly from measurements shared with every other patient.

A second baseline, :class:`TransdiagnosticRawFeatures`, skips PCA and
simply returns the normalized transdiagnostic features directly as the
embedding. It is useful as a bare-bones control for Stage C comparisons —
if a fancier embedding model cannot beat this, the effort is not justified.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    import umap
except ImportError:
    umap = None  # type: ignore[assignment]

from face_stratification.harmonization.feature_schema import FeatureSchema
from face_stratification.harmonization.harmonizer import HarmonizedDataset
from face_stratification.harmonization.normalization import (
    fit_normalization,
    transform_normalization,
)
from face_stratification.graph.transdiagnostic import (
    TransdiagnosticFeatureSet,
    select_transdiagnostic_features,
)
from face_stratification.models.base import (
    BaseEmbeddingModel,
    PatientEmbedding,
)

logger = logging.getLogger(__name__)


# ─── Transdiagnostic PCA ─────────────────────────────────────────────────────


class TransdiagnosticPCA(BaseEmbeddingModel):
    """PCA on the data-driven transdiagnostic feature subset.

    Pipeline
    --------
    1. Select the transdiagnostic feature set from the harmonized dataset
       (``select_transdiagnostic_features``). Features not in this set are
       ignored — it is the "trusted" subspace.
    2. Robust-normalize the selected columns (reusing Stage A's
       ``fit_normalization`` / ``transform_normalization``).
    3. For the rare missing cells that remain (a selected feature with
       ~100% coverage can still have a handful of NaNs), fill with 0 —
       which, on a median-centered robust z-score matrix, represents
       "exactly at the pooled median". This is the one small concession
       the baseline makes, and it only touches features that are already
       declared trustworthy across cohorts.
    4. Run ``sklearn.decomposition.PCA`` with a configurable number of
       components (default: ``min(n_features, 8)``).

    Parameters
    ----------
    n_components:
        Target dimensionality. Capped at ``|transdiagnostic_set|``.
    l2_normalize:
        If True, every patient's embedding row is L2-normalized after PCA.
        Helps comparability with graph-based spectral views.
    """

    name = "transdiagnostic_pca"

    def __init__(
        self,
        *,
        n_components: int = 8,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__(n_components=n_components, l2_normalize=l2_normalize)
        self._n_components = n_components
        self._l2 = l2_normalize
        self._embedding: pd.DataFrame | None = None
        self._feature_set: TransdiagnosticFeatureSet | None = None
        self._schema_version: str = "unknown"

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> TransdiagnosticPCA:
        schema = dataset.schema
        fs = select_transdiagnostic_features(dataset.X, dataset.metadata, schema)
        if fs.n_selected == 0:
            raise RuntimeError(
                "TransdiagnosticPCA: transdiagnostic feature set is empty. "
                "Lower schema.transdiagnostic_selection.min_cohort_coverage "
                "or check that every cohort has adequate coverage."
            )

        cols = list(fs.feature_ids)
        X_sub = dataset.X[cols]

        # Reuse Stage A normalization (fit on this subset).
        stats = fit_normalization(X_sub, schema)
        Xn = transform_normalization(X_sub, stats)

        # Fill the few remaining NaNs with 0 (= pooled median post-normalization).
        # The Category-A features in the transdiagnostic set are ~100% observed,
        # so this touches a handful of cells — audit emits the exact count.
        n_nan_filled = int(Xn.isna().sum().sum())
        if n_nan_filled > 0:
            logger.info(
                "TransdiagnosticPCA: filling %d residual NaN cells with 0 "
                "(= pooled median post-normalization) across %d trusted features",
                n_nan_filled,
                fs.n_selected,
            )
        Xn_filled = Xn.fillna(0.0).to_numpy(dtype=np.float64)

        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise ImportError(
                "TransdiagnosticPCA requires scikit-learn. "
                "Install the 'stratification' extra: pip install -e '.[stratification]'."
            ) from exc

        k = min(self._n_components, fs.n_selected, Xn_filled.shape[0])
        pca = PCA(n_components=k, random_state=0)
        components = pca.fit_transform(Xn_filled)
        if self._l2:
            components = self._l2_normalize_rows(components)

        col_names = [f"{self.name}_{i}" for i in range(k)]
        self._embedding = pd.DataFrame(
            components,
            index=dataset.X.index,
            columns=col_names,
            dtype=np.float64,
        )
        self._feature_set = fs
        self._schema_version = schema.version
        self._config.update(
            {
                "n_components_requested": self._n_components,
                "n_components_effective": k,
                "n_transdiagnostic_features": fs.n_selected,
                "transdiagnostic_feature_ids": list(fs.feature_ids),
                "n_residual_nan_filled": n_nan_filled,
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            }
        )
        self._fitted = True
        return self

    def transform(self) -> PatientEmbedding:
        self._ensure_fitted()
        assert self._embedding is not None
        return PatientEmbedding(
            values=self._embedding,
            model_name=self.name,
            model_config=self.config,
            view_dims={self.name: self._embedding.shape[1]},
            n_isolated_nodes=0,
            schema_version=self._schema_version,
        )


# ─── Raw transdiagnostic features (sanity control) ──────────────────────────


class TransdiagnosticRawFeatures(BaseEmbeddingModel):
    """Baseline-of-baselines: use the normalized transdiagnostic features as-is.

    Skips PCA entirely. Useful as a control for Stage C — any fancier
    model should be measurably better than this on downstream clustering
    metrics; if it isn't, the extra complexity is not justified.
    """

    name = "transdiagnostic_raw"

    def __init__(self, *, l2_normalize: bool = True) -> None:
        super().__init__(l2_normalize=l2_normalize)
        self._l2 = l2_normalize
        self._embedding: pd.DataFrame | None = None
        self._feature_set: TransdiagnosticFeatureSet | None = None
        self._schema_version: str = "unknown"

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> TransdiagnosticRawFeatures:
        schema = dataset.schema
        fs = select_transdiagnostic_features(dataset.X, dataset.metadata, schema)
        if fs.n_selected == 0:
            raise RuntimeError("transdiagnostic feature set is empty")

        cols = list(fs.feature_ids)
        X_sub = dataset.X[cols]
        stats = fit_normalization(X_sub, schema)
        Xn = transform_normalization(X_sub, stats).fillna(0.0)

        arr = Xn.to_numpy(dtype=np.float64)
        if self._l2:
            arr = self._l2_normalize_rows(arr)

        col_names = [f"{self.name}_{c}" for c in cols]
        self._embedding = pd.DataFrame(
            arr, index=dataset.X.index, columns=col_names, dtype=np.float64
        )
        self._feature_set = fs
        self._schema_version = schema.version
        self._config.update(
            {
                "n_features": fs.n_selected,
                "feature_ids": list(fs.feature_ids),
            }
        )
        self._fitted = True
        return self

    def transform(self) -> PatientEmbedding:
        self._ensure_fitted()
        assert self._embedding is not None
        return PatientEmbedding(
            values=self._embedding,
            model_name=self.name,
            model_config=self.config,
            view_dims={self.name: self._embedding.shape[1]},
            n_isolated_nodes=0,
            schema_version=self._schema_version,
        )


# ─── Transdiagnostic UMAP ───────────────────────────────────────────────────


class TransdiagnosticUMAP(BaseEmbeddingModel):
    """UMAP on the data-driven transdiagnostic feature subset.

    Mirrors :class:`TransdiagnosticPCA` but uses UMAP for non-linear
    dimensionality reduction with cosine distance, which better preserves
    local structure in the clinical feature space.

    Parameters
    ----------
    n_components:
        Target dimensionality.
    l2_normalize:
        If True, every patient's embedding row is L2-normalized after UMAP.
    """

    name = "transdiagnostic_umap"

    def __init__(
        self,
        *,
        n_components: int = 2,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__(n_components=n_components, l2_normalize=l2_normalize)
        self._n_components = n_components
        self._l2 = l2_normalize
        self._embedding: pd.DataFrame | None = None
        self._feature_set: TransdiagnosticFeatureSet | None = None
        self._schema_version: str = "unknown"

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
        metadata: Any | None = None,
    ) -> TransdiagnosticUMAP:
        if umap is None:
            raise ImportError(
                "TransdiagnosticUMAP requires umap-learn. "
                "Install it with: pip install umap-learn"
            )

        schema = dataset.schema
        fs = select_transdiagnostic_features(dataset.X, dataset.metadata, schema)
        if fs.n_selected == 0:
            raise RuntimeError(
                "TransdiagnosticUMAP: transdiagnostic feature set is empty. "
                "Lower schema.transdiagnostic_selection.min_cohort_coverage "
                "or check that every cohort has adequate coverage."
            )

        cols = list(fs.feature_ids)
        X_sub = dataset.X[cols]

        stats = fit_normalization(X_sub, schema)
        Xn = transform_normalization(X_sub, stats)

        n_nan_filled = int(Xn.isna().sum().sum())
        if n_nan_filled > 0:
            logger.info(
                "TransdiagnosticUMAP: filling %d residual NaN cells with 0 "
                "(= pooled median post-normalization) across %d trusted features",
                n_nan_filled,
                fs.n_selected,
            )
        Xn_filled = Xn.fillna(0.0).to_numpy(dtype=np.float64)

        k = min(self._n_components, fs.n_selected)
        reducer = umap.UMAP(
            n_components=k, metric="cosine", random_state=42,
        )
        components = reducer.fit_transform(Xn_filled)
        if self._l2:
            components = self._l2_normalize_rows(components)

        col_names = [f"{self.name}_{i}" for i in range(k)]
        self._embedding = pd.DataFrame(
            components,
            index=dataset.X.index,
            columns=col_names,
            dtype=np.float64,
        )
        self._feature_set = fs
        self._schema_version = schema.version
        self._config.update(
            {
                "n_components": k,
                "n_transdiagnostic_features": fs.n_selected,
                "transdiagnostic_feature_ids": list(fs.feature_ids),
                "n_residual_nan_filled": n_nan_filled,
            }
        )
        self._fitted = True
        return self

    def transform(self) -> PatientEmbedding:
        self._ensure_fitted()
        assert self._embedding is not None
        return PatientEmbedding(
            values=self._embedding,
            model_name=self.name,
            model_config=self.config,
            view_dims={self.name: self._embedding.shape[1]},
            n_isolated_nodes=0,
            schema_version=self._schema_version,
        )
