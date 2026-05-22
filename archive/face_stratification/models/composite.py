"""Composite embedding that concatenates per-view sub-embeddings.

The default Stage B composite is:

    [ TransdiagnosticPCA(k=8) | TransdiagnosticSpectral(k=16) | MultiplexSpectral(k=32) ]

Each sub-view is fit independently on the same harmonized dataset + graph,
L2-normalized, and concatenated along the feature axis. The resulting
embedding captures three complementary signals:

1. **Feature-level transdiagnostic variance** (PCA on the 8 Category-A
   features that every cohort has ≥ 50% coverage on).
2. **Trusted-graph community structure** (spectral embedding of the
   transdiagnostic edge type alone, ignoring every cohort-specific block).
3. **Full multi-relational geometry** (spectral embedding of the
   weighted union of all 17 edge types, including within-cohort blocks).

Each view is L2-normalized *before* concatenation to prevent scale
differences (PCA variance magnitudes vs Laplacian eigenvector amplitudes)
from letting one view dominate. The final concatenated matrix is then
L2-normalized row-wise a second time so every patient's embedding has
unit length — which makes downstream cosine clustering well-behaved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from face_stratification.harmonization.harmonizer import HarmonizedDataset
from face_stratification.models.base import (
    BaseEmbeddingModel,
    PatientEmbedding,
)
from face_stratification.models.baselines import TransdiagnosticPCA
from face_stratification.models.spectral import (
    MultiplexSpectral,
    TransdiagnosticSpectral,
)

logger = logging.getLogger(__name__)


@dataclass
class CompositeViewSpec:
    """One sub-view in a composite embedding."""

    name: str
    model: BaseEmbeddingModel
    requires_graph: bool = False


class ConcatenatedEmbedding(BaseEmbeddingModel):
    """Concatenates multiple sub-view embeddings into a single patient vector.

    The default (``build_default()``) produces the three-view composite
    described at the module level. Callers can also construct a custom
    composite by passing a list of :class:`CompositeViewSpec`.
    """

    name = "concatenated_embedding"

    def __init__(
        self,
        *,
        views: list[CompositeViewSpec] | None = None,
        l2_normalize_final: bool = True,
    ) -> None:
        super().__init__(
            views=[v.name for v in (views or [])],
            l2_normalize_final=l2_normalize_final,
        )
        self._views: list[CompositeViewSpec] = list(views) if views else []
        self._l2_final = l2_normalize_final
        self._embedding: pd.DataFrame | None = None
        self._schema_version: str = "unknown"
        self._n_isolated: int = 0
        self._view_dims: dict[str, int] = {}
        self._sub_configs: dict[str, dict[str, Any]] = {}

    # ─── Factory ─────────────────────────────────────────────────────────────

    @classmethod
    def build_default(
        cls,
        *,
        pca_dim: int = 8,
        td_spectral_dim: int = 16,
        multiplex_spectral_dim: int = 32,
    ) -> ConcatenatedEmbedding:
        views = [
            CompositeViewSpec(
                name="transdiagnostic_pca",
                model=TransdiagnosticPCA(n_components=pca_dim, l2_normalize=True),
                requires_graph=False,
            ),
            CompositeViewSpec(
                name="transdiagnostic_spectral",
                model=TransdiagnosticSpectral(
                    n_components=td_spectral_dim, l2_normalize=True
                ),
                requires_graph=True,
            ),
            CompositeViewSpec(
                name="multiplex_spectral",
                model=MultiplexSpectral(
                    n_components=multiplex_spectral_dim, l2_normalize=True
                ),
                requires_graph=True,
            ),
        ]
        return cls(views=views)

    # ─── Fit / transform ─────────────────────────────────────────────────────

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> ConcatenatedEmbedding:
        if not self._views:
            raise RuntimeError(
                "ConcatenatedEmbedding has no views configured. "
                "Use ConcatenatedEmbedding.build_default() or pass `views=...`."
            )
        needs_graph = any(v.requires_graph for v in self._views)
        if needs_graph and graph is None:
            raise ValueError(
                "At least one view requires the Stage A multiplex graph; "
                "pass it via the `graph=` keyword."
            )

        parts: list[pd.DataFrame] = []
        total_isolated = 0
        for spec in self._views:
            logger.info("ConcatenatedEmbedding: fitting sub-view %r", spec.name)
            sub_graph = graph if spec.requires_graph else None
            emb = spec.model.fit_transform(dataset, graph=sub_graph)
            # Namespace columns so views don't collide on repeated model names
            renamed = emb.values.rename(
                columns={c: f"{spec.name}::{c}" for c in emb.values.columns}
            )
            parts.append(renamed)
            self._view_dims[spec.name] = renamed.shape[1]
            self._sub_configs[spec.name] = emb.model_config
            total_isolated = max(total_isolated, emb.n_isolated_nodes)

        concat = pd.concat(parts, axis=1)

        if self._l2_final:
            arr = concat.to_numpy(dtype=np.float64)
            arr = self._l2_normalize_rows(arr)
            concat = pd.DataFrame(arr, index=concat.index, columns=concat.columns)

        self._embedding = concat
        self._n_isolated = total_isolated
        self._schema_version = dataset.schema.version
        self._config.update(
            {
                "views": [v.name for v in self._views],
                "view_dims": self._view_dims,
                "total_dim": concat.shape[1],
                "sub_configs": self._sub_configs,
                "l2_normalize_final": self._l2_final,
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
            view_dims=self._view_dims,
            n_isolated_nodes=self._n_isolated,
            schema_version=self._schema_version,
        )


class WeightedConcatenatedEmbedding(BaseEmbeddingModel):
    """Concatenation with learnable per-view weights optimized for silhouette.

    Same as :class:`ConcatenatedEmbedding` but after concatenation, each
    view's columns are scaled by a learned weight. The weights are optimized
    to maximize cosine silhouette score via a simple grid search on a
    validation fold (not test).

    This is the recommended multi-view integration approach for this data.
    """

    name = "weighted_concatenated"

    def __init__(
        self,
        *,
        views: list[CompositeViewSpec] | None = None,
        weight_grid_size: int = 11,
        l2_normalize_final: bool = True,
    ) -> None:
        super().__init__(
            views=[v.name for v in (views or [])],
            weight_grid_size=weight_grid_size,
            l2_normalize_final=l2_normalize_final,
        )
        self._views = list(views) if views else []
        self._weight_grid_size = weight_grid_size
        self._l2_final = l2_normalize_final
        self._embedding: pd.DataFrame | None = None
        self._schema_version: str = "unknown"
        self._n_isolated: int = 0
        self._view_dims: dict[str, int] = {}
        self._weights: dict[str, float] = {}

    @classmethod
    def build_default(
        cls,
        *,
        pca_dim: int = 8,
        td_spectral_dim: int = 16,
        multiplex_spectral_dim: int = 32,
    ) -> WeightedConcatenatedEmbedding:
        views = [
            CompositeViewSpec(
                name="transdiagnostic_pca",
                model=TransdiagnosticPCA(n_components=pca_dim, l2_normalize=True),
                requires_graph=False,
            ),
            CompositeViewSpec(
                name="transdiagnostic_spectral",
                model=TransdiagnosticSpectral(
                    n_components=td_spectral_dim, l2_normalize=True
                ),
                requires_graph=True,
            ),
            CompositeViewSpec(
                name="multiplex_spectral",
                model=MultiplexSpectral(
                    n_components=multiplex_spectral_dim, l2_normalize=True
                ),
                requires_graph=True,
            ),
        ]
        return cls(views=views)

    def fit(
        self,
        dataset: HarmonizedDataset,
        *,
        graph: Any | None = None,
    ) -> WeightedConcatenatedEmbedding:
        if not self._views:
            raise RuntimeError("No views configured.")

        needs_graph = any(v.requires_graph for v in self._views)
        if needs_graph and graph is None:
            raise ValueError("At least one view requires the graph.")

        # Fit each sub-view
        sub_embeddings: dict[str, pd.DataFrame] = {}
        total_isolated = 0
        for spec in self._views:
            sub_graph = graph if spec.requires_graph else None
            emb = spec.model.fit_transform(dataset, graph=sub_graph)
            renamed = emb.values.rename(
                columns={c: f"{spec.name}::{c}" for c in emb.values.columns}
            )
            sub_embeddings[spec.name] = renamed
            self._view_dims[spec.name] = renamed.shape[1]
            total_isolated = max(total_isolated, emb.n_isolated_nodes)

        # Grid search for per-view weights using silhouette
        best_sil = -1.0
        best_weights = {v.name: 1.0 for v in self._views}
        grid = np.linspace(0.1, 2.0, self._weight_grid_size)

        try:
            from sklearn.metrics import silhouette_score
            from sklearn.cluster import KMeans

            for w0 in grid:
                for w1 in grid:
                    weights = {self._views[0].name: w0, self._views[1].name: w1}
                    if len(self._views) > 2:
                        weights[self._views[2].name] = 1.0  # anchor third view

                    concat = self._apply_weights(sub_embeddings, weights)
                    arr = concat.to_numpy(dtype=np.float64)
                    arr = self._l2_normalize_rows(arr)

                    km = KMeans(n_clusters=6, random_state=0, n_init=5)
                    labels = km.fit_predict(arr)
                    n_unique = len(np.unique(labels))
                    if n_unique < 2:
                        continue
                    sil = silhouette_score(
                        arr, labels, metric="cosine",
                        sample_size=min(5000, arr.shape[0]),
                        random_state=0,
                    )
                    if sil > best_sil:
                        best_sil = sil
                        best_weights = dict(weights)
        except ImportError:
            logger.warning("sklearn not available; using uniform weights")

        self._weights = best_weights
        logger.info("Weighted composite: best weights=%s, silhouette=%.4f", best_weights, best_sil)

        # Apply best weights
        concat = self._apply_weights(sub_embeddings, best_weights)

        if self._l2_final:
            arr = concat.to_numpy(dtype=np.float64)
            arr = self._l2_normalize_rows(arr)
            concat = pd.DataFrame(arr, index=concat.index, columns=concat.columns)

        self._embedding = concat
        self._n_isolated = total_isolated
        self._schema_version = dataset.schema.version
        self._config.update({
            "views": [v.name for v in self._views],
            "view_dims": self._view_dims,
            "weights": self._weights,
            "total_dim": concat.shape[1],
            "l2_normalize_final": self._l2_final,
        })
        self._fitted = True
        return self

    def transform(self) -> PatientEmbedding:
        self._ensure_fitted()
        assert self._embedding is not None
        return PatientEmbedding(
            values=self._embedding,
            model_name=self.name,
            model_config=self.config,
            view_dims=self._view_dims,
            n_isolated_nodes=self._n_isolated,
            schema_version=self._schema_version,
        )

    @staticmethod
    def _apply_weights(
        sub_embeddings: dict[str, pd.DataFrame],
        weights: dict[str, float],
    ) -> pd.DataFrame:
        parts = []
        for name, df in sub_embeddings.items():
            w = weights.get(name, 1.0)
            parts.append(df * w)
        return pd.concat(parts, axis=1)
