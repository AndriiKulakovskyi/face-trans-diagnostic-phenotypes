"""Stage B — patient embedding models (no imputation, graph-respecting).

This package produces patient embeddings from the Stage A harmonized dataset
and multi-relational similarity graph. The first PR ships baselines that
are pure ``numpy``/``scipy``/``sklearn`` (no torch dependency) so Stage C
can cluster and compare against DSM labels immediately.

A future PR will add torch-geometric GNN subclasses (R-GCN / HGT /
contrastive SSL) under the same :class:`BaseEmbeddingModel` interface.
"""

from face_stratification.models.base import (
    BaseEmbeddingModel,
    PatientEmbedding,
)
from face_stratification.models.baselines import (
    TransdiagnosticPCA,
    TransdiagnosticRawFeatures,
    TransdiagnosticUMAP,
)
from face_stratification.models.spectral import (
    MultiplexSpectral,
    TransdiagnosticSpectral,
)
from face_stratification.models.composite import (
    CompositeViewSpec,
    ConcatenatedEmbedding,
)
from face_stratification.models.raw_baseline import (
    RawFeatureBaseline,
)
from face_stratification.models.pipeline import (
    fit_and_save_embedding,
    fit_embedding,
)

__all__ = [
    # interface
    "BaseEmbeddingModel",
    "PatientEmbedding",
    # concrete models
    "TransdiagnosticPCA",
    "TransdiagnosticRawFeatures",
    "TransdiagnosticUMAP",
    "TransdiagnosticSpectral",
    "MultiplexSpectral",
    "CompositeViewSpec",
    "ConcatenatedEmbedding",
    # raw baseline
    "RawFeatureBaseline",
    # pipeline
    "fit_embedding",
    "fit_and_save_embedding",
]
