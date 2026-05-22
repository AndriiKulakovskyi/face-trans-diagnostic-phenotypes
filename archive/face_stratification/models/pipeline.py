"""End-to-end Stage B pipeline: harmonized dataset + graph → saved embedding.

This is a tiny orchestrator that wires Stage A outputs into one of the
Stage B models and persists the result as a parquet + manifest bundle.
It is the exact surface Stage C clustering will consume.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from face_stratification.harmonization.harmonizer import HarmonizedDataset
from face_stratification.harmonization.normalization import (
    fit_normalization,
    transform_normalization,
)
from face_stratification.graph.patient_similarity import build_multiplex_graph
from face_stratification.models.base import (
    BaseEmbeddingModel,
    PatientEmbedding,
)
from face_stratification.models.composite import ConcatenatedEmbedding

logger = logging.getLogger(__name__)


def fit_embedding(
    dataset: HarmonizedDataset,
    *,
    model: BaseEmbeddingModel | None = None,
    k_neighbours: int = 10,
    reuse_graph: Any | None = None,
) -> tuple[PatientEmbedding, Any]:
    """Fit a Stage B model on a Stage A dataset.

    Parameters
    ----------
    dataset:
        A :class:`HarmonizedDataset` produced by
        :func:`face_stratification.build_harmonized_dataset`.
    model:
        An already-instantiated :class:`BaseEmbeddingModel`. If ``None``,
        the default composite (
        :meth:`ConcatenatedEmbedding.build_default`) is used.
    k_neighbours:
        Passed to ``build_multiplex_graph`` if the graph needs to be built.
    reuse_graph:
        Optional pre-built multiplex graph. If ``None``, a fresh one is
        built from the normalized matrix.

    Returns
    -------
    ``(embedding, graph)``. The graph is returned alongside the embedding
    so callers can reuse it for other views / stages without re-building.
    """
    model = model or ConcatenatedEmbedding.build_default()

    # Normalization is deterministic and cheap — always re-fit.
    stats = fit_normalization(dataset.X, dataset.schema)
    Xn = transform_normalization(dataset.X, stats)

    if reuse_graph is None:
        logger.info("Building multiplex graph (k=%d)", k_neighbours)
        graph, _block_graphs, _td = build_multiplex_graph(
            Xn, dataset.schema, k=k_neighbours, metadata=dataset.metadata
        )
    else:
        graph = reuse_graph

    normalized_dataset = HarmonizedDataset(
        X=Xn,
        metadata=dataset.metadata,
        feature_metadata=dataset.feature_metadata,
        schema=dataset.schema,
    )
    embedding = model.fit_transform(normalized_dataset, graph=graph)
    logger.info(
        "Fitted %s: %d × %d (isolated nodes: %d)",
        embedding.model_name,
        embedding.n_patients,
        embedding.dim,
        embedding.n_isolated_nodes,
    )
    return embedding, graph


def fit_and_save_embedding(
    dataset: HarmonizedDataset,
    out_dir: str | Path,
    *,
    model: BaseEmbeddingModel | None = None,
    k_neighbours: int = 10,
) -> Path:
    """Fit the embedding and persist it to ``out_dir`` as parquet + manifest."""
    embedding, _graph = fit_embedding(
        dataset, model=model, k_neighbours=k_neighbours
    )
    return embedding.save(out_dir)
