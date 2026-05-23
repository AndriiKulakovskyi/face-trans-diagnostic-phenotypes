"""Internalized stratification engine (vendored from the sister
``face_stratification`` project, trimmed to exactly what the FACE
trans-diagnostic pipeline uses).

Provenance: the algorithmic modules (``masked_similarity``, ``spectral_base``,
``multipartite``, ``enrichment``, ``feature_schema``) are copied verbatim;
``harmonized_dataset`` and ``clustering`` are minimal extracts (the cohort-
specific builders, ``face_rlvr`` / ``cohort_adapters`` dependencies, the
``config/`` glossary loader, and unused clustering algorithms are dropped). No
behaviour relied on by the pipeline is changed.
"""
from __future__ import annotations

from .clustering import ClusterAssignment, bootstrap_stability, run_kmeans
from .enrichment import compute_cluster_feature_enrichment
from .feature_schema import (
    FeatureBlock,
    FeatureSchema,
    FeatureType,
    TemporalScope,
    UnifiedFeature,
)
from .harmonized_dataset import HarmonizedDataset
from .masked_similarity import masked_cosine, masked_similarity
from .multipartite import CoveragePartition, MultipartiteSpectralEmbedding
from .spectral_base import BaseEmbeddingModel, PatientEmbedding

__all__ = [
    "FeatureSchema",
    "FeatureType",
    "TemporalScope",
    "FeatureBlock",
    "UnifiedFeature",
    "HarmonizedDataset",
    "MultipartiteSpectralEmbedding",
    "CoveragePartition",
    "PatientEmbedding",
    "BaseEmbeddingModel",
    "masked_cosine",
    "masked_similarity",
    "compute_cluster_feature_enrichment",
    "run_kmeans",
    "bootstrap_stability",
    "ClusterAssignment",
]
