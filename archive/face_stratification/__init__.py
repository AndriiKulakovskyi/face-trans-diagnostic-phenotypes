"""face_stratification — transdiagnostic patient stratification on FACE V1 data.

This sub-project builds on ``face_rlvr.profiles`` to produce data-driven,
transdiagnostic patient clusters from the baseline (V1) visit of the four
FACE cohorts (BP, SZ, DR, ASP).

Design guarantees
-----------------
- **No imputation on the default path.** Every pairwise similarity is
  computed strictly on features observed by both patients (pairwise-complete
  masked cosine / euclidean / Gower).
- **Semantic overlap edge constraint.** An edge between two patients exists
  only if they share at least the block's ``min_shared_features`` observed
  measurements.
- **Data-driven transdiagnostic feature set.** A parallel edge type uses only
  features whose observed coverage is above a configurable threshold in every
  cohort — nothing "partially transdiagnostic" sneaks in.
- **V1-only.** Longitudinal / follow-up fields are rejected at four layers
  (Pydantic enum, Pydantic field validators, cohort adapters, runtime guard).
"""

from face_stratification.harmonization.harmonizer import (
    HarmonizedDataset,
    build_harmonized_dataset,
)
from face_stratification.harmonization.feature_schema import (
    FeatureBlock,
    FeatureSchema,
    FeatureType,
    TemporalScope,
    TransdiagnosticSelectionConfig,
    UnifiedFeature,
    load_feature_schema,
)
from face_stratification.models.base import (
    BaseEmbeddingModel,
    PatientEmbedding,
)
from face_stratification.models.composite import (
    ConcatenatedEmbedding,
    WeightedConcatenatedEmbedding,
)
from face_stratification.models.pipeline import (
    fit_and_save_embedding,
    fit_embedding,
)
from face_stratification.models.baselines import TransdiagnosticUMAP
from face_stratification.models.raw_baseline import RawFeatureBaseline
from face_stratification.models.kernel_methods import (
    DiffusionMapEmbedding,
    KernelPCAEmbedding,
)
from face_stratification.models.deep_baselines import VanillaAE, VAE
from face_stratification.harmonization.missingness import characterize_missingness
from face_stratification.harmonization.dsm_subtypes import extract_dsm_subtypes
from face_stratification.analysis.meta_stability import (
    MetaStabilityResult,
    compute_meta_stability,
)
from face_stratification.analysis.safety_analysis import (
    SafetyAnalysisResult,
    run_safety_analysis,
)
from face_stratification.graph.patient_similarity import (
    build_balanced_knn_graph,
    build_mutual_knn_graph,
)
from face_stratification.graph.multipartite import (
    CoveragePartition,
    MultipartiteSpectralEmbedding,
    identify_coverage_partitions,
)
from face_stratification.stage_b2.gcn import get_device
from face_stratification.clustering import (
    ClusterAssignment,
    ClusterMetrics,
    KSelectionResult,
    bootstrap_stability,
    compute_assignment_entropy,
    compute_cluster_metrics,
    identify_boundary_patients,
    kmeans_sweep,
    run_dual_criterion_k_selection,
    run_gmm_soft,
    run_kmeans,
)
from face_stratification.clustering.algorithms import (
    run_bayesian_gmm,
    run_gmm_variants,
    run_hdbscan,
    run_hierarchical,
    run_kmedoids,
    run_minibatch_kmeans,
    run_spectral_clustering,
)
from face_stratification.clustering.metrics import compute_information_theoretic
from face_stratification.evaluation import (
    StratifiedCohortSplit,
    create_loco_splits,
    create_repeated_stratified_kfold,
    create_stratified_split,
)

__all__ = [
    # Stage A
    "HarmonizedDataset",
    "build_harmonized_dataset",
    "FeatureBlock",
    "FeatureSchema",
    "FeatureType",
    "TemporalScope",
    "TransdiagnosticSelectionConfig",
    "UnifiedFeature",
    "load_feature_schema",
    # Stage B — models
    "BaseEmbeddingModel",
    "PatientEmbedding",
    "ConcatenatedEmbedding",
    "fit_embedding",
    "fit_and_save_embedding",
    # Stage B — clustering
    "ClusterAssignment",
    "ClusterMetrics",
    "KSelectionResult",
    "bootstrap_stability",
    "compute_assignment_entropy",
    "compute_cluster_metrics",
    "identify_boundary_patients",
    "kmeans_sweep",
    "run_dual_criterion_k_selection",
    "run_gmm_soft",
    "run_kmeans",
    # Harmonization extras
    "characterize_missingness",
    "extract_dsm_subtypes",
    # Models extras
    "RawFeatureBaseline",
    "TransdiagnosticUMAP",
    "KernelPCAEmbedding",
    "DiffusionMapEmbedding",
    "VanillaAE",
    "VAE",
    "WeightedConcatenatedEmbedding",
    # Clustering extras
    "run_kmedoids",
    "run_minibatch_kmeans",
    "run_gmm_variants",
    "run_bayesian_gmm",
    "run_hdbscan",
    "run_spectral_clustering",
    "run_hierarchical",
    "compute_information_theoretic",
    # Evaluation
    "StratifiedCohortSplit",
    "create_stratified_split",
    "create_loco_splits",
    "create_repeated_stratified_kfold",
    # Analysis
    "compute_meta_stability",
    "MetaStabilityResult",
    "run_safety_analysis",
    "SafetyAnalysisResult",
    # Graph
    "build_balanced_knn_graph",
    "build_mutual_knn_graph",
    "CoveragePartition",
    "MultipartiteSpectralEmbedding",
    "identify_coverage_partitions",
    # Stage B2
    "get_device",
]
