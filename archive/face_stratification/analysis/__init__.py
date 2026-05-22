"""Stage B review: ablation, enrichment, medoid extraction, visualization.

This package provides the tooling for the Stage B review pass:

- :mod:`face_stratification.analysis.ablation` — global vs per-cohort
  normalization ablation study.
- :mod:`face_stratification.analysis.enrichment` — per-cluster feature
  enrichment with Benjamini-Hochberg FDR.
- :mod:`face_stratification.analysis.medoids` — per-cluster medoid
  extraction and French-vignette retrieval via ``face_rlvr``.
- :mod:`face_stratification.analysis.visualization` — matplotlib helpers
  for 2D projection scatter plots, cluster × cohort heatmaps, and
  per-cluster enrichment bar charts.
- :mod:`face_stratification.analysis.treatment_validation` — treatment
  response proxy computation and cluster validation.
"""

from face_stratification.analysis.ablation import (
    AblationResult,
    run_normalization_ablation,
)
from face_stratification.analysis.enrichment import (
    FeatureEnrichmentResult,
    compute_cluster_feature_enrichment,
)
from face_stratification.analysis.medoids import (
    ClusterMedoid,
    MedoidVignetteResult,
    extract_cluster_medoids,
    fetch_medoid_vignettes,
)
from face_stratification.analysis.meta_stability import (
    MetaStabilityResult,
    compute_meta_stability,
)
from face_stratification.analysis.safety_analysis import (
    SafetyAnalysisResult,
    run_safety_analysis,
)
from face_stratification.analysis.treatment_validation import (
    TreatmentValidationResult,
    compute_adherence_by_cluster,
    compute_functioning_by_cluster,
    compute_treatment_profiles,
    run_treatment_validation,
)
from face_stratification.analysis.visualization import (
    plot_cluster_cohort_heatmap,
    plot_embedding_projection,
    plot_enrichment_bars,
    plot_kmeans_sweep,
    tsne_project,
    umap_project,
)

__all__ = [
    "AblationResult",
    "run_normalization_ablation",
    "FeatureEnrichmentResult",
    "compute_cluster_feature_enrichment",
    "ClusterMedoid",
    "MedoidVignetteResult",
    "extract_cluster_medoids",
    "fetch_medoid_vignettes",
    "MetaStabilityResult",
    "compute_meta_stability",
    "SafetyAnalysisResult",
    "run_safety_analysis",
    "TreatmentValidationResult",
    "compute_adherence_by_cluster",
    "compute_functioning_by_cluster",
    "compute_treatment_profiles",
    "run_treatment_validation",
    "plot_cluster_cohort_heatmap",
    "plot_embedding_projection",
    "plot_enrichment_bars",
    "plot_kmeans_sweep",
    "tsne_project",
    "umap_project",
]
