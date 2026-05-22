"""Stage C — consensus clustering, formal DSM comparison, narrative cards.

This package consumes a Stage A :class:`HarmonizedDataset` and a Stage B
:class:`PatientEmbedding` and produces:

- a multi-algorithm consensus partition of the cohort,
- per-patient confidence scores from the co-association matrix,
- a formal cluster-vs-DSM comparison (chi², Cramér's V, purity, entropy),
- a full ablation grid over (algorithm × k × view),
- per-cluster clinical narrative cards with French ``face_rlvr`` vignettes.
"""

from face_stratification.stage_c.algorithms import (
    ALGORITHMS,
    run_algorithm,
    run_gmm,
    run_kmeans,
    run_spectral,
    run_ward,
)
from face_stratification.stage_c.consensus import (
    ConsensusResult,
    align_labels_to_reference,
    build_coassociation_matrix,
    compute_per_patient_confidence,
    consensus_partition,
    run_consensus_clustering,
)
from face_stratification.stage_c.comparison import (
    FullDSMComparison,
    chi_square_independence,
    cramers_v,
    full_dsm_comparison,
    per_cluster_cohort_entropy,
    per_cohort_purity,
)
from face_stratification.stage_c.ablation import (
    compute_optimization_score,
    pick_best_configuration,
    run_algorithm_k_grid,
    run_embedding_view_ablation,
)
from face_stratification.stage_c.narrative import (
    ClusterCard,
    build_cluster_cards,
    write_cluster_cards,
)
from face_stratification.stage_c.pipeline import (
    StageCResult,
    run_stage_c,
)
from face_stratification.stage_c.clinical_panels import (
    EMBEDDING_INPUT_FEATURES,
    MIN_PANEL_POSITIVES,
    ClinicalFeaturePanel,
    ClinicalFeaturePanelValidationResult,
    default_clinical_feature_whitelist,
    discover_all_clinical_feature_panels,
    discover_clinical_feature_panel,
    validate_all_clinical_feature_panels_cv,
    validate_clinical_feature_panel_cv,
)
# Legacy aliases (deprecated; kept so older notebooks keep importing).
from face_stratification.stage_c.biomarkers import (  # noqa: F401
    BiomarkerPanel,
    BiomarkerValidationResult,
    default_biomarker_whitelist,
    discover_all_biomarker_panels,
    discover_biomarker_panel,
    validate_all_biomarker_panels_cv,
    validate_biomarker_panel_cv,
)

__all__ = [
    # algorithms
    "ALGORITHMS",
    "run_algorithm",
    "run_gmm",
    "run_kmeans",
    "run_spectral",
    "run_ward",
    # consensus
    "ConsensusResult",
    "align_labels_to_reference",
    "build_coassociation_matrix",
    "compute_per_patient_confidence",
    "consensus_partition",
    "run_consensus_clustering",
    # comparison
    "FullDSMComparison",
    "chi_square_independence",
    "cramers_v",
    "full_dsm_comparison",
    "per_cluster_cohort_entropy",
    "per_cohort_purity",
    # ablation
    "compute_optimization_score",
    "pick_best_configuration",
    "run_algorithm_k_grid",
    "run_embedding_view_ablation",
    # narrative
    "ClusterCard",
    "build_cluster_cards",
    "write_cluster_cards",
    # pipeline
    "StageCResult",
    "run_stage_c",
    # clinical-feature panels (canonical names)
    "ClinicalFeaturePanel",
    "ClinicalFeaturePanelValidationResult",
    "EMBEDDING_INPUT_FEATURES",
    "MIN_PANEL_POSITIVES",
    "default_clinical_feature_whitelist",
    "discover_clinical_feature_panel",
    "discover_all_clinical_feature_panels",
    "validate_clinical_feature_panel_cv",
    "validate_all_clinical_feature_panels_cv",
    # Legacy aliases (deprecated)
    "BiomarkerPanel",
    "BiomarkerValidationResult",
    "default_biomarker_whitelist",
    "discover_all_biomarker_panels",
    "discover_biomarker_panel",
    "validate_all_biomarker_panels_cv",
    "validate_biomarker_panel_cv",
]
