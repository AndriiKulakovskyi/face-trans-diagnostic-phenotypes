"""V1 feature harmonization across FACE cohorts.

This package turns cohort-specific ``PatientProfile`` objects (from
``face_rlvr.profiles``) into a single unified cross-sectional feature matrix
suitable for downstream graph construction and representation learning.

All feature definitions live in ``config/face_stratification/feature_schema.yaml``
and are validated by Pydantic models in :mod:`feature_schema`. Python code in
this package contains only logic — there are no hardcoded feature names or
clinical thresholds.
"""

from face_stratification.harmonization.feature_schema import (
    FeatureBlock,
    FeatureSchema,
    FeatureType,
    TemporalScope,
    UnifiedFeature,
    load_feature_schema,
)
from face_stratification.harmonization.cohort_adapters import (
    adapt_asp_profile,
    adapt_bp_profile,
    adapt_dr_profile,
    adapt_sz_profile,
)
from face_stratification.harmonization.harmonizer import (
    HarmonizedDataset,
    build_harmonized_dataset,
)
from face_stratification.harmonization.normalization import (
    NormalizationStats,
    fit_normalization,
    transform_normalization,
)
from face_stratification.harmonization.missingness import (
    augment_with_missingness_indicators,
    characterize_missingness,
    compute_missingness_mask,
    impute_block_knn,
    impute_block_mice,
)
from face_stratification.harmonization.dsm_subtypes import (
    extract_dsm_subtypes,
)

__all__ = [
    "FeatureBlock",
    "FeatureSchema",
    "FeatureType",
    "TemporalScope",
    "UnifiedFeature",
    "load_feature_schema",
    "adapt_asp_profile",
    "adapt_bp_profile",
    "adapt_dr_profile",
    "adapt_sz_profile",
    "HarmonizedDataset",
    "build_harmonized_dataset",
    "NormalizationStats",
    "fit_normalization",
    "transform_normalization",
    "augment_with_missingness_indicators",
    "characterize_missingness",
    "compute_missingness_mask",
    "impute_block_knn",
    "impute_block_mice",
    "extract_dsm_subtypes",
]
