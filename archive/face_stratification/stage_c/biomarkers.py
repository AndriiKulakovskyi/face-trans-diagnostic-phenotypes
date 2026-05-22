"""Backwards-compatibility shim for the renamed clinical-feature panel module.

The module was renamed from ``biomarkers`` to :mod:`clinical_panels` because
the panels it discovers are clinically-actionable **phenotypic descriptors**,
not biological biomarkers in the FDA–NIH BEST sense. The old public names
(``BiomarkerPanel``, ``discover_biomarker_panel``, …) are re-exported here as
thin aliases so existing notebooks, scripts and tests continue to work while
we migrate call sites and documentation.

New code should import from :mod:`face_stratification.stage_c.clinical_panels`.
"""

from __future__ import annotations

import warnings

from face_stratification.stage_c.clinical_panels import (
    EMBEDDING_INPUT_FEATURES,
    MIN_PANEL_POSITIVES,
    ClinicalFeaturePanel,
    ClinicalFeaturePanelValidationResult,
    _CLINICAL_FEATURE_WHITELIST_BY_TYPE,
    default_clinical_feature_whitelist,
    discover_all_clinical_feature_panels,
    discover_clinical_feature_panel,
    validate_all_clinical_feature_panels_cv,
    validate_clinical_feature_panel_cv,
)

# ─── Legacy aliases ────────────────────────────────────────────────────────
#
# These names are kept verbatim so callers that imported
# ``BiomarkerPanel``/``discover_biomarker_panel``/etc. keep working. The
# *behaviour* is identical to the new API — in particular, the default
# whitelist still excludes the eight embedding-input features, which is the
# leakage-safe behaviour you should prefer.

BiomarkerPanel = ClinicalFeaturePanel
BiomarkerValidationResult = ClinicalFeaturePanelValidationResult
_BIOMARKER_WHITELIST_BY_TYPE = _CLINICAL_FEATURE_WHITELIST_BY_TYPE


def default_biomarker_whitelist(
    *, exclude_embedding_inputs: bool = True
) -> list[str]:
    """Deprecated alias for :func:`default_clinical_feature_whitelist`."""
    warnings.warn(
        "default_biomarker_whitelist is deprecated; use "
        "default_clinical_feature_whitelist from "
        "face_stratification.stage_c.clinical_panels.",
        DeprecationWarning,
        stacklevel=2,
    )
    return default_clinical_feature_whitelist(
        exclude_embedding_inputs=exclude_embedding_inputs
    )


def discover_biomarker_panel(*args, **kwargs):
    """Deprecated alias for :func:`discover_clinical_feature_panel`."""
    warnings.warn(
        "discover_biomarker_panel is deprecated; use "
        "discover_clinical_feature_panel from "
        "face_stratification.stage_c.clinical_panels.",
        DeprecationWarning,
        stacklevel=2,
    )
    return discover_clinical_feature_panel(*args, **kwargs)


def discover_all_biomarker_panels(*args, **kwargs):
    """Deprecated alias for :func:`discover_all_clinical_feature_panels`."""
    warnings.warn(
        "discover_all_biomarker_panels is deprecated; use "
        "discover_all_clinical_feature_panels from "
        "face_stratification.stage_c.clinical_panels.",
        DeprecationWarning,
        stacklevel=2,
    )
    return discover_all_clinical_feature_panels(*args, **kwargs)


def validate_biomarker_panel_cv(*args, **kwargs):
    """Deprecated alias for :func:`validate_clinical_feature_panel_cv`."""
    warnings.warn(
        "validate_biomarker_panel_cv is deprecated; use "
        "validate_clinical_feature_panel_cv from "
        "face_stratification.stage_c.clinical_panels.",
        DeprecationWarning,
        stacklevel=2,
    )
    return validate_clinical_feature_panel_cv(*args, **kwargs)


def validate_all_biomarker_panels_cv(*args, **kwargs):
    """Deprecated alias for :func:`validate_all_clinical_feature_panels_cv`."""
    warnings.warn(
        "validate_all_biomarker_panels_cv is deprecated; use "
        "validate_all_clinical_feature_panels_cv from "
        "face_stratification.stage_c.clinical_panels.",
        DeprecationWarning,
        stacklevel=2,
    )
    return validate_all_clinical_feature_panels_cv(*args, **kwargs)


__all__ = [
    # New canonical names
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
    "discover_biomarker_panel",
    "discover_all_biomarker_panels",
    "validate_biomarker_panel_cv",
    "validate_all_biomarker_panels_cv",
]
