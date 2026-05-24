from .axes import AXIS_INDEX_TO_NAME, AXIS_LABELS, AXIS_NAMES, AXIS_SHORT
from .filters import (
    IDENTIFIER_COLUMNS,
    PatientFilterReport,
    V0Anchor,
    VariableFilterReport,
    filter_patients,
    filter_variables,
    select_v0_anchor,
)
from .loader import YEARLY_VISIT_MAP, build_unified_dataframe
from .rules import RULES, identity_cast, register
from .variable import Variable, load_variables

# Engine bridge (imports face_stratification; available whenever the vendored
# engine is on the path — always true via pyproject pythonpath / scripts' setup).
try:  # pragma: no cover - exercised in integration, not unit tests
    from .adapter import (
        ADMINISTRATIVE_FEATURES,
        CLINICAL_SECTIONS,
        COHORT_TO_CODE,
        normalize_for_embedding,
        residualize_features,
        to_harmonized_dataset,
    )
    from .domains import (
        BIOLOGY_COMPOSITES,
        DOMAIN_SECTIONS,
        build_domain_scores,
    )
    from .schema_gen import DEFAULT_SCHEMA_VERSION, build_feature_schema
except ImportError:  # engine not importable in this environment
    COHORT_TO_CODE = None  # type: ignore[assignment]
    ADMINISTRATIVE_FEATURES = None  # type: ignore[assignment]
    CLINICAL_SECTIONS = None  # type: ignore[assignment]
    normalize_for_embedding = None  # type: ignore[assignment]
    residualize_features = None  # type: ignore[assignment]
    to_harmonized_dataset = None  # type: ignore[assignment]
    build_domain_scores = None  # type: ignore[assignment]
    BIOLOGY_COMPOSITES = None  # type: ignore[assignment]
    DOMAIN_SECTIONS = None  # type: ignore[assignment]
    build_feature_schema = None  # type: ignore[assignment]
    DEFAULT_SCHEMA_VERSION = None  # type: ignore[assignment]

__all__ = [
    "Variable",
    "load_variables",
    "RULES",
    "register",
    "identity_cast",
    "build_unified_dataframe",
    "YEARLY_VISIT_MAP",
    "IDENTIFIER_COLUMNS",
    "VariableFilterReport",
    "PatientFilterReport",
    "V0Anchor",
    "filter_variables",
    "filter_patients",
    "select_v0_anchor",
    "to_harmonized_dataset",
    "normalize_for_embedding",
    "residualize_features",
    "build_domain_scores",
    "build_feature_schema",
    "COHORT_TO_CODE",
    "ADMINISTRATIVE_FEATURES",
    "CLINICAL_SECTIONS",
    "BIOLOGY_COMPOSITES",
    "DOMAIN_SECTIONS",
    "DEFAULT_SCHEMA_VERSION",
    "AXIS_NAMES",
    "AXIS_SHORT",
    "AXIS_LABELS",
    "AXIS_INDEX_TO_NAME",
]
