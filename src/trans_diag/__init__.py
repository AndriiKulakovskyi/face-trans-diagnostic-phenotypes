from .axes import AXIS_INDEX_TO_NAME, AXIS_LABELS, AXIS_NAMES, AXIS_SHORT, ORTHOGONAL_DIMENSIONS
from .filters import (
    IDENTIFIER_COLUMNS,
    V0Anchor,
    VariableFilterReport,
    filter_patients,
    filter_variables,
    select_v0_anchor,
)
from .loader import YEARLY_VISIT_MAP, build_unified_dataframe
from .rules import RULES, identity_cast, register
from .skip_logic import SUICIDE_SKIP_RULES, SkipRule, decode_skip_logic
from .variable import Variable, load_variables

# Engine bridge: the adapter + domains pull in the internalized engine
# (src/trans_diag/engine/). Guarded so the core (loader/rules/variable) still
# imports even if an optional engine dependency is unavailable.
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
        COGNITIVE_COMPOSITES,
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
    COGNITIVE_COMPOSITES = None  # type: ignore[assignment]
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
    "decode_skip_logic",
    "SkipRule",
    "SUICIDE_SKIP_RULES",
    "YEARLY_VISIT_MAP",
    "IDENTIFIER_COLUMNS",
    "VariableFilterReport",
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
    "COGNITIVE_COMPOSITES",
    "DOMAIN_SECTIONS",
    "DEFAULT_SCHEMA_VERSION",
    "AXIS_NAMES",
    "AXIS_SHORT",
    "AXIS_LABELS",
    "AXIS_INDEX_TO_NAME",
    "ORTHOGONAL_DIMENSIONS",
]
