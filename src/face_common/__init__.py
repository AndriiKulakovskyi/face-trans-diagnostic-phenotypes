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
    from .adapter import COHORT_TO_CODE, to_harmonized_dataset
    from .schema_gen import DEFAULT_SCHEMA_VERSION, build_feature_schema
except ImportError:  # engine not importable in this environment
    COHORT_TO_CODE = None  # type: ignore[assignment]
    to_harmonized_dataset = None  # type: ignore[assignment]
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
    "build_feature_schema",
    "COHORT_TO_CODE",
    "DEFAULT_SCHEMA_VERSION",
]
