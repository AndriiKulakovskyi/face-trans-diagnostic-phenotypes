"""Generate the engine's ``FeatureSchema`` from our common-variables dictionary.

The vendored ``face_stratification`` engine describes its feature matrix with a
strict pydantic :class:`FeatureSchema` (blocks + features). To drive that engine
from OUR pipeline we translate the common-variables dictionary into the same
vocabulary:

    dictionary section   → ``FeatureBlock``  (id = slug of the section)
    dictionary row       → ``UnifiedFeature`` (id = canonical_name)
    ``dtype``            → ``FeatureType``   (binary / ordinal / categorical / continuous)
    per-cohort source col → ``cohorts``       (which of bp/sz/dr can provide it)

Only the math-irrelevant parts of the schema are inferred with defaults:
``temporal_scope`` is fixed to ``current`` (we always feed a single visit, V0)
and ``direction`` to ``none`` (we have not curated clinical sign conventions).
Neither field is read by the embedding/clustering code — they exist so the
engine's optional block-level and clinical-panel analyses can run later.

See ``face_stratification.harmonization.feature_schema`` for the target model.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

from face_stratification.harmonization.feature_schema import FeatureSchema

from .variable import Variable

__all__ = ["build_feature_schema", "feature_cohorts", "DEFAULT_SCHEMA_VERSION"]

DEFAULT_SCHEMA_VERSION = "face_common-0.1.0"

# canonical cohort order used throughout the engine
_COHORT_COLUMNS = (("bp", "bp_csv_col"), ("sz", "sz_csv_col"), ("dr", "dr_csv_col"))


def _slug(text: str) -> str:
    """ASCII snake_case slug for a (possibly accented, spaced) section label."""
    ascii_text = (
        unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    )
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", ascii_text).strip("_").lower()
    return slug or "misc"


def _feature_type(dtype: str) -> str:
    """Map a dictionary ``dtype`` string to a :class:`FeatureType` value.

    The dictionary uses labels like ``"int8 binary"``, ``"int8 ordinal"``,
    ``"int8 categorical"``, ``"float"``, ``"category"``, ``"string"``. Non-numeric
    types fall back to ``categorical`` (they never reach the cosine matrix, but
    the schema still needs a valid type).
    """
    d = dtype.lower()
    if "binary" in d:
        return "binary"
    if "ordinal" in d:
        return "ordinal"
    if "categor" in d or d == "category":
        return "categorical"
    if "float" in d or "continuous" in d or "double" in d or "numeric" in d:
        return "continuous"
    return "categorical"


def feature_cohorts(variable: Variable) -> tuple[str, ...]:
    """Cohort codes (lowercase) that declare a source column for this variable."""
    return tuple(
        code
        for code, attr in _COHORT_COLUMNS
        if getattr(variable, attr) is not None
    )


def build_feature_schema(
    variables: Iterable[Variable],
    feature_ids: Sequence[str],
    *,
    version: str = DEFAULT_SCHEMA_VERSION,
) -> FeatureSchema:
    """Build a validated :class:`FeatureSchema` for ``feature_ids``.

    Parameters
    ----------
    variables:
        The full dictionary (``load_variables(...)``). Only rows whose
        ``canonical_name`` appears in ``feature_ids`` and that declare at least
        one cohort source column are emitted as features.
    feature_ids:
        The exact columns present in the harmonized matrix (so schema features
        line up 1:1 with matrix columns). Order is preserved.
    version:
        Schema version string recorded on the embedding artifacts.
    """
    by_name = {v.canonical_name: v for v in variables}

    blocks: dict[str, dict[str, str]] = {}
    features: list[dict[str, object]] = []
    seen: set[str] = set()

    for fid in feature_ids:
        if fid in seen:
            continue
        var = by_name.get(fid)
        if var is None:
            continue
        cohorts = feature_cohorts(var)
        if not cohorts:
            continue  # a feature with no source column cannot enter the schema
        seen.add(fid)

        block_id = _slug(var.section)
        if block_id not in blocks:
            section_label = var.section or block_id
            blocks[block_id] = {
                "id": block_id,
                "label_fr": section_label,
                "description": f"FACE common-variables section: {section_label}",
            }

        features.append(
            {
                "id": fid,
                "label_fr": var.label or fid,
                "block": block_id,
                "type": _feature_type(var.dtype),
                "temporal_scope": "current",
                "unit": None,
                "direction": "none",
                "cohorts": cohorts,
                "description": var.clinical_rationale or None,
            }
        )

    if not features:
        raise ValueError("No dictionary-backed features to build a schema from.")

    return FeatureSchema.model_validate(
        {
            "version": version,
            "blocks": list(blocks.values()),
            "features": features,
        }
    )
