"""Pydantic v2 schema for the V3 data layer's feature matrix.

``v3.data.schema_gen.build_feature_schema`` builds a ``FeatureSchema`` directly from the
common-variables dictionary (validated with ``extra="forbid"`` everywhere — any typo fails loudly
rather than silently producing a bad matrix). The engine consumes that schema to describe its blocks
and features.

Invariants enforced here:

- ``UnifiedFeature.temporal_scope`` cannot be ``trajectory``. A feature may be
  ``current`` (measured at the baseline visit V0), ``lifetime`` (lifetime history
  up to and including the baseline visit), or ``static`` (time-invariant
  demographics / developmental / family history). Anything longitudinal is
  rejected.
- Every feature declares the cohorts that can provide it. Unknown cohorts are
  rejected.
- Feature ids must be unique across the whole schema.
- Every feature belongs to exactly one ``FeatureBlock`` that is itself declared
  in the schema.

The schema is intentionally flat: features are grouped logically by ``block``
but stored in a single list, so the harmonizer can emit one column per feature
without extra indirection.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ─── Enums ────────────────────────────────────────────────────────────────────


class FeatureType(StrEnum):
    """Statistical type of a unified feature."""

    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    BINARY = "binary"
    CATEGORICAL = "categorical"


class TemporalScope(StrEnum):
    """Temporal interpretation of a feature value relative to the baseline visit.

    ``trajectory`` is *intentionally absent* — this sub-project is cross-sectional
    by design and the loader rejects any attempt to introduce it.
    """

    CURRENT = "current"  # assessed at baseline visit
    LIFETIME = "lifetime"  # lifetime history up to baseline
    STATIC = "static"  # time-invariant (sex, developmental, family)


# Canonical cohort codes.
_VALID_COHORTS: frozenset[str] = frozenset({"bp", "sz", "dr", "asp"})


# ─── Pydantic models ─────────────────────────────────────────────────────────


class FeatureBlock(BaseModel):
    """A logical grouping of related features used for graph edge construction.

    The ``min_shared_features`` and ``min_fraction_present`` fields control the
    semantic overlap edge constraint (used by the engine's
    patient-similarity graph builder):

    - ``min_fraction_present``: a patient must have at least this fraction of
      the block's features measured to be considered as a candidate node in
      this block's graph at all.
    - ``min_shared_features``: an edge between two candidate patients is only
      created if they share at least this many observed features in the block.
      When ``None``, defaults to ``max(2, ⌈|block| / 2⌉)`` at graph-build time.

    Both are enforced by the graph builder; none of them trigger imputation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(..., description="Short machine id, e.g. 'cognition'")
    label_fr: str = Field(..., description="Human-readable French label")
    description: str = Field(..., description="What this block captures clinically")
    metric: Literal["cosine", "euclidean", "gower", "manhattan"] = Field(
        "cosine",
        description="Default similarity metric for intra-block kNN graphs",
    )
    min_fraction_present: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fraction of block features a patient must have observed "
            "to participate in this block's graph."
        ),
    )
    min_shared_features: int | None = Field(
        None,
        ge=1,
        description=(
            "Minimum shared observed features required for a pair of patients "
            "to receive an edge in this block. None = auto (half the block)."
        ),
    )


class TransdiagnosticSelectionConfig(BaseModel):
    """Declarative config for the data-driven transdiagnostic feature set.

    A feature is admitted into the transdiagnostic graph only if its
    observed-value coverage is at least ``min_cohort_coverage`` in **every**
    cohort of the harmonized dataset. The transdiagnostic graph itself then
    enforces a semantic overlap constraint proportional to the size of the
    selected set (see ``min_shared_features_fraction``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_cohort_coverage: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum per-cohort observed coverage for a feature to be admitted "
            "to the transdiagnostic feature set."
        ),
    )
    min_shared_features_fraction: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the selected transdiagnostic set that two patients "
            "must share to receive a transdiagnostic-graph edge."
        ),
    )
    metric: Literal["cosine", "euclidean", "gower", "manhattan"] = Field(
        "cosine",
        description="Similarity metric for the transdiagnostic graph",
    )
    excluded_feature_ids: tuple[str, ...] = Field(
        (),
        description=(
            "Optional hard-exclude list: features that should never enter the "
            "transdiagnostic set even if their coverage crosses the threshold."
        ),
    )


class UnifiedFeature(BaseModel):
    """One column of the unified feature matrix.

    Features are per-patient (single value) and strictly cross-sectional:
    they describe the baseline visit or a static / lifetime attribute. Any
    feature whose value would require two timepoints (delta, RCI, follow-up)
    is rejected at load time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(..., description="Unique machine id, e.g. 'cog_tmt_b_seconds'")
    label_fr: str = Field(..., description="Human-readable French label")
    block: str = Field(..., description="Id of the FeatureBlock this feature belongs to")
    type: FeatureType = Field(..., description="Statistical type")
    temporal_scope: TemporalScope = Field(
        ..., description="Must be current / lifetime / static (never trajectory)"
    )
    unit: str | None = Field(None, description="Unit of measurement, if meaningful")
    direction: Literal["higher_is_worse", "higher_is_better", "none"] = Field(
        "none",
        description="Clinical direction (for normalization sign-flipping)",
    )
    cohorts: tuple[str, ...] = Field(
        ...,
        description="Cohort codes that can provide this feature",
        min_length=1,
    )
    allowed_values: tuple[str, ...] | None = Field(
        None,
        description="For categorical/ordinal features: canonical values in order",
    )
    description: str | None = Field(None, description="Clinical description")
    rdoc_domain: Literal[
        "negative_valence",
        "positive_valence",
        "cognitive_systems",
        "social_processes",
        "arousal_regulatory",
        "sensorimotor",
    ] | None = Field(None, description="RDoC/HiTOP domain mapping")

    @field_validator("id")
    @classmethod
    def _id_snake_case(cls, v: str) -> str:
        if not v.islower() or " " in v:
            raise ValueError(f"Feature id must be snake_case lowercase: {v!r}")
        return v

    @field_validator("cohorts")
    @classmethod
    def _cohorts_valid(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        bad = [c for c in v if c not in _VALID_COHORTS]
        if bad:
            raise ValueError(
                f"Unknown cohort code(s) {bad}. Valid cohorts: {sorted(_VALID_COHORTS)}"
            )
        return tuple(dict.fromkeys(v))  # dedupe while preserving order

    @field_validator("id")
    @classmethod
    def _reject_longitudinal_id(cls, v: str) -> str:
        """Refuse feature ids that look longitudinal.

        This is the YAML-level guardrail for the V1-only invariant. Combined
        with the ``temporal_scope`` enum it prevents longitudinal features from
        ever entering the matrix.
        """
        forbidden_suffixes = (
            "_n1",
            "_followup",
            "_delta",
            "_rci",
            "_change",
            "_v2",
            "_visit2",
        )
        for suffix in forbidden_suffixes:
            if v.endswith(suffix):
                raise ValueError(
                    f"Feature id {v!r} looks longitudinal (suffix {suffix!r}); "
                    "this sub-project is V1-only."
                )
        return v


class FeatureSchema(BaseModel):
    """Top-level schema: a list of blocks + a list of features.

    Features reference blocks by id. The model validator below enforces
    global invariants (unique ids, block references resolve, etc.).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(..., description="Semantic version of the schema")
    blocks: tuple[FeatureBlock, ...] = Field(..., min_length=1)
    features: tuple[UnifiedFeature, ...] = Field(..., min_length=1)
    transdiagnostic_selection: TransdiagnosticSelectionConfig = Field(
        default_factory=TransdiagnosticSelectionConfig,
        description="Data-driven transdiagnostic feature selection config",
    )

    @model_validator(mode="after")
    def _validate_cross_references(self) -> FeatureSchema:
        # 1. Unique block ids
        block_ids = [b.id for b in self.blocks]
        if len(block_ids) != len(set(block_ids)):
            dupes = {b for b in block_ids if block_ids.count(b) > 1}
            raise ValueError(f"Duplicate block ids: {sorted(dupes)}")

        # 2. Unique feature ids
        feat_ids = [f.id for f in self.features]
        if len(feat_ids) != len(set(feat_ids)):
            dupes = {f for f in feat_ids if feat_ids.count(f) > 1}
            raise ValueError(f"Duplicate feature ids: {sorted(dupes)}")

        # 3. Every feature's block exists
        known_blocks = set(block_ids)
        for f in self.features:
            if f.block not in known_blocks:
                raise ValueError(
                    f"Feature {f.id!r} references unknown block {f.block!r}"
                )

        return self

    # ─── Query helpers ────────────────────────────────────────────────────────

    def feature_ids(self) -> tuple[str, ...]:
        return tuple(f.id for f in self.features)

    def block_ids(self) -> tuple[str, ...]:
        return tuple(b.id for b in self.blocks)

    def features_by_id(self) -> dict[str, UnifiedFeature]:
        return {f.id: f for f in self.features}

    def features_by_block(self) -> dict[str, tuple[UnifiedFeature, ...]]:
        out: dict[str, list[UnifiedFeature]] = {b.id: [] for b in self.blocks}
        for f in self.features:
            out[f.block].append(f)
        return {k: tuple(v) for k, v in out.items()}

    def features_for_cohort(self, cohort: str) -> tuple[UnifiedFeature, ...]:
        c = cohort.lower()
        return tuple(f for f in self.features if c in f.cohorts)

    def block(self, block_id: str) -> FeatureBlock:
        for b in self.blocks:
            if b.id == block_id:
                return b
        raise KeyError(f"Unknown block id: {block_id!r}")
