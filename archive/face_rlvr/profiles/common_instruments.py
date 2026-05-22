"""Shared clinical instrument infrastructure and definitions.

This module contains:
- Core data structures (SeverityLevel, InstrumentDefinition, ScoreInterpretation)
- The interpret_score function used by all cohorts
- Shared instrument definitions used by 2+ cohorts
- Shared severity threshold lists for instruments with cohort-specific columns

All cohort-specific instrument files import from here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ─── Core Data Structures ────────────────────────────────────────────────────


@dataclass
class SeverityLevel:
    """A single severity band for a clinical instrument."""

    min_score: float
    max_score: float
    code: str  # "normal", "mild", "moderate", "severe"
    label_fr: str  # Short French label
    clinical_meaning_fr: str  # Full sentence clinical interpretation


@dataclass
class InstrumentDefinition:
    """Complete definition of a clinical assessment instrument."""

    name: str  # Abbreviation, e.g. "MADRS"
    full_name: str  # Full name
    full_name_fr: str  # Full name in French
    domain: str  # Clinical domain
    total_column: str  # CSV column for the total score
    subscale_columns: dict[str, str] = field(default_factory=dict)
    score_range: tuple[float, float] = (0, 100)
    higher_is_worse: bool = True  # True for most psychiatric scales
    severity_thresholds: list[SeverityLevel] = field(default_factory=list)
    screening_threshold: float | None = None  # For screening instruments
    screening_positive_label_fr: str = "Dépistage positif"
    screening_negative_label_fr: str = "Dépistage négatif"
    evaluation_type: str = "hetero"  # "hetero" (clinician-rated) or "auto" (self-report)
    unit: str = ""  # Display unit
    clinical_note_fr: str = ""


@dataclass
class ScoreInterpretation:
    """Interpreted score for a single instrument."""

    instrument: str
    raw_score: float | None
    severity_code: str  # "normal", "mild", etc., or "positive"/"negative"
    severity_label_fr: str
    clinical_interpretation_fr: str
    subscales: dict[str, float | None] = field(default_factory=dict)
    is_screening: bool = False
    score_available: bool = True
    suspect_value: bool = False  # True if value is outside valid range or likely erroneous


# ─── Helper functions ─────────────────────────────────────────────────────────


def _safe_float(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _fmt_score(val: float) -> str:
    """Format a score: integer if whole number, one decimal otherwise."""
    if val == int(val):
        return str(int(val))
    return f"{val:.1f}"


# ─── Interpretation function ─────────────────────────────────────────────────


def interpret_score(
    instrument: InstrumentDefinition,
    raw_score: float | None,
    subscale_values: dict[str, float | None] | None = None,
) -> ScoreInterpretation:
    """Interpret a raw score using the instrument's severity thresholds."""
    if raw_score is None:
        return ScoreInterpretation(
            instrument=instrument.name,
            raw_score=None,
            severity_code="missing",
            severity_label_fr="Non disponible",
            clinical_interpretation_fr=f"Score {instrument.name} non disponible.",
            subscales=subscale_values or {},
            score_available=False,
        )

    score_str = _fmt_score(raw_score)

    # Detect suspect values (outside valid instrument range or known bad patterns)
    lo, hi = instrument.score_range
    suspect = False
    if raw_score < lo or raw_score > hi:
        suspect = True
    # Maximum possible score is clinically unusual — possible acquiescence bias or data error
    if raw_score == hi:
        suspect = True
    # EQ-5D exactly 0.0 is likely missing data, not actual score
    if instrument.name == "EQ-5D" and raw_score == 0.0:
        suspect = True

    suspect_suffix = " [valeur suspecte]" if suspect else ""

    # Screening instruments: binary positive/negative
    if instrument.screening_threshold is not None:
        if instrument.higher_is_worse:
            is_positive = raw_score >= instrument.screening_threshold
        else:
            is_positive = raw_score <= instrument.screening_threshold

        code = "positive" if is_positive else "negative"
        label = instrument.screening_positive_label_fr if is_positive else instrument.screening_negative_label_fr
        interp = (
            f"{instrument.name} = {score_str} "
            f"(seuil {_fmt_score(instrument.screening_threshold)}) : {label.lower()}.{suspect_suffix}"
        )
        return ScoreInterpretation(
            instrument=instrument.name,
            raw_score=raw_score,
            severity_code=code,
            severity_label_fr=label,
            clinical_interpretation_fr=interp,
            subscales=subscale_values or {},
            is_screening=True,
            suspect_value=suspect,
        )

    # Severity-based instruments: find matching band
    for level in instrument.severity_thresholds:
        if level.min_score <= raw_score <= level.max_score:
            interp = (
                f"{instrument.name} = {score_str}"
                f"{(' ' + instrument.unit) if instrument.unit else ''} "
                f"({level.label_fr}). {level.clinical_meaning_fr}{suspect_suffix}"
            )
            return ScoreInterpretation(
                instrument=instrument.name,
                raw_score=raw_score,
                severity_code=level.code,
                severity_label_fr=level.label_fr,
                clinical_interpretation_fr=interp,
                subscales=subscale_values or {},
                suspect_value=suspect,
            )

    # Score outside all defined bands
    return ScoreInterpretation(
        instrument=instrument.name,
        raw_score=raw_score,
        severity_code="unclassified",
        severity_label_fr="Non classifiable",
        clinical_interpretation_fr=(
            f"{instrument.name} = {score_str} : valeur hors des seuils de référence habituels.{suspect_suffix}"
        ),
        subscales=subscale_values or {},
        suspect_value=True,
    )


# Shared severity threshold bands and instrument definitions live entirely in
# ``config/glossary/common/thresholds.yaml`` and ``config/glossary/common/instruments.yaml``.
# Access them via the glossary loader:
#     from face_rlvr.profiles.glossary_loader import load_common_glossary, _load_common_thresholds
#     load_common_glossary()["instruments"]["MADRS"]
#     _load_common_thresholds().bands["PSQI_THRESHOLDS"]
