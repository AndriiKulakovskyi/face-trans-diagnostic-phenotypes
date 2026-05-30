"""Canonical names for the six trans-diagnostic dimensional axes — single source of truth.

The dimensional model (``07_dimensional_refine.py``) writes its axes as generic ``axis1``…``axis6``
in ``dimensional_final_{scores.parquet,loadings.csv}``, ordered by descending sum-of-squares of
the loadings. This module fixes the *interpretation* of that order so downstream scripts and the
manuscript don't each re-hardcode the names (which drift).

History: with neuropsychology restricted to BP/SZ the model locked at seven *symptom* axes
(depression, later-onset, illness-burden, a pure mania axis, a separate externalizing axis,
metabolic, work-disability). Closing the DR neuropsych extraction gap (2026-05) and admitting
cognition into the model re-derived the structure to K=6: ONE reproducible, confound-clean
cognitive axis emerges (verbal/working-memory ability), the mania and externalizing axes re-merge
into one mania/activation axis, and work-disability is no longer separately resolved. Excluded
cognitive constructs (each for a documented reason — see ``domains.COGNITIVE_COMPOSITES`` and
MANUSCRIPT §2.12): processing speed & executive/TMT (incoherent across cohorts), verbal fluency
(its axis was a cohort artifact), CVLT memory & matrix reasoning (BP/SZ-only).

  ``axis1`` → depression_severity     (Depression / internalizing)
  ``axis2`` → later_onset             (Later onset)
  ``axis3`` → mania_activation        (Mania / activation, with externalizing: impulsivity, childhood ADHD)
  ``axis4`` → illness_burden          (Illness / hospitalization burden)
  ``axis5`` → cognition_verbal        (Verbal / crystallized cognition: verbal reasoning, working memory)
  ``axis6`` → metabolic               (Metabolic / inflammatory)

Import from the package: ``from trans_diag import AXIS_NAMES, AXIS_SHORT, AXIS_LABELS``.
If 07's axis ordering is ever re-locked, update only this file.
"""
from __future__ import annotations

# snake_case names in axis1..axis6 (sum-of-squares) order — the order 07 writes.
AXIS_NAMES: list[str] = [
    "depression_severity",
    "later_onset",
    "mania_activation",
    "illness_burden",
    "cognition_verbal",
    "metabolic",
]

# Short display labels (compact figure axes, tables).
AXIS_SHORT: dict[str, str] = {
    "depression_severity": "Depression",
    "later_onset": "Later onset",
    "mania_activation": "Mania / externalizing",
    "illness_burden": "Illness burden",
    "cognition_verbal": "Verbal cognition",
    "metabolic": "Metabolic",
}

# Full descriptive labels (figure titles, captions).
AXIS_LABELS: dict[str, str] = {
    "depression_severity": "Depression / internalizing",
    "later_onset": "Later onset",
    "mania_activation": "Mania / activation (with externalizing: impulsivity, childhood ADHD)",
    "illness_burden": "Illness / hospitalization burden",
    "cognition_verbal": "Verbal / crystallized cognition (verbal reasoning, working memory)",
    "metabolic": "Metabolic / inflammatory",
}

# Map the generic axis index (as written by 07: "axis1".."axis6") to the canonical name.
AXIS_INDEX_TO_NAME: dict[str, str] = {f"axis{i + 1}": n for i, n in enumerate(AXIS_NAMES)}

__all__ = ["AXIS_NAMES", "AXIS_SHORT", "AXIS_LABELS", "AXIS_INDEX_TO_NAME"]
