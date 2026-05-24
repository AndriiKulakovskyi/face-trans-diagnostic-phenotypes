"""Canonical names for the six trans-diagnostic dimensional axes — single source of truth.

The dimensional model (``07_dimensional_refine.py``) writes its axes as generic ``axis1``…``axis6``
in ``dimensional_final_{scores.parquet,loadings.csv}``, ordered by descending sum-of-squares of
the loadings. This module fixes the *interpretation* of that order so downstream scripts and the
manuscript don't each re-hardcode the names (which drift — e.g. the imputation-free re-derivation
renamed axis 6 from "adhd_impulsivity_trauma" to "work_disability", and stale copies lingered in
several scripts).

  ``axis1`` → depression_severity     (Depression / internalizing)
  ``axis2`` → later_onset             (Later onset)
  ``axis3`` → mania_activation        (Mania / activation; incl. impulsivity)
  ``axis4`` → illness_burden          (Illness / hospitalization burden)
  ``axis5`` → metabolic               (Metabolic / inflammatory)
  ``axis6`` → work_disability         (Socio-occupational / work-disability)

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
    "metabolic",
    "work_disability",
]

# Short display labels (compact figure axes, tables).
AXIS_SHORT: dict[str, str] = {
    "depression_severity": "Depression",
    "later_onset": "Later onset",
    "mania_activation": "Mania",
    "illness_burden": "Illness burden",
    "metabolic": "Metabolic",
    "work_disability": "Work-disability",
}

# Full descriptive labels (figure titles, captions).
AXIS_LABELS: dict[str, str] = {
    "depression_severity": "Depression / internalizing",
    "later_onset": "Later onset",
    "mania_activation": "Mania / activation",
    "illness_burden": "Illness / hospitalization burden",
    "metabolic": "Metabolic / inflammatory",
    "work_disability": "Socio-occupational / work-disability",
}

# Map the generic axis index (as written by 07: "axis1".."axis6") to the canonical name.
AXIS_INDEX_TO_NAME: dict[str, str] = {f"axis{i + 1}": n for i, n in enumerate(AXIS_NAMES)}

__all__ = ["AXIS_NAMES", "AXIS_SHORT", "AXIS_LABELS", "AXIS_INDEX_TO_NAME"]
