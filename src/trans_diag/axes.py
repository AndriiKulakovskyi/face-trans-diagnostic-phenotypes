"""Canonical names for the seven trans-diagnostic dimensional axes — single source of truth.

The dimensional model (``07_dimensional_refine.py``) writes its axes as generic ``axis1``…``axis7``
in ``dimensional_final_{scores.parquet,loadings.csv}``, ordered by descending sum-of-squares of
the loadings. This module fixes the *interpretation* of that order so downstream scripts and the
manuscript don't each re-hardcode the names (which drift).

History: the imputation-free re-derivation first renamed axis 6 from "adhd_impulsivity_trauma" to
"work_disability" (the mean-fill ADHD/trauma axis was a co-observation artifact; §3.8). Locking
the model at K=7 (the maximum reproducible dimensionality; split-half min >=0.85 through K=7,
collapse at K>=8) then *split* the K=6 mania/activation+impulsivity axis into a pure mania axis
and a distinct externalizing/neurodevelopmental axis — the genuine, imputation-free counterpart
of that ADHD/trauma signal. The SS order therefore changed: illness-burden now precedes the
(purified) mania axis, and the externalizing axis enters at position 5.

  ``axis1`` → depression_severity     (Depression / internalizing)
  ``axis2`` → later_onset             (Later onset)
  ``axis3`` → illness_burden          (Illness / hospitalization burden)
  ``axis4`` → mania_activation        (Mania / activation — pure)
  ``axis5`` → externalizing           (Externalizing / neurodevelopmental: impulsivity, childhood ADHD, early adversity)
  ``axis6`` → metabolic               (Metabolic / inflammatory)
  ``axis7`` → work_disability         (Socio-occupational / work-disability)

Import from the package: ``from trans_diag import AXIS_NAMES, AXIS_SHORT, AXIS_LABELS``.
If 07's axis ordering is ever re-locked, update only this file.
"""
from __future__ import annotations

# snake_case names in axis1..axis7 (sum-of-squares) order — the order 07 writes.
AXIS_NAMES: list[str] = [
    "depression_severity",
    "later_onset",
    "illness_burden",
    "mania_activation",
    "externalizing",
    "metabolic",
    "work_disability",
]

# Short display labels (compact figure axes, tables).
AXIS_SHORT: dict[str, str] = {
    "depression_severity": "Depression",
    "later_onset": "Later onset",
    "illness_burden": "Illness burden",
    "mania_activation": "Mania",
    "externalizing": "Externalizing",
    "metabolic": "Metabolic",
    "work_disability": "Work-disability",
}

# Full descriptive labels (figure titles, captions).
AXIS_LABELS: dict[str, str] = {
    "depression_severity": "Depression / internalizing",
    "later_onset": "Later onset",
    "illness_burden": "Illness / hospitalization burden",
    "mania_activation": "Mania / activation",
    "externalizing": "Externalizing / neurodevelopmental (impulsivity, childhood ADHD, early adversity)",
    "metabolic": "Metabolic / inflammatory",
    "work_disability": "Socio-occupational / work-disability",
}

# Map the generic axis index (as written by 07: "axis1".."axis7") to the canonical name.
AXIS_INDEX_TO_NAME: dict[str, str] = {f"axis{i + 1}": n for i, n in enumerate(AXIS_NAMES)}

__all__ = ["AXIS_NAMES", "AXIS_SHORT", "AXIS_LABELS", "AXIS_INDEX_TO_NAME"]
