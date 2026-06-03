"""Canonical names for the v2 trans-diagnostic dimensions — single source of truth.

The v2 hierarchical/bifactor pipeline (scripts/30-35_*.py) re-derived the dimensional structure
from zero. It writes generic ``dim1``..``dim4`` in ``results/hfa/stage3_loadings.csv`` (paf order,
descending eigenvalue). This module fixes their interpretation so downstream code + the manuscript
do not re-hardcode names.

(The superseded v1 6-axis solution — depression, later_onset, mania_activation, illness_burden,
cognition_verbal, metabolic — is archived at git tag ``v1-archive-2026-05-30`` and is not used here.)

History (LABBOOK V2-9..V2-11): on the re-curated v2 dictionary + the hybrid measurement model, the
structure locks at **K=4** (per-factor split-half congruence; K=6 sensitivity adds cardiac/somatic-
history + childhood-trauma). There is **no general p-factor** (ECV 0.36). The four axes are confound-
clean, leave-cohort-out reproducible, and granularity-invariant (Stage 4). Crucially, **mania and
suicidality are valid, well-measured constructs that are ORTHOGONAL to the four axes** (|r| <= 0.09):
they do not share enough variance with the others to anchor a second-order factor, so they are
reported as independent standalone dimensions, NOT as part of the correlated trans-diagnostic
structure. (This is why v2 has no separate "mania" axis, unlike v1.)

  ``dim1`` -> internalizing      (depression / anxiety / poor functioning; higher = more severe)
  ``dim2`` -> cognition          (cognitive impairment; higher = more impaired)
  ``dim3`` -> illness_course     (age of onset + inverse hospitalization burden; higher = LATER onset / LOWER chronicity)
  ``dim4`` -> cardiometabolic    (metabolic + inflammatory + autonomic burden; higher = worse)

Import: ``from trans_diag import AXIS_NAMES, AXIS_SHORT, AXIS_LABELS, ORTHOGONAL_DIMENSIONS``.
If Stage 3's dim ordering is ever re-locked, update only this file.
"""
from __future__ import annotations

# snake_case names in dim1..dim4 (paf / descending-eigenvalue) order — the order stage 33 writes.
AXIS_NAMES: list[str] = [
    "internalizing",
    "cognition",
    "illness_course",
    "cardiometabolic",
]

# Short display labels (compact figure axes, tables).
AXIS_SHORT: dict[str, str] = {
    "internalizing": "Internalizing",
    "cognition": "Cognition",
    "illness_course": "Illness course",
    "cardiometabolic": "Cardiometab-infl",
}

# Full descriptive labels (figure titles, captions) — including polarity.
AXIS_LABELS: dict[str, str] = {
    "internalizing": "Internalizing (depression / anxiety / poor functioning) — higher = more severe",
    "cognition": "Cognitive impairment — higher = more impaired",
    "illness_course": "Illness course (onset + inverse burden) — higher = later onset / lower chronicity",
    "cardiometabolic": "Cardiometabolic–inflammatory burden — higher = worse",
}

# dim index (as written by stage 33: "dim1".."dim4") -> canonical name.
AXIS_INDEX_TO_NAME: dict[str, str] = {f"dim{i + 1}": n for i, n in enumerate(AXIS_NAMES)}

# Valid, well-measured constructs that are ORTHOGONAL to the four correlated axes (|r| <= 0.09;
# Stage 4). Reported as independent standalone dimensions — included as features in stratification,
# but NOT part of the correlated trans-diagnostic factor structure.
ORTHOGONAL_DIMENSIONS: dict[str, str] = {
    "mania_activation": "Mania / activation (Altman + YMRS) — orthogonal standalone dimension",
    "suicidal_ideation": "Suicidal ideation (ISF) — orthogonal standalone dimension",
}

__all__ = ["AXIS_NAMES", "AXIS_SHORT", "AXIS_LABELS", "AXIS_INDEX_TO_NAME", "ORTHOGONAL_DIMENSIONS"]
