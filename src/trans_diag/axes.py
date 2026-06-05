"""Canonical names for the v2 trans-diagnostic dimensions — single source of truth.

The v2 hierarchical/bifactor pipeline (scripts/01-15_*.py) re-derives the dimensional structure
from zero. It writes generic ``dim1``..``dimK`` in ``results/hfa/stage3_loadings.csv`` (paf order,
descending eigenvalue). This module fixes their interpretation so downstream code + the manuscript
do not re-hardcode names.

(The superseded v1 6-axis solution — depression, later_onset, mania_activation, illness_burden,
cognition_verbal, metabolic — is archived at git tag ``v1-archive-2026-05-30`` and is not used here.)

History (LABBOOK V2-9..V2-20): on the re-curated v2 dictionary + the hybrid measurement model, the
structure first locked at K=4. Adding the lifetime alcohol/cannabis **substance_use_disorder**
construct (2026-06-04) to the second-order extraction collapses the K=4 split-half congruence
(0.96 -> 0.31; counterfactual-confirmed) and the data lock moves to **K=3**: the previously weakest
axis, *illness_course* (age-of-onset + inverse hospitalization burden), is no longer a reproducible
standalone second-order factor — its content disperses (the inverse-burden part is partly absorbed by
the cardiometabolic axis). There is **no general p-factor** (ECV ~0.42). As before, **mania,
suicidality and substance-use disorder are valid, well-measured constructs that are ORTHOGONAL to the
axes** (|loading| < 0.10): they do not share enough variance to anchor a second-order factor, so they
are reported as independent standalone dimensions, NOT part of the correlated trans-diagnostic
structure.

  ``dim1`` -> internalizing      (depression / anxiety / poor functioning; higher = more severe)
  ``dim2`` -> cognition          (cognitive impairment; higher = more impaired)
  ``dim3`` -> cardiometabolic    (metabolic + inflammatory + autonomic burden; higher = worse)

Import: ``from trans_diag import AXIS_NAMES, AXIS_SHORT, AXIS_LABELS, ORTHOGONAL_DIMENSIONS``.
If Stage 3's dim ordering is ever re-locked, update only this file.
"""
from __future__ import annotations

# snake_case names in dim1..dimK (paf / descending-eigenvalue) order — the order stage 3 writes.
AXIS_NAMES: list[str] = [
    "internalizing",
    "cognition",
    "cardiometabolic",
]

# Short display labels (compact figure axes, tables).
AXIS_SHORT: dict[str, str] = {
    "internalizing": "Internalizing",
    "cognition": "Cognition",
    "cardiometabolic": "Cardiometab-infl",
}

# Full descriptive labels (figure titles, captions) — including polarity.
AXIS_LABELS: dict[str, str] = {
    "internalizing": "Internalizing (depression / anxiety / poor functioning) — higher = more severe",
    "cognition": "Cognitive impairment — higher = more impaired",
    "cardiometabolic": "Cardiometabolic–inflammatory burden — higher = worse",
}

# dim index (as written by stage 3: "dim1".."dimK") -> canonical name.
AXIS_INDEX_TO_NAME: dict[str, str] = {f"dim{i + 1}": n for i, n in enumerate(AXIS_NAMES)}

# Valid, well-measured constructs that are ORTHOGONAL to the correlated axes (|loading| < 0.10;
# Stage 3/4). Reported as independent standalone dimensions — included as features in stratification
# and the predictive arm, but NOT part of the correlated trans-diagnostic factor structure.
ORTHOGONAL_DIMENSIONS: dict[str, str] = {
    "mania_activation": "Mania / activation (Altman + YMRS) — orthogonal standalone dimension",
    "suicidal_ideation": "Suicidal ideation (ISF) — orthogonal standalone dimension",
    "substance_use_disorder": "Lifetime alcohol/cannabis use disorder (MINI, BP/SZ) — orthogonal standalone dimension",
}

__all__ = ["AXIS_NAMES", "AXIS_SHORT", "AXIS_LABELS", "AXIS_INDEX_TO_NAME", "ORTHOGONAL_DIMENSIONS"]
