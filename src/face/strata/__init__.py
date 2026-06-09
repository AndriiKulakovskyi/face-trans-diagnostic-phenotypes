"""M2 — patient stratification engine (probabilistic decision regions on the M1 coordinates).

Methods of record: ``docs/STRATIFICATION_MODEL.md``. This package acts on the M1 9-dim per-patient
coordinates (with uncertainty), never on raw indicators or diagnosis. Modules (built incrementally,
one per stage of the M2 pipeline):

    scoring   — M2.0 prep: full-N projection of the explicit (non-Gaussian) axes + uncertainty export.
    structure — M2.1 structure-discovery gate (Mapper / dip / Hopkins) — cluster vs continuum vs branched.
    mixture   — M2.2 Model A: measurement-error Bayesian mixture (hard probabilistic regions).
    archetypes— M2.3 Model B: archetypal analysis (soft archetype membership).
    validation— M2.4 the Q1–Q4 battery.
    viz       — figures (UMAP/PCA embeddings, Mapper, profile heatmaps) — viz-only, never a model input.
"""
from __future__ import annotations
