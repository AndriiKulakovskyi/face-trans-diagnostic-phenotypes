"""M3 temporal-coherence engine — score follow-up visits onto the FIXED M1/M2 model.

V0 defines, follow-up validates: M3 never re-discovers the 8-dim map or the M2 strata on
later visits. It scores each patient's **observed** V1/V2 cells onto the certified M1 loadings
(on the frozen V0 scale), propagates M1/M2 per-patient uncertainty, and asks whether the
measurement and the memberships *cohere and persist* over time. Methods of record:
docs/TEMPORAL_MODEL.md. Scope: V0 -> V1 -> V2 (yearly), internal validity only.

Modules (built incrementally, one per concern; mirrors `face.strata`):
  dropout      — G6 retention table + (later) raw dropout extractor + informative-dropout model
  standardize  — the V0 standardization spec (capture V0 moments, apply to any visit)   [stage 32]
  panel        — per-visit model-ready tables + the longitudinal panel substrate          [stage 32/34]
  score        — score one visit on the fixed model (thin reuse of the M1/M2 scorers)      [stage 34]
  membership   — frozen archetype + tessellation assignment per visit                      [stage 34]
  invariance   — G1 longitudinal measurement invariance (adapts scripts/06)                [stage 33]
  variance     — G3 trait/state variance components + reliable-change index                [stage 35]
  persistence  — G4 membership persistence + spine-vs-corner trajectories                  [stage 36]
"""
from __future__ import annotations

# The M3 follow-up window (decision locked 2026-06-09): yearly V0 -> V1 -> V2, where all three
# cohorts are well-represented. V3+ and interim `_mois` visits are out of scope for M3.
VISITS: tuple[str, ...] = ("V0", "V1", "V2")

# Canonical M1 dimension order — must match the strata engine's CANON (the order written into
# coordinates_full.parquet + archetype_profiles.csv). 8-factor: immunometabolic merge.
CANON: tuple[str, ...] = (
    "overall_severity", "cognition", "immunometabolic", "sleep", "mania_activation",
    "suicidality", "developmental_risk", "substance",
)

__all__ = ["VISITS", "CANON"]
