# Refactor Step 5 (M2) — regenerate strata + reconcile

**Date:** 2026-07-05 · Fit: `face fit m2 --detach` (deterministic numpy/EM, seed 20260621) on the M1 primary map.

## What ran
`face.strata.run.run_m2` walked the six-stage plan (coordinates → structure → regions → archetypes →
usefulness → consolidate) on `results/m1_measurement/primary`. Result: **A=5 archetypes, operational K=2,
N=9,013**. Hand-off `results/m2_strata/consolidate/{patient_strata.parquet, k_family_menu.csv,
archetype_profiles.csv}` + coordinates under `results/m2_strata/coordinates/`.

## Reconciliation vs `reference/oracle/m2/` — **EXACT**
- **Archetype profiles** (`archetype_profiles.csv`, arm A_all9): 5 archetypes, labels aligned 1:1, max |Δ|
  over archetypes × 8 axes = **0.0000**. Arm B_specifics aligns 1:1 (the G axis is structurally NaN in the
  G-residualized arm; all non-G axes exact).
- **K-family menu** (`k_family_menu.csv`): K∈{2,3,4}, 13 numeric metrics (BIC, confident-dominant frac,
  entropy, seed-ARI, η²…) max |Δ| = **0.0000**.
- A=5 selected (largest A with cross-seed Tucker ≥ 0.8), operational K=2 tessellation — matches the reported map.

## Converged?
**YES — exact reproduction.** Deterministic strata engine + the bit-identical M1 map ⇒ the A=5 simplex and
nested K-family reproduce exactly. Hand-off ready for M3. Next: `face fit m3`.
