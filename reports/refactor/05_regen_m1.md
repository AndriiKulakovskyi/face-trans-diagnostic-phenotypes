# Refactor Step 5 (M1) — regenerate the transdiagnostic map from sources + reconcile

**Branch:** `refactor/ground-up-2026-07` · **Date:** 2026-07-05 · Fit: `face fit m1 --mode production --detach`.

## What ran
- `face build-data` + `face build-covariates` → `data/processed/baseline_v{0,1,2}.parquet` (+ covariates/site/metadata).
- `face fit m1 --mode production --detach` — the canonical two-phase recipe (`face.measurement.run`):
  **Phase 1** balanced horseshoe merge ladder `hs_s1_merged → hs_s3_merged → hs_s5_merged_xc` (N=2000),
  **Phase 2** full-N cohort-weighted salvage with substance pinned orthogonal, warm-started from Phase 1.
  → `results/m1_measurement/primary/idata.nc`. Detached, `caffeinate`-wake-locked; survived overnight.
- `run_export_loadings.py --idata results/m1_measurement/primary/idata.nc` → loadings + Φ (CI-aware).

## What was checked (convergence)
| stage | R-hat | ESS | div | time |
|---|---|---|---|---|
| hs_s1_merged (continuous core) | 1.01 | 2285 | 0 | 5.4 min |
| hs_s3_merged (+dev, mania) | 1.00 | 976 | 0 | 10.7 min |
| hs_s5_merged_xc (8-factor mixed, balanced) | 1.06 | 77 | 0 | 37 min |
| **primary (full-N weighted, substance ⊥)** | **1.03** | 97 | **0** | 3.2 h |

Matches the oracle headline (R-hat 1.03, 0 divergences) and REPRODUCE.md's documented convergence profile.

## Reconciliation vs `reference/oracle/m1/` — **BIT-IDENTICAL**
- **Loadings:** Tucker congruence φ = **1.0000** for all 8 factors (worst 1.0000; whole-matrix 1.0000); max |Δ loading| = **0.0000**. Gate ≥ 0.99 → **pass**.
- **Φ:** max |Δ off-diagonal| = **0.0000** (gate ≤ 0.05); substance-orthogonality and bifactor-G ⊥ preserved exactly.
- **Cross-loadings:** exactly the 3 earned cells recovered — ctq37→cognition −0.080, psqi11→cognition +0.047,
  psqi17→cognition −0.051 — same sign, all 95% CI exclude 0. Gate (exactly 3, same sign) → **pass**.
- **Spot biology:** bmi→immunometabolic 0.946 (oracle 0.95), crp→immunometabolic 0.366 (oracle 0.37).

Same seed (20260605) + bit-identical `baseline_v0` + unchanged numerical kernels (renamed only) + the env of
record (Python 3.13.9, pymc 6.0.1, numpyro 0.21.0, jax 0.10.1) yield an exact reproduction of the map.

## Converged?
**YES — exact reproduction.** The clean `face.measurement` pipeline reproduces the certified M1 transdiagnostic
map bit-identically. Hand-off `results/m1_measurement/primary/{idata.nc, loadings_summary.csv, phi.csv,
manifest.json}` is the fixed input for M2. Next: `face fit m2`.
