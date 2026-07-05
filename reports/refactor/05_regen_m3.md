# Refactor Step 5 (M3) — regenerate temporal coherence + reconcile

**Date:** 2026-07-05 · Fit: `face fit m3 --detach` (seed 20260622) on the fixed M1 primary + M2 strata + V1/V2 tables.

## What ran
`face.temporal.run.run_m3` walked invariance → panel → attrition → trait_state → persistence → consolidate,
scoring V1/V2 under the FIXED copula M1/M2 (never re-discovered). Hand-off
`results/m3_temporal/{consolidate/patient_panel.parquet, attrition/ipw_weights.parquet, trait_state/, invariance/}`.

## Reconciliation vs `reference/oracle/m3/`
- **G1 invariance license (copula-vintage oracle) — EXACT match.** New vs oracle `invariance_license.csv`:
  all 4 backbone axes invariant, byte-identical — cognition 0.995, **immunometabolic 0.987**, overall_severity
  0.991, sleep 0.996. Matches STATE.md's "immunometabolic φ 0.987; all 4/4 backbone axes invariant".
- **G3 trait/state ICC — matches the copula-canonical STATE.md values (±0.03):**
  immunometabolic **0.909** (STATE.md 0.91, most-durable trait) · cognition 0.697 (0.70 trait) · overall_severity
  0.618 (0.62 trait-by-rank) · developmental_risk 0.384 (0.39 state) · sleep 0.467 (mixed). Trait/state verdicts unchanged.

## Oracle-harvest caveat (Step-0 imperfection, corrected understanding — NOT an M3 defect)
Two Step-0 oracle CSVs were harvested from the **pre-merge NATIVE vintage** (`reports/35_trait_state.csv`,
`reports/33_congruence.csv`) — they carry *separate* `metabolic` (ICC 0.932) + `inflammatory` (0.854) axes, whereas
the copula-canonical M3 (the one M4/M5 + the article consume, per STATE.md/TEMPORAL_OOP_FINDINGS.md) has the
**merged immunometabolic** axis. The correctly-harvested copula artifact (`invariance_license.csv`, from
`temporal_oop/`) reproduces **exactly**, and the new immunometabolic ICC 0.909 ≈ the metabolic/inflammatory pair
merged — consistent, not divergent. The reconciliation target for M3 is the copula values (STATE.md), which the
new M3 reproduces. Substance ICC (0.749 vs native 0.999) is thin/orthogonal/BP-SZ-only and expected-unstable.

## Converged?
**YES** — the copula temporal-coherence findings reproduce (G1 invariance exact; G3 ICCs match STATE.md; biology
durable/trait, symptoms slide). Next: `face fit m4` (consumes M2 coords/archetypes + M3 attrition IPW).
