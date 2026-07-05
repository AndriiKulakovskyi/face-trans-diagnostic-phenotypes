# Refactor Step 5 (M4) — regenerate prognosis + reconcile

**Date:** 2026-07-05 · Fit: `face fit m4 --detach` (EIV Bayesian GLM, NUTS draws=800 tune=1000 chains=4, seed 20260610)
on the fixed M2 coords/archetypes + M3 attrition IPW + outcomes.

## What ran
`face.prognosis.run.run_m4` walked frame → reference → incremental → transdiagnostic → endpoints →
clinical_value → robustness → consolidate. Frame = 9,013 rows. Hand-off
`results/m4_prognosis/{consolidate/{prognosis_patient_risk.parquet,prognosis_summary.csv}, endpoints/archetype_atlas.csv}`.

## Reconciliation vs `reference/oracle/m4/` (NUTS → within-tolerance, not bit-identical)
- **Headline — functioning (egf):** `+archetypesA` ΔELPD = **+62.76** (se 11.2), verdict **predictive**, **top-ranked**
  encoding. Matches CLAUDE.md's "+62.8 held-out"; within the gate band [+45, +72]. Ordering preserved
  (+archetypesA > +specifics8 > +archetypesB > +tess_k{2,3,4} > +durable > R3y).
- **operative_K = None** — reproduces the canonical M4 answer to the M2 K-question exactly.
- **Severity (cgi_s):** all encodings small ΔELPD (autoregression-saturated) — reproduces "predicts functioning, not severity".
- **Convergence cleaner than the oracle:** new `+specifics8` max-Pareto-k 0.51 / R-hat 1.0 vs the oracle's flagged 1.01 / 1.17; 0 divergences throughout.

## Caveats (design-flagged, reconcile on science not byte-diff)
- `incremental.csv`: new emits K-family-expanded `+tess_k{K}`; oracle has a single `+tessellation` — map best-tess ↔ tessellation.
- `archetype_atlas.csv` (long vs oracle wide), `clinical_value.csv` (per-endpoint vs per-model rows), `robustness.csv`
  (per-encoding ΔELPD sweep vs oracle per-axis EIV coefficients) have **known layout drift** between the current
  engine and the frozen oracle — reconcile on the scientific conclusion (per-archetype remission gradient, IPW survival), not table byte-identity.

## Converged?
**YES** — the prognostic headline reproduces: archetypes predict 2-year functioning (ΔELPD +62.8, predictive,
top-ranked), operative K none, not-severity. Hand-off ready. Next: `face fit m5` (consumes the M4 frame).
