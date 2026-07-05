# Refactor Step 5 (M5) — regenerate treatment + reconcile · + Step-5 completion

**Date:** 2026-07-05 · Fit: `face fit m5 --detach` (overlap → propensity → DR-EIV moderation + E-value, seed 20260605)
on the fixed M4 frame.

## M5 reconciliation vs `reference/oracle/m5/` (NUTS → within-tolerance)
Treatment verdict table (`treatment_summary.csv`) matches the oracle essentially cell-for-cell:
- **lithium-BP functioning:** E 1.19 / 1.28 (oracle 1.20 / 1.28), `moderation_any_axis`=False, verdict **bounded null** — both. ✓
- **antipsychotic-BP functioning:** E 1.80 / 1.79 (oracle 1.80 / 1.79), `any_axis`=True, verdict **suggestive** — both. ✓
- **clozapine-SZ:** underpowered — both. ✓ (one borderline underpowered cgi_response cell flips True/False across NUTS draws; does not change the conclusion.)
- ATEs match to ±0.01; E-values to ±0.01; `propensity_summary` overlap fractions + verdicts match (all estimable; clozapine-active_comparator "residual imbalance (caution)" both).

The per-contrast null-vs-signal verdict is preserved (the MANIFEST invariant): map is prognostic + descriptive, not prescriptive.
Moderation/atlas layout drift vs oracle is the design-flagged known drift — reconciled on the numeric keys + verdicts.

## Step 5 COMPLETE — full M1→M5 vertical regenerated from raw sources
Per-milestone reconciliation vs `reference/oracle/`:
| milestone | fit | reconciliation |
|---|---|---|
| M1 measurement | `face fit m1` (production, ~4 h detached) | **bit-identical** — loadings Tucker φ=1.0000 ×8, Φ Δ=0, 3 cross-loadings exact, R-hat 1.03 / 0 div |
| M2 strata | `face fit m2` (deterministic) | **exact** — archetype profiles Δ=0, k-family Δ=0 (A=5, K=2) |
| M3 temporal | `face fit m3` | **copula-canonical** — invariance license exact (immunometabolic φ 0.987), ICCs match STATE.md |
| M4 prognosis | `face fit m4` | **within-tolerance** — +archetypesA ΔELPD +62.76 (=+62.8), operative_K none, top-ranked |
| M5 treatment | `face fit m5` | **within-tolerance** — E-values ±0.01, null-vs-signal verdicts preserved |

**Global fingerprint** (`reference/oracle/fingerprint.py`, retargeted to the clean result tree): M1/M2 coordinates
rank-hash 8/8 exact + moment Δ 0; M2 & M4 archetype assignments dominant-hash EXACT + sizes match; M3 panel
within tolerance (mean Δ ≤ 0.09, rank-hash 5/8 exact — NUTS V1/V2 projection). Deterministic layers bit-identical
per-patient; the one stochastic scoring layer within tolerance.

## Converged?
**YES — the entire research program reproduces from raw sources through the clean `face` pipeline.** Remaining:
Step 6 (figures + article re-point/rebuild) and Step 7 (docs rewrite + final sign-off).
