# 13 — invariance of mania + substance across cohorts (§8)

Per-cohort **joint 9-dim mixed** fits (thin factors identified via the shared structure); Tucker congruence φ of the target loadings on the cohorts where each factor is identified (φ ≥ 0.85 = invariant). substance is a **2-cohort axis** (alcohol/cannabis SUD BP/SZ-only) — tested BP-vs-SZ; mania's Altman is BP/DR-only — tested BP-vs-DR.

**Convergence.** The overall structural R-hat (BP 1.03 · SZ 1.86 · DR 1.23) is inflated by the factor each cohort *cannot* identify (mania-in-SZ has no Altman; substance-in-DR has no SUD). The **target loadings themselves converged** — substance lh_ R-hat ≤ 1.06 (BP+SZ), mania R-hat ≤ 1.04 (BP+DR) — so the φ below are trustworthy, not artefacts.

## Target loadings by cohort (blank = item absent in that cohort)
| factor           | item                 |   BP |     SZ |   DR |
|:-----------------|:---------------------|-----:|-------:|-----:|
| mania_activation | ymrs                 | 0.57 |   0.16 | 0.41 |
| mania_activation | altman               | 0.76 | nan    | 0.1  |
| substance        | suoccur_alcool       | 0.4  |   0.49 | 0.63 |
| substance        | suoccur_cannabis     | 0.37 |   0.49 | 0.62 |
| substance        | sudose_cigarettes_lt | 0.75 |   0.83 | 0.67 |
| substance        | fagers               | 0.54 |   0.53 | 0.74 |

## Congruence (testable pairs)
| factor    | pair   | items                            |   tucker_phi | verdict                          |
|:----------|:-------|:---------------------------------|-------------:|:---------------------------------|
| mania     | BP–DR  | YMRS, Altman                     |        0.764 | partial (Altman ✗ in DR, YMRS ✓) |
| substance | BP–SZ  | alcohol/cannabis SUD, cigarettes |        0.997 | invariant                        |

## Verdict
- **substance — INVARIANT across its two cohorts (BP–SZ): φ = 0.997.** Alcohol/cannabis SUD + cigarettes load congruently (alcohol 0.40/0.49 · cannabis 0.37/0.49 · cigarettes 0.75/0.83); loadings converged (R-hat ≤ 1.06). Declared a 2-cohort axis — not claimed for DR (no SUD).
- **mania — PARTIALLY invariant (BP–DR): φ = 0.764 < 0.85.** **YMRS holds** (BP 0.57 · DR 0.41) but **Altman does not transfer to DR** (BP 0.76 → DR 0.10). The DR loadings converged (R-hat ≤ 1.04), so this is **real, not a sampling artefact**: self-rated manic activation (Altman) is a near-floor signal in a depression-at-risk cohort. Documented partial invariance (§8) — alongside G-in-SZ and inflammatory-in-DR on the backbone.

## Implication
- substance scores are comparable across BP/SZ. **mania scores should lean on YMRS in DR** (Altman is non-discriminating there) and BP-vs-DR mania comparisons carry that caveat; mania is anyway a 2-indicator, lower-reliability axis (flagged *partial* for every patient in §7 scoring).

Artifacts: per-cohort idata `results/face/inv9_{bp,sz,dr}/`, `reports/13_invariance9_loadings.csv`.