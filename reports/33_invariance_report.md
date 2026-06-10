# 33 — G1: longitudinal measurement invariance (V0 → V1 → V2)

Per-visit **simple-structure** backbone (cohort-balanced N≈1800, in-sample z-scored, 3 seed(s)). Metric invariance = Tucker congruence φ of the primary loadings per factor vs V0 (φ≥0.95 invariant · ≥0.85 partial). Tucker φ is scale-invariant, so the frozen V0 spec is deliberately not used here (this tests loading *shape*). A passing axis licenses interpreting its V1/V2 coordinate change as patient change (G3/G4).

## Convergence — 8/9 fits converged (R-hat ≤ 1.05 · ESS ≥ 100 · 0 div)
| fit          |   rhat |   ess |   div | converged   |
|:-------------|-------:|------:|------:|:------------|
| V0 s20260609 |   1.02 |   373 |     0 | True        |
| V1 s20260609 |   1.01 |   498 |     0 | True        |
| V2 s20260609 |   1.02 |   298 |     0 | True        |
| V0 s20260610 |   1.02 |   554 |     0 | True        |
| V1 s20260610 |   1.02 |   240 |     0 | True        |
| V2 s20260610 |   1.01 |   162 |     8 | False       |
| V0 s20260611 |   1.02 |   473 |     0 | True        |
| V1 s20260611 |   1.02 |   381 |     0 | True        |
| V2 s20260611 |   1.01 |   453 |     0 | True        |

- ⚠ 1 non-converged fit(s) excluded from the φ averages — congruence rests only on fits that passed the gate (each axis still has ≥1 converged seed at every visit).

## Metric invariance — Tucker congruence φ vs V0 (mean over seeds)
| factor             | visit   |   n_items |   phi_mean |   phi_min | verdict   |
|:-------------------|:--------|----------:|-----------:|----------:|:----------|
| overall_severity   | V1      |        10 |      0.988 |     0.987 | invariant |
| cognition          | V1      |        11 |      0.993 |     0.989 | invariant |
| metabolic          | V1      |        29 |      0.991 |     0.99  | invariant |
| inflammatory       | V1      |         8 |      0.924 |     0.828 | partial   |
| sleep              | V1      |         9 |      0.998 |     0.998 | invariant |
| developmental_risk | V1      |         7 |      0.965 |     0.955 | invariant |
| overall_severity   | V2      |        11 |      0.992 |     0.992 | invariant |
| cognition          | V2      |        11 |      0.994 |     0.991 | invariant |
| metabolic          | V2      |        29 |      0.991 |     0.991 | invariant |
| inflammatory       | V2      |         8 |      0.901 |     0.822 | partial   |
| sleep              | V2      |         9 |      0.996 |     0.994 | invariant |
| developmental_risk | V2      |         7 |      0.959 |     0.952 | invariant |

## Per-axis license — 5 invariant · 1 partial (backbone axes)
| axis               |   min_phi | license   |
|:-------------------|----------:|:----------|
| cognition          |     0.993 | invariant |
| developmental_risk |     0.959 | invariant |
| inflammatory       |     0.901 | partial   |
| metabolic          |     0.991 | invariant |
| overall_severity   |     0.988 | invariant |
| sleep              |     0.996 | invariant |

- **Not tested in stage 33** (mania_activation, suicidality, substance): mania has only 2 indicators (φ on 2 items is unstable); suicidality/substance are explicit (binary/count) — not in the continuous backbone (a heavier mixed-model refit). These carry `license=not-tested` in the panel; their change is reported descriptively, not as licensed patient-change.

## Loading-DIF — largest cross-visit loading spread (seed 1)
| item          | factor             |   spread |   load_V0 |   load_V1 |   load_V2 |
|:--------------|:-------------------|---------:|----------:|----------:|----------:|
| wbc           | inflammatory       |    0.519 |     0.554 |     1.072 |     0.943 |
| mono_lbstresc | inflammatory       |    0.468 |     0.905 |     0.535 |     0.436 |
| eos           | inflammatory       |    0.398 |     0.654 |     0.28  |     0.256 |
| baso_lbstresc | inflammatory       |    0.361 |     0.481 |     0.12  |     0.134 |
| ctq35         | developmental_risk |    0.35  |     0.498 |     0.722 |     0.371 |
| agepere       | developmental_risk |    0.34  |     0.021 |     0.361 |     0.342 |
| neut          | inflammatory       |    0.302 |     0.528 |     0.83  |     0.775 |
| fast25        | overall_severity   |    0.302 |     1.073 |     0.771 |     0.888 |
| ctq33         | developmental_risk |    0.276 |     0.842 |     0.606 |     0.566 |
| hba1c         | metabolic          |    0.215 |     0.416 |     0.201 |     0.31  |

- 10 item(s) with cross-visit spread > 0.20.

## Verdict
The continuous backbone is **largely invariant** across V0→V1→V2 on the testable axes: cognition (invariant, φ=0.993), developmental_risk (invariant, φ=0.959), inflammatory (partial, φ=0.901), metabolic (invariant, φ=0.991), overall_severity (invariant, φ=0.988), sleep (invariant, φ=0.996). Where φ is only partial, the axis's change is interpreted with that caveat (documented partial, not hidden). The licenses gate stage 34's panel.

Artifacts: `reports/33_{congruence,dif_items}.csv` · `results/face/m3/invariance_license.parquet` · `docs/figures/33_congruence.png`.