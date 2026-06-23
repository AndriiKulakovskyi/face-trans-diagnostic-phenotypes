# 12 — biology⊥G confound sensitivity (medication / adiposity / site)

Correlated-G marginalized model, N≈2000 balanced, 2 seed(s). Each continuous item is partialled (FWL) on a growing covariate design before the factor model; Φ(G,·) compares the G-entanglement of each domain across the adjustment ladder.

| arm | adjusts for |
|---|---|
| A0_unadjusted | nothing (the reported value) |
| A1_demo_site | age(spline) + sex + education + site |
| A2_antipsychotic | A1 + antipsychotic exposure **(conservative headline)** |
| A3_bmi | A2 + BMI moved to the covariate block **(exploratory / partly circular)** |

## Φ(G, domain) across the adjustment ladder
| domain       |   A0_unadjusted |   A1_demo_site |   A2_antipsychotic |   A3_bmi |
|:-------------|----------------:|---------------:|-------------------:|---------:|
| inflammatory |           0.064 |          0.051 |              0.06  |    0.051 |
| metabolic    |           0.127 |          0.066 |              0.069 |    0.06  |
| cognition    |           0.402 |          0.261 |              0.265 |    0.273 |
| sleep        |           0.371 |          0.358 |              0.361 |    0.357 |

## Convergence
| arm              | seed   |   rhat |   ess |   div |
|:-----------------|:-------|-------:|------:|------:|
| A0_unadjusted    | s1     |   1.01 |   338 |     0 |
| A0_unadjusted    | s2     |   1.01 |   301 |     0 |
| A1_demo_site     | s1     |   1.02 |   400 |     0 |
| A1_demo_site     | s2     |   1.01 |   455 |     0 |
| A2_antipsychotic | s1     |   1.01 |   367 |     0 |
| A2_antipsychotic | s2     |   1.01 |   450 |     0 |
| A3_bmi           | s1     |   1.83 |     2 |     0 |
| A3_bmi           | s2     |   1.84 |     2 |     0 |

## Verdict (on A2 — antipsychotic-adjusted)
- **Biology⊥G survives medication + site adjustment**: metabolic +0.069 and inflammatory +0.060 remain the least severity-entangled domains (both below cognition/sleep, min 0.265). The headline is confound-robust.
- A3 (BMI-as-covariate) shows convergence trouble — read as a circularity/identification flag, not a result; it is exploratory because BMI is itself a metabolic indicator.

## Honest limits
- Antipsychotic coverage ~54 % (NaN mean-imputed for the design; BP lifetime vs SZ/DR current).
- Antipsychotic is on the causal path to metabolic load, so adjusting for it is conservative-to-over-conservative (it can remove real signal).
- A site dummy is coarser than full cross-platform assay/batch harmonization.
- Internal sensitivity on the correlated-G measurement structure; not external validation.
