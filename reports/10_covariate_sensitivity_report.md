# 10 — covariate-adjusted biology⊥G sensitivity (P0-04)

Correlated-G marginalized model, N≈2000 balanced, 2 seed(s). Each continuous item is partialled on age(spline)+age×sex+sex+edulevel+site before the factor model (FWL-equivalent to the published β_jᵀc_i). Φ(G,·) compares the unadjusted vs covariate-adjusted G-correlations.

## Convergence
| arm        | seed   |   rhat |   ess |   div |
|:-----------|:-------|-------:|------:|------:|
| unadjusted | s1     |   1.01 |   421 |     0 |
| unadjusted | s2     |   1.01 |   460 |     0 |
| adjusted   | s1     |   1.02 |   724 |     0 |
| adjusted   | s2     |   1.01 |   495 |     0 |

## Biology⊥G — unadjusted vs covariate-adjusted
| domain       |   phi_G_unadjusted |   phi_G_adjusted |   delta |
|:-------------|-------------------:|-----------------:|--------:|
| inflammatory |              0.071 |            0.056 |  -0.015 |
| metabolic    |              0.124 |            0.058 |  -0.067 |
| cognition    |              0.385 |            0.229 |  -0.156 |
| sleep        |              0.422 |            0.409 |  -0.013 |

- **Survives covariate adjustment:** metabolic and inflammatory remain the least severity-entangled domains after adjusting for age/sex/education/site.
