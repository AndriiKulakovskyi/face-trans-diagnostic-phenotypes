# S5 correlated-G sensitivity — the biology⊥G refinement (§3.1)

All factors freely correlated (G not held orthogonal), simple structure, marginalized continuous model, N≈2000 balanced, 2 seed(s). Reads G's correlation with each specific vs the bifactor's near-zero direct G-loading — the dual-identification test.

## Convergence
| seed   |   rhat |   ess |   div |
|:-------|-------:|------:|------:|
| s1     |   1.01 |   421 |     0 |
| s2     |   1.01 |   460 |     0 |

## Biology⊥G under both identifications
| domain       |   bifactor_loading_on_G |   corrG_phi_with_G |   seed_range |
|:-------------|------------------------:|-------------------:|-------------:|
| inflammatory |                    0.07 |              0.071 |        0.027 |
| metabolic    |                    0.08 |              0.124 |        0.03  |
| cognition    |                    0.26 |              0.385 |        0.018 |
| sleep        |                    0.25 |              0.422 |        0.038 |

- **Bifactor** holds G⊥specifics (direct G-loadings ≈ 0 for biology). **Correlated-G** lets G correlate: biology still shows the **lowest** G-correlations (inflammatory < metabolic) — well below cognition/sleep. So biology is **not strictly orthogonal** to severity but is the **least severity-entangled domain** (largely severity-independent); the bifactor's strict orthogonality slightly overstated it. The load-bearing premise — biological strata capture heterogeneity that severity misses — holds.
