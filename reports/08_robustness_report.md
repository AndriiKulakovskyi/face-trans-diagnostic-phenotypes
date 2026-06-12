# 08 — robustness of the reported map (§8/§3.6)

Tucker congruence φ of the primary loadings (G + cognition/metabolic/inflammatory/sleep) vs the certified full-N **S2 reference**, under four perturbations (φ≥0.85 = robust). Marginalized continuous backbone; K=5 resamples per bootstrap arm.

## LOCO
| perturbation   |   overall_severity |   cognition |   metabolic |   inflammatory |   sleep |
|:---------------|-------------------:|------------:|------------:|---------------:|--------:|
| drop BP        |              0.958 |       0.999 |       0.985 |          0.985 |   0.995 |
| drop SZ        |              0.999 |       0.997 |       0.996 |          0.987 |   0.999 |
| drop DR        |              0.997 |       1     |       0.998 |          0.998 |   0.998 |

## diagnosis-balanced
| perturbation   |   overall_severity |   cognition |   metabolic |   inflammatory |   sleep |
|:---------------|-------------------:|------------:|------------:|---------------:|--------:|
| seed 1         |              0.995 |       0.998 |       0.995 |          0.985 |   0.999 |
| seed 2         |              0.995 |       0.996 |       0.996 |          0.983 |   0.999 |
| seed 3         |              0.995 |       0.997 |       0.996 |          0.991 |   0.999 |
| seed 4         |              0.995 |       0.998 |       0.996 |          0.986 |   0.999 |
| seed 5         |              0.995 |       0.999 |       0.997 |          0.98  |   0.999 |

## site-bootstrap
| perturbation   |   overall_severity |   cognition |   metabolic |   inflammatory |   sleep |
|:---------------|-------------------:|------------:|------------:|---------------:|--------:|
| resample 1     |              1     |       0.999 |       0.994 |          0.993 |   1     |
| resample 2     |              0.998 |       0.997 |       0.998 |          0.994 |   0.999 |
| resample 3     |              0.999 |       0.997 |       0.998 |          0.987 |   1     |
| resample 4     |              0.999 |       1     |       0.998 |          0.994 |   1     |
| resample 5     |              1     |       0.999 |       0.998 |          0.99  |   0.999 |

## weighted (1/n_cohort)
| perturbation   |   overall_severity |   cognition |   metabolic |   inflammatory |   sleep |
|:---------------|-------------------:|------------:|------------:|---------------:|--------:|
| all-N weighted |              0.994 |       0.997 |       0.998 |          0.982 |   0.999 |

## Summary
| arm                   |   min_phi | worst_factor     |
|:----------------------|----------:|:-----------------|
| LOCO                  |     0.958 | overall_severity |
| diagnosis-balanced    |     0.98  | inflammatory     |
| site-bootstrap        |     0.987 | inflammatory     |
| weighted (1/n_cohort) |     0.982 | inflammatory     |

- **Min Tucker φ across all arms/factors = 0.958** ⇒ the reported map is **robust** (φ≥0.85). The loading structure holds under leave-one-cohort-out, diagnosis-balanced subsampling, site cluster-bootstrap, and 1/n_cohort weighting — it is not an artefact of cohort imbalance, any single cohort, or recruitment-site clustering.