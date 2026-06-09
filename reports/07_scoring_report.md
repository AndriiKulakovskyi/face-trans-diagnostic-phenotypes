# 07 — per-patient dimension scoring (§7), 9-dim joint map

Per-patient coordinates with uncertainty for **9,013 patients**, fit-once-score-all (§3.6). Six continuous-anchored dimensions (G + cognition/metabolic/inflammatory/sleep/**mania**) via draw-wise analytic conditional-Gaussian scores from the **certified 9-dim joint** loadings; three explicit (suicidality/developmental/**substance**) via f_e from the same fit (subsample n=1,884). Orientation: higher = more burden. Each dimension carries mean · SD · HDI · #observed home indicators · reliability tier.

## Reliability — patients per tier, by continuous-anchored dimension
(well = ≥3 observed home indicators · partial = 1–2 · prior-dominated = 0)
|                  |   partial |   prior-dominated |   well |
|:-----------------|----------:|------------------:|-------:|
| overall_severity |       249 |               158 |   8606 |
| cognition        |        56 |              2506 |   6451 |
| metabolic        |        67 |               431 |   8515 |
| inflammatory     |       227 |              1684 |   7102 |
| sleep            |       122 |              1369 |   7522 |
| mania_activation |      8594 |               419 |      0 |

## Dimension summary (posterior-mean scores, z-scored, higher = more burden)
|                  |   mean |   sd_across_patients |   mean_posterior_SD |
|:-----------------|-------:|---------------------:|--------------------:|
| overall_severity |   0.09 |                 0.82 |                0.29 |
| cognition        |  -0    |                 0.75 |                0.44 |
| metabolic        |  -0.01 |                 0.93 |                0.27 |
| inflammatory     |  -0    |                 0.77 |                0.55 |
| sleep            |  -0.01 |                 0.9  |                0.28 |
| mania_activation |  -0.01 |                 0.83 |                0.66 |

## Notes
- A patient with few observed indicators for a dimension gets a **prior-dominated** flag and a wider posterior SD — downstream strata (M2) must propagate this uncertainty, not treat all coordinates as equally characterised.
- **Suicidality/developmental/substance are scored on the S5 subsample** (their explicit f_e); full-N projection of the non-Gaussian block (a logistic/count projection, not Gaussian) is a documented follow-on for M2.

Artifacts: `results/face/patient_scores.parquet` (per-patient, gitignored).