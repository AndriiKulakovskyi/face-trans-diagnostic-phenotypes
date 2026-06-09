# 23 — M2.3 archetypal analysis (soft archetype membership — the lead view)

Continuum verdict (M2.1) ⇒ each patient = a convex blend of **extreme phenotypes**, not a hard cluster. Soft simplex weights = the probabilistic decision regions. Native latent z-scale; uncertainty from projecting M1 draws onto fixed archetypes. **No biotype claim.**

## Number of archetypes A = **8** (data-driven knee on Arm A = 8)
- explained variance at A=8: Arm A **0.791** · Arm B 0.857
- stability across seeds: min Tucker congruence **0.999**, mean profile SD 0.012
- membership: **25%** of patients have a clear dominant archetype (max weight ≥0.5); mean normalized entropy 0.67 (1 = fully blended — expected on a continuum)

## Archetype profiles — Arm A (full phenotype, z-units; the extreme phenotypes)
|                                                  |   overall_severity |   cognition |   metabolic |   inflammatory |   sleep |   mania_activation |   suicidality |   developmental_risk |   substance |
|:-------------------------------------------------|-------------------:|------------:|------------:|---------------:|--------:|-------------------:|--------------:|---------------------:|------------:|
| A0: ↓overall_severity ↓sleep ↓developmental_risk |              -1.34 |       -0.99 |       -1.14 |          -1.18 |   -1.21 |              -1.14 |         -0.29 |                -1.2  |       -0.79 |
| A1: ↑suicidality ↑developmental_risk ↑metabolic  |              -0.13 |        0.22 |        2.44 |          -0.61 |    0.75 |              -0.13 |          8.13 |                 2.59 |        2.1  |
| A2: ↑cognition ↑overall_severity ↓suicidality    |               2.13 |        2.3  |       -1.16 |          -0.33 |   -1.32 |               0    |         -1.38 |                -0.8  |       -0.28 |
| A3: ↑sleep ↓cognition ↓developmental_risk        |               0.44 |       -1.48 |       -1.1  |          -0.75 |    2.59 |              -0.24 |         -0.42 |                -1.2  |       -0.85 |
| A4: ↑metabolic ↓suicidality ↓developmental_risk  |              -0.63 |        0.75 |        3.74 |          -0.08 |   -0.59 |              -0.38 |         -1.47 |                -1.21 |        0.3  |
| A5: ↑inflammatory ↑substance ↓suicidality        |               0.42 |        0.8  |        0.75 |           6.59 |    0.27 |              -0.19 |         -1.27 |                 0.27 |        2.38 |
| A6: ↑developmental_risk ↓metabolic ↑sleep        |               0.62 |       -0.57 |       -0.86 |          -0.36 |    0.81 |               0.46 |         -0.1  |                 5.1  |       -0.03 |
| A7: ↑mania_activation ↑sleep                     |               0.07 |       -0.01 |       -0.59 |           0.52 |    0.63 |               5.04 |         -0.48 |                 0.31 |       -0.02 |

Population share by dominant archetype: {0: 0.369, 1: 0.015, 2: 0.165, 3: 0.159, 4: 0.132, 5: 0.019, 6: 0.085, 7: 0.055}

## Archetype profiles — Arm B (pure profile, G removed)
|                                                         |   cognition |   metabolic |   inflammatory |   sleep |   mania_activation |   suicidality |   developmental_risk |   substance |
|:--------------------------------------------------------|------------:|------------:|---------------:|--------:|-------------------:|--------------:|---------------------:|------------:|
| B0: ↑sleep ↓cognition ↓substance                        |       -1.3  |       -1.21 |          -0.94 |    2.6  |              -0.19 |         -0.32 |                -1.19 |       -1.28 |
| B1: ↑metabolic ↓suicidality ↓developmental_risk         |       -0.12 |        3.97 |          -0.16 |   -0.1  |              -0.17 |         -1.24 |                -1.08 |        0.28 |
| B2: ↓sleep ↓developmental_risk ↓metabolic               |       -0.65 |       -1.18 |          -0.69 |   -1.49 |              -0.91 |         -0.73 |                -1.36 |       -0.58 |
| B3: ↑suicidality ↑developmental_risk ↑metabolic         |        0.22 |        2.44 |          -0.61 |    0.75 |              -0.13 |          8.13 |                 2.59 |        2.1  |
| B4: ↑cognition ↓sleep ↓metabolic                        |        4.57 |       -0.95 |          -0.91 |   -1.38 |               0.07 |         -0.9  |                -0.35 |       -0.28 |
| B5: ↑inflammatory ↑substance ↓suicidality               |        0.93 |        0.77 |           6.81 |    0.25 |              -0.19 |         -1.49 |                 0.43 |        2.54 |
| B6: ↑developmental_risk ↓metabolic ↓cognition           |       -0.88 |       -0.98 |          -0.64 |    0.72 |               0.12 |         -0.19 |                 4.77 |       -0.33 |
| B7: ↑mania_activation ↑inflammatory ↑developmental_risk |       -0.33 |       -0.85 |           1.02 |    0.79 |               5.77 |          0.24 |                 0.99 |        0.49 |

## Diagnostic composition (Q3 preview — two granularities; validation-only)
By cohort (counts):
|   archetype |   bp |   dr |   sz |
|------------:|-----:|-----:|-----:|
|           0 | 2652 |   56 |  616 |
|           1 |   98 |    5 |   34 |
|           2 |  504 |  222 |  760 |
|           3 | 1100 |  155 |  182 |
|           4 |  755 |   57 |  380 |
|           5 |  102 |   12 |   60 |
|           6 |  612 |   43 |  110 |
|           7 |  429 |    2 |   67 |

By DSM-5 subtype (counts):
| archetype                                    |   Bipolaire de type 1 |   Bipolaire de type 2 |   Bipolaire non spécifié |   Schizophrénie |   Trouble dépressif majeur |   Trouble schizo-affectif |   Trouble schizophréniforme |
|:---------------------------------------------|----------------------:|----------------------:|-------------------------:|----------------:|---------------------------:|--------------------------:|----------------------------:|
| ↑cognition ↑overall_severity ↓suicidality    |                   213 |                   222 |                       69 |             633 |                        222 |                       119 |                           8 |
| ↑developmental_risk ↓metabolic ↑sleep        |                   222 |                   333 |                       57 |              88 |                         43 |                        21 |                           1 |
| ↑inflammatory ↑substance ↓suicidality        |                    50 |                    43 |                        9 |              48 |                         12 |                        12 |                           0 |
| ↑mania_activation ↑sleep                     |                   149 |                   242 |                       38 |              37 |                          2 |                        30 |                           0 |
| ↑metabolic ↓suicidality ↓developmental_risk  |                   391 |                   298 |                       66 |             280 |                         57 |                        93 |                           7 |
| ↑sleep ↓cognition ↓developmental_risk        |                   319 |                   629 |                      152 |             135 |                        155 |                        46 |                           1 |
| ↑suicidality ↑developmental_risk ↑metabolic  |                    42 |                    46 |                       10 |              20 |                          5 |                        14 |                           0 |
| ↓overall_severity ↓sleep ↓developmental_risk |                  1249 |                  1143 |                      260 |             451 |                         56 |                       141 |                          24 |

## Reading
- Archetypes are **corners of the data's convex hull** — extreme phenotypes that span the continuum; most patients are blends (high entropy), consistent with M2.1.
- This is the lead M2 deliverable; the mixture (M2.2) will overlay a soft tessellation. Whether archetypes are **better than DSM-5** is the M4/M5 predictive/treatment head-to-head (§1.7), not decided here.

## Artifacts
- `results/face/m2/archetypes.parquet` — per-patient weights (mean+sd) · dominant · entropy.
- `results/face/m2/archetype_profiles.csv` — both arms' archetype profiles.
- Figures: `docs/figures/23_{scree,profiles,membership}.png`.