# 22 — M2.2 measurement-error mixture (soft TESSELLATION, not biotypes)

Continuum verdict (M2.1) ⇒ this is a **soft tessellation** of the continuum — a discrete decision-region overlay, **not** natural-kind biotypes. Fit by Extreme Deconvolution (x_i ~ Σ_k π_k N(m_k, V_k+S_i)): **propagates the M1 per-patient uncertainty S_i**, so the components are the underlying noise-free cloud and prior-dominated / DR-absent coordinates self-down-weight.

## K = 4 (tessellation granularity; M2.1 uncertainty-aware mode 4 + BIC plateau)
XD BIC over K (no sharp optimum expected on a continuum):
| K | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| BIC | 200,425 | 199,607 | 199,325 | 199,307 | 199,439 | 199,643 | 199,868 |

- membership: **92%** of patients have a confident component (max responsibility ≥0.5); mean normalized entropy 0.42.
- population share by component: {0: np.float64(0.307), 1: np.float64(0.116), 2: np.float64(0.323), 3: np.float64(0.253)}

## Tessellation component profiles (m_k, z-units; higher = more burden)
|                                                  |   overall_severity |   cognition |   metabolic |   inflammatory |   sleep |   mania_activation |   suicidality |   developmental_risk |   substance |
|:-------------------------------------------------|-------------------:|------------:|------------:|---------------:|--------:|-------------------:|--------------:|---------------------:|------------:|
| T0: ↓developmental_risk ↓sleep ↓overall_severity |              -0.42 |        0.07 |       -0.04 |          -0.12 |   -0.55 |              -0.37 |         -0.2  |                -0.75 |       -0.07 |
| T1: ↑mania_activation ↑developmental_risk ↑sleep |               0.01 |       -0.07 |        0.07 |           0.12 |    0.54 |               1.3  |          0.43 |                 1.24 |        0.23 |
| T2: ↑overall_severity ↑metabolic                 |               0.58 |        0.32 |        0.42 |           0.19 |    0.05 |              -0.11 |          0.01 |                -0.07 |        0.23 |
| T3: ↓metabolic ↓cognition                        |              -0.03 |       -0.5  |       -0.57 |          -0.19 |    0.22 |              -0.15 |          0.13 |                 0.23 |       -0.14 |

## Diagnostic composition (validation-only; two granularities)
By cohort:
|   component |   bp |   dr |   sz |
|------------:|-----:|-----:|-----:|
|           0 | 2053 |    6 |  712 |
|           1 |  919 |   13 |  114 |
|           2 | 1470 |  403 | 1039 |
|           3 | 1810 |  130 |  344 |

By DSM-5 subtype:
| component                                    |   Bipolaire de type 1 |   Bipolaire de type 2 |   Bipolaire non spécifié |   Schizophrénie |   Trouble dépressif majeur |   Trouble schizo-affectif |   Trouble schizophréniforme |
|:---------------------------------------------|----------------------:|----------------------:|-------------------------:|----------------:|---------------------------:|--------------------------:|----------------------------:|
| ↑mania_activation ↑developmental_risk ↑sleep |                   326 |                   505 |                       88 |              61 |                         13 |                        52 |                           1 |
| ↑overall_severity ↑metabolic                 |                   630 |                   677 |                      163 |             831 |                        403 |                       204 |                           4 |
| ↓developmental_risk ↓sleep ↓overall_severity |                  1052 |                   819 |                      182 |             530 |                          6 |                       152 |                          30 |
| ↓metabolic ↓cognition                        |                   627 |                   955 |                      228 |             270 |                        130 |                        68 |                           6 |

## Reading
- The components are **regions of a continuum**, not discrete kinds — read with the archetypes (M2.3), which are the lead view. Whether the tessellation is **better than DSM-5** is the M4/M5 predictive/treatment head-to-head (§1.7), set up in M2.4.

## Artifacts
- `results/face/m2/tessellation.parquet` — per-patient soft responsibilities · MAP · entropy.
- `results/face/m2/tessellation_profiles.csv` — component profiles.
- Figures: `docs/figures/22_{bic,profiles,membership}.png`.