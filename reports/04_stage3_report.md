# Stage 3 — continuous core + Φ + windows (ESEM) (full-N V0)

N=4,000 · J=83 · factors=['overall_severity', 'cognition', 'metabolic', 'inflammatory', 'sleep', 'developmental_risk']. Marginalized (Woodbury), observed-cell Gaussian likelihood, no imputation. Φ ~ LKJ(2) over specifics (G ⊥ specifics); 8 MADRS/QIDS/STAI window cross-loadings; 0 specific↔specific cross-loadings (metabolic↔inflammatory association carried by Φ).

## Certification — **CERTIFIED**
- max R-hat **1.010** · min ESS 832 · divergences 0 · Heywood False (gates: R-hat≤1.01, ESS≥400, div=0)

## G (functional burden) — anchor loadings
| item     |   loading |
|:---------|----------:|
| fast     |     0.92  |
| fast25   |     0.798 |
| fast27   |     0.737 |
| egf      |     0.718 |
| fast30   |     0.712 |
| eq5d     |     0.701 |
| eq5d0206 |     0.643 |
| cgi01    |     0.617 |
| fast26   |     0.557 |
| fast28   |     0.42  |
| lvsbjind |     0.018 |

## Specific factors — mean primary home loading
| factor             |   loading |
|:-------------------|----------:|
| cognition          |      0.58 |
| developmental_risk |      0.41 |
| inflammatory       |      0.39 |
| metabolic          |      0.32 |
| sleep              |      0.47 |

## Bifactor — mean |loading on G| of each specific factor's items (G ⊥ biology check)
| home               |   loading |
|:-------------------|----------:|
| cognition          |      0.26 |
| developmental_risk |      0.14 |
| inflammatory       |      0.07 |
| metabolic          |      0.09 |
| sleep              |      0.26 |

## Inter-dimension correlations Φ (specific block; G orthogonal by construction)
|                    |   cognition |   metabolic |   inflammatory |   sleep |   developmental_risk |
|:-------------------|------------:|------------:|---------------:|--------:|---------------------:|
| cognition          |        1    |        0.17 |           0.07 |   -0.12 |                -0.06 |
| metabolic          |        0.17 |        1    |           0.21 |   -0.03 |                -0.03 |
| inflammatory       |        0.07 |        0.21 |           1    |   -0.03 |                -0.01 |
| sleep              |       -0.12 |       -0.03 |          -0.03 |    1    |                 0.16 |
| developmental_risk |       -0.06 |       -0.03 |          -0.01 |    0.16 |                 1    |

- mean |off-diagonal| = **0.09**

## MADRS / QIDS / STAI windows — where they land (signed cross-loadings)
| item     | factor           |   loading |
|:---------|:-----------------|----------:|
| madrs    | cognition        |    -0.054 |
| madrs    | overall_severity |     0.814 |
| madrs    | sleep            |     0.13  |
| qidsr120 | cognition        |    -0.05  |
| qidsr120 | overall_severity |     0.759 |
| qidsr120 | sleep            |     0.221 |
| staya    | overall_severity |     0.649 |
| staya    | sleep            |     0.184 |

_Specific↔specific (metabolic↔inflammatory) cross-loadings not freed: rotationally aliased with Φ and not separately identifiable — Φ carries that association (see Φ above)._

Artifacts: `reports/04_stage3_loadings.csv` · `04_stage3_phi.csv` · `results/face/stage3/` (per-patient, gitignored).