# Stage 4 — continuous core + Φ + windows (ESEM) (full-N V0)

N=4,000 · J=84 · factors=['overall_severity', 'cognition', 'metabolic', 'inflammatory', 'sleep', 'developmental_risk', 'anhedonia']. Marginalized (Woodbury), observed-cell Gaussian likelihood, no imputation. Φ ~ LKJ(2) over specifics (G ⊥ specifics); 10 MADRS/QIDS/STAI window cross-loadings; 0 specific↔specific cross-loadings (metabolic↔inflammatory association carried by Φ).

## Certification — **NOT certified — provisional**
- max R-hat **1.540** · min ESS 7 · divergences 0 · Heywood False (gates: R-hat≤1.01, ESS≥400, div=0)

## G (functional burden) — anchor loadings
| item     |   loading |
|:---------|----------:|
| fast     |     0.937 |
| fast25   |     0.808 |
| fast27   |     0.745 |
| fast30   |     0.721 |
| egf      |     0.715 |
| eq5d     |     0.687 |
| eq5d0206 |     0.631 |
| cgi01    |     0.609 |
| fast26   |     0.579 |
| fast28   |     0.43  |
| lvsbjind |     0.019 |

## Specific factors — mean primary home loading
| factor             |   loading |
|:-------------------|----------:|
| anhedonia          |      0.42 |
| cognition          |      0.58 |
| developmental_risk |      0.41 |
| inflammatory       |      0.39 |
| metabolic          |      0.32 |
| sleep              |      0.47 |

## Bifactor — mean |loading on G| of each specific factor's items (G ⊥ biology check)
| home               |   loading |
|:-------------------|----------:|
| anhedonia          |      0.61 |
| cognition          |      0.26 |
| developmental_risk |      0.14 |
| inflammatory       |      0.07 |
| metabolic          |      0.09 |
| sleep              |      0.25 |

## Inter-dimension correlations Φ (specific block; G orthogonal by construction)
|                    |   cognition |   metabolic |   inflammatory |   sleep |   developmental_risk |   anhedonia |
|:-------------------|------------:|------------:|---------------:|--------:|---------------------:|------------:|
| cognition          |        1    |        0.16 |           0.06 |   -0.12 |                -0.05 |       -0.03 |
| metabolic          |        0.16 |        1    |           0.2  |   -0.03 |                -0.03 |       -0.01 |
| inflammatory       |        0.06 |        0.2  |           1    |   -0.03 |                -0.01 |        0.01 |
| sleep              |       -0.12 |       -0.03 |          -0.03 |    1    |                 0.17 |        0.11 |
| developmental_risk |       -0.05 |       -0.03 |          -0.01 |    0.17 |                 1    |        0.06 |
| anhedonia          |       -0.03 |       -0.01 |           0.01 |    0.11 |                 0.06 |        1    |

- mean |off-diagonal| = **0.07**

## MADRS / QIDS / STAI windows — where they land (signed cross-loadings)
| item     | factor           |   loading |
|:---------|:-----------------|----------:|
| madrs    | anhedonia        |     0.087 |
| madrs    | cognition        |    -0.031 |
| madrs    | overall_severity |     0.793 |
| madrs    | sleep            |     0.1   |
| qidsr120 | anhedonia        |     0.365 |
| qidsr120 | cognition        |    -0.023 |
| qidsr120 | overall_severity |     0.738 |
| qidsr120 | sleep            |     0.156 |
| staya    | overall_severity |     0.628 |
| staya    | sleep            |     0.196 |

_Specific↔specific (metabolic↔inflammatory) cross-loadings not freed: rotationally aliased with Φ and not separately identifiable — Φ carries that association (see Φ above)._

Artifacts: `reports/04_stage4_loadings.csv` · `04_stage4_phi.csv` · `results/face/stage4/` (per-patient, gitignored).