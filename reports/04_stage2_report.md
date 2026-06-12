# Stage 2 — continuous core + Φ + windows (ESEM) (full-N V0)

N=9,013 · J=71 · factors=['overall_severity', 'cognition', 'metabolic', 'inflammatory', 'sleep']. Marginalized (Woodbury), observed-cell Gaussian likelihood, no imputation. Φ ~ LKJ(2) over specifics (G ⊥ specifics); 8 MADRS/QIDS/STAI window cross-loadings; 0 specific↔specific cross-loadings (metabolic↔inflammatory association carried by Φ).

## Certification — **CERTIFIED**
- max R-hat **1.010** · min ESS 676 · divergences 0 · Heywood False (gates: R-hat≤1.01, ESS≥400, div=0)

## G (functional burden) — anchor loadings
| item     |   loading |
|:---------|----------:|
| fast     |     0.912 |
| fast25   |     0.781 |
| egf      |     0.732 |
| fast27   |     0.712 |
| eq5d     |     0.695 |
| fast30   |     0.694 |
| eq5d0206 |     0.637 |
| cgi01    |     0.622 |
| fast26   |     0.576 |
| fast28   |     0.402 |
| lvsbjind |     0.007 |

## Specific factors — mean primary home loading
| factor       |   loading |
|:-------------|----------:|
| cognition    |      0.58 |
| inflammatory |      0.38 |
| metabolic    |      0.32 |
| sleep        |      0.48 |

## Bifactor — mean |loading on G| of each specific factor's items (G ⊥ biology check)
| home         |   loading |
|:-------------|----------:|
| cognition    |      0.27 |
| inflammatory |      0.07 |
| metabolic    |      0.08 |
| sleep        |      0.26 |

## Inter-dimension correlations Φ (specific block; G orthogonal by construction)
|              |   cognition |   metabolic |   inflammatory |   sleep |
|:-------------|------------:|------------:|---------------:|--------:|
| cognition    |        1    |        0.15 |           0.06 |   -0.09 |
| metabolic    |        0.15 |        1    |           0.2  |   -0.03 |
| inflammatory |        0.06 |        0.2  |           1    |   -0.01 |
| sleep        |       -0.09 |       -0.03 |          -0.01 |    1    |

- mean |off-diagonal| = **0.09**

## MADRS / QIDS / STAI windows — where they land (signed cross-loadings)
| item     | factor           |   loading |
|:---------|:-----------------|----------:|
| madrs    | cognition        |    -0.055 |
| madrs    | overall_severity |     0.802 |
| madrs    | sleep            |     0.139 |
| qidsr120 | cognition        |    -0.052 |
| qidsr120 | overall_severity |     0.766 |
| qidsr120 | sleep            |     0.242 |
| staya    | overall_severity |     0.661 |
| staya    | sleep            |     0.205 |

_Specific↔specific (metabolic↔inflammatory) cross-loadings not freed: rotationally aliased with Φ and not separately identifiable — Φ carries that association (see Φ above)._

Artifacts: `reports/04_stage2_loadings.csv` · `04_stage2_phi.csv` · `results/face/stage2/` (per-patient, gitignored).