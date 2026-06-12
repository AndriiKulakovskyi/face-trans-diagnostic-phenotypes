# Stage 1 — continuous-core bifactor (full-N V0)

N=9,013 · J=68 · factors=['overall_severity', 'cognition', 'metabolic', 'inflammatory', 'sleep']. Marginalized (Woodbury), observed-cell Gaussian likelihood, no imputation.

## Certification — **CERTIFIED**
- max R-hat **1.010** · min ESS 1939 · divergences 0 · Heywood False (gates: R-hat≤1.01, ESS≥400, div=0)

## G (functional burden) — anchor loadings
| item     |   loading |
|:---------|----------:|
| fast     |     1.035 |
| fast25   |     0.803 |
| fast27   |     0.763 |
| fast26   |     0.742 |
| fast30   |     0.707 |
| egf      |     0.686 |
| eq5d     |     0.579 |
| cgi01    |     0.543 |
| eq5d0206 |     0.525 |
| fast28   |     0.467 |
| lvsbjind |     0.009 |

## Specific factors — mean primary home loading
| factor       |   loading |
|:-------------|----------:|
| cognition    |      0.57 |
| inflammatory |      0.38 |
| metabolic    |      0.32 |
| sleep        |      0.5  |

## Bifactor — mean |loading on G| of each specific factor's items (G ⊥ biology check)
| f            |   loading |
|:-------------|----------:|
| cognition    |      0.27 |
| inflammatory |      0.07 |
| metabolic    |      0.08 |
| sleep        |      0.22 |

Artifacts: `reports/04_stage{S}_loadings.csv` · `results/face/stage{S}/` (per-patient, gitignored).