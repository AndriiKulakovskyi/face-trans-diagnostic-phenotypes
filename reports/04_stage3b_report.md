# Stage 3b — mixed-likelihood suicidality + developmental (full-N V0)

N=4,000 · continuous J=84 + 18 non-Gaussian (binary/ordinal/count). Explicit latents f_e=['overall_severity', 'suicidality', 'developmental_risk']; the continuous specifics are marginalized and coupled to f_e through the shared Φ (conditional decomposition). Observed-cell likelihoods, no imputation.

## Certification — **NOT certified — provisional**
- max R-hat **1.060** · min ESS 58 (structural) · z_e latent min ESS 194 · divergences 0 · Heywood False (gates: R-hat≤1.01, ESS≥400, div=0)

## Inter-dimension correlations Φ (specific block; G orthogonal)
|                    |   cognition |   metabolic |   inflammatory |   sleep |   suicidality |   developmental_risk |
|:-------------------|------------:|------------:|---------------:|--------:|--------------:|---------------------:|
| cognition          |        1    |        0.16 |           0.07 |   -0.12 |         -0.13 |                -0.06 |
| metabolic          |        0.16 |        1    |           0.21 |   -0.03 |         -0.09 |                -0.02 |
| inflammatory       |        0.07 |        0.21 |           1    |   -0.03 |         -0.01 |                -0.01 |
| sleep              |       -0.12 |       -0.03 |          -0.03 |    1    |          0.11 |                 0.16 |
| suicidality        |       -0.13 |       -0.09 |          -0.01 |    0.11 |          1    |                 0.22 |
| developmental_risk |       -0.06 |       -0.02 |          -0.01 |    0.16 |          0.22 |                 1    |

- mean |off-diagonal| = **0.10**

## Suicidality factor — where its indicators load (home loading · G bifactor)
- isf01: home +2.51 · G +0.46
- isf02: home +2.69 · G +0.39
- isf03: home +3.27 · G +0.56
- isf04: home +3.13 · G +0.50
- isf05: home +3.22 · G +0.52
- isf08: home +1.77 · G +0.00
- isf09: home +1.90 · G +0.28
- isf08a: home +1.70 · G -0.01
- isf09a: home +1.54 · G +0.22

## Developmental-risk factor — non-Gaussian indicators (home · G)
- autneuro_mhoccur: home +0.14 · G +0.17
- epilepsie_mhoccur: home +0.07 · G +0.29
- honeonat: home +0.09 · G +0.08
- mere_structure: home +0.30 · G -0.07
- naisstyp: home +0.03 · G +0.03
- pere_structure: home +0.21 · G -0.04
- traumacra_mhoccur: home +0.21 · G -0.03
- ctq40: home +0.00 · G -0.12
- prembrth: home +0.03 · G -0.05

Artifacts: `reports/04_stage3b_loadings.csv` · `04_stage3b_phi.csv` · `results/face/stage3b/` (per-patient, gitignored).