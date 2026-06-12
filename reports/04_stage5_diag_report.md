# stage5_diag — mixed-likelihood global fit (suicidality + developmental + continuous core)

N=2,000 · continuous J=84 + 18 non-Gaussian (binary/ordinal/count). Explicit latents f_e=['overall_severity', 'suicidality', 'developmental_risk']; the continuous specifics are marginalized and coupled to f_e through the shared Φ (conditional decomposition). Observed-cell likelihoods, no imputation.

## Certification — **NOT certified — provisional**
- max R-hat **1.100** · min ESS 30 (structural) · z_e latent min ESS 82 · divergences 0 · Heywood False (gates: R-hat≤1.01, ESS≥400, div=0)

## Inter-dimension correlations Φ (specific block; G orthogonal)
|                    |   cognition |   metabolic |   inflammatory |   sleep |   suicidality |   developmental_risk |
|:-------------------|------------:|------------:|---------------:|--------:|--------------:|---------------------:|
| cognition          |        1    |        0.13 |           0.05 |   -0.07 |         -0.14 |                -0.04 |
| metabolic          |        0.13 |        1    |           0.15 |   -0.06 |         -0.03 |                -0.06 |
| inflammatory       |        0.05 |        0.15 |           1    |   -0.02 |         -0.01 |                 0    |
| sleep              |       -0.07 |       -0.06 |          -0.02 |    1    |          0.16 |                 0.19 |
| suicidality        |       -0.14 |       -0.03 |          -0.01 |    0.16 |          1    |                 0.24 |
| developmental_risk |       -0.04 |       -0.06 |           0    |    0.19 |          0.24 |                 1    |

- mean |off-diagonal| = **0.09**

## Suicidality factor — where its indicators load (home loading · G bifactor)
- isf01: home +2.45 · G +0.25
- isf02: home +2.42 · G +0.32
- isf03: home +2.71 · G +0.44
- isf04: home +2.64 · G +0.33
- isf05: home +2.88 · G +0.45
- isf08: home +1.45 · G +0.17
- isf09: home +2.20 · G +0.03
- isf08a: home +1.36 · G +0.17
- isf09a: home +1.68 · G +0.02

## Developmental-risk factor — non-Gaussian indicators (home · G)
- autneuro_mhoccur: home +0.13 · G +0.02
- epilepsie_mhoccur: home +0.23 · G +0.23
- honeonat: home +0.09 · G -0.04
- mere_structure: home +0.34 · G -0.10
- naisstyp: home +0.09 · G +0.02
- pere_structure: home +0.27 · G -0.01
- traumacra_mhoccur: home +0.30 · G -0.07
- ctq40: home +0.00 · G -0.11
- prembrth: home +0.08 · G +0.12

Artifacts: `reports/04_stage5_diag_loadings.csv` · `04_stage5_diag_phi.csv` · `results/face/stage5_diag/` (per-patient, gitignored).