# stage5_rp — mixed-likelihood global fit (suicidality + developmental + continuous core)

N=2,000 · continuous J=84 + 18 non-Gaussian (binary/ordinal/count). Explicit latents f_e=['overall_severity', 'suicidality', 'developmental_risk']; the continuous specifics are marginalized and coupled to f_e through the shared Φ (conditional decomposition). Observed-cell likelihoods, no imputation.

## Certification — **NOT certified — provisional**
- max R-hat **1.090** · min ESS 47 (structural) · z_e latent min ESS 62 · divergences 0 · Heywood False (gates: R-hat≤1.01, ESS≥400, div=0)

## Inter-dimension correlations Φ (specific block; G orthogonal)
|                    |   cognition |   metabolic |   inflammatory |   sleep |   suicidality |   developmental_risk |
|:-------------------|------------:|------------:|---------------:|--------:|--------------:|---------------------:|
| cognition          |        1    |        0.13 |           0.05 |   -0.06 |         -0.13 |                -0.04 |
| metabolic          |        0.13 |        1    |           0.15 |   -0.06 |         -0.03 |                -0.06 |
| inflammatory       |        0.05 |        0.15 |           1    |   -0.02 |         -0.01 |                 0    |
| sleep              |       -0.06 |       -0.06 |          -0.02 |    1    |          0.16 |                 0.2  |
| suicidality        |       -0.13 |       -0.03 |          -0.01 |    0.16 |          1    |                 0.25 |
| developmental_risk |       -0.04 |       -0.06 |           0    |    0.2  |          0.25 |                 1    |

- mean |off-diagonal| = **0.09**

## Suicidality factor — where its indicators load (home loading · G bifactor)
- isf01: home +2.46 · G +0.18
- isf02: home +2.42 · G +0.26
- isf03: home +2.72 · G +0.37
- isf04: home +2.65 · G +0.26
- isf05: home +2.89 · G +0.37
- isf08: home +1.45 · G +0.14
- isf09: home +2.20 · G -0.02
- isf08a: home +1.37 · G +0.14
- isf09a: home +1.68 · G -0.02

## Developmental-risk factor — non-Gaussian indicators (home · G)
- autneuro_mhoccur: home +0.13 · G +0.01
- epilepsie_mhoccur: home +0.23 · G +0.21
- honeonat: home +0.09 · G -0.05
- mere_structure: home +0.34 · G -0.13
- naisstyp: home +0.10 · G +0.02
- pere_structure: home +0.27 · G -0.03
- traumacra_mhoccur: home +0.29 · G -0.10
- ctq40: home +0.00 · G -0.10
- prembrth: home +0.08 · G +0.11

Artifacts: `reports/04_stage5_rp_loadings.csv` · `04_stage5_rp_phi.csv` · `results/face/stage5_rp/` (per-patient, gitignored).