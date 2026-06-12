# S5 — global 7-dimension mixed-likelihood fit (the reported map; N=5,000 subsample)

N=5,000 · continuous J=84 + 18 non-Gaussian (binary/ordinal/count). Explicit latents f_e=['overall_severity', 'suicidality', 'developmental_risk']; the continuous specifics are marginalized and coupled to f_e through the shared Φ (conditional decomposition). Observed-cell likelihoods, no imputation.

## Certification — **NOT certified — provisional**
- max R-hat **1.040** · min ESS 73 (structural) · z_e latent min ESS 273 · divergences 0 · Heywood False (gates: R-hat≤1.01, ESS≥400, div=0)

## Inter-dimension correlations Φ (specific block; G orthogonal)
|                    |   cognition |   metabolic |   inflammatory |   sleep |   suicidality |   developmental_risk |
|:-------------------|------------:|------------:|---------------:|--------:|--------------:|---------------------:|
| cognition          |        1    |        0.15 |           0.07 |   -0.09 |         -0.13 |                -0.08 |
| metabolic          |        0.15 |        1    |           0.19 |   -0.03 |         -0.05 |                -0.05 |
| inflammatory       |        0.07 |        0.19 |           1    |    0.02 |          0.02 |                 0.01 |
| sleep              |       -0.09 |       -0.03 |           0.02 |    1    |          0.14 |                 0.19 |
| suicidality        |       -0.13 |       -0.05 |           0.02 |    0.14 |          1    |                 0.23 |
| developmental_risk |       -0.08 |       -0.05 |           0.01 |    0.19 |          0.23 |                 1    |

- mean |off-diagonal| = **0.10**

## Suicidality factor — where its indicators load (home loading · G bifactor)
- isf01: home +2.67 · G +0.33
- isf02: home +2.86 · G +0.37
- isf03: home +3.41 · G +0.52
- isf04: home +3.28 · G +0.44
- isf05: home +3.44 · G +0.56
- isf08: home +1.84 · G +0.12
- isf09: home +1.98 · G +0.34
- isf08a: home +1.74 · G +0.12
- isf09a: home +1.60 · G +0.31

## Developmental-risk factor — non-Gaussian indicators (home · G)
- autneuro_mhoccur: home +0.15 · G +0.00
- epilepsie_mhoccur: home +0.06 · G +0.25
- honeonat: home +0.10 · G +0.06
- mere_structure: home +0.30 · G -0.08
- naisstyp: home +0.03 · G +0.11
- pere_structure: home +0.26 · G -0.05
- traumacra_mhoccur: home +0.20 · G -0.07
- ctq40: home +0.00 · G -0.02
- prembrth: home +0.03 · G -0.04

Artifacts: `reports/04_stage5_loadings.csv` · `04_stage5_phi.csv` · `results/face/stage5/` (per-patient, gitignored).