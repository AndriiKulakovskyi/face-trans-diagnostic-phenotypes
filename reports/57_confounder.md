# 57 — M5.2c treatment-as-confounder for M4

M4's functioning prognosis re-fit on the **treatment-data subset** (N=1324; bp=718, sz=504, dr=102), WITH vs WITHOUT the harmonized drug-class exposures (antipsychotic/antidepressant/mood-stab/lithium/anxiolytic) as covariates — same sample, so the contrast isolates treatment adjustment. EGF z-scored; durable axes EIV.

## Durable-axis effect on future functioning, with vs without treatment adjustment

| axis         |   beta_no_treat | hdi_no          |   beta_with_treat | hdi_with        | survives   |   attenuation_% |
|:-------------|----------------:|:----------------|------------------:|:----------------|:-----------|----------------:|
| cognition    |          -0.037 | [-0.088,+0.013] |            -0.038 | [-0.089,+0.013] | False      |            -1.2 |
| metabolic    |          -0.051 | [-0.098,-0.003] |            -0.048 | [-0.098,-0.001] | True       |             4.4 |
| inflammatory |          -0.046 | [-0.104,+0.012] |            -0.047 | [-0.106,+0.016] | False      |            -0.7 |

## Read
- **survives** = the durable-axis coefficient's 94% HDI still excludes 0 after adjusting for treatment; **attenuation_%** = how much the point estimate shrank.
- A surviving metabolic / inflammatory effect answers the standing objection: **the map's functional forecast is not merely unmodelled treatment** — it holds controlling for the drug classes the patient was on. (Treatment here is the observed exposure; residual/unmeasured prescribing is bounded by the M5.2b E-values, not eliminated.)

Artifact: `results/face/m5/confounder.csv`.