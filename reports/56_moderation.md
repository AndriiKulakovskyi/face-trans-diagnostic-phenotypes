# 56 — M5.2b stratum × treatment moderation

Per estimable question × co-primary outcome (functioning EGF; CGI response): the EIV outcome GLM on the propensity common-support sample, stabilized-IPTW + covariate-adjusted (doubly robust). **ATE** = treatment main effect (SD units / log-odds) + **E-value** (confounding sensitivity); **moderation** = the durable-axis × treatment interaction (ΔELPD vs no-interaction; any axis HDI excluding 0).

## ATE + moderation by question × outcome

| question         | outcome      |   n |    ate |   ate_lo |   ate_hi |   e_value |   moderation_d_elpd |   moderation_se | moderation_any_axis   |
|:-----------------|:-------------|----:|-------:|---------:|---------:|----------:|--------------------:|----------------:|:----------------------|
| lithium_bp       | functioning  | 660 | -0.01  |   -0.133 |    0.112 |      1.11 |               -1.35 |            2.39 | False                 |
| lithium_bp       | cgi_response | 631 |  0.095 |   -0.228 |    0.417 |      1.28 |               -2.82 |            1.8  | False                 |
| antipsychotic_bp | functioning  | 700 | -0.24  |   -0.38  |   -0.108 |      1.79 |                4.6  |            4.24 | True                  |
| antipsychotic_bp | cgi_response | 674 | -0.29  |   -0.662 |    0.075 |      1.58 |               -2.17 |            1.87 | False                 |
| clozapine_sz     | functioning  | 513 |  0.023 |   -0.234 |    0.286 |      1.17 |               -3.11 |            0.83 | False                 |
| clozapine_sz     | cgi_response | 488 |  0.342 |   -0.282 |    0.962 |      1.66 |                1.01 |            2.45 | True                  |

## Per-axis moderation coefficients (treat × axis, 94% HDI)

- **lithium_bp · functioning**: cognition +0.052 [-0.090,+0.202]; metabolic +0.072 [-0.066,+0.211]; inflammatory +0.159 [-0.031,+0.341]
- **lithium_bp · cgi_response**: cognition +0.131 [-0.232,+0.507]; metabolic +0.192 [-0.196,+0.586]; inflammatory +0.246 [-0.239,+0.739]
- **antipsychotic_bp · functioning**: cognition +0.011 [-0.134,+0.159]; metabolic -0.151 [-0.267,-0.034]*; inflammatory -0.263 [-0.423,-0.098]*
- **antipsychotic_bp · cgi_response**: cognition -0.181 [-0.599,+0.260]; metabolic +0.097 [-0.236,+0.436]; inflammatory -0.426 [-0.949,+0.075]
- **clozapine_sz · functioning**: cognition -0.058 [-0.355,+0.241]; metabolic -0.018 [-0.284,+0.247]; inflammatory -0.038 [-0.424,+0.343]
- **clozapine_sz · cgi_response**: cognition -0.114 [-0.888,+0.640]; metabolic +0.564 [-0.145,+1.338]; inflammatory -1.313 [-2.467,-0.299]*

(* = HDI excludes 0.)

## Read
- **ATE**: the treatment association after propensity + outcome adjustment; the **E-value** is how strong an unmeasured confounder (on both treatment and outcome) would need to be to null it — small E-values mean the association is fragile to confounding by indication.
- **Moderation** is the M5 question: a credible `treat × axis` interaction (ΔELPD > 2·SE and an axis HDI excluding 0) means the map identifies *who benefits* — over and above the average effect.
- **Honest expectation**: average treatment effects on observational data are confounded (low E-values); the moderation interaction is the cleaner target but is typically underpowered. A null moderation is a legitimate, publishable result.

Artifacts: `results/face/m5/moderation.csv` · `docs/figures/56_moderation.png`.