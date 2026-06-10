# 43 — M4.3 incremental validity: does the map beat diagnosis + severity?


Each map representation added on top of the R3y bar; **ΔELPD vs R3y** is the held-out added value (Q1), and the durable-axis **β 94% HDI excluding 0** is the in-sample read. The durable coordinates enter as errors-in-variables (M1 SD propagated). A small/ambiguous ΔELPD with a credibly non-zero coefficient is an honest, reportable result — the biology adds a real but modest signal against a strong autoregressive baseline.

## egf  (N = 2114)

| model         |   elpd_loo |   d_elpd_vs_ref |   se_d_elpd | verdict    |   max_pareto_k |   rhat |
|:--------------|-----------:|----------------:|------------:|:-----------|---------------:|-------:|
| +durable      |   -2559.95 |            6.73 |        4.48 | ambiguous  |           0.5  |   1.01 |
| +archetypes   |   -2508.32 |           58.36 |       11.06 | predictive |           0.41 |   1.01 |
| +tessellation |   -2519.46 |           47.23 |        9.71 | predictive |           0.4  |   1.01 |
| +specifics8   |   -2512.84 |           53.84 |       10.47 | predictive |           0.99 |   1.17 |

Durable-axis effects (standardized β on the z-scored outcome; EIV, 94% HDI):

| axis         |       mean |    eti_lo |     eti_hi |   p_direction |
|:-------------|-----------:|----------:|-----------:|--------------:|
| cognition    | -0.0221474 | -0.06279  |  0.0214441 |     0.16      |
| metabolic    | -0.0621253 | -0.102671 | -0.0224146 |     0.0021875 |
| inflammatory | -0.0600885 | -0.111743 | -0.0110207 |     0.00875   |

Q2 — same axes under the **error-aware G** severity (must survive both):

| axis         |       mean |     eti_lo |     eti_hi |   p_direction |
|:-------------|-----------:|-----------:|-----------:|--------------:|
| cognition    | -0.0222691 | -0.0628624 |  0.018211  |      0.140313 |
| metabolic    | -0.054566  | -0.0950406 | -0.0153099 |      0.004375 |
| inflammatory | -0.035008  | -0.0854889 |  0.0126297 |      0.091875 |

## cgi_s  (N = 2345)

| model         |   elpd_loo |   d_elpd_vs_ref |   se_d_elpd | verdict    |   max_pareto_k |   rhat |
|:--------------|-----------:|----------------:|------------:|:-----------|---------------:|-------:|
| +durable      |   -3023.48 |           -0.96 |        2.04 | ambiguous  |           0.41 |   1    |
| +archetypes   |   -3009.69 |           12.83 |        6.65 | ambiguous  |           0.33 |   1    |
| +tessellation |   -3012.3  |           10.22 |        5.23 | ambiguous  |           0.3  |   1    |
| +specifics8   |   -3005.26 |           17.26 |        7.19 | predictive |           0.85 |   1.12 |

Durable-axis effects (standardized β on the z-scored outcome; EIV, 94% HDI):

| axis         |       mean |      eti_lo |    eti_hi |   p_direction |
|:-------------|-----------:|------------:|----------:|--------------:|
| cognition    | -0.0106299 | -0.0525611  | 0.033048  |      0.321875 |
| metabolic    |  0.0366884 | -0.00346501 | 0.0769195 |      0.957187 |
| inflammatory |  0.0147841 | -0.0365108  | 0.0656091 |      0.710938 |

## Read

- **Representations compared**: continuous durable coords vs the 8 archetypes vs the 4-region tessellation vs the 8-specifics ceiling — which carries predictive value, and whether the deployable archetypes retain it.
- **Q2**: a durable effect is only credited if its HDI excludes 0 under *both* the manifest CGI-S and the error-aware G severity (egf; for cgi_s the two coincide since CGI-S is the baseline outcome).
- Held-out ΔELPD is the honest performance metric; the calibration scatter (in-sample R²) and the added-value bars are in `docs/figures/43_{added_value,calibration}.png`.

## Decision for the gate
Confirm which representations clear Q1/Q2 per outcome before the transdiagnostic / head-to-head-vs-DSM-5 stage (44) and the robustness sweep (46).

Artifacts: `results/face/m4/{incremental_comparison.csv, coef_durable.csv}` · `docs/figures/43_{added_value,calibration}.png`.