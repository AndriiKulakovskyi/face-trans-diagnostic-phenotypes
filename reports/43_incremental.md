# 43 — M4.3 incremental validity: does the map beat diagnosis + severity?


Each map representation added on top of the R3y bar; **ΔELPD vs R3y** is the held-out added value (Q1), and the durable-axis **β 94% HDI excluding 0** is the in-sample read. The durable coordinates enter as errors-in-variables (M1 SD propagated). A small/ambiguous ΔELPD with a credibly non-zero coefficient is an honest, reportable result — the biology adds a real but modest signal against a strong autoregressive baseline.

## egf  (N = 2114)

| model         |   elpd_loo |   d_elpd_vs_ref |   se_d_elpd | verdict    |   max_pareto_k |   rhat |
|:--------------|-----------:|----------------:|------------:|:-----------|---------------:|-------:|
| +durable      |   -2559.61 |            7.1  |        4.45 | ambiguous  |           0.57 |   1    |
| +archetypesA  |   -2508.48 |           58.22 |       11.06 | predictive |           0.51 |   1.01 |
| +archetypesB  |   -2520.27 |           46.44 |       10.27 | predictive |           0.47 |   1.01 |
| +tessellation |   -2519.52 |           47.19 |        9.74 | predictive |           0.47 |   1    |
| +specifics8   |   -2509.87 |           56.84 |       10.36 | predictive |           1.01 |   1.17 |

Durable-axis effects (standardized β on the z-scored outcome; EIV, 94% HDI):

| axis         |       mean |     eti_lo |      eti_hi |   p_direction |
|:-------------|-----------:|-----------:|------------:|--------------:|
| cognition    | -0.0217316 | -0.0630405 |  0.0204885  |     0.160625  |
| metabolic    | -0.0618859 | -0.101602  | -0.0220119  |     0.000625  |
| inflammatory | -0.0599875 | -0.111582  | -0.00793849 |     0.0134375 |

Q2 — same axes under the **error-aware G** severity (must survive both):

| axis         |       mean |     eti_lo |     eti_hi |   p_direction |
|:-------------|-----------:|-----------:|-----------:|--------------:|
| cognition    | -0.0225595 | -0.0608886 |  0.0164196 |     0.12875   |
| metabolic    | -0.0541756 | -0.0924973 | -0.0155374 |     0.0028125 |
| inflammatory | -0.0364547 | -0.0855003 |  0.0110841 |     0.079375  |

## cgi_s  (N = 2345)

| model         |   elpd_loo |   d_elpd_vs_ref |   se_d_elpd | verdict    |   max_pareto_k |   rhat |
|:--------------|-----------:|----------------:|------------:|:-----------|---------------:|-------:|
| +durable      |   -3023.1  |           -0.34 |        2.03 | ambiguous  |           0.4  |   1    |
| +archetypesA  |   -3009.79 |           12.98 |        6.65 | ambiguous  |           0.36 |   1    |
| +archetypesB  |   -3007.58 |           15.18 |        6.99 | predictive |           0.39 |   1.01 |
| +tessellation |   -3012.08 |           10.69 |        5.2  | predictive |           0.34 |   1    |
| +specifics8   |   -3006.09 |           16.68 |        7.2  | predictive |           1.17 |   1.03 |

Durable-axis effects (standardized β on the z-scored outcome; EIV, 94% HDI):

| axis         |       mean |      eti_lo |    eti_hi |   p_direction |
|:-------------|-----------:|------------:|----------:|--------------:|
| cognition    | -0.0104823 | -0.0531543  | 0.0321809 |      0.31     |
| metabolic    |  0.0370666 | -0.00195457 | 0.0769096 |      0.960938 |
| inflammatory |  0.0135686 | -0.0363103  | 0.0651393 |      0.68375  |

## Read

- **Representations compared**: continuous durable coords · 8 archetypes **Arm A (full phenotype, includes G)** vs **Arm B (G-residualized, ⊥G)** · 4-region tessellation · the 8-specifics ceiling. Arm A−Arm B gap = how much of the strata's added value is a richer severity profile vs genuinely orthogonal-to-G structure.
- **Q2**: a durable effect is only credited if its HDI excludes 0 under *both* the manifest CGI-S and the error-aware G severity (egf; for cgi_s the two coincide since CGI-S is the baseline outcome).
- Held-out ΔELPD is the honest performance metric; the calibration scatter (in-sample R²) and the added-value bars are in `docs/figures/43_{added_value,calibration}.png`.

## Decision for the gate
Confirm which representations clear Q1/Q2 per outcome before the transdiagnostic / head-to-head-vs-DSM-5 stage (44) and the robustness sweep (46).

Artifacts: `results/face/m4/{incremental_comparison.csv, coef_durable.csv}` · `docs/figures/43_{added_value,calibration}.png`.