# 42 — M4.2 reference models (the diagnosis + severity bar)


The strictly-nested ladder per primary outcome (R0 nuisance → R1 +diagnosis → R2 +severity → R3y +baseline outcome), fit on the complete-case V0→V2 sample. ΔELPD is vs **R0** (how much each clinician-available block improves held-out prediction). **R3y is the bar** the transdiagnostic map must beat in stage 43.

## egf  (N = 2114, severity = cgi_s__V0)

| model   |   elpd_loo |   se_elpd |   d_elpd_vs_ref |   se_d_elpd | verdict    |   max_pareto_k |   rhat |
|:--------|-----------:|----------:|----------------:|------------:|:-----------|---------------:|-------:|
| R0      |   -2840.17 |     28.82 |            0    |        0    | ambiguous  |           0.48 |      1 |
| R1      |   -2737.96 |     30.17 |          102.2  |       15.25 | predictive |           0.51 |      1 |
| R2      |   -2653.98 |     31.17 |          186.18 |       20.29 | predictive |           0.45 |      1 |
| R3y     |   -2566.56 |     32.11 |          273.61 |       23.57 | predictive |           0.47 |      1 |

R3y standardized coefficients (top |effect|; outcome z-scored):

| term                          |       mean |     eti_lo |      eti_hi |   p_direction |
|:------------------------------|-----------:|-----------:|------------:|--------------:|
| arm_Schizophrénie             | -0.512998  | -0.640349  | -0.390237   |       0       |
| arm_Trouble schizo-affectif   | -0.37898   | -0.551348  | -0.208907   |       0       |
| arm_Trouble schizophréniforme | -0.37419   | -0.887028  |  0.13633    |       0.082   |
| egf__V0                       |  0.361397  |  0.312598  |  0.409667   |       1       |
| sev::cgi_s__V0                | -0.0545758 | -0.101814  | -0.00697118 |       0.016   |
| arm_Bipolaire non spécifié    |  0.0526854 | -0.0919008 |  0.200341   |       0.75125 |

## cgi_s  (N = 2345, severity = overall_severity__mean)

| model   |   elpd_loo |   se_elpd |   d_elpd_vs_ref |   se_d_elpd | verdict    |   max_pareto_k |   rhat |
|:--------|-----------:|----------:|----------------:|------------:|:-----------|---------------:|-------:|
| R0      |   -3219.41 |     26.61 |            0    |        0    | ambiguous  |           0.34 |      1 |
| R1      |   -3154.84 |     28.2  |           64.56 |       11.42 | predictive |           0.31 |      1 |
| R2      |   -3035.57 |     29.98 |          183.83 |       19.25 | predictive |           0.32 |      1 |
| R3y     |   -3022.42 |     30.46 |          196.98 |       20.23 | predictive |           0.43 |      1 |

R3y standardized coefficients (top |effect|; outcome z-scored):

| term                          |      mean |     eti_lo |     eti_hi |   p_direction |
|:------------------------------|----------:|-----------:|-----------:|--------------:|
| arm_Schizophrénie             |  0.487761 |  0.363745  |  0.615324  |       1       |
| arm_Trouble schizo-affectif   |  0.411445 |  0.23354   |  0.590955  |       1       |
| arm_Trouble schizophréniforme |  0.29449  | -0.225861  |  0.817899  |       0.857   |
| arm_Trouble dépressif majeur  | -0.29336  | -0.507193  | -0.0839492 |       0.00325 |
| sev::overall_severity__mean   |  0.235169 |  0.181822  |  0.287484  |       1       |
| cgi_s__V0                     |  0.146164 |  0.0957222 |  0.196328  |       1       |

## Read

- The ladder shows the **bar**: how well diagnosis + severity + baseline value already predict the V2 outcome. A large R0→R3y ΔELPD that saturates by R3y means the autoregressive baseline carries most of the signal — exactly why stage 43 must beat **R3y**, not R0.
- For **cgi_s**, severity = the G coordinate (CGI-S itself is the baseline outcome at R3y); for **egf**, severity = baseline CGI-S. Both point estimates here; the error-aware G enters at stage 43.
- Convergence (max R-hat, Pareto-k) reported per rung; any rung breaching the gate is re-fit before stage 43 builds on it.

## Decision for the gate
Confirm the reference ladders converged and the R3y bar is established per primary outcome, before adding the durable-coordinate / strata blocks (stage 43).

Artifacts: `results/face/m4/{elpd_reference.csv, coef_reference_*.csv}` · `docs/figures/42_reference.png`.