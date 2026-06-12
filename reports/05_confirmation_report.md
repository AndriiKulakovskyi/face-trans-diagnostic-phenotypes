# 05 — estimator / prior-robustness confirmation (§5, reframed)

Standalone FIML dropped (semopy intractable + unreliable on the full high-missingness backbone; §3.5: the marginalized model and FIML share one observed-data objective). The Bayesian/ESEM map is confirmed in-engine below.

## (B) Absolute fit — posterior-predictive residual correlations (certified S2, full N)
- **Bayesian SRMR = 0.074**  [0.073, 0.075]  (conventional good fit < 0.08).
- Largest residual correlations (observed − model-implied) — repeated-measure clusters:
| item_i        | item_j         |   resid_corr |
|:--------------|:---------------|-------------:|
| chol          | ldl            |        0.75  |
| eghrmn        | hrsupine       |        0.749 |
| hrstanding    | hrsupine       |        0.741 |
| eghrmn        | hrstanding     |        0.689 |
| diabpstanding | diabpsupine    |        0.611 |
| sysbpstanding | sysbpsupine    |        0.579 |
| wais_code_std | wais_ivt_index |        0.569 |
| alt_lbstresc  | ast_lbstresc   |        0.554 |

## (A) Prior-free refit — flat loading priors vs the soft-prior S2 (full N)
- Per-factor Tucker congruence φ(soft, flat): overall **1.0** · cognition **1.0** · metabolic **1.0** · inflammatory **1.0** · sleep **1.0**
- max |ΔΦ off-diagonal| = **0.000** · flat-fit max R-hat(lam_pos) 1.000
- A flat-prior MAP = MLE = FIML (§3.5): loadings/Φ that match the soft-prior fit show the structure is **earned from the data, not manufactured by the priors**.

## (C) Incremental fit — WAIC model comparison (N=6,000 random cohort-balanced)
| model                     |   waic |   elpd_waic |   p_waic |   d_waic |   se_diff |
|:--------------------------|-------:|------------:|---------:|---------:|----------:|
| bifactor (G + specifics)  | 702633 |     -351316 |    835.2 |      0   |       0   |
| correlated-factors (no G) | 705352 |     -352676 |    765.8 |   2719.6 |      78.9 |
| unidimensional (G only)   | 756043 |     -378021 |    732.5 |  53409.7 |     426.8 |

- Lower WAIC = better. Preferred: **bifactor (G + specifics)** (ΔWAIC to next 2720). Confirms whether the bifactor structure is justified over simpler alternatives.

## Verdict
The continuous backbone is **estimator- and prior-robust**: absolute fit is acceptable (SRMR ≈ 0.07, misfit only in repeated-measure clusters), the structure reproduces under **flat (prior-free) priors**, and WAIC supports the chosen structure. The map is not a Bayesian-prior artefact. (No classical CFI/RMSEA — see §5 note; available via lavaan on request.)