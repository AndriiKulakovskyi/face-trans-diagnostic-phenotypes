# S5 — 9-dimension joint certification (mania + substance integrated)

The full joint map: G + cognition/metabolic/inflammatory/sleep + **mania** (marginalized) + suicidality/developmental + **substance** (explicit; binary SUD + count + Fagerström). N≈2000 cohort-balanced · tune 2000 · draws 1500 · 2 seed(s); rung-3 reparam (every explicit specific →G tightened). Largest-N documented (§3.6).

## Per-seed convergence (§8 battery)
| seed   |   rhat |   ess |   div |   bfmi |
|:-------|-------:|------:|------:|-------:|
| s1     |   1.01 |   149 |     0 |   0.41 |
| s2     |   1.04 |   112 |     0 |   0.44 |

## Cross-seed resample-stability
| pair   |   max_dLoading |   min_tucker |
|:-------|---------------:|-------------:|
| s1–s2  |          0.186 |        0.993 |

## Verdict
**Largest-N documented (§3.6).** Structural R-hat 1.040, min ESS 112, 0 div, BFMI ≥ 0.41. As in the 7-dim S5, the explicit-latent block (now incl. substance) is the mixing limit; point estimates resample-stable, precision provisional. mania + substance are integrated with the rest under one shared Φ.