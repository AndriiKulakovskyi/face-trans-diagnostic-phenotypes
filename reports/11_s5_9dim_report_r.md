# S5 — 9-dimension joint certification (mania + substance integrated)

The full joint map: G + cognition/metabolic/inflammatory/sleep + **mania** (marginalized) + suicidality/developmental + **substance** (explicit; binary SUD + count + Fagerström). N≈2000 cohort-balanced · tune 2000 · draws 1500 · 2 seed(s); rung-3 reparam (every explicit specific →G tightened). Largest-N documented (§3.6).

## Per-seed convergence (§8 battery)
| seed   |   rhat |   ess |   div |   bfmi |
|:-------|-------:|------:|------:|-------:|
| s1     |   1.55 |     7 |     0 |   0.39 |
| s2     |   1.03 |   144 |     0 |   0.47 |

## Cross-seed resample-stability
| pair   |   max_dLoading |   min_tucker |
|:-------|---------------:|-------------:|
| s1–s2  |          0.274 |        0.995 |

## Verdict
**Largest-N documented (§3.6).** Structural R-hat 1.550, min ESS 7, 0 div, BFMI ≥ 0.39. As in the 7-dim S5, the explicit-latent block (now incl. substance) is the mixing limit; point estimates resample-stable, precision provisional. mania + substance are integrated with the rest under one shared Φ.