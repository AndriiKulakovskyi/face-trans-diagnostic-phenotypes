# S5 certification — reported 7-dimension map (multi-seed)

N≈2000 cohort-balanced · tune 2000 · draws 1500 · ta 0.9 · 2 seed(s). §4.4 rung-3 reparam on (dev/suic→G tightened; biology→G free). The reported map is the global mixed fit; only it is interpreted (§4.3).

## Per-seed convergence — §8 sampler battery
| seed   |   rhat |   ess |   div |   bfmi |
|:-------|-------:|------:|------:|-------:|
| s1     |   1.03 |   158 |     0 |    0.4 |
| s2     |   1.03 |   114 |     0 |    0.4 |

## Cross-seed resample-stability (point estimates)
| pair   |   max_dLoading |   max_dPhi |   min_tucker |
|:-------|---------------:|-----------:|-------------:|
| s1–s2  |          0.184 |       0.05 |        0.993 |

- Small |ΔΛ| / |ΔΦ| + high Tucker φ ⇒ the reported loadings/Φ are **resample-stable** even where the suic~dev Φ *precision* (ESS) is the documented limit.

## Verdict
**Largest-N documented (§3.6).** Structural R-hat 1.030, min ESS 114, 0 div, BFMI ≥ 0.40. The biology→G estimand and the continuous backbone mix well; the **suicidality~developmental Φ + explicit-latent coupling** is the limiting block — its point estimates are resample-stable, its precision provisional (as flagged in RESULTS/§5). The reported map's loadings and Φ are read from these stable point estimates.
