# 10b — soft-zero vs hard-zero unlikely cross-loadings (P0-05)

The methods doc says `unlikely` cells carry a soft `Normal(0, 0.05)` prior; the engine hard-zeros them. This fits the S2 backbone under both (`prepare(soft_unlikely=…)`) and compares the map.

## Hard-zero vs soft-zero agreement
| seed   |   hard_rhat |   soft_rhat |   min_loading_tucker |   max_abs_dPhi |
|:-------|------------:|------------:|---------------------:|---------------:|
| s1     |        1.02 |        1.02 |               0.9667 |         0.1114 |
| s2     |        1.01 |        1.02 |               0.9688 |         0.0738 |

- **The map is robust to the soft-zero specification:** loading congruence Tucker 0.967 ≥ the 0.95 invariance bar; max |ΔΦ| 0.111 (one Φ cell). The ~980 `unlikely` cells carry little signal — the soft `Normal(0, 0.05)` shrinks them to ≈0 and reproduces the hard-zero map to within the invariance threshold (congruent, not byte-identical). The reported hard-zero fit stands as primary; report the soft-zero arm as a congruent sensitivity and reword the methods (a Bayesian sparse bifactor with selected ESEM windows; the unlikely cells fixed/shrunk to ~0).
