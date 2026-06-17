# 24b — missingness-artefact test under imbalance-robust metrics (P3-06)

The Q4 "not-an-artefact" check asks whether the **coverage** pattern (per-axis observed-indicator counts)
can predict tessellation membership — if it can, membership is a missingness artefact rather than a
signal. The original test used plain classifier accuracy vs the majority baseline, which is weak under
class imbalance. This re-runs it with **balanced accuracy, macro-F1, log-loss, and a permutation test**
(`face.strata.validation.coverage_artifact`, P3-06).

## Result (coherent-coords tessellation, K=4)

| metric | value | chance / baseline | reading |
|---|---:|---:|---|
| classifier accuracy | 0.246 | 0.323 (majority) | below baseline |
| **balanced accuracy** | **0.222** | 0.250 | **below chance** |
| macro-F1 | 0.206 | — | poor |
| log-loss | 5.58 | — | poor probabilistic fit |
| **permutation p-value** | **1.00** | — | worse than all 30 label-permuted nulls |

## Reading

Under every imbalance-robust metric the coverage pattern predicts membership **no better than chance —
in fact worse** (balanced accuracy below 1/K; permutation p = 1.00). The "membership is not a missingness
artefact" conclusion is upheld and strengthened: the stronger metrics the reviewer requested do not
overturn it. (Coverage `n_obs` is scoring-independent, so this holds for both the old and coherent
coordinates.)
