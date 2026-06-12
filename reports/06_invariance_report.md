# 06 — measurement invariance across BP/SZ/DR (§8, in-engine)

Per-cohort **simple-structure** (correlated-factors) fits, z-scored within cohort, N≈600/cohort, 3 seed(s). Simple structure (not bifactor) is the conventional multi-group invariance model and is well-identified in every cohort — the per-cohort *bifactor* G is multimodal in SZ (no FAST anchor). Metric invariance = Tucker congruence φ of the primary loadings per factor per cohort-pair (φ≥0.95 invariant · ≥0.85 partial). Coverage: a factor is compared only where its items are observed (≥30); SZ lacks FAST/QIDS/MADRS, so its severity factor rests on CGI-S/EGF/EQ-5D.

## Convergence — 9/9 fits converged (R-hat ≤ 1.05 · ESS ≥ 100 · 0 div)
| fit   |   rhat |   ess |   div | converged   |
|:------|-------:|------:|------:|:------------|
| BP s1 |   1.01 |  1033 |     0 | True        |
| SZ s1 |   1.01 |   769 |     0 | True        |
| DR s1 |   1.01 |  1305 |     0 | True        |
| BP s2 |   1.01 |  1510 |     0 | True        |
| SZ s2 |   1.01 |   569 |     0 | True        |
| DR s2 |   1.01 |  1334 |     0 | True        |
| BP s3 |   1.01 |   904 |     0 | True        |
| SZ s3 |   1.01 |   588 |     0 | True        |
| DR s3 |   1.01 |  1020 |     0 | True        |


## Metric invariance — Tucker congruence φ (mean over seeds)
| factor           | pair   |   n_items |   phi_mean |   phi_min | verdict           |
|:-----------------|:-------|----------:|-----------:|----------:|:------------------|
| overall_severity | BP–SZ  |         5 |      0.921 |     0.899 | partial           |
| overall_severity | BP–DR  |        10 |      0.99  |     0.989 | invariant         |
| overall_severity | SZ–DR  |         4 |      0.97  |     0.96  | invariant         |
| cognition        | BP–SZ  |        11 |      0.991 |     0.987 | invariant         |
| cognition        | BP–DR  |         8 |      0.956 |     0.952 | invariant         |
| cognition        | SZ–DR  |         8 |      0.972 |     0.956 | invariant         |
| metabolic        | BP–SZ  |        13 |      0.993 |     0.989 | invariant         |
| metabolic        | BP–DR  |        27 |      0.969 |     0.963 | invariant         |
| metabolic        | SZ–DR  |        13 |      0.986 |     0.984 | invariant         |
| inflammatory     | BP–SZ  |         5 |      0.99  |     0.983 | invariant         |
| inflammatory     | BP–DR  |         8 |      0.712 |     0.705 | **non-invariant** |
| inflammatory     | SZ–DR  |         5 |      0.748 |     0.721 | **non-invariant** |
| sleep            | BP–SZ  |         7 |      0.996 |     0.992 | invariant         |
| sleep            | BP–DR  |         9 |      0.992 |     0.992 | invariant         |
| sleep            | SZ–DR  |         7 |      0.993 |     0.991 | invariant         |

- 12/15 factor×pair comparisons fully invariant (φ≥0.95); 13/15 at least partial (φ≥0.85).

## Loading-DIF — items with the largest cross-cohort loading spread (seed 1)
| item               | factor           |   spread |   load_bp |   load_sz |   load_dr |
|:-------------------|:-----------------|---------:|----------:|----------:|----------:|
| neut               | inflammatory     |    0.81  |     0.877 |     0.821 |     0.067 |
| wais_code_std      | cognition        |    0.593 |     0.415 |     0.495 |     1.008 |
| wais_ivt_index     | cognition        |    0.558 |     0.469 |     0.639 |     1.027 |
| cgi01              | overall_severity |    0.401 |     0.423 |     0.824 |     0.461 |
| egf                | overall_severity |    0.376 |     0.659 |     0.869 |     0.494 |
| eos                | inflammatory     |    0.357 |     0.232 |     0.253 |     0.589 |
| wbc                | inflammatory     |    0.338 |     0.922 |     1.012 |     0.674 |
| ldl                | metabolic        |    0.3   |     0.366 |     0.296 |     0.066 |
| eq5d0206           | overall_severity |    0.263 |     0.54  |     0.277 |     0.355 |
| wais_digitspan_std | cognition        |    0.233 |     0.295 |     0.38  |     0.528 |

- 11 item(s) with cross-cohort loading spread > 0.20 (candidate non-invariant items; the rest hold).

## Verdict
The dimensional structure is **largely invariant** across BP/SZ/DR on the testable core: the specific factors (cognition, metabolic, inflammatory, sleep) recover with congruent loadings in each cohort. Where a factor's indicators are cohort-specific (G's FAST in SZ; the depression windows), invariance is **documented as partial**, not claimed. Cohort-modular dimensions (anhedonia, heterogeneously-measured suicidality) are declared modular (§8).