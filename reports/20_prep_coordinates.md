# 20 — M2.0 prep: full-N 9-dim coordinates + validation table

All **9,013** patients now carry **all 9** dimensions with uncertainty (M1 left suicidality/developmental/substance on the ~1,884 fit subsample). **Coherent joint scoring** (P2-01/02/04): every 9D draw comes from ONE model state — the explicit-block latents f_e (incl the explicit-block G) plus the marginalized specifics f_m conditioned on that same f_e under the shared Phi; full-N projection under fixed certified parameters (no re-fit, no imputation). Exports the joint draws AND the full per-patient covariance S_i.

**Projection sampler:** R-hat(z_e) max **1.041** · divergences **0** · draws 800. Runtime 0.6 min.

## QC — projection reproduces the certified f_e on the fit subsample (Pearson r)
|                    |     r |
|:-------------------|------:|
| overall_severity   | 1     |
| suicidality        | 1     |
| developmental_risk | 1     |
| substance          | 0.999 |

Ordinal re-coding to the certified categories (top-category absorption): ctq40=0, isf08a=3, prembrth=0 patients re-mapped.

## Reliability — patients per tier, by dimension
(well = ≥3 observed home indicators · partial = 1–2 · prior-dominated = 0)
|                    |   well |   partial |   prior-dominated |
|:-------------------|-------:|----------:|------------------:|
| overall_severity   |   8606 |       249 |               158 |
| cognition          |   6451 |        56 |              2506 |
| metabolic          |   8515 |        67 |               431 |
| inflammatory       |   7102 |       227 |              1684 |
| sleep              |   7522 |       122 |              1369 |
| mania_activation   |      0 |      8594 |               419 |
| suicidality        |   8216 |         3 |               794 |
| developmental_risk |   8153 |       649 |               211 |
| substance          |   3382 |      5269 |               362 |

## Dimension summary (posterior-mean coordinate, z-scored, higher = more burden)
|                    |   mean |   sd_across_patients |   mean_posterior_SD |
|:-------------------|-------:|---------------------:|--------------------:|
| overall_severity   |   0.09 |                 0.82 |                0.29 |
| cognition          |  -0.01 |                 0.75 |                0.44 |
| metabolic          |  -0.01 |                 0.93 |                0.27 |
| inflammatory       |  -0    |                 0.77 |                0.54 |
| sleep              |  -0.01 |                 0.9  |                0.28 |
| mania_activation   |  -0.01 |                 0.83 |                0.66 |
| suicidality        |   0.03 |                 0.96 |                0.48 |
| developmental_risk |  -0.01 |                 1.07 |                0.16 |
| substance          |  -0    |                 0.58 |                0.8  |

## Validation table (validation-only; never a clustering input)
- columns: ['age', 'sex', 'education_years', 'siteid_city', 'arm'] · rows 9,013
- coverage: age 9006, sex 9013, education_years 5336, siteid_city 9013, arm 9013

## Artifacts (results/face/m2/, gitignored)
- `coordinates_full.parquet` — per-patient 9-dim mean/SD/HDI/n_obs/reliability (the M2 input).
- `coordinates_draws.npz` — [200, 9013, 9] coherent joint posterior draws (archetypes-over-draws / structure gate).
- `coordinates_cov.npz` — [9013, 9, 9] full per-patient covariance S_i (coherent; the full-S_i XD arm, P2-04).
- `validation_table.parquet` — cohort/arm/age/sex/education/site.
- `proj.npz` — raw explicit projection (mean/sd/draws).

Figure: `docs/figures/20_coverage.png`.