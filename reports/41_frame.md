# 41 — M4.1 analysis-frame build (the EIV substrate)

One row per V0-roster patient: baseline coordinates + per-patient SD (the errors-in-variables predictors), the 8 archetypes + 4-region tessellation, reference covariates, the native baseline & horizon outcomes (+ derived remission/response), and the M3 IPW weights. The aligned predictor draw tensor is persisted alongside. Nothing re-scored, nothing imputed.

## Integrity checks

- **Roster**: 9013 rows, 9013 unique `(cohort, patient_id)` (expected 9,013 — the V0 roster).
- **Predictor draw tensor**: shape (200, 9013, 9) `[draws, patients, axes]`, **0 patients unaligned** (expected 0 — every V0 patient has draws).
- **Alignment QC**: corr(mean(draws), panel mean) = **0.9955** (min across axes, ≈1.0 confirms the tensor is in frame order). The residual per-patient gap matches the 200-draw Monte-Carlo error (median gap ≈ median sd/√200), not a misalignment.
- **Durable-axis completeness** (posterior mean present): cognition 9013, metabolic 9013, inflammatory 9013 — cognition is prior-dominated for the untested patients (wide SD, down-weighted by EIV, not missing).
- **IPW**: `w_retained_V2` present for 9013 patients.

## Outcome coverage in the frame (re-derived — must match M4.0)

| outcome      | role      |   n_V0 |   n_paired_V02 | remission   | response   |
|:-------------|:----------|-------:|---------------:|:------------|:-----------|
| egf          | primary   |   7486 |           2121 | True        | False      |
| cgi_s        | primary   |   8129 |           2345 | True        | True       |
| fast         | secondary |   6188 |           1991 | True        | False      |
| eq5d_vas     | secondary |   5581 |           1393 | False       | False      |
| madrs        | secondary |   6580 |           2176 | True        | True       |
| ymrs         | secondary |   8435 |           2660 | False       | False      |
| psqi         | secondary |   7268 |           2234 | False       | False      |
| cssrs_active | secondary |   1408 |            227 | False       | False      |

- Primary outcomes (egf, cgi_s) carry ≈2,100–2,350 paired V0→V2 rows — the modelling N for the headline incremental test. Binary remission/response columns are derived where the config gives thresholds.

## Frame schema (persisted)
- `results/face/m4/analysis_frame.parquet` — 9013 × 124: ids + `arm`/`cohort`/`age`/`sex`/`education_years`/`siteid_city`; `{axis}__{mean,sd,hdi_lo,hdi_hi,n_obs,reliability}` for the 9 axes; `arch_*` + `tess_*`; `{outcome}__{V0,V1,V2}` + `__remission_/__response_`; `{p,w}_retained_{V1,V2}`.
- `results/face/m4/predictor_draws.npz` — `draws [S, N, 9]` aligned row-for-row to the frame, `dims`, `patient_uid`, `visit`.

## Decision for the gate
Confirm the frame integrity (roster size, zero unaligned draws, alignment gap ≈ 0, paired-N matching M4.0) before fitting the reference models (stage 42).

Artifacts: `results/face/m4/{analysis_frame.parquet, predictor_draws.npz}` · `docs/figures/41_frame.png`.