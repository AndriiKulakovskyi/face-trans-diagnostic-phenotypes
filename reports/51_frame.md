# 51 — M5.1 analysis-frame build

The fixed M4 predictor side + the treatment-response endpoints at the horizon. One row per V0-roster patient; nothing re-scored or imputed.

- **Rows:** 9013 (V0 roster); columns 129.
- **Predictors present:** durable trio (cognition/metabolic/inflammatory mean) for 9013; baseline severity (G + CGI-S) for 8129.
- **Map representations + covariates + IPW** carried from `results/face/m4/analysis_frame.parquet`.

## Endpoint coverage by cohort (non-missing at the horizon)

| endpoint           |   n_total |   bp |   sz |   dr |
|:-------------------|----------:|-----:|-----:|-----:|
| response           |      2179 | 1677 |  502 |    0 |
| therapeutic_effect |      1993 | 1529 |  464 |    0 |
| resistance         |      2158 | 1657 |  501 |    0 |
| side_effects       |      1972 | 1510 |  462 |    0 |
| low_adherence      |      2452 | 1942 |  510 |    0 |

- The CGI response endpoints are **BP/SZ only** (DR `n=0` — no CGI efficacy index); `low_adherence` excludes DR (MARS mis-scaled, M5.0). The modelling sample for the tolerability test is BP/SZ.

## Decision for the gate
Confirm the frame (endpoint coverage, predictor presence) before the tolerability test (stage 52).

Artifact: `results/face/m5/analysis_frame.parquet`.