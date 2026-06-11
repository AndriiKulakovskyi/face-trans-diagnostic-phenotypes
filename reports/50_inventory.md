# 50 — M5.0 treatment-response inventory (feasibility + circularity + severity-confound)

What treatment-response signal is available, what endpoints it yields, and how badly each is confounded with baseline severity (the hazard that decides M5). M5 = response *heterogeneity* (stratify response to treatment-as-usual), not treatment *selection* — TAU is unobserved. No model, no imputation.

## Response-signal coverage (raw harmonized layer, by visit)

| signal   |   n_V0 |   n_V1 |   n_V2 |   V2_bp |   V2_sz |   V2_dr |
|:---------|-------:|-------:|-------:|--------:|--------:|--------:|
| cgi02    |   2456 |   3404 |   2319 |    1782 |     537 |       0 |
| cgi03a   |   2571 |   3234 |   2209 |    1688 |     521 |       0 |
| cgi03b   |   1643 |   2902 |   1972 |    1510 |     462 |       0 |
| cgi01    |   8129 |   3799 |   2531 |    1862 |     559 |     110 |
| mars     |   7909 |   3720 |   2551 |    1942 |     510 |      99 |

- The signals are present at follow-up (the modelling visits) but absent from the processed tables — stage 51 extracts them into the M5 frame.
- **Data QC (gate catches):** DR has **no CGI efficacy index** at V2 (cgi02/03a/03b $n=0/0/0$) → the response / therapeutic_effect / resistance / side_effects endpoints are **BP/SZ only** (DR generalization untestable, as for the M4 two-cohort outcomes). And **DR MARS is mis-scaled** (mean 2.8 vs BP 7.7 on 0–10) → the DR low\_adherence rate is a harmonization artefact; **exclude DR from the adherence endpoint** pending a data-layer fix.

## Endpoint prevalence at V2 (overall + by cohort)

| endpoint           | polarity   | role      |    n |   rate |   rate_bp |   rate_sz |   rate_dr |
|:-------------------|:-----------|:----------|-----:|-------:|----------:|----------:|----------:|
| response           | good       | primary   | 2179 |  0.516 |     0.519 |     0.504 |   nan     |
| therapeutic_effect | good       | primary   | 1993 |  0.856 |     0.852 |     0.871 |   nan     |
| resistance         | poor       | primary   | 2158 |  0.307 |     0.267 |     0.439 |   nan     |
| side_effects       | poor       | primary   | 1972 |  0.159 |     0.149 |     0.193 |   nan     |
| low_adherence      | poor       | secondary | 2551 |  0.159 |     0.116 |     0.18  |     0.889 |

- Definitions: response `cgi02∈{1,2}`; therapeutic_effect `cgi03a∈{1,2}`; resistance `cgi01≥4 & cgi02≥3`; side_effects `cgi03b≥3`; low_adherence `mars≤5`.

## Circularity audit — are response signals M1 map indicators?

- **Clean (not map indicators, no overlap):** cgi02, cgi03a, cgi03b, mars.
- **In the map:** cgi01 — `cgi01` (CGI-S) is the G anchor, so it enters M5 only as the *severity adjustment* and inside the *resistance* definition (which is severity-entangled by construction, see below); it is never a credited response predictor.

## Severity-confound audit (the make-or-break for Q2)

Correlation of each endpoint with **baseline** CGI-S, and its rate in the low- vs high-baseline-severity tertile. A large gap means baseline severity drives the endpoint — so the map must beat a diagnosis+severity bar (R2/R3), not raw prevalence.

| endpoint           |    n | polarity   |   corr_baseline_cgis |   rate_lowsev |   rate_highsev |
|:-------------------|-----:|:-----------|---------------------:|--------------:|---------------:|
| response           | 2023 | good       |               -0.081 |         0.556 |          0.364 |
| therapeutic_effect | 1855 | good       |               -0.066 |         0.88  |          0.735 |
| resistance         | 2002 | poor       |                0.267 |         0.181 |          0.609 |
| side_effects       | 1836 | poor       |                0.081 |         0.125 |          0.228 |
| low_adherence      | 2334 | poor       |                0.177 |         0.096 |          0.331 |

- **Most severity-confounded** (|corr| ≥ 0.2): resistance — `resistance` is confounded by design (it contains CGI-S). These set the bar's burden; tolerability/adherence should be the *least* severity-driven (cleaner map tests).

## Decision for the gate
Confirm the endpoint set + the severity-confound profile before building the M5 frame (stage 51). The beyond-severity gate (Q2) is the milestone's crux; M5.0 shows exactly how much work it must do.

Artifacts: `reports/50_{signal_coverage,endpoint_prevalence,severity_confound}.csv` · `docs/figures/50_response_inventory.png`.