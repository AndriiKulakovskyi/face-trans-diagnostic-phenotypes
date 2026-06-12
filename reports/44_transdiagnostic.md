# 44 — M4.4 head-to-head vs DSM-5 + transdiagnostic generalization


Does the transdiagnostic map (Arm-B archetypes, ⊥G) beat the **7 DSM-5 subtypes** at predicting the V2 outcome, and does its edge hold *within* each cohort? Dominance reads the asymmetry **map beyond DSM-5 (B−A)** vs **DSM-5 beyond map (B−C)**; generalization is read from the **within-cohort** fits (the honest evidence) + a saturation diagnostic. 'Better' = outcome ELPD, never agreement with DSM-5.

## egf  (N = 2114)  —  **dominance: co-informative**

Dominance contrasts (ΔELPD vs the noted reference):

| contrast                      |   d_elpd |    se | verdict    |
|:------------------------------|---------:|------:|:-----------|
| DSM-5 beyond foundation (A−D) |    28.6  |  8.94 | predictive |
| map beyond foundation (C−D)   |    36.29 |  9.34 | predictive |
| map beyond DSM-5 (B−A)        |    47.18 | 10.3  | predictive |
| DSM-5 beyond map (B−C)        |    39.5  | 10.2  | predictive |

- Both the map and DSM-5 add beyond each other → **co-informative**: the map adds real prognostic value beyond diagnosis+severity (B−A), and diagnosis is **not** redundant (B−C). The map *complements* DSM-5, it does not replace it.
- **Raw showdown** (map-only vs DSM-5-only, no autoregressive baseline): C−A = -35.8 ± 19.3 → **ambiguous** (negative = DSM-5 alone classifies better — expected, since the ⊥G map removes the severity axis that drives the categorical functioning gaps).

**Within-cohort map ΔELPD** (the honest generalization evidence):

| cohort   |    n |   map_d_elpd |     se | verdict    |
|:---------|-----:|-------------:|-------:|:-----------|
| bp       | 1490 |        43.5  |   9.97 | predictive |
| sz       |  519 |        -1.75 |   3.5  | ambiguous  |
| dr       |  105 |       nan    | nan    | too-thin   |

**Why it differs — saturation diagnostic** (OLS; a small foundation R² leaves room for the map, a large one saturates it):

| cohort   |    n |   outcome_sd |   foundation_r2 |   map_dR2 |   arch_spread |   coord_unc |
|:---------|-----:|-------------:|----------------:|----------:|--------------:|------------:|
| bp       | 1490 |         14.6 |           0.168 |     0.056 |         0.135 |        0.35 |
| sz       |  519 |         14.9 |           0.261 |     0.016 |         0.125 |        0.38 |
| dr       |  105 |         19.5 |           0.087 |     0.068 |         0.11  |        0.44 |

- The map's value tracks **residual prognostic uncertainty**, not diagnosis: it adds where the foundation (baseline+severity) is weak (BP, DR — episodic courses) and little where the foundation already saturates the predictable variance (SZ — more baseline-locked). SZ outcome variance and map/coordinate spread are comparable to BP, so the SZ null is **not** a floor effect, narrow range, or noisy coordinates.
- The cohort×map interaction ELPD (-10.6 ± 3.8, not-predictive) does **not** prove homogeneity — it only lacks power to justify 14 cohort-specific parameters on held-out ELPD; the within-cohort fits above are the evidence, and they show the effect is course-dependent.

## cgi_s  (N = 2345)  —  **dominance: co-informative**

Dominance contrasts (ΔELPD vs the noted reference):

| contrast                      |   d_elpd |   se | verdict    |
|:------------------------------|---------:|-----:|:-----------|
| DSM-5 beyond foundation (A−D) |    30.53 | 8.58 | predictive |
| map beyond foundation (C−D)   |    10.44 | 6.15 | ambiguous  |
| map beyond DSM-5 (B−A)        |    15.17 | 6.99 | predictive |
| DSM-5 beyond map (B−C)        |    35.26 | 9.01 | predictive |

- Both the map and DSM-5 add beyond each other → **co-informative**: the map adds real prognostic value beyond diagnosis+severity (B−A), and diagnosis is **not** redundant (B−C). The map *complements* DSM-5, it does not replace it.
- **Raw showdown** (map-only vs DSM-5-only, no autoregressive baseline): C−A = -32.4 ± 14.8 → **not-predictive** (negative = DSM-5 alone classifies better — expected, since the ⊥G map removes the severity axis that drives the categorical functioning gaps).

**Within-cohort map ΔELPD** (the honest generalization evidence):

| cohort   |    n |   map_d_elpd |     se | verdict        |
|:---------|-----:|-------------:|-------:|:---------------|
| bp       | 1694 |        16.61 |   7.06 | predictive     |
| sz       |  544 |        -4.76 |   2.21 | not-predictive |
| dr       |  107 |       nan    | nan    | too-thin       |

**Why it differs — saturation diagnostic** (OLS; a small foundation R² leaves room for the map, a large one saturates it):

| cohort   |    n |   outcome_sd |   foundation_r2 |   map_dR2 |   arch_spread |   coord_unc |
|:---------|-----:|-------------:|----------------:|----------:|--------------:|------------:|
| bp       | 1694 |          1.5 |           0.117 |     0.025 |         0.135 |        0.36 |
| sz       |  544 |          1.2 |           0.206 |     0.008 |         0.125 |        0.38 |
| dr       |  107 |          1.7 |           0.179 |     0.116 |         0.113 |        0.46 |

- The map's value tracks **residual prognostic uncertainty**, not diagnosis: it adds where the foundation (baseline+severity) is weak (BP, DR — episodic courses) and little where the foundation already saturates the predictable variance (SZ — more baseline-locked). SZ outcome variance and map/coordinate spread are comparable to BP, so the SZ null is **not** a floor effect, narrow range, or noisy coordinates.
- The cohort×map interaction ELPD (+0.3 ± 4.9, ambiguous) does **not** prove homogeneity — it only lacks power to justify 14 cohort-specific parameters on held-out ELPD; the within-cohort fits above are the evidence, and they show the effect is course-dependent.

## Read

- **Dominance = co-informative.** The ⊥G map adds real prognostic value beyond the 7 DSM-5 subtypes + severity + baseline; DSM-5 adds beyond the map too. The map is a complementary lens on prognosis, consistent with the project's four-layer design (diagnosis stays metadata).
- **Generalization is course-dependent, not uniformly transdiagnostic.** The map's incremental value is large in BP (and DR) and small in SZ — explained by foundation saturation (SZ functioning/severity is more baseline-determined), not by the map failing in SZ.
- **Honest limitation for the paper**: predictive value is concentrated in the episodic (BP/DR) courses; DR is statistically thin (N≈105, ELPD untestable but the OLS ΔR² agrees with BP); the SZ increment is null on held-out ELPD.

## Decision for the gate
Confirm the co-informative dominance + the course-dependent (saturation) generalization read before the robustness sweep (stage 46: IPW, leave-one-cohort-out, reliability-stratified, RTM).

Artifacts: `results/face/m4/{h2h_dsm5, transdiagnostic, transdiagnostic_percohort, transdiagnostic_saturation}.csv` · `docs/figures/44_{dominance,transdiagnostic}.png`.