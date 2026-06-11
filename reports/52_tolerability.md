# 52 — M5.2 tolerability + the treatment-resistance atlas

The novel, severity-clean test (map → side-effect burden), the per-archetype resistance / tolerability atlas, and the confirmatory response/resistance checks. Bar = nuisance + DSM-5 arm + baseline CGI-S + error-corrected G. BP/SZ only.

## Does the map predict side-effect burden beyond diagnosis + severity? (the novel test)

| model       |   elpd_loo |   d_elpd_vs_ref |   se_d_elpd | verdict   |
|:------------|-----------:|----------------:|------------:|:----------|
| +archetypes |    -788.48 |            3.41 |        4.39 | ambiguous |
| +durable    |    -790.55 |            1.34 |        2.95 | ambiguous |
| reference   |    -791.89 |            0    |        0    | ambiguous |

Durable-axis effect on side-effects (EIV, 94% HDI) — the metabolic-phenotype × side-effects bet:

- **cognition**: β = +0.150 [-0.000, +0.297]
- **metabolic**: β = +0.143 [+0.008, +0.276]  ← excludes 0
- **inflammatory**: β = +0.030 [-0.147, +0.209]

- Clinical value (side-effects): reference AUC 0.598 → +map 0.605 (ΔAUC 0.007 [-0.015,+0.029]).

## Treatment-resistance / tolerability atlas (per archetype, sorted by side-effects)

| archetype          |    n |   side_effects |   resistance |   response |
|:-------------------|-----:|---------------:|-------------:|-----------:|
| suicidality        |  137 |          0.36  |        0.536 |      0.357 |
| developmental      |  765 |          0.232 |        0.339 |      0.459 |
| inflammatory       |  174 |          0.216 |        0.5   |      0.475 |
| high-sev+cognitive | 1486 |          0.196 |        0.488 |      0.44  |
| metabolic          | 1192 |          0.194 |        0.295 |      0.528 |
| mania/activation   |  498 |          0.182 |        0.319 |      0.484 |
| sleep/circadian    | 1437 |          0.17  |        0.399 |      0.422 |
| low-burden         | 3324 |          0.109 |        0.201 |      0.59  |

## Confirmatory: response & resistance (expected M4-redundant)

| endpoint   |   d_elpd_vs_ref |   se_d_elpd | verdict   |
|:-----------|----------------:|------------:|:----------|
| response   |            3.87 |        4.75 | ambiguous |
| resistance |            3.24 |        4.67 | ambiguous |

- As predicted by the M5.0 severity-confound audit, the response/resistance signals largely restate the M4 prognosis (severity outcomes) — reported honestly, not as new evidence.

## The boundary (what M5 cannot do here) + the data-ask

- M5 stratifies **response / resistance / tolerability** to treatment-as-usual; it **cannot** say which drug to give — FACE records no treatment identity. True treatment **moderation / selection** (the precision-psychiatry payoff) requires linking **prescription / medication or trial-arm data** (a future *M5b*). That data-acquisition check is the program's next step.
- The program's *demonstrated* clinical value culminates at **M4 (prognosis)**; M5 adds a novel tolerability signal and reframes resistance, within the data's limits.

Artifacts: `results/face/m5/{tolerability.csv, response_atlas.csv}` · `docs/figures/52_treatment_atlas.png`.