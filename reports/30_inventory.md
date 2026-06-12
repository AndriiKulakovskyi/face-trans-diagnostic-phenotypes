# 30 — M3.0 longitudinal-coverage inventory (V0 → V1 → V2)

The M3 feasibility gate: retention, per-axis follow-up coverage, and an **empirical** re-administered vs carried-forward split (within-patient V0→V1 change rate; the dictionary's `temporal_scope` is hardcoded `current`, so it cannot be used). No scoring, no imputation.

> **Read the change rate as re-administration, not state.** A high rate means the item was *re-collected with varying answers*, not that the construct moved. The CTQ childhood-trauma items (a fixed history) reach change rates ~0.9 from recall noise alone. This stage decides **coverage** only; **trait vs state is G3** (stage 35), which deconvolves measurement error.

## Retention (unique patients per visit; V0–V2 is the M3 window)
| visit   |   bp |   dr |   sz |   total |
|:--------|-----:|-----:|-----:|--------:|
| V0      | 6252 |  552 | 2209 |    9013 |
| V1      | 3074 |  233 |  963 |    4270 |
| V2      | 2228 |  135 |  595 |    2958 |

Fraction of each cohort's V0 roster retained:
| visit   |    bp |    dr |    sz |   total |
|:--------|------:|------:|------:|--------:|
| V0      | 1     | 1     | 1     |   1     |
| V1      | 0.492 | 0.422 | 0.436 |   0.474 |
| V2      | 0.356 | 0.245 | 0.269 |   0.328 |

- All three cohorts well-represented at V1/V2 (total 4270 / 2958 vs 9013 at V0). Full visit grid in `reports/30_retention.csv`. Attrition is *characterized* in G6 (stage 31), never filled.

## Per-axis follow-up coverage (modeled indicators with ≥30 obs)

`verdict` = coverage only (✅ trackable: ≥1 indicator ≥30 obs at V1 *and* V2 · ⚠️ thin: ≤2 indicators · ⛔ coverage-limited). `n_readministered`/`n_carried` = re-administration split (not trait/state).

| axis               |   n_indicators |   items_ge30_V0 |   items_ge30_V1 |   items_ge30_V2 |   n_readministered |   n_carried |   median_change_rate | verdict   | v   |
|:-------------------|---------------:|----------------:|----------------:|----------------:|-------------------:|------------:|---------------------:|:----------|:----|
| overall_severity   |             14 |              14 |              10 |              12 |                 10 |           0 |                0.732 | trackable | ✅  |
| cognition          |             11 |              11 |              11 |              11 |                 11 |           0 |                0.895 | trackable | ✅  |
| metabolic          |             32 |              32 |              32 |              32 |                 32 |           0 |                0.952 | trackable | ✅  |
| inflammatory       |             14 |              14 |              14 |              14 |                 14 |           0 |                0.793 | trackable | ✅  |
| sleep              |              9 |               9 |               9 |               9 |                  9 |           0 |                0.554 | trackable | ✅  |
| mania_activation   |              2 |               2 |               2 |               2 |                  2 |           0 |                0.691 | thin      | ⚠️  |
| suicidality        |             30 |              28 |              21 |              21 |                 21 |           0 |                0.432 | trackable | ✅  |
| developmental_risk |             23 |              23 |              19 |              18 |                 16 |           1 |                0.221 | trackable | ✅  |
| substance          |              4 |               4 |               4 |               4 |                  4 |           0 |                0.368 | trackable | ✅  |

- Windows (MADRS/QIDS/STAI cross-loaders, no home axis): 3 item(s) [madrs, qidsr120, staya] — inform severity/cognition/sleep via cross-loadings.
- Per-indicator detail (n_obs per visit, change rate, class): `reports/30_indicator_temporal.csv`.

## Feasibility read (gate)
- **Trackable** (fresh follow-up data at V1 *and* V2): overall_severity, cognition, metabolic, inflammatory, sleep, suicidality, developmental_risk, substance.
- **Thin** (≤2 indicators — scored but caveated): mania_activation.
- **Coverage-limited at follow-up**: none.
- **Carried-forward / identical** indicators (cannot inform change): epilepsie_mhoccur — a single indicator; **no axis is carry-forward**, so every axis is scored from its own observed cells at each visit (correcting the earlier 'developmental_risk is static' assumption — its CTQ items are re-administered; G3 will test whether that variation is genuine state or recall noise).

## Decision for the gate
Confirm the V0→V1→V2 window and the per-axis coverage above before building the scoring substrate (stage 32). All trackable axes are scored per visit; `mania_activation` (2 indicators) is scored but flagged thin; trait vs state for every axis is deferred to G3.

Artifacts: `reports/30_{retention,axis_coverage,indicator_temporal}.csv` · `docs/figures/30_{retention,axis_coverage}.png`.