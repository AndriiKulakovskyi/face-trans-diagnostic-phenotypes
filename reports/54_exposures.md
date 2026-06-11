# 54 — M5.1 harmonized treatment-exposure table

One row per (cohort, patient_id) of common drug-class exposures at **V0** (the moderation baseline), harmonized across the three capture mechanisms (ATC / class-string / lifetime-flag). No imputation — a patient with no medication record at V0 is NaN, not unexposed.

- **Rows:** 9013 ; with a V0 medication record: bp=2571 · sz=1889 · dr=427.
- **Join to the M5 frame** (exposed-or-recorded / frame patients): bp 2571/6252 · sz 1889/2209 · dr 427/552.

## Exposure coverage (n exposed) by class × cohort (V0)

| cohort   | temporality   |   n_with_med_record |   antipsychotic |   antidepressant |   mood_stabilizer |   lithium |   anxiolytic |   clozapine |
|:---------|:--------------|--------------------:|----------------:|-----------------:|------------------:|----------:|-------------:|------------:|
| bp       | lifetime      |                2571 |            1843 |             2183 |              1789 |      1140 |         1718 |           0 |
| sz       | current       |                1889 |             532 |              238 |               140 |        24 |          231 |         180 |
| dr       | current       |                 427 |              83 |              274 |                47 |         0 |           97 |           0 |

## The powered moderation questions

- **lithium-response-in-BP** — BP on lithium 1140 / off 1353; +plasma n=1274
- **clozapine-in-SZ** — SZ on clozapine 180 / off 2029
- **antipsychotic (BP/SZ/DR)** — bp=1843 · sz=532 · dr=83
- **antidepressant (BP/SZ/DR)** — bp=2183 · sz=238 · dr=274

- **Temporality**: SZ/DR are **current** (the V0 medication); BP is **lifetime** (`cmoccur_*`, ever-by-baseline) — the BP exposures are illness-history-confounded and carry the target-trial caveat in the M5.2 design.

## Decision for the gate
Confirm the exposure table (coverage, join rate, the on/off split per question) before the propensity models (M5.2 / scripts 55) + the stratum × treatment moderation (56).

Artifact: `results/face/m5/treatment_exposures.parquet`.