# 37 — M3 consolidation: the longitudinal hand-off + axis coherence summary

**M3 complete.** Per-(patient, visit) panel over V0, V1, V2 for **9,013 patients** (16,241 rows) → `results/face/patient_panel.parquet` (the M4 substrate). Axis-level coherence summary below.

## Axis-level M3 summary (the temporal verdict per dimension)
| axis               | g1_license   |   icc_trait | trait_state   |   pop_slide_v0v2 |   reliable_change_rate | durable_for_m4   |
|:-------------------|:-------------|------------:|:--------------|-----------------:|-----------------------:|:-----------------|
| overall_severity   | invariant    |       0.656 | trait         |           -0.344 |                  0.33  | False            |
| cognition          | invariant    |       0.776 | trait         |           -0.157 |                  0.102 | True             |
| metabolic          | invariant    |       0.932 | trait         |            0.102 |                  0.166 | True             |
| inflammatory       | partial      |       0.854 | trait         |            0.049 |                  0.062 | True             |
| sleep              | invariant    |       0.49  | mixed         |           -0.082 |                  0.532 | False            |
| mania_activation   | not-tested   |       0.72  | trait         |           -0.18  |                  0.083 | False            |
| suicidality        | not-tested   |       0.46  | mixed         |           -0.886 |                  0.318 | False            |
| developmental_risk | invariant    |       0.39  | state         |           -0.166 |                  0.088 | False            |
| substance          | not-tested   |       0.999 | uninformative |           -0.074 |                  0.003 | False            |

- **Durable stratify-on axes** (G1-licensed + G3-trait + stable over time — the biology corners worth stratifying / predicting on): **cognition, metabolic, inflammatory**.
- **Spine / monitoring axes** (move over time → track, don't stratify): **severity** (rank-stable but the cohort slides — the spine), suicidality, sleep.
- **Caveats carried forward:** developmental_risk's apparent state is CTQ recall noise (trait by design); inflammatory is partial-invariant; substance is uninformative (signal ≪ noise); mania / suicidality / substance are not G1-tested (explicit block).

## The M3 coherence verdict (G1–G6)
- **G1 (invariance):** the V0 map measures the same constructs at V1/V2 — severity, cognition, metabolic, sleep, developmental invariant; inflammatory partial. The precondition holds.
- **G3 ⟷ G4 (trait/state ⟷ geometry):** both routes agree — biology/cognition are durable (trait, ranks/positions/archetype-identity persist) while severity + symptoms move (state, population slide). The M2 geometry is temporally coherent.
- **G6 (attrition):** dropout is mild/cognition-leaning; verdicts robust to survivorship.
- **Bottom line:** the transdiagnostic map and strata are **temporally coherent** — *stratify on the durable biology, monitor the moving symptoms.* Persists ≠ predicts (M4).

Docs: `docs/TEMPORAL_FINDINGS.md` (paper-facing) · `docs/TEMPORAL_MODEL.md` (methods) · `docs/TEMPORAL_RESULTS.md` (per-stage). Hand-off: `results/face/patient_panel.parquet`.