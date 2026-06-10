# 40 — M4.0 prognostic-outcome inventory (feasibility + the added-value audit)

What is actually predictable, and where the map can legitimately claim *added value*. The effective modelling sample for an *incremental-beyond-baseline* test is **n_paired_V0V2** (both the V0 baseline *and* the V2 horizon observed — ANCOVA needs the pair). We forecast every state axis; the question is whether the durable biology adds value beyond today's value. No model, no imputation.

## Outcome registry & coverage

Parsed 8 outcomes from `configs/m4_outcomes.yaml` (primary: egf, cgi_s). Skipped (absent from the harmonized tables): mars. `n_paired_V0V2` is the headline modelling N; `paired_{bp,sz,dr}` drive the within-cohort (Q3) feasibility.

| outcome      | source_var   | family    | role      | cohort_scope   |   n_V0 |   n_paired_V0V1 |   n_paired_V0V2 |   paired_bp |   paired_sz |   paired_dr |
|:-------------|:-------------|:----------|:----------|:---------------|-------:|----------------:|----------------:|------------:|------------:|------------:|
| egf          | egf          | gaussian  | primary   | bp+sz+dr       |   7486 |            3196 |            2121 |        1495 |         520 |         106 |
| cgi_s        | cgi01        | gaussian  | primary   | bp+sz+dr       |   8129 |            3540 |            2345 |        1694 |         544 |         107 |
| fast         | fast         | gaussian  | secondary | bp+dr          |   6188 |            2669 |            1991 |        1920 |           0 |          71 |
| eq5d_vas     | eq5d0206     | gaussian  | secondary | bp+sz+dr       |   5581 |            2234 |            1393 |        1005 |         269 |         119 |
| madrs        | madrs        | gaussian  | secondary | bp+dr          |   6580 |            3092 |            2176 |        2052 |           0 |         124 |
| ymrs         | ymrs         | gaussian  | secondary | bp+sz+dr       |   8435 |            3852 |            2660 |        2034 |         532 |          94 |
| psqi         | psqi         | gaussian  | secondary | bp+sz+dr       |   7268 |            3223 |            2234 |        1911 |         254 |          69 |
| cssrs_active | cssrs01      | bernoulli | secondary | bp+dr          |   1408 |             469 |             227 |         123 |           0 |         104 |

- **Primary horizon V2** (2-yr); **replication V1** (1-yr, larger N). Retention thins V0 9,013 → V1 4,270 → V2 2,958; attrition is mild/MAR-given-V0 (M3 G6), corrected by IPW at the modelling stages, never by imputation.
- **SZ has no FAST / MADRS / C-SSRS at follow-up** (`paired_sz = 0`) — those outcomes are BP/DR only; their Q3 reduces to BP-vs-DR and the SZ generalization is explicitly untested.
- **DR is thin at V2** (paired ~ a hundred even on the 3-cohort outcomes) — DR-specific verdicts will be documented-partial.

## Added-value audit — predictor axis ↔ outcome item overlap

Each (axis, outcome) is classified by how the outcome's own item enters that axis in the M1 loading matrix: **defines** (primary / g_anchor → the axis is built from the outcome → that axis is *autoregressive* for this outcome), **cross** (plausible cross-loading), **pinned** (`g_anchor_on_specific`, a soft-zero at prior_sd 0.001 → negligible), **soft0** (unlikely_cross), **none**.

**Autoregressive pairs — the baseline bar, NOT credited as added value:**
- `egf` ← overall_severity
- `cgi_s` ← overall_severity
- `fast` ← overall_severity
- `eq5d_vas` ← overall_severity
- `ymrs` ← mania_activation
- `psqi` ← sleep
- `cssrs_active` ← suicidality

These are not forbidden — we *do* forecast each of these outcomes. But when a dimension is built from the outcome's own items, its contribution enters as the **autoregressive baseline `Y_V0` (R3y), the bar to beat**, and is never reported as the transdiagnostic map earning its keep (that would be a trivial self-prediction).

**The added-value test the milestone hinges on (clean / cross-construct):** the durable trio (cognition, metabolic, inflammatory) → the primary outcomes (egf, cgi_s) — **6/6 pairs share no items** with the outcome. The functioning/severity outcomes load on the durable axes only as `g_anchor_on_specific` (pinned ~0), so these coordinates are genuinely not built from them: a real, non-circular forecast. Only the general factor **overall_severity** (anchored on EGF/CGI-S/FAST/EQ-5D) and the same-construct axes are autoregressive — exactly the planned guard, now data-derived.

> The clinically useful question is cross-construct *and* clean: *given two patients equal on an outcome today, does their durable biology forecast who diverges in a year?* That lift over the autoregressive bar is the finding; the bar itself is the thing it must beat.

## Data contract for M4 (resolved here)
- **Outcomes**: read native-scale from `data/processed/baseline_v{0,1,2}.parquet` (EGF 0–100, CGI-S 0–7, … verified), `(cohort, patient_id)`-indexed, NaN = missing.
- **Predictors**: baseline coordinates + per-patient SD from `results/face/patient_panel.parquet` (V0 rows) and the draw tensor `results/face/m3/panel_draws.npz`; the three map representations (continuous dims · 8 archetypes · 4-region tessellation) from `results/face/patient_strata.parquet`.
- **Reference covariates** (age, sex, siteid_city): pulled from the data layer at stage 41 (absent from the processed item tables); `arm` (DSM-5 subtype) + `cohort` from the panel.
- **Attrition**: `results/face/m3/ipw_weights.parquet` (`w_retained_V2`).

## Decision for the gate
Confirm the outcome set + the V2-paired sample sizes above, and the autoregressive list, before assembling the analysis frame (stage 41). The durable-biology → EGF/CGI-S added-value test is clean and adequately powered (≈2,400 paired on the 3-cohort primaries); SZ-absent and DR-thin outcomes are flagged for documented-partial verdicts.

Artifacts: `reports/40_{outcome_coverage,overlap_audit}.csv` · `docs/figures/40_coverage.png`.