# FINDINGS — FACE V3 precision psychiatry — running log

Paper-oriented log of empirical + methodological findings for **V3**. Plan of record:
[`V3_PLAN.md`](V3_PLAN.md). Every number must be reproducible from the V3 pipeline once built.

> **Status.** V3 modeling has **not been run yet** — the patient-level Bayesian / FIML discovery engine
> (Phases E–M) is not implemented. The runnable code (`src/trans_diag/`, `scripts/01–15`) is the **V2
> benchmark implementation**; its complete results log is preserved at
> [`legacy_v2/FINDINGS.md`](legacy_v2/FINDINGS.md). **Do not carry V2 numbers forward as conclusions** —
> they enter V3 only as priors, baselines, and the hypotheses below.

## How V2 findings enter V3 (hypotheses, not conclusions)

Each V2 result is a **falsifiable hypothesis** to confirm / refine / **downgrade** under the V3
patient-level observed-likelihood model (V3_PLAN §0C, Phase G3). The V2 evidence is in
[`legacy_v2/FINDINGS.md`](legacy_v2/FINDINGS.md).

| # | V2 finding (benchmark) | V3 hypothesis / action | Verdict |
|---|---|---|---|
| H1 | 3 weakly-correlated axes: internalizing · cognition · cardiometabolic | retest under FIML + Bayesian mixed-likelihood; adjudicate {confirm/split/merge} | ⬜ open |
| H2 | No dominant general factor (Schmid–Leiman ECV 0.42) | estimate `G` **directly**; test whether specifics survive beyond it | ⬜ open |
| H3 | **Symptoms ⊥ biology** (between-block mean \|r\| ≈ 0.03); p-factor is symptom-only | retest under observed-data likelihood + posterior uncertainty | ⬜ open |
| H4 | Cardiometabolic axis robust but possibly mixed | **test split** into metabolic load vs inflammatory load | ⬜ open |
| H5 | Cognition = strongest fully-transdiagnostic axis | keep core; refine into cognitive-flexibility / broader cognition if supported | ⬜ open |
| H6 | Internalizing axis (mood scales 0% in FACE-SZ → SZ-proxy) | model as affective/anhedonic extension unless invariance supports all-cohort status | ⬜ open |
| H7 | Standalone suicidality / mania / substance-use (orthogonal) | model suicidality with mixed binary/ordinal/count likelihoods; mania = activation/impulsivity **proxy**; substance = module/covariate | ⬜ open |
| H8 | No discrete subtypes (continuum) | accept; build **probabilistic decision regions**, not natural clusters | ⬜ open |
| H9 | Modest prognosis increment over DSM (functioning ΔR²≈0.04; de-confounded relapse ΔAUC≈+0.036) | the **minimum benchmark** the V3 model ladder + decision-curve utility must beat | ⬜ open |

## V3 log

*(empty — append dated entries as Phases A–M produce results)*

- **V3-0 · Project reframed to precision-psychiatry V3 — 2026-06-05.** V3 plan adopted as the single
  source of truth ([`V3_PLAN.md`](V3_PLAN.md), [`V3_PLAN_SOURCE.md`](V3_PLAN_SOURCE.md)); V2 demoted to a
  benchmark/reference arm under [`legacy_v2/`](legacy_v2/README.md). No V3 modeling run yet. Next:
  Phase A (V3 data contract) → Phase B (missingness atlas) → Phase C (soft-prior map).
