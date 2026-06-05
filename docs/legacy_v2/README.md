# `legacy_v2/` — the V2 benchmark / reference arm (traceability only)

> **This directory is not the current project.** It preserves the **V2 study** — a completed,
> high-quality *dimensional* analysis — as the **benchmark / reference arm** of the V3 program.
> The current source of truth is the **V3 precision-psychiatry plan**: [`../V3_PLAN.md`](../V3_PLAN.md),
> with the forward roadmap in [`../ROADMAP.md`](../ROADMAP.md) and the target pipeline in
> [`../PIPELINE.md`](../PIPELINE.md).

## Why V2 is kept (and what role it now plays)

V2 is a strong dimensional prototype. It is **frozen, reproducible, and demoted to three explicit
roles** inside V3 — never the active roadmap:

1. **Reproducibility baseline.** The V2 masked pairwise-complete correlation → PAF → Schmid–Leiman
   pipeline is re-run on the V3-curated data to confirm that data-curation changes did not destroy
   the known structure (V3 plan, Phase D).
2. **Soft-prior source.** V2's named constructs and axes become the **soft prior loading map** that
   seeds — but does not fix — the V3 latent model (V3 plan, Phase C / S2). They are hypotheses, not
   labels.
3. **Minimum benchmark.** Every V3 dimension, stratum, and prognosis model must **beat or defensibly
   refine** the V2 result; reproducing V2 with heavier machinery is not a success (V3 plan, §0A.8).

**V2 conclusions are now hypotheses to retest, not settled facts.** In particular the V2 storylines —
*"symptoms are orthogonal to biology," "no dominant p-factor," "no discrete subtypes," K=3 axes* — are
treated as **claims to confirm / refine / downgrade** under the V3 patient-level observed-likelihood
model, not as the project's conclusions. See [`../FINDINGS.md`](../FINDINGS.md) for the carry-forward
of each V2 finding into a V3 hypothesis.

## What the V2 code is

The Python package (`src/trans_diag/`) and the numbered scripts (`scripts/01–15`, `qa_harmonization`,
`sensitivity_*`, `figures_manuscript`, `build_manuscript`) **are the V2 benchmark implementation**.
They remain runnable and are the reproducibility baseline; they are **not** the V3 discovery engine
(which is the patient-level Bayesian / FIML latent model the V3 plan specifies and which is not yet
implemented). Treat the masked estimator as the *benchmark*, not the *primary model*.

## Contents

| File | What it is | V3 role |
|---|---|---|
| [`PIPELINE.md`](PIPELINE.md) | The V2 end-to-end analysis (masked correlation → PAF → Schmid–Leiman → stratification → validation A–D), diagrams + math | Reproducibility baseline (Phase D) |
| [`FINDINGS.md`](FINDINGS.md) | The V2 results log (K=3 axes, no p-factor, symptoms⊥biology, dimensional/no-subtypes, modest prognosis) | Hypotheses to retest (§0A.4) |
| [`AGGREGATION_RATIONALE.md`](AGGREGATION_RATIONALE.md) | Why V2 aggregated items → constructs before geometry (count-bias + structured-missingness derivations) | Background for the soft-prior construct map |
| [`PHENOTYPE_ATLAS.md`](PHENOTYPE_ATLAS.md) | V2 named factors as predictive features | Source for the V3 **prior atlas** (S2) |
| [`LABBOOK.md`](LABBOOK.md) | Chronological V2 lab notebook (entries V2-1 … V2-23) | Historical record |
| [`CLEANUP_PLAN.md`](CLEANUP_PLAN.md) | The executed V1→V2 repository cleanup | Historical record |
| [`planning/`](planning/) | V2 pre-registration plans (hierarchical-FA, validation A–D, manuscript) | Historical record |

The **V2 manuscript** — `../../results/manuscript/manuscript.md` (→ `FACE_trans_diagnostic_v2.docx` /
`.pdf`) — is the benchmark-arm paper (*"Symptoms are orthogonal to biology…"*). It is kept for
traceability and is **not** the current project objective.

## Earlier generations

The pre-V2 **v1** study is archived deeper, at git tag `v1-archive-2026-05-30` (branch
`archive/v1-research`). Do not carry v1 numbers forward.
