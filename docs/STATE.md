# STATE — where the project is right now

> **Read this first.** Updated 2026-06-06.

## TL;DR

The project has been **replanned** around **Milestone 1 (M1): the transdiagnostic dimensional map** on the
FACE **V0** baseline. The methods and mathematics are **fixed** in
[`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md) (the single methods/plan of record). The previous
"Engine A" stage results (the old `03/04` Bayesian engine and its "no general factor" headline) are
**discarded** — superseded by the global, full-sample, explicit-latent approach in the methods doc. The
repository is on a **clean base**; implementation of M1 is the next step.

## What's decided

- **Model:** one **global** Bayesian sparse bifactor / ESEM — mixed likelihoods, soft priors,
  observed-cell likelihood (no imputation), **full V0 sample**. Estimated via a **staged continuation**
  (S1→S5); **only the global fit (S5) is interpreted.**
- **Confirmation:** **FIML** on the continuous backbone (masked-PAF dropped).
- **Dimension set (V0):** `G(severity)` · `cognition` · `metabolic` · `inflammatory` · `sleep` ·
  `suicidality` · `developmental-risk` (3-cohort) + `anhedonia` (BP/DR, thin). Dropped: impulsivity,
  negative symptoms, sensory.
- **Stack:** lean — PyMC (dev) + NumPyro/JAX-CUDA on the **RTX 4090** (full fits); YAML configs; Parquet
  model-ready persistence (raw stays CSV); a Jupyter notebook to run + display; per-stage reports.
- **Repo:** namespace `src/v3/…` → `src/face/…`; pipeline `scripts/01_build_data … 07_score`.

## What exists vs. not

- **Exists:** the data layer (`src/.../data` — harmonization + sanity bounds + skip-logic, no imputation);
  tests (`tests/v3/`, **84 passing**); the candidate-eligibility map (the soft-priors workbook + `configs/`).
- **Next (M1 build):** the `src/face/…` restructure · the pipeline `scripts/01…07` · the Parquet
  persistence layer · the staged global fit · FIML confirmation · adjudication · scoring.
- **Later milestones (not started):** strata (M2) · temporal coherence V1–V4 (M3) · prognosis (M4) ·
  treatment (M5).

## Open methods choices (flagged for the PI)

Sparsity prior (soft-normal vs horseshoe) · Student-t vs Gaussian continuous default · item- vs
factor-level covariates · acceptance-gate numbers. Defaults are set in
[`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md); confirm or overrule before S1.

## What to read

[`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md) (methods + math) · [`../README.md`](../README.md) (overview)
· [`../CLAUDE.md`](../CLAUDE.md) (guide) · [`DATA.md`](DATA.md) (data contract).
