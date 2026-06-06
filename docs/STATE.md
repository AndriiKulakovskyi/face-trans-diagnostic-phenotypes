# STATE — where the project is right now

> **Read this first.** Updated 2026-06-06.

## TL;DR

The project has been **replanned** around **Milestone 1 (M1): the transdiagnostic dimensional map** on the
FACE **V0** baseline. The methods and mathematics are **fixed** in
[`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md) (the single methods/plan of record). The previous
"Engine A" stage results (the old `03/04` Bayesian engine and its "no general factor" headline) are
**discarded** — superseded by the global, full-sample, explicit-latent approach in the methods doc. The
repository is on a **clean base**, and M1 implementation is underway — the data layer + the
marginalized measurement engine are built, and **S1 (the continuous core) is certified at full N
(N = 9,013) on the Mac** (no 4090 required).

## What's decided

- **Model:** one **global** Bayesian sparse bifactor / ESEM — mixed likelihoods, soft priors,
  observed-cell likelihood (no imputation), **full V0 sample**. Estimated via a **staged continuation**
  (S1→S5); **only the global fit (S5) is interpreted.**
- **Confirmation:** **FIML** on the continuous backbone (masked-PAF dropped).
- **Dimension set (V0):** `G(severity)` · `cognition` · `metabolic` · `inflammatory` · `sleep` ·
  `suicidality` · `developmental-risk` (3-cohort) + `anhedonia` (BP/DR, thin). Dropped: impulsivity,
  negative symptoms, sensory.
- **Stack:** lean — PyMC + **NumPyro/JAX**. The marginalized (Woodbury) engine **certifies on the Mac M4
  (CPU)**; the RTX 4090 is optional (faster for later mixed-likelihood stages). YAML configs; Parquet
  model-ready persistence (raw stays CSV); per-stage reports; notebook later.
- **Repo:** package **`src/face/…`** (renamed from `src/v3`, tests green). Pipeline built so far:
  `scripts/01_build_data` (full-N V0 → Parquet) · `scripts/04_fit` (staged engine: marginalized Woodbury
  default, explicit-latent + `--gpu` optional).

## What exists vs. not

- **Exists:** `src/face/data` (harmonization + skip-logic, no imputation); `configs/` ontology +
  `prior_loading_matrix_v3.csv` (143 indicators × 10 factors) + the **prior atlas**
  (`docs/PRIOR_ATLAS.md`); `scripts/01_build_data` (Parquet persistence) + `scripts/04_fit` + the
  marginalized/explicit engine; tests (`tests/v3/`, **84 passing**).
- **First result — S1 continuous core CERTIFIED (full N):** see "S1 result" below.
- **Next (M1 build):** S2 (ESEM cross-loadings + the MADRS/QIDS/STAI cross-loading windows) → S3
  (mixed-likelihood suicidality + developmental-risk) → S4 (anhedonia) → **S5 global = the reported fit**
  → FIML confirmation → adjudication → scoring → the **empirical atlas + prior→posterior comparison**.
- **Later milestones (not started):** strata (M2) · temporal coherence V1–V4 (M3) · prognosis (M4) ·
  treatment (M5).

## S1 result — continuous core (CERTIFIED, full N = 9,013, no imputation)

Marginalized (Woodbury) bifactor, NumPyro/JAX-CPU: **R-hat 1.010 · ESS 1,939 · 0 divergences**
(415,531 observed cells, ~72 min on the Mac). Factors: G + cognition/metabolic/inflammatory/sleep
(continuous block).
- **G = functional burden / illness severity**, anchored cleanly by functioning + global severity only
  (FAST 1.04, EGF 0.69, EQ-5D 0.58, CGI-S 0.54 — no symptom content by design).
- **Biology ⊥ G:** mean |loading on G| = metabolic **0.08**, inflammatory **0.07** vs cognition 0.27,
  sleep 0.22 — biology sits *off* the general-burden axis; cognition/sleep partly track it. The earlier
  explicit-latent run reproduced these loadings (the two parameterizations triangulate).
- *Continuous backbone only* (independent-specifics bifactor, Φ = I); cross-loadings, the symptom blocks,
  and inter-factor correlations come at S2–S5. **Full writeup + interpretation:
  [`RESULTS.md`](RESULTS.md) §S1.** Artifacts: `reports/04_stage1_report.md` + `_loadings.csv`.

## Open methods choices (flagged for the PI)

Sparsity prior (soft-normal vs horseshoe) · Student-t vs Gaussian continuous default · item- vs
factor-level covariates · acceptance-gate numbers. Defaults are set in
[`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md); confirm or overrule before S1.

## What to read

[`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md) (methods + math) · [`RESULTS.md`](RESULTS.md) (findings log)
· [`PRIOR_ATLAS.md`](PRIOR_ATLAS.md) (prior map) · [`../README.md`](../README.md) (overview) ·
[`../CLAUDE.md`](../CLAUDE.md) (guide) · [`DATA.md`](DATA.md) (data contract).
