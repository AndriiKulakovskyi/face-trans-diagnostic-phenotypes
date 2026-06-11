# CLAUDE.md — FACE clinical-biological transdiagnostic stratification (BP · SZ · DR)

> Guide for collaborators and AI assistants. Keep it short.
> **Methods + math of record: [docs/MEASUREMENT_MODEL.md](docs/MEASUREMENT_MODEL.md).**
> Current state: [docs/STATE.md](docs/STATE.md) · Data contract: [docs/DATA.md](docs/DATA.md).

## What this is

A project that turns the harmonized 3-cohort FACE **baseline (V0)** data (BP · SZ · DR) into a
**transdiagnostic dimensional map**, then — on that map — into validated patient strata and
prognosis/treatment decision models. Four layers that must not be collapsed:

```text
diagnostic cohorts → transdiagnostic dimensions → validated strata → prognosis / treatment
  (entry metadata)     (M1 — complete, 9-dim)       (M2 — next)        (later)
```

The discovery engine is **one global, missingness-aware Bayesian sparse bifactor / ESEM model** with mixed
likelihoods and **soft loading priors**; confirmation is **in-engine** (prior-free refit + PPC + WAIC — a
standalone FIML proved redundant, §5). The full specification —
logic, mathematics, staged estimation, acceptance gates — is in
**[docs/MEASUREMENT_MODEL.md](docs/MEASUREMENT_MODEL.md)**; read it before any modeling work.

## Load-bearing invariants (do not break)

1. **No naive imputation.** Estimate structure from each patient's observed cells (observed-data
   likelihood / FIML). Never build a mean/KNN/MICE-filled matrix. Deterministic skip-logic structural-zero
   decoding is allowed (it is not imputation).
2. **Diagnosis is metadata** — covariate / invariance grouping / validation only, never a dimension indicator.
3. **Baseline (V0) defines dimensions; later visits validate.** No discovery on V1–V4.
4. **The 10 candidates are soft priors, not labels.** The data may confirm / split / merge / proxy /
   reject / declare `not_testable` any of them. A construct with no indicators is `not_testable`, never an
   invented proxy.
5. **Only the global fit is interpreted.** Staged fits (S1–S4) are convergence checkpoints, never reported claims.

## Data layer (the foundation)

The self-contained data layer reads each dictionary variable from its per-cohort source column →
harmonization rule + per-variable **sanity bounds** (out-of-range → NaN, never imputed) → native clinical
scale, with deterministic **skip-logic** structural-zero decoding. It carries each variable's likelihood
family and missingness type (the data contract — [docs/DATA.md](docs/DATA.md)). Identifiers
(`usubjid_patients`, `cohort`, `arm`, `visit`, `siteid_city`) are never modelled on; `cohort`/`arm` are
covariates / validation labels.

## Conventions

- **Python ≥ 3.11.** Lean stack — **no DVC / Hydra / MLflow.** Configs in YAML; model-ready tables
  persisted as **Parquet** (raw stays CSV).
- **No naive imputation, ever.** Observed-data likelihood only.
- **Determinism:** fixed seeds.
- **Compute:** develop + test in **PyMC** on Mac M4 Pro, 24 GB RAM;
- **Cadence:** each stage writes a report (`reports/NN_*.md`) + figures, followed by a discussion gate
  before advancing. Consolidate, don't accrete; one canonical doc per concern.
- **Output:** scripts write aggregates to `results/`, figures to `docs/figures/`.

## Current state — M1 + M2 + M3 + M4 complete (next: M5 treatment)

The package is **`src/face/…`**; the engine (`src/face/models/bayesian/continuous_core` + `confirm`,
`runner`, `scoring`) and pipeline (`scripts/01_build_data`, `04_fit`, `05_confirm`, `06_invariance`,
`07_score`, `08_robustness`, `09_atlas`, `s5_certify{,9}`, `s5_corrg`) consume
`configs/prior_loading_matrix_v3.csv`. **M1 is complete (pending PI sign-off):** a **certified 9-dimension**
transdiagnostic map — G + cognition/metabolic/inflammatory/sleep/developmental-risk/suicidality **+ mania +
substance** — built, hardened (confirmation §5 / invariance §8 / robustness §8), certified (§4), scored (§7),
and adjudicated (§6). **Findings + discussion (paper-facing, read first): [docs/M1_FINDINGS.md](docs/M1_FINDINGS.md).**
Current status: **[docs/STATE.md](docs/STATE.md)**; per-candidate verdict:
**[docs/ADJUDICATION.md](docs/ADJUDICATION.md)**; per-stage detail: [docs/RESULTS.md](docs/RESULTS.md).
**M2 strata COMPLETE** (pending PI sign-off) — methods **[docs/STRATIFICATION_MODEL.md](docs/STRATIFICATION_MODEL.md)**,
findings **[docs/STRATA_FINDINGS.md](docs/STRATA_FINDINGS.md)**, atlas **[docs/STRATA_ATLAS.md](docs/STRATA_ATLAS.md)**,
detailed dev record **[docs/STRATA_RESULTS.md](docs/STRATA_RESULTS.md)**.
On the 9-dim coordinates (uncertainty-propagated, diagnosis = validation-only) the transdiagnostic space is a
**continuum, not biotypes**: 8 soft **archetypes** (lead) + a 4-region measurement-error **tessellation** —
transdiagnostic (ARI≈0 vs the 7 DSM-5 subtypes), specific-axis-driven (biology⊥G as phenotypes), stable, not
a missingness artefact, and a tighter *description* than DSM-5 (predictive/treatment validity → M4/M5).
Engine `src/face/strata/`; pipeline `scripts/20–26`.
**M3 temporal coherence COMPLETE** (pending PI sign-off) — methods **[docs/TEMPORAL_MODEL.md](docs/TEMPORAL_MODEL.md)**,
findings (paper-facing, read first) **[docs/TEMPORAL_FINDINGS.md](docs/TEMPORAL_FINDINGS.md)**, dev record
**[docs/TEMPORAL_RESULTS.md](docs/TEMPORAL_RESULTS.md)**. Scoring follow-up (V0→V1→V2) onto the **fixed** M1/M2
model (observed cells, uncertainty propagated, never re-discovered), the map + strata are **temporally
coherent**: the measurement holds (G1 invariance: 5/6 backbone axes invariant, inflammatory partial), and the
M2 geometry replays — **biology/cognition are durable (trait) while severity + symptoms slide (state)**, and
archetype identity persists (G3 variance ⟷ G4 geometry agree). Honest caveats: developmental's apparent state
is CTQ recall-noise (trait by design); G5-vs-DSM5 deferred to M4 (`arm` time-invariant). Clinical logic:
*stratify on the durable biology, monitor the moving symptoms.* Engine `src/face/temporal/`; pipeline
`scripts/30–37`; hand-off `results/face/patient_panel.parquet`.
**M4 prognosis COMPLETE** (pending PI sign-off) — methods **[docs/PROGNOSIS_MODEL.md](docs/PROGNOSIS_MODEL.md)**,
findings (paper-facing, read first) **[docs/PROGNOSIS_FINDINGS.md](docs/PROGNOSIS_FINDINGS.md)**, clinician-facing
prognostic atlas **[docs/PROGNOSIS_ATLAS.md](docs/PROGNOSIS_ATLAS.md)**, dev record **[docs/PROGNOSIS_RESULTS.md](docs/PROGNOSIS_RESULTS.md)**.
On the fixed M1/M2/M3 objects (panel + draws + strata + IPW; never re-scored), an errors-in-variables
Bayesian GLM tests whether a baseline coordinate/stratum predicts a 2-year outcome **incrementally beyond
DSM-5 + severity + the baseline outcome**. *Persists became predicts — for functioning, in the open-course
patients:* the durable **metabolic/inflammatory** ⊥G axes and the **8 archetypes** predict future
**functioning** (archetypes ΔELPD +46; remission AUC +0.017; metabolic survives the error-corrected-G
severity), robust to attrition/reliability/permutation — but **not severity** (autoregression-saturated).
**Co-informative with DSM-5** (complements, not replaces) and **course-dependent** (large in episodic
BP/DR, null in baseline-saturated SZ). The archetype prognostic atlas: 2-year functional remission
**14%→60%**, transdiagnostic. The map's value is **group-level stratification + continuous functional
forecasting**, not a large individual-binary boost — honest limits: scale trajectories not events,
internal validity, 2-year horizon. Engine `src/face/prognosis/`; pipeline `scripts/40–48`; hand-off
`results/face/m4/{prognosis_summary.csv, prognosis_patient_risk.parquet}`. Next: **M5 treatment** (does a
stratum *moderate treatment response*?).
