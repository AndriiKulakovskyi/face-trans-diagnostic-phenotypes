# CLAUDE.md — FACE clinical-biological transdiagnostic stratification (BP · SZ · DR)

> Guide for collaborators and AI assistants. Keep it short.
> **Methods + math of record: [docs/MEASUREMENT_MODEL.md](docs/MEASUREMENT_MODEL.md).**
> Current state: [docs/STATE.md](docs/STATE.md) · Data contract: [docs/DATA.md](docs/DATA.md).

## What this is

A project that turns the harmonized 3-cohort FACE **baseline (V0)** data (BP · SZ · DR) into a
**transdiagnostic dimensional map**, then — on that map — into validated patient strata and
prognosis/treatment decision models. Four layers that must not be collapsed:

```text
diagnostic cohorts → transdiagnostic dimensions → continuous map + A=5 archetypes → prognosis / treatment
  (entry metadata)     (M1 — complete, 8-factor)     (M2 — continuum, no privileged K)  (M4 / M5)
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

## Current state — M1 + M2 + M3 + M4 + M5 complete (next: PI sign-off; a true M5b needs randomized data)

The package is **`src/face/…`**; the canonical engine is the Gaussian-copula OOP fit
(`src/face/models/bayesian/measurement_model_oop.py`), consuming `configs/prior_loading_matrix_v3.csv` and the
**data/config layer** (`scripts/01_build_data` → `data/processed/`, `scripts/02_build_covariates`). Only the
data/config layer + utilities remain as scripts; all milestone modeling lives in the OOP engines below.

**M1 — the 8-factor map (complete, PI sign-off 2026-06-27).** One global, missingness-aware Bayesian
sparse-bifactor / ESEM model, mixed likelihoods, Gaussian-copula (rank-INT) continuous block, marginalized
(Woodbury); fit at full N = 9,013, cohort-weighted (R-hat 1.03, 0 divergences; 88 indicators). **8 latent
dimensions:** G (overall burden) ⊥ 7 specifics — cognition, **immunometabolic** (one biology factor:
cardiometabolic + inflammatory markers), sleep, mania/activation, suicidality, developmental-risk, **substance**
(pinned orthogonal — its cross-factor correlations are non-identifiable). The map is mostly simple-structure
with **3 earned cross-loadings** (CTQ-37 → cognition, PSQI-latency → cognition, PSQI-daytime → cognition; all
95% CI exclude 0): a **regularized ("Finnish") horseshoe** prior on every off-home specific↔specific loading is
default-off (global shrinkage) / evidence-on (heavy-tailed local shrinkage) / magnitude-capped (slab), which
protects the thin factors while letting small, clinically real cross-talk emerge — a continuous sparse-ESEM
validation freed all off-home cells and ~83% shrank to ≈0, so the simple structure is *earned*. Findings (read
first): **[docs/HORSESHOE_ESEM.md](docs/HORSESHOE_ESEM.md)** / **[docs/M1_FINDINGS.md](docs/M1_FINDINGS.md)**;
loadings/Φ at `reports/copula_8factor_{loadings,phi}.csv`. Map location:
`results/face/.../copula/weighted_8d/hs_s5_merged_xc`.

**M2 — stratification (complete).** Methods **[docs/STRATIFICATION_MODEL.md](docs/STRATIFICATION_MODEL.md)**,
findings **[docs/STRATA_OOP_FINDINGS.md](docs/STRATA_OOP_FINDINGS.md)**, atlas
**[docs/STRATA_OOP_ATLAS.md](docs/STRATA_OOP_ATLAS.md)**. On the 8-dim coordinates (uncertainty-propagated,
diagnosis = validation-only) the space is a **continuum, not biotypes** (silhouette 0.140 ≈ structureless null
0.137 ± 0.002, z = 1.13 n.s.; HDBSCAN 0 clusters). Load-bearing objects = the continuous coordinates + a stable
**A = 5 archetype simplex** (clean stability cliff at A = 6): **A0** activation/sleep, **A1** severe clean-biology,
**A2** immunometabolic (the biology corner), **A3** trauma/suicidality, **A4** low-burden/well. Transdiagnostic
(ARI 0.006 vs DSM-5), not-just-severity (driven by mania + suicidality ≫ G), tighter than DSM-5 at lower BIC.
The soft tessellation is exported as a **nested K-family (2/3/4) with no privileged K** — the operative K is
decided by M4/M5 incremental validity (answer: none). Engine `src/face/strata/strata_model_oop.py`; driver
`notebooks/run_strata_model_oop.py`; hand-off
`results/face/strata_oop/consolidate/{patient_strata.parquet, k_family_menu.csv}` (9,013 × 50) + coords in
`results/face/strata_oop/coordinates/`.

**M3 — temporal coherence (complete).** Methods **[docs/TEMPORAL_MODEL.md](docs/TEMPORAL_MODEL.md)**, findings
**[docs/TEMPORAL_OOP_FINDINGS.md](docs/TEMPORAL_OOP_FINDINGS.md)**. Scoring V1/V2 under the **fixed** M1/M2 model
(observed cells, uncertainty propagated, never re-discovered; V0 reproduced at r ≈ 0.99) shows the map + strata
are temporally coherent: **G1 invariance** all 4/4 backbone axes invariant (G, cognition, immunometabolic φ
0.987, sleep); **G3 trait/state** immunometabolic ICC 0.91 the single most durable axis, cognition 0.70 trait,
severity trait-by-rank with population improvement, symptoms slide; **G4 persistence** archetype weights persist
(Arm-B cosine median 0.81). Clinical logic: **stratify on the durable biology, monitor the moving symptoms.**
Engine `src/face/temporal/temporal_model_oop.py`; driver `notebooks/run_temporal_model_oop.py`; hand-off
`results/face/temporal_oop/` (its strata-independent IPW at `.../attrition/ipw_weights.parquet` feeds M4 +
repbench).

**M4 — prognosis (complete).** Methods **[docs/PROGNOSIS_MODEL.md](docs/PROGNOSIS_MODEL.md)**, findings + atlas
**[docs/PROGNOSIS_OOP_FINDINGS.md](docs/PROGNOSIS_OOP_FINDINGS.md)**. On the fixed M1/M2/M3 objects, an
errors-in-variables Bayesian GLM tests whether a baseline coordinate/stratum predicts a 2-year outcome
incrementally beyond DSM-5 + severity + baseline outcome. The **A = 5 archetypes predict 2-year functioning**
(ΔELPD +62.8 held-out; IPW-robust +54.4; permutation-null; **co-informative with DSM-5**; course-dependent,
BP-led) but **not severity** (autoregression-saturated). **Operative K = none** (archetypes dominate every
tessellation). Prognostic atlas: 2-yr functional remission **17% → 52%**, the immunometabolic corner (A2) the
worst-prognosis pole, within-diagnosis (composition explains ~4%). Honest: small individual-binary lift (AUC
+0.010 — the value is group-level stratification + continuous forecasting). **Representation benchmark**
(`src/face/prognosis/repbench/`, [docs/M4_REPRESENTATION_BENCHMARK.md](docs/M4_REPRESENTATION_BENCHMARK.md)): vs
raw indicators under a matched XGBoost the map is **sufficient for deterioration** (AUC tie) and
**near-sufficient for recovery** (raw +0.04 AUC; 92–97% within-factor compression — residual is item-level, not
a missing axis). Engine `src/face/prognosis/prognosis_model_oop.py`; driver
`notebooks/run_prognosis_model_oop.py`; hand-off `results/face/prognosis_oop/`.

**M5 — treatment (complete, bounds-and-defends).** Methods **[docs/TREATMENT_MODEL.md](docs/TREATMENT_MODEL.md)**,
findings **[docs/TREATMENT_OOP_FINDINGS.md](docs/TREATMENT_OOP_FINDINGS.md)**. Treatment data live in the
per-cohort thesaurus `TRAITEMENTS` tabs, harmonized to common drug-class exposures (ATC[SZ] / class-string[DR] /
lifetime-flag[BP]). A causal pipeline (**overlap gate → propensity[severity+diagnosis+demographics+map] →
doubly-robust EIV moderation [treat×durable-axis] + A=5 archetype interaction + E-value**) asks whether the map
*moderates / selects* treatment. **(1) Ceiling** — on observational TAU it does not reliably: lithium-BP a
well-identified MDE-bounded null (E 1.20–1.28, interaction MDE ≈ 0.20), antipsychotic-BP a confounded average
effect (E 1.80) with suggestive-but-unconfirmed moderation, clozapine-SZ underpowered; the map is **prognostic +
descriptive, not prescriptive**. **(2) Defends M4** — the prognostic carrier **survives treatment adjustment**:
the **A2 immunometabolic archetype corner** (attenuation 7.7% / 6.4% IPW) and the immunometabolic durable axis
(6.4% / 4.1%); not a treatment proxy. **(3) Describes course** — the immunometabolic corner faces the hardest
2-year course (resistance 44%, side-effects 25% vs the well pole's 20% / 11%), discrimination clears for
response/side-effects (archetype ΔAUC +0.012 / +0.034 / +0.042). Genuine treatment **selection** needs
randomized/trial-arm data (a future **M5b**). Engine `src/face/treatment/treatment_model_oop.py`; driver
`notebooks/run_treatment_model_oop.py`; hand-off `results/face/treatment_oop/`. **Full vertical synthesis:
[docs/COPULA_VERTICAL_FINDINGS.md](docs/COPULA_VERTICAL_FINDINGS.md).**

**Open follow-ups:** FondaMental treatment-data (RCT/prescription) check for M5b; a DR-MARS harmonization fix.
