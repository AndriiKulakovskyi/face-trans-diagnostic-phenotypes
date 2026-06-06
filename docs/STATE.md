# STATE — where V3 actually is

> **Read this first.** One-page ground truth, updated 2026-06-06. Where this disagrees with
> [`V3_RESULTS.md`](V3_RESULTS.md) or [`FINDINGS.md`](FINDINGS.md), **this file wins** until those
> are reconciled. Plan of record (direction, unchanged): [`V3_PLAN.md`](V3_PLAN.md).

## TL;DR

- **One engine is canonical:** the config-first soft-prior **ESEM-bifactor** model in
  `src/v3/latent_models/bayesian/`, driven by `configs/bayesian_model.yaml`, run via
  `scripts/v3/04_fit_measurement.py --stage S`.
- **Certified through Stage 2.** The decisive result (**Stage 1**) is that **a general factor `G`
  identifies** — a *functional-impairment / clinical-distress* axis (functioning + mood + some
  cognition), **orthogonal to metabolic/inflammatory biology**.
- This **overturns the earlier "no general factor" headline**, which came from a model that omitted
  the severity/functioning indicators `G` is built from.
- **Stage 3** (sharpen `G` with CGI severity) is **not yet converged** (R-hat 1.53) — in progress.
  **Stage 4** (mixed-likelihood suicidality/substance) is coded but **not run**.
- **Downstream layers — strata, prognosis, treatment — are NOT built.**

## The pipeline (run in order)

| # | script | does | output |
|---|--------|------|--------|
| 01 | `01_eligibility_audit.py` | candidate-dimension eligibility + V0 coverage | `results/v3/eligibility/` |
| 02 | `02_missingness_atlas.py` | observation matrix + missingness mechanism | `results/v3/missingness/` |
| 03 | `03_build_prior_matrix.py` | config ontology → `prior_loading_matrix_v3.csv` | `configs/` |
| 04 | `04_fit_measurement.py --stage S` | staged Bayesian measurement model | `results/v3/bayesian/stageS/` |

## Stage status (engine = `04_fit_measurement.py`)

| stage | question | certified | R-hat | dropped | takeaway |
|-------|----------|:---------:|:-----:|:-------:|----------|
| 0 | reproduce old core | ✅ | 1.01 | 23% | matches the old engine exactly |
| 1 | does `G` identify? | ✅ | 1.010 | 32% | **yes — `G` = impairment/distress, ⊥ biology** |
| 2 | ESEM cross-loadings | ✅ | 1.01 | 32% | simple structure mostly holds |
| 3 | sharpen `G` w/ CGI severity | ❌ | 1.53 | 38% | degenerate (hospitalization count) — debugging |
| 4 | mixed-likelihood suic/subst | — | — | — | coded, not run |

"Certified" here = the certification gate in `bayesian_model.yaml` (R-hat ≤ 1.01 · 0 divergences ·
ESS ≥ 400 · no Heywood). See caveats — this is **convergence**, not scientific validation.

## What the certified model (Stage 1–2) says

- **`G` (general factor) = functional impairment / distress.** Anchored by FAST 1.04, EGF 0.75,
  EQ-5D 0.60; affective items load strongly on it (MADRS 0.82, QIDS 0.69), cognition moderately
  (CVLT 0.32), **metabolic/inflammatory ≈ 0** (BMI 0.13, WBC 0.10, CRP 0.11).
- **Specific dimensions survive `G`** and stay weakly correlated among themselves (model Φ):
  metabolic×inflammatory 0.17 · cognition×metabolic 0.18 · sleep×affective 0.32 · the rest ≈ 0.
- **Net:** "no general factor" is **overturned**; "symptoms/severity ⊥ biology" is **strengthened**
  (biology sits off the general axis).

## Caveats — read before quoting any number

- **"Certified" = MCMC converged**, not validated: no out-of-sample test, no measurement invariance,
  no posterior predictive checks yet.
- **N = 1,500** (500 most-complete per cohort, balanced) of 9,013; **single visit V0**.
- **~23–38% of patients dropped** as rare missingness patterns (`min_group`); the fraction grows
  with the number of indicators.
- Results are through **Stage 2**; the severity-sharpened `G` (Stage 3) is not yet converged.

## File map (V3-only — V2 and the first-generation engine were deleted)

- **Foundation:** `src/v3/data/` · `scripts/v3/01,02` · `configs/candidate_dimensions_v3.yaml`,
  `likelihood_map_v3.yaml`, `soft_loading_priors_v3.csv` (the last three are audit outputs of `01`).
- **Canonical engine:** `src/v3/latent_models/bayesian/` · `src/v3/priors/` · `scripts/v3/03,04` ·
  `configs/dimensions.yaml`, `priors.yaml`, `likelihoods.yaml`, `bayesian_model.yaml`,
  `prior_loading_matrix_v3.csv`.
- **Tests:** all under `tests/v3/` — foundation (`test_adapter`, `test_filters`, `test_skip_logic`,
  `test_sanity_and_encoding`, now testing `v3.data`) + engine (`test_prior_matrix`). **84 passing.**
- **Superseded but kept:** `docs/V3_RESULTS.md` (old headline, banner-marked) · `docs/figures/v3/*.png`
  (first-generation figures — regenerate once Stage ≥ 3 certifies). Deleted code is in git history.

## Cleanup debt (tracked, not yet done)

- **Two ontology files:** `candidate_dimensions_v3.yaml` (eligibility, read by `01`/`02`) vs
  `dimensions.yaml` (modeling). Unify onto `dimensions.yaml` once `01`/`02` are migrated — defer to the
  roadmap re-think.
