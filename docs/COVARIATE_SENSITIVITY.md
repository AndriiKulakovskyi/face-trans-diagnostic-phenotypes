# Covariate-adjusted measurement sensitivity (resolves issue P0-04)

> Paper-facing record of the covariate-adjustment arm. Methods of record: [MEASUREMENT_MODEL.md](MEASUREMENT_MODEL.md) §3.1.
> Engine: `src/face/models/bayesian/continuous_core.py` (`prepare(covariate_adjust=...)`).
> Driver: `scripts/10_covariate_sensitivity.py`. Covariate table: `scripts/02_build_covariates.py`.

## The issue (P0-04)

The published measurement equation adjusts each indicator's mean for item-level covariates,
`η_ij = α_j + λ_jG·G_i + Σ_k λ_jk·D_ik + β_jᵀ·c_i` with `c_i =` age, sex, education, site (…),
but the primary marginalized engine z-scores items to zero mean and implemented **no covariate term
and no intercept**, and the processed baseline carried none of those covariates. As fitted, therefore,
the headline "metabolic/inflammatory burden is the least severity-entangled domain" was **not adjusted
for the obvious confounders** of a biology-vs-severity contrast (age, sex, site). This is the reviewer's
sharpest scientific concern, and it was a genuine report↔code mismatch.

## What was implemented

1. **Covariate sourcing** (`scripts/02_build_covariates.py`) — `data/processed/covariates_v0.parquet`
   (age, sex, education_years, edulevel), N = 9,013, aligned to the baseline `(cohort, patient_id)` index
   via the same harmonization pipeline as `01_build_data`. Site comes from `site_v0.parquet`.
   Coverage: age 99.9%, sex 100%, edulevel 78%, education_years 59% (we use `edulevel`).
2. **Covariate adjustment** (`prepare(covariate_adjust=True)`) — each continuous indicator is OLS-residualized
   on the covariate design **before** z-scoring: intercept + natural-spline(age) + age×sex + sex + edulevel +
   site dummies. For a Gaussian item this is **Frisch–Waugh–Lovell-equivalent** to the published `β_jᵀ c_i`,
   so the marginalized Woodbury kernel is untouched (still zero-mean) and missingness is preserved
   (no imputation of indicators; covariate NaNs mean-imputed in the design only). The flag is **off by
   default** — the primary encoding is byte-for-byte unchanged (94 engine tests pass).
3. **Scope** (per project invariant *diagnosis is metadata*): we adjust age/sex/education/site but **not**
   cohort/diagnosis — those remain validation / invariance grouping, so the transdiagnostic between-cohort
   signal is preserved rather than partialled away.

## Result

Correlated-G marginalized model (G freely correlated with the specifics), N ≈ 2,000 cohort-balanced
(N = 1,884 after the balanced draw), 2 seeds. **All four fits pass the strict gate** (R-hat ≤ 1.02,
ESS ≥ 421, 0 divergences); the adjusted and unadjusted arms used the **same patients** per seed
(matching persisted `index_hash`), i.e. a proper paired comparison.

| domain | Φ(G,·) unadjusted | Φ(G,·) **adjusted** | Δ |
|---|---:|---:|---:|
| inflammatory | 0.071 | **0.056** | −0.015 |
| metabolic | 0.124 | **0.058** | −0.067 |
| cognition | 0.385 | 0.229 | −0.156 |
| sleep | 0.422 | 0.409 | −0.013 |

(Source: `reports/10_covariate_sensitivity_report.md` + `.csv`; manifests in `results/manifests/covar_*`.)

## Interpretation

**The biology⊥G headline survives covariate adjustment — and strengthens.** After partialling out
age/sex/education/site, metabolic and inflammatory remain by far the least severity-entangled domains
(Φ ≈ 0.06, versus 0.23 cognition and 0.41 sleep), and the ordering
(inflammatory < metabolic ≪ cognition < sleep) is preserved.

Critically, adjustment **lowers** metabolic~G (0.124 → 0.058) rather than raising it: the age/sex/site
confounding was real and was *inflating* the apparent metabolic–severity association, so removing it
makes biology **more** orthogonal to functional burden, not less. Cognition also drops substantially
(0.385 → 0.229), consistent with age confounding both cognition and functioning. The load-bearing
premise — biological strata capture heterogeneity that severity misses — holds under adjustment.

## Status / follow-ups

- This is the **continuous correlated-G arm**, which is where the biology⊥G estimand lives (G is anchored
  by continuous functioning items; metabolic/inflammatory are continuous labs). The non-Gaussian
  explicit indicators (suicidality/substance) are not part of this estimand; extending item-level
  covariate terms to their logit/log-η is a documented secondary follow-up.
- The manuscript equation in `article/sections/05_methods.tex` should describe covariate adjustment as
  this implemented (residualization / FWL) sensitivity arm, and report biology⊥G under both arms.
