# FACE V3 — Precision-psychiatry stratification & decision-modeling plan

> **This is the single source of truth for the project's direction.** Companion current-facing docs:
> [`ROADMAP.md`](ROADMAP.md) (what/why), [`PIPELINE.md`](PIPELINE.md) (target architecture),
> [`DATA.md`](DATA.md) (data contract).

> **Progress (2026-06-06).** Phases **A·B·C done**; Phase **F measurement engine converged through
> Stage 2** (config-first soft-prior ESEM-bifactor; `src/v3/latent_models/bayesian/`, run via
> `scripts/v3/` `03` build-prior-matrix · `04` fit-measurement). **Headline (Stage 1, certified):** a
> general factor `G` (functional impairment / distress) **identifies, orthogonal to
> metabolic/inflammatory biology** — this **overturns** the earlier "no general factor." 3 of 10
> candidates remain unsupported in the common dictionary. **Current state + caveats:**
> [`STATE.md`](STATE.md); step journal: [`LABBOOK_V3.md`](LABBOOK_V3.md). Next: certify Stage 3 (CGI
> severity) + Stage 4 (suicidality/substance) → Phase H invariance → Phase J strata.

## 0 · One sentence

V3 uses **patient-level, missingness-aware observed-data latent models** — a Bayesian sparse
bifactor / ESEM-like model as the **primary discovery engine**, with FIML/SEM as a confirmatory
check — to discover and **validate** transdiagnostic dimensions, project patients into
uncertainty-aware dimension space,
derive **probabilistic patient strata as validated decision regions**, and test whether those strata
improve **prognosis and treatment-relevant decisions** beyond DSM diagnosis and conventional severity.

## 0A · The conceptual contract (four layers — do not collapse them)

```text
diagnostic cohorts                         (entry point + validation metadata, NOT clustering features)
        → transdiagnostic dimension discovery   (measurement layer: continuous latent coords + uncertainty)
        → validated patient strata               (decision layer: probabilistic risk/treatment profiles)
        → prognosis / treatment decision models  (clinical-utility layer: calibrated predictions, decision curves, causal where feasible)
```

| Layer | Scientific role | V3 output |
|---|---|---|
| Diagnostic cohorts | Entry + validation metadata | BP / SZ / DR labels — **covariate/validation, never a clustering feature** |
| Transdiagnostic dimensions | Measurement | continuous latent coordinates **with posterior uncertainty** |
| Patient strata | Segmentation / decision | **probabilistic** risk- or treatment-relevant profiles (soft assignments) |
| Prognosis / treatment | Clinical utility | calibrated predictions, decision curves, causal estimates where feasible |

**Dimension discovery describes variation; stratification converts validated variation into
decision-relevant territories.** A dimension is a latent clinical/biological gradient explaining
stable covariance among observed indicators — it need not be an independent mechanism, and dimensions
**may be correlated** (a correlated clinical coordinate system, not a Cartesian grid).

## 0B · The 10 candidate dimensions are a *soft starting ontology*, not fixed scores

The candidate constructs are:

1. Impulsivity · 2. Cognitive flexibility · 3. Negative symptoms · 4. Anhedonia ·
5. Metabolism / immunometabolism · 6. Sleep / circadian dysregulation · 7. Overall clinical severity ·
8. Sensory abnormalities · 9. Neurodevelopment / neurodevelopmental alterations · 10. Suicidality.

They define a **starting ontology and a soft prior loading map** — **not** 10 assumed final
dimensions, and **not** hand-tagged composite scores. The discovery contract is:

```text
10 candidate constructs → soft prior loading map → missingness-aware hybrid latent modeling
        → empirical factor structure → {confirmed | split | merged | module | proxy | unsupported}
        → patient strata → prognosis & treatment validation
```

The model is **explicitly allowed to confirm, split, merge, reject, downgrade, or cross-load** any
candidate. The final dimension set may be **smaller or different** from the initial 10 — that is the
*expected* output of a hybrid discovery framework, not a failure. V3 must allow cross-loadings,
factor splitting/merging/rejection, diagnosis-specific modules, a general severity factor, method
factors, missingness sensitivity, and external longitudinal/prognostic validation.

### FACE-specific eligibility *before* modeling (don't ask the model to discover unmeasured constructs)

| Candidate dimension | V3 starting status | Reason |
|---|---|---|
| Overall clinical severity | **core** — general factor `G` | CGI/GAF/EQ-5D/functioning/hospitalization measure it directly |
| Cognitive flexibility / cognition | **core** | neuropsychology directly supports a cognition axis |
| Metabolism / immunometabolism | **core, but test split** | a cardiometabolic-inflammatory block may hide metabolic vs inflammatory subdomains |
| Sleep / circadian dysregulation | **core if PSQI/sleep indicators present** | measurable across cohorts; circadian specificity may be partial |
| Suicidality | **core risk dimension** | likely near-standalone; needs mixed binary/ordinal/count likelihoods + skip-logic awareness |
| Anhedonia | **extension / module** | likely BP/DR-measured; SZ may be proxy only |
| Impulsivity | **proxy / module** | no dedicated common impulsivity measure unless verified |
| Negative symptoms | **SZ module** unless common direct indicators exist | do not infer from poor functioning alone |
| Neurodevelopmental alterations | **proxy / module** | use developmental-risk / adversity / early-onset proxy, not direct neurodevelopmental biology |
| Sensory abnormalities | **unsupported** unless direct indicators exist | the model cannot discover an unmeasured construct |

V3 must keep three objects distinct: a **candidate construct** (clinical idea →
soft prior/hypothesis), an **empirical latent dimension** (stable axis of covariance → discovered &
validated), and a **patient stratum** (recurring decision-relevant profile → derived after dimension
validation).

### V3 success criterion

V3 succeeds only if the pipeline delivers **at least one** of: (1) defensible
patient-level dimension scores with uncertainty; (2) sharp empirical adjudication of the 10
candidates; (3) a clear verdict on symptom–biology orthogonality under observed-data
likelihood; (4) validated probabilistic strata that are *not* just natural clusters; (5) improved
prognosis calibration/discrimination/decision-utility beyond diagnosis and severity;
(6) feasible treatment-relevant modeling under target-trial assumptions. **Elegant machinery without
downstream value is not a success.**

---

## 1 · Scientific objective & primary question

**Objective.** Estimate empirically-supported, clinically-interpretable transdiagnostic latent
dimensions across FACE BP/SZ/DR using patient-level missingness-aware models, then derive validated
patient strata that improve prognosis and treatment-relevant decision modeling beyond DSM and
conventional severity.

**Primary question.** Do patient-level latent dimensions and derived strata add clinically meaningful
predictive or decision value beyond `diagnosis + age + sex + site + baseline clinical severity`?

**Main claim to target (only if supported).** *FACE patients can be represented by a validated set of
transdiagnostic latent dimensions, and probabilistic strata derived from these dimensions improve
prognosis or treatment-relevant decision modeling beyond DSM diagnosis and standard severity.*

**Do NOT claim** (unless strongly validated): definitive biological disease subtypes; true mechanistic
biotypes; causal treatment response without target-trial emulation; that all 10 candidate constructs
are measured; sensory abnormalities or negative symptoms if the common FACE variables do not directly
measure them.

## 2 · Target architecture

```text
FACE BP/SZ/DR V0 baseline
  → dictionary correction + unit harmonization + skip-logic decoding
  → missingness atlas + structural/design/informative/sporadic classification
  → soft prior loading matrix (10 candidate dimensions)
  → patient-level FIML / Bayesian mixed-likelihood latent models
  → empirical dimension adjudication {confirmed | split | merged | cohort-module | unsupported}
  → posterior dimension scores + uncertainty + measurement coverage
  → probabilistic patient strata / decision regions
  → strata validation: statistical · clinical · cohort · site · missingness · longitudinal
  → prognosis models + treatment-relevant target-trial analyses
  → precision-psychiatry decision framework
```

The end-to-end diagram and the missing-data doctrine are in [`PIPELINE.md`](PIPELINE.md).

---

## 3 · Phased plan (A–T)

Each phase lists its intent and primary artifacts. Phases A–C build the foundation; E–G are the
modeling core; H–N validate and interpret; O–T are structure, acceptance, deliverables, risk, and the
project-management layer.

### Phase A — Build the analytical foundation — ✅ done (V3-1)
- **A1 Patient-level analytical base.** Stand up the V3 code (`src/v3/data`, `scripts/v3/`) and outputs
  (`results/v3/`) on the harmonized 3-cohort data, with the data contract under `configs/`.
- **A2 V3 data contract.** Strict patient-level schema (one row/patient at baseline) + long-format
  observed-cell tables. Extend the dictionary with per-variable **likelihood family, missingness type,
  prior-loading metadata, covariate/outcome status, and modeling role** (see [`DATA.md`](DATA.md)).
  → `data_dictionary_v3.csv`, `variable_schema_v3.yaml`, `likelihood_map_v3.yaml`,
  `construct_prior_map_v3.yaml`.
- **A3 Re-check harmonization, units, score direction.** Validate ranges/units; parse free-text labs;
  reverse-code so **higher = more burden/dysfunction** unless documented (GAF/EGF, EQ-5D VAS, HDL…);
  `log1p` skewed labs (CRP); residualize TMT-B on TMT-A/age/education/site; preserve ordinal/binary/
  count nature (PSQI, C-SSRS/ISF). **Keep deterministic scaling where useful but do not force one
  pseudo-continuous metric — the observation likelihood carries the variable type.**
- **A4 Skip-logic decoding, separated from imputation.** Use deterministic structural-zero decoding for
  gated variables (e.g. `attempt_count = 0` when `attempt_ever = 0`); **never** overwrite observed values
  or create values where the gate is unknown.

### Phase B — Missingness atlas & measurement eligibility — ✅ done (V3-2)
- **B1 Missingness matrix** `R_ij ∈ {0,1}`, summarized by cohort/site/wave/age/sex/severity/block/
  instrument/candidate-dimension. In FACE, missingness reflects cohort design, site practice,
  questionnaire routing, clinical feasibility, sometimes severity — it is an **explicit object**, not
  noise.
- **B2 Classify each variable's missingness mechanism:** Structural | Design | Clinical-skip |
  Sporadic | Informative | Outcome-related.
- **B3 Model observation probability** for key blocks (`Observed_j ~ cohort + site + age + sex +
  education + diagnosis + severity_proxy`) to see whether missingness is design/cohort/site/severity-
  driven; feed into sensitivity analyses.
- **B4 Eligibility tiers:** Core all-cohort | Partial extension | Diagnosis-specific module |
  Covariate | Outcome | Excluded. **A dimension can only be discovered if it is actually measured.**

### Phase C — Soft-prior construct map — ✅ done (V3-1)
- **C1 Convert the 10 candidate constructs into a soft prior loading matrix** `T_jk` with
  statuses {primary_expected · plausible_cross_loading · unlikely · covariate_only · invalid ·
  outcome_only}. Goal: **let theory guide discovery while the data may split/merge/reject/cross-load.**
- **C2 Adjudicate which candidates are testable** (confirmed-core | extension | proxy-only |
  not-testable) — see the eligibility table in §0B. → `soft_loading_prior_matrix.{csv,yaml}`,
  `core_dimensions_v3.yaml`, `extension_dimensions_v3.yaml`, `unsupported_dimensions_v3.md`.

### Phase E — FIML latent-model confirmation
- **E1** Fit patient-level **FIML SEM/ESEM** models (E1 one-factor; E2 three-axis; E3 +standalones;
  E4 candidate 6–8 dim; E5 bifactor `G`+specifics; E6 ESEM cross-loadings). FIML uses each patient's
  observed data — **complete-data ML is precluded by missingness, but observed-data FIML is compatible
  with no naive imputation.**
- **E2** Use FIML to test general severity `G` vs specific dimensions:
  `X_ij = ν_j + λ_jG·G_i + Σ_k λ_jk·D_ik + ε_ij`.

### Phase F — Bayesian sparse bifactor model with mixed likelihoods (**primary discovery engine**) — 🟢 certified core (V3-5)
- **F1 Generative model.** `G_i ~ N(0,1)`, `D_ik ~ N(0,1)`; `η_ij = α_j + λ_jG·G_i + Σ_k λ_jk·D_ik +
  covariates`; `X_ij ~ likelihood_j(η_ij, params_j)`. Models the observed variables themselves — **no
  completed dataset is ever created.**
- **F2 Mixed likelihoods by variable type:** Gaussian/Student-t (standardized continuous), lognormal/
  Student-t after log (skewed labs), ordered logistic/probit (ordinal items), Bernoulli-logit (binary
  history/suicide), negative-binomial/Poisson (counts), zero-inflated NB (zero-heavy counts);
  longitudinal outcomes get a **separate** outcome model, never the baseline dimension model.
- **F3 Soft loading priors** from the C1 matrix: primary `λ_jk ~ N(0.6, 0.3)`; plausible cross-load
  `~ N(0, 0.25)`; unlikely `~ N(0, 0.05)`; or horseshoe / spike-and-slab for stronger discovery. **The
  10 candidates are priors, not forced labels** — the data can keep unexpected cross-loadings or shrink
  unsupported ones to zero.
- **F4 General burden `G`** estimated directly; specifics explain residual domain variance (`G ⊥ D` for
  identifiability in the first pass; correlated `D` in a sensitivity model). Tests whether a dominant
  general factor is supported under patient-level likelihood.
- **F5 Diagnosis/site/age/sex/education as covariates in the measurement model — never as dimension
  indicators.** Separates diagnosis/site mean differences from within-patient latent dimensions.
- **F6 Missingness handling.** Start with an observed-likelihood **MAR** model
  `p(X_observed | dimensions, params)`; then **sensitivity** models for informative missingness
  `R_ij ~ Bernoulli(logit⁻¹(a_j + γ_jG·G_i + γ_jk·D_ik + cohort + site))`.
- **F7 Fit & diagnose.** Core model first (severity + sleep + cognition + metabolic + inflammatory +
  developmental + suicidality), then extensions; **mandatory diagnostics** (R-hat, ESS, divergences,
  prior/posterior predictive checks, loading stability, posterior uncertainty by cohort).

### Phase G — Model comparison & dimension discovery
- **G1** Compare competing structures (1-factor, three-axis ±standalones, 10-candidate, 6–8 empirical,
  bifactor, ESEM, Bayesian sparse) by approximate-LOO / predictive log-likelihood, posterior predictive
  checks, loading interpretability, reliability, coverage, invariance, resampling stability, outcome
  validity. **Adjudicate, don't assume, dimensionality.**
- **G2 Dimension adjudication table** — every candidate → {Confirmed | Split | Merged | Module |
  Proxy-only | Unsupported}. → `final_empirical_dimensions_v3.yaml`.
- **G3 Validate the discovered structure** (symptom–biology orthogonality, general-factor strength,
  cognition transdiagnosticity, cardiometabolic structure, SZ internalizing gap, standalone
  suicidality/mania/substance, subtype-vs-continuum, prognostic value) → confirm / refine / **downgrade**.

### Phase H — Measurement invariance & transdiagnostic validity
- **H1** Per-dimension transdiagnosticity: coverage + loading stability + score reliability by cohort,
  cohort-residualized loadings, leave-one-cohort-out congruence, within-cohort covariance reproduction.
- **H2** Measurement invariance / DIF across BP/SZ/DR, sex, age bands, site, year (multi-group FIML,
  Bayesian group-specific loadings/intercepts, posterior DIF, partial-invariance models).

### Phase I — Patient-level dimension scoring
- **I1 Posterior dimension scores** per patient (`*_mean`, `*_sd`) for `G` and each `D`, **plus**
  number of observed indicators, posterior reliability proxy, cohort, site, baseline severity,
  missingness pattern. V3 = posterior scores **with uncertainty**, not point Thomson scores.
- **I2 V3 phenotype atlas** (`docs/PHENOTYPE_ATLAS_V3.md` when produced) — a *probabilistic* atlas with
  posterior loadings, coverage, direction, state/trait/fixed-historical status, reliability, limits,
  clinical reading.

### Phase J — Stratification as precision decision regions
- **J1 Reframe the objective.** Strata are **validated recurrent patient profiles in dimension space**
  (latent profiles, Bayesian mixtures, risk territories, outcome- or treatment-relevant profiles) — a
  stratum can be valid even when the global distribution is continuous, provided it is stable,
  interpretable, non-artefactual, prognostic, and useful, and is represented with **posterior class
  probabilities, not forced hard labels**. The objective is `continuous dimension space → probabilistic
  validated decision regions`.
- **J2 Fit probabilistic strata** on posterior dimension scores (LPA, Bayesian GMM, mixture-of-factors,
  consensus clustering, risk-threshold regions); keep `P(stratum=k)` and assignment entropy.
- **J3 Sensitivity:** direct raw-data stratification (observed-cell similarity, Gower/HDBSCAN,
  missingness-aware autoencoder) compared to dimension-based strata — a secondary control, not the headline.

### Phase K — Strata validation
- **K1 Statistical stability** (bootstrap, diagnosis-balanced subsampling, leave-one-site-out, posterior
  class stability, sensitivity to #dimensions and to high-missingness exclusion).
- **K2 Artefact checks** — reject/relabel any stratum that is mostly one site / cohort / missingness
  pattern / completeness level / medication artefact. **Precision strata must not be assessment-protocol
  strata.**
- **K3 Clinical profile validation** (dimension profile + raw clinical/functioning/suicide/cognitive/
  biological/course characteristics → clinically interpretable decision profiles).
- **K4 Longitudinal/outcome validation** — baseline strata predict **future** outcomes not used to
  define them (GAF/EGF, FAST, relapse, hospitalization, suicidality, nonadherence, metabolic worsening,
  dropout): `Y_future ~ diagnosis + age + sex + site + baseline severity + dimensions + strata probs`.

### Phase L — Prognosis modeling
- **L1 Model ladder** per outcome: `M0 = age+sex+site` → `M1 +DSM` → `M2 +conventional severity` →
  `M3 +posterior dimension scores` → `M4 +stratum probabilities` → `M5 +selected raw vars + missingness
  indicators` → `M6 +early-course trajectory`. Quantifies the **incremental** value of dimensions/strata.
- **L2 Missingness-aware learners** (CatBoost/LightGBM/XGBoost/HistGradientBoosting, survival variants;
  penalized logistic/Cox on dimension scores as classical baselines) — do **not** impute high-missing
  raw variables before prediction.
- **L3 Validation to clinical-prediction-model standards** — nested CV, GroupKFold by patient,
  leave-one-site-out, diagnosis-stratified, temporal if dates allow; report AUROC, AUPRC (rare
  outcomes), Brier, calibration slope/intercept, C-index, integrated Brier, **net benefit (decision
  curves)**, subgroup performance. A pooled AUC is insufficient in a BP-dominant imbalanced dataset.

### Phase M — Treatment & decision modeling
- **M1 Separate prognosis from treatment effect** (FACE is observational; treatment is confounded by
  indication — do not infer treatment effects from predictive associations).
- **M2 Target-trial emulations** for selected questions (eligibility, time-zero, strategies, grace
  period, follow-up, outcome, causal contrast, confounders, censoring, positivity). Candidates: lithium
  initiation in BP; LAI vs oral / adherence in SZ; depression augment-vs-switch.
- **M3 Heterogeneous treatment effects by stratum** (IPW/AIPW, double-ML, causal forests/CATE,
  stratum×treatment interactions) — *does a stratum identify patients with different expected
  benefit-risk?* This is the move from description to decision support.

### Phase N — Clinical interpretation & reporting
- **N1** Dimension cards + stratum cards (label, indicators, posterior loadings, coverage, uncertainty,
  cohort representation, outcome risks, actionability, limits).
- **N2** Model cards + reporting checklists (intended use, population, outcome, features, missingness
  handling, performance, calibration, subgroup, limits, not-for-use) — **TRIPOD-AI**, **PROBAST-AI**.

### Phase O — Repository & implementation structure
Target layout: `configs/ · src/v3/{data,missingness,priors,fiml,bayesian,dimensions,strata,prognosis,
treatment,reporting} · scripts/v3/ · results/v3/ · docs/ · tests/`. The data layer lives in
`src/v3/data`; the staged pipeline in `scripts/v3/` writes to `results/v3/`; the data contract is under
`configs/`. New modeling modules are added as sub-packages (`src/v3/bayesian/`, `.../fiml/`,
`.../missingness/`, `.../priors/`, `.../strata/`, `.../prognosis/`, `.../treatment/`). Keep the
**no-imputation** and **V0-anchor** invariants. Add V3 tests: `test_schema`, `test_skip_logic`,
`test_likelihood_assignments`, `test_prior_matrix`, `test_no_outcome_leakage`.

### Phase P — Acceptance criteria
- **Dimension** accepted into the V3 core only with most of: sufficient direct indicators; acceptable
  BP/SZ/DR coverage; stable posterior loadings; not explained by site/cohort/missingness; theoretically
  interpretable; acceptable posterior reliability; (partial) invariance defensible; a longitudinal/
  external validity signal.
- **Stratum** accepted only if: statistically stable; probabilistically assignable; clinically
  interpretable; not a site/missingness artefact; not merely DSM; associated with a future outcome or
  decision value.
- **Prognosis model** useful only if it beats DSM/severity on calibration, AUPRC/AUROC (as appropriate),
  Brier, **decision-curve net benefit**, subgroup, and site/dataset robustness. *A tiny pooled-AUC gain
  without calibration or decision benefit is not enough.*
- **Treatment model** interpretable only with: a defined target trial; positivity within relevant
  strata; plausible confounding adjustment; reported sensitivity analyses; causal language limited to
  the design.

### Phase Q — Deliverables
1. V3 data dictionary + missingness atlas · 2. FIML confirmation ·
3. Bayesian mixed-likelihood sparse bifactor model · 4. Dimension adjudication report · 5. V3 phenotype
atlas · 6. Probabilistic patient strata · 7. Strata validation report · 8. Prognosis model-ladder
report · 9. Treatment target-trial feasibility report. **Manuscript sequence:** (1) measurement model;
(2) dimensions → validated strata; (3) precision-psychiatry decision modeling.

### Phase R — Risk register (abridged)
Bayesian model too slow (→ start core, variational/Pyro later) · non-identifiability (→ sparse priors,
staged models, diagnostics) · `G` dominates strata (→ bifactor with specifics) · BP dominance (→
balanced resampling + cohort weights) · DR too small (→ report uncertainty, avoid DR overclaim) ·
structural missingness → spurious proxy dimensions (→ explicit eligibility/module status) · site
effects → artificial strata (→ leave-one-site-out) · missingness patterns masquerading as phenotype (→
stratum artefact audit) · treatment confounding (→ target-trial only) · neural-model overfit (→
secondary sensitivity only).

### Phases S–T — Project-management layer & the decision tree
- **S1** Dual-track evidence: run every major V3 result against {FIML · Bayesian} and
  record confirm/refine/split/merge/reject/cannot-test.
- **S2** Maintain the **prior atlas** (named constructs → expected loadings + plausible cross-loadings +
  module status + cohort coverage) as the bridge from soft ontology to discovered structure.
- **S3** The missing-data doctrine, stated precisely (see [`PIPELINE.md`](PIPELINE.md) §"Missing-data
  doctrine"): **No** completed-data imputation before discovery; **no** mean/KNN/MICE-filled matrix for
  clustering; **yes** to deterministic skip-logic decoding, observed-data likelihood, posterior
  uncertainty over latent dimensions, and explicit missingness models when missingness is informative.
- **S4** Frame stratification as **decision-strata development** rather than natural-subtype testing
  (keep a subtype test as a negative control).
- **S5** Precision psychiatry is contingent on **utility, not elegance** — each final dimension/stratum
  must show ≥1 downstream value (calibration, discrimination, decision-curve net benefit, subgroup
  prognosis, treatment-effect heterogeneity, or clear interpretability).

**T — Decision tree** (adjudicate the whole project): measured & harmonized? → missingness class? →
enough indicators? → stable latent dimension? → transdiagnostic across BP/SZ/DR? → survives `G`? →
predicts/validates externally? → combinations define stable interpretable strata? → strata improve
prognosis or treatment decisions? Only a "yes" at the end earns the **precision-psychiatry decision
framework** label.

---

## Final operating principle

V3 does not ask only *"do FACE patients lie on dimensions rather than DSM categories?"* It asks:

```text
Which transdiagnostic dimensions are empirically supported by patient-level observed data?
Which patients occupy similar validated regions of this dimension space?
Do those regions improve prognosis or treatment-relevant decisions beyond DSM and severity?
```

That is what makes V3 a precision-psychiatry **stratification and decision-modeling** framework rather
than a descriptive dimensional psychopathology analysis.
