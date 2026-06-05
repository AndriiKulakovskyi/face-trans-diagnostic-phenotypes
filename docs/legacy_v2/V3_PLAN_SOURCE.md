# FACE Transdiagnostic V3 Research Plan

> **Source plan (verbatim), kept for full traceability.** The repo-native plan of record — mapped to
> this repository's actual paths and the existing V2 benchmark code — is [`V3_PLAN.md`](V3_PLAN.md).
> Where the proposed repository layout below differs from this repo, `V3_PLAN.md` Phase O governs.

## Patient-level Bayesian / FIML latent modeling with mixed likelihoods and soft priors

**Goal:** transform the current FACE V2 transdiagnostic dimensional analysis into a **clinical-biological precision-psychiatry stratification and decision-modeling framework**. The V3 plan integrates the clinical-biological context: transdiagnostic dimensions are a measurement layer, patient strata are a decision layer, and the 10 candidate constructs are soft hypotheses rather than fixed final dimensions.

V2 is a strong dimensional prototype. V3 should preserve its strengths — harmonization, no naive imputation, baseline anchoring, longitudinal validation, and skepticism about artificial clusters — while replacing the core estimator with a **patient-level observed-data latent model** and reframing stratification as **validated decision regions**, not only natural density-separated subtypes.

---

## 0. Executive transformation

### V2 in one sentence

V2 uses a harmonized FACE BP/SZ/DR baseline dictionary, a strict no-imputation principle, masked pairwise-complete correlations, principal-axis factoring, hierarchical construct aggregation, Schmid–Leiman general-factor testing, and a parallel masked-similarity clustering arm to show a dimensional rather than categorical structure.

### V3 in one sentence

V3 should use **patient-level observed-data likelihood models** — Bayesian sparse bifactor / ESEM-like latent models and FIML SEM benchmarks — to discover and validate transdiagnostic dimensions, project patients into uncertainty-aware dimension space, derive probabilistic patient strata, and test whether those strata improve prognosis and treatment-relevant decision-making.

### Sharp divergence from V2

| Layer | V2 | V3 advancement |
|---|---|---|
| Missingness | No cell imputation; masked pairwise covariance | No naive imputation; observed-data likelihood using each patient's observed cells |
| Estimator | Pairwise-complete correlation + PAF/promax | Patient-level FIML SEM + Bayesian mixed-likelihood latent model |
| Data types | Variables scaled to `[-1, 1]` | Variable-specific likelihoods: Gaussian, Student-t, Bernoulli, ordered logistic, negative binomial, lognormal |
| Clinical priors | Constructs clinically anchored, then data-revised | Construct map becomes a **soft prior loading matrix**, not a hard ontology |
| General factor | Tested by Schmid–Leiman ECV | Estimated directly as latent general burden `G`, with specific dimensions beyond `G` |
| Dimensions | Three correlated axes + standalone dimensions | Empirically adjudicated dimensions: confirmed, split, merged, module, rejected |
| Stratification | Test for natural discrete subtypes | Build **probabilistic decision strata** from validated dimension profiles |
| Prediction | DSM vs dimensions, modest prognosis increment | Model ladder: diagnosis → severity → dimensions → strata → raw missing-aware ML → treatment/causal models |
| Clinical objective | Dimensional account of psychopathology | Precision-psychiatry stratification and decision-support framework |


---

## 0A. Added context: what V3 is actually transforming

### 0A.1 Conceptual contract

V3 is **not** just V2 with a more advanced estimator. It changes the scientific product.

The framework has four layers:

```text
diagnostic cohorts
        -> transdiagnostic dimension discovery
        -> validated patient strata
        -> prognosis / treatment decision models
```

These layers must not be collapsed into one clustering pipeline.

| Layer | Scientific role | V3 output |
|---|---|---|
| Diagnostic cohorts | Entry point and validation metadata | BP / SZ / DR labels, not clustering features |
| Transdiagnostic dimensions | Measurement layer | continuous latent coordinates with uncertainty |
| Patient strata | Segmentation / decision layer | probabilistic risk or treatment-relevant profiles |
| Prognosis / treatment models | Clinical utility layer | calibrated predictions, decision curves, causal estimates where feasible |

The key point is that **dimension discovery describes variation**; stratification **converts validated variation into decision-relevant territories**. A dimension is not necessarily an independent mechanism. It is a latent clinical/biological gradient that explains stable covariance among observed indicators. Dimensions may be correlated. Correlation does not invalidate them; it means the patient map is a correlated clinical coordinate system, not a Cartesian grid.

### 0A.2 The 10 dimensions are a starting ontology, not final truth

The candidate dimensions are:

1. Impulsivity
2. Cognitive flexibility
3. Negative symptoms
4. Anhedonia
5. Metabolism / immunometabolism
6. Sleep / sleep-circadian dysregulation
7. Overall clinical severity
8. Sensory abnormalities
9. Neurodevelopment / neurodevelopmental alterations
10. Suicidality / suicidal behaviours

In V3 these are treated as **candidate constructs**. They define a starting ontology and a soft prior loading map. They are not assumed to exist as 10 final dimensions.

The V3 discovery contract is:

```text
10 candidate constructs
        -> soft prior loading map
        -> missingness-aware hybrid latent modeling
        -> empirical factor structure
        -> confirmed / split / merged / module / proxy / unsupported dimensions
        -> patient strata
        -> prognosis and treatment validation
```

This prevents the naive workflow:

```text
manual tagging -> simple scores -> clustering -> over-interpreted strata
```

V3 must explicitly allow:

- cross-loadings;
- factor splitting;
- factor merging;
- factor rejection;
- diagnosis-specific modules;
- general severity factor;
- method factors;
- missingness sensitivity;
- external longitudinal/prognostic validation.

### 0A.3 What V2 already established and should be preserved

V2 is a strong benchmark and should remain reproducible. Its core facts become V3 priors, stress tests, and baseline comparators.

| V2 element | V2 status | V3 treatment |
|---|---|---|
| Cohorts | FACE-BD, FACE-SZ, FACE-DR, V0 baseline anchor | keep V0 as dimension-discovery anchor; use later visits for validation only |
| Sample structure | large imbalance: BP dominant, SZ medium, DR small and longitudinally thin | use cohort weights, diagnosis-balanced resampling, and cautious DR claims |
| No-imputation principle | no cell imputation; skip-logic zeros decoded | preserve no naive imputation; replace pairwise covariance with observed-data likelihood |
| Harmonized dictionary | clinical, biological, cognitive, suicide, antecedent, substance, social variables | extend dictionary with likelihood family, missingness type, prior loading metadata, and modeling role |
| Measurement model | 196 baseline items -> 95 first-order constructs -> second-order dimensions | convert construct map into soft priors rather than hard aggregation |
| V2 dimensions | internalizing, cognition, cardiometabolic-inflammatory + standalones | retest under patient-level mixed-likelihood model; split/merge/reject if needed |
| General factor | no dominant p-factor by Schmid-Leiman ECV | estimate general burden `G` directly and test whether specific dimensions survive beyond it |
| Stratification | no strong density-separated subtypes beyond DSM | do not interpret this as failure of precision stratification; build decision strata, not only natural clusters |
| Prognosis | modest but real gains for functioning and de-confounded relapse | use as minimum benchmark for V3 model ladder and decision-utility testing |

### 0A.4 V2 findings become V3 hypotheses, not fixed conclusions

V2 reported a three-axis correlated backbone plus orthogonal standalone dimensions. V3 must retest these using patient-level Bayesian/FIML likelihood rather than accepting them as final.

| V2 finding | V3 interpretation | V3 action |
|---|---|---|
| Internalizing axis | likely valid but BP/DR-anchored; SZ measured partly by proxy | model as affective/anhedonic extension unless invariance supports all-cohort status |
| Cognition axis | strongest fully transdiagnostic candidate | keep as core dimension; refine into cognitive flexibility / broader cognition if data support it |
| Cardiometabolic-inflammatory axis | robust biological signal but may combine distinct biology | test split into metabolic load and inflammatory load |
| Suicidality standalone | likely independent patient-risk dimension | model with binary/ordinal/count likelihoods; keep separate from future suicide outcomes |
| Mania/activation standalone | may inform impulsivity/activation proxy | do not call true impulsivity unless direct indicators support it |
| Substance-use standalone | behavioural-risk module | treat as module or covariate depending on outcome |
| Childhood adversity / illness course | developmental-risk proxy | do not overclaim neurodevelopmental alteration |
| No p-factor | important legacy result | retest with explicit `G` factor and mixed likelihoods |
| No natural subtypes | patients lie on continua | derive validated decision regions, not necessarily natural density clusters |
| Symptoms nearly orthogonal to biology | central V2 hypothesis | retest under observed-data latent modeling and posterior uncertainty |

### 0A.5 V3 must distinguish three objects that V2 partially blended

| Object | Definition | Example | V3 handling |
|---|---|---|---|
| Candidate construct | clinical-theoretical idea | anhedonia, impulsivity, sleep | soft prior / hypothesis |
| Empirical latent dimension | stable axis of observed covariance | sleep burden, cognition, inflammation | discovered and validated |
| Patient stratum | recurring decision-relevant profile | high sleep + high suicidality + preserved cognition | derived after dimension validation |

The final V3 dimension set may be smaller or different from the initial 10. That is not a failure. It is the expected output of a hybrid discovery framework.

### 0A.6 FACE-specific dimension eligibility before modeling

V3 must not ask the Bayesian model to discover constructs that FACE does not measure.

| Candidate dimension | V3 starting status | Reason |
|---|---|---|
| Overall clinical severity | core general factor `G` | directly measurable through CGI/GAF/EQ5D/functioning/hospitalization |
| Sleep dysregulation | core if PSQI/common sleep indicators are available | measurable across cohorts, but circadian specificity may be partial |
| Cognitive flexibility / cognition | core | V2 found robust cognition axis; neuropsychology supports it |
| Metabolism / immunometabolism | core, but test split | V2 cardiometabolic-inflammatory may hide metabolic vs inflammatory subdomains |
| Suicidality | core risk dimension | V2 standalone; requires mixed likelihoods and skip-logic aware modeling |
| Neurodevelopmental alterations | proxy/module | use developmental risk/adversity/early-onset proxy, not direct neurodevelopmental biology |
| Anhedonia | extension or module | likely BP/DR measured; SZ may be proxy only |
| Impulsivity | proxy/module | no dedicated common impulsivity measure unless verified |
| Negative symptoms | SZ module unless common direct indicators exist | do not infer from poor functioning alone |
| Sensory abnormalities | unsupported unless direct indicators exist | model cannot discover an unmeasured construct |

### 0A.7 V3 definition of precision-psychiatry stratification

V2 tested whether patients form **natural discrete subtypes**. V3 should test whether patients form **validated decision strata**.

These are not the same.

A V3 stratum can be valid even if the global patient distribution is continuous, provided that the stratum is:

- statistically stable;
- clinically interpretable;
- not a diagnosis/site/missingness artefact;
- prognostically meaningful;
- useful for risk thresholds or treatment-relevant decision-making;
- represented with posterior class probabilities rather than forced hard labels.

Therefore, V3 should not try to reverse the V2 conclusion that transdiagnostic variation is continuous. It should build on it:

```text
continuous dimension space -> probabilistic validated decision regions
```

### 0A.8 V3 success criterion

V3 succeeds only if the advanced pipeline demonstrates at least one of the following beyond V2:

1. more defensible patient-level dimension scores with uncertainty;
2. sharper empirical adjudication of the 10 candidate constructs;
3. replication or refinement of the V2 symptom-biology orthogonality claim under observed-data likelihood;
4. validated probabilistic patient strata that are not just natural clusters;
5. improved prognosis calibration / discrimination / decision utility beyond diagnosis, severity, and V2 dimensions;
6. feasible treatment-relevant modeling under target-trial assumptions.

If V3 only reproduces V2 with more complex code, it has not delivered the precision-psychiatry framework.

---

## 1. V3 scientific objective

### Primary objective

Estimate empirically supported, clinically interpretable, transdiagnostic latent dimensions across FACE BP, SZ, and DR cohorts using patient-level missingness-aware models, then derive validated patient strata that improve prognosis and treatment-relevant decision modeling beyond DSM diagnosis and conventional severity.

### Primary scientific question

Do patient-level latent dimensions and derived patient strata add clinically meaningful predictive or decision value beyond:

```text
diagnosis + age + sex + site + baseline clinical severity
```

### Main V3 claim to target

Use this claim only if supported:

> FACE patients can be represented by a validated set of transdiagnostic latent dimensions, and probabilistic strata derived from these dimensions improve prognosis or treatment-relevant decision modeling beyond DSM diagnosis and standard severity.

### Claims to avoid unless strongly validated

Do **not** claim:

- definitive biological disease subtypes;
- true mechanistic biotypes;
- causal treatment response without target-trial emulation;
- all 10 candidate transdiagnostic constructs are measured;
- sensory abnormalities or negative symptoms if the common FACE variables do not directly measure them.

---

## 2. V3 target architecture

```text
FACE BP/SZ/DR V0 baseline data
        ↓
Data dictionary correction + unit harmonization + skip-logic decoding
        ↓
Missingness atlas + structural/design/informative missingness classification
        ↓
Soft prior loading matrix from V2 constructs + 10 candidate dimensions
        ↓
Patient-level FIML / Bayesian mixed-likelihood latent models
        ↓
Empirical dimension adjudication
confirmed / split / merged / cohort-specific module / unsupported
        ↓
Posterior dimension scores + uncertainty + measurement coverage
        ↓
Probabilistic patient strata / decision regions
        ↓
Strata validation: statistical, clinical, cohort, site, missingness, longitudinal
        ↓
Prognosis models and treatment-relevant target-trial analyses
        ↓
Precision-psychiatry decision framework
```

---

## 3. Step-by-step V3 plan

Each step below states:

1. **What to do**
2. **Why we do it**
3. **How this advances beyond V2**
4. **Expected output**

---

# Phase A — Rebuild the analytical foundation

---

## Step A1 — Freeze V2 and define the V3 fork

### What to do

Freeze the current V2 pipeline and artifacts as the historical benchmark. Create a new V3 branch/repository.

Recommended naming:

```text
face-transdiagnostic-v2-benchmark
face-transdiagnostic-v3-latent-stratification
```

### Why

V2 should remain reproducible and serve as a reference arm. V3 should not silently overwrite the V2 logic.

### Advancement from V2

V2 is no longer the final estimator. It becomes the benchmark against which the stronger patient-level latent models are compared.

### Expected output

```text
git tag: v2_final_benchmark
git branch: v3_patient_level_latent_model
reports/v2_benchmark_summary.md
```

---

## Step A2 — Define the V3 data contract

### What to do

Create a strict data schema with one row per patient at baseline and separate long-format observed-cell tables for modeling.

Minimum patient-level schema:

```text
patient_id
cohort: BP / SZ / DR
site
baseline_date
age
sex
education
diagnosis / DSM arm
V0 variables
follow-up outcome availability
```

Minimum variable-level schema:

```text
variable_name
source_column_BP
source_column_SZ
source_column_DR
instrument
clinical_section
data_type
likelihood_family
allowed_range
unit
higher_score_meaning
candidate_dimensions
primary_expected_dimension
plausible_cross_loadings
covariate_status
outcome_status
missingness_type
structural_zero_rule
use_in_core_model
use_in_extension_model
```

### Why

V3 depends on explicit modeling assumptions. These assumptions must live in a machine-readable dictionary, not only in code comments or manuscript prose.

### Advancement from V2

V2 already has a common dictionary and harmonization logic. V3 extends it with likelihood family, soft-prior metadata, missingness type, and modeling role.

### Expected output

```text
data_dictionary_v3.csv
variable_schema_v3.yaml
likelihood_map_v3.yaml
construct_prior_map_v3.yaml
```

---

## Step A3 — Recheck harmonization, units, and score direction

### What to do

For every variable:

- validate allowed range;
- validate units;
- convert free-text biological values to numeric values where possible;
- apply deterministic parsing rules;
- reverse-code where needed;
- ensure higher score means higher burden/dysfunction unless explicitly documented otherwise.

Examples:

```text
EGF/GAF          → reverse-code for burden models
EQ5D VAS         → reverse-code if used as health-burden indicator
CRP              → parse numeric, then log1p transform
HDL              → reverse-code if used as metabolic burden
TMT-B            → log-transform and residualize on TMT-A/age/education/site
PSQI subscales   → preserve ordinal coding if possible
ISF/C-SSRS       → preserve binary/ordinal/count nature
```

### Why

Latent model loadings are interpretable only if variables are harmonized and oriented coherently.

### Advancement from V2

V2 used type-aware scaling to `[-1, 1]`. V3 keeps deterministic scaling where useful but does not force all variables into one pseudo-continuous metric. The observation likelihood carries the variable type.

### Expected output

```text
face_v3_baseline_wide.parquet
face_v3_baseline_long_observed.parquet
unit_validation_report.html
range_validation_report.csv
score_direction_report.csv
```

---

## Step A4 — Preserve skip-logic decoding but separate it from imputation

### What to do

Keep V2's structural-zero decoding for gated variables.

Example:

```text
if suicide_attempt_ever == 0 and attempt_count is blank:
    attempt_count = 0  # structural zero
```

But never overwrite observed values and never create values where the gate is unknown.

### Why

Skip-logic blanks are not the same as missing observations. Some blanks mean clinically meaningful zero.

### Advancement from V2

This is a direct carry-over from V2. V3 formalizes it in the data contract and downstream likelihood.

### Expected output

```text
skip_logic_rules.yaml
structural_zero_audit.csv
coverage_before_after_skip_decoding.csv
```

---

# Phase B — Missingness atlas and measurement eligibility

---

## Step B1 — Build the missingness matrix

### What to do

Create:

```text
R_ij = 1 if variable j observed for patient i
R_ij = 0 if variable j missing for patient i
```

Summarize missingness by:

```text
cohort
site
year / baseline wave
age band
sex
severity band
variable block
instrument
dimension candidate
```

### Why

In FACE, missingness is not noise. It reflects cohort design, site practice, questionnaire routing, clinical feasibility, and sometimes severity.

### Advancement from V2

V2 avoided imputation and used masked pairwise support. V3 makes missingness itself an explicit object to audit and potentially model.

### Expected output

```text
missingness_matrix.parquet
missingness_by_cohort.csv
missingness_by_site.csv
missingness_by_variable_block.csv
missingness_heatmaps/
```

---

## Step B2 — Classify missingness mechanism per variable

### What to do

Assign every variable to one primary missingness class:

| Class | Meaning | Example |
|---|---|---|
| Structural | not collected in one or more cohorts | mood scale absent in SZ |
| Design | collected only at certain sites/years | lab module variation |
| Clinical skip | only asked if gate item positive | suicide-attempt details |
| Sporadic | isolated unstructured missingness | missing lab value |
| Informative | missingness likely related to severity/function | neuropsych test not completed |
| Outcome-related | only available at follow-up | relapse, future GAF |

### Why

Different missingness types require different handling. Structural missingness should not be treated as if a value could have been observed.

### Advancement from V2

V2 used pairwise support and coverage gates. V3 uses missingness classification to decide whether a variable is eligible for core dimensions, extension modules, or outcome modeling.

### Expected output

```text
variable_missingness_taxonomy.csv
structural_missingness_variables.csv
informative_missingness_candidates.csv
```

---

## Step B3 — Model observation probability

### What to do

For key variables or blocks, fit missingness models:

```text
Observed_j ~ cohort + site + age + sex + education + diagnosis + severity_proxy
```

Use logistic regression or missing-aware tree models.

### Why

This tests whether missingness is mainly design-driven, cohort-driven, site-driven, or severity-driven.

### Advancement from V2

V2 reported per-pair support and missingness diagnostics. V3 estimates explicit observation mechanisms and uses them in sensitivity analyses.

### Expected output

```text
missingness_mechanism_models/
observation_probability_by_variable.csv
highly_informative_missingness_report.md
```

---

## Step B4 — Define core, extension, module, and excluded variables

### What to do

Classify variables and candidate dimensions:

| Status | Definition |
|---|---|
| Core all-cohort | enough direct indicators in BP/SZ/DR |
| Partial extension | valid in two cohorts or a subset |
| Diagnosis-specific module | valid mainly in one cohort |
| Covariate | adjust for but not an indicator |
| Outcome | not used in baseline dimensions if predicting it later |
| Excluded | invalid or too sparse |

### Why

A dimension can only be discovered if it is actually measured. This prevents overclaiming unsupported dimensions.

### Advancement from V2

V2 qualified internalizing and cardiometabolic transdiagnosticity after analysis. V3 makes measurement eligibility explicit before fitting and uses it in model design.

### Expected output

```text
feature_eligibility_v3.csv
core_dimension_indicator_pool.csv
extension_module_indicator_pool.csv
excluded_variable_justification.csv
```

---

# Phase C — Soft-prior construct map

---

## Step C1 — Convert V2 constructs into a soft prior loading matrix

### What to do

Use the V2 construct map and the 10 candidate dimensions to create:

```text
T_jk = prior relationship between variable/construct j and dimension k
```

Statuses:

```text
primary_expected_indicator
plausible_cross_loading
unlikely_loading
covariate_only
invalid_indicator
outcome_only
```

Example:

| Variable/construct | Primary prior | Plausible cross-loading |
|---|---|---|
| PSQI latency | sleep dysregulation | general severity |
| TMT-B residual | cognitive flexibility | developmental proxy, severity |
| CRP | inflammation | metabolic load |
| BMI/waist | metabolic load | inflammation |
| CTQ | developmental/adversity | suicidality, severity |
| ISF attempt count | suicidality | impulsivity proxy |
| QIDS anhedonia | anhedonia | severity, suicidality |

### Why

The goal is not to hard-tag variables. The goal is to let theory guide discovery while allowing the data to split, merge, reject, or cross-load dimensions.

### Advancement from V2

V2 had clinically anchored constructs revised by data. V3 turns this into an explicit Bayesian prior system.

### Expected output

```text
soft_loading_prior_matrix.csv
soft_loading_prior_matrix.yaml
prior_map_review_report.md
```

---

## Step C2 — Decide which of the 10 candidate dimensions are testable

### What to do

For each candidate dimension, classify it as:

```text
confirmed candidate for core model
candidate for extension model
proxy-only candidate
not testable with current common variables
```

Recommended initial V3 expectation:

| Candidate dimension | V3 status |
|---|---|
| Overall clinical severity | core, modeled as general factor `G` |
| Sleep / sleep-circadian dysregulation | core for sleep; circadian extension if variables exist |
| Cognitive flexibility | core, likely broader cognition/flexibility |
| Metabolism / immunometabolism | core, likely split into metabolic and inflammatory load |
| Suicidality / suicidal behaviours | core, preferably mixed binary/ordinal/count model |
| Neurodevelopment / neurodevelopmental alterations | proxy only: developmental risk / adversity / early onset |
| Anhedonia | BP/DR extension unless direct all-cohort indicators exist |
| Impulsivity | proxy only unless dedicated impulsivity scale exists |
| Negative symptoms | SZ-specific module unless direct all-cohort indicators exist |
| Sensory abnormalities | not testable unless direct sensory/perceptual indicators exist |

### Why

This protects the scientific validity of the project. Unsupported dimensions should not be forced into the core model.

### Advancement from V2

V2 reported standalone dimensions such as mania and suicidality. V3 formally adjudicates each candidate construct against measurement availability.

### Expected output

```text
candidate_dimension_status_table.md
core_dimensions_v3.yaml
extension_dimensions_v3.yaml
unsupported_dimensions_v3.md
```

---

# Phase D — V2 benchmark replication under V3 data

---

## Step D1 — Re-run the V2 masked estimator as benchmark

### What to do

Re-run the V2 logic on the re-harmonized V3 data:

```text
masked pairwise-complete correlation
nearest positive-definite repair
PAF extraction
promax / varimax rotation
Thomson factor scores on observed support
Schmid–Leiman ECV
split-half congruence
bootstrap stability
```

### Why

Before replacing the estimator, confirm that data curation changes did not destroy the V2 structure.

### Advancement from V2

This is not the V3 primary model. It is a benchmark and reproducibility control.

### Expected output

```text
v2_replicated_loadings.csv
v2_replicated_scores.parquet
v2_vs_v3_data_change_report.md
v2_benchmark_figures/
```

---

## Step D2 — Establish benchmark metrics

### What to do

Record benchmark values:

```text
number of reproduced axes
loading congruence
ECV / general factor strength
symptom-biology correlations
cohort residualized congruence
longitudinal coherence
prediction increments over DSM
stratification density-cluster tests
```

### Why

V3 must show whether stronger estimators confirm, refine, or contradict the V2 findings.

### Advancement from V2

V2 results become falsifiable hypotheses under a stronger likelihood-based framework.

### Expected output

```text
v2_benchmark_metrics.json
v2_claims_to_retest.md
```

---

# Phase E — FIML latent model benchmark

---

## Step E1 — Build a FIML SEM/ESEM benchmark model

### What to do

Fit patient-level FIML models on approximately continuous variables or construct scores:

Candidate models:

```text
Model E1: one general factor
Model E2: V2 three-axis model
Model E3: V2 three axes + standalone dimensions
Model E4: candidate 6–8 dimension model
Model E5: bifactor model with general burden + specifics
Model E6: ESEM-like target model with cross-loadings
```

### Why

FIML uses patient-level observed data and is a bridge between V2 masked correlations and the full Bayesian mixed-likelihood model.

### Advancement from V2

V2 stated that complete-data ML was precluded by missingness. V3 clarifies that **complete-data ML is precluded**, but **observed-data FIML is compatible with no naive imputation**.

### Expected output

```text
fiml_model_fit_indices.csv
fiml_loadings.csv
fiml_factor_scores.parquet
fiml_model_comparison.md
```

---

## Step E2 — Use FIML to test general severity versus specific dimensions

### What to do

Fit a bifactor model:

```text
X_ij = nu_j + lambda_jG * G_i + sum_k lambda_jk * D_ik + epsilon_ij
```

where:

```text
G_i = general clinical burden
D_ik = sleep, cognition, metabolic, inflammatory, developmental, suicidality, etc.
```

### Why

This tests whether domain-specific dimensions exist beyond global severity.

### Advancement from V2

V2 tested a general factor post hoc using Schmid–Leiman ECV. V3 estimates the general factor directly as part of the patient-level model.

### Expected output

```text
general_factor_loadings.csv
specific_factor_loadings_after_G.csv
specific_dimension_survival_report.md
```

---

# Phase F — Bayesian sparse bifactor model with mixed likelihoods

---

## Step F1 — Specify the core Bayesian generative model

### What to do

Define the patient-level latent structure:

```text
G_i ~ Normal(0, 1)
D_ik ~ Normal(0, 1)
```

Observation model:

```text
eta_ij = alpha_j + lambda_jG * G_i + sum_k lambda_jk * D_ik + covariates
X_ij ~ likelihood_j(eta_ij, parameters_j)
```

### Why

This models patient-level latent dimensions directly from observed data without creating a completed dataset.

### Advancement from V2

V2 estimated covariance first. V3 models the observed variables themselves.

### Expected output

```text
bayesian_model_specification.md
bayesian_model_code/
```

---

## Step F2 — Use mixed likelihoods by variable type

### What to do

Assign likelihoods:

| Variable type | Likelihood |
|---|---|
| standardized continuous clinical score | Gaussian or Student-t |
| skewed biological marker | lognormal or Student-t after log transform |
| ordinal questionnaire item | ordered logistic / ordered probit |
| binary clinical/suicide item | Bernoulli-logit |
| count variable | negative binomial or Poisson |
| zero-heavy count | zero-inflated negative binomial if justified |
| longitudinal outcome | separate outcome model, not baseline dimension model |

### Why

Clinical scales, labs, ordinal items, binary history variables, and suicide counts are not the same statistical object.

### Advancement from V2

V2 scaled all variables to a common numeric range. V3 preserves distributional meaning.

### Expected output

```text
likelihood_assignment_table.csv
mixed_likelihood_model.py
likelihood_diagnostics_report.md
```

---

## Step F3 — Encode soft loading priors

### What to do

For each loading `lambda_jk`, set priors according to the soft prior matrix:

```text
primary expected loading:
    lambda_jk ~ Normal(0.6, 0.3)

plausible cross-loading:
    lambda_jk ~ Normal(0.0, 0.25)

unlikely loading:
    lambda_jk ~ Normal(0.0, 0.05)
```

Alternative for stronger discovery:

```text
lambda_jk ~ horseshoe / spike-and-slab shrinkage prior
```

### Why

The 10 candidate dimensions become hypotheses, not forced labels. The data can retain unexpected cross-loadings or shrink unsupported ones to zero.

### Advancement from V2

V2 had clinical anchors revised by EFA. V3 formalizes this as probabilistic soft constraints.

### Expected output

```text
loading_prior_summary.csv
posterior_loading_summary.csv
prior_vs_posterior_loading_shift_report.md
```

---

## Step F4 — Include general burden factor without blindly orthogonalizing everything

### What to do

Estimate `G_i` as a general burden factor and allow specific dimensions to explain residual domain variation.

Recommended initial model:

```text
G_i independent of D_ik for identifiability
D_ik independent in first pass
correlated D_ik in sensitivity model
```

### Why

Psychiatric data often contain a global severity gradient. If not modeled, it can dominate dimensions and strata.

### Advancement from V2

V2 concluded no dominant general factor under masked-correlation Schmid–Leiman testing. V3 retests this directly under patient-level likelihood.

### Expected output

```text
posterior_G_scores.parquet
posterior_specific_dimension_scores.parquet
G_dominance_report.md
```

---

## Step F5 — Add diagnosis, site, age, sex, education as covariates, not dimensions

### What to do

Use covariates in the measurement model where needed:

```text
eta_ij = alpha_j + beta_j_cohort + beta_j_site + beta_j_age + beta_j_sex + lambda_jG*G_i + Lambda_j*D_i
```

Do not use diagnosis as a dimension indicator.

### Why

This separates mean differences due to diagnosis/site from within-patient latent dimensions.

### Advancement from V2

V2 tested cohort confounding after deriving axes. V3 can adjust for cohort/site effects inside the measurement model and then still validate post hoc.

### Expected output

```text
covariate_adjusted_loadings.csv
cohort_effects_on_indicators.csv
site_effects_report.md
```

---

## Step F6 — Decide whether to model missingness explicitly

### What to do

Start with an observed-likelihood MAR model:

```text
p(X_observed | latent dimensions, parameters)
```

Then run sensitivity models for informative missingness:

```text
R_ij ~ Bernoulli(logit^{-1}(a_j + gamma_jG*G_i + gamma_jk*D_ik + cohort + site))
```

### Why

Some missingness may be informative, especially neuropsych completion or biological sampling.

### Advancement from V2

V2 avoided filling missing cells. V3 can additionally test whether missingness patterns are themselves related to latent dimensions.

### Expected output

```text
mar_model_results/
missingness_sensitivity_results/
latent_dimension_shift_under_missingness_model.csv
```

---

## Step F7 — Fit and diagnose the Bayesian model

### What to do

Fit initially on a smaller core model, then scale up.

Recommended sequence:

```text
F7.1 Core model: severity + sleep + cognition + metabolic + inflammatory + developmental + suicidality
F7.2 Add affective/anhedonia extension
F7.3 Add impulsivity proxy if justified
F7.4 Add SZ negative-symptom module if direct indicators exist
F7.5 Sensitivity: correlated specific factors
```

Diagnostics:

```text
R-hat
effective sample size
divergences
posterior predictive checks
prior predictive checks
loading stability
posterior uncertainty by cohort
```

### Why

Complex Bayesian models can look plausible while being poorly identified. Diagnostics are mandatory.

### Advancement from V2

V2 used deterministic factor extraction and bootstrap stability. V3 adds posterior uncertainty and model diagnostics.

### Expected output

```text
bayesian_idata_core.nc
bayesian_posterior_diagnostics.html
posterior_predictive_checks/
posterior_dimension_scores.parquet
posterior_loading_summaries.csv
```

---

# Phase G — Model comparison and dimension discovery

---

## Step G1 — Compare competing latent structures

### What to do

Compare:

```text
1-factor general burden model
V2 three-axis model
V2 three-axis + standalone model
10 candidate construct model
6–8 empirically supported candidate model
bifactor model
ESEM-like cross-loading model
Bayesian sparse model
```

Metrics:

```text
predictive log likelihood / approximate LOO
posterior predictive checks
loading interpretability
factor reliability
dimension coverage
measurement invariance
stability under resampling
clinical interpretability
outcome validity
```

### Why

The correct number and meaning of dimensions should be adjudicated, not assumed.

### Advancement from V2

V2 locked dimensionality by split-half congruence and bootstrapping. V3 adds likelihood-based model comparison and posterior predictive validation.

### Expected output

```text
latent_model_comparison_table.csv
model_selection_scorecard.md
preferred_dimension_model.md
```

---

## Step G2 — Adjudicate each candidate dimension

### What to do

Classify every candidate dimension:

| Decision | Meaning |
|---|---|
| Confirmed | stable all-cohort dimension with direct indicators |
| Split | candidate divides into multiple empirical dimensions |
| Merged | candidate absorbed by another dimension/general burden |
| Module | valid only in a cohort/subset |
| Proxy-only | indirect but useful proxy construct |
| Unsupported | insufficient evidence or no valid indicators |

### Why

Dimension discovery is not simply recovering all 10 proposed dimensions. It is an empirical negotiation between theory and data.

### Advancement from V2

V2 reported three axes plus standalone dimensions. V3 produces an explicit adjudication table for the 10 candidate precision-psychiatry constructs.

### Expected output

```text
dimension_adjudication_table.md
final_empirical_dimensions_v3.yaml
unsupported_constructs_report.md
```

---

## Step G3 — Retest the V2 headline claims

### What to do

Under the V3 model, retest:

```text
symptom-biology orthogonality
general-factor strength
cognition as fully transdiagnostic
cardiometabolic structure
internalizing measurement gap in SZ
standalone suicidality / mania / substance-use
no natural density-separated subtypes
modest prognosis increment over DSM
```

### Why

V2's conclusions are strong. V3 should either confirm, refine, or downgrade them under a stronger estimator.

### Advancement from V2

V3 makes the original claims more defensible if they survive patient-level mixed-likelihood modeling.

### Expected output

```text
v2_claims_retested_under_v3.md
v2_v3_concordance_table.csv
claims_confirmed_refined_rejected.md
```

---

# Phase H — Measurement invariance and transdiagnostic validity

---

## Step H1 — Test whether dimensions are transdiagnostic

### What to do

For each dimension, evaluate:

```text
coverage by BP/SZ/DR
loading stability by cohort
posterior factor score reliability by cohort
cohort-residualized loadings
leave-one-cohort-out congruence
within-cohort covariance reproduction
```

### Why

A dimension is not transdiagnostic merely because patients from three cohorts were pooled.

### Advancement from V2

V2 did cohort-residualization and within-cohort checks. V3 performs the same logic at the patient-level latent-model stage.

### Expected output

```text
transdiagnostic_validity_by_dimension.csv
cohort_invariance_report.md
leave_cohort_out_results.csv
```

---

## Step H2 — Test measurement invariance / differential item functioning

### What to do

Test whether indicators behave similarly across:

```text
BP / SZ / DR
sex
age bands
site
assessment year
```

Methods:

```text
multi-group FIML SEM
Bayesian group-specific loading/intercept models
posterior DIF checks
partial invariance models
```

### Why

The same observed score may mean different things in different diagnoses or sites.

### Advancement from V2

V2 tested whether axes were cohort artifacts. V3 asks whether the measurement model itself is comparable across cohorts.

### Expected output

```text
measurement_invariance_report.md
DIF_indicator_table.csv
partial_invariance_model_results/
```

---

# Phase I — Patient-level dimension scoring

---

## Step I1 — Generate posterior dimension scores

### What to do

For each patient, estimate:

```text
G_general_mean
G_general_sd
D_sleep_mean
D_sleep_sd
D_cognition_mean
D_cognition_sd
D_metabolic_mean
D_metabolic_sd
D_inflammatory_mean
D_inflammatory_sd
D_developmental_mean
D_developmental_sd
D_suicidality_mean
D_suicidality_sd
...
```

Also store:

```text
number of observed indicators per dimension
posterior reliability proxy
cohort
site
baseline severity
missingness pattern
```

### Why

Precision psychiatry needs patient-level profiles, not only group-level factor loadings.

### Advancement from V2

V2 computed Thomson factor scores on observed support. V3 computes posterior patient scores with uncertainty.

### Expected output

```text
patient_dimension_scores_v3.parquet
patient_dimension_uncertainty_v3.parquet
dimension_score_coverage_report.csv
```

---

## Step I2 — Create the V3 phenotype atlas

### What to do

For each final empirical dimension, document:

```text
name
definition
main indicators
cross-loadings
coverage by cohort
direction
state/trait/fixed-historical status
posterior reliability
measurement limitations
clinical interpretation
```

### Why

This is the bridge from statistical latent dimensions to clinical use.

### Advancement from V2

V2 released a predictive-feature atlas. V3 releases a probabilistic phenotype atlas with uncertainty and model provenance.

### Expected output

```text
docs/PHENOTYPE_ATLAS_V3.md
reports/phenotype_profiles/
```

---

# Phase J — Patient stratification as precision decision regions

---

## Step J1 — Reframe stratification objective

### What to do

Do not only test for natural density-separated subtypes. Define strata as:

```text
validated recurrent patient profiles in dimension space
```

Possible types:

```text
latent profiles
Bayesian mixture classes
risk territories
outcome-enriched profiles
treatment-relevant profiles
```

### Why

A clinically useful stratum does not need to be a natural disease subtype. It must be stable, interpretable, and useful for prognosis or decision-making.

### Advancement from V2

V2 concluded continuous variation with no novel discrete subtypes. V3 accepts that but asks a different precision-medicine question: are there useful validated regions of the dimension space?

### Expected output

```text
stratification_objective_v3.md
strata_definition_rules.md
```

---

## Step J2 — Fit probabilistic strata on dimension scores

### What to do

Use patient dimension scores as inputs:

```text
[G, sleep, cognition, metabolic, inflammatory, developmental, suicidality, ...]
```

Candidate models:

```text
latent profile analysis
Bayesian Gaussian mixture model
mixture-of-factor model
consensus clustering on posterior scores
risk-threshold decision regions
```

Keep soft assignments:

```text
P(stratum = 1)
P(stratum = 2)
...
assignment entropy
```

### Why

Hard clusters are unrealistic in psychiatry. Patients can sit between profiles.

### Advancement from V2

V2 tested HDBSCAN, masked kernels, silhouettes, and ARI. V3 uses probabilistic strata and evaluates their clinical utility.

### Expected output

```text
strata_model_K2_to_K6_results.csv
patient_strata_probabilities.parquet
strata_entropy.csv
strata_profile_plots/
```

---

## Step J3 — Compare dimension-based strata to direct raw-data stratification

### What to do

Run direct stratification only as sensitivity:

```text
raw observed-data mixture model
Gower/hierarchical clustering
masked autoencoder + clustering
HDBSCAN on construct scores
```

Compare to dimension-based strata.

### Why

This tests whether the dimension-first strategy misses unexpected structure.

### Advancement from V2

V2's direct masked-similarity stratification remains useful, but is now secondary.

### Expected output

```text
direct_vs_dimension_strata_comparison.md
cluster_concordance_ARI.csv
raw_stratification_sensitivity_results/
```

---

# Phase K — Strata validation

---

## Step K1 — Statistical stability validation

### What to do

Assess:

```text
bootstrap stability
balanced diagnosis subsampling
leave-one-site-out stability
posterior class stability
sensitivity to number of dimensions
sensitivity to excluding high-missingness patients
```

### Why

A stratum is not valid because the algorithm found it once.

### Advancement from V2

V2 used bootstrap and structure tests. V3 extends this to probabilistic strata and posterior uncertainty.

### Expected output

```text
strata_stability_report.md
bootstrap_ARI_matrix.csv
leave_site_out_strata_results.csv
```

---

## Step K2 — Artefact checks

### What to do

For each stratum, report:

```text
cohort distribution
site distribution
sex / age / education distribution
missingness burden
number of observed indicators
medication exposure
follow-up availability
```

Reject or relabel strata that are mostly:

```text
one site
one cohort
one missingness pattern
one measurement-completeness level
one medication artefact
```

### Why

Precision psychiatry strata must not be assessment-protocol strata.

### Advancement from V2

V2 checked cohort confounding. V3 checks cohort, site, missingness, medication, and follow-up availability at the stratum level.

### Expected output

```text
strata_artefact_audit.csv
strata_validity_red_flags.md
```

---

## Step K3 — Clinical profile validation

### What to do

For every stratum, describe:

```text
dimension profile
main raw clinical characteristics
functioning profile
suicide risk profile
cognitive profile
biological profile
course/chronicity profile
```

### Why

A stratum must be clinically interpretable.

### Advancement from V2

V2 mainly answered natural subtype vs continuum. V3 creates clinically interpretable decision profiles.

### Expected output

```text
strata_clinical_profile_table.csv
strata_radar_plots/
strata_clinical_naming_report.md
```

---

## Step K4 — Longitudinal and outcome validation

### What to do

Test whether baseline strata predict future outcomes not used to define them:

```text
future GAF / EGF / functioning
FAST disability
relapse from remission
hospitalization
future suicidal ideation / attempt
medication nonadherence
metabolic worsening
follow-up dropout
```

Models:

```text
Y_future ~ diagnosis + age + sex + site + baseline severity + dimensions + strata probabilities
```

### Why

A stratum becomes clinically meaningful only when it predicts or explains something outside the variables used to create it.

### Advancement from V2

V2 tested dimensions versus DSM. V3 tests dimensions plus strata versus DSM and baseline severity.

### Expected output

```text
strata_outcome_validation.csv
longitudinal_strata_survival_models/
strata_incremental_value_report.md
```

---

# Phase L — Prognosis modeling framework

---

## Step L1 — Build the prognosis model ladder

### What to do

For each outcome, compare nested models:

```text
M0 = age + sex + site
M1 = M0 + DSM diagnosis
M2 = M1 + conventional severity indicators
M3 = M2 + posterior dimension scores
M4 = M3 + posterior stratum probabilities
M5 = M4 + selected raw variables + missingness indicators
M6 = M5 + early-course trajectory information, if available
```

### Why

This quantifies the incremental value of dimensions and strata.

### Advancement from V2

V2 compared DSM and dimensions. V3 adds conventional severity, uncertainty-aware dimensions, strata probabilities, raw missing-aware models, and trajectory models.

### Expected output

```text
prognosis_model_ladder_results.csv
incremental_value_summary.md
```

---

## Step L2 — Use missingness-aware prediction algorithms

### What to do

Use algorithms that can handle missing values directly or via explicit missingness indicators:

```text
CatBoost
LightGBM
XGBoost
sklearn HistGradientBoosting
scikit-survival / XGBoost survival models
penalized logistic/Cox models on dimension scores as classical benchmarks
```

### Why

Imputing high-missingness raw variables before prediction risks distorting the signal. Tree-based missing-aware learners are better for raw-variable prediction layers.

### Advancement from V2

V2 used ridge, logistic regression, and histogram gradient boosting. V3 extends this into a full missing-aware model stack.

### Expected output

```text
model_performance_by_outcome.csv
model_calibration_by_diagnosis.csv
model_calibration_by_site.csv
feature_importance_and_shap_reports/
```

---

## Step L3 — Validate prognosis models correctly

### What to do

Use:

```text
nested cross-validation
GroupKFold by patient for longitudinal intervals
leave-one-site-out validation
diagnosis-stratified validation
temporal validation if dates allow
bootstrap confidence intervals
calibration curves
decision curve analysis
```

Report:

```text
AUROC
AUPRC for rare outcomes
Brier score
calibration slope/intercept
C-index for survival
integrated Brier score
net benefit
subgroup performance
```

### Why

A pooled AUC is insufficient in an imbalanced BP-dominant dataset.

### Advancement from V2

V2 already used cross-validated predictive validation. V3 upgrades validation to clinical prediction-model standards.

### Expected output

```text
validation_design.md
prognosis_performance_tables/
calibration_figures/
decision_curve_figures/
```

---

# Phase M — Treatment and decision modeling

---

## Step M1 — Separate prognosis from treatment effect

### What to do

State explicitly:

```text
Prognosis model: Who is likely to worsen?
Treatment model: Which treatment strategy would improve this patient's outcome?
```

Do not infer treatment effects from predictive associations.

### Why

FACE is observational. Treatment is confounded by indication.

### Advancement from V2

V2 focused on prognosis. V3 becomes a decision-modeling framework but must do so with causal discipline.

### Expected output

```text
prognosis_vs_treatment_scope.md
causal_claims_policy.md
```

---

## Step M2 — Define target-trial emulations for selected treatment questions

### What to do

For each treatment question, define:

```text
eligibility
time zero
treatment strategies
grace period
follow-up window
outcome
causal contrast
confounders
censoring rules
positivity checks
```

Candidate examples:

```text
lithium initiation vs no initiation in BP-eligible patients
LAI vs oral antipsychotic in SZ, if available
antipsychotic adherence vs nonadherence in SZ
depression augmentation vs switching, if available
```

### Why

Target-trial emulation is required before treatment-effect modeling.

### Advancement from V2

V2 did not attempt causal treatment modeling. V3 can add it as a later stage once dimensions and strata are validated.

### Expected output

```text
target_trial_protocols/
positivity_reports/
confounder_sets.yaml
```

---

## Step M3 — Estimate heterogeneous treatment effects by stratum

### What to do

Use methods such as:

```text
inverse probability weighting
augmented inverse probability weighting
double machine learning
causal forests / CATE models
stratum-by-treatment interactions
```

Use strata as possible effect modifiers:

```text
Outcome ~ treatment + strata + treatment:strata + confounders
```

### Why

The precision-psychiatry question is whether strata identify patients with different expected benefit-risk profiles.

### Advancement from V2

This is the major move from dimensional description to decision support.

### Expected output

```text
heterogeneous_treatment_effect_results.csv
stratum_treatment_interaction_report.md
causal_sensitivity_analysis.md
```

---

# Phase N — Clinical interpretation and reporting

---

## Step N1 — Build clinical profiles for dimensions and strata

### What to do

For each final dimension and stratum, produce:

```text
clinical label
core indicators
posterior loading profile
coverage
uncertainty
cohort representation
outcome risks
possible clinical actionability
limitations
```

### Why

The framework must be interpretable to psychiatrists, not only data scientists.

### Advancement from V2

V2 was a dimensional measurement paper. V3 becomes a precision-psychiatry decision framework.

### Expected output

```text
docs/DIMENSION_CARDS_V3.md
docs/STRATUM_CARDS_V3.md
```

---

## Step N2 — Produce model cards and reporting checklists

### What to do

For every prognosis/treatment model, document:

```text
intended use
training population
outcome definition
features used
missingness handling
performance
calibration
subgroup performance
limitations
not-for-use conditions
```

### Why

Clinical decision tools require transparent documentation.

### Advancement from V2

V2 reports predictive validity. V3 documents decision-readiness and limitations.

### Expected output

```text
model_cards/
TRIPOD_AI_checklist.md
PROBAST_AI_risk_of_bias.md
```

---

# Phase O — Repository and implementation structure

---

## Recommended repository layout

```text
face_v3_precision_psychiatry/
│
├── configs/
│   ├── data_schema.yaml
│   ├── variable_likelihoods.yaml
│   ├── soft_loading_priors.yaml
│   ├── dimensions.yaml
│   ├── outcomes.yaml
│   ├── models.yaml
│
├── data/
│   ├── raw/                 # secure, not committed
│   ├── interim/             # secure, not committed
│   ├── processed/           # secure, not committed
│
├── src/
│   ├── data/
│   │   ├── harmonize.py
│   │   ├── validate_schema.py
│   │   ├── skip_logic.py
│   │   └── build_baseline.py
│   │
│   ├── missingness/
│   │   ├── atlas.py
│   │   ├── mechanism_models.py
│   │   └── reports.py
│   │
│   ├── priors/
│   │   ├── build_loading_matrix.py
│   │   └── validate_priors.py
│   │
│   ├── v2_benchmark/
│   │   ├── masked_correlation.py
│   │   ├── paf.py
│   │   └── schmid_leiman.py
│   │
│   ├── fiml/
│   │   ├── sem_models.py
│   │   └── invariance.py
│   │
│   ├── bayesian/
│   │   ├── model_core.py
│   │   ├── mixed_likelihoods.py
│   │   ├── posterior_scores.py
│   │   └── diagnostics.py
│   │
│   ├── dimensions/
│   │   ├── adjudication.py
│   │   └── phenotype_atlas.py
│   │
│   ├── strata/
│   │   ├── latent_profiles.py
│   │   ├── bayesian_mixture.py
│   │   ├── validation.py
│   │   └── stratum_cards.py
│   │
│   ├── prognosis/
│   │   ├── model_ladder.py
│   │   ├── survival.py
│   │   ├── calibration.py
│   │   └── decision_curves.py
│   │
│   ├── treatment/
│   │   ├── target_trial.py
│   │   ├── causal_models.py
│   │   └── positivity.py
│   │
│   └── reporting/
│       ├── tables.py
│       ├── figures.py
│       └── model_cards.py
│
├── notebooks/
│   ├── 01_v3_data_audit.ipynb
│   ├── 02_missingness_atlas.ipynb
│   ├── 03_soft_prior_map.ipynb
│   ├── 04_v2_benchmark_replication.ipynb
│   ├── 05_fiml_models.ipynb
│   ├── 06_bayesian_latent_model.ipynb
│   ├── 07_dimension_adjudication.ipynb
│   ├── 08_patient_strata.ipynb
│   ├── 09_prognosis_models.ipynb
│   └── 10_treatment_models.ipynb
│
├── reports/
│   ├── missingness_atlas/
│   ├── latent_models/
│   ├── dimension_adjudication/
│   ├── strata_validation/
│   ├── prognosis/
│   └── treatment/
│
├── docs/
│   ├── PHENOTYPE_ATLAS_V3.md
│   ├── DIMENSION_CARDS_V3.md
│   ├── STRATUM_CARDS_V3.md
│   └── METHODS_V3.md
│
├── tests/
│   ├── test_schema.py
│   ├── test_skip_logic.py
│   ├── test_likelihood_assignments.py
│   ├── test_prior_matrix.py
│   └── test_no_outcome_leakage.py
│
├── pyproject.toml
├── dvc.yaml
└── README.md
```

---

# Phase P — Acceptance criteria

---

## Dimension acceptance criteria

A dimension is accepted into the V3 core model only if it satisfies most of:

```text
sufficient direct indicators
acceptable coverage in BP/SZ/DR
stable posterior loadings
not explained only by site/cohort/missingness
theoretically interpretable
posterior score reliability acceptable
invariance or partial invariance defensible
longitudinal or external validity signal
```

---

## Stratum acceptance criteria

A patient stratum is accepted only if it is:

```text
statistically stable
probabilistically assignable
clinically interpretable
not a site artifact
not a missingness artifact
not merely DSM diagnosis
associated with future outcome or decision value
```

---

## Prognosis model acceptance criteria

A prognosis model is useful only if it improves over DSM/severity by:

```text
calibration
AUPRC or AUROC, depending on outcome
Brier score
decision-curve net benefit
subgroup performance
site/dataset robustness
```

A tiny pooled AUC gain without calibration or decision benefit is not enough.

---

## Treatment model acceptance criteria

A treatment model is interpretable only if:

```text
target trial is defined
positivity holds within relevant strata
confounding adjustment is plausible
sensitivity analyses are reported
causal language is limited to the design
```

---

# Phase Q — Concrete V3 deliverables

## Core scientific deliverables

```text
1. V3 data dictionary and missingness atlas
2. V2 benchmark replication report
3. FIML latent model benchmark
4. Bayesian mixed-likelihood sparse bifactor model
5. Dimension adjudication report
6. V3 phenotype atlas
7. Probabilistic patient strata
8. Strata validation report
9. Prognosis model ladder report
10. Treatment-target-trial feasibility report
```

## Manuscript-level outputs

Suggested manuscript sequence:

### Paper 1 — Measurement model

```text
Patient-level missingness-aware latent modeling of transdiagnostic dimensions across FACE BP/SZ/DR cohorts
```

Main claim:

```text
V3 confirms/refines the empirical dimensional structure under patient-level mixed-likelihood modeling.
```

### Paper 2 — Stratification

```text
From transdiagnostic dimensions to validated patient strata in FACE cohorts
```

Main claim:

```text
Probabilistic strata derived from validated dimensions define clinically interpretable risk profiles beyond DSM.
```

### Paper 3 — Prognosis / decision modeling

```text
Precision-psychiatry decision modeling using transdiagnostic dimensions and patient strata
```

Main claim:

```text
Dimensions and strata improve selected prognosis or treatment-relevant decisions beyond diagnosis and conventional severity.
```

---

# Phase R — Risk register

| Risk | Why it matters | Mitigation |
|---|---|---|
| Bayesian model too slow | 9K patients, many variables | start with core model; use variational inference or Pyro later |
| Model not identifiable | too many factors/cross-loadings | sparse priors, staged models, strong diagnostics |
| General severity dominates | strata become mild/moderate/severe | bifactor model with `G` and specific residual dimensions |
| BP dominance | factor structure may be bipolar-driven | diagnosis-balanced resampling and cohort weights |
| DR too small | weak depression validation | report uncertainty; avoid DR-specific overclaiming |
| Structural missingness | unsupported dimensions may appear by proxy | explicit eligibility and module status |
| Site effects | artificial strata | leave-one-site-out validation |
| Missingness patterns form strata | assessment protocol masquerades as phenotype | stratum artefact audit |
| Treatment confounding | false treatment-response claims | target-trial emulation only |
| Neural models overfit | latent space hard to interpret | neural models only as secondary sensitivity |

---


# Phase S — Explicit V2-to-V3 transformation map

This phase is a management layer: it makes the transformation from V2 to V3 operational and prevents conceptual drift.

## S1 — Maintain a dual-track evidence structure

### What to do

Run every major V3 result against the V2 benchmark:

```text
V2 masked estimator result
V3 FIML benchmark result
V3 Bayesian mixed-likelihood result
```

For each dimension and stratum, document whether V3:

```text
confirms V2
refines V2
splits V2
merges V2
rejects V2
cannot test V2
```

### Why

V2 is not discarded. It is the historical reference arm. V3 has to prove that its added complexity changes scientific validity, not only software sophistication.

### Advancement from V2

V2 was a single coherent analysis. V3 becomes a benchmarked methodological upgrade with explicit confirmation/refinement logic.

### Expected output

```text
reports/v2_v3_concordance_matrix.csv
reports/v2_v3_dimension_translation.md
figures/v2_v3_loading_comparison.png
```

---

## S2 — Convert the V2 phenotype atlas into a V3 prior atlas

### What to do

Take V2's named constructs and axes and encode them as prior information:

```text
V2 construct name
V2 axis / standalone status
candidate V3 dimension
expected loading strength
plausible cross-loadings
allowed module status
coverage by cohort
```

Example translation:

| V2 construct/finding | V3 prior role |
|---|---|
| QIDS/MADRS/STAI/FAST internalizing | expected affective/anhedonic/severity loading; BP/DR anchored |
| CVLT/TMT/fluency/coding cognition | expected cognition/flexibility loading; core transdiagnostic |
| adiposity/lipids/glucose/BP | expected metabolic load |
| CRP/WBC/neutrophils/platelets | expected inflammatory load |
| ISF suicidality | expected suicidality loading with binary/ordinal/count likelihoods |
| Altman/YMRS mania | activation/impulsivity-proxy module |
| alcohol/cannabis SUD | behavioural-risk/substance module |
| CTQ/age of onset/family history | developmental-risk proxy |

### Why

This preserves V2's clinical and empirical knowledge without freezing its factor structure.

### Advancement from V2

V2 used construct aggregation and second-order factoring. V3 uses the same construct intelligence as soft priors that the data can override.

### Expected output

```text
priors/v3_prior_loading_matrix.csv
priors/v3_prior_strength_dictionary.yaml
reports/prior_coverage_by_cohort.md
```

---

## S3 — Reframe the no-imputation principle precisely

### What to do

State the V3 missing-data doctrine explicitly:

```text
No completed-data imputation before discovery.
No mean/KNN/MICE-filled matrix for clustering.
Yes to deterministic skip-logic decoding.
Yes to observed-data likelihood.
Yes to posterior uncertainty over latent dimensions.
Yes to explicit missingness models when missingness is informative.
```

### Why

V2 correctly rejected naive imputation. V3 must clarify that FIML and Bayesian observed likelihood are not equivalent to creating a filled dataset.

### Advancement from V2

V2 used masked pairwise covariance because complete-data ML was unsuitable. V3 uses patient-level likelihood over observed cells, which preserves the no-naive-imputation principle while avoiding pairwise-correlation limitations.

### Expected output

```text
methods/v3_missingness_doctrine.md
reports/observed_likelihood_vs_imputation_note.md
```

---

## S4 — Replace natural-subtype testing with decision-strata development

### What to do

Keep the V2 natural-subtype test as a negative-control benchmark, but add decision-strata modeling:

```text
Dimension scores + uncertainty
        -> latent profile / Bayesian mixture / risk-threshold models
        -> posterior stratum probabilities
        -> outcome validation
        -> decision-curve analysis
```

### Why

V2 found no robust natural density-separated subtypes. That does not preclude clinically useful strata. A risk stratum may be a validated region of a continuum rather than a natural disease subtype.

### Advancement from V2

V2 answered: "Are there discrete natural subtypes beyond DSM?"  
V3 answers: "Can dimension profiles define validated decision regions that improve prognosis or treatment-relevant choices?"

### Expected output

```text
reports/natural_clusters_vs_decision_strata.md
models/v3_patient_strata_probabilities.parquet
figures/decision_region_profiles.png
```

---

## S5 — Make precision psychiatry contingent on utility, not elegance

### What to do

For each final V3 dimension and stratum, require proof of at least one downstream value:

```text
improved calibration
improved discrimination
improved decision-curve net benefit
improved prognosis in a clinically relevant subgroup
support for treatment-effect heterogeneity under target-trial design
clear clinical interpretability
```

### Why

A latent dimension can be statistically elegant but clinically inert. Precision psychiatry requires decision value.

### Advancement from V2

V2 showed modest prognosis increment. V3 must define whether these increments become useful for decision-making, triage, monitoring, or treatment-effect hypotheses.

### Expected output

```text
reports/clinical_utility_scorecard.md
reports/strata_decision_value.md
reports/treatment_feasibility_by_stratum.md
```

---

# Phase T — V3 final decision tree

Use this decision tree to adjudicate the entire project.

```text
1. Is the variable measured and harmonized?
   no  -> exclude or mark unsupported
   yes -> continue

2. Is missingness structural/design/informative/sporadic?
   structural -> module or extension, not imputed core
   design/informative -> model missingness sensitivity
   sporadic -> observed likelihood / FIML / Bayesian model

3. Does the candidate construct have enough indicators?
   no  -> proxy/module/unsupported
   yes -> build soft prior loading

4. Does the latent model support a stable dimension?
   no  -> reject or merge
   yes -> continue

5. Is the dimension transdiagnostic across BP/SZ/DR?
   no  -> diagnosis-specific or BP/DR/SZ module
   yes -> core dimension

6. Does it survive general burden G?
   no  -> severity expression, not independent dimension
   yes -> specific dimension

7. Does it predict or validate externally?
   no  -> descriptive phenotype
   yes -> candidate decision dimension

8. Do combinations of dimensions define stable, interpretable strata?
   no  -> use continuous scores only
   yes -> define probabilistic strata

9. Do strata improve prognosis or treatment decision value?
   no  -> report as descriptive profiles
   yes -> precision-psychiatry decision framework
```

---

# Final V3 operating principle

V3 should not ask only:

```text
Do FACE patients lie on dimensions rather than DSM categories?
```

It should ask:

```text
Which transdiagnostic dimensions are empirically supported by patient-level observed data?
Which patients occupy similar validated regions of this dimension space?
Do those regions improve prognosis or treatment-relevant decisions beyond DSM and severity?
```

That is the transformation from a dimensional psychopathology analysis into a precision-psychiatry stratification and decision-modeling framework.
