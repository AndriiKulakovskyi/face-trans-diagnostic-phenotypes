# M1 — The transdiagnostic measurement model: logic and mathematical specification

> This is methods-of-record for **Milestone 1 (M1): the transdiagnostic dimensional map on the FACE
> V0 baseline.** This document fixes the *scientific logic* and the *mathematics* of the measurement
> model and its estimation; it is written to feed the manuscript's Methods section directly. It is the
> **single methods/plan of record** for the measurement layer. Scope is **V0 measurement only** —
> strata, prognosis, and treatment are separate milestones (all now complete), each specified in its own
> methods-of-record, not here (§10).
>
> *Math is written in plain-text/unicode (code blocks for display, `code spans` inline) so it renders in
> any markdown viewer; it transcribes directly to LaTeX for the paper.*

---

## 1. Scientific logic

### 1.1 What we are building, and in what order

Psychiatric heterogeneity has two levels that must not be conflated:

1. **Transdiagnostic dimensional analysis** — a *measurement* problem: place each patient on continuous,
   diagnosis-agnostic axes of variation (clinical, cognitive, behavioural, biological). This *builds the
   map*. It is aligned with NIMH **RDoC** (dysfunction as continua across units of analysis) and
   **HiTOP** (dimensional structure over arbitrary diagnostic boundaries).
2. **Patient stratification** — a *segmentation* problem: use a patient's position on the map to define
   decision-relevant subgroups. This *draws validated territories on the map*.

The architecture is strictly ordered:

```
diagnostic cohorts → transdiagnostic dimensions → validated strata → prognosis / treatment
   (entry metadata)      (M1 — this document)        (later)            (later)
```

M1 delivers only the first arrow: the **measurement model**, the **dimension atlas**, and **per-patient
dimension coordinates with uncertainty** on V0.

### 1.2 Hybrid discovery, not theory-imposed scoring

Manually tagging instruments to dimensions, summing scores, and clustering is a *theory-imposed scoring
system*, not discovery. We instead run a **hybrid** design: theory proposes a candidate structure; the
data estimate the actual factor structure, loadings, cross-loadings, hierarchy, and validity. The ten
candidate dimensions are a **soft prior ontology**, not fixed outputs. The data may **confirm, split,
merge, downgrade (proxy), reject, or declare not-testable** any candidate, and may place cross-loadings,
a general factor, and cohort-specific modules.

We keep three objects distinct throughout:

- **candidate construct** — a clinical idea → a soft prior over loadings;
- **empirical dimension** — a stable, data-supported axis of covariance;
- **patient stratum** — a recurring decision-relevant region (later milestone).

### 1.3 Three load-bearing invariants

- **No naive imputation.** Structure is estimated from each patient's *observed* cells via an
  observed-data likelihood; no completed patient×variable matrix is ever created (§3.4). This is not a
  convenience — naive imputation distorts covariance and manufactures artefactual strata when missingness
  is cohort-, site-, or severity-dependent.
- **Diagnosis is metadata.** BP/SZ/DR labels are never indicators. They enter only as covariates, as a
  measurement-invariance grouping variable, and as a post-hoc validation/interpretation layer.
- **Baseline defines, follow-up validates.** Dimensions are estimated on **V0** only. Later visits
  (V1–V4) are reserved for temporal coherence and prognosis — used *after* the map is fixed.
- **One global structure; cohort is missingness, not structure.** There is a single factor structure on
  one harmonized matrix. An instrument absent for a cohort is simply a **missing cell** (observed
  likelihood) — there are **no cohort-specific "modules" or "extensions."** Coverage is emergent and
  flagged per patient (a factor's scores are low-information for patients lacking its indicators), never a
  structural sub-model.

> **A dimension requires indicators.** The model can discover a statistical dimension; it cannot discover
> a construct that is not measured. A candidate with no valid indicators is reported `not_testable`, never
> back-filled with an invented proxy.

---

## 2. The candidate ontology and the eligibility map

The ten candidate dimensions: **(1) Impulsivity · (2) Cognitive Flexibility · (3) Negative Symptoms ·
(4) Anhedonia · (5) Metabolism / Immunometabolism · (6) Sleep / Sleep-Circadian Dysregulation ·
(7) Overall Clinical Severity · (8) Sensory Abnormalities · (9) Neurodevelopment · (10) Suicidality**.

Mapping these against the FACE *common-variable* instrument coverage (per the dimension-readiness
workbook) fixes the eligible model set. **This map is itself a primary result of the analysis.**

| Candidate | Verdict | Role in the model | Indicators (FACE common) | Cohorts |
|---|---|---|---|---|
| 7 Overall Severity | core anchor | **G** (functional burden — one job) | CGI-S, EGF⁻, EQ5D-VAS⁻, FAST, work-stoppage, employment — functioning + global severity **only** | 3-cohort |
| 2 Cognitive Flexibility | core | **cognition** | TMT-B (resid. on TMT-A/age/edu/site), WAIS coding/digit-span, fluency | 3-cohort |
| 5 Metabolism/Immuno | core — **split** | **metabolic** + **inflammatory** | BMI, waist, BP, glucose, HbA1c, lipids / logCRP, WBC, neutrophils, platelets | 3-cohort |
| 6 Sleep / Circadian | core (sleep) | **sleep** | PSQI total + 7 sub-scores (ESS/CSM circadian = BP/DR extension) | 3-cohort |
| 10 Suicidality | core | **suicidality** (mixed likelihood) | ISF ideation/attempt; C-SSRS lethality (BP/DR extension) | 3-cohort ISF |
| 9 Neurodevelopment | core **proxy — relabelled** | **developmental-risk** | perinatal, CTQ, age-of-onset, family history, education | 3-cohort |
| 4 Anhedonia | partial | **anhedonia** (BP/DR extension; thin) | QIDS anhedonia item (± MADRS/QIDS pleasure items) | BP/DR |
| 1 Impulsivity | weak proxy | **dropped** from core (hypothesis layer only) | YMRS/Altman/WURS proxies | — |
| 3 Negative Symptoms | not directly measurable | **dropped** (functional proxy at most) | no PANSS-neg / SANS | — |
| 8 Sensory Abnormalities | gap | **not_testable** | none | — |

**Eligible model set (V0).** 3-cohort core factors `G` (severity), `cognition`, `metabolic`,
`inflammatory`, `sleep`, `suicidality`, `developmental-risk`; **BP/DR extension** `anhedonia` (thin — may
not survive as a standalone factor and could merge into `G` or be rejected). Three candidates are dropped
for lack of indicators — *stating this is a result, not a failure.*

Notes that the data will adjudicate, not us: (i) `metabolic` vs `inflammatory` is a hypothesised **split**
of candidate 5; (ii) `developmental-risk` is a **proxy** relabelling of candidate 9; (iii) the composite
depression/anxiety instruments (`madrs`, `qidsr120`, `staya`) are **cross-loading windows**, not a
dimension: each is seeded `plausible_cross` on the axes it clinically touches (`G`, anhedonia, sleep,
suicidality, cognition) and the data place it — there is **no separate affective factor and no 11th
"depression" dimension** (one may only *emerge* via model comparison, §6). `G` does **one job** —
transdiagnostic functional burden, anchored by functioning + global severity **only** — so symptom
severity informs the axes it belongs to rather than contaminating the general factor.

### 2.1 Soft-prior loading roles

Every (indicator, dimension) cell receives a prior **role** in the soft-prior matrix `T`:

| Role | Meaning | Prior on the loading `λ_jk` |
|---|---|---|
| `expected` | indicator should load on this dimension (primary) | `Normal₊(0.70, 0.25)` (truncated > 0) |
| `plausible_cross` | indicator *may* load | `Normal(0, 0.25)` |
| `unlikely` | indicator should be near zero | `Normal(0, 0.05)` |
| `forbidden` | invalid clinical relation | `0` (fixed) |
| `covariate_only` | adjust for it, not an indicator | — (not in Λ) |
| `outcome_only` | reserved for validation | — (excluded from baseline) |

The default is **shrinkage, not hard exclusion**: theory says where variables *should* load; the
likelihood is free to disagree.

### 2.2 Coverage — seed generously, let the soft priors adjudicate

A hybrid soft-prior ESEM is **safe to over-seed**: an indicator with no real signal shrinks toward zero
under its `unlikely` / `plausible_cross` prior, so including a weak indicator costs little, while
*dropping* a usable one is a permanent loss of signal. We therefore seed **every construct-relevant usable
common variable** as an indicator (core or extension) and reserve `covariate_only` for genuine
confounders / identifiers.

Audit of the harmonized dictionary (`data/face-common-vars.xlsx`): **201 usable variables** (100 READY
3-cohort + 91 PARTIAL 2-cohort). The initial pools mapped 119; the audit recovered **~80 unmapped usable
variables**, re-allocated as:

| Recovered block | → dimension | n | notes |
|---|---|---|---|
| C-SSRS ideation battery + attempt-lethality (`cssrs01–12`, `ltsg*`, `ltsv*`) | **suicidality** (BP/DR ext) | ~24 | mixed likelihoods (binary + ordinal); alongside the 3-cohort ISF core |
| family psychiatric history (`mere_structure`, `pere_structure`, 3-cohort) + perinatal (`agemere`, `agepere`, `honeonat`, `brthcirc`) | **developmental-risk** | ~6 | a 3-cohort family-liability indicator that had been missed |
| employment + work-stoppage (`jobclas`, `stprof`, `*_arret_travail`) | **G** (functioning) | ~4 | functional-impairment indicators |
| 2-cohort biology — thyroid, liver, renal/electrolyte, red-cell, vit-D, resting HR | **metabolic / inflammatory** (BP/DR ext, soft) | ~30 | seeded `plausible_cross`; the data keep the informative ones |
| inflammatory / metabolic comorbidity flags (psoriasis, eczema, migraine / HTA, CV, endocrine) | **inflammatory / metabolic** (soft) | ~10 | binary comorbidity signal, adjudicated |
| hormonal Tx, menopause, QT/RR, height, smoking-history ages, PRISE-M | `covariate_only` | ~8 | genuine confounders, not indicators |

As encoded (`configs/dimensions.yaml` → `prior_loading_matrix_v3.csv`), **143 of 201 usable variables
enter as modeled indicators** across **10 factors** (G + 9 specifics: cognition, metabolic, inflammatory,
sleep, suicidality, developmental-risk, anhedonia, mania_activation, substance) plus **3 cross-loading
windows** (`madrs`/`qidsr120`/`staya`). The remaining ~58 are covariates / identifiers — including
ambiguous-direction labs (electrolytes, red-cell indices), CGI improvement/efficacy ratings, ECG
intervals, reproductive/hormonal and smoking-history confounders, and nominal suicide-method types. **Where
an instrument is unobserved for a cohort, its cells are simply missing** (observed likelihood; no cohort
sub-model), so a factor's transdiagnosticity is *adjudicated and flagged*, never assumed.

### 2.3 The atlas, and the theory-vs-data comparison (the deliverable)

The map is reported as two aligned **atlases** and their comparison — this *is* the scientific story:

- **Prior atlas** — the soft-prior loading matrix, drawn as an indicator×factor heatmap (theory: where each
  instrument is *expected* to load). Generated now from `configs/` alone — see [`PRIOR_ATLAS.md`](PRIOR_ATLAS.md).
- **Empirical atlas** — the posterior loading matrix `Λ` from the fitted global model (data: where each
  instrument *actually* loads, with uncertainty) + the factor-correlation matrix `Φ`.
- **Prior → posterior comparison** — the two heatmaps side by side, with a per-candidate verdict
  (`confirmed | split | merged | proxy | rejected | not_testable`; §6). It shows the 10 theoretical
  candidates being **confirmed, reshaped, or dropped by the FACE data** — the hybrid model adjusting theory
  with evidence.

It is *expected* that some candidates do not survive (sensory, negative symptoms, impulsivity have no
instruments; anhedonia is thin). Demonstrating **which** survive, **how** they reshape (e.g. metabolic
splitting from inflammatory), and that they were earned from cohort data — with uncertainty and validation
— is the scientific and clinical value of the measurement layer.

---

## 3. Mathematical specification

Index patients `i = 1, …, N` (V0; `N = 9,013 = BP 6,252 + SZ 2,209 + DR 552`), indicators `j = 1, …, J`,
specific dimensions `k = 1, …, K`, and one general factor `G`.

### 3.1 Generative model

Latent factors per patient:

```
G_i  ~  Normal(0, 1)
D_i  =  (D_i1, …, D_iK)
```

Two identifications of the specific factors are estimated:

- **Bifactor (primary):** `D_ik ~ Normal(0, 1)`, `Cor(D_k, D_l) = 0`, and `G ⊥ D_k`. This is the cleanest
  identification for a first stable fit.
- **Correlated specifics / correlated-G (sensitivity):** `D_i ~ Normal(0, Φ)` with `Φ ~ LKJ(η)`, and a
  variant in which `G` is *allowed* to correlate with the biological specifics. This variant is essential:
  it tests whether "`G ⊥ biology`" is an empirical finding or merely an artefact of the bifactor constraint.

Linear predictor for indicator `j` in patient `i`:

```
η_ij  =  α_j  +  λ_jG · G_i  +  Σ_k ( λ_jk · D_ik )  +  β_j^T · c_i
```

where `c_i` are **covariates** (age, sex, education, site, cohort, assessment year, and — later —
medication), entering at the item level. Covariates adjust indicator means; they are **never** dimension
indicators. `α_j` is the item intercept, `λ_jG` the general loading, `λ_jk` the specific loading.

### 3.2 Mixed-likelihood observation model

Each indicator is given the likelihood appropriate to its type — we do **not** coerce everything to a
shared continuous scale:

```
X_ij  ~  F_j( η_ij , θ_j )
```

| Indicator type | Likelihood | Form (linear predictor `η_ij` → likelihood) |
|---|---|---|
| continuous clinical / z-scored neuropsych | Student-t (Gaussian as ν→∞) | `X_ij ~ t(ν_j, η_ij, σ_j)` |
| skewed lab (CRP, triglycerides) | log-Student-t | `log X_ij ~ t(ν_j, η_ij, σ_j)` |
| ordinal item (`C_j` categories) | ordered logistic | `P(X_ij ≤ c) = logit⁻¹(τ_jc − η_ij)`, ordered `τ_j1 < … < τ_j,Cj−1` |
| binary (ISF ideation/attempt) | Bernoulli-logit | `X_ij ~ Bernoulli(logit⁻¹(η_ij))` |
| count (attempts, hospitalizations) | negative binomial | `X_ij ~ NegBin(μ_ij = exp(η_ij), φ_j)` |

(Zero-inflated NB is available if excess structural zeros remain after decoding, §3.4.) For non-Gaussian
likelihoods the intercept is absorbed into the cutpoints/offset and the factor scale is fixed for
identification.

### 3.3 Priors

```
λ_jk | role =
    Normal₊(0.70, 0.25)   if expected (primary)      [truncated > 0]
    Normal(0,    0.25)    if plausible cross
    Normal(0,    0.05)    if unlikely
    0                     if forbidden
```

Remaining priors: intercepts `α_j ~ Normal(0, 1.5)`; residual scale `σ_j ~ Half-t`; ordinal cutpoints
`τ_j,·` ordered-normal; count dispersion `φ_j ~ Exponential(1)`; factor correlation `Φ ~ LKJ(η)`; latent
scores `G_i, D_ik ~ Normal(0, 1)`.

The soft-prior block above is the formal mechanism of "theory suggests, data decides": most cross-loadings
are shrunk toward zero (sparsity), but the likelihood can pull any of them away if the data demand it.

### 3.4 Missing data — observed-likelihood, no imputation

Let `O_i ⊆ {1, …, J}` be the indicators **observed** for patient `i`. The likelihood sums over observed
cells only:

```
log p( X_obs | Θ )  =  Σ_i  Σ_{ j ∈ O_i }  log F_j( X_ij | η_ij , θ_j )
```

Missing cells contribute **no term**; a patient's latent coordinates are informed only by their observed
indicators. There is no completed matrix at any point.

- **Structural zeros.** Gated items are decoded to an observed `0` only where the gate is *explicitly*
  negative (e.g. `attempt_count = 0` when `ever_attempt = 0`); never where the gate is unknown, never
  overwriting an observed value. Decoded zeros are **observed values**, logged with rule + provenance.
- **Cohort-specific measurement.** Where an instrument is absent by design in a cohort (e.g. anhedonia in
  SZ), those patients simply contribute **no cells** to that factor; the factor is identified from the
  cohorts that measure it, and the unmeasured patients' scores are flagged low-information — **not
  imputed**.
- **Informative missingness (optional MNAR arm).** Where missingness is demonstrably informative, a
  selection model may be added as a *sensitivity* analysis (not the primary fit), with `R_ij` the observed
  indicator for cell `(i, j)`:

  ```
  R_ij  ~  Bernoulli( π_ij )
  logit(π_ij)  =  a_j  +  γ_j^T · c_i  +  δ_jG · G_i  +  Σ_k ( δ_jk · D_ik )
  ```

  used only where identification is acceptable.

### 3.5 The Gaussian-block marginal — and its identity with FIML

For the continuous indicators the latent factors can be integrated out analytically. Stacking the
continuous loadings into `Λ`, the full factor covariance over `G` and the `K` specifics into `Φ_full`, and
the residual variances into `Ψ = diag(σ_j²)`, the marginal over patient `i`'s observed continuous
sub-vector `X_{i, Oi^c}` is

```
X_{i, Oi^c}  ~  Normal( μ_{i, Oi^c} ,  Σ_{Oi^c, Oi^c} )

Σ  =  Λ · Φ_full · Λ^T  +  Ψ ,        μ_i  =  α  +  B · c_i
```

Summing `log Normal(·)` over each patient's observed continuous sub-vector is **exactly the FIML
objective**. Hence the Bayesian marginalized model and the FIML confirmation (§5) optimise the *same*
observed-data likelihood and differ only by the presence of priors — which is precisely why FIML is the
right, non-redundant confirmatory estimator, and why this marginal is also our computational fallback
(§4.4): it removes the per-patient latent funnel for the Gaussian block while keeping the full sample.

### 3.6 Sample, cohort balance, and the fit–score separation

**Sample.** The measurement model is fit on the **full V0 sample** (N = 9,013; BP 6,252 · SZ 2,209 ·
DR 552) — **no completeness selection.** A missingness-aware observed-likelihood model must see the full
missingness structure; selecting the most-complete patients would estimate the map on the least-missing,
least-representative sub-population and forfeit the very property the observed likelihood provides (§3.4).
This is a stated acceptance criterion (§8), not a convenience.

**Cohort imbalance** (BP is ~11× DR) is handled *without discarding data*. For a measurement model,
imbalance distorts the loadings only if the structure is **non-invariant** across cohorts; we therefore
(i) **test measurement invariance** across BP/SZ/DR directly (does each loading hold per cohort? §8) —
detecting the problem rather than hiding it — and (ii) report a **1/n_cohort-weighted** fit as a
sensitivity arm (equalizing each cohort's influence using all patients). Balancing by *subsampling* is
rejected: a "500/cohort" scheme would discard ~92% of BP to match DR — a large real loss for a balance
that weighting achieves for free.

**Fit–score separation.** Fitting and scoring are distinct. The model is fit **once** on the full sample
to estimate loadings `Λ` and correlations `Φ`; **scores** for any patient are then a projection of that
patient's *observed* cells onto the fitted model (§7). So a downstream cohort — e.g. patients with
adequate follow-up (≥ V3) reserved for prognosis (M4) — is **scored** from the full-sample model; it does
**not** drive the measurement fit. This delivers reusability for predictive modeling *without* importing
attrition/retention selection into the map: selecting the measurement sample on follow-up completeness
would bias the structure toward treatment-retained (systematically healthier, more-adherent) patients —
exactly the selection we avoid at the measurement layer.

**Compute (hard ceiling: M4 Pro, 24 GB).** Full-N is made tractable by **Gaussian-block marginalization**
(§3.5; the continuous core carries only a few hundred parameters, no per-patient latent funnel) plus
**staged warm-starts** (§4.2). The frontier is the mixed-likelihood stages (S3+), where non-Gaussian
indicators carry per-patient latents at full N. *If* a stage exceeds the ceiling there, the fallback is a
**random, cohort-balanced** subsample for that stage (random → realistic missingness; with a
resample-stability check) — **never** completeness- or attrition-selected. The reported map targets the
full sample, or the largest N that certifies, documented.

---

## 4. Estimation strategy — staged continuation to the global fit

### 4.1 The target and the difficulty

The scientific object is the **single global posterior** `p(Θ | X_obs)` with
`Θ = { Λ, α, θ, Φ_full, latent }` for **all** eligible dimensions, **all** indicators, **all**
likelihoods, on the **full V0 sample**. Fitting it cold is hard for three reasons, each a concrete failure
mode:

1. **Funnel geometry at scale.** ~9,000 patients × ~8 latent dimensions ≈ 70k per-patient latent
   parameters with hierarchical variance → Neal's funnel → NUTS divergences.
2. **Weak identification / rotational indeterminacy.** `G` + correlated specifics + cross-loadings is
   identified only up to rotation/sign/label; on sparse, cohort-structured missingness the posterior can
   be multimodal → chains land in different modes → R-hat inflates and loadings look unstable.
3. **Mixed-likelihood stiffness.** Gaussian + ordinal + Bernoulli + NB across cohort-specific missingness
   at full `N` is a large, stiff joint model.

### 4.2 Continuation (homotopy) staging

We do **not** treat the build order as a scientific sequence. Staging is a **continuation method**: solve
an easy, well-conditioned sub-model, then deform toward the full target one component at a time, carrying
each converged posterior forward as the warm-start for the next. Formally we fit a nested sequence
`M1 ⊂ M2 ⊂ … ⊂ M5 = global`, where each `Ms` adds exactly one source of difficulty, and initialise

```
θ_init^(s+1)  =  E[ θ | Ms ]
```

What this buys: **(i) fault isolation** — a convergence failure at stage `s` implicates only what stage
`s` added (everything prior is certified); **(ii) warm-starts** — the hardest fit starts in a good basin;
**(iii) risk quarantine** — each failure mode in §4.1 is introduced at a known stage with its specific
mitigation; **(iv) stability-under-elaboration** as robustness evidence; **(v) fail-fast** compute and a
natural QC/discussion gate after each step.

Latent scores use a **non-centered parameterization** (`D_ik = z_ik`, `z_ik ~ Normal(0, 1)`, scale carried
by the loadings) to mitigate the funnel; rotational indeterminacy is controlled by sign-anchored primary
loadings (oriented so *higher = more burden*) and the soft-prior sparsity.

| Stage | Adds | Primarily tests / yields |
|---|---|---|
| **S1** | continuous core (`G`, cognition, metabolic, inflammatory, sleep), explicit-latent, **full N** | is the funnel controlled at scale? — *the make-or-break gate* |
| **S2** | ESEM soft-prior cross-loadings | does identification survive freeing cross-loadings? |
| **S3** | mixed-likelihood suicidality (Bernoulli/NB) + developmental-risk | do non-Gaussian indicators compose with the shared `Φ_full`? |
| **S4** | anhedonia (BP/DR, thin) | does a cohort-specific, thin factor identify at all? (adjudication input) |
| **S5** | **all dimensions jointly** + correlated-G variant | **the reported fit:** `Λ`, `Φ_full`, adjudication, the `G ⊥ biology` test |

### 4.3 The interpretation rule (the guardrail)

> **Only `M5` is interpreted or reported.** Stages S1–S4 are convergence/identification checkpoints with
> *provisional* reads. The loadings, factor correlations, adjudication, and patient scores are read
> **only** from the global fit. Publishing an intermediate stage is exactly the error to avoid (a prior
> iteration concluded "no general factor" from a sub-model that omitted the severity anchors).

Each stage emits a report (`reports/NN_stage*.md`) + figures and is followed by a discussion gate before
advancing — the standing QC cadence.

### 4.4 Acceptance gate per stage, and the fallback ladder

A stage advances only if: `R-hat ≤ 1.01`; `ESS ≥ 400` (bulk and tail); `0` divergences; no Heywood
(`max_j |λ_j|` within a cap); posterior-predictive checks not grossly violated; and an identification
check (loadings stable across chains, no sign/label switching).

If a stage will not certify, climb the fallback ladder (we will know early which rung we are on):

1. **Marginalize the Gaussian block** (§3.5) — analytic integration removes the latent funnel for the
   continuous indicators while keeping the full sample; explicit latents only for the non-Gaussian block.
2. **QR / orthogonal loading parameterization** to break rotational symmetry.
3. **Reduce free cross-loadings** (tighten `plausible_cross` toward `unlikely`).
4. **Block-wise fits with a shared `Φ_full`**.

None of these reintroduce subsampling or imputation — the full-sample, no-imputation invariants hold on
every rung.

### 4.5 Compute

Develop and smoke-test the model in **PyMC** (readable spec); run the full-sample global fits with the
**NumPyro / JAX-CUDA** backend (`pm.sample(nuts_sampler="numpyro")`) on the **RTX 4090** (24 GB), which
comfortably accommodates ~9k patients × ~8 latent dimensions with long, well-tuned chains. Running the full
sample on GPU is what lets us *refuse* the completeness-selected subsample that biased the previous
iteration — the single most important rigor decision in M1.

---

## 5. Estimator and prior robustness (in-engine confirmation)

The empirical structure must not depend on one estimator or on the soft priors. We originally specified a
separate **FIML SEM** here, but §3.5 shows the marginalized Bayesian model and FIML optimize the **same**
observed-data objective — so a separate SEM adds little independent evidence. In practice a classical FIML
on the full high-missingness backbone is also intractable/unreliable (semopy is slow and returns
inconsistent fit indices), while complete-case FIML would reintroduce the completeness selection the
project forbids (§3.6). We therefore confirm the map **in the existing engine**, which answers the same
questions more faithfully (script `scripts/05_confirm.py`, module `src/face/confirm.py`):

1. **Prior-free refit.** Re-fit the continuous backbone with *flat* loading priors (identification
   constraints only — sign-oriented home cells, signed cells centered at 0). A flat-prior MAP = MLE = FIML
   (§3.5); loadings/Φ that match the soft-prior fit show the structure is **earned from the data, not
   manufactured by the priors**. *(Result, full N: Tucker congruence φ = 1.00 for every factor; max |ΔΦ|
   = 0.00; flat-fit R-hat 1.00.)*
2. **Posterior-predictive checks.** Model-implied vs observed pairwise correlations → a **Bayesian SRMR**
   and residual-correlation matrix (absolute fit), without the χ² asymptotics that fail under heavy
   missingness + non-normal indicators. *(Result: SRMR ≈ 0.07; misfit confined to repeated-measure item
   clusters — HR/BP positions, chol/LDL, AST/ALT, WAIS subtests.)*
3. **WAIC model comparison.** Bifactor vs unidimensional vs correlated-factors (incremental fit). *(Result:
   the bifactor is decisively preferred — ΔWAIC ≈ 2.7k over correlated-factors, ≈ 53k over unidimensional.)*

This composes with the explicit-vs-marginalized parameterization agreement (§4) and the prior-sensitivity
arm (§8). **Classical fit indices (CFI/TLI, RMSEA/SRMR)** are not reported — they are convention, not new
evidence, and statistically weak here; if a venue requires them, lavaan (R) provides proper missing-data
FIML on request. Suicidality's ordinal/binary/count indicators remain Bayesian-only throughout.

---

## 6. Dimension adjudication

Empirical factors are mapped back to candidate constructs and assigned one verdict:
`confirmed | split | merged | module | proxy | rejected | not_testable`.

A dimension is **confirmed** only if it passes most of: adequate indicator coverage (`≥ 3` meaningful
indicators where possible); posterior primary loadings `|λ| ≥ 0.30` with credible intervals away from `0`;
clinically coherent loadings and acceptable score reliability; **not reducible to `G`**; **not reducible to
cohort / site / missingness**; stable under bootstrap / diagnosis-balanced subsampling (congruence
`≥ 0.85`); acceptable or documented-partial measurement invariance; (in later milestones) a relation to
external validators. Thresholds are guidelines, not automatic rules. Output: the **empirical dimension
atlas**.

---

## 7. Patient scoring

Each patient receives, per surviving dimension: posterior **mean**, **SD**, and **HDI**; the **number of
observed indicators**; a **reliability/quality flag**; and a consistent **orientation** (higher = more
burden, unless documented). Uncertainty is preserved (optionally as posterior draws) so downstream
strata/prognosis can propagate it — a patient with one observed indicator for a dimension is *not* treated
as equally characterised as one with six.

---

## 8. Validation and acceptance for M1

- **Sampler diagnostics:** R-hat, ESS, divergences, BFMI; prior- and posterior-predictive checks;
  loading sign/identifiability; prior sensitivity.
- **Robustness:** diagnosis-balanced subsampling and site bootstrap (as *checks on the full-sample fit*,
  not as the fit itself); leave-one-cohort-out loading congruence.
- **Measurement invariance** where testable — the 3-cohort core (cognition, metabolic, inflammatory,
  sleep, `G`): multi-group loadings/intercepts/thresholds and Bayesian DIF; partial invariance documented.
  Cohort-modular dimensions (anhedonia; heterogeneously-measured suicidality) are **declared modular** and
  not claimed as fully invariant.
- **The selection-bias resolution** (full-sample fit, no completeness subsample) is itself a stated
  acceptance criterion for M1.

---

## 9. Estimands, and what we will / will not claim

**Estimands.** The loading matrix `Λ`; the factor correlation matrix `Φ_full` (including the `G`–biology
relationship under the correlated-G variant); per-patient posterior dimension coordinates with uncertainty;
the adjudication verdicts.

**We will claim** (if supported): an empirically-supported subset of transdiagnostic dimensions, their
measurement structure, and per-patient coordinates with uncertainty — under **internal** validity.

**We will not claim** (in M1): external validity (no external cohort yet); natural biotypes/subtypes;
any causal, treatment, or follow-up prognostic effect. "Converged" denotes sampler convergence, **not**
scientific validation.

---

## 10. Position in the roadmap (named, not specified here)

M1 (this document) → **M2 strata** (probabilistic decision regions on the M1 coordinates) →
**M3 temporal coherence** (do scores persist V1–V4?) → **M4 prognosis** (incremental value beyond
diagnosis + severity) → **M5 treatment** (target-trial emulation, only if the data support it). Each is a
separate milestone with its own gate; none is started before M1 is locked.

> **All five are now complete** (pending PI sign-off) — see [STATE.md](STATE.md) and the per-milestone
> findings. The calibrated outcome: scientific validity demonstrated (a stable continuum map that forecasts
> functioning beyond severity), strong-sense clinical utility (individual prediction, treatment guidance)
> not — M5's target-trial emulation found the data support a moderation *test* but no reliable moderation
> on treatment-as-usual.

---

## 11. Repository and reproducibility (lean)

```
src/face/{data, missingness, priors, models/{bayesian, fiml}, dimensions, scoring, reporting}
scripts/      01_build_data → 02_missingness → 03_priors → 04_fit_global(S1..S5) → 05_fiml → 06_adjudicate → 07_score
data/         raw CSV (read-only, confidential) ; processed/*.parquet (model-ready persistence)
configs/      data, harmonization, missingness, dimensions, priors, model, fiml  (YAML)
notebooks/    run_pipeline.ipynb  (executes stages + displays results/figures)
reports/      NN_*.md per stage (+ figures)  ·  docs/  (this spec, atlas, plan)
```

Raw data stay **CSV**; the persistence layer adds **Parquet** for the model-ready tables only. No DVC /
Hydra / MLflow in M1 — deliberately lean. Every number reproducible from `scripts/` → `reports/`.
