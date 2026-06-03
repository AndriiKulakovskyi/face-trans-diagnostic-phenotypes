---
title: "Symptoms are orthogonal to biology: an integrated, imputation-free dimensional model dissolves the general psychopathology factor across bipolar disorder, schizophrenia and major depression"
author:
  - "FACE Trans-diagnostic Study Group^[Author list, affiliations and corresponding author to be confirmed. Data: Fondation FondaMental, FACE cohorts (FACE-BD, FACE-SZ, FACE-DR).]"
date: "Working draft — v2 analysis (re-derived from zero on the re-curated dictionary)"
abstract: |
  **Background.** Dimensional models of psychopathology (HiTOP, RDoC) increasingly challenge categorical diagnosis, and many recover a single general "*p*-factor." Yet most of this evidence rests on **symptom self-reports alone**, usually with imputed missing data. Whether a general factor survives when biology and cognition are placed in the *same* latent space, estimated without imputation, is unknown.

  **Methods.** We harmonized three deeply-phenotyped FACE cohorts — bipolar disorder (n = 6,252), schizophrenia (n = 2,209) and major depression (n = 552; total **N = 9,013**) — into a 220-variable common dictionary spanning symptoms, biology (labs, vitals/ECG) and cognition (WAIS/TMT, verbal memory, verbal fluency). Under a strict **no-imputation** rule, all structure was estimated from a masked pairwise-complete correlation with regression (Thomson) factor scores computed on each patient's observed support only. We fitted a **hierarchical/bifactor measurement model** (194 baseline items → 94 first-order constructs → second-order trans-diagnostic dimensions), tested a general factor by Schmid–Leiman explained common variance (ECV), and locked dimensionality by split-half Tucker congruence. A parallel masked-similarity stratification arm tested for discrete subtypes. Four pre-registered validation studies probed cohort confounding (A), symptom–biology orthogonality and the *p*-factor (B), longitudinal coherence (C), and incremental prediction over DSM (D).

  **Results.** Trans-diagnostic variation was **dimensional, not categorical**: four reproducible axes — **internalizing, cognition, illness-course, cardiometabolic–inflammatory** — with **no dominant general factor** (ECV = 0.34) and **no discrete subtypes** beyond the DSM cohorts themselves. The axes were not a cohort artifact (cohort-residualized congruence ≥ 0.96) and were granularity-invariant (top-three canonical r = 0.99/0.90/0.79). The headline result: **symptoms and biology are nearly orthogonal** (between-block mean |r| = 0.03), and the general factor is a **symptom-only artifact** — its first-factor share falls monotonically from **0.33 (symptoms only)** to **0.09 (integrated)** as *structured* biology and cognition are admitted. The structure was longitudinally coherent (cardiometabolic the most trait-stable axis). Against DSM, the dimensions added a **modest but real** prognostic increment — robustly for functioning (GAF ΔR² = +0.046; FAST +0.036) and, once a regression-to-the-mean confound was removed, modestly for relapse (de-confounded ΔAUC = +0.036 by logistic regression; early-course prognosis reached AUC ≈ 0.70).

  **Conclusions.** An integrated, imputation-free account of three major psychoses is genuinely **multidimensional with separable symptom, cognitive and biological substrates**. The general psychopathology factor appears to be an artifact of symptom-only measurement rather than a property of disorder. The dimensions are at least DSM-equivalent for prognosis and add a small, biologically-grounded increment. We state plainly where the "trans-diagnostic" label is qualified by measurement design.
---

# Introduction

For seventy years psychiatric nosology has been **categorical**: the DSM and ICD sort patients into discrete disorders defined by symptom checklists [1]. The categorical model has organized clinical communication and trials, but its scientific limits are now widely acknowledged — extensive comorbidity, within-category heterogeneity, arbitrary thresholds, and weak mapping to biology [2,3]. Two influential programs propose a **dimensional** alternative: the Research Domain Criteria (RDoC) recast psychopathology as continua grounded in neurobiological systems [2], and the Hierarchical Taxonomy of Psychopathology (HiTOP) organizes empirically-derived symptom dimensions into a hierarchy [3]. At the apex of that hierarchy many investigators place a single **general factor of psychopathology**, the "*p*-factor," interpreted as a trait-like liability common to all disorders [4–6].

The *p*-factor is, however, contested. Bifactor models that produce it are statistically permissive — they tend to fit well even on data with no genuine general dimension, and their general factor is often unreliable or unstable [7,8]. More fundamentally, almost all of the evidence — for HiTOP dimensions and for the *p*-factor alike — is built from **symptom self-reports and clinician ratings**. Biology and cognition, when measured, are typically related to the symptom dimensions *after the fact* rather than estimated **jointly, in the same latent space**. This leaves a basic question unanswered: **is the general factor a property of psychopathology, or a property of symptom questionnaires?** If biology and cognition were placed inside the same factor model, would a single dimension still span them — or would the structure resolve into separable substrates?

Answering this requires three things rarely combined: (i) **deep, multimodal phenotyping** — symptoms, blood biology, vitals/ECG and neurocognition — in the *same* patients; (ii) genuine **trans-diagnostic breadth**, so that structure is not the signature of one disorder; and (iii) an estimator that does not distort the multimodal covariance. The third point is usually under-appreciated. Multimodal clinical data are **pervasively and non-randomly missing**; the standard remedy — imputation — silently re-imports exactly the cohort-by-missingness confounds one is trying to avoid (we make this precise in Methods). Deep models (autoencoders, VAEs) do not escape this: they require complete input vectors, of which there are none.

We address all three using the FACE network — three deeply-phenotyped French national cohorts of **bipolar disorder, schizophrenia and major depression** (N = 9,013) sharing a common assessment battery. We harmonize them into one dictionary and, under a strict **no-imputation** rule, estimate structure from a **masked** (pairwise-complete) covariance with factor scores computed only on each patient's observed cells. On this foundation we fit a **hierarchical/bifactor measurement model** that respects the data's ragged granularity and lets us *test*, rather than assume, a general factor.

We ask three questions:

1. **Is trans-diagnostic structure dimensional or categorical?** (continuous axes vs discrete patient subtypes);
2. **Does a general psychopathology factor survive integration** of biology and cognition with symptoms?
3. **Do the dimensions add prognostic value over DSM diagnosis** out of sample?

Our central contribution is the answer to (2): **symptoms are nearly orthogonal to biology, and the general factor is an artifact of symptom-only measurement** — an integrated model of three major psychoses is genuinely multidimensional, with no dominant general factor. Around this we report a dimensional (not categorical) structure with four reproducible axes, a careful refutation of the obvious cohort-confound, a measurement-design map of what is *truly* trans-diagnostic versus disorder-anchored, and an honest accounting of modest-but-real incremental prognosis over DSM. Throughout we have tried to anticipate the sceptical reviewer, and we devote a section of the Discussion to doing so explicitly.

# Methods

We give the full formalism, the pipeline, and the key derivations. All analyses re-derive the structure **from zero** on a re-curated dictionary; no prior result is assumed. Software and reproducibility are described in §2.10.

## Cohorts and design

The FACE network comprises standardized "expert-centre" assessments of three diagnostic groups: **FACE-BD** (bipolar disorder), **FACE-SZ** (schizophrenia) and **FACE-DR** (treatment-resistant/major depression). Patients are assessed at baseline (**V0**) and annually to four years (**V4**). The analytic anchor is **V0**: dimensions are *defined* at baseline and later visits are used only to test their temporal coherence (§2.9, Study C), never to define structure. Cohort sizes and attrition are given in **Table 1**. Data are confidential (Fondation FondaMental) and were never exported from the secure environment.

| Visit | BP | SZ | DR | Total |
|---|---:|---:|---:|---:|
| **V0** (baseline) | 6,252 | 2,209 | 552 | **9,013** |
| V1 (1 yr) | — | — | — | 4,270 |
| V2 (2 yr) | — | — | — | 2,958 |
| V3 (3 yr) | — | — | — | 1,955 |
| V4 (4 yr) | — | — | — | 779 |

Table: **Cohort sizes and longitudinal attrition.** Cohort imbalance is marked (BP 69% / SZ 24% / DR 6%). Attrition is steep and likely non-random; FACE-DR collapses by V3 (n ≈ 3), so all longitudinal and predictive analyses are effectively **BP + SZ**, and V4 is reported but never led upon.

## The no-imputation principle, and why it matters

No cell is ever imputed, anywhere in the pipeline. This is not fastidiousness: under non-random, cohort-structured missingness, mean- or model-imputation **re-weights every correlation by co-observation** and so re-imports a cohort-by-missingness confound. If two variables $A$ and $B$ are observed on $n_A$ and $n_B$ patients with $n_{AB}$ in common, filling the unobserved cells with column means shrinks the recovered correlation toward

$$\operatorname{corr}_{\text{fill}}(A,B) \;\approx\; O_{AB}\,\operatorname{corr}_{\text{masked}}(A,B), \qquad O_{AB} \;=\; \frac{n_{AB}}{\sqrt{n_A\,n_B}} \in [0,1], \tag{1}$$

so a pair that is rarely co-observed (overlap $O_{AB}\!\to\!0$) has its true association attenuated toward zero purely as a function of *who was measured*. Because missingness in FACE is cohort-patterned, the attenuation factor $O_{AB}$ is itself a cohort signal — exactly the artifact a trans-diagnostic study must avoid. We therefore estimate every covariance on observed cells only (§2.6) and accept the resulting need for masked, rather than complete-data maximum-likelihood, machinery.

## Harmonization and dictionary

A re-curated common-variables dictionary maps each of **220 usable variables** to its per-cohort source column, applies a harmonization rule (text→code recoding, unit reconciliation — e.g. haematocrit L/L → %, MCHC g/L → g/dL), and enforces a per-variable **sanity bound**; out-of-range values are set to missing and never imputed. Quality control confirmed that 196/196 loaded variables pass loading and sanity checks. Variables span clinical sections (auto-/hetero-questionnaires, suicide, medical evaluation, antecedents, substances, social, hospitalization), **biology** (blood panels, vitals/ECG) and **cognition** (WAIS subtests, Trail-Making A/B, California Verbal Learning Test verbal memory, and phonemic/semantic verbal fluency). Administrative identifiers (patient id, cohort, arm, visit, site) are retained for stratification but excluded from all feature sets.

A further harmonization step decodes instrument **skip-logic**: conditional items that a gate question leaves blank (e.g. the suicide-attempt counts ISF07/08a/09a, asked only when "ever attempted" = yes) are *structural zeros*, not missing data. Where a gate is explicitly "no" and the dependent cell is blank, we set the structural zero — never overwriting an observed value, never where the gate is unknown. This is decoding, not imputation, and it recovers the attempt-count coverage from 25–38% to 72–92% across cohorts.

## Three-stage processing, with scaling equations

Processing proceeds through three stages.

**Stage 1 — native clinical scale.** Each variable lands on its native scale (TMT seconds, WAIS 1–19, Likert 0–3, binary 0/1, labs in clinical units) after harmonization and sanity bounding.

**Stage 2 — type-aware scaling to $[-1,1]$.** Scaling is by variable type. Binary/ordinal/Likert variables are min–max mapped, $x' = 2(x-\min)/(\max-\min) - 1$. Continuous variables are robust-standardized: an optional $\log(1+x)$ for heavy right-skew (e.g. prolactin), winsorization at the 1st/99th percentile, then a median/MAD robust-$z$ clipped at $\pm 5$ and rescaled,

$$\tilde z_j \;=\; \frac{1}{5}\,\operatorname{clip}\!\left(\frac{x_j - \operatorname{median}(x_j)}{1.4826\,\operatorname{MAD}(x_j)},\; -5,\; +5\right) \;\in\; [-1,1]. \tag{2}$$

This places a lab in the thousands and a 0/1 flag on the same footing and bounds outliers (it corrected an early robust-$z$ explosion, prolactin $|z|\approx106 \to \le 5$). All 196 features land in $[-1,1]$.

**Stage 3 — model inputs.** Standardization fixes *scale* but, as we now show, not two further problems that distort every inner-product or squared-error method. These motivate the measurement model of §2.7.

## Why aggregate: the count/redundancy bias

Feeding standardized **items** directly to a factor, clustering or autoencoder model is *not* "analysis-ready," for two reasons that scaling cannot touch.

**(i) Count/redundancy bias.** Any method that sums over columns (covariance, distance, reconstruction error) gives a construct geometric weight proportional to **how many items its questionnaire happens to contain** — an accident of instrument design, not clinical importance. For $m$ items with average inter-item correlation $\rho$, the shared variance piles into a single leading eigenvalue

$$\lambda_1 \;=\; 1 + (m-1)\,\rho, \tag{3}$$

so an $m$-item construct can dominate "the largest axis of variation" while a clinically equal single-item construct cannot form a factor at all. In a controlled synthetic check (two equally-important constructs, one with 1 and one with 10 correlated items), the 10-item construct's contribution to distance was **6.4×** larger and PC1 loaded $\approx 0.32$ on each of its items and $\approx 0$ on the single-item construct. The bias is **method-agnostic** — a linear autoencoder *is* PCA, and a Gaussian VAE likelihood *is* weighted MSE — so switching to deep models inherits, not dissolves, it. On our data the top five instruments occupy **25%** of all item-axes and the suicide block alone occupies **19%** (34 of 183 item-columns).

**(ii) Structured missingness.** Under no-imputation, item-level covariance is poorly conditioned and dominated by the largest cohort. Aggregating items to construct scores both removes the count bias and densifies coverage:

| level | median co-obs/pair | pairs zeroed (<100) | neg-eigenvalue mass | condition number |
|---|---:|---:|---:|---:|
| item (183) | 3,453 | 10.1% | 0.6% | ≈ 1.4 × 10⁹ |
| construct (72) | 4,102 | 3.4% | 0.0% | ≈ 1.1 × 10² |

Crucially, **within schizophrenia only about 60% of items exist and most item-pairs have fewer than 100 co-observed patients** — an item-level structure would largely be *bipolar's* structure, a direct threat to a trans-diagnostic claim. Aggregation is therefore a deliberate, low-variance **measurement-model prior**; we estimate that model from the data rather than asserting flat averages (§2.7), and we test that the headline axes do not depend on the choice of granularity (§2.9, granularity invariance).

## The masked estimator

All factor structure derives from a single imputation-free estimator.

**Masked correlation.** Each entry of the correlation matrix uses only patients observed on **both** variables; pairs with fewer than a minimum of 100 co-observed patients are set to 0 (treated as uncorrelated — a covariance choice, never the imputation of a value). The result is projected to the nearest positive-definite correlation matrix,

$$\tilde R \;=\; \arg\min_{X \succeq 0,\; \operatorname{diag}(X)=1}\; \lVert R - X\rVert_F, \tag{4}$$

approximated by eigenvalue clipping ($\lambda \leftarrow \max(\lambda,\varepsilon)$) and diagonal renormalization. Pre-repair negative-eigenvalue mass is reported as a conditioning diagnostic.

**Principal-axis factoring (PAF).** Loadings are extracted by iterated communalities: initialize $h^2$ at the squared multiple correlations, place $h^2$ on the diagonal of $\tilde R$, take the top-$k$ eigenpairs $L = V_k \operatorname{diag}(\sqrt{\lambda_k})$, update $h^2 \leftarrow \sum_k L^2$, and iterate to convergence. Orthogonal simple structure is obtained by Kaiser **varimax**; correlated structure by **promax**.

**Masked posterior factor scores.** A patient is scored on the factors using only her observed entries. With standardized observed sub-vector $z_{i,o}$, loading sub-matrix $L_o$ and uniquenesses $\Psi = I - \operatorname{diag}(LL^\top)$ (floored to guard Heywood cases), the regression (Thomson) score is

$$\hat f_i \;=\; \bigl(I_k + L_o^\top \Psi_o^{-1} L_o\bigr)^{-1} L_o^\top \Psi_o^{-1}\, z_{i,o}. \tag{5}$$

No imputed value ever enters a score; rows with fewer than $k$ observed entries are left missing.

## The hierarchical/bifactor measurement model

We replace flat construct means with an estimated, two-level measurement model in **hybrid** mode — clinically anchored construct boundaries, revised by the data. Formally,

$$z = \Lambda_1 f_1 + \varepsilon_1,\quad \operatorname{Cov}(f_1) = \Phi_1; \qquad f_1 = \Lambda_2 f_2 + \varepsilon_2 \;\Longrightarrow\; \Phi_1 = \Lambda_2\,\Phi_2\,\Lambda_2^\top + \Psi_2, \tag{6}$$

where $\Lambda_1$ maps items to **first-order construct factors** (depression, mania, adiposity, inflammation, executive function, …) and $\Lambda_2$ maps those to **second-order trans-diagnostic dimensions**. The model is fitted in four stages.

**Stage 0 — item set.** We froze **194 V0 items** comprising every valid measurement, including 34 labs/vitals that flat composites had silently dropped (thyroid, vitamin-D, full blood count, orthostatic BP, heart rate) and the six dictionary-review additions (CVLT verbal-memory recall ×3, phonemic/semantic fluency, QIDS-13 anhedonia). We excluded identifiers; age and sex (residualized); treatment/pregnancy markers (clozapine, oxcarbazepine, β-hCG); the deeply-conditional suicide method/lethality items (LTSV/LTSG) with 0–1 complete cases — the gated attempt *counts* (ISF07/08a/09a) are instead recovered by skip-logic decoding (§2.3) and retained; and one by-construction-collinear TMT difference. The item correlation was factorable (leading eigenvalues 13.3, 10.2, 7.7, 6.3, 5.2, …; 56 eigenvalues > 1) but near-singular ($\kappa \approx 1.4\times10^9$; shrunk-KMO 0.69), confirming that item-level inference is ill-posed and motivating aggregation.

**Stage 1 — exploratory first-order structure.** Masked EFA with Horn parallel analysis [9] returned **43 nameable first-order factors** that independently confirmed the aggregation problems: the metabolic block split into adiposity / blood-pressure / lipids / cholesterol; childhood-trauma *denial* items split from genuine trauma; C-SSRS split into severity / intensity; and the recovered labs/vitals formed coherent autonomic-HR, red-cell, inflammation and vitamin-D factors. The substantive factors reproduced on a leave-bipolar-out split (mean Tucker congruence 0.91), i.e. they are not bipolar-driven.

**Stage 2 — first-order construct scores.** Each construct's score is the **within-construct masked one-factor posterior** (Eq. 5 with $k=1$) over its sign-oriented items, oriented so that higher = more pathological where a severity pole exists. This estimates item weights (rather than flat $1/m$), drops misfitting items, and splits multidimensional constructs — yielding **94 constructs** (including a clean verbal-memory construct, $\mathrm{VAF}_1=1.00$, and verbal-fluency, 1.00, from the additions). Per-construct unidimensionality is summarized by the variance accounted for by the first factor, $\mathrm{VAF}_1 = \lambda_1/\sum_j \lambda_j$. Splitting concentrated signal that flat means had diluted (adiposity $\mathrm{VAF}_1 = 0.93$, cholesterol 0.90, autonomic-HR 0.86 vs the collapsed metabolic mean 0.40). The construct correlation $\Phi_1$ showed coherent second-order seeds (a minority of pairs with $|r|>0.3$, max 0.74) — warranting a second level. A 24-flag medical-comorbidity bin ($\mathrm{VAF}_1=0.38$) was decomposed, data-anchored, into a stable cardiac-history construct ($\mathrm{VAF}_1=0.50$) and a weak atopic/inflammatory one (0.26); 13 flags with <2% prevalence were removed from the inputs and retained as **Stage-4 validators**.

**Stage 3 — second-order dimensions and the general-factor test.** We factored $\Phi_1$ (restricted to 81 constructs with ≥30% coverage, standardized; 0% negative-eigenvalue mass — the conditioning gain over item level) with PAF and an oblique **promax** rotation, giving second-order loadings $\Lambda_2$ and dimension correlations $\Phi_2$. A general factor is **tested, not assumed**, by a Schmid–Leiman orthogonalization [10]: let $\gamma$ be the loadings of the $K$ oblique dimensions on a single second-order general factor (the leading eigenvector of $\Phi_2$ scaled by $\sqrt{\lambda}$); then each construct's general and specific loadings are

$$g = \Lambda_2\,\gamma, \qquad S = \Lambda_2 \odot \sqrt{1-\gamma^{2}}, \qquad \mathrm{ECV} = \frac{\sum_j g_j^{2}}{\sum_j g_j^{2} + \sum_{j,k} S_{jk}^{2}}, \tag{7}$$

where the **explained common variance (ECV)** is the share of common variance attributable to the general factor; $\mathrm{ECV} \gtrsim 0.5$ would warrant a dominant *p*-factor.

**Dimensionality $K$** is locked by **masked split-half reproducibility**, not by eigenvalue->1 rules (which over-extract here). For each candidate $K$ we repeatedly split patients, re-extract varimax loadings on each half, Hungarian-match factors, and record the minimum **Tucker congruence**

$$\phi(a,b) = \frac{a^\top b}{\sqrt{(a^\top a)(b^\top b)}}, \tag{8}$$

averaged over splits. Because split-half congruence is non-monotonic in $K$ (a known trap — naïve "max-$K$ above threshold" rules recover spurious high-$K$ solutions with improper Heywood loadings), we lock $K$ at the **first collapse minus one** and confirm by a **per-factor** congruence refinement (a single unstable factor can drag the minimum down without indicting the stable ones).

## Stratification arm (test for discrete subtypes)

In parallel, we tested whether patients form **discrete subtypes** rather than lying on continua. Pairwise patient similarities were computed with masked kernels (cosine; Gower for mixed types [11]) on observed-shared features only, each returning an overlap count used to enforce a minimum-support constraint. The masked similarity graph was embedded by a multipartite spectral method and subjected to a structure-test battery: **HDBSCAN** density clustering [12]; the gap between the real silhouette and a Gaussian-null silhouette across $k$; per-axis **Sarle bimodality**

$$b = \frac{g_1^{2} + 1}{\,g_2 + \dfrac{3(n-1)^{2}}{(n-2)(n-3)}\,}, \qquad b > 0.555 \Rightarrow \text{possible bimodality}, \tag{9}$$

(with $g_1,g_2$ the sample skewness and excess kurtosis); bootstrap cluster stability; and agreement with DSM labels by the adjusted Rand index (ARI). Two feature sets were used: **A**, the 4 dimensions plus the 2 orthogonal standalone constructs; **B**, the 75 construct scores.

## Validation (Studies A–D)

Four pre-registered studies tested whether the structure is real and useful.

**Study A — cohort confound.** Because the three cohorts *are* the three diagnoses, the axes might encode between-cohort differences. We re-derived the structure (i) **within** each cohort and (ii) after **residualizing each construct on cohort** (removing all between-cohort means), and measured Tucker congruence (Eq. 8) against the pooled solution. A within-cohort, cohort-residualization-robust axis is genuinely trans-diagnostic.

**Study B — symptom–biology orthogonality and the *p*-factor.** Constructs were assigned to **symptom**, **biology**, **cognition** or **other** blocks by their items' clinical section. We computed the distribution of construct–construct $|r|$ **within** versus **between** blocks, and the general-factor strength — both first-factor share $\lambda_1/\sum_j\lambda_j$ and ECV (Eq. 7) — for the nested sets symptom-only → +cognition → +biology → full. Because the internalizing (mood) scales are measured only in BP + DR (Study A), the **clean** analysis is **within BP + DR**; pooled is a sensitivity.

**Study C — longitudinal coherence.** Applying the V0 construct definitions to V1 and V2, we re-estimated the $K=4$ loadings at each visit and measured (i) **structural invariance** (Tucker congruence vs V0) and (ii) **score stability** (rank-order Spearman test–retest of projected scores), separating the persistence of the axes from individual movement along them.

**Study D — predictive validity vs DSM.** Out-of-sample, cohort-stratified cross-validation compared nested predictor sets for each outcome: $M_0$ = age + sex (+ the V0 baseline of the outcome for symptom outcomes); $M_1 = M_0 + \text{DSM}$ (finest diagnosis, one-hot); $M_2 = M_0 + \text{dimensions}$ (4 axes + mania + suicidality); $M_3 = M_0 + \text{DSM} + \text{dimensions}$; and $M_{2x}$, a **cross-domain** set dropping internalizing to give a strictly non-circular test. Continuous outcomes (GAF, FAST) used ridge regression and out-of-sample $R^2$; binary outcomes used logistic regression and AUC; increments $\Delta R^2$, $\Delta\text{AUC}$ carried bootstrap 95% CIs. **Circularity** was controlled by always including the V0 baseline of any symptom outcome and by leading on cross-domain/hard outcomes; **attrition** was checked by predicting follow-up availability from the axes.

The relapse outcome was derived with care. A hospitalization-count relapse was **rejected** because the lifetime count is non-monotone in the data (41% of consecutive pairs decrease — a recording artifact). The change-based clinical relapse (CGI-S rising ≥2 points or crossing into "moderately ill") proved confounded by **regression to the mean**. The clean outcome is therefore a **remission-based discrete-time survival** model: among patients remitted at V0 (CGI-S 1–3), relapse is deterioration to CGI-S ≥ 4, modelled as a pooled discrete-time hazard over person-intervals,

$$\operatorname{logit}\,\Pr(T_i = t \mid T_i \ge t,\, x_i) = \alpha_t + \beta^\top x_i, \tag{10}$$

with an interval baseline-hazard term $\alpha_t$, **GroupK-fold by patient** (a patient's intervals never split train/test), and bootstrap CIs resampling **patients**. Two estimators were run identically per predictor set: regularized logistic regression and histogram gradient boosting.

## Software and reproducibility

Analyses use Python 3.13 (NumPy, pandas, SciPy, scikit-learn, `factor_analyzer`, Matplotlib). Every stage is a numbered, deterministic (fixed-seed) script writing aggregate artifacts; figures regenerate from those artifacts. The full v1 study is archived at a git tag; the present v2 analysis is independent.

# Results

## A four-dimension trans-diagnostic structure, with no general factor

The second-order model resolved **four reproducible trans-diagnostic dimensions** (**Fig. 2**):

1. **Internalizing** — depression (QIDS 0.94, MADRS 0.88), anxiety (STAI 0.82), anhedonia (QIDS-13 loss of interest, 0.75), disability (FAST 0.72) and poor functioning/quality-of-life (GAF, EQ-5D, reverse-keyed);
2. **Cognition** — now anchored by **verbal episodic memory** (CVLT total recall 0.79, delayed free recall 0.77), with verbal fluency (semantic 0.65, phonemic 0.55), executive function (TMT-B 0.70), processing speed (0.67) and psychomotor speed (0.63) co-loading, and education a negative correlate;
3. **Illness course** — later age at first hospitalization (0.84), treatment (0.76) and episode (0.66) against inverse lifetime hospitalization burden (higher = later-onset / lower-chronicity);
4. **Cardiometabolic–inflammatory** — HDL/triglycerides (0.49), inflammation (CRP/WBC, 0.48), autonomic heart rate (0.47), adiposity (0.46), blood pressure, hepatic enzymes, cholesterol and lymphocytes.

The axes are only weakly correlated (mean $|\Phi_2| = 0.16$; the largest is internalizing↔course $-0.35$, reflecting that more chronic illness co-occurs with worse mood). Critically, the Schmid–Leiman test returned **ECV = 0.34** — **no dominant general (*p*-)factor**; the structure is genuinely multidimensional. Two well-measured constructs, **mania/activation** (Altman + YMRS, $\mathrm{VAF}_1=0.71$) and **suicidal ideation** (ISF), proved **orthogonal** to all four axes ($|r| \le 0.10$) and are reported as independent standalone dimensions rather than forced into the correlated structure — itself a finding (mania's independence from internalizing). Notably, suicidal ideation remained orthogonal *even after* skip-logic decoding recovered its attempt-count items (§2.3; coverage 25–38% → 72–92%), so its standalone status is a structural fact, not a missing-data artifact.

The four-dimension solution passed every validation check (**Table 2**): it was confound-clean (no axis explained >0.25 by cohort, sex, age, site or per-patient missingness; cognition's cohort $\eta^2=0.16$ is matched by an education $\eta^2=0.17$, a genuine correlate); trans-diagnostic yet clinically valid (internalizing highest in depression, cognition worst in schizophrenia); **leave-cohort-out reproducible** (dropping BP, SZ, DR gave minimum per-axis congruence 0.78 / 0.96 / 0.98); and **granularity-invariant** — canonical correlations between the hierarchical axes and a flat-domain factor solution were **0.99, 0.90, 0.79, 0.38** (permutation null ≈ 0.04), so the top three axes are *not* an artifact of how items were grouped (the fourth differs only because the hierarchical model adds the recovered labs/vitals the flat domains dropped). A polychoric (tetrachoric) sensitivity reproduced the four dimensions exactly (congruence 1.00), and suicidal ideation remained sub-threshold (max loading 0.29) — even after skip-logic recovered its coverage — confirming its exclusion is neither a Pearson-attenuation nor a missing-data artifact.

![**Figure 1. Study design and analytic pipeline.** Three FACE cohorts (N = 9,013) are harmonized into a 220-variable dictionary and processed in three stages (native scale → type-aware scaling to [-1,1] → V0 item matrix) under a strict no-imputation rule. A hierarchical/bifactor measurement model maps 194 baseline items → 94 first-order constructs (within-construct masked one-factor posteriors) → four second-order trans-diagnostic dimensions plus two orthogonal standalone constructs (ECV 0.34 → no *p*-factor). Two analysis arms (dimensional; stratification) feed four validation studies (A–D).](../reports/figures/fig1_pipeline.png){width=6.6in}

![**Figure 2. The four trans-diagnostic dimensions.** Defining constructs per axis (|second-order loading| > 0.30; red = higher score more pathological, blue = reverse-keyed). The Φ₂ panel shows weak inter-axis correlations (mean |Φ₂| = 0.16); Schmid–Leiman ECV = 0.34 indicates no dominant general factor. Mania and suicidal ideation are valid but orthogonal (|r| ≤ 0.10) and are not axes.](../reports/figures/fig2_axes.png){width=6.3in}

## Trans-diagnostic variation is dimensional, not categorical

Both analysis arms agreed that the structure is **continuous, not categorical** (**Fig. 4**). On the dimensions (set A), HDBSCAN found **no real density structure** (only two micro-pockets at 87% noise, themselves unrelated to cohort, ARI 0.04); the real silhouette barely exceeded a Gaussian-null silhouette and did not peak at any $k$ (gaps 0.01–0.05); every axis was unimodal (Sarle $b < 0.555$); and $k$-means partitions bore essentially no relation to DSM (ARI ≈ 0.03). On the 75 construct scores (set B), the **only** dense clusters HDBSCAN recovered were the **three cohorts themselves** (ARI = 1.00) — i.e. the sole categorical structure in the data is the DSM diagnosis, and finer granularity revealed **no novel subtypes**. (High $k$-means bootstrap stability, 0.79–0.93, is a continuum artifact — $k$-means partitions a blob reproducibly — and is over-ruled by the absence of density clusters and the unimodality.) Combined with the absence of a general factor, this yields a clean dimensional account: **continuous axes, no *p*-factor, no discrete subtypes**.

![**Figure 4. Trans-diagnostic variation is dimensional, not categorical.** (a) The real-data silhouette tracks a Gaussian-null silhouette and never peaks; HDBSCAN finds no meaningful density structure (only micro-pockets at ~87% noise) and k-means bears no relation to DSM (ARI ≈ 0.03). (b) Every axis is unimodal (Sarle b < 0.555). (c) The cohorts overlap heavily in axis space — a continuum, not separated clusters.](../reports/figures/fig4_continuum.png){width=6.8in}

## Symptoms are orthogonal to biology — the *p*-factor is a symptom-only artifact

This is the central result (**Fig. 3**). Within the clean BP + DR sample, construct–construct correlations were substantial **within** the symptom block (mean $|r| = 0.24$) and within cognition (0.42, a coherent battery), but **near zero between blocks**: symptom↔biology **0.03**, symptom↔cognition 0.06, biology↔cognition 0.04. The single strongest symptom–biology link was only 0.15 (FAST↔lipids), and **no** symptom–biology pair exceeded 0.15. Symptoms, biology and cognition are, to a first approximation, **mutually orthogonal**.

It follows that a general factor cannot span the integrated space — and indeed it does not. The first-factor share fell **monotonically** as structured biology and cognition were admitted: **0.33 (symptoms only) → 0.24 (+cognition) → 0.16 (+biology) → 0.09 (full integrated)**, with ECV moving consistently downward. This is not a dilution-by-noise effect: biology and cognition are themselves **structured** — they form the coherent cardiometabolic and cognition axes — yet they are orthogonal to symptoms, so no single dimension can span them. The headline, made falsifiable and confirmed: **a general psychopathology factor is an artifact of symptom-only measurement; an integrated symptom + biology + cognition model is genuinely multidimensional with no dominant general factor.** The result was robust from BP + DR to the pooled sample.

![**Figure 3. Symptoms are orthogonal to biology; the *p*-factor is a symptom-only artifact.** (a) Block-ordered construct-correlation matrix (BP+DR): dense structure within the symptom and cognition blocks, near-zero correlations between blocks (symptom↔biology mean |r| = 0.03). (b) General-factor strength (first-factor share) falls monotonically from 0.33 with symptoms only to 0.09 when structured biology and cognition are admitted — the general factor dissolves under integration.](../reports/figures/fig3_orthogonality.png){width=6.8in}

## The axes are not a cohort artifact — and a map of what is truly trans-diagnostic

Study A refuted the obvious confound and, in doing so, produced an important qualification. Re-deriving the structure after **residualizing every construct on cohort** (removing all between-cohort means) reproduced all four axes with congruence ≥ 0.96; within bipolar disorder alone (n = 6,252) all four reproduced ≥ 0.95. The axes are therefore **within-cohort covariance**, not between-cohort batch effects.

The within-schizophrenia analysis revealed *why* one axis is weaker there, and it is a **measurement-coverage** fact, not a clinical subtlety. The internalizing axis's defining scales — MADRS, QIDS, STAI, FAST, Altman, PRISM, CSM — are **0% observed in FACE-SZ**, which used a psychosis battery by design; schizophrenia patients are scored on internalizing only through surviving three-cohort proxies (GAF, CGI, PSQI, EQ-5D), giving within-SZ congruence 0.76. By contrast, cognition and illness-course are cleanly three-cohort (top constructs 43–92% observed in SZ), and cardiometabolic is three-cohort in its core (lipids, adiposity, glycaemia, CRP) with only peripheral BP+DR additions. The consequence, which we state plainly throughout: the **fully** trans-diagnostic axes are **cognition and illness-course**, with **cardiometabolic** three-cohort in its biological core but only weakly reproduced in the small FACE-DR within-cohort test (n = 552; congruence 0.35, underpowered) though robust under cohort-residualization, whereas the **internalizing (mood) axis is directly measured only in BP + DR and represented by proxy in schizophrenia**. The two weakly-reproduced axes (internalizing, cardiometabolic) are flagged accordingly. The dimensional, no-*p*-factor and no-subtype results are unaffected; the "trans-diagnostic" label is qualified for internalizing (a measurement gap in SZ) and, with a power caveat, for cardiometabolic in DR.

## Longitudinal coherence

The structure persisted at follow-up wherever it could be re-measured (**Fig. 6**, Study C). Re-estimated per visit, the loadings were highly congruent with V0 for internalizing (0.99 / 0.98 at V1 / V2) and cardiometabolic (0.98 / 0.97), and recovered for cognition at V2 (0.87; ~0 at V1 because the WAIS battery is baseline-anchored and ~5% re-measured at V1). Rank-order score stability cleanly separated **trait, state and fixed-historical** axes: cardiometabolic was the most trait-stable (Spearman 0.66 / 0.62), internalizing was moderately stable as befits episodic mood (0.59 / 0.53), and illness-course showed low test–retest (0.12 / 0.16) **by design** — its age-of-onset constructs are baseline-only historical items, not re-measurable states. Together with Study B, this identifies the **most measurement-robust trans-diagnostic substrate as biological/cognitive**, with symptom structure more state- and cohort-specific.

![**Figure 6. Longitudinal coherence (V0 → V1 → V2).** (a) Structural invariance: per-visit loadings are highly congruent with V0 for internalizing and cardiometabolic; cognition recovers at V2 (its battery is baseline-anchored). (b) Score test–retest separates trait (cardiometabolic), state/episodic (internalizing) and fixed-historical (illness-course, whose age-of-onset core is collected only at baseline) axes.](../reports/figures/fig6_longitudinal.png){width=6.6in}

## Predictive validity versus DSM: modest but real

The make-or-break test was whether the dimensions add prognostic value over DSM diagnosis out of sample (**Fig. 5**, **Table 3**). The axes did not bias the completer sample (V0 axes predicted follow-up availability at AUC 0.531, near chance).

For **functioning**, the dimensions added a modest but robust, **non-circular** increment over DSM: for GAF at V2 (n = 2,043) $\Delta R^2 = +0.046$ (95% CI +0.030, +0.062) over DSM, +0.033 cross-domain; for FAST disability (BP + DR, n = 1,878) the dimensions **beat** DSM outright ($\Delta R^2 = +0.038$ over DSM, which added nothing), +0.026 cross-domain. The functional signal was led by illness-course and internalizing; cognition and cardiometabolic added little individually.

For **relapse**, the honest story required removing a confound. A change-based relapse appeared near-perfectly predicted by baseline severity (AUC 0.765) with no room for the dimensions — but that was **regression to the mean**: under the de-confounded remission-based discrete-time survival model, baseline-only prediction fell to AUC **0.578**. There the dimensions **did** add a modest increment over DSM (logistic $\Delta\text{AUC} = +0.036$, CI +0.014, +0.057; gradient boosting +0.012, ns — i.e. linearly detectable but not boosting-robust), carried by residual internalizing symptoms. Relapse-from-remission remained hard to predict (best AUC ≤ 0.65), and enriching the baseline with all 75 constructs did not break 0.64. AUC ≈ 0.70 was reachable only by adding the **early course** (predicting the V1→V2 interval from V0 + V1 trajectory, with V1 severity controlled and no leakage): there the trajectory model reached AUC ≈ 0.70 and beat DSM by +0.05 — a different, clinically sensible question (early-response prognosis), honestly distinguished from baseline prediction.

The verdict is therefore **partial but genuine**: the dimensions are at least DSM-equivalent and add a small, significant, non-circular increment to **functional** prognosis, and a **modest** de-confounded increment to **relapse** — more than descriptive, but not a transformation of prediction.

![**Figure 5. Predictive validity versus DSM.** (a) Incremental performance of the dimensions over DSM (ΔR² for functioning, ΔAUC for relapse) with 95% CIs; green = CI excludes 0. Functioning (GAF, FAST) gains are robust and non-circular; the de-confounded relapse increment is significant by logistic regression. (b) The relapse narrative: apparent baseline dominance (AUC 0.77) is a regression-to-the-mean confound (de-confounded baseline 0.58); the dimensions then add a modest increment, and early-course (V0+V1) trajectory information reaches AUC ≈ 0.70.](../reports/figures/fig5_predictive.png){width=6.8in}

| Outcome (n) | Metric | M0 base | M1 +DSM | M2 +dims | M3 +both | Δ dims over DSM [95% CI] |
|---|---|---:|---:|---:|---:|---|
| GAF @ V2 (2,043) | $R^2$ | 0.270 | 0.310 | 0.314 | 0.357 | **+0.046 [+0.030, +0.062]** |
| FAST @ V2, BP+DR (1,878) | $R^2$ | 0.265 | 0.265 | 0.303 | 0.301 | **+0.036 [+0.021, +0.051]** |
| Relapse, change-based (3,378) | AUC | 0.765 | 0.764 | 0.768 | 0.766 | +0.002 [-0.004, +0.008] (ns) |
| Relapse, de-confounded — logistic (1,766) | AUC | 0.578 | 0.614 | 0.631 | 0.650 | **+0.036 [+0.014, +0.057]** |
| Relapse, de-confounded — gboost (1,766) | AUC | 0.576 | 0.610 | 0.597 | 0.621 | +0.012 [-0.017, +0.041] (ns) |
| Relapse, early-course — logistic (989) | AUC | 0.640 | 0.653 | 0.681 | 0.691 | **+0.046 vs DSM [+0.004, +0.087]** |

Table: **Predictive validity versus DSM, out of sample.** Functioning outcomes use ridge $R^2$; relapse uses AUC. Bold increments have 95% CIs excluding 0. The change-based relapse is shown to document the regression-to-the-mean confound that the remission-based survival model removes.

# Discussion

## Principal findings

Across three deeply-phenotyped major psychoses, integrated under a strict no-imputation design, trans-diagnostic structure is **dimensional, not categorical** — four reproducible continuous axes (internalizing, cognition, illness-course, cardiometabolic–inflammatory), with no discrete subtypes beyond the DSM cohorts and **no dominant general factor**. The central, non-derivative result is that **symptoms are nearly orthogonal to biology**, and that the general psychopathology factor is a **symptom-only artifact**: it dissolves monotonically as structured biology and cognition enter the same latent space. The axes are not a cohort artifact, are longitudinally coherent, and add a modest-but-real increment over DSM for functioning and (de-confounded) relapse.

## Interpretation

The orthogonality result reframes the *p*-factor debate. A general factor is routinely recovered from symptom batteries and interpreted as a substantive liability [4–6]. Our data suggest a more parsimonious reading: the general factor indexes covariance **among symptom reports** — plausibly reflecting shared method, distress and help-seeking — and does **not** extend to the biological and cognitive systems usually invoked to explain it. When those systems are measured and modelled jointly, the integrated space is multidimensional with separable substrates. This is consistent with long-standing cautions that bifactor general factors are statistically over-permissive and often unreliable [7,8], and it gives the RDoC premise [2] an empirical edge over a single-liability view: the biological/cognitive axes here are the *most* measurement-robust and the *most* orthogonal to symptoms. It also sharpens HiTOP [3] — the symptom hierarchy is real, but its apex does not generalize to biology.

## Measurement design sets the limits — stated up front

We foreground, rather than bury, the design limits. **Internalizing is BP + DR-anchored**: its defining mood/anxiety/functioning scales are absent in FACE-SZ by protocol, so schizophrenia is scored by proxy and the "trans-diagnostic" claim is qualified for that axis alone. **Cognition is baseline-anchored** (the WAIS battery is largely not re-administered), so its longitudinal test is limited to V2. **Illness-course is fixed-historical** — age-of-onset is recorded only at baseline — so its low test–retest is a design feature, not instability. **Cardiometabolic is the most trait-stable, symptom-orthogonal** axis with a three-cohort biological core; its one weakness is the small FACE-DR within-cohort test (n = 552, underpowered), where it reproduced at only 0.35 although it survives cohort-residualization (≥ 0.96). The depression cohort is small (n = 552) and collapses longitudinally, so all temporal and predictive claims are effectively bipolar + schizophrenia; reassuringly, leave-depression-out congruence was 0.98, so the structure does not depend on it.

## Clinical implications

The clinical reading is deliberately measured. The dimensions are **at least DSM-equivalent** for prognosis and add a small, significant, biologically-grounded increment — robust for functioning, modest for de-confounded relapse — but they are not a transformation of prediction, and we do not present them as one. Two points have practical weight nonetheless. First, the **regression-to-the-mean confound** we expose is a general hazard for relapse prediction from baseline severity, and the remission-based survival design is a transferable remedy. Second, reaching AUC ≈ 0.70 required **early-course** information (the first follow-up year), suggesting that prognostic value in these disorders lies more in **trajectory** than in any cross-sectional snapshot.

## Anticipated objections

We close by addressing the objections a sceptical reviewer will (rightly) raise.

1. *"This merely confirms HiTOP/RDoC."* The dimensional structure is partly confirmatory; the **non-derivative** contribution is the integrated-model result — symptoms ⊥ biology and the *p*-factor as a symptom-only artifact — which symptom-only designs cannot reach.
2. *"The three cohorts are the three diagnoses, so the axes are batch effects."* Refuted in Study A: cohort-residualized re-derivation reproduces all four axes ≥ 0.96, and within-bipolar ≥ 0.95. The structure is within-cohort covariance.
3. *"Aggregating items into constructs manufactures the structure."* Granularity invariance (top-three canonical r = 0.99/0.90/0.79 vs flat-domain and item-level solutions; permutation null 0.04) shows the headline axes do not depend on grouping.
4. *"The dissolution of the *p*-factor is just dilution by adding noise variables."* No: biology and cognition are **structured** (they form coherent axes) yet orthogonal to symptoms, so no single factor can span them. Both first-factor share and ECV fall monotonically.
5. *"You over-claim 'trans-diagnostic' when internalizing isn't measured in schizophrenia."* We state this prominently and restrict the fully trans-diagnostic claim to cognition, illness-course and core cardiometabolic.
6. *"No imputation with masked correlations is unstable and you're hiding non-PSD matrices."* We report conditioning explicitly (item κ ≈ 1.3×10⁹ → construct κ ≈ 110; construct-level negative-eigenvalue mass 0%), repair to the nearest PD matrix (Eq. 4), and show a polychoric sensitivity reproduces the axes (congruence 1.00).
7. *"Why not complete-data ML confirmatory SEM / Bayesian CFA?"* The no-imputation rule precludes complete-data maximum likelihood; we instead report cross-validated congruence and **test** (not assume) the general factor by ECV — and document the hybrid measurement model as an explicit bias–variance choice for a small, unbalanced cohort.
8. *"The prognostic gains are clinically trivial."* We agree they are modest and report them as such (ΔR² ≈ 0.04), without overselling; the contribution is honesty about a small, non-circular, de-confounded increment, not a claim of clinical transformation.
9. *"The relapse outcome is arbitrary or gameable."* The hospitalization-count outcome was rejected on data-quality grounds (41% non-monotone lifetime counts); the remission-based discrete-time survival design removes regression to the mean; GroupK-fold and patient-level bootstrap remove leakage.
10. *"Depression is tiny and longitudinally thin."* Acknowledged throughout; longitudinal/predictive analyses are BP + SZ, V4 is never led upon, and the structure is leave-DR-out robust (0.98).
11. *"Cross-sectional factor analysis can't speak to course or cause."* We make no causal claim; V0 defines the structure and later visits only test its coherence (Study C).
12. *"Mania should be an axis (as in many models)."* Mania/activation is a valid, well-measured construct, but it is **orthogonal** to the four correlated axes ($|r| \le 0.09$, robust to polychoric estimation); forcing it into the structure would misrepresent the data. Its independence from internalizing is itself reported.

## Strengths and limitations

Strengths are the integrated multimodal phenotyping, the genuine trans-diagnostic breadth, the strict imputation-free estimator, the explicit general-factor test, and the pre-registered confound/longitudinal/predictive validation. Limitations are the measurement-coverage asymmetries detailed above, the small and longitudinally thin depression cohort, the reliance on a single network (external replication is needed), and the absence of neuroimaging or genomics — modalities that could either extend or further fragment the biological axis.

## Conclusion

An integrated, imputation-free model of bipolar disorder, schizophrenia and major depression is multidimensional, with separable symptom, cognitive and biological substrates and **no dominant general factor**. The general psychopathology factor appears to be an artifact of symptom-only measurement. The resulting dimensions are at least DSM-equivalent for prognosis and add a small, biologically-grounded increment — a rigorous, honestly-bounded step toward a trans-diagnostic account that takes biology as seriously as symptoms.

# References

1. American Psychiatric Association. *Diagnostic and Statistical Manual of Mental Disorders*, 5th ed. Washington, DC: APA; 2013.
2. Insel T, Cuthbert B, Garvey M, et al. Research Domain Criteria (RDoC): toward a new classification framework for research on mental disorders. *Am J Psychiatry*. 2010;167(7):748–751.
3. Kotov R, Krueger RF, Watson D, et al. The Hierarchical Taxonomy of Psychopathology (HiTOP): a dimensional alternative to traditional nosologies. *J Abnorm Psychol*. 2017;126(4):454–477.
4. Caspi A, Houts RM, Belsky DW, et al. The *p* factor: one general psychopathology factor in the structure of psychiatric disorders? *Clin Psychol Sci*. 2014;2(2):119–137.
5. Lahey BB, Applegate B, Hakes JK, et al. Is there a general factor of prevalent psychopathology during adulthood? *J Abnorm Psychol*. 2012;121(4):971–977.
6. Caspi A, Moffitt TE. All for one and one for all: mental disorders in one dimension. *Am J Psychiatry*. 2018;175(9):831–844.
7. Bonifay W, Lane SP, Reise SP. Three concerns with applying a bifactor model as a structure of psychopathology. *Clin Psychol Sci*. 2017;5(1):184–186.
8. Watts AL, Poore HE, Waldman ID. Riskier tests of the validity of the bifactor model of psychopathology. *Clin Psychol Sci*. 2019;7(6):1285–1303.
9. Horn JL. A rationale and test for the number of factors in factor analysis. *Psychometrika*. 1965;30(2):179–185.
10. Schmid J, Leiman JM. The development of hierarchical factor solutions. *Psychometrika*. 1957;22(1):53–61.
11. Gower JC. A general coefficient of similarity and some of its properties. *Biometrics*. 1971;27(4):857–871.
12. Campello RJGB, Moulavi D, Sander J. Density-based clustering based on hierarchical density estimates. *PAKDD*. 2013;160–172.
13. Reise SP. The rediscovery of bifactor measurement models. *Multivariate Behav Res*. 2012;47(5):667–696.
14. Lorenzo-Seva U, ten Berge JMF. Tucker's congruence coefficient as a meaningful index of factor similarity. *Methodology*. 2006;2(2):57–64.
15. Marquand AF, Wolfers T, Mennes M, Buitelaar J, Beckmann CF. Beyond lumping and splitting: a review of computational approaches for stratifying psychiatric disorders. *Biol Psychiatry Cogn Neurosci Neuroimaging*. 2016;1(5):433–447.
16. Schürhoff F, Fond G, Berna F, et al. A National network of schizophrenia expert centres (FACE-SZ). *Eur Psychiatry*. 2015;30(6):728–735.
17. Henry C, Etain B, Godin O, et al. The FondaMental Advanced Centres of Expertise in Bipolar Disorders (FACE-BD). *J Affect Disord*. 2017 (and updates).
18. Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*. 2007;8(1):118–127.
19. McCullagh P, Nelder JA. *Generalized Linear Models*, 2nd ed. London: Chapman & Hall; 1989.
20. Singer JD, Willett JB. It's about time: using discrete-time survival analysis to study duration and the timing of events. *J Educ Behav Stat*. 1993;18(2):155–195.
