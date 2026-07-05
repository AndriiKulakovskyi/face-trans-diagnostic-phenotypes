# Constructing and validating a transdiagnostic dimensional map for precision psychiatry: a missingness-aware measurement model and its archetypes across bipolar disorder, schizophrenia and depression

*Working methods-paper draft — milestone-by-milestone (M1–M5). The **subject is the method**: a five-stage pipeline that builds a transdiagnostic dimensional map from harmonised multi-cohort data and then stress-tests it for structure, temporal coherence, prognostic validity and treatment moderation. Throughout, we **report the map uniformly and separate measurement from interpretation**. All figures are the reconciled, reported **Gaussian-copula 8-factor / A = 5** objects; legacy native and A = 4 numbers have been superseded (see Methods note). Internal-validity findings on observational baseline + follow-up data; no causal or prescriptive claims are made beyond what is stated.*

---

## Abstract

We present a **methodology for constructing and validating a transdiagnostic dimensional map** for precision psychiatry, and apply it to the harmonised baseline (V0) assessments of three FACE-network cohorts — bipolar disorder (BP), schizophrenia (SZ) and a depression/at-risk cohort (DR), N = 9,013. The pipeline has five reusable stages. **(M1)** A single global, missingness-aware Bayesian sparse bifactor / ESEM measurement model with mixed likelihoods and a Gaussian-copula continuous block estimates an **8-dimensional map** — a general functional-burden axis (G) and seven specific axes — directly from each patient's *observed cells*, without imputation. **(M2)** On the fixed map, a structure test plus archetypal analysis with propagated measurement uncertainty summarises the patient space as a **continuum, not biotypes**, with a stable **A = 5 archetype simplex** and a no-privileged-K tessellation family. **(M3)** Scoring two yearly follow-ups under the *frozen* model — never re-discovering structure — quantifies temporal coherence: which axes keep their meaning, which are trait vs state, and whether archetype identity persists. **(M4)** An errors-in-variables Bayesian GLM tests the **incremental prognostic validity** of a baseline coordinate beyond diagnosis, severity and the baseline outcome, with a representation benchmark against raw indicators. **(M5)** A bounds-and-defends causal pipeline (overlap → propensity → doubly-robust moderation → E-value) asks whether the map moderates treatment on observational data.

The methodological contribution is the **end-to-end, no-imputation, uncertainty-propagating pipeline** and the discipline of holding the discovered map fixed at every downstream stage. As a substantive demonstration, **one property of the map recurs across all five stages**: a single *immunometabolic* axis that is distinct from the rest of the clinical picture (M1), anchors a stable archetype corner (M2), is the most temporally durable dimension (M3), is the worst-prognosis pole for 2-year functioning (M4), and survives treatment adjustment (M5) — the kind of biologically-grounded, severity-independent signal a diagnosis-blind map can surface and a severity- or DSM-based view misses. We report all eight factors uniformly and separate measurement from clinical interpretation throughout; every finding is an internal-validity result on observational data, with no causal or prescriptive claim beyond those stated.

---

## Background and methodological framing

Diagnostic categories (DSM-5) organise psychiatric care but align weakly with biology and course. A dimensional, diagnosis-blind alternative is attractive, but is only credible if the *measurement* is done well: comparable across cohorts, faithful to mixed data types, honest about missingness and uncertainty, and held fixed when it is later validated. This paper's contribution is a **method that meets those requirements end-to-end**, organised as a four-layer pipeline that must not be collapsed:

```
diagnostic cohorts → transdiagnostic dimensions → continuous map + archetypes → prognosis / treatment
  (entry metadata)      (M1, 8-factor)               (M2, A = 5)                  (M4 / M5)
```

Three invariants are load-bearing at every stage. (1) **No imputation** — structure is estimated from each patient's observed cells via an observed-data (FIML) likelihood, never a filled matrix. (2) **Diagnosis is metadata** — cohort/DSM subtype is a covariate and a validation label, never a clustering input or a dimension indicator. (3) **Baseline defines, follow-up validates** — dimensions and strata are discovered on V0 and then *held fixed* when scoring later visits, treatment and outcome. Two reporting commitments follow from the methods-paper stance: we present the eight factors **uniformly** (loadings, direct G-loadings, factor correlations), and we keep **measurement separate from interpretation** — the clinically interesting properties of the map are read off the neutral structural tables, not built into them.

**Methods note (provenance reconciliation).** All milestones consume one chain of *reported* objects: the cohort-weighted full-N Gaussian-copula map `copula/weighted_8d/hs_s5_merged_xc` (M1) → the A = 5 strata `strata_oop_2026_06_26_v2_8factor` (M2) → the temporal panel `temporal_oop_2026_06_26_v2_8factor` (M3) → the prognosis frame `prognosis_oop_2026_06_27_v2` (M4) → the treatment pipeline `treatment_oop_2026_06_27_v3` (M5). Each milestone's data lineage was audited and confirmed to consume the copula A = 5 objects; earlier *native* (9-indicator, A = 8) and *A = 4* artifacts that survive on disk are superseded. The numbers below are the reconciled copula A = 5 values. Full vertical synthesis: [`docs/VERTICAL_FINDINGS.md`](docs/VERTICAL_FINDINGS.md).

---

## M1 — The measurement model and the 8-factor map

### What was done and why

The first and central methodological component is the **measurement model itself**. The task is to convert 109 harmonised clinical and biological indicators — spanning functioning, depression/anxiety/mania scales, neurocognition, sleep, childhood adversity, suicidality, substance use, and a broad biology panel (anthropometry, lipids, glycaemia, blood pressure, inflammatory markers) — into a small number of interpretable transdiagnostic dimensions that are, by construction, *shared* across BP, SZ and DR. Rather than fitting per-cohort models and reconciling them, we estimate **one global model** on the pooled V0 sample. (The 109 indicators are a curated modelling subset of the 225-variable common dictionary; the selection rule and full variable accounting are given in [`docs/DATA.md`](docs/DATA.md) — briefly, a variable is promoted to an *indicator* only if it is comparable across all three cohorts, a current clinical/biological state rather than a history/onset descriptor, and not redundant with another indicator; the remainder serve as covariates, identifiers, or validation labels.)

The estimator is a **missingness-aware Bayesian sparse bifactor / ESEM** with mixed likelihoods: an empirical-rank inverse-normal (**Gaussian-copula**) transform gaussianises the 88 continuous/high-cardinality indicators, while 21 explicit binary/count/ordinal items keep native probit/logit links. The Gaussian latent scores are integrated out analytically (a marginalised Woodbury / matrix-determinant-lemma observed-cell FIML likelihood), so missing cells contribute no term and nothing is imputed. Cohort imbalance (BP is ~11× DR) is handled by a 1/n-cohort weighting rather than subsampling, retaining every patient. Off-home cross-loadings are governed by a regularised ("Finnish") horseshoe prior — default-off globally, evidence-on locally, magnitude-capped — so thin factors are protected while genuine cross-talk can still emerge. The model is identified as a bifactor (G orthogonal to the specifics) for the headline structure, with a freely-correlated-G arm reserved to *measure* the biology↔severity relationship rather than assume it.

### Results and interpretation

The fit certifies at the full N = 9,013 (R-hat 1.03, ESS 97, 0 divergences; 4 chains, ~3 h on a Mac M4 Pro CPU) and resolves **8 latent dimensions**: a general factor **G** plus seven specifics — cognition, **immunometabolic** (cardiometabolic *and* inflammatory markers on one axis), sleep, mania/activation, suicidality, developmental-risk, and substance (pinned orthogonal because its cross-factor correlations are non-identifiable). The map is near-simple-structure: among the specifics the mean absolute factor correlation is ≈ 0.08, and only **three cross-loadings are earned** (CTQ-37, PSQI-latency, PSQI-daytime → cognition, each 95% CI excluding 0). A sparse-ESEM validation that freed all 294 off-home cells saw ~83% shrink to ≈ 0 — so the simple structure is *earned from the data*, not imposed.

**How each domain relates to G (reported uniformly).** Because the map is identified as a bifactor — G orthogonal to the specifics *by construction* — a domain's relationship to general burden is read from its indicators' **direct loadings on G**, not from the factor-correlation matrix Φ (where G's row is fixed at 0). The mean |G-loading| per domain is:

| domain | cognition | sleep | mania | suicidality | developmental | substance | immunometabolic |
|---|--:|--:|--:|--:|--:|--:|--:|
| mean \|G-loading\| | 0.20 | 0.20 | 0.13 | 0.11 | 0.07 | 0.07 | 0.06 |

A freely-correlated-G refit (which lets G correlate with each specific — the *measured* estimand) reproduces the same ordering for the well-covered axes: G↔cognition ≈ 0.39, G↔sleep ≈ 0.42, immunometabolic ≈ 0.10. **G itself** is read as a transdiagnostic impairment/distress axis, not a latent "p-factor" — its anchors are functioning scales with no symptom content (a subjective-burden item loads ≈ 0), and depression/anxiety/mania scales surface as *windows* onto G rather than separate dimensions. **Φ** (the latent factor correlations) is near-simple-structure across the six freely-correlating specifics (mean |off-diagonal| ≈ 0.08), with G and substance pinned to zero by construction; the only non-trivial couplings are sleep–mania +0.23, sleep–developmental +0.20, suicidality–developmental +0.20 and cognition–sleep −0.16.

**Interpretation — a periphery of three weakly-G axes, of which biology is the *earned island*.** With the structure reported, the interpretation is now read *off* it. Three axes load only weakly on general burden — immunometabolic (0.06), substance (0.07) and developmental-risk (0.07) — a near-tie on the G-loading. They are peripheral for *different* reasons, and only one is earned. **Substance** is orthogonal by construction (pinned; thin, two-cohort), so its independence is imposed rather than estimated. **Developmental-risk** is weak on G but **couples to the symptom axes** (sleep/suicidality, Φ ≈ 0.20) and is a historical antecedent rather than a current state (the least temporally durable axis, M3). **Immunometabolic load** is the only domain weakly tied to the *entire* clinical picture — both general burden (0.06 / correlated-G ≈ 0.10) **and** the symptom axes (max Φ 0.076, versus ≥ 0.16 for every other freely-correlating factor) — on a freely-estimated basis. It is, uniquely, an *earned island* in the map. *(One honest caveat: the correlated-G arm cleanly separates biology from cognition/sleep but does not cleanly separate it from developmental-risk — the one fit including both placed them at a comparable ≈ 0.28, with biology later refined to ≈ 0.10; the immuno-vs-developmental distinction therefore rests on the Φ symptom-decoupling, not the correlated-G.)* Clinically this means **two patients with the same overall impairment can carry very different cardiometabolic/inflammatory profiles** — severity is not a proxy for biological load. This property is what the downstream stages carry forward and re-test.

The map is robust as a measurement object. A prior-free refit reproduces loadings and Φ to three decimals (Tucker φ = 1.00); resampling (leave-one-cohort-out, diagnosis-balanced, site cluster-bootstrap) gives a minimum Tucker φ of 0.958; a BMI/weight/waist-excluded refit shows the immunometabolic axis survives on blood pressure / lipids / inflammation (Tucker φ ≈ 0.90), so the biology axis is not merely adiposity; and a variational (SVI) re-fit is congruent with NUTS on loadings and coordinates (the only gap being an ~21% attenuation of the Φ off-diagonals, a known mean-field bias — so NUTS remains the Φ authority). One honest caveat: the formal in-engine confirmation battery (WAIC model comparison, posterior-predictive checks, longitudinal/cross-cohort invariance) was reported on *sibling* fits (a staged continuous backbone at reduced N, and a 9-dim variant), not re-run on the exact 8-factor specification; the map's own confirmation rests on the flat-prior refit and the resampling robustness.

### Documentation and data

- Methods of record: [`docs/MEASUREMENT_MODEL.md`](docs/MEASUREMENT_MODEL.md); indicator selection & variable accounting: [`docs/DATA.md`](docs/DATA.md)
- Findings: [`docs/M1_FINDINGS.md`](docs/M1_FINDINGS.md); horseshoe/ESEM validation: [`docs/HORSESHOE_ESEM.md`](docs/HORSESHOE_ESEM.md)
- Engine: `src/face/models/bayesian/measurement_model_oop.py`; variational parallel: `src/face/models/variational/`
- Loadings / Φ: `reports/copula_8factor_{loadings,phi}.csv`; posterior + manifest: `results/face/oop_measurement/copula/weighted_8d/hs_s5_merged_xc/`

### Key results and takeaways

- **Method:** one global, missingness-aware Bayesian copula bifactor/ESEM fits an **8-factor transdiagnostic map** at full N = 9,013 (R-hat 1.03, 0 divergences) from observed cells only — 109 indicators, mixed likelihoods, no imputation.
- **Uniform reporting:** all eight factors are reported with primary loadings, direct G-loadings and Φ; G is a functional-burden axis (not a p-factor), depression/anxiety scales are windows onto G, and simple structure is *earned* (3 data-insisted cross-loadings out of 294 freed).
- **A periphery of three weakly-G axes** (immunometabolic 0.06, substance 0.07, developmental 0.07 — a near-tie), of which **immunometabolic is the only *earned* island**: weak on both G *and* the symptom axes (max Φ 0.076) on a freely-estimated basis (substance is orthogonal by construction; developmental couples to symptoms).
- **The biology property is robust** to estimator, prior, resampling and adiposity exclusion — and is the property the downstream stages re-test.

---

## M2 — Stratification: a continuum and an A = 5 archetype simplex

### What was done and why

The second method takes the fixed map and asks whether the transdiagnostic space carves into clinically useful *strata*. We deliberately frame M2 as a **coordinate system plus reading guide**, not a typology: patients are scored on all 8 axes from the copula posterior with measurement uncertainty propagated (diagnosis remains validation-only), and we first test what *shape* the cloud has before fitting any partition. Only then do we summarise it with interpretable lenses — extreme-phenotype archetypes and a soft tessellation — chosen so as not to over-claim discrete kinds. The methodological point is that the *summary* layer is held subordinate to the continuous coordinates, and its granularity is left to be decided by downstream validity rather than by internal parsimony.

### Results and interpretation

**The patient space is a continuum, not a set of natural kinds.** Before fitting any partition we run an uncertainty-aware structure-discovery gate on the 8-dimensional coordinate cloud, and it is decisive on every axis of evidence. The best silhouette over all candidate K is **0.140**, statistically indistinguishable from a structureless Gaussian null (**0.137 ± 0.002, z = 1.13, n.s.**) — "separation indistinguishable from a structureless continuum." **HDBSCAN returns 0 clusters** (100% of patients labelled noise); the **gap statistic optimum is K = 1**; and the **dip test is unimodal on every coordinate (p ≈ 0.99) except mania and suicidality** (p ≈ 0, the only two genuinely multimodal axes). The methodological consequence governs everything downstream: there are no reproducible, well-separated clusters to name, so K is a *granularity convention* rather than a count of disorders, and **the continuous coordinates — not any discrete label — are the primary load-bearing object.** This is exactly the conclusion an uncertainty-propagating pipeline is built to draw honestly: a naive clustering of the same cloud would have returned *some* partition and invited over-interpretation, but the null comparison shows that partition would have been an artefact.

**A = 5 archetypes: extreme phenotypes, not centroids.** Because the space is graded, we summarise it with **archetypal analysis** rather than clustering. Archetypal analysis places a small number of *corners* (extreme phenotypes) on the convex hull of the data and writes every patient as a soft convex mixture of them — so a patient is not "in cluster 3" but, say, 60% biology-corner + 25% severe + 15% well. The number of corners A is chosen by **cross-seed reproducibility** — the largest A whose corners reproduce across random restarts with Tucker congruence ≥ 0.8 — sweeping A = 2–8. There is a clean **stability cliff at A = 6** (cross-seed stability 0.98 → 0.44) while explained variance keeps rising smoothly, so **A = 5 is the reproducibility ceiling**, not a knee in fit. The five corners and their z-scored coordinate profiles (arm A, all 8 axes; positive = more of that axis, in SD units):

| archetype (size) | severity G | cognition | immunometab. | sleep | mania | suicid. | developm. | substance |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **A0** activation / sleep (16%) | −1.53 | −0.28 | +0.49 | **+2.79** | **+1.67** | −0.04 | −0.55 | −0.03 |
| **A1** severe / clean-biology (18%) | **+1.88** | +0.59 | **−1.93** | −0.45 | −0.70 | −1.25 | **−2.59** | +0.11 |
| **A2** immunometabolic corner (16%) | +2.36 | −0.01 | **+3.46** | −0.04 | −0.06 | +1.25 | +1.21 | −0.12 |
| **A3** trauma / suicidality (22%) | +0.09 | −0.57 | −2.07 | +0.84 | +0.72 | **+1.35** | **+2.83** | +0.11 |
| **A4** low-burden / well (28%) | **−2.03** | +0.38 | +0.28 | **−2.67** | −1.37 | −1.16 | −1.08 | −0.08 |

Membership is genuinely soft — even dominant-assigned patients carry only ~0.43–0.45 weight on their nearest corner, and ~74% sit in a "boundary" confidence tier — the honest representation of a continuum. The robustness battery passes 8/9 checks; the single conditional is itself informative: under *propagated measurement uncertainty* the extreme corners wobble (median Tucker 0.69), which is exactly why the continuous coordinates, not the hard corners, are treated as primary. A pipeline that ignored coordinate uncertainty would have reported spuriously sharp corners.

**The archetypes are diagnosis-blind: every corner spans all three cohorts.** Validated against the held-out diagnostic labels (never an input), the archetype assignment is essentially orthogonal to DSM-5 (adjusted Rand index **0.021**) and each corner is populated in BP, SZ and DR. The composition is genuinely transdiagnostic rather than a relabelling of diagnosis:

| cohort | dominant-corner composition (share) |
|---|---|
| **BP** | well 0.30 · trauma/suicidality 0.24 · activation/sleep 0.19 · severe-clean-biology 0.14 · immunometabolic 0.13 |
| **DR** | severe-clean-biology 0.31 · immunometabolic 0.30 · trauma/suicidality 0.22 · well 0.11 · activation/sleep 0.06 |
| **SZ** | well 0.29 · severe-clean-biology 0.25 · immunometabolic 0.19 · trauma/suicidality 0.18 · activation/sleep 0.10 |

Each diagnosis is a *mixture* of corners and each corner draws from every diagnosis — the activation/sleep corner is BP-enriched (mania is a bipolar feature) and the biology corners (A1/A2) are DR/SZ-enriched, but none is diagnosis-specific. This is the stratification analogue of the M1 result that diagnosis is metadata: the map's organising axes cut across the diagnostic categories.

**The archetypes describe the patient ≈ 9.7× more tightly than the diagnosis does.** We quantify "tightly" as the share of coordinate variance a grouping explains (η², averaged over the 8 axes — how much of *where a patient sits on the clinical/biological map* the grouping accounts for). The **DSM-5 grouping explains only 2.6%** (mean η² 0.026); the **A = 5 archetypes explain 25.6%** (mean η² 0.256) — a **9.7-fold** improvement, at a lower BIC than the DSM-5 grouping. Clinically this is a precise restatement of why categorical diagnosis is a weak organiser of clinical state: a patient's DSM-5 label tells you almost nothing about their position on the 8-dimensional map, whereas their archetype mixture tells you a great deal — and does so *across* diagnoses. Because this is a *descriptive* (variance-explained) comparison, it establishes only that the archetypes are a **better summary of who the patient is now**; whether that summary also *predicts the future* better is deferred to, and answered in, M4.

**A correction worth stating precisely: the archetypes separate on severity *and* profile.** It would be wrong to call the archetypes "severity-free." At the archetype level, **severity (G) is in fact the single strongest separating axis (η² 0.470)** — the simplex spans a clear severity gradient from the well pole (A4) to the severe poles (A1/A2). But it is not *only* severity: the archetypes separate almost as strongly on **immunometabolic load (η² 0.39), sleep (0.38) and developmental-risk (0.35)**. This is precisely the added value over a one-number severity score: *at any given severity*, the archetype distinguishes patients by their biological and symptom **profile** — A1 and A2 are equally severe yet opposite in biology (clean vs immunometabolic), and a severity score cannot tell them apart. (The sharper "driven by mania + suicidality ≫ G, not just severity" statement applies to the **K = 2 tessellation's first split** — η² ≈ 0.22 each vs G 0.05 — a different, coarser object than the A = 5 simplex; we keep the two distinct.) The biology property recurs here, *within* the severity gradient rather than instead of it.

**What the archetype lens adds over clustering.** Three things, each a deliberate modelling choice rather than a default. (i) *Honesty about the continuum* — soft convex mixtures do not impose boundaries the data do not contain, where a k-means or Gaussian-mixture partition would have manufactured them on a cloud we have shown has none. (ii) *Interpretability by extremes* — corners are *extreme, characterisable phenotypes* (the biology corner, the well pole, the trauma corner), which read more naturally to a clinician than the blurry "average patient" centroid a clustering returns; a patient is described by their proximity to clinically meaningful poles. (iii) *A reference frame, not a bin* — the five weights are a compact, transdiagnostic coordinate that preserves where a patient sits relative to *every* pole, so two patients who would land in the same cluster can still be distinguished by their mixtures. The hard tessellation (a soft mixture partition) is exported alongside as a nested K-family (2/3/4) with **no privileged K**; which K, if any, is operative is left to downstream incremental validity — answered in M4 (none).

### Documentation and data

- Methods: [`docs/STRATIFICATION_MODEL.md`](docs/STRATIFICATION_MODEL.md)
- Findings (incl. the copula A = 5 validation, *Result 4b*): [`docs/STRATA_FINDINGS.md`](docs/STRATA_FINDINGS.md); atlas: [`docs/STRATA_ATLAS.md`](docs/STRATA_ATLAS.md); archetype robustness: [`docs/ARCHETYPE_ROBUSTNESS.md`](docs/ARCHETYPE_ROBUSTNESS.md)
- Engine: `src/face/strata/strata_model_oop.py`; driver: `notebooks/run_strata_model_oop.py`
- Hand-off: `results/face/strata_oop/consolidate/{patient_strata.parquet (9,013×50), archetype_profiles.csv, k_family_menu.csv}`; copula A = 5 validation: `results/face/strata_oop/usefulness/a5_archetype_validation.{json,csv}` (driver `notebooks/compute_a5_archetype_validation.py`)

### Key results and takeaways

- **Method:** an uncertainty-aware structure test decides *shape* before any partition; **archetypal analysis** (soft convex mixtures of extreme phenotypes) summarises the continuum *without imposing boundaries* — chosen over clustering precisely because the data have none.
- **The space is a continuum, not biotypes** (silhouette 0.140 ≈ null 0.137 ± 0.002, z = 1.13; HDBSCAN 0 clusters; gap K = 1; only mania/suicidality multimodal) — so the continuous coordinates are primary and the labels are a reading lens.
- **A stable A = 5 simplex** (cross-seed reproducibility ceiling; clean cliff at A = 6): A0 activation/sleep · A1 severe-clean-biology · A2 immunometabolic · A3 trauma/suicidality · A4 well; membership is soft (~74% in a boundary tier).
- **Diagnosis-blind** — ARI vs DSM-5 0.021; every corner spans BP/SZ/DR, and each diagnosis is itself a mixture of corners.
- **≈ 9.7× tighter than DSM-5** (mean η² 0.256 vs 0.026, lower BIC): the archetype mixture describes a patient's clinical/biological state far better than the diagnostic label — descriptively (predictive validity is M4).
- **Severity + profile, not severity-free:** G is the strongest separating axis (η² 0.47) but immuno/sleep/developmental (0.39/0.38/0.35) separate almost as strongly — so at equal severity, A1 vs A2 differ in biology. The "mania + suicidality ≫ G" property is the **K = 2 tessellation's**, not the simplex's.
- **What archetypes add over clustering:** honesty about the continuum (no false boundaries), interpretability by extremes (clinically meaningful poles, not blurry centroids), and a reference frame (mixtures, not bins). No privileged K; tessellation K deferred to M4.

---

## M3 — Temporal coherence of the map and archetypes

### What was done and why

A dimensional map is only useful clinically if it *holds up over time*, and a clean way to test that is to **score follow-up under the frozen model** rather than re-fitting. M3 scores the two yearly follow-ups (V1, V2) by replaying the V0 standardisation and projecting each patient's observed follow-up cells onto the fixed loadings, factor correlations and archetypes, with measurement uncertainty propagated and nothing re-discovered. As a precondition that validates the scorer, re-scoring V0 under the frozen pipeline reproduces the original coordinates almost exactly (archetype-weight r = 0.9999; dominant-archetype agreement 99.5%), so any movement at V1/V2 reflects real change rather than scorer drift. The panel spans 16,241 visit-rows (V0 9,013 → V1 4,270 → V2 2,958).

### Results and interpretation

The map is **temporally coherent**, and we establish it stage by stage rather than asserting it.

**Measurement invariance — does each axis keep its meaning?** Comparing the within-visit loadings to V0 by Tucker congruence, all four continuous-backbone axes are invariant across visits — sleep 0.998, cognition 0.995, **immunometabolic 0.987**, severity (G) 0.991 — so a unit of "immunometabolic load" means the same thing at year 2 as at baseline, and a change is a change in the patient, not in the instrument. (The thin/explicit axes — mania, suicidality, developmental, substance — are not longitudinally licensed and are read descriptively.)

**Trait vs state — which axes are durable?** For each axis we decompose its variance over the panel into a **between-patient** component (stable individual differences), a **within-patient** component (real change) and a **measurement** component; the intraclass correlation ICC = between ÷ (between + within + measurement) is the share of the axis that is a durable patient *trait*. This is a direct pay-off of propagating measurement uncertainty end-to-end: only by isolating the measurement term can one separate genuine durability from a high test–retest correlation that is really noise.

| axis | ICC | reading |
|---|--:|---|
| **immunometabolic** | **0.91** | trait — the single most durable axis |
| mania / activation | 0.79 | *uninformative* — measurement-dominated (signal ratio 0.49), **not** durability |
| cognition | 0.70 | trait |
| severity (G) | 0.62 | trait-by-rank — patients hold rank while the population improves |
| substance | 0.49 | mixed |
| sleep | 0.47 | mixed |
| suicidality | 0.43 | mixed |
| developmental-risk | 0.39 | state (likely retrospective re-administration noise) |

The mania row is the cautionary case the decomposition is designed to catch: a naive test–retest reading would call ICC 0.79 "durable," but its between-patient variance is tiny relative to measurement, so the apparent stability is an artefact — flagged *uninformative*, not trait. This is the kind of error an uncertainty-aware pipeline avoids and a point-estimate one does not.

**Population trajectory — what moves?** Symptoms improve at the population level — suicidality slides −0.84 SD and severity −0.46 SD over two years — while **immunometabolic load stays essentially flat (+0.04 SD)**; geometrically, the severity "spine" reliably changes in ~35% of patients versus ~20% for the biology corner. So severity is durable *by rank* (who is more ill tends to stay more ill) even as everyone improves, and biology is durable in *level*.

**Archetype persistence — does identity hold?** The soft archetype-membership vector is durable (cosine 0.81 over followed patients) even though the hard argmax label churns (40% agreement, κ 0.19). This is not a failure but the signature of a continuum: central patients cross many argmax boundaries while their weight vector barely moves, so the durable object is again the *continuous* membership, not the discrete label. A deliberately reported null sharpens it — axis-level trait-ness does **not** predict archetype self-persistence (Spearman 0.071, p = 0.87) — confirming that durability lives in the continuous biology coordinate, not in any one corner. Finally, the same stage yields a **strata-independent, stabilised inverse-probability-of-retention weight** (bounded ~1.5–2.0; dropout is informative but severity-neutral) that lets M4 de-bias attrition without circularity.

The added value of the modelling is exactly this **separation of durable from moving signal**, on a map whose axes are first shown to keep their meaning. The **biology property recurs, now temporally** — of all eight axes the immunometabolic one is the most stable over two years — and the clinical reading is compact and actionable: **stratify on the durable biology; monitor the moving symptoms.**

### Documentation and data

- Methods: [`docs/TEMPORAL_MODEL.md`](docs/TEMPORAL_MODEL.md); findings: [`docs/TEMPORAL_FINDINGS.md`](docs/TEMPORAL_FINDINGS.md)
- Engine: `src/face/temporal/temporal_model_oop.py`; driver: `notebooks/run_temporal_model_oop.py`
- Hand-off: `results/face/temporal_oop/{trait_state/trait_state.csv, invariance/{congruence,license}.csv, persistence/{persistence.json,reliable_change.csv}, attrition/ipw_weights.parquet}`

### Key results and takeaways

- **Method:** follow-up is *scored under the frozen model* (V0 reproduced at r = 0.9999), so V1/V2 change is real change; a between/within/measurement **variance decomposition** yields per-axis durability (ICC) and catches measurement-dominated pseudo-stability (mania, ICC 0.79 but uninformative); attrition is handled with strata-independent IPW.
- **Map keeps its meaning over time** — all 4 backbone axes longitudinally invariant (immunometabolic φ 0.987).
- **The biology property recurs:** immunometabolic load is the **most durable axis** (ICC 0.91); cognition trait (0.70); severity trait-by-rank with population improvement; symptoms slide while biology holds.
- **Archetype identity persists in the soft weights** (cosine 0.81), not the churning hard label — durability is in the continuous coordinate, not the simplex corner.

---

## M4 — Prognosis: does a baseline coordinate predict 2-year functioning?

### What was done and why

The fourth method tests *incremental prognostic validity* — the right bar for a representation that claims clinical value. On the fixed M1/M2/M3 objects, does a baseline coordinate or archetype predict a 2-year outcome **beyond the hard reference of DSM-5 + baseline severity + the patient's own baseline outcome** (an ANCOVA autoregression, which dodges regression-to-the-mean)? The model is a measurement-error-aware (errors-in-variables) Bayesian GLM — each coordinate's M1 posterior SD is plugged in so the latent predictor is attenuation-corrected — and candidate encodings (the A = 5 archetypes, the continuous coordinates, the durable axis, each tessellation K) are added one at a time to the reference and judged by held-out expected log predictive density (ΔELPD-LOO), with the coefficient credible interval required to exclude 0. The two PI-locked outcomes are EGF (global functioning) and CGI-S (clinician severity) at the 2-year visit. A companion *representation benchmark* asks the complementary question — is the low-dimensional map a sufficient summary of the raw indicators? Provenance was audited: the prognosis frame was regenerated on the A = 5 strata and ingests the five archetype weights with their copula corner names (the engine's version-string date is a stale label, corrected to `prognosis_oop_2026_06_27_v2`).

### Results and interpretation

**The incremental-validity test: ΔELPD over a hard autoregression bar.** The reference model `R3y` already adjusts for diagnosis, baseline severity (CGI-S *and* an error-corrected G) and the patient's own baseline outcome (an ANCOVA autoregression); on functioning this reference is strong on its own (it improves held-out ELPD by ≈ +273 over a demographics-only baseline). The question is whether any baseline *representation* of the map adds predictive signal **on top of that bar**. Each encoding is added to `R3y` and scored by held-out ΔELPD-LOO (n = 2,114 for functioning, 2,345 for severity):

| encoding added to R3y | ΔELPD — functioning (EGF) | ΔELPD — severity (CGI-S) |
|---|--:|--:|
| **+ A = 5 archetypes (all axes)** | **+62.8** | +11.9 |
| + continuous coordinates (8 axes) | +38.0 | +13.3 |
| + A = 5 archetypes (⊥ G) | +33.5 | +14.0 |
| + tessellation K = 3 / 4 / 2 | +19.6 / +16.6 / +15.9 | +5.6 / +4.4 / +4.9 |
| + durable (immunometabolic) axis | +2.3 *(ambiguous)* | −0.7 *(ambiguous)* |
| reference `R3y` | 0 | 0 |

Two readings follow at once. First, **the A = 5 archetype encoding is the best predictor of 2-year functioning**, by a clear margin, and the continuous/archetype representations dominate every hard tessellation — which **answers M2's deferred question: operative K = none** (no discrete K adds anything the continuous map does not). Second, the *same* encoding gains almost nothing on **severity** (+11.9 vs +62.8): CGI-S is largely saturated by its own baseline, so the map predicts **functional trajectory, not severity level** — a genuinely different and harder target, and evidence that the prognostic signal is not just relabelled severity.

**The functioning signal is real, robust and complementary to diagnosis.** It survives attrition reweighting (IPW ΔELPD +54), **collapses under label permutation** (−2.4 — so it is not an artefact of model flexibility), and is **co-informative with DSM-5**: archetypes-plus-diagnosis (+62.6) beat diagnosis-alone (+29.0) and map-alone (+17.3), so the map adds ≈ +34 on top of the diagnostic label *and* the label still adds on top of the map. It is also **course-dependent and bipolar-led**: removing BP collapses the increment (+7, ambiguous) while removing SZ or DR does not (+62 / +56), and the within-cohort signal concentrates in the open-course bipolar cohort (likelihood-ratio p = 3 × 10⁻¹²) versus null in the more baseline-determined SZ/DR — the map adds prognostic information *where the future is genuinely open*.

**The prognostic atlas (the clinician-facing read).** Collapsing the continuous forecast to a 2-year functional-remission rate per archetype turns the result into an atlas, and the **biology property recurs prognostically** — the immunometabolic corner is the worst-prognosis pole, the well pole the best:

| archetype | remission BP | DR | SZ | pooled |
|---|--:|--:|--:|--:|
| **A4** well | 0.73 | 0.72 | 0.25 | **0.63 (best)** |
| A0 activation/sleep | 0.49 | 0.46 | 0.17 | 0.46 |
| A3 trauma/suicidality | 0.44 | 0.40 | 0.11 | 0.37 |
| A1 severe/clean-biology | 0.48 | 0.36 | 0.12 | 0.34 |
| **A2** immunometabolic | 0.27 | 0.31 | 0.09 | **0.22 (worst)** |

The 22% → 63% gradient is genuinely **within-diagnosis** — cohort composition explains only ≈ 4% of it, and the rank holds inside every cohort (BP 27% → 73%, SZ 9% → 25%, DR 31% → 72%), though SZ sits on a uniformly lower floor.

**Honest ceiling, and a sufficiency check.** At the *individual-binary* level the lift is small — adding the map to a clinical foundation model moves functional-remission AUC from 0.745 to 0.755 (+0.010) — so the value is **group-level stratification and continuous-trajectory forecasting, not an individual risk calculator**. A companion *representation benchmark* asks the complementary question — is the 8-dimensional map a *sufficient* summary of the raw indicators under a matched gradient-boosted model? For deterioration it is (raw − map AUC ≈ +0.005, a tie); for recovery it is near-sufficient (raw adds ≈ +0.04 AUC; the arm ladder runs REF 0.65 → + coordinates 0.70 → + archetypes 0.71 → + raw 0.75), with **92–97% of raw's predictive signal already captured within the modelled factors** — the small residual is item-level detail (depression/anxiety window items), not a missing axis. (These benchmark artifacts were regenerated on A = 5; the conclusions are unchanged.)

### Documentation and data

- Methods: [`docs/PROGNOSIS_MODEL.md`](docs/PROGNOSIS_MODEL.md); findings + atlas: [`docs/PROGNOSIS_FINDINGS.md`](docs/PROGNOSIS_FINDINGS.md); representation benchmark: [`docs/M4_REPRESENTATION_BENCHMARK.md`](docs/M4_REPRESENTATION_BENCHMARK.md)
- Engine: `src/face/prognosis/prognosis_model_oop.py` (+ `src/face/prognosis/repbench/`); drivers: `notebooks/run_prognosis_model_oop.py`, `notebooks/run_representation_benchmark.py`, `notebooks/run_repbench_recovery_shap.py`
- Hand-off: `results/face/prognosis_oop/{incremental/{incremental_comparison.csv,operative_k.json}, endpoints/archetype_atlas.csv, within_cohort/decomposition.json, clinical_value/clinical_value.csv}`; benchmark: `results/face/m4_repbench/`

### Key results and takeaways

- **Method:** incremental validity is tested with an errors-in-variables GLM and held-out ΔELPD *over a DSM-5 + severity + baseline-outcome autoregression bar*; a representation benchmark checks sufficiency vs raw indicators.
- **Archetypes predict 2-year functioning** (ΔELPD **+62.8**; IPW-robust, permutation-null, co-informative with DSM-5, BP-led) but **not severity** (CGI-S autoregression-saturated, +11.9).
- **Operative K = none** — the continuous/archetype encoding dominates every tessellation (resolves M2's deferred question).
- **The biology property recurs prognostically:** atlas **22% → 63%** functional remission, the **immunometabolic corner (A2) the worst-prognosis pole**, within-diagnosis (composition ~4%).
- **Honest ceiling** — small individual lift (AUC +0.010); the value is group-level + continuous forecasting. The map is a **sufficient representation** for deterioration, near-sufficient for recovery (92–97% within-factor compression).

---

## M5 — Treatment: does the map moderate response? (bounds-and-defends)

### What was done and why

The final method addresses the prescriptive question — does the map identify *who benefits from which treatment*? — under the constraint that the available data are **observational treatment-as-usual, not randomised arms**. The methodological response is an explicitly *bounds-and-defends* causal pipeline that states what observational data can and cannot support: an overlap gate first establishes which contrasts are even estimable; a propensity model adjusts for confounding-by-indication (conditioning on severity, diagnosis, demographics *and* the map); a doubly-robust errors-in-variables outcome model tests a treatment × durable-axis (and treatment × A = 5 archetype) moderation term; and an E-value quantifies how much unmeasured confounding would overturn each effect. (Treatment data were recovered late, from per-cohort `TRAITEMENTS` thesaurus tabs — a reminder to check raw/per-cohort sources before declaring a data boundary — and harmonised to common drug-class exposures: ATC codes for SZ, class-strings for DR, lifetime flags for BP.) M5 consumes the M4 A = 5 prognosis frame directly (provenance confirmed; version corrected to `treatment_oop_2026_06_27_v3`).

### Results and interpretation

**What observational data can and cannot identify — read off one table.** For each treatment contrast the pipeline reports, in order: whether the exposed and unexposed groups *overlap* enough to compare at all (max standardised mean difference, before → after weighting); the average treatment effect on functioning (ATE) with its E-value (how strong an unmeasured confounder, on the risk-ratio scale, would have to be to explain it away); and whether adding a treatment × map *interaction* improves held-out fit (moderation ΔELPD).

| contrast | overlap (max SMD →) | ATE (functioning) | E-value | moderation ΔELPD | verdict |
|---|--:|--:|--:|--:|---|
| **lithium → BP** | 0.30 → 0.01 | −0.03 *(CI incl. 0)* | 1.20–1.28 | −1.6 | well-identified **null** |
| **antipsychotic → BP** | 0.71 → 0.08 | −0.24 *(excl. 0)* | **1.80** | +3.4 (immuno axis) | confounded effect; **suggestive-only** moderation |
| **clozapine → SZ** | 0.33 → 0.07 | +0.03 *(CI incl. 0)* | 1.21 | −1.8 | **underpowered** (n ≈ 515) |

The reading is disciplined. Lithium in BP is the cleanest contrast (near-perfect overlap) and a **well-identified null** — its E-value of 1.20–1.28 means a *trivial* confounder would erase even the point estimate, and no axis interaction improves fit. Antipsychotic in BP shows a non-zero average effect, but its E-value of only 1.80 means a *modest* confounder — very plausible under treatment-by-indication — would overturn it, and the one suggestive moderation signal (a treat × immunometabolic interaction) is not held-out-confirmed. Clozapine in SZ is underpowered, with residual imbalance in the active-comparator contrast. With a minimum detectable interaction of ≈ 0.20 SD, the moderation arm simply lacks power to confirm realistic effects. **The map is prognostic and descriptive, not prescriptive — on these data, and we state so explicitly** rather than over-reading a borderline interaction.

**Defending M4: the prognostic signal is not an unmodelled treatment effect.** A natural objection to M4 is that the immunometabolic prognostic gradient could be a treatment artefact (patients with worse biology are medicated differently). The confounder analysis refutes it: adding drug-class exposure to the M4 outcome model barely moves the immunometabolic carrier, which keeps a credible interval excluding 0 throughout.

| carrier | coefficient (no-treat → +treat) | attenuation (unweighted / IPW) | survives? |
|---|--:|--:|:--:|
| immunometabolic **durable axis** | −0.049 → −0.046 | 6.4% / 4.1% | ✓ |
| A2 immunometabolic **archetype corner** | −0.21 → −0.19 | 7.7% / 6.4% | ✓ |

An attenuation of ≤ 8% means treatment explains almost none of the prognostic biology signal — so M4's claim stands on its own, and the map is not a covert treatment proxy.

**Describing course: who faces the hardest road.** Even where the map cannot prescribe, it stratifies *course*. Across three 2-year endpoints the immunometabolic corner (A2) is consistently the hardest and the well pole (A4) the easiest, and the archetype adds descriptively beyond severity, substance and demographics:

| endpoint | A2 immunometabolic | A4 well | archetype ΔAUC (perm. p) |
|---|--:|--:|---|
| treatment resistance | **44%** | 20% | +0.012 (p = 0.21, marginal) |
| side-effects | **25%** | 11% | +0.042 (p = 0.015) |
| response | 48% | 61% *(best)* | +0.034 (p = 0.010) |

Discrimination clears the permutation null for response and side-effects but is marginal for resistance. The closing position is deliberately bounded: **genuine treatment *selection* — who should get which drug — requires randomised or trial-arm data, a future M5b** (e.g. a prescription/RCT linkage within the FACE network).

### Documentation and data

- Methods: [`docs/TREATMENT_MODEL.md`](docs/TREATMENT_MODEL.md); findings: [`docs/TREATMENT_FINDINGS.md`](docs/TREATMENT_FINDINGS.md)
- Engine: `src/face/treatment/treatment_model_oop.py`; driver: `notebooks/run_treatment_model_oop.py`
- Hand-off: `results/face/treatment_oop/{propensity/propensity_summary.csv, moderation/moderation.csv, confounder/confounder.csv, atlas/{atlas_gates.csv,treatment_course_atlas.csv}}`

### Key results and takeaways

- **Method:** a bounds-and-defends causal pipeline (overlap → propensity → doubly-robust EIV moderation → E-value) that states what observational TAU can and cannot support.
- **Ceiling** — the map does **not reliably moderate** treatment: lithium-BP a well-identified null (E 1.20–1.28), antipsychotic-BP a confounded average effect (E 1.80) with suggestive-only moderation, clozapine-SZ underpowered.
- **Defends M4** — the immunometabolic prognostic carrier **survives treatment adjustment** (durable axis 6.4%/4.1% IPW; A2 corner 7.7%/6.4% IPW); not a treatment proxy.
- **Describes course** — the immunometabolic corner faces the hardest 2-year course (resistance 44%, side-effects 25% vs 20%/11% at the well pole).
- **Prescription needs randomisation** — a true treatment-selection claim awaits an M5b on trial-arm data.

---

## Synthesis and limitations

The contribution is a **method**: a five-stage, no-imputation, uncertainty-propagating pipeline that builds a transdiagnostic dimensional map from harmonised multi-cohort data and then validates it under a single discipline — the discovered map is *held fixed* at every downstream stage, diagnosis is metadata throughout, and measurement is reported separately from interpretation. Each stage contributes a reusable component: a missingness-aware copula bifactor/ESEM (M1), an uncertainty-aware structure-test-then-archetype layer (M2), frozen-model temporal scoring with a trait/state decomposition (M3), an incremental-validity test against an autoregression bar plus a representation benchmark (M4), and a bounds-and-defends causal design for observational treatment data (M5).

Run end-to-end on the FACE data, the pipeline also delivers one coherent substantive demonstration: **a single immunometabolic axis behaves as a distinct, durable, prognostic biological signal at every layer** — weakly tied to the rest of the clinical picture (M1), anchoring its own archetype corner (M2), the most stable dimension over two years (M3), the worst-prognosis pole for functioning (M4), and robust to treatment adjustment (M5). The recurring lesson is that **severity is not a proxy for biology**, and a map that separates the two surfaces a signal that severity-staging and DSM-5 categories miss. We are careful, though, to present this as a *property the method reveals*, not the premise it was built on.

The limitations are stated plainly and consistently. All findings are **internal-validity** results on observational baseline-plus-follow-up data: there is no external replication cohort, the horizon is two years with ~33% reaching V2, individual-level predictive lift is modest, and — decisively for the prescriptive question — there are no randomised treatment arms, so M5 is bounded rather than confirmatory. On the methods side, the M1 confirmation battery (WAIC/PPC/invariance) was reported on sibling fits rather than the exact 8-factor specification, and the archetype simplex is a soft reading lens over a continuum whose hard labels are intrinsically unstable. None of these undercut the methodological contribution, but each marks where confirmatory work — an external cohort, longer follow-up, and trial-arm data for M5b — would convert these reconciled internal findings into deployable claims.

*Full vertical synthesis of record: [`docs/VERTICAL_FINDINGS.md`](docs/VERTICAL_FINDINGS.md). Project map and current state: [`CLAUDE.md`](CLAUDE.md), [`docs/STATE.md`](docs/STATE.md).*
