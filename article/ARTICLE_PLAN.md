# FACE-ATLAS — Plan for a top-tier journal article

> **Purpose.** A working plan to turn the FACE-ATLAS program (Milestones 1–5) into **one comprehensive,
> discovery-framed research article** held to top-tier standards (Nature-family / Lancet Psychiatry class).
> The article itself will be written in LaTeX; this document is the blueprint that gets us there.
>
> **Decisions locked with the PI (2026-06-17):**
> 1. **Scope** — *one comprehensive paper* spanning the full arc (the map exists → organizes → persists →
>    predicts → treatment boundary).
> 2. **Journal** — *no fixed venue yet*; write to the **highest academic standard** and keep the structure
>    journal-agnostic. A recommended target + ranking is in §1.3, but nothing in the plan depends on it.
> 3. **Framing** — *Discovery: the map.* The lead message is the transdiagnostic clinical–biological
>    dimensional map: biology rides on axes orthogonal to functional-burden severity, and the space is a
>    **continuum, not biotypes**. M3–M5 are framed as "how far the map can be pushed," ending at an
>    honestly-drawn boundary.
>
> **Follow-up decisions (2026-06-17, resolved):**
> - **Table 1** — approved to compute aggregate-only sample characteristics from the confidential data.
>   **DONE:** `scripts/60_table1.py` → `reports/table1_characteristics.{csv,md}` +
>   `article/tables/table1_characteristics.md` (no per-patient values leave the machine).
> - **Treatment (M5)** — the moderation forest moves to **Extended Data**; the main text ends on prognosis
>   + the treatment boundary stated in prose (keeps the discovery in focus).
> - **Venue** — proceed **journal-agnostic** to the highest standard (recommendation ranking in §1.3 stands).
>
> **How to read this.** §1 fixes strategy; §2–§4 give title/abstract/section-by-section content (every claim
> tagged with its source report so drafting is copy-traceable); §5 is the display-items plan; **§6 is the
> figure/table gap report you asked for**; §7–§10 cover references, the claims ledger, the reviewer
> pre-mortem, and boilerplate; §11 is the writing workflow + LaTeX layout; §12 is open questions for you.

---

## 1. Strategy

### 1.1 The one-sentence thesis

> Across bipolar disorder, schizophrenia and major depression (FACE, *N* = 9,013), a single
> missingness-aware Bayesian bifactor/ESEM model places clinical **and** metabolic/inflammatory measurements
> in one transdiagnostic coordinate system, and shows that **biological load is the domain least entangled
> with overall functional severity**, that the patient space is a **graded continuum rather than discrete
> biotypes**, and that this geometry **persists over two years** and carries a **real-but-modest,
> group-level prognostic signal for functioning** — while honest causal analysis finds **no reliable
> treatment moderation** in observational data.

The paper's distinctive virtue is **calibration**: it reports a genuine discovery *and* its limits in the
same breath. That is the antidote to the biotype/biomarker over-claiming the field is now correcting for,
and it is a feature to foreground, not hide.

### 1.2 The five-beat arc (one paper, five Results blocks)

| Beat | Claim | Milestone | Evidential status |
|---|---|---|---|
| **Exists** | A certified 9-dimension transdiagnostic map; **biology ⟂ G** | M1 | strong, certified |
| **Organizes** | A **continuum, not biotypes**; 8 archetypes; biology corners survive | M2 | strong (internal/descriptive) |
| **Persists** | Temporally coherent: durable biology, moving symptoms (trait/state) | M3 | moderate–strong |
| **Predicts** | Modest, group-level, functioning-only, course-dependent | M4 | modest, honest |
| **(Doesn't yet) guide treatment** | No reliable moderation on observational TAU; boundary *earned* | M5 | null/boundary |

The arc is a **funnel**: each beat is a stronger test than the last, and the paper's honesty is that it
reports where the signal thins. The title and abstract live in beats 1–2; beats 3–5 are the stress test.

### 1.3 Target journal — recommendation (journal-agnostic plan)

We will draft to a journal-agnostic **IMRaD + Methods + Extended Data** template (§4) that converts cleanly
to any of the targets. Recommended order, with the reasoning:

1. **Nature Mental Health / Molecular Psychiatry** *(recommended primary).* Both reward a
   methods-and-biology discovery with a dimensional/normative-modeling lineage and are comfortable with a
   large computational model and a calibrated, partly-null result. Best fit for the "Discovery: the map"
   framing. Format: ~4,000–5,000 words, 4–6 main display items, Extended Data.
2. **Lancet Psychiatry.** Highest clinical reach; receptive to dimensional reframing of diagnosis and to
   honestly-reported modest/null findings. Would need the clinical translation (M4/M5) and the
   continuum-vs-DSM message brought slightly forward. Format: ~4,500 words, structured abstract, panels.
3. **Biological Psychiatry.** Strongest if we foreground "metabolic/inflammatory markers in the
   transdiagnostic factor space" and the Bayesian engine. ~4,000 words.
4. **JAMA Psychiatry / AJP.** Clinical flagships; would foreground stratification/prognosis — the hardest
   sell given the deliberately modest M4 increment. Keep as fallback, not lead.

**Working budget (assume primary):** main text ≈ 4,500 words; **6 main figures, 1–2 main tables**; Methods
≈ 3,000 words (often uncounted/online); Extended Data ≈ 8–12 items; Supplementary unlimited. We design to
this and trim per the venue we ultimately pick.

### 1.4 Novelty statement (defensible, from the literature dossier §6)

No prior work does all of the following **together**, which is the novelty we claim:

- Prior transdiagnostic structure work is **symptom-only** (Caspi/p-factor; Kotov/HiTOP) or
  **neuroimaging-only** (Wolfers, Marquand, B-SNIP). FACE puts **metabolic and inflammatory laboratory
  markers directly into the transdiagnostic factor space** alongside cognition, sleep, suicidality, mania
  and substance, across **three diagnostic cohorts (BP/SZ/MDD) at once**.
- It **quantifies the entanglement** between each biological axis and a general functional-burden factor
  (the ≈ 0.06–0.14 correlations) — the immuno-metabolic literature asserts a *separable* dimension but does
  not report its correlation with a bifactor general factor.
- It does so with a **full-sample, no-imputation, mixed-likelihood Bayesian bifactor/ESEM** — a combination
  not seen in the retrieved transdiagnostic-biology papers, most of which subsample or use single-modality
  data.

One-line framing of the contribution: **the first quantitative measurement of how independent
metabolic/inflammatory load is from a transdiagnostic functional-burden axis**, with uncertainty, across
three SMI cohorts — converting a qualitative, independently-supported pattern into a measured coordinate.

---

## 2. Title and abstract

### 2.1 Title options (discovery-first)

1. *(lead)* **"A transdiagnostic clinical–biological map of severe mental illness: metabolic and
   inflammatory load are largely independent of overall severity"**
2. **"Biology rides on axes that clinical severity does not see: a transdiagnostic dimensional map across
   bipolar disorder, schizophrenia and depression"**
3. **"A continuum, not biotypes: a missingness-aware Bayesian map of clinical and biological variation in
   9,013 patients with severe mental illness"**
4. *(clinical-venue variant)* **"Transdiagnostic dimensions of severe mental illness: a continuum that
   persists, predicts functioning modestly, and does not yet guide treatment"**

Decision rule: titles 1/3 if primary is Nature Mental Health / Mol Psychiatry; title 4 if Lancet Psychiatry.
Avoid the word "independent" unqualified (see §9.1).

### 2.2 Abstract (draft to refine — ~200–250 words, convertible to structured)

> Psychiatric diagnosis aggregates biologically and prognostically heterogeneous patients into single
> categories. We reorganized the harmonized three-cohort FACE baseline (bipolar disorder, schizophrenia,
> major depression; *N* = 9,013) around continuous, diagnosis-agnostic axes of variation, estimated by one
> global, missingness-aware **Bayesian sparse bifactor/ESEM** model with mixed likelihoods, fit to each
> patient's observed cells only (no imputation, full sample), with diagnosis held out as metadata. From ten
> candidate constructs the data certified a **nine-dimension map** — a general functional-burden factor (G)
> plus eight weakly-correlated axes (cognition, metabolic, inflammatory, sleep, developmental-risk,
> suicidality, mania, substance). The central finding: **metabolic and inflammatory load are the least
> severity-entangled domains** (correlation with G ≈ 0.06 and 0.14, vs 0.39 cognition, 0.44 sleep). On these
> coordinates the patient space is a **graded continuum, not discrete biotypes** (five structure-discovery
> methods agree; adjusted Rand index ≈ 0 vs DSM-5 subtypes), describable by eight extreme phenotypes whose
> two-year functional-remission rates range 14%–60%. Scored forward onto follow-up the geometry **persists**
> (durable biology, moving symptoms). It **predicts** future functioning incrementally beyond diagnosis and
> severity, but **modestly and at the group level** (remission AUC +0.017; course-dependent). On
> observational treatment-as-usual the map does **not** reliably moderate response. FACE-ATLAS delivers a
> real, stable, transdiagnostic clinical–biological continuum and a calibrated account of its prognostic
> reach.

A **structured** version (Background / Methods / Findings / Interpretation / Funding) is a 30-minute
rewrite if we go to Lancet Psychiatry; keep both in the LaTeX as commented alternates.

---

## 3. Cohort, scope, and what the article will and will not claim

- **Sample.** FACE cohorts (Fondation FondaMental), V0 baseline *N* = 9,013 (BP 6,252 · SZ 2,209 · DR 552);
  21 recruitment sites; 143 modeled indicators (88 continuous, 55 explicit); mean cell missingness 39.8%,
  preserved, never imputed (`reports/01_build_data.md`). Longitudinal follow-up V1 47.4% (4,270), V2 32.8%
  (2,958) (`reports/30_retention.csv`, `31_attrition.md`).
- **Internal validity only.** V0 discovery; follow-up validates; **no external cohort, no incident-event
  outcomes, no randomized treatment**. This is stated up front, in the abstract's last sentence and a
  dedicated Limitations paragraph — it is the single most important honesty move (see §9.5).
- **Confidentiality.** The FACE database is confidential (FondaMental); only aggregate parameters, derived
  structure and code description appear. This constrains Table 1 (see §6) and the data-availability
  statement (§10).

---

## 4. Manuscript structure — section by section

> Convention below: each block lists **(a)** what it must say, **(b)** the headline numbers with their
> source report, and **(c)** the display item it points to. Drafting = expand (a), drop in (b), cite the
> figure in (c). Discovery-framed venues lead with Results; clinical venues use IMRaD — the content blocks
> are identical, only the order of Methods vs Results flips.

### 4.1 Introduction (~600–750 words)

(a) Four moves, no more:
1. **The problem.** DSM categories aggregate heterogeneous patients; comorbidity, instability and
   within-category variance limit biological and prognostic traction. Cite RDoC (Insel 2010), HiTOP (Kotov
   2017), the p-factor read as impairment (Caspi 2014).
2. **The gap.** Transdiagnostic structure has been mapped from **symptoms** or **neuroimaging**, but
   **routine metabolic/inflammatory biology has not been put into the transdiagnostic factor space** across
   BP/SZ/MDD at once, with uncertainty and without imputation.
3. **The live debate we enter.** Continuum (normative-modeling: Wolfers 2018, Marquand 2016) vs **biotypes**
   (B-SNIP: Clementz 2016) — name it as unsettled; we contribute a result about *this* clinical–biological
   feature space, not a universal verdict.
4. **What we did, in one paragraph**, ending on the five-beat arc and the calibrated bottom line.

(b) No results in the intro beyond the one-sentence thesis. (c) → Fig 1 (study overview).

### 4.2 Results

> Six results subsections, mapping to the arc. Lead with the map; let each subsequent beat be visibly a
> harder test.

**R1 — A certified nine-dimension transdiagnostic map (M1).**
(a) One global model on observed cells; ten candidates in, nine dimensions out; depression/anxiety enter as
**cross-loading windows** (load 0.66–0.80 on G), anhedonia **rejected**, impulsivity/negative-symptoms/
sensory **not_testable** — *and stating so is a result.* G is **functional burden, not a symptom p-factor**
(anchored by FAST 0.90, EGF 0.73; no symptom content). The candidate "biology" **splits** into metabolic +
inflammatory (Φ ≈ 0.19). (b) `docs/M1_FINDINGS.md` §2–§3, `ADJUDICATION.md`, `reports/11_s5_9dim_*`. (c) →
**Fig 2** (prior→posterior atlas + Φ).

**R2 — Biology is the least severity-entangled domain (M1, the headline).**
(a) Under the correlated-G sensitivity model, G correlates **+0.06 inflammatory, +0.14 metabolic** vs
**+0.39 cognition, +0.44 sleep**: metabolic/immune load is carried on axes overall severity does not see.
Word it as **"largely independent of a general functional-burden axis"** with boundary conditions (§9.1).
(b) `M1_FINDINGS.md` F2, `reports/07_corrG_report.md`, `04_stage5_corrG_phi.csv`. (c) → **Fig 3**.

**R3 — The space is a continuum, not biotypes (M2).**
(a) A pre-registered five-method structure gate (gap-stat K=1; HDBSCAN 0 clusters; unimodal PC1, dip
p≈0.99; smooth archetype scree; flat mixture-BIC basin) → **continuum**. Represented by **8 stable extreme
phenotypes** (cross-seed congruence 0.999) + a 4-region tessellation; **75% of patients are blends**.
**Transdiagnostic** (ARI ≈ 0 vs cohort and vs 7 DSM-5 subtypes), **driven by specific/biological axes not G**
(max specific η² 0.45 > G 0.31), **not a missingness artefact** (coverage→membership classifier −0.076
lift), and a **tighter description than DSM-5** (mixture BIC 199,325 vs 206,016; coordinate variance
explained ~21% vs ~5%). Distinct **metabolic and inflammatory corners** = biology⟂G survives into phenotypes.
(b) `STRATA_FINDINGS.md`, `reports/21_structure.md`, `22_tessellation.md`, `23_archetypes.md`,
`24_validation.md`. (c) → **Fig 4**.

**R4 — The geometry persists over two years (M3).**
(a) Scored forward onto V1/V2 on the **fixed** model (observed cells, uncertainty propagated, never
re-discovered): measurement invariance holds (5/6 backbone axes φ 0.96–1.00; inflammatory partial 0.90).
**Trait/state split:** metabolic ICC 0.93, inflammatory 0.85, cognition 0.78 (durable) vs sleep 0.49,
suicidality 0.46, developmental 0.39 (moving); population *slides* down severity/symptoms while individual
biological rank is *held*. **Spine moves 2.2× more than the biology corner**; archetype identity persists
52% (κ 0.27 vs 12.5% chance). Honest caveat: developmental "state" is CTQ recall-noise, not change.
(b) `TEMPORAL_FINDINGS.md`, `reports/33_invariance_report.md`, `35_variance_report.md`,
`36_persistence_report.md`. (c) → **Fig 5**.

**R5 — It predicts future functioning, modestly and at the group level (M4).**
(a) Errors-in-variables Bayesian GLM, incremental **beyond diagnosis + severity + baseline outcome**.
**Functioning yes, severity no:** archetypes add held-out ΔELPD +46; durable **metabolic β −0.062
[−0.103, −0.022]**, **inflammatory β −0.060 [−0.112, −0.011]**; metabolic **survives error-corrected G
severity**. **Co-informative with DSM-5** (each adds beyond the other) and **course-dependent** (large in
BP/DR, null in baseline-saturated SZ). Archetype prognostic atlas: 2-year functional remission **14%→60%**
(partly baseline-severity). Individual discrimination small (**remission ΔAUC +0.017 [+0.009, +0.026]**) —
value is **group-level stratification + continuous forecasting**, not an individual risk calculator. Honest
de-scope: **scale trajectories, not recorded relapse/hospitalization events**. (b) `PROGNOSIS_FINDINGS.md`,
`reports/43_incremental.md`, `44_transdiagnostic.md`, `45_endpoints.md`, `46_clinical_value.md`,
`47_robustness.md`. (c) → **Fig 6a–b**.

**R6 — It does not yet guide treatment — a boundary drawn by the evidence (M5).**
(a) Treatment data found in per-cohort `TRAITEMENTS` thesauri, harmonized to drug-class exposures; causal
pipeline (overlap gate → propensity → doubly-robust EIV moderation + E-value). **Lithium-in-BP** (100%
overlap) = **well-identified null**; **antipsychotic-BP** = suggestive-unconfirmed metabolic/inflammatory ×
functioning (ATE E-value 1.79); **clozapine-SZ** = channeled, non-estimable. ATEs confounding-fragile
(E 1.1–1.8). **M5 strengthens M4:** metabolic→functioning survives drug-class adjustment (β −0.051→−0.048,
4.4% attenuation). Boundary **earned, not assumed**; treatment *selection* needs randomized/trial-arm data
(a future M5b). (b) `TREATMENT_FINDINGS.md`, `reports/55_propensity.md`, `56_moderation.md`,
`57_confounder.md`, `59_m5b_feasibility.md`. (c) → **Extended Data** (moderation forest); main text carries
this beat in **prose only** (decided 2026-06-17 — keeps the discovery in focus).

### 4.3 Discussion (~900–1,100 words)

(a) Five paragraphs:
1. **What the map is and why biology⟂G matters** — biological risk is decoupled from a clinician's global
   impression, which is exactly what a biology-aware stratification can exploit.
2. **Continuum vs biotypes, honestly** — position against B-SNIP: different measurement spaces; B-SNIP
   itself found DSM diagnoses to be a severity continuum (Clementz 2016/2024). We do not claim "no biotypes
   exist"; we claim none in *this* 9-D clinical–biological space.
3. **Convergent external validation** — the immuno-metabolic-depression program (Penninx 2024; Milaneschi
   2020; Lamers 2013/2020) reached a separable biological dimension by entirely different methods → strong
   convergent validity; metabolic burden as treatment/adiposity/chronicity-driven (Vancampfort 2015/16;
   Pillinger 2017; Perry 2019).
4. **The calibrated reach** — persists, predicts functioning modestly/group-level, no treatment moderation;
   why a well-identified null (lithium-BP) is a result, and a corrective to over-claiming.
5. **Clinical reading** — *stratify on the durable biology, monitor the moving symptoms*; group-level
   prognostic enrichment + a hypothesis for prospective metabolic/inflammatory × antipsychotic testing.

(b) Each paragraph pins one tension from `LITERATURE_EVIDENCE.md` and answers it. (c) no new figure.

### 4.4 Limitations (own subsection — do not bury)

Internal validity only; no external cohort; **scale trajectories not incident events** (M4 de-scope);
observational treatment (confounding by indication, E 1.1–1.8); documented invariance partials (G in SZ;
inflammatory in DR; mania-Altman in DR); mania a 2-indicator axis, substance a 2-cohort axis; the
`isf09a` item-level PPC misfit (factor unaffected); informative dropout handled by IPW; three visits / 2-year
window. Sources: each milestone's "Honesty and limits" section.

### 4.5 Methods (~3,000 words, online)

Subsections, each traceable to a methods-of-record doc:
1. **Cohorts & harmonization** — FACE BP/SZ/DR; dictionary; per-variable sanity bounds; deterministic
   skip-logic structural-zero decoding; no imputation (`DATA.md`).
2. **Candidate ontology & soft priors** — 10 candidates; soft-prior loading roles (expected
   N₊(0.70,0.25)…) (`MEASUREMENT_MODEL.md` §2–§3.3).
3. **Model** — bifactor (primary) + correlated-G (sensitivity); mixed-likelihood observation model
   (Gaussian/Student-t, log-t, ordered-logit, Bernoulli, NegBin); item-level covariates (`MM` §3.1–3.3).
4. **Observed-likelihood / missing data** — observed-cell sum; structural zeros; optional MNAR arm
   (`MM` §3.4).
5. **Gaussian-block marginalization = FIML** — Woodbury/matrix-determinant lemma; why this makes full-N
   tractable on a workstation; identity with FIML (`MM` §3.5, §4.5).
6. **Staged continuation estimation** S1→S5; interpret only the global fit; acceptance gates (R-hat, ESS,
   divergences, BFMI) (`MM` §4).
7. **In-engine confirmation** — flat-prior refit (Tucker φ=1.00), PPC/Bayesian SRMR ≈ 0.07, WAIC model
   comparison; why standalone FIML is redundant (`MM` §5).
8. **Invariance & robustness** — multi-group loadings, Bayesian DIF; LOCO, diagnosis-balanced subsampling,
   site cluster-bootstrap, 1/n-cohort weighting (min φ 0.958) (`reports/06,08,13`).
9. **Per-patient scoring** — posterior mean/SD/HDI + reliability tiers (`MM` §7).
10. **Stratification** — structure gate; measurement-error Bayesian mixture (Extreme Deconvolution with
    per-patient S_i); archetypal analysis; validation gates (`STRATIFICATION_MODEL.md`).
11. **Temporal** — forward scoring; longitudinal invariance; measurement-error random-intercept trait/state
    with plugged M1 variance + visit fixed effects; reliable-change geometry; IPW for attrition
    (`TEMPORAL_MODEL.md`).
12. **Prognosis** — errors-in-variables Bayesian GLM; nested model comparison (D/+DSM-5/+map/+both); ELPD,
    AUC, decision curves; permutation/reliability/IPW robustness (`PROGNOSIS_MODEL.md`).
13. **Treatment** — exposure harmonization; overlap gate; stabilized IPTW propensity; doubly-robust EIV
    moderation; E-values (`TREATMENT_MODEL.md`).
14. **Compute & reproducibility** — PyMC/NumPyro-JAX, M4 Pro/24 GB, fixed seeds, config-first, tests;
    `scripts/01–57`.

---

## 5. Display items plan (main + Extended Data)

> Target: **6 main figures + 1–2 main tables**; everything else → Extended Data / Supplementary. Mapping to
> existing assets in `report/figures/` and `docs/figures/`. "Status" = ready / rework / **new**.

### 5.1 Main figures

| # | Working title | Panels / content | Source asset(s) | Status |
|---|---|---|---|---|
| **Fig 1** | Study overview & the five questions | Cohort/coverage + no-imputation pipeline + the exists→…→treatment arc as one schematic | TikZ `\discoveryflow`/`\programstrip` in report; SVGs `m3_*`; `20_coverage.png` | **NEW** (compose one clean publication panel) |
| **Fig 2** | The certified nine-dimension map | (a) prior→posterior loading atlas; (b) Φ heatmap | `fig_empirical_atlas.png`, `fig_prior_posterior.png`, `fig_phi_heatmap.png` | rework (merge → multi-panel, 300 dpi, panel labels) |
| **Fig 3** | Biology is least severity-entangled | bar of axis–G correlation + bifactor \|λ_G\| | `fig_biology_g.png` | ready (minor: vector export, font) |
| **Fig 4** | A continuum, not biotypes | (a) structure-gate verdict; (b) UMAP colored by cohort/severity/inflammatory; (c) 8 archetype profiles | `21_selection.png`, `m2_embedding.png`, `m2_profiles.png` | rework (**embedding is low-res — regenerate**; strip "Arm A" jargon) |
| **Fig 5** | The geometry persists (trait/state) | (a) trait/state ICC + population slide; (b) spine-slides-while-corners-hold | `35_trait_state.png`, `36_spine_corner.png`, SVGs `m3_two_lens`, `m3_spine_slides` | rework (turn schematic SVGs into final, unify style) |
| **Fig 6** | Prognostic reach (functioning) | (a) archetype prognostic atlas 14%→60%; (b) incremental value + durable-axis forest | `m4_atlas.png`, `43_added_value.png`/`m4_value.png` | rework (2-panel; treatment moderation moved to Extended Data per 2026-06-17 decision) |

### 5.2 Main tables

| # | Title | Content | Status |
|---|---|---|---|
| **Table 1** | Sample characteristics by cohort | N, age, sex, education, DSM-5 subtype distribution, V1/V2 retention — aggregate only | **DONE** — `scripts/60_table1.py` → `reports/table1_characteristics.{csv,md}`, `article/tables/` (English subtype labels) |
| **Table 2** | The nine dimensions: indicators, loadings, adjudication, invariance | One row per dimension; anchor indicators, mean \|λ\|, G-correlation, verdict, invariance note | ready (assemble from `ADJUDICATION.md` + `11_s5_9dim_loadings.csv`) |

### 5.3 Extended Data (8–12 items, candidates)

Prior atlas (theory); WAIC model comparison (`fig_waic.png`); dual-block PPC + SRMR (`fig_ppc.png`,
`mixed_ppc.png`); reliability tiers (`fig_reliability.png`); cross-cohort invariance (`fig_invariance.png`,
`33_congruence.png`); robustness (LOCO/site/weighting, `47_robustness.png`); tessellation/membership
(`22_*`, `23b_compare.png`); attrition/IPW & longitudinal CONSORT flow (`31_attrition.png`, **new flow**);
prognosis calibration/decision curves (`46_*`); **treatment moderation forest (`m5_moderation.png`) — the
M5 beat lives here**; treatment overlap/propensity (`55_overlap.png`); the soft-prior-shrinkage mechanism
(`fig_soft_priors.png`).

---

## 6. Figure & table gap report (you asked me to flag these)

**Verdict:** the figure set is unusually complete and already publication-quality in *style* (clean
matplotlib, sensible color use). The gaps are about **print specification, composition, jargon, and two
genuinely missing items** — not about missing science. Ordered by severity.

### 6.1 Missing — must create

1. ~~**Table 1 (sample characteristics by cohort).**~~ **DONE (2026-06-17).** `scripts/60_table1.py` reads
   the confidential demographics and writes **aggregate-only** outputs (`reports/table1_characteristics.csv`
   + `.md`, copy in `article/tables/`): N, age (mean/SD + median[IQR]), % female, education, recruitment
   sites, demographic missingness, V1/V2 retention, and an English-labelled DSM-5 subtype panel. No
   per-patient value leaves the machine. *Note for the manuscript: education is 40.8% missing overall
   (74.5% in SZ) — a real coverage feature, consistent with the observed-likelihood/no-imputation design;
   worth a one-line footnote.*
2. **Figure 1 (study-overview / pipeline schematic) as one polished panel.** The pieces exist (report TikZ
   diagrams, M3 SVGs, coverage plot) but not as a single 300-dpi/vector main figure. **Action:** compose in
   a vector tool or matplotlib; keep the "no-imputation, diagnosis-as-metadata, V0-defines" invariants
   visible.
3. **Longitudinal CONSORT-style retention flow** (V0 9,013 → V1 4,270 → V2 2,958, with dropout reasons and
   the IPW note). Currently only `31_attrition.png` (odds ratios). **Action:** new flow diagram → Extended
   Data. Strengthens the M3/M4 attrition story reviewers will probe.

### 6.2 Rework — exists but not print-ready

4. **`m2_embedding.png` is low-resolution** (screen thumbnail). Must be **regenerated at ≥300 dpi** (or
   vector) for Fig 4. Highest-priority rework.
5. **Panel jargon leaks project-internal language.** "Arm A / Arm B", "tessellation", "S5", "Φ", axis name
   `mania_activation` need plain-language equivalents or a one-line legend gloss in any main figure
   (`m2_profiles.png`, M3 SVGs). 
6. **Schematic SVGs (`m3_*`) are explainer-grade**, not final figures. Decide per panel: promote to a
   polished vector figure, or keep as Extended Data conceptual aids.
7. **Uniform figure system.** Adopt one palette (colorblind-safe; verify the blue/teal vs red in
   `m5_moderation.png` and the diverging maps pass deuteranopia), one font, consistent panel labels (a, b,
   c), and **export every main figure as vector PDF/EPS + 300-dpi TIFF**. Current assets are PNG at screen
   resolution. `report/make_figures.py` is the right place to centralize this.

### 6.3 Decisions, not gaps

8. ~~**Fig 6 is dense (3 panels spanning M4+M5).**~~ **DECIDED (2026-06-17):** Fig 6 = prognosis only (a,b);
   the treatment moderation forest (`m5_moderation.png`) → **Extended Data**, with the M5 beat carried in
   main-text prose. Keeps the main narrative on the discovery; treatment is the boundary coda.
9. **A causal DAG for M5** (treatment) — now an Extended Data companion to the moderation forest (recommended,
   since M5 is Extended-Data-only): a small DAG makes the confounding-by-indication argument legible.

**Bottom line for you:** only **#1 (Table 1 aggregates)** needs new data work and your go-ahead on computing
confidential aggregates; **#2–#3** are new figure composition; **#4–#7** are reprocessing existing assets to
print spec. None require re-running the models.

---

## 7. References & citation strategy

- A verified, ready-to-use evidence base already exists: **`docs/LITERATURE_EVIDENCE.md`** (every citation
  retrieved and field-checked in PubMed, with PMIDs/DOIs; rejected citations quarantined in its §8). This is
  the backbone of Intro + Discussion.
- **Anchor citations (must-cite):** RDoC (Insel 2010); HiTOP (Kotov 2017); p-factor (Caspi 2014); bifactor
  method (Reise 2012); normative modeling (Marquand 2016; Wolfers 2018); **B-SNIP biotypes (Clementz 2016,
  2024)** — engage explicitly; immuno-metabolic depression (Penninx 2024; Milaneschi 2020; Lamers 2013/2020);
  metabolic-burden-not-severity (Vancampfort 2015/16; Pillinger 2017; Perry 2019). Tension set (state
  openly): Howren 2009; Osimo 2020; Penninx & Lange 2018; Kappelmann 2021; Heinrich 2021; Levin-Aspenson
  2020; Watts 2021.
- **Flag (honesty):** Faugère 2025 is FACE-adjacent (same cohort family) — cite, don't omit.
- **Mechanics.** Build `article/references.bib` (BibTeX) from the dossier PMIDs/DOIs; the existing report
  uses a manual `thebibliography` — for the journal article switch to `.bib` + the venue's `.bst`. I can
  generate the `.bib` entries from the dossier on request.

---

## 8. The claims ledger (reviewer-facing; keep in the LaTeX as a comment block)

| We **claim** | We **do not claim** |
|---|---|
| A real, certified 9-D transdiagnostic measurement map (internal validity) | Natural biotypes/subtypes; that no biotype exists in any space |
| Biology **largely independent of a general functional-burden axis** (quantified, with boundary conditions) | Strict statistical orthogonality; independence from *being ill* |
| A graded continuum tighter than DSM-5 **descriptively** | That the strata are clinically superior to DSM-5 (predictive/treatment) beyond what M4/M5 show |
| Temporal coherence: durable biology, moving symptoms | That developmental-risk is genuinely "state" (it's CTQ recall noise) |
| Modest, **group-level** incremental prognosis for **functioning**, course-dependent | A large individual-risk gain; prediction of **recorded** relapse/hospitalization/events; a deployable rule |
| A **well-identified null** for lithium-BP moderation; a testable antipsychotic hypothesis | That the map guides treatment selection; any causal treatment effect on TAU |
| Convergent validity with the immuno-metabolic literature | External or out-of-sample validation (none yet) |

This table is the spine of both the Limitations section and the response-to-reviewers; keep it canonical.

---

## 9. Reviewer pre-mortem (anticipated objections + our answer)

**9.1 "'Independent of severity' is overstated."** *Answer:* reword to **"largely independent of a general
clinical/functional-burden axis"**; report the ≈0.06–0.14 values as a *quantification at the low end* of a
known pattern; name boundary conditions (case-control inflammation elevations are real; BMI confounding;
concentration in a metabolic subgroup). Emphasize G is **functional burden**, not symptom severity. (Source:
`LITERATURE_EVIDENCE.md` §1.)

**9.2 "Biotypes do exist (B-SNIP)."** *Answer:* different measurement space (brain biomarkers vs clinical–
biological coordinates); B-SNIP itself found DSM diagnoses to be a severity continuum; we claim no biotypes
*in this 9-D space*, consistent with the normative-modeling heterogeneity literature. Cite Clementz 2016/2024
and Pan 2026 (clustering can resurface biotypes — analytic-choice caveat) and note our five-method gate ran
*before* clustering.

**9.3 "The prognostic gain is tiny (ΔAUC +0.017)."** *Answer:* by design — the conservative
incremental-beyond-baseline frame; the value is group-level stratification (14%→60% remission spread) and
continuous functional forecasting, course-dependent; we explicitly *do not* sell an individual calculator.

**9.4 "Treatment analysis is null/observational."** *Answer:* that is the point — a boundary earned via a
proper causal pipeline; lithium-BP is a *well-identified* null; channeling makes clozapine non-estimable;
we specify the randomized data a real treatment-selection test needs. Reporting the null is the corrective.

**9.5 "Internal validity only — no external cohort, no events."** *Answer:* acknowledged prominently;
framed as a **measurement/discovery** paper (the map and its structure), with persistence and incremental
prediction as internal stress tests; external validation and incident-event outcomes named as the required
next studies.

**9.6 "Mania (2 indicators) / substance (2 cohorts) are thin."** *Answer:* flagged *partial* everywhere,
never "well-characterised"; both confirmed in the certified joint fit (cross-seed φ 0.993) but reported with
explicit reliability caveats.

**9.7 "Bifactor models are unstable / the general factor is artefactual."** *Answer:* flat-prior refit
reproduces loadings/Φ exactly (Tucker φ=1.00); WAIC decisively prefers bifactor; G anchored only on
functioning items sidesteps the "what does p mean" instability (Heinrich 2021; Levin-Aspenson 2020).

**9.8 "Confidential data → not reproducible."** *Answer:* code, configs, dictionary, and aggregate results
are shareable and tracked; every number is reproducible from `scripts/`→`reports/`; data access via
FondaMental governance (data-availability statement, §10).

---

## 10. Boilerplate to prepare (journal-required)

- **Data availability** — FACE/FondaMental governed access; aggregates + code shared; per-patient data
  confidential.
- **Code availability** — `src/face/`, `scripts/`, `configs/`, tests; license (repo `LICENSE`).
- **Ethics** — FACE cohort approvals / consent (obtain exact IRB references and registration from the PI).
- **Funding & roles** — FondaMental / cohort funders (PI to supply).
- **Author contributions, conflicts, acknowledgements** — PI to supply; include cohort investigators per
  FACE authorship policy.
- **Reporting checklist** — prepare the venue's checklist (e.g. STROBE for observational; Nature reporting
  summary). *(Open question §12.)*

---

## 11. Writing workflow & LaTeX layout

### 11.1 Phased plan

1. **Phase 0 — lock decisions** (this doc) + your answers to §12.
2. **Phase 1 — skeleton** `article/` LaTeX: `main.tex`, `sections/*.tex`, `references.bib`, journal-agnostic
   class; paste the claims ledger + abstract as the anchor.
3. **Phase 2 — Methods first** (it's the most reusable and the rigor backbone; lift from methods-of-record).
4. **Phase 3 — Results** R1→R6, each written against its source report with numbers locked.
5. **Phase 4 — Intro + Discussion + Limitations** using `LITERATURE_EVIDENCE.md`.
6. **Phase 5 — display items**: ~~produce Table 1~~ (DONE), regenerate/compose Figs 1, 4, 5, 6 to print
   spec; finalize Table 2; build the Extended Data set (incl. the M5 moderation forest + optional DAG).
7. **Phase 6 — abstract/title final, reference `.bib`, boilerplate, checklist.**
8. **Phase 7 — internal review**: claims-ledger audit, number re-verification against `reports/`, a
   read-through by a "reviewer" subagent, language/formatting pass.

### 11.2 Proposed `article/` layout

```
article/
  ARTICLE_PLAN.md            ← this document
  main.tex                   ← title, abstract, \input sections (journal-agnostic preamble)
  sections/
    00_abstract.tex
    01_introduction.tex
    02_results.tex           ← R1..R6 (or split per-beat files)
    03_discussion.tex
    04_limitations.tex
    05_methods.tex
    90_boilerplate.tex       ← data/code availability, ethics, contributions
  references.bib             ← built from docs/LITERATURE_EVIDENCE.md (PMIDs/DOIs)
  figures/                   ← print-spec exports (vector PDF/EPS + 300-dpi TIFF)
  tables/                    ← table1_characteristics + table2_dimensions
  submission/                ← cover letter, reporting checklist, response templates
```

We do **not** reuse `report/FACE-ATLAS.tex` directly — it is a ~chapter-length technical report; the article
is a fresh, compressed manuscript that *borrows its figures and numbers*. Keep the report as the canonical
long-form backup and Supplementary source.

### 11.3 Division of labor (proposed)

- **I draft** Methods, Results, the figure reworks/compositions, the `.bib`, Table 2, and the
  claims/limitations text — all traceable to committed reports.
- **You (PI) supply** ethics/registration/funding/authorship, the final journal choice (or confirm
  journal-agnostic), and scientific sign-off on wording (esp. §9.1).

---

## 12. Open questions for you (let's decide together)

**Resolved 2026-06-17:** ① Table 1 aggregates — **approved & produced** (`scripts/60_table1.py`).
② Treatment (M5) placement — **Extended Data** (main text in prose). ③ Venue — **journal-agnostic**, drafted
to the highest standard (ranking in §1.3 stands as a recommendation, not a constraint).

Still open:

4. **Authorship & cohort policy.** Are there FACE/FondaMental authorship or co-author-review requirements
   that affect scope or timing? (Also need exact IRB/ethics references + funding for the boilerplate, §10.)
5. **Companion paper?** You chose one comprehensive paper. Should I still reserve a methods-heavy companion
   (the Bayesian engine + Woodbury marginalization) as a fallback if a venue asks us to cut Methods?
6. **The biology⟂severity wording (§9.1).** Confirm you're comfortable adopting "largely independent of a
   general functional-burden axis" with stated boundary conditions as the canonical phrasing.
7. **Next build step.** Shall I proceed to **Phase 1–2** (scaffold `article/main.tex` + `sections/` and
   draft Methods first), and assemble **Table 2** (the nine-dimension table) from `ADJUDICATION.md` +
   `11_s5_9dim_loadings.csv`?

---

*Prepared 2026-06-17. Every numeric claim above is sourced to a committed `reports/NN_*.md` or `docs/*.md`;
the verification pass (plan task #4) re-checks them before drafting begins.*
