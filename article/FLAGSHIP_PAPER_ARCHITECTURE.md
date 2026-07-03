# FACE-ATLAS — Flagship paper: story architecture (from-scratch draft)

_A high-level storyboard for one comprehensive, discovery-framed paper aimed at the very best
journals. Not the manuscript — the blueprint: thesis, narrative spine, section-by-section
organization, a rewritten discovery-forward abstract, a display-item plan, and the delta from
the current draft. All numbers are the manuscript's own canonical values._

---

## 0. The one decision this encodes

**One paper, discovery-forward.** The finding and the method are mutually load-bearing (the biology
is convergent-with-literature without the rigorous measurement; the pipeline is engineering-without-a-payoff
without the finding). The single most consequential change from the current draft is the **frame**:

> The current abstract closes with *"the contribution is the measurement-model pipeline."* That sentence
> is what caps the paper at a methods-journal ceiling. For a flagship, **the contribution is the map and
> the one property it reveals; the pipeline is why you believe it.** Method becomes the enabler, not the hero.

---

## 1. The one-sentence thesis

> When routinely-collected biology is placed as a **co-equal latent axis inside** a diagnosis-blind
> transdiagnostic map — measured honestly, never imputed, uncertainty carried end-to-end — one axis,
> **immunometabolic load, separates from the entire clinical severity picture, stays put over two years,
> and marks the worst-prognosis pole** — making it a concrete cohort-enrichment and monitoring target
> that a diagnosis or a severity score cannot see.

Everything in the paper is in service of that sentence. If a paragraph doesn't advance it, it goes to Methods or Supplement.

---

## 2. What is genuinely new — and what is not (state both, up front)

A top reviewer's first question is "what's actually new here?" Answer it *inside the paper* before they ask.

**New (the paper's claims to fame):**
1. **Biology as a co-equal latent axis**, not a correlate of symptom dimensions — to our knowledge the first
   such diagnosis-blind map spanning bipolar disorder, schizophrenia and major depression (N=9,013).
2. **A measured, not assumed, separation**: turning "biology is separable from severity" from a modelling
   constraint into an *estimated* quantity (general-factor correlation ≈0.10 for immunometabolic vs 0.39/0.42
   for cognition/sleep) — with the honesty machinery (no imputation, mixed likelihoods, uncertainty propagated,
   map frozen before validation) that makes the number trustworthy.
3. **One property that recurs across all four validation stages** — the immunometabolic axis is simultaneously
   the most *separable* (0.10), the most *durable* (ICC 0.91), and the *worst-prognosis* pole (22–63%
   functional-remission gradient). Convergence across independent tests is the argument.

**Not new (say so — it is a strength, not a weakness):**
- That immunometabolic/inflammatory biology matters in severe mental illness, and that it forms a partly
  separable dimension, is established (van Campfort; Pillinger; Perry; the Penninx/Milaneschi/Lamers IMD programme).
- Bifactor/ESEM models, the Woodbury identity, and value-of-information test design each pre-exist.

**Why "not new" makes the paper stronger:** the biology result **convergent-validates against a large
independent literature** — which is exactly why we state it as *relative*, measured independence, and why it
reads as discovery rather than cohort-specific noise. The novelty is the **measurement discipline that lets a
qualitatively-known pattern be quantified inside one coherent map**.

---

## 3. Title options (ordered biology-forward → method-forward)

1. **_A transdiagnostic map places immunometabolic burden as a distinct, durable, prognostic axis of severe mental illness_**  _(biology-forward — recommended for Nature Medicine / Nature Mental Health / Lancet Psychiatry)_
2. **_Biology as a co-equal axis: a diagnosis-blind dimensional atlas of 9,013 patients with bipolar disorder, schizophrenia and depression_**  _(map-forward — recommended for Nature / Nature Human Behaviour)_
3. **_Measuring what a diagnosis blurs: an uncertainty-honest transdiagnostic atlas and its one emergent biological axis_**  _(integration-forward — Molecular Psychiatry / Biological Psychiatry)_
4. _An immunometabolic island in the transdiagnostic landscape of severe mental illness_  _(evocative short-title option)_

---

## 4. The narrative spine (the through-line every section serves)

A single escalating argument, not four parallel analyses. The four validation stages become **four
consecutive tightenings of one claim**, culminating in the immunometabolic island:

| Beat | Claim | The move | Payoff number |
|---|---|---|---|
| **Exists** | A trustworthy map can be built at all | Diagnosis-blind, no imputation, uncertainty carried | 8 axes from 143 indicators, N=9,013 |
| **Organizes** | It is a continuum, not biotypes | Structure test vs single-Gaussian null; archetypes | apparent clusters reproduced by null; state described ~9.7× tighter than DSM-5 |
| **Separates** | Biology stands apart from severity | Free-correlation re-fit → *measured* estimand | immuno 0.10 vs 0.39/0.42; ~1% vs 15–18% shared variance |
| **Persists** | The separable axis is stable, symptoms move | Frozen map scored onto 2-yr follow-up; trait/state split | immunometabolic ICC 0.91 (most durable) |
| **Predicts** | It carries prognosis — honestly bounded | Errors-in-variables GLM vs diagnosis+severity | worst-prognosis pole 22→63%; ΔELPD +62.8 held-out, **but** AUC +0.010 |

The last two columns are the spine: **the same axis wins on separability, durability, and prognosis.** That
triple convergence — not any single p-value — is the flagship-level result. End on the honest bound (group-level,
not individual) reframed as the *correct* claim: an **enrichment/monitoring target**, not a treatment rule.

---

## 5. Rewritten abstract — discovery-forward (concrete draft)

> Psychiatric diagnoses group biologically heterogeneous patients, and dimensional alternatives have struggled
> to show that any one biological signal is more than a correlate of overall severity. We built **FACE-ATLAS**, a
> diagnosis-blind dimensional map of **N=9,013** patients with bipolar disorder, schizophrenia or major depression
> (three harmonized FACE cohorts, 21 expert-centre sites), in which routinely-collected biology is placed as a
> **co-equal latent axis** alongside cognition, sleep and symptom dimensions rather than as a downstream correlate.
> A single global Bayesian bifactor model — fit to each patient's observed cells only (no imputation), with mixed
> likelihoods, uncertainty propagated end-to-end, and the map frozen before any validation — yields eight axes: a
> general functional-burden factor and seven specific dimensions. The patient space is a **graded continuum, not
> discrete biotypes** (apparent clustering is reproduced by a single-Gaussian null) and is summarized by a
> five-archetype simplex that locates a patient ~9.7× more tightly than the DSM-5 label. Allowing the general
> factor to correlate freely with the specific axes, one property recurs across every validation stage: an
> **immunometabolic axis nearly independent of general burden** (correlation ≈0.10 vs 0.39/0.42 for cognition/sleep;
> ~1% vs 15–18% shared variance; robust to medication, adiposity and site), which is the **most temporally durable**
> dimension (test–retest ICC 0.91) and the **worst-prognosis pole** for two-year functioning (functional remission
> 22→63% across corners). This separation is *relative* and *measured*, and converges with an independent
> immunometabolic-depression literature. The map improves prediction of future functioning beyond diagnosis and
> severity at the group level (held-out ΔELPD +62.8) but not at the individual level (AUC +0.010), positioning the
> immunometabolic axis as a **cohort-enrichment and monitoring target rather than an individual treatment rule**.
> All results are internal-validity findings on observational data; individual-level utility would require external
> validation and incident-event outcomes.

_Note what moved: the map and the biology now open; the method is one clause; the honest bound closes the argument
instead of a claim about pipelines. Every number is the manuscript's own._

---

## 6. Section-by-section organization (Nature-family research-article format)

Flagship general/psychiatry venues use **Intro → Results (methods embedded lightly) → Discussion → Methods (at end)**,
~3,000–4,500 main-text words, 4–6 display items, with an Extended Data tier. Map the current content onto that:

**Introduction (~500 w).** Three beats: (1) diagnoses aggregate heterogeneity; dimensional alternatives are only as
good as their measurement; (2) prior transdiagnostic work treats biology as a correlate and tolerates imputation /
point estimates — so it cannot say whether a biological axis is *separable* from severity; (3) we build a
diagnosis-blind map that puts biology inside it and ask one question of every stage: *what stands apart, stays, and
predicts?* End on the immunometabolic island as the answer. **Drop** the "(i)–(iv) pipeline enumeration" and the
"three commitments" restatement from the intro — those are Methods.

**Results (~2,200 w, 5 subsections = the spine of §4):**
1. _A diagnosis-blind map of severe mental illness_ (Exists) — Fig 1.
2. _The patient space is a continuum, summarized by five archetypes_ (Organizes) — Fig 2.
3. _Immunometabolic load separates from general severity_ (Separates — the hinge) — Fig 3.
4. _The separable axis is the durable one_ (Persists) — Fig 4a.
5. _It marks the worst-prognosis pole — at the group level_ (Predicts, honestly bounded) — Fig 4b/5.

**Discussion (~900 w).** What the convergence means; why *relative* independence; the stratify-vs-monitor design
principle; limitations as a numbered, confident list (observational; group-level prognosis; measurement horizon).
End on the enrichment/monitoring target framing.

**Methods (at end, unlimited).** The pipeline is the star *here*: observed-cell likelihood, sparse bifactor/ESEM,
Gaussian-copula block, Woodbury marginalization, freeze-before-validation protocol, errors-in-variables GLM,
invariance/robustness battery. This is where "the contribution is the discipline" belongs — as rigor, not as headline.

**Extended Data (8–10 items) + Supplementary Information.** Annexes A–H become Extended Data figures + SI. The
**value-of-information / optimal-instrument-selection** block moves here (or becomes the companion paper — see §9).

---

## 7. Display-item plan (the figures ARE the paper at this tier)

Flagship reviewers read the figures first. Five main display items carry the spine; everything else is Extended Data.

- **Fig 1 — The atlas exists.** Schematic of the diagnosis-blind pipeline (compact) + the eight-axis map "at a
  glance" (loadings heatmap or factor summary). One panel must show *biology sits as its own axis*.
- **Fig 2 — Continuum, not biotypes.** Structure test vs single-Gaussian null + the five-archetype simplex, colored
  by cohort to show DSM cuts across it (ARI 0.021). The "no density gaps" panel.
- **Fig 3 — The hinge (biology ⟂ severity, measured).** The bifactor-vs-correlated-arm contrast: immunometabolic
  ≈0.10 vs cognition/sleep 0.39/0.42, with the shared-variance gap (1% vs 15–18%) and the robustness ladder
  (medication/adiposity/site → 0.07). **This is the single most important figure** — it is the evidence for the thesis.
- **Fig 4 — Persists & predicts.** (a) trait/state variance thermometer, immunometabolic most durable (ICC 0.91);
  (b) archetype → 2-yr functional-remission gradient (22→63%), immunometabolic corner worst.
- **Fig 5 — The honest bound.** Incremental prognosis: map vs diagnosis vs severity (ΔELPD +62.8 group-level;
  AUC +0.010 individual) — the figure that turns a limitation into the correct claim (enrich/monitor, not treat).

**Extended Data (candidates):** VoI / optimal-battery, invariance across BP/SZ/DR, leave-one-cohort-out, variational
re-estimation, full loading tables, consort/attrition, sensitivity arms.

---

## 8. Kill-your-darlings — what leaves the main text

The cut-list pass already flagged most of these; the flagship format makes the calls non-negotiable:
- **Value-of-information / adaptive-battery block** → Extended Data or companion paper (§9). It is a *design tool*,
  a digression from the biology spine.
- **Woodbury algebra, copula construction, identification proofs** → Methods.
- **Self-referential rigor statements** ("we report null findings as deliberately as positive ones") → cut.
- **The (i)–(iv) pipeline enumeration and three-commitments restatement** → Methods (stated once).
- **Coarse two-region tessellation aside** → SI.

---

## 9. Target journals (primary + alternates, with the format implication)

| Venue | Fit | Format note |
|---|---|---|
| **Nature Medicine** _(primary if clinical translation is foregrounded)_ | Biology + prognosis + "enrichment target" reads as translational | Clinical structured elements; wants a path to utility (state it as enrichment, honestly) |
| **Nature Mental Health / Nature Human Behaviour** | Transdiagnostic dimensional map is squarely in scope | Nature-family format; figure-driven |
| **Lancet Psychiatry** | High clinical reach; cohort-scale dimensional work | Structured abstract; clinical framing; SAP-style rigor expected |
| **Molecular Psychiatry / Biological Psychiatry** | If the immunometabolic biology is the lead | More biology-specialist; method welcomed |
| _Nature_ (main) | Only if framed as a general-science measurement advance *with* the biology as proof | Highest bar; needs the "first co-equal biology axis" claim to land broadly |

**Recommendation:** target **Nature Medicine or Nature Mental Health first** with title option 1/2, discovery-forward
abstract (§5). Keep the structure journal-agnostic (as the PI already decided) so a resubmit needs only abstract/format surgery.

**The companion-paper option (from the prior discussion):** if length forces a split, spin off the
**instrument-optimization / value-of-information** methodology to a measurement venue (_Psychological Methods_,
_Assessment_, _Behavior Research Methods_) — it is self-contained and its removal *tightens* the flagship. Do **not**
split methods from clinical; that halves both.

---

## 10. The delta from the current draft (what actually changes)

The science and numbers are unchanged. The rebuild is **framing, order, and proportion**:
1. **Reframe the thesis**: map + biology as the contribution; pipeline as Methods. (Rewrite the abstract's closing; rewrite the intro's last paragraph.)
2. **Reorder Results** into the five-beat spine (§6), making "Separates" the explicit hinge.
3. **Split the two senses of "primary"** — the bifactor is the *primary map* (coordinate system); the free-correlation vector is the *primary estimand* for the biology question. (The three clarity edits already drafted this session.)
4. **Tag the abstract numbers** as coming from the correlated-G arm.
5. **Demote** the VoI block and algebra to Extended Data / Methods.
6. **Rebuild the figures** around the five-item plan, with Fig 3 as the load-bearing panel.
7. **State novelty and non-novelty explicitly** (§2) so convergent validity reads as strength.

---

## 11. Risks a top reviewer will raise — and how this structure pre-empts them

- _"The biology is already known."_ → §2 states it first and turns it into convergent validity; the novelty is the measured axis inside one map.
- _"Two models — which is real?"_ → §6.3 frames the bifactor as coordinate choice and the correlated arm as the estimand; they are nested, not rival.
- _"Prognosis is weak (AUC +0.010)."_ → Fig 5 makes the honest bound the *claim*: enrichment/monitoring target, not treatment rule. Owning it disarms it.
- _"Observational, single consortium."_ → limitations list + external-validation caveat, stated confidently, not buried.
- _"Is the immunometabolic axis just adiposity?"_ → robustness ladder (Fig 3) + the CRP-vs-body-composition discussion.

---

_Deliverable: a storyboard, not a manuscript. Next steps on request: (a) draft the rewritten Introduction and
Discussion to match §4–6; (b) build the Fig-1–5 outline as a figure plan; (c) produce the companion-paper
scoping memo. Nothing in the current manuscript files is modified by this document._
