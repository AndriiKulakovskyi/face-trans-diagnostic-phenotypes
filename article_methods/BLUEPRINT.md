# FACE-ATLAS methods paper — architecture / blueprint

_A storyboard for a standalone **measurement-methods** paper built from the instrument-centric
material currently demoted inside `article/`. Companion to the clinical/discovery paper
(`article_v2/`). This is the blueprint — thesis, narrative spine, section map, display-item plan,
new-analysis specification, and the explicit content division that keeps the two papers from
overlapping. Not the manuscript. All numbers here are the source manuscript's own canonical values;
new-analysis numbers are filled in once those analyses run (Phase 1)._

---

## 0. The one decision this encodes

`article/` and `article_v2/` are two framings of the **same** work. `article_v2/` already made its
choice: **biology is the headline, the pipeline is the Methods** — the immunometabolic axis that
separates, persists, and predicts. That paper is the discovery/clinical story and it is largely
built.

This paper makes the *opposite, complementary* choice, and it is the choice the project's own
`FLAGSHIP_PAPER_ARCHITECTURE.md` (§8–§9) already anticipated:

> "The **value-of-information / optimal-instrument-selection** block moves here (or becomes the
> companion paper). … it is self-contained and its removal *tightens* the flagship. Do **not**
> split methods from clinical; that halves both."

So the methods paper does not compete with `article_v2` for the biology result — it takes the
material `article_v2` deliberately sheds (projection, information importance, adaptive sampling,
value-of-information, uncertainty calibration, archetypes-as-summary) and makes it the **hero**.
The immunometabolic finding appears here only as a *worked example of the instrument*, never as
the claim.

**The single most consequential reframing:** in `article/` the instrument material is a demoted
digression — the cut-candidates review (#15, #16) flags four consecutive Results subsections
(information accumulation, minimal-indicator rule, "a few well-chosen indicators", battery design)
as a "methodological digression that outweighs its narrative payoff" and recommends moving them to
a supplement. **This paper promotes exactly that block from digression to thesis.** What was cut
from the flagship *is* the methods paper.

---

## 1. The one-sentence thesis

> A transdiagnostic dimensional map is not a fixed set of point-coordinates but a **self-calibrating
> measurement instrument**: fit to each patient's observed cells only (no imputation), it returns for
> every patient a posterior coordinate *and its uncertainty ellipsoid*, quantifies exactly how much
> information each indicator contributed (Fisher information, not loading), prescribes the cheapest
> next measurement that shrinks that uncertainty, and propagates that uncertainty into every
> downstream use — turning a questionnaire bank into an auditable, designable instrument.

Everything in the paper serves that sentence. If a paragraph advances a *clinical* claim rather than
an *instrument* property, it belongs in `article_v2`, not here.

---

## 2. What is genuinely new — and what is not (state both, up front)

**New (the paper's methods claims):**
1. **The map read as an instrument, end-to-end.** Not "we built a bifactor model" (those pre-exist)
   but: one missingness-native model whose closed-form structure makes four operational instrument
   properties — projection, information accounting, adaptive sampling, value-of-information —
   computable *on the same object*, and validated together.
2. **Loading ≠ information, made operational on a mixed-type map.** The quantity that governs how
   sharply a patient is located is the per-indicator **Fisher information under its own likelihood
   family**, not the loading. On a mixed Gaussian / logistic / graded-response / negative-binomial
   bank this dissociates sharply (a high-loading, ~1%-prevalence binary flag contributes almost no
   information). The paper turns this into a design lens.
3. **A closed-form value-of-information design rule.** Via the matrix-determinant lemma applied to
   the posterior precision, the entropy reduction from any not-yet-measured indicator is closed-form
   and *depends on what is already measured*, so a greedy cross-axis rule assembles a minimal shared
   battery and localizes each cohort's under-measurement — an actionable consortium specification,
   not a post-hoc audit.
4. **New in this paper (Phase 1 analyses):** (a) a fully patient-adaptive (CAT-style) selection order
   benchmarked against the population-mean upper bound; (b) simulation-based **calibration** of the
   posterior coordinate (empirical coverage of the credible ellipsoids under known truth + real
   missingness); (c) **archetype reconstruction fidelity** — how faithfully the five-corner convex
   blend summarizes the eight-dimensional continuum.

**Not new (say so — it is a strength):**
- Bifactor / ESEM models, the Woodbury identity, EAP scoring, Fisher information, computerized
  adaptive testing (CAT), and archetypal analysis each pre-exist and are cited as such.
- Placing routinely-collected biology inside the map, and the immunometabolic finding, are the
  contribution of the **companion** paper; here they are only an illustrative worked example.

**Why "not new" makes the paper stronger:** the instrument is assembled from well-understood,
individually-trusted pieces — which is exactly why a reviewer should believe the composite. The
novelty is the **integration and its validation on a real, pervasively-incomplete, mixed-type
transdiagnostic bank at N=9,013**, not any single estimator.

---

## 3. Title options (methods-forward)

1. **_A transdiagnostic dimensional map as a self-calibrating measurement instrument: projection,
   information, and value-of-information under structured missingness_** _(recommended — full scope)_
2. **_Loading is not information: reading a mixed-type psychiatric factor map as an adaptive
   measurement instrument_** _(hook-forward, leads with the sharpest single idea)_
3. **_Measuring how well we measured: uncertainty-honest projection and instrument design for
   transdiagnostic dimensional maps_**
4. **_From questionnaire bank to designed instrument: value-of-information battery construction for a
   diagnosis-blind dimensional map of 9,013 patients_**

Recommended venue class: **measurement / quantitative-methods** — _Psychological Methods_,
_Behavior Research Methods_, _Multivariate Behavioral Research_, _Educational and Psychological
Measurement_; methods-forward biostatistics (_Statistics in Medicine_, _Biometrical Journal_) as
alternates. Keep the structure venue-agnostic so a resubmit needs only abstract/format surgery.

---

## 4. The narrative spine (the through-line every section serves)

A single escalating argument: **you can only trust a map as far as you can measure it — so measure
the measurement.** Five beats, each an instrument property, each grounded in one closed-form result
already in the source manuscript.

| Beat | Claim (instrument property) | The move | Key object / number |
|---|---|---|---|
| **Project** | Every patient is a posterior *coordinate + uncertainty*, never a false-precise point | EAP score map + Woodbury reduction to the (K+1)-dim factor space; unmeasured axes return the prior | Eq. `m-scores`, `m-woodbury`; worked patient posterior SD 0.22 (6 items) vs ~0.99 (0 items) |
| **Weigh** | Loading is not information | Per-indicator Fisher information, exact per likelihood family, at the population mean; the "paradox items" | `i(λ)=λ²/(1−λ²)`; binary info `∝ λ²·p(1−p)` collapses at ~1% endorsement |
| **Sample** | A few well-chosen items recover most of an axis; some axes are bank-capped | Minimal-indicator rule + **patient-adaptive (CAT) order** (Phase 1) vs the population-mean upper bound | 3–6 items to reliability ~0.85; mania capped 0.408, substance 0.429 |
| **Design** | The map prescribes the cheapest next measurement | Closed-form VoI via matrix-determinant lemma; greedy cross-axis battery + cohort under-measurement map | 27-item shared battery at mean reliability 0.70; SZ 20.7 vs BP 34.9 immuno items |
| **Trust & summarize** | The instrument is calibrated, and its low-dim summary is faithful | **Coverage calibration on synthetic truth** (Phase 1) + **archetype reconstruction fidelity** (Phase 1); S_i propagated downstream | nominal-vs-empirical coverage; variance retained by the 5-corner blend; archetypes locate ~9.7× tighter than DSM-5 |

The spine is the escalation **project → weigh → sample → design → trust**. It ends not on a clinical
payoff but on the instrument's own credibility (calibration) and the fidelity of its most-used
summary (archetypes) — the two places a methods reviewer will push hardest, answered with new
analyses.

---

## 5. Rewritten abstract — instrument-forward (working draft)

> Dimensional maps of psychopathology are only as trustworthy as the measurement that produces them,
> yet they are typically read as fixed point-coordinates that hide how sharply each patient is
> located, how much each instrument contributed, and how cheaply the map could be measured better. We
> present the transdiagnostic dimensional map as a **measurement instrument** rather than a finding.
> A missingness-native Bayesian bifactor / exploratory structural-equation model — fit to each
> patient's observed cells only (no imputation), under mixed likelihoods and a Gaussian-copula
> continuous block — returns for every one of N=9,013 patients (bipolar disorder, schizophrenia,
> major depression; 21 sites) a posterior coordinate **and** its uncertainty ellipsoid, and its
> closed-form structure makes four instrument properties computable on that one object. **Projection:**
> an expected-a-posteriori score map, reduced to the eight-dimensional factor space by a Woodbury
> identity, returns the prior — never a false-precise point — on unmeasured axes. **Information:**
> because indicators are conditionally independent given the factors, posterior precision accumulates
> additively, and the quantity that governs localization is the Fisher information each indicator
> contributes at its own likelihood family, *not* its loading — high-loading, low-prevalence flags
> contribute almost nothing. **Adaptive sampling:** a loading-dependent minimal-indicator rule and a
> patient-adaptive selection order recover most of each axis's reliable signal in three to six items,
> and expose two axes that are intrinsically capped by the instrument bank (mania 0.408, substance
> 0.429) rather than by missingness. **Value of information:** the entropy reduction from a
> not-yet-measured indicator is closed-form (matrix-determinant lemma), so a greedy rule assembles a
> shared 27-item battery at mean reliability 0.70 and localizes where each cohort is under-measured.
> We show by simulation that the posterior coordinate is well-calibrated (empirical coverage matches
> nominal under known truth and real missingness), that a five-archetype convex-blend simplex
> summarizes the continuum with quantified reconstruction fidelity while locating a patient ~9.7×
> more tightly than the DSM-5 label, and that the per-patient covariance propagates into every
> downstream use. The contribution is the instrument: a map that reports its own uncertainty, weighs
> instruments by the information they carry, and prescribes how to measure the next cohort more
> cheaply.

_Note the frame: the map and its instrument properties open and close; no clinical finding is a
claim. The immunometabolic worked example is an illustration of projection, not a result._

---

## 6. Section-by-section organization (methods-journal format)

Measurement venues use **Introduction → Methods → Results → Discussion** (methods before results,
unlike Nature-family). Generous Methods, real derivations in the main text or a tight appendix.

**Introduction (~700 w).** Three beats: (1) dimensional maps are proliferating, but are consumed as
fixed point-scores that hide measurement quality; imputation and point-estimate scoring make it
impossible to say how well any given patient is located; (2) the pieces to do better exist
individually (bifactor/ESEM, EAP, Fisher information, CAT, VoI) but have not been assembled and
validated *together* on a real, incomplete, mixed-type transdiagnostic bank; (3) we read one such
map as an instrument and ask four operational questions of it — where is a patient and how sharply;
which items earned that precision; how few items suffice; what to measure next. End on the thesis
(§1). **Out of scope, stated explicitly:** the clinical/biological findings, which are the companion
paper.

**Methods (~2,000 w).** The instrument machinery as the star:
1. _Cohorts and the no-imputation data layer_ (brief; the observed-cell invariant).
2. _The measurement model_ — bifactor/ESEM, mixed likelihoods, Gaussian-copula block, the
   observed-cell likelihood (Eq. `obslik`), Σ = ΛΦΛᵀ + Ψ and why imputation is forbidden.
3. _The loading as a measurement slope_ — Eq. `m-slope`, the loading as the Jacobian ∂η/∂f; sets up
   "loading ≠ information".
4. _Projection: EAP scoring and the Woodbury reduction_ — Eqs. `m-scores`, `m-woodbury`; per-pattern
   reuse; the prior returned on unmeasured axes.
5. _Information accounting_ — additive precision (Eq. `res-precision`); exact per-family Fisher
   information; the minimal-indicator rule (Eq. `res-mincount`).
6. _Adaptive sampling and value of information_ — greedy selection; matrix-determinant-lemma entropy
   reduction; the CAT extension (Phase 1).
7. _Archetypes as a low-dimensional summary_ — convex-blend simplex (Eq. `m-archetype`); a modelling
   summary of the continuum, not imposed clusters.
8. _Calibration protocol_ — synthetic truth + real missingness; coverage of the credible ellipsoids
   (Phase 1).

**Results (~1,800 w, the five-beat spine of §4):** Project → Weigh → Sample → Design → Trust &
summarize. Each subsection = one display item + its number.

**Discussion (~900 w).** What an instrument reading buys (auditability, principled battery design,
honest downstream uncertainty); scope and limits (population-mean upper bound vs the CAT realized
curve; pooled item parameters assume common ordering; point-estimate loadings do not themselves
propagate parameter uncertainty into the VoI ranking; single consortium); and the explicit division
of labour with the clinical companion paper.

**Appendix / SI.** Woodbury derivation, copula construction, Fisher-information forms per family,
CAT simulation detail, calibration protocol detail.

---

## 7. Display-item plan (5–6 main items)

Methods reviewers read the figures first. Six main display items carry the spine.

- **Fig 1 — Project (the worked patient).** _Most-arresting asset._ One real, de-identified patient
  end-to-end: posterior coordinate + 95% HDIs on all eight axes (immunometabolic +3.57 SD, SD 0.22
  from 6 items; cognition/suicidality prior-dominated, SD ~0.99) → 86% archetype-A2 weight →
  interval-valued prognosis → the single most informative next item (one verbal-memory test,
  cognition reliability 0 → 0.65). Source: `fig_worked_patient.py` / `fig_localization`.
- **Fig 2 — Weigh (loading ≠ information).** Per-item loading vs Fisher information contributed
  (common latent metric), by likelihood family; the paradox items (high loading, ~0 information);
  the binary-information-vs-prevalence mechanism panel. Source: `fig_loading_vs_info.py`.
- **Fig 3 — Sample (adaptive assessment).** Reliability vs items administered, per axis,
  most-informative first; the two bank-capped axes; **plus the new patient-adaptive (CAT) curve vs
  the population-mean upper bound** (Phase 1). Source: `fig_adaptive_assessment.py` + new CAT track.
- **Fig 4 — Design (value of information).** Greedy cross-axis 27-item battery at mean reliability
  0.70 (first item per axis = its canonical instrument); the cohort under-measurement map. Source:
  `fig_value_of_information.py`.
- **Fig 5 — Trust (calibration).** _NEW (Phase 1)._ Nominal vs empirical coverage of the posterior
  credible ellipsoids on synthetic patients with known truth + real missingness, per axis and by
  observed-item count.
- **Fig 6 — Summarize (archetype simplex + reconstruction fidelity).** Membership-entropy "fog" of
  all 9,013 patients in the five-corner simplex (`fig_simplexfog`) **+ new reconstruction-fidelity
  panel** (variance retained / reconstruction error of the convex blend vs the full 8-dim
  coordinate; Phase 1).

**Extended Data / SI candidates:** full loading × Fisher-information table; per-family
Fisher-information derivations; CAT vs upper-bound per-axis detail; calibration by missingness
stratum; Woodbury per-pattern reuse timing; variational (GLLVM) re-estimation cross-check.

---

## 8. New analyses this paper adds (Phase 1) — specification

Each turns a *stated limitation* of the existing analysis into a *result*, giving the methods paper
its own contribution beyond reframing.

1. **Patient-adaptive (CAT) reliability curve vs the population-mean upper bound.**
   - _Why:_ `article/` states the reliability curves are "an upper bound on the efficiency a fully
     adaptive test … would realise" because Fisher information is evaluated at the population mean.
     The methods paper should *realize* that adaptive test.
   - _What:_ simulate CAT-style selection that re-evaluates each item's information at the patient's
     provisional posterior estimate; compare realized reliability-vs-items to the population-mean
     curve, per axis, over representative simulated patients.
   - _Output:_ `analysis/cat_vs_upperbound.csv` + figure-ready data → Fig 3.
2. **Posterior-coordinate calibration (coverage under known truth).**
   - _Why:_ `article/` notes the point-estimate scoring "does not itself propagate parameter
     uncertainty"; a methods reviewer will ask whether the reported credible ellipsoids are honest.
   - _What:_ generate synthetic patients with known latent coordinates and the real FACE missingness
     pattern; project them through the frozen EAP map; measure empirical coverage of the credible
     ellipsoids / per-axis HDIs vs nominal, overall and by observed-item count.
   - _Output:_ `analysis/coverage_calibration.csv` + figure-ready data → Fig 5.
3. **Archetype-simplex reconstruction fidelity.**
   - _Why:_ archetypes are used as the map's low-dimensional summary; the paper should quantify how
     much is lost by the five-corner convex blend, not just assert it summarizes.
   - _What:_ reconstruct the eight-dim coordinates (and, where available, standardized indicators)
     from `W·Z`; report variance retained, per-axis reconstruction error, and information lost vs the
     full coordinate, across all 9,013 patients.
   - _Output:_ `analysis/archetype_reconstruction.csv` + figure-ready data → Fig 6.

All three read only committed model objects (`results/face/gllvm_oop/…`, `strata_oop/…`) and derived
aggregates; no per-patient raw clinical value is emitted.

---

## 9. Content-division table — methods paper vs `article_v2` (no overlap)

The organizing principle: **`article_methods` owns instrument *properties*; `article_v2` owns
clinical *findings*.** The same objects (loadings, Φ, coordinates, archetypes, S_i) appear in both,
but each paper asks a different question of them.

| Material | `article_methods` (this paper) | `article_v2` (clinical companion) |
|---|---|---|
| No-imputation observed-cell likelihood | **Core method** — the invariant the instrument rests on | Brief Methods mention |
| Bifactor/ESEM + copula + mixed likelihoods | **Core method**, full detail | Methods, condensed |
| Loading as slope / Jacobian (Eq. `m-slope`) | **Own subsection** (sets up loading≠info) | Not present |
| EAP projection + Woodbury (Eqs `m-scores`,`m-woodbury`) | **Beat 1 (Project)** — hero | Methods, one line |
| Fisher information / loading≠information | **Beat 2 (Weigh)** — hero, Fig 2 | Not present |
| Minimal-indicator rule + adaptive/CAT | **Beat 3 (Sample)** — hero, Fig 3 (+new) | Not present |
| Value of information / battery design | **Beat 4 (Design)** — hero, Fig 4 | Not present |
| Posterior-coordinate calibration | **Beat 5 (Trust)** — NEW, Fig 5 | Not present |
| Archetypes | As a **low-dim summary**; reconstruction fidelity (NEW, Fig 6) | As **clinical strata** with prognostic gradient (their Fig 5) |
| Continuum-not-biotypes structure test | Mentioned as *why archetypes* (a summary of a continuum) | **Own result** (their "Organizes" beat, Fig 2) |
| Immunometabolic separates from G (φ≈0.10) | Only as a **worked-example** illustration of projection | **The hinge** — their central finding, Fig 3 |
| Immunometabolic durability (ICC 0.91) | Not a claim here | **"Persists" beat**, their Fig 4 |
| Archetype → 2-yr functional remission (22–63%) | Only as the worked patient's interval prognosis | **"Predicts" beat / money figure**, their Fig 5 |
| Errors-in-variables prognosis GLM | Cited as *where S_i is propagated* (an instrument property) | **Own result** (incremental validity), their Fig 6 |
| Temporal trait/state decomposition | Cited as *where S_i is propagated* | **Own result** (durable biology, moving symptoms) |

**One-line test for a paragraph:** does it make a claim about *how well we measured* (→ methods
paper) or *what we found about patients* (→ `article_v2`)? The immunometabolic axis is the clearest
shared object: here it is a pin on the worked-example map; there it is the discovery.

**What must NOT appear in the methods paper:** the biology-as-co-equal-axis novelty claim, the
separability/durability/prognosis triple, the DSM-vs-dimensional clinical argument, the
enrichment/monitoring-target framing. Those are `article_v2`'s claims to fame; repeating them here
would split the discovery across two papers and weaken both (per `FLAGSHIP_PAPER_ARCHITECTURE.md`
§9).

---

## 10. Build & provenance

- New parallel folder `article_methods/`; **nothing in `article/` or `article_v2/` is modified.**
- Figures regenerated from committed model objects and aggregates only (no per-patient raw value),
  same house style and 300-dpi PNG+PDF as `article/`.
- Same BasicTeX/tectonic toolchain as the sibling folders; `references.bib` reused from `article/`.
- New-analysis result tables live in `article_methods/analysis/` with the script that produced them.

---

## 11. Risks a methods reviewer will raise — and how this structure pre-empts them

- _"The pieces all pre-exist."_ → §2 states it first; novelty is the validated integration on a real
  incomplete mixed-type bank, not a new estimator.
- _"Your reliability curves are an upper bound."_ → Fig 3 adds the realized CAT curve (Phase 1
  analysis 1), so the bound and the realized efficiency are both shown.
- _"Are the credible ellipsoids honest?"_ → Fig 5 is a simulation-based coverage calibration (Phase 1
  analysis 2) — the direct answer.
- _"Archetypes throw away information."_ → Fig 6 quantifies exactly how much (reconstruction fidelity,
  Phase 1 analysis 3); the claim is bounded, not asserted.
- _"VoI ranking uses point-estimate item parameters."_ → stated as a limitation; the gaps are
  empirical, the ranking assumes a common ordering — owned, not buried.
- _"Isn't this just the clinical paper again?"_ → §9 content-division table; the immunometabolic axis
  appears only as a worked example, never as a claim.

---

_Deliverable: a storyboard. Next (per the approved plan): Phase 1 runs the three new analyses; Phase
2 composes the six figures; Phase 3 drafts the manuscript; Phase 4 compiles and QAs. Nothing in
`article/` or `article_v2/` is modified by this document._
