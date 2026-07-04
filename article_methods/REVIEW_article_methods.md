# Editorial & utility review — `article_methods`

*A transdiagnostic dimensional map as a self-calibrating measurement instrument.*
Two independent passes over the current draft (17-pp main + 12-pp supplement, six figures):
a **handling-editor story review** (does the narrative earn peer review?) and a **utility
review** (does the *method* demonstrably earn its keep?). Both read the full manuscript and
the assembled figure deck.

---

## Bottom line — two verdicts that point different directions

| Lens | Verdict | Meaning |
|---|---|---|
| **Story / narrative** (handling editor) | **Would send for review: YES** | Fig 1 is a genuine hook; the arc hook→mechanism→evidence→application is coherent; content is in roughly the right figures. |
| **Method utility / readiness** (methods reviewer) | **Major revision — added value only *partly* demonstrated** | The machinery is real and calibration + value-of-information are genuine wins, but the paper never runs the one comparison a methods venue requires, so two of its three most novel-sounding contributions currently read as *losing to simpler baselines*. |

**The one-sentence problem:** the instrument is internally coherent and its
calibration/VoI machinery is real, but the paper **never benchmarks the instrument against a
single cheap baseline** (sum-score + SE, FIML/lavaan EAP, mean-imputation + PCA) — so its two
most quotable numbers (patient-adaptive CAT *coincides* with the fixed order; archetype
reconstruction 59% *trails* PCA's 80%) read, on their face, as the novel machinery being beaten
by the naive alternative rather than as demonstrated wins.

This is not a narrative problem you fix by rewriting. It is a **"the method must be shown to
win somewhere"** problem — and the good news is that the three highest-leverage fixes are
low-effort reruns on the model object you already have.

---

## What is already convincing (keep and foreground)

1. **Loading ≠ information is a clean, falsifiable, useful result** (Fig 2). The closed-form
   Woodbury/Fisher machinery *quantifies* the divergence on the same 109-indicator fit rather
   than asserting it, and it has an immediate actionable consequence (item-count targets by
   loading × prevalence).
2. **Calibration is unusually thorough** (Fig 5) — four nominal levels *and* stratified by
   observed-item count, including the by-construction zero-item check. More rigorous than the
   usual single-point coverage claim.
3. **Value-of-information battery design is a concrete deliverable** (Fig 4): a named 27-item
   battery at mean reliability 0.70 and a *quantified* per-cohort under-measurement diagnosis
   (schizophrenia 20.7 vs bipolar 34.9 immunometabolic items) a clinic could act on. **This is
   the paper's single strongest utility demonstration.**
4. **The paper runs its own nulls instead of hiding them** — the CAT-vs-fixed simulation, the
   PCA/k-means reconstruction foils, the non-significant silhouette. Intellectually honest
   reporting most measurement papers suppress.
5. **The no-imputation invariant is mathematically motivated** (deletion = marginalization),
   giving the missingness-native claim real teeth rather than a design preference.
6. **Bank-limited axes are identified and distinguished from missingness-limited axes**
   (mania 0.41 / 2-item, substance 0.43 / 4-item) — a genuinely useful diagnostic a static
   point-estimate map cannot produce.

---

## The core problem, stated precisely

A methods paper lives or dies on showing the method **earns its keep versus what a reader
already has on their desk.** Right now:

- **Calibration** demonstrates *internal* consistency (the machine's posterior matches its own
  generative truth) — necessary, but any correctly-coded Bayesian estimator passes it *by
  construction*. It is not yet a comparative win, because no simpler estimator was pushed
  through the same protocol to be shown *mis*-calibrated by contrast.
- **Adaptive CAT** returns a null (coincides with the fixed order to 2×10⁻⁴). Honest — but a
  skeptical reviewer reads "the adaptive contribution adds nothing beyond a static list."
- **Archetype simplex** reconstructs 59% of variance vs **80%** for linear PCA-5. A reviewer
  reads "the interpretable summary is simply worse than a baseline the field already has."
- **Nothing quantifies what the *whole instrument* buys** (Woodbury EAP + Fisher info +
  per-patient Sᵢ) over the obvious cheap competitor. The central claim that this is worth
  building versus off-the-shelf tools is **asserted, not shown.**

Everything below turns each of these from a liability into a demonstrated strength.

---

## Improvement list — prioritized

### Tier 0 — the gate: show the method wins somewhere (do before submission)

**T0.1 — Head-to-head against a cheap baseline on *downstream* utility.** *(blocker; medium)*
Rerun the Fig 1C-style prognosis with (a) the full EAP + Sᵢ propagated, versus (b) a
sum-score / mean-imputed point estimate, on the same patients. Report a **decision-curve or
calibration difference** — a decision that is made *better* because the uncertainty was
carried. The paper asserts uncertainty is "not decorative"; this shows it.

**T0.2 — Misspecification / perturbed-DGP calibration stress test.** *(blocker; high)*
Generate synthetic data from a *perturbed* model (extra correlated residual, wrong link,
omitted class) and show coverage either degrades gracefully or is flagged. Without this,
"calibrated" is circular. If out of scope, **say so explicitly** in the Discussion: this is an
implementation-correctness check, not external validation.

**T0.3 — Information-ranked vs loading-ranked battery at matched N.** *(major; LOW effort)*
Build the 27-item battery a second way — ranked by raw loading instead of Fisher information —
and report the reliability gap in Fig 4's own terms. This is the cheapest possible proof that
the information-based rule *changes a real design decision*. If loading-ranked ties, the whole
Fig 2 thesis weakens; if it loses, you have your headline utility number.

### Tier 1 — reframe the honest nulls as the strengths they are (pure writing, high impact)

**T1.1 — CAT null → validated upper-bound + deployment saving.** Present the coincidence as a
*theoretical prediction confirmed* (population-mean curve is a tight upper bound, proven via
local independence), whose actionable payoff is: **deploy a fixed, pre-registered battery with
no adaptive-delivery infrastructure.** Then add one contrastive simulated axis with strong
θ-dependent information to show the method *does* detect adaptive gains when they exist — proving
the machinery is sensitive, not inert.

**T1.2 — 59% vs 80% → fidelity-for-interpretability with a named price.** State the 20.6-point
gap as a trade, then *earn* it: show the PCA-5 components are **not** interpretable as clinical
corners (report what they actually mix), and run the downstream check (does replacing archetypes
with PCA-5 change any prognosis conclusion?). If conclusions are unchanged, the R² gap
"doesn't matter where it counts."

**T1.3 — Silhouette n.s. → lead with it as the argument FOR archetypes.** This is the single
best reframe available. The non-significant silhouette (0.140 vs 0.137 ± 0.002, z = 1.13) is not
a weakness to bury — it is *why archetypal analysis is the right tool*: because there are **no
natural clusters**, a hard-partition biotype claim would be unjustified, and a convex-blend
simplex is the only defensible summary of a continuum. Make this the **lead sentence** of the
archetype section, paired with the entropy distribution (median 0.77, 74% no-majority blend) as
converging evidence.

**T1.4 — Calibration → relabel as implementation-correctness.** In Methods/Discussion, call the
self-DGP coverage what it is: confirmation that the Fisher-scoring approximation and Woodbury
reduction are bug-free — not that the model is correctly specified for real patients. Honest
labelling pre-empts the "circular" objection.

### Tier 2 — convert asserted → demonstrated (medium effort, strong payoff)

**T2.1 — Missingness-native vs imputation, quantified.** Refit the projection with mean- and
single-imputation (MICE) versions of the same bank; compare reliability curves and coverage.
The paper *claims* imputation "injects covariance the model misreads as a factor" but never
measures the bias. *(major; high)*

**T2.2 — Does closing the measurement gap change prognostic accuracy?** Compare
remission-prediction AUC/calibration under the current uneven per-cohort battery vs the proposed
27-item battery. Gives Fig 4 a downstream consequence. *(missing panel; medium)*

**T2.3 — Runtime: Woodbury vs naive inversion.** Report wall-clock/memory for the O(|Cᵢ|³) →
O((K+1)³) reduction at N=9,013. The speedup is asserted but never measured. *(minor; LOW)*

**T2.4 — Sensitivity of headline numbers.** Report the 27-item / 0.70, 59% R², and bank ceilings
across A = 4/5/6 and battery sizes 20/27/35, so conclusions are visibly not fragile to arbitrary
choices. *(minor; LOW)*

**T2.5 — Add the null-summary R² anchor to Fig 6c.** Plot R² of "keep the raw 8-D coordinate,
no summary" (= 100%) and a random-rotation floor alongside PCA/k-means, so the reader sees the
full 0→100% range the 59% sits in. *(minor; LOW)*

### Tier 3 — figure & narrative polish (handling editor)

- **Reorder the arc:** calibration (Fig 5, *evidence*) should come **before** value-of-information
  (Fig 4, *application*) — you must earn trust before you spend it on design. Current order puts
  the application before the evidence that licenses it.
- **Thread Fig 1 with one patient as connective tissue.** Add a before/after ghost interval:
  show this patient's cognition/suicidality whiskers (panel A) *shrinking* if the panel-D
  recommended items were administered. Turns a 2×2 capability grid into one causal chain.
- **Move the CAT-convergence result (2×10⁻⁴) into Fig 1**, next to panel D — it is what
  *licenses* the "next best item" recommendation as adaptive-optimal despite a fixed lookup.
- **Move the bank-limited annotation from Fig 3 into Fig 4's heatmap** — that is where readers
  must not misread mania/substance red cells as "go collect more" (structurally impossible).
- **Kill list → supplement/caption:** the k-means-5 strawman line in Fig 6c (nobody proposes hard
  5-clustering; it dilutes the real PCA gap); the dense unannotated scatter in Fig 2a (keep the
  3–4 annotated exemplars); the six overlapping near-identical curves in Fig 3a (the diminishing-
  returns point is made once).
- **Generalization (reviewer Q4):** every headline number comes from one fit on one dataset. A
  cross-validation or held-out-cohort replication of at least the reliability curves and 27-item
  battery would blunt the "elaborate in-sample description" objection. *(medium/high — scope call)*

---

## Baselines a methods reviewer will demand (none currently in the paper)

| Baseline | Metric that settles it |
|---|---|
| Sum-score + SE (classical test theory) | Coverage/reliability of sum-score CI vs EAP + Sᵢ on identical simulated patients |
| lavaan / Mplus FIML EAP factor scores | Correlation/RMSE vs EAP coordinates; relative reliability at matched item counts |
| Mean-imputation + PCA on completed matrix | Reconstruction R² / reliability vs the missingness-native projection, same masks |
| Single/multiple imputation + factor scores | Calibration coverage vs the no-imputation model's 0.949 |
| Loading-ranked battery at matched size (27) | Mean reliability across 8 axes vs the greedy entropy battery's 0.70 |
| Legacy clinical battery (if one exists) | Per-axis reliability at native item count vs the VoI-selected battery |

---

## The five hardest questions a reviewer will ask

1. If patient-adaptive CAT gives essentially the same reliability as a fixed order, what is left
   of the "adaptive sampling" contribution beyond publishing a static list — and does "adaptive"
   over-sell the title?
2. Calibration is simulated from the model's own parameters — isn't that true by construction,
   and what happens to coverage under plausible misspecification?
3. Archetype simplex keeps 59% vs PCA-5's 80%, and the silhouette found no clusters — why prefer
   the archetype summary over just publishing PCA-5 scores?
4. Every headline number comes from one fitted model on one dataset — where is the
   cross-validation / held-out / replication evidence that the pipeline generalizes?
5. You claim the method "earns its keep" over simpler alternatives but never run one head-to-head
   comparison — how is a reviewer to know the whole apparatus beats tools already in every
   applied researcher's toolkit?

*Questions 1–3 and 5 are answered by Tier 0–1 above. Question 4 is the Tier-3 generalization
item and the main remaining scope decision.*

---

## Recommended path to "ready"

The cheapest route to a convincing submission is **not** more machinery — it is three low-effort
reruns plus four paragraph-level reframes:

1. **T0.3** (loading- vs information-ranked battery) — one rerun, directly demonstrates Fig 2's
   thesis changes a decision.
2. **T2.3 + T2.4 + T2.5** (runtime, A/size sensitivity, null-R² anchor) — all low-effort, all
   close "is this fragile / is this necessary" doubts.
3. **T1.1–T1.4** (reframe the four honest nulls) — pure writing, converts the paper's most
   quotable liabilities into strengths.
4. Then commit to **one** of the two blockers — **T0.1** (downstream decision utility vs
   sum-score) is the higher-impact, more-tractable choice than T0.2.

That package moves the utility verdict from *"added value partly demonstrated"* to *"demonstrated
where it counts,"* without touching the narrative the handling editor already passed.
