# M2 — Strata: Findings & Discussion

> **The paper-facing synthesis of Milestone 2**: what we did, what we observed, what we found, and what it
> means. Canonical *findings + discussion* record for the stratification layer (PI review + manuscript).
> Companions: methods of record → [`STRATIFICATION_MODEL.md`](STRATIFICATION_MODEL.md); the atlas + per-view
> detail → [`STRATA_ATLAS.md`](STRATA_ATLAS.md); the detailed development record (methods rationale, ideas
considered, per-stage observations) → [`STRATA_RESULTS.md`](STRATA_RESULTS.md); the measurement layer it builds on →
> [`M1_FINDINGS.md`](M1_FINDINGS.md); live status → [`STATE.md`](STATE.md). Every numeric claim is backed by
> a committed `reports/2x_*.md`. Updated 2026-06-09.

---

## 1. Summary

On the **M1 9-dimension coordinates** for all **N = 9,013** FACE V0 patients (BP 6,252 · SZ 2,209 ·
DR 552), each carrying per-patient posterior uncertainty, we asked whether decision-relevant **strata** exist
— and, critically, *what kind of structure* the transdiagnostic space has, before imposing any clustering.
The answer is a **graded continuum, with no evidence for well-separated discrete biotypes**: cluster-tendency, modality, density, and
topology diagnostics converge on "no natural kinds" (gap-statistic K=1, HDBSCAN 0 clusters, unimodal PC1,
a smooth archetype scree with no elbow, a flat mixture-BIC basin). We therefore represent the continuum two
complementary, uncertainty-propagating ways: **(i) eight stable extreme phenotypes (archetypes)** — the
corners of the cloud, each the high pole of one specific axis, plus a low-burden pole — with every patient a
**soft convex blend** of them; and **(ii) a four-region soft tessellation** (a coarser decision-region
overlay). Both are **fully transdiagnostic** (adjusted Rand index ≈ 0 against both the 3 cohorts and the 7
DSM-5 subtypes), **driven by the specific/biological axes rather than overall severity**, **stable**, **not
an artefact of missingness**, and form a **tighter description of the data than DSM-5** (mixture BIC 199.3k
vs 206.0k with fewer components; DSM-5 explains only ~5% of the coordinate variance vs ~21%). The headline:
**the biology⊥G signal survives into the phenotypes** — there are *distinct* metabolic and inflammatory
extreme phenotypes, while overall severity is the continuum's *spine* (it has no corner of its own). All of
this is **internal/descriptive validity**; whether the strata are *clinically better* than DSM-5 —
predictive and treatment validity — is deferred by design to M3/M4/M5.

---

## 2. The empirical map

The stratification space is the M1 9-axis coordinate cloud (orientation: higher = more burden). M2 adds
three nested objects, in the order the work established them:

1. **Structure verdict — a continuum.** The cloud is one diffuse, graded mass with smooth gradients of
   severity and biological load running through it in different directions, and with the cohorts and all 7
   DSM-5 subtypes fully intermixed. There are **no discrete clusters** (§3 F1).
2. **Eight extreme phenotypes (archetypes — the lead view).** The corners of the cloud's convex hull map
   almost one-to-one onto axis extremes: a **low-burden pole** plus one corner each for high **cognition+
   severity**, **sleep**, **metabolic**, **developmental-risk**, **mania**, **inflammatory(+substance)**, and
   **suicidality**. Each patient is a soft simplex blend (§3 F2–F4).
3. **A four-region soft tessellation (a coarse overlay).** Low-burden (31%) · severity+metabolic (32%) ·
   low-metabolic/better-cognition (25%) · mania+developmental+sleep (12%) — regions of the continuum, **not**
   kinds.

Per-patient membership for both views (with uncertainty) is the M2 hand-off, `results/face/patient_strata.parquet`.

---

## 3. Principal findings

Each finding is stated as *observation → result → interpretation*.

### F1 — The transdiagnostic space is a continuum, not biotypes
A pre-registered structure-discovery gate (run *before* any clustering, and uncertainty-aware over the M1
posterior draws) gives a unanimous verdict: **gap-statistic K = 1**, **HDBSCAN finds 0 clusters (100% of
points "noise")**, **PC1 is unimodal** (Hartigan dip p ≈ 0.99), the archetype **scree is smooth with no
elbow**, and the mixture **BIC has a flat basin** (no separating K). **Result:** the 9-dimensional cloud is
graded, not partitioned. **Interpretation:** psychiatric heterogeneity here is dimensional, not a set of
natural kinds — consistent with the dimensional (RDoC/HiTOP) view and with why "biotype" cluster claims so
often fail to replicate. Building the gate first is what let the data say "continuum" rather than us
imposing a K.

### F2 — Eight stable extreme phenotypes span the continuum
Archetypal analysis recovers **A = 8** extreme phenotypes that are **highly reproducible** (cross-seed
Tucker congruence 0.999). They map almost one-to-one onto **axis extremes**: a low-burden pole + a high pole
for each specific axis. **Interpretation:** the honest representation of a continuum is its *extremes* and
the blends between them — archetypes provide an interpretable, out-of-sample-projectable soft basis without
fabricating discrete groups.

### F3 — Biology⊥G survives into the phenotypes (the headline)
Two findings together. **(a)** There are **distinct metabolic and inflammatory extreme phenotypes** — the
inflammatory corner appears only once the resolution is fine enough (A=8), but it is real and separate from
the metabolic corner. **(b)** **Overall severity (G) has no corner of its own** — at every resolution it is
the *spine* the cloud varies along, not an extreme. **Result:** the validation confirms it quantitatively —
the partition's variance-explained is led by the **specific axes** (mania η² 0.45, developmental 0.35,
metabolic 0.21, sleep 0.19, cognition 0.17) more than by G (0.31), with the maximum specific axis exceeding
G. **Interpretation:** the M1 premise — biological load varies on axes the clinical severity picture does not
see — carries all the way into the patient phenotypes. Two patients at the same severity can be opposite
biological phenotypes. This is the precise value a biology-aware stratification adds.

### F4 — Most patients are blends — the strata are genuinely soft
**75% of patients have no single dominant archetype** (max simplex weight < 0.5; mean normalized entropy
0.67) — they live in the *interior* of the simplex, between extremes. **Interpretation:** hard assignment
would be a fiction here. The decision-region object is the *distribution* over phenotypes, which is exactly
what M1's per-patient uncertainty and the continuum structure demand. (The coarse 4-region tessellation
assigns more confidently — 92% — precisely because it tiles broad regions rather than resolving corners; the
two views are the coarse-label and fine-blend ends of the same continuum.)

### F5 — The strata are fully transdiagnostic
**Adjusted Rand index of the partition against diagnosis is ≈ 0** — 0.007 (cohort) and 0.020 (the 7 DSM-5
subtypes) for the tessellation; 0.06 / 0.05 for the archetypes. **Result:** the data-driven structure shares
almost no information with the diagnostic taxonomy; every phenotype and region mixes all cohorts and all
subtypes (with only weak gradients — Cramér's V 0.18–0.28). **Interpretation:** the strata **cut across**
DSM-5 — the necessary precondition for adding anything beyond it. Clinically coherent gradients remain
(the mania corner is BP-heavy with almost no depression-cohort patients; the severity+cognition corner draws
the most schizophrenia and major-depression patients), but no phenotype *is* a diagnosis.

### F6 — The strata describe the cloud better than DSM-5 (descriptively)
Head-to-head on the coordinates (§1.7): a free 4-region mixture reaches **BIC 199,325 versus 206,016 for a
DSM-5-constrained 7-group mixture** — a decisive win with *fewer* components — and the free partition
explains **~21% of the coordinate variance versus ~5% for DSM-5**. **Interpretation:** the 7 diagnostic
boxes barely organize the transdiagnostic coordinate space; the data's own regions describe it far more
tightly. This is a **descriptive** superiority — the *predictive and treatment* head-to-head (the validators
that matter) is M4/M5 and is **not** claimed here.

### F7 — The strata are real, not a missingness artefact
The central risk of clustering uncertainty-laden, heavily-missing coordinates is manufacturing groups out of
*who-was-measured*. **Result:** a classifier predicting stratum membership from the per-axis coverage pattern
achieves **0.248 accuracy versus a 0.323 majority baseline — a negative lift**; and the partition is
seed-stable (tessellation ARI 0.987, archetype congruence 0.999). **Interpretation:** membership is governed
by what patients *are*, not by what was *measured* on them — the measurement-error mixture (which deconvolves
the known per-patient noise) and the no-imputation handling did their job. This is the M2 vindication of M1's
"propagate the uncertainty" design.

### F8 — Severity is the spine and substance is absorbed — two honest non-corners
Neither **overall severity** nor **substance** forms an extreme phenotype at any resolution (A = 5–8).
Severity does not because it is the general axis the whole cloud varies along (a corner would require it to
be an independent extreme — it is not). Substance does not because it is the noisiest, only-2-cohort axis
(no DR data), so it self-down-weights and never anchors a corner (it appears only as a side-loading on the
inflammatory phenotype). **Interpretation:** both non-results are informative and consistent with the model —
G as a bifactor spine, substance as a low-information 2-cohort axis — and were *predicted* in the M2.0 prep.

---

## 4. Methodological contributions

- **Uncertainty propagated end-to-end, not discarded.** Every step consumes the M1 per-patient posterior:
  the structure gate runs over draws; the mixture is fit by **Extreme Deconvolution** (`x_i ~ Σ_k π_k
  N(m_k, V_k + S_i)`, known per-patient noise `S_i`); the archetype weights carry draw-propagated SDs. This
  is what makes F7 (not-a-missingness-artefact) hold — and it is the direct M2 expression of M1's
  observed-likelihood, no-imputation invariant, now at the coordinate layer.
- **Structure-gate-first.** Asking "*is there cluster structure at all?*" before clustering — with topology
  (Mapper), modality (dip), tendency (Hopkins), density (HDBSCAN), and model-selection — converted a hidden
  assumption (K biotypes) into a *tested, reported result* (a continuum). It is the M2 analogue of M1's
  "the eligibility map is itself a result."
- **Archetypes-as-lead, given the continuum.** A continuum has no natural clusters but it has *extremes*;
  archetypal analysis is the representation that fits that geometry — interpretable, out-of-sample, soft —
  while the Gaussian mixture is demoted to a soft *tessellation* (an overlay, not biotypes). Reporting both,
  and noting where they agree, is the robustness argument.
- **A full-N projection that reproduces the joint fit.** M1 had scored three non-Gaussian axes
  (suicidality/developmental/substance) only on its fit subsample; M2.0 projected them onto all 9,013 under
  the fixed parameters and **reproduced the fixed latent at Pearson r ≈ 1.00** on the overlap —
  extending the map to full N with no re-fit and no imputation.
- **A fair head-to-head vs DSM-5.** Comparing a free mixture to a DSM-5-*constrained* mixture under the
  identical measurement-error likelihood, and decomposing coordinate variance per partition, operationalizes
  "better description than DSM-5" without circularity — and cleanly separates the *descriptive* claim M2 can
  make from the *predictive/treatment* claim it cannot.

---

## 5. Discussion

**What the map is for.** M2 converts the M1 coordinate system into decision-region objects — soft phenotype
memberships with uncertainty — that the later milestones act on. It deliberately stops at *internal*
validity: it establishes that the strata are real, stable, transdiagnostic, value-driven, and a tighter
description than diagnosis, and it sets up the head-to-head that M4/M5 will run on outcomes and treatment.

**Why a continuum is the right — and most useful — answer.** Finding no biotypes is not a null result; it
is a finding with teeth. It says the actionable object is *position on continuous axes / proximity to
extreme phenotypes*, not membership in a box — which is both more honest and more flexible for individualized
decisions than a new set of categories. It also reframes "better than DSM-5" precisely: the question for M4
becomes *does a continuous, biology-aware coordinate/archetype representation out-predict the 7 DSM-5
subtypes (and severity) on course and treatment response?* — a sharper, more defensible bar than "are our
clusters the true kinds?"

**Why biology⊥G is again the consequential finding.** That metabolic and inflammatory load form their own
extreme phenotypes — independent of the severity spine — is what makes the map worth building. A stratification
that only recovered severity tiers would be a re-dressed CGI-S; this one separates patients who look equally
ill but are biologically opposite, which is the precise hypothesis a precision-psychiatry layer should test.

**Honesty as a design principle (continued from M1).** The pipeline was built to let the data refuse
discreteness, and it did; to flag substance as a non-corner, and it did; to catch missingness-driven
artefacts, and it found none. The reported strata are what survived adversarial structure-testing and
validation, not what was assumed.

---

## 6. Limitations

1. **Internal/descriptive validity only.** V0 baseline; no outcomes. The strata are *not yet* shown to be
   clinically actionable — predictive validity (M4) and treatment-moderation (M5) are the decisive tests and
   are deferred by design.
2. **Continuum ⇒ K and A are granularity choices, not natural numbers.** The tessellation K = 4 and the
   archetype A = 8 are parsimony/interpretability choices (PI-confirmed), justified by the BIC basin and the
   need to resolve both biology corners; neighbouring values are equally valid soft bases.
3. **Two rare extreme corners.** The inflammatory (1.9%) and suicidality (1.5%) archetypes are the long tails
   of skewed latents — real but sparsely populated; their per-patient weights carry wide uncertainty.
4. **Heteroscedastic, partial axes carry less weight.** Mania (partial for all patients), substance
   (2-cohort, DR-absent), and prior-dominated cognition/inflammatory contribute less to the structure (they
   self-down-weight) — a correct behaviour, but it means those axes are under-resolved for some patients.
5. **Embeddings are visualization-only.** UMAP/PCA panels illustrate the continuum but are never clustering
   inputs (all structure work is on the 9-D coordinates).

---

## 7. Future work

- **M3 — temporal coherence (V1–V4):** do the coordinates and the phenotype memberships *persist*? Is a
  stratum more temporally stable than a DSM-5 diagnosis (which itself shifts, e.g. MDD→BP, schizophreniform→
  schizophrenia)?
- **M4 — prognosis (the first actionability test):** do the strata add **incremental predictive value beyond
  diagnosis + severity** on course/relapse/hospitalization/functioning — the head-to-head-vs-DSM-5 on
  *predictive* validators (§1.7)? The natural engine is **Bayesian profile regression** (cluster the
  coordinates jointly with the outcome), a direct extension of the M2 mixture.
- **M5 — treatment:** do the strata **moderate treatment response** (stratum × treatment interaction) — the
  strongest "actionable" test, the point at which knowing a patient's phenotype changes management.
