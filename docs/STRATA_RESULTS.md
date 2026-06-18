# M2 — Strata: detailed results, observations, ideas & discussion (development record)

> The curated, article-grade **development log** of Milestone 2: per-stage methods + observations, the
> **ideas and alternatives considered** (and why we chose what we chose), the reasoning behind every
> decision, the engineering/reproducibility notes, and the extended discussion. This is the record that
> feeds the manuscript's Methods-rationale, Results, and Discussion sections — the layer the synthesis docs
> compress out. Companions: methods of record → [`STRATIFICATION_MODEL.md`](STRATIFICATION_MODEL.md);
> paper-facing synthesis → [`STRATA_FINDINGS.md`](STRATA_FINDINGS.md); the atlas → [`STRATA_ATLAS.md`](STRATA_ATLAS.md);
> the measurement layer → [`M1_FINDINGS.md`](M1_FINDINGS.md) / [`RESULTS.md`](RESULTS.md). Every number is
> reproducible from `scripts/20–26` → `reports/2x_*.md`. Accumulates per stage. Updated 2026-06-09.

---

## 0. The M2 arc and design philosophy

M2 is the third layer of the project (`cohorts → dimensions (M1) → strata (M2) → prognosis/treatment
(M4/M5)`). It acts only on the M1 9-dimension coordinates (N = 9,013; BP 6,252 · SZ 2,209 ·
DR 552), each carrying per-patient posterior uncertainty, and never on raw indicators or diagnosis.

Four design commitments shaped everything and are themselves results worth stating:

1. **Test structure before imposing it.** We did not assume "biotypes" exist. A structure-discovery gate
   (M2.1) asks *is there cluster structure at all?* and is a reported result — the M2 analogue of M1's
   "the eligibility map is itself a result." This is what let the data return a **continuum** rather than a
   forced K.
2. **Propagate uncertainty end-to-end.** Every step consumes the M1 per-patient posterior (SD/draws). This
   is the M2 expression of M1's no-imputation, observed-likelihood invariant, now at the coordinate layer —
   and it is what defeats the dominant failure mode (clustering on *who-was-measured*; §6 Q4).
3. **Diagnosis is metadata.** BP/SZ/DR and the 7 DSM-5 subtypes are validation/interpretation only — never
   inputs.
4. **Honest scope.** M2 delivers *internal/descriptive* validity. "Decision-relevance" (predictive,
   treatment) is deferred to M3/M4/M5. "Converged/validated" never means "clinically better" here.

The deliverable is two complementary, uncertainty-propagating soft representations of the continuum —
**archetypes (lead)** and a **soft tessellation** — plus their validation.

---

## 1. Ideas & modelling approaches considered (the rationale record)

This section documents the design space we weighed — valuable as the manuscript's Methods rationale and as
the answer to "why this method and not that one?"

### 1.1 The representation: patients are *distributions*, not points
After M1, each patient is **not a point but a ~Gaussian blob** in 9-D: a posterior mean coordinate plus a
**per-patient, per-dimension SD**, and (for some axes/patients) a missing or prior-dominated coordinate. The
uncertainty is strongly **heteroscedastic** (§2). The single most consequential implication: a naïve
hard clustering of posterior means (k-means/standard GMM) would be dominated by the precisely-measured axes
and would place patients by the *prior-mean artefacts* of their unmeasured axes — manufacturing strata out
of **missingness patterns**. This is why "probabilistic decision regions" is the correct framing, not a
stylistic one, and why every method below had to either propagate `S_i` or be relegated to a diagnostic.

### 1.2 Method families weighed

| approach | uncertainty | missingness | data-driven K | non-convex | soft | out-of-sample | interpretable | role taken |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| **measurement-error mixture** (XD) | yes | yes | yes | no | yes | yes | yes | **engine** (tessellation) |
| **archetypal analysis** | (draws) |  |  | n/a | yes | yes | yesyes | **lead view** |
| spectral / community graph |  |  |  | yes |  | no |  | not used (see 1.3) |
| HDBSCAN / density | no | no | yes | yes |  |  |  | structure diagnostic |
| GNN embedding | no |  | n/a | yes |  | yes | no | parked → M4 (see 1.3) |
| TDA / Mapper, dip, gap |  |  | n/a | yes | n/a | no | yes(shape) | structure gate |

### 1.3 Graph clustering and GNNs — considered and deliberately parked (a Methods-rationale point)
Graph methods (spectral clustering, Louvain/Leiden community detection, Similarity Network Fusion) and graph
neural networks are the workhorses of much multi-omics patient stratification, so we considered them
explicitly. The decisive argument against them *here*: **the M1 projection has already done the work graph
methods exist to do.** Their main advantages — manifold-following in high, un-denoised dimensions, and
multi-view fusion — are either moot (we are in a clean, interpretable 9-D Euclidean-ish latent space with
quantified uncertainty) or upstream of where we now sit. Against that, graph partitions *fight* the two
properties we most need: they discard the per-patient uncertainty unless one hand-builds an uncertainty-aware
affinity (e.g. a Bhattacharyya/2-Wasserstein kernel between the patient Gaussians), they are mostly
*transductive* (no out-of-sample assignment for new patients — needed for M4/external cohorts), and they
return hard partitions with K set by a resolution knob. **GNNs** are an additional mismatch at M2: the graph
would be *synthetic* (k-NN on coordinates — no real edges, so a GNN largely re-learns geometry we already
hold explicitly), the setting is unsupervised with only 9 features (GNNs are data/structure-hungry and shine
with a learning signal), there is no native input-uncertainty propagation, and the embedding is a black box
that works against M2's interpretability deliverable. **Where they earn their place is later:** M4, given a
*real* graph and a learning signal — a multi-view patient-similarity network (SNF-style) or a temporal
patient–visit graph (M3), combined with **outcomes** for semi-supervised prognosis (graph regularization /
label propagation handles heavy missingness well). Graph thinking is not wasted at M2: the **consensus /
co-assignment** stability analysis *is* a patient graph, and an uncertainty-aware spectral pass remains a
valid triangulation/shape-check. The verdict — graphs/GNNs are mismatched to a clean, low-dimensional,
unsupervised stratification but well-matched to the outcome-bearing later milestones — is itself a useful
contribution to state.

### 1.4 The two decisions that shaped the build
- **Cluster-vs-continuum elevated to a first-class question** (not just "what K?"). The honest null is not
  only "K=1" but "the structure is *continuous*." We therefore made the structure gate (TDA/dip/Hopkins/
  density/model-selection) a gating step that decides which representation leads. *(This proved decisive —
  the answer was "continuum.")*
- **G handled both ways.** Because M1's specific factors are bifactor-orthogonal to G *by construction*,
  "drop G from the feature set" **is** the G-residualized (pure-profile) view — no ad-hoc regression. We ran
  **Arm A** (all 9 — severity×profile) and **Arm B** (8 specifics — pure profile) throughout and compared.
  *(They agreed on the continuum verdict and on the archetype geometry; biology is already ≈⊥G, so the arms
  differ mainly on cognition/sleep, which partly track severity.)*

---

## 2. M2.0 — full-N coordinates + uncertainty (the substrate)

### 2.1 Method
M1 left three explicit, non-Gaussian axes (suicidality, developmental-risk, substance) scored only on its
~1,884-patient fit subsample. M2.0 projects them to all 9,013 by a **conditional posterior under fixed
parameters**: holding the fixed loadings, cutpoints, dispersions, and Φ fixed, we sample each
patient's explicit latent `f_e` from their observed binary/ordinal/count cells (reusing the exact S5 mixed
likelihood; NumPyro NUTS). This is a projection, **not** a re-fit, and **no imputation** — DR patients have
no substance items, so their substance coordinate is prior-dominated, never filled. The six
continuous-anchored axes use draw-wise analytic conditional-Gaussian scores. We export, per patient, mean ·
SD · HDI · observed-indicator count · reliability tier, **and** a thinned posterior-draws array
(`coordinates_draws.npz`, `[200, 9013, 9]`) — the uncertainty-faithfulness arm for the mixture's `S_i` and
the archetype-over-draws fit.

### 2.2 Faithfulness QC (the key check)
The full-N projection **reproduces the fixed `f_e` at Pearson r ≈ 1.00** on the fit-subsample overlap
(suicidality / developmental / severity 1.000, substance 0.999; 0 divergences; R-hat(z_e) 1.04 — per-patient
latent mixing, point estimates exact). So the projection faithfully extends the joint fit to the other
~7,100 patients.

### 2.3 Observations worth recording
- **Coverage is heavily structured.** Prior-dominated counts: cognition 2,506 (no neuropsych testing),
  inflammatory 1,684 (no labs); **mania is *partial* for every patient** (only 2 indicators); substance is
  mostly *partial* (3,382 well / 5,269 partial — a 2-cohort axis). This degrades gracefully because the
  models down-weight high-uncertainty coordinates.
- **Uncertainty is strongly heteroscedastic** (mean posterior SD): developmental 0.16, metabolic 0.27,
  sleep 0.28, severity 0.29 (tight) vs cognition 0.44, suicidality 0.48, inflammatory 0.55, mania 0.66,
  substance 0.80 (loose). This is precisely what the measurement-error machinery consumes.
- **Cross-cohort means are clinically coherent** (independent sanity that the projection produced signal,
  not noise): mania highest in BP (0.03 vs −0.08 SZ / −0.14 DR); suicidality lowest in SZ (−0.26 vs +0.12
  BP / +0.13 DR); developmental-risk highest in DR (0.34); overall severity highest in DR (1.16) > SZ
  (0.38) > BP (−0.11).
- **An inter-dimension correlation preview** (correlations of posterior means; mean |off-diagonal| 0.135):
  the strongest couplings are **substance↔inflammatory 0.52** and **substance↔metabolic 0.39**, plus the
  M1 immunometabolic link (metabolic↔inflammatory 0.28), sleep↔mania 0.31, and suicidality↔developmental
  0.30. The substance/biological-load coupling foreshadowed the archetype geometry (substance rides on the
  inflammatory corner, §4).
- **A methods detail (ordinal alignment).** The attempt-ordinal `isf08a` has rare high categories (raw 6/7/
  10) in full-N that the certified fit subsample never saw; we re-encode full-N ordinals to the **certified
  category coding** with top-category absorption so the fixed cutpoints stay valid (3 patients re-mapped).

### 2.4 Boundary
M2.0 is the substrate only — it makes all 9 axes full-N with honest uncertainty; no structure is claimed
here. Artifacts: `results/face/m2/{coordinates_full.parquet, coordinates_draws.npz, validation_table.parquet}`,
`reports/20_prep_coordinates.md`, `docs/figures/20_coverage.png`.

---

## 3. M2.1 — structure discovery: the pivotal result (CONTINUUM)

### 3.1 Method
A battery run on both G-arms and **uncertainty-aware** (over M1 draws / jittered by `S_i`), on the native
latent z-scale (no re-standardisation — that would inflate the noisy low-variance axes; the cross-patient
SDs span only ~0.58–1.07): **Hopkins** (cluster tendency), **Hartigan's dip** (modality, per axis + PC1),
**GMM-BIC sweep**, **KMeans silhouette**, **Tibshirani gap statistic**, **HDBSCAN** (density), and a minimal
**Mapper** graph (lens = severity). A conservative synthesis declares "clustered" only on converging
evidence and otherwise defaults to "continuum" — the honest null.

### 3.2 Results (each diagnostic + what it contributes)
| diagnostic | value | contribution |
|---|---|---|
| Gap statistic | **K = 1** | strongest single signal: no clustering preferred over one blob |
| HDBSCAN | **0 clusters, noise fraction 1.00** | no density peaks separated by valleys |
| Hartigan dip (PC1) | **p ≈ 0.99** (Arm A 0.996, B 0.994) | the principal axis is unimodal |
| KMeans silhouette | peak **≈ 0.18** (A 0.174, B 0.191), only declines | no well-separated groups at any K |
| GMM-BIC | best K = 12/11 but **monotone** (drop to K≈3, then flat) | tiles a continuum; no interior optimum |
| Mapper | 11 nodes, **1 connected component** (a chain) | graded backbone, not islands |
| uncertainty sweep | Hopkins 0.81 ± 0.01; GMM K_best **mode 4** (3:3, 4:15, 5:2) | verdict stable under measurement error; the noise washes fine over-segmentation toward ~4 |
| Hopkins | **0.85–0.86** | the *lone* high signal — see caveat |

### 3.3 The Hopkins caveat (honesty)
Hopkins 0.85 reads "clustered" on its face, but Hopkins is **biased upward in structured, high-dimensional
data** (a correlated cloud with gradients inflates it without any clusters). It is outweighed by the gap
statistic, HDBSCAN, dip, and silhouette, all of which directly address cluster-vs-continuum and all say
continuum. We record this rather than hide it.

### 3.4 The UMAP observation
The 2-D embedding (visualisation-only, never a clustering input — UMAP distorts density, so a PCA companion
guards against illusory clusters) shows **one diffuse cloud** with the three cohorts and all 7 DSM-5
subtypes **fully intermixed**, and two smooth continuous **gradients** crossing it in different directions —
a severity gradient and a biological (inflammatory) gradient. A landscape with a ridge and crossing slopes,
not a constellation of islands.

### 3.5 Why this is the pivotal result
It converts the field's default assumption (discrete biotypes) into a *tested, falsified* hypothesis on this
data, and it sets the entire downstream framing: a continuum has no natural kinds but it has **extremes** and
**gradients**, so the lead representation becomes archetypes (corners) + a soft tessellation (regions), never
biotypes. It also reframes "better than DSM-5" (§6.5): the question becomes whether a *continuous /
soft-archetype* representation out-predicts the 7 boxes — a sharper, more defensible bar. Artifacts:
`reports/21_structure.md`, `docs/figures/21_{selection,embedding,mapper}.png`.

---

## 4. M2.3 — archetypes (the LEAD view)

### 4.1 Why archetypes lead, and the method
A continuum is best described by its **extremes** and the blends between them. Archetypal analysis (PCHA;
the `archetypes` library) represents each patient as a convex combination of `A` archetypes that are
themselves convex combinations of patients — i.e. the **corners of the data's convex hull**, with each
patient a point on the simplex. Fit on the native z-scale; uncertainty propagated by **projecting M1 draws
onto the fixed anchor archetypes** (per-patient weight SDs); out-of-sample membership = projecting a new
patient onto the fixed archetypes.

### 4.2 Choosing A — a parsimony decision, transparently made
The explained-variance **scree is smooth with no elbow** (ev 0.24 → 0.41 → 0.51 → 0.59 → 0.68 → 0.74 → 0.79
for A = 2…8) — a *third* independent continuum signal: there is no natural number of archetypes, only a
granularity choice. A dedicated comparison (`reports/23b_archetype_compare.md`) tabulated which **axis-corners
survive** at each A:

| corner appears at | A=5 | A=6 | A=7 | A=8 |
|---|:--:|:--:|:--:|:--:|
| metabolic, developmental, suicidality, sleep | yes | yes | yes | yes |
| cognition | – | yes | yes | yes |
| mania | – | – | yes | yes |
| **inflammatory** | – | – | – | **yes** |

**A = 8 was chosen (PI-confirmed)** because it is the *only* resolution that resolves **both** biology
corners (metabolic *and* inflammatory) — the operational biology⊥G story — and it gives the cleanest
narrative: one extreme per measurable specific axis + a low-burden pole. The archetypes are **highly stable**
(cross-seed Tucker congruence 0.999; mean profile SD 0.012), so the choice is interpretability, not fit;
every A is a valid soft basis for the same continuum.

### 4.3 The eight extreme phenotypes (profiles in z-units; share = % with this as dominant)
| # | label | peak axes (z) | share | diagnostic note |
|---|---|---|---|---|
| A0 | low-burden pole | all ≈ −1 | 37% | BP-heavy; few DR |
| A2 | high severity + cognitive burden | cognition 2.3, severity 2.1 | 16% | draws most SZ (760) **and** DR (222 = 40% of DR) |
| A3 | sleep/circadian | sleep 2.6, ↓cognition | 16% | BP-leaning |
| A4 | metabolic | metabolic 3.7 | 13% | mixed (BP 755 · SZ 380 · DR 57) |
| A6 | developmental / early-adversity | developmental 5.1 | 8.5% | BP/SZ-leaning |
| A7 | mania / activation | mania 5.0, ↑sleep | 5.5% | **almost entirely BP** (DR ≈ 2) |
| A5 | inflammatory (+ substance) | inflammatory 6.6, substance 2.4 | 1.9% | rare tail |
| A1 | suicidality | suicidality 8.1, developmental 2.6 | 1.5% | rare tail |

### 4.4 The geometric findings (the elegant part)
- **Severity is the spine, not a corner.** Overall severity (G) forms no archetype at any A — it is the axis
  the whole cloud is stretched along (every archetype sits at some severity level). This is the bifactor
  structure made visible as geometry: severity = "how much," the specific axes = "what kind," largely independent.
- **Substance is absorbed, not a corner.** Substance never anchors an extreme; it appears only as a
  side-loading on the inflammatory corner (A5). Consistent with it being the noisiest, only-2-cohort axis —
  it self-down-weighted, exactly as predicted in M2.0. Both non-corners are *informative* results.
- **Biology⊥G survives into the phenotypes:** distinct metabolic (A4) and inflammatory (A5) corners,
  separate from each other and from the severity spine.

### 4.5 Membership & uncertainty
**75% of patients are blends** (max simplex weight < 0.5; mean normalized entropy 0.67) — they live in the
interior, between extremes. Hard assignment would be a fiction; the decision-region object is the
*distribution* over phenotypes. Explained variance 0.79 (Arm A) / 0.86 (Arm B). Artifacts:
`results/face/m2/{archetypes.parquet, archetype_profiles.csv}`, `reports/23_archetypes.md`,
`docs/figures/23_{scree,profiles,membership}.png`, `23b_compare.png`.

---

## 5. M2.2 — the soft tessellation (coarse overlay)

### 5.1 Why a tessellation, and why Extreme Deconvolution
Given the continuum, the Gaussian mixture is reported as a **soft tessellation** (a discrete decision-region
overlay), **not** biotypes. It must still propagate `S_i`, which a standard GMM cannot (no per-point known
noise). We therefore fit **Extreme Deconvolution** (Bovy et al. 2011): the EM for
`x_i ~ Σ_k π_k N(m_k, V_k + S_i)`. This deconvolves the known per-patient measurement variance, so the
recovered components (m_k, V_k) describe the **underlying noise-free cloud**; prior-dominated coordinates
and DR's absent substance cell (both large `S_i`) self-down-weight — the no-imputation principle, again, at
the coordinate layer.

### 5.2 K and the four regions
BIC over K is a **flat basin** (K=2 200,425 · K=3 199,607 · **K=4 199,325 · K=5 199,307** · K=6 199,439 · …)
— no sharp optimum, continuum-consistent. **K = 4** is reported (the M2.1 uncertainty-aware GMM mode + BIC
plateau onset). The four deconvolved regions: **T0 low-burden** (31%), **T2 severity+metabolic** (32%,
DR/SZ-heavy — the acute/impaired region), **T3 low-metabolic/better-cognition** (25%, BP-leaning), **T1
mania+developmental+sleep** (12%, BP-heavy). **92% of patients have a confident MAP component** (vs 25% for
the finer 8 archetypes): coarse, broad regions assign sharply even on a continuum, whereas the archetype
*corners* expose the blending. The two views are the coarse-label and fine-blend ends of the same continuum,
and their agreement is the robustness argument. Artifacts: `results/face/m2/{tessellation.parquet,
tessellation_profiles.csv}`, `reports/22_tessellation.md`, `docs/figures/22_{bic,profiles,membership}.png`.

---

## 6. M2.4 — validation + the head-to-head vs DSM-5

The four gates are the M2 analogue of M1's adjudication/validation. Applied to both views; diagnosis
validation-only.

### 6.1 Q1 — existence
The honest answer is a **continuum** (M2.1; reinforced by the smooth scree §4.2 and flat BIC §5.2). The
strata layer is therefore a soft representation of a continuum, not natural kinds.

### 6.2 Q2 — not just severity (the headline test)
Per-axis η² of the tessellation partition: **mania 0.45, developmental 0.35**, severity 0.31, metabolic
0.21, sleep 0.19, cognition 0.17 (inflammatory 0.056, suicidality 0.054, substance 0.094 — the rare/noisy
axes). **η²(G) = 0.31 vs mean η²(specifics) = 0.20, with the maximum specific (mania, 0.45) exceeding G.**
The partition is driven by the *specific/biological* axes, not overall severity — a stratification that only
recovered severity tiers would be a re-dressed CGI-S, and this is not that. Archetypes separate even more
strongly on specifics (mean η² 0.32).

### 6.3 Q3 — transdiagnostic (two granularities)
Validated against **both** the 3 cohorts and the **7 DSM-5 subtypes** (BP-I 2,635 / BP-II 2,956 / BP-NOS
661; Schizophrenia 1,692 / Schizoaffective 476 / Schizophreniform 41; MDD 552 — the granularity at which a
data-driven stratum can align with, cut across, or split a diagnosis, and where the schizoaffective boundary
group is most informative). **Adjusted Rand index ≈ 0**: tessellation 0.007 (cohort) / 0.020 (DSM-5);
archetypes 0.060 / 0.046. Cramér's V 0.18–0.28 (weak — informative gradients, not redundancy). The strata
**cut across** diagnosis: every phenotype/region mixes all cohorts and subtypes, with clinically coherent
gradients (mania corner BP-only; severity+cognition corner draws the most SZ and DR).

### 6.4 Q4 — stable & not a missingness artefact 
Seed-stability: tessellation MAP ARI **0.987** (min 0.967); archetype congruence **0.999**. The critical
artefact check: a classifier predicting MAP membership from the **per-axis coverage pattern** achieves
**0.248 accuracy vs a 0.323 majority baseline — a *negative* lift (−0.08)**. Membership is governed by what
patients *are*, not by what was *measured* on them. This is the M2 vindication of the uncertainty-propagation
design (§0.2) — the dominant failure mode of clustering missing clinical data was specifically tested and
ruled out.

### 6.5 Head-to-head vs DSM-5 — the "better description" test (§1.7), and its epistemology
**Design.** We compare the free 4-region mixture to a mixture **constrained to the 7 DSM-5 subtypes** under
the *identical* measurement-error likelihood (`xd_fixed_labels`), and we decompose coordinate variance by
partition. This avoids circularity and isolates a fair descriptive comparison.

**Result.** Free **XD BIC 199,325 (K=4) vs DSM-5 206,016 (7 groups)** — the data's own regions describe the
cloud decisively better, with *fewer* components. Mean coordinate η²: **free 0.209 vs DSM-5 0.048** — the 7
diagnostic boxes explain only ~5% of where patients sit in this biological/cognitive space; the strata ~21%.

**Epistemology (stated, because it governs M4/M5).** "Better than DSM-5" cannot mean "agrees with DSM-5"
(DSM-5 is a consensus taxonomy with weak biological validity, not ground truth; **principled divergence** is
a precondition for value, which Q3's ARI ≈ 0 supplies). It must mean **higher validity on what matters**,
and there are two senses, earned at different milestones: (i) a **better description** (a tighter fit to the
heterogeneity that exists) — testable now, and won here; (ii) **better for decisions / actionable**
(predictive + treatment validity) — the Robins–Guze/Kendler validators that matter, testable only with
outcomes (M4/M5). **M2 establishes (i) only.** The classification-validator families and where each is
tested are recorded in `STRATIFICATION_MODEL.md` §1.7. Artifacts: `reports/24_validation.md`,
`docs/figures/24_validation.png`.

---

## 7. Discussion (extended)

**Why a continuum is the honest *and* the useful answer.** Finding no evidence for well-separated discrete biotypes is a finding with teeth. It
explains the field's biotype non-replication crisis (those results impose K on continua), aligns with the
dimensional turn in psychiatry (RDoC, HiTOP), and changes the actionable object from "which box?" to
"position on continuous axes / proximity to extreme phenotypes" — more honest and more flexible for
individualised decisions. It sharpens M4's question to: *does a continuous, biology-aware coordinate /
archetype representation out-predict the 7 DSM-5 subtypes (and severity) on course and treatment response?*

**Why biology⊥G is again the consequential finding.** That metabolic and inflammatory load form their own
extreme phenotypes, largely independent of the severity spine, is what makes the map worth building: two patients who
look equally ill can be biologically opposite phenotypes — exactly the heterogeneity a precision layer should
exploit, and exactly what a severity-only view is blind to.

**Honesty as a design principle (continued from M1).** The pipeline was built to let the data refuse
discreteness (it did), to flag substance as a non-corner (it did), and to catch missingness-driven artefacts
(it found none, with a *negative* lift). The reported strata are what survived adversarial structure-testing
and validation, not what was assumed — and the actionability claim is explicitly withheld until M4/M5.

**Relation to the literature.** Methodologically the build draws on archetypal analysis (Cutler & Breiman;
Mørup & Hansen) as a continuum-honest alternative to clustering, Extreme Deconvolution (Bovy et al. 2011) for
measurement-error-aware mixtures, and topological/structure diagnostics (Mapper; Hartigan's dip; Tibshirani's
gap) to test cluster tendency before modelling. Scientifically it sits with RDoC/HiTOP dimensionality and the
literature questioning discrete psychiatric biotypes.

---

## 8. Methodological & engineering record (reproducibility)

- **Full-N projection faithfulness.** The fixed-parameter conditional projection reproduces the certified
  joint `f_e` at r ≈ 1.00 (§2.2) — the projection is not a new model, and it is exact on the overlap.
- **Uncertainty propagated at every step** — structure gate over draws; tessellation via XD with known
  `S_i`; archetype weights via draw projection. This through-line is the methodological spine of M2.
- **An AA convergence benchmark** (a process observation worth keeping): explained variance plateaus by
  ~100 iterations (ev 0.514 at iter 100 vs 0.515 at 300), so we cap `max_iter = 120` — a 3× speedup with no
  loss. (An initial over-sized sweep — `max_iter=400` × selection on both arms × many inits — was diagnosed
  as compute-bound and right-sized; lesson logged.)
- **Determinism & runs.** Fixed seeds throughout; long fits run detached under `caffeinate` (the M1 pattern).
  New dependencies installed this session: `diptest` 0.11.0 (dip test), `archetypes` 0.12.2 (PCHA AA);
  `umap-learn`, `hdbscan`, `networkx`, `scikit-learn`, `scipy` already present. **90 data-layer tests green**
  (M2 added no regressions).
- **Engine & pipeline.** One engine `src/face/strata/{scoring, structure, mixture, archetypes, validation}.py`;
  pipeline `scripts/20_prep_coordinates → 21_structure → 22_mixture → 23_archetypes(+23b) → 24_validate →
  26_score`; per-stage `reports/2x_*.md` + `docs/figures/2x_*.png`; configs/seeds inline. Lean stack (no
  DVC/Hydra/MLflow). Every number reproducible from `scripts/` → `reports/`.
- **The per-patient hand-off.** `results/face/patient_strata.parquet` (9,013 × 29): archetype weights (+SD),
  tessellation responsibilities, dominant labels, entropies, `arm` (validation-only). Underlying coordinates
  + draws in `results/face/m2/`.

---

## 9. Limitations & open methods choices (consolidated)

**Limitations.** (1) Internal/descriptive validity only — no outcomes; actionability deferred to M3/M4/M5.
(2) Continuum ⇒ K = 4 and A = 8 are parsimony/interpretability choices (PI-confirmed), not natural numbers;
neighbouring values are equally valid soft bases. (3) The inflammatory (1.9%) and suicidality (1.5%)
archetypes are sparsely-populated long tails of skewed latents, with wide per-patient weight uncertainty.
(4) Heteroscedastic/partial axes (mania partial-for-all; substance 2-cohort, DR-absent; prior-dominated
cognition/inflammatory) are under-resolved for some patients — a correct self-down-weighting, but a coverage
limit. (5) Embeddings (UMAP/PCA) are visualisation-only, never clustering inputs.

**Open methods choices, and how each was resolved.** `A` (archetypes) → 8, PI-confirmed via corner-survival
(§4.2). `K` (tessellation) → 4, via the BIC basin + M2.1 uncertainty mode. `S_i` fidelity → the diagonal
posterior SD is the default; the full draws export supports a covariance-faithful sensitivity arm. G in the
clustering → **both arms** reported (§1.4). Prior-dominated handling → variance-inflation (large `S_i`
self-down-weights), the XD-native treatment. Lead representation if the gate is ambiguous → resolved by the
continuum verdict (archetypes lead; mixture = tessellation).

---

## 10. What M2 hands forward

- **M3 — temporal coherence (V1–V4):** do the coordinates and the phenotype memberships persist? Is a
  stratum more temporally stable than a DSM-5 diagnosis (which itself shifts, e.g. MDD→BP, schizophreniform→
  schizophrenia)?
- **M4 — prognosis (first actionability test):** do the strata add **incremental predictive value beyond
  diagnosis + severity** on course/relapse/hospitalisation/functioning (the *predictive* head-to-head vs
  DSM-5, §6.5)? Natural engine: **Bayesian profile regression** — cluster the coordinates jointly with the
  outcome — a direct extension of the M2 mixture; or a graph/GNN approach once a real graph + outcomes exist
  (§1.3).
- **M5 — treatment:** do the strata **moderate treatment response** (stratum × treatment interaction) — the
  strongest "actionable" test, the point at which knowing a patient's phenotype changes management.
