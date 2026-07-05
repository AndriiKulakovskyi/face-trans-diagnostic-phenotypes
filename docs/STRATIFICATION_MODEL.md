# M2 — The patient-stratification model: logic and mathematical specification

> Methods-of-record for **Milestone 2 (M2): probabilistic decision regions on the M1 coordinates.**
> This document fixes the *scientific logic* and the *mathematics* of the stratification model and its
> estimation; it is written to feed the manuscript's Methods section directly, and is the **single
> methods/plan of record** for the strata layer. It is the sibling of
> [`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md) (M1). Scope is **stratification of the V0 baseline
> coordinates only** — temporal coherence (M3), prognosis (M4), and treatment (M5) are later milestones,
> named in §10 but not specified here.
>
> *Math is plain-text/unicode so it renders in any markdown viewer and transcribes directly to LaTeX.*
> *Status: PLAN — PI sign-off 2026-06-27 at the discussion gate.*
>
> **Coordinate frame (the M1 map M2 acts on).** M2 is run on the **8-dimension** M1 map: **G (overall burden)**
> ⊥ **7 specific axes** {cognition, **immunometabolic** (cardiometabolic + inflammatory markers on one axis),
> sleep, mania/activation, suicidality, developmental-risk, substance}, with 3 earned cross-loadings
> (CTQ/PSQI → cognition) and **substance pinned orthogonal** to the correlated block (its cross-factor
> correlations are non-identifiable — `with_substance_orthogonal()`; full-N weighted fit converges, R̂ 1.03).
> Two consequences for the coordinate frame: (i) the reliability split is `CONT_AXES = {G, cognition,
> immunometabolic, sleep, mania}`, `EXPL_AXES = {suicidality, developmental_risk, substance}` (Ke = 4); (ii)
> substance carries **no** off-diagonal Φ, so its coordinate is informed by its own SUD items only
> (DR = prior-dominated, not imputed). The cross-seed-stable archetype count is **A = 5** — see
> [STRATA_FINDINGS.md](STRATA_FINDINGS.md).

---

## 1. Scientific logic

### 1.1 Where M2 sits, and what it must not collapse

The project has four layers that must stay distinct (per [`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md) §1.1):

```
diagnostic cohorts → transdiagnostic dimensions → continuous map + A=5 archetypes → prognosis / treatment
   (entry metadata)     (M1 — complete, 8-factor)      (M2 — this; continuum)          (M4 / M5)
```

M1 delivers the **measurement model**: an 8-dimension map and **per-patient coordinates with
uncertainty** on V0. M2 delivers the third arrow: **patient strata** — *recurring, decision-relevant
regions* of that 8-dimensional space. We keep three objects distinct throughout (M1 §1.2):

- **empirical dimension** — a stable, data-supported *axis* of covariance (M1's product);
- **patient stratum** — a recurring, decision-relevant *region* of the coordinate space (M2's product);
- **decision rule** — a map from a patient's stratum (+ uncertainty) to an action (M4/M5).

A stratum is a **region**, not a label glued to a diagnosis. The strata are *discovered from the
coordinates* and only then interpreted against diagnosis and covariates.

### 1.2 Discovery, not theory-imposed grouping

As in M1, M2 is **hybrid**: we do not pre-declare a fixed number of strata or hand-draw their boundaries.
The data decide how many regions the 8-dimensional coordinate cloud supports, where they sit, and how sharp
their boundaries are. The number of strata `K` (and of archetypes `A`, §1.4) is an **estimand**, not a
setting (§5). Just as M1 let the data reject anhedonia, fold the biological markers onto one immunometabolic
axis, and demote neurodevelopment, M2 must
let the data say "there is only severity here," or "there are no discrete clusters at all, only a
continuum" — those null/continuum results are permitted, reportable outcomes (§1.3).

### 1.3 Why "probabilistic decision regions" (the central design constraint)

The M1 coordinates are **not point estimates** — each carries a posterior SD/HDI and a reliability tier,
and they are **wildly heteroscedastic** across dimensions and patients (e.g. severity/immunometabolic/sleep
posterior SD ≈ 0.27–0.29, but cognition 0.44, mania 0.66; 2,506 patients are
prior-dominated on cognition, large sub-populations are prior-dominated on immunometabolic, and mania is
*partial for every patient*). A hard
clustering of the posterior **means** would be dominated by the precisely-measured axes and would place
patients on the basis of **prior-mean artefacts** of their unmeasured axes — manufacturing strata out of
*missingness patterns*, the exact failure M1's no-imputation apparatus exists to prevent.

The honest object is therefore a **soft, probabilistic partition**: every patient receives a *distribution*
over strata (responsibilities / soft membership), and a coordinate's *uncertainty propagates into that
distribution* — an uncertain axis pulls a patient diffusely rather than committing it. "Probabilistic
decision regions" is not stylistic; it is the statistically correct response to the coordinate object M1
produced.

### 1.4 Two complementary views + a structure gate

M2 does **not** assume discrete clusters exist. It first **tests the shape of the cloud** and then reports
**two complementary representations** of probabilistic membership — because clustered vs continuous
structure is itself the scientific question, not a setting.

1. **Structure-discovery gate (§3.1).** Before committing to "strata exist," characterize the *shape* of the
   coordinate cloud — clustered (multimodal), continuum (unimodal, graded), or branched/manifold (flares,
   loops) — using topology (Mapper / persistent homology), cluster-tendency (Hopkins), and modality (dip)
   tests, run **uncertainty-aware** (over M1 draws). This gate is a *reported result* (the M2 analogue of
   M1's "the eligibility map is itself a primary result"), and it decides which view leads the narrative.
2. **Model A — measurement-error mixture (hard probabilistic regions, the primary engine; §3.2).** Soft
   responsibilities over `K` density regions; the natural object for decision rules with thresholds, and the
   one tightly aligned with M1's Bayesian, uncertainty-propagating machinery.
3. **Model B — archetypal analysis (soft archetype membership, co-primary; §3.3).** Each patient as a convex
   blend of a few **extreme phenotypes**; the *continuum-honest* view that does not force discreteness, is
   highly interpretable clinically, and degrades gracefully whatever the structure gate finds.

Reporting **both** A and B is robustness: where they agree, the stratification is solid; where they
disagree, that is itself informative (typically signalling a continuum). The structure gate sets which is
the lead representation.

### 1.5 Load-bearing invariants (inherited + new)

1. **No naive imputation — now at the coordinate layer.** A patient's membership is informed only by their
   *observed/identified* coordinates. Unobserved or structurally-absent coordinates (e.g. substance for DR;
   any prior-dominated cell, if treated as missing) contribute **no term** to the likelihood — integrated
   out, never filled with a prior mean treated as data (§3.2.4). The direct M2 analogue of M1's observed-cell
   likelihood.
2. **Uncertainty must propagate.** The per-patient M1 posterior (SD, and ideally covariance/draws) enters as
   known measurement error (§3.2.2) or via draw-wise refitting (§3.3). Strata may not treat a prior-dominated
   coordinate as equally informative as a well-characterised one.
3. **Diagnosis is metadata.** BP/SZ/DR and DSM `arm` are **never** clustering inputs — validation/
   interpretation only.
4. **Baseline defines; follow-up validates.** Strata are discovered on **V0** coordinates only. Persistence
   across V1–V4 is M3; predictive/decision value is M4.
5. **One global result; only the certified fit is interpreted.** Staged fits (§6) are
   convergence/identification checkpoints; the *reported* strata/archetypes come only from the certified
   global models.

### 1.6 Scope of M2 (the honesty boundary)

At V0 there are **no outcomes** — so "decision-relevant" cannot yet be *demonstrated* against prognosis or
treatment response. M2's honest deliverable is therefore **internal**: discover probabilistic strata /
archetypes and validate them as *stable, transdiagnostic, not-reducible-to-severity, and
not-an-artefact-of-missingness*. The **decision-relevance** of the regions (do they predict outcomes? do
they moderate treatment effect?) is **deferred to M3/M4**, exactly as M1 deferred temporal and external
validity. M2 builds and certifies the regions; the later milestones earn the word "decision." *(Decided at
the M2 planning gate, 2026-06-09.)*

### 1.7 Evaluation philosophy — what "better than DSM-5" means

"Better" must mean **higher validity on things that matter**, *never* **agreement with DSM-5** — DSM-5 is a
consensus taxonomy with weak biological/genetic validity (the motivation for this project), not ground
truth. If the strata merely reproduced the diagnostic groups they would add nothing; **principled
divergence** from DSM-5 is a *precondition* for value (Q3), not a defect. Two senses of "better", earned at
different milestones:

- **Better description (M2, testable now):** the strata fit the actual patient heterogeneity *more tightly*
  than the DSM-5 partition — model evidence (WAIC/BIC) of a free mixture vs one constrained to the 7 DSM-5
  subtypes; variance-explained (η²) / silhouette on the coordinates + held-out indicators.
- **Better for decisions / actionable (M3–M5, the prize):** the strata carry **predictive and treatment
  validity** — the validators that matter. Judged head-to-head vs DSM-5 on **external** variables not used
  to build the strata: for any validator V, compare `V ~ strata` vs `V ~ DSM-5`, and crucially
  `V ~ DSM-5 + strata` vs `V ~ DSM-5` — "better" = strata **predict more and/or add incremental signal
  beyond diagnosis** (+ severity + demographics). The classification-validator families (Robins–Guze /
  Kendler):

  | validator family | example | milestone |
  |---|---|---|
  | antecedent / concurrent | childhood adversity, held-out biology/cognition, demographics | M2 (partial) |
  | **predictive**  | relapse, hospitalization, functioning trajectory, suicide attempt | **M4** |
  | **treatment**  | stratum × treatment-response interaction (changes management) | **M5** |

This operationalizes the project's central bet (**biology⊥G**): biology-aware strata should predict
biologically-relevant course/outcomes (immunometabolic burden, treatment side-effects) that DSM-5 +
severity miss. **M2 may claim only the preconditions (§7 Q1–Q4) + the "tighter description" head-to-head; it
may NOT claim actionability** — that is earned in M3 (persistence) → M4 (prognosis) → M5 (treatment). Two
cautions: **more clusters ≠ better** (overfitting — K is chosen out-of-sample, §5), and the **continuum null
is permitted** (then "better" becomes "a continuous coordinate system out-predicts 7 boxes").

---

## 2. The object M2 acts on

The input is the M1 per-patient coordinate table (`results/face/patient_scores.parquet`), extended to
full-N on all eight dimensions by the M2.0 prep step (§6).

**Eight coordinate axes** (orientation: higher = more burden), each with a posterior **mean**, **SD**, HDI,
observed-indicator count, and a reliability tier (`well ≥3` · `partial 1–2` · `prior-dominated 0`):

| # | dimension | M1 source | full-N as scored by M1? |
|---|---|---|---|
| 0 | overall_severity (G) | continuous, analytic conditional-Gaussian | yes all 9,013 |
| 1 | cognition | continuous | yes (2,506 prior-dominated) |
| 2 | immunometabolic | continuous (cardiometabolic + inflammatory markers) | yes (large sub-population prior-dominated) |
| 3 | sleep | continuous | yes |
| 4 | mania_activation | continuous (2 indicators) | yes (**partial for all**) |
| 5 | suicidality | explicit non-Gaussian `f_e` |  subsample only → **M2.0** |
| 6 | developmental_risk | explicit non-Gaussian `f_e` |  subsample only → **M2.0** |
| 7 | substance | explicit non-Gaussian `f_e` |  subsample only + **2-cohort (no DR)** → **M2.0** |

**Decision (planning gate):** strata/archetypes are built on **all eight** dimensions. This makes M2.0 — the
**full-N projection of the three explicit (non-Gaussian) axes** — a *prerequisite*, not a later extension
(§6).

**The faithful representation is a per-patient distribution, not a point.** M2.0 exports, per patient, not
only `(mean, SD)` but the per-patient posterior **covariance** (or a thinned set of **draws**). This
captures two things the diagonal-SD summary drops: (i) **non-Gaussianity** of the explicit axes (suicidality
is skewed/censored — most patients pile at "no ideation"); (ii) **within-patient cross-dimension posterior
correlation** (shared indicators inform multiple factors; Φ ≠ 0). Both the mixture's `S_i` (§3.2.2) and the
draw-wise archetypal fit (§3.3) consume this; the diagonal `(mean, SD)` summary remains the lightweight
default with the full covariance/draws as the faithfulness arm.

**A validation table** (built alongside, never a clustering input): `patient_id · cohort · arm (DSM-5
subtype) · age · sex · education · site` + follow-up-availability flags (for the M3/M4 hand-off). Used only
for §7 validation and the §8 atlas. **`arm` is the finer DSM-5 taxonomy** — diagnosis is *not* just the 3
cohorts: BP splits into **Bipolar I (2,635) / Bipolar II (2,956) / Bipolar NOS (661)**, SZ into
**Schizophrenia (1,692) / Schizoaffective (476) / Schizophreniform (41)**, DR is **Major depressive
disorder (552)** — **7 subtypes** (schizophreniform n=41 is small → flag). Strata are validated against
diagnosis at **both** granularities (§7 Q3).

**Coverage facts that shape the model:** measurement precision is dimension- and patient-specific; **mania
carries low information for everyone** and immunometabolic/cognition are prior-dominated for large
sub-populations; **substance is undefined for DR**. The models in §3 are chosen precisely so these facts
degrade gracefully (uninformative axes self-down-weight) rather than corrupting the partition.

---

## 3. Mathematical specification — structure discovery + two complementary models

Index patients `i = 1, …, N` (N = 9,013), dimensions `d = 1, …, D` (D = 8), strata `k = 1, …, K`,
archetypes `a = 1, …, A`.

### 3.1 Structure discovery — the gate (cluster vs continuum vs branched)

Before fitting or reporting any partition, characterize the **shape** of the coordinate cloud. This answers
"*are there strata at all?*" with shape evidence, not merely a likelihood ratio, and it is run
**uncertainty-aware** — on M1 posterior draws (or point estimates jittered by `S_i`) so the shape is not an
artefact of treating blobs as points.

- **Cluster tendency** — Hopkins statistic (is the cloud more clustered than uniform/unimodal?).
- **Modality** — Hartigan's dip test on G, on each specific axis, and on informative projections;
  GMM-vs-single-Gaussian by WAIC/BIC (does `K ≥ 2` beat `K = 1`?); the gap statistic / silhouette profile
  over `K`.
- **Topology** — a **Mapper** graph of the cloud (lens = G, density, or a leading archetype coordinate;
  overlapping cover + per-bin clustering) to reveal a single blob vs **branches/flares** (a "Y": a severity
  trunk with biological flares) vs loops; optionally **persistent homology** to quantify connected
  components / loops with persistence.

**The decision rule the gate encodes (a reported result):**

- **Clustered** (multimodal, `K ≥ 2` ≻ 1, clear gaps) → the **mixture** (Model A) leads; archetypes
  complement.
- **Continuum** (unimodal, graded, no gaps/branches) → honest finding "no discrete strata beyond a
  continuum"; the **archetypal/soft** view (Model B) leads, and the mixture is reported as a *soft
  tessellation of a continuum*, **not** as natural kinds.
- **Branched / manifold** (Mapper shows flares/loops) → describe the topology explicitly; report both views;
  consider a manifold-aware refinement.

### 3.2 Model A — measurement-error mixture (hard probabilistic regions; the primary engine)

#### 3.2.1 Generative model (strata = a mixture of regions)

```
π          ~  Dirichlet(α/K · 1)                      [mixing weights; sparse α prunes K — §5]
z_i        ~  Categorical(π)                           [latent stratum of patient i]
θ_i | z_i=k ~  Normal(m_k, Σ_k)                        [θ_i = patient i's TRUE 8-d coordinate]
```

Each stratum `k` is a **region** with centroid `m_k ∈ ℝ^D` and spread `Σ_k` (a `D×D` covariance — an
ellipsoidal/oriented region).

#### 3.2.2 Measurement model (M1's uncertainty enters here — the crux)

M1 gives, per patient, a coordinate estimate `x_i` with a **known** posterior covariance `S_i` (diagonal
`diag(s_i1², …, s_iD²)` by default; full covariance via the §2 export). Treat `x_i` as a noisy reading:

```
x_i | θ_i  ~  Normal(θ_i , S_i)
```

Integrating out the latent true coordinate `θ_i` (a Gaussian convolution) gives the **marginal likelihood
actually fit**:

```
x_i  ~  Σ_k  π_k · Normal( m_k ,  Σ_k + S_i )
```

A finite Gaussian mixture in which **every observation carries its own additive, known covariance `S_i`.**
A wide `S_i` (prior-dominated axes, the always-partial mania axis) makes the patient consistent with many
components → **diffuse responsibility**; tight coordinates place a patient sharply. Each coordinate's
contribution scales by how well it was measured. *(Default likelihood is Student-t per component for
robustness to heavy tails — `Normal` is the `ν → ∞` limit; the convolution `Σ_k + S_i` is applied on the
Gaussian core.)*

#### 3.2.3 Responsibilities — the probabilistic decision regions

```
r_ik  =  π_k · 𝓝(x_i | m_k, Σ_k + S_i)  /  Σ_l π_l · 𝓝(x_i | m_l, Σ_l + S_i)
```

`r_i = (r_i1, …, r_iK)` **is** the probabilistic decision-region membership. We report it in full, plus a
hard MAP label `argmax_k r_ik` and an **assignment-confidence** summary (max responsibility and/or
normalized entropy `H(r_i)/log K`). Boundary patients are flagged, never hidden. Computed post-hoc from
posterior draws, so each `r_i` carries its own posterior.

#### 3.2.4 Missing / structurally-absent coordinates — no imputation

Let `O_i ⊆ {1, …, D}` be the dimensions observed/identified for patient `i`. The likelihood uses only the
observed sub-vector:

```
x_{i,O_i}  ~  Σ_k  π_k · 𝓝( m_{k,O_i} ,  (Σ_k + S_i)_{O_i,O_i} )
```

Unobserved coordinates are marginalized — they neither place the patient nor are filled in. This covers
**structural absence** (substance for every DR patient → DR placed by its other seven axes; substance-defined
strata cannot contain DR, and we say so) and **prior-dominated cells** (`n_obs = 0`): default **mask**
(strict no-imputation); sensitivity arm **variance-inflate** (`s_id²` = prior variance, down-weighted to
≈ 0). Report if they differ.

#### 3.2.5 Priors, identification, estimation

```
m_k  ~ Normal(0, τ²),  τ ~ Half-t          Σ_k = diag(σ_k)·Ω_k·diag(σ_k),  σ_kd ~ Half-t,  Ω_k ~ LKJ(η)
π    ~ Dirichlet(α/K · 1)   [α small ⇒ sparse/overfitted finite mixture — §5]
```

Identification (label-switching, empty/degenerate components): **sparse Dirichlet** pruning (§5); **LKJ +
Half-t** regularization to forbid collapsing (Heywood-type) components; **label-invariant** reporting (the
co-assignment matrix; responsibilities up to permutation) with a relabeling algorithm (ECR / Stephens') for
presentation only; fixed + multi-seed agreement as the identification check (parallel to M1's cross-seed
Tucker φ). The discrete `z_i` are **marginalized analytically**, so the model is continuous and NUTS-friendly
— fit with **NumPyro/JAX** on the Mac (tiny next to M1: `O(K·D²)` globals, no per-patient latent funnel).

### 3.3 Model B — archetypal analysis (soft archetype membership; the co-primary view)

Represent each patient as a convex combination of `A` **extreme phenotypes (archetypes)**, each archetype
itself a convex combination of patients:

```
minimize  Σ_i ‖ x_i − Σ_a w_ia z_a ‖²        s.t.   z_a = Σ_j b_ja x_j ,
          w_i on the simplex (Σ_a w_ia = 1, w_ia ≥ 0) ,   b_·a on the simplex
```

The **simplex weights `w_i`** are the soft membership — the "soft-archetype" probabilistic-region view
(patient = a blend of extremes, e.g. "0.7 high-immunometabolic + 0.3 cognitive-impairment archetype").

- **Uncertainty propagation** — fit AA over M1 posterior **draws** (default; reuses the §2 export) → a
  *posterior* over archetypes `z_a` and over each `w_i` → uncertainty-aware soft membership. (A probabilistic
  AA formulation is an optional alternative.)
- **Missingness** — masked / weighted-AA over observed cells only (no imputation; substance-for-DR masked),
  consistent with §1.5.
- **Number of archetypes `A`** — scree/elbow on explained variance, stability across resamples,
  interpretability (§5). Archetypes are *extreme points*, so `A` is typically small (≈ 3–6).
- **Out-of-sample** — a new patient is projected onto the fixed archetypes (simplex least-squares) → soft
  membership for M4/M5 / external cohorts.

### 3.4 How the two views relate

- **Mixture (A):** "which region, with what probability?" — discrete decision regions / density modes;
  natural for threshold decision rules. **Archetypes (B):** "what blend of extremes?" — graded soft
  membership; natural when structure is continuous.
- **They cross-check.** Under clustered structure, archetypes sit near cluster cores and `A ≈ K` tells the
  same story; under a continuum, the mixture is a soft tessellation while archetypes describe the spanning
  extremes. **Agreement = robustness; disagreement = a continuum signal.** Both are reported; the §3.1 gate
  sets the lead.

### 3.5 The role of G — two arms (severity×profile vs pure profile)

How the general factor **G** (overall functional burden) enters the clustering changes what a *stratum
means*. We run **both** arms and compare — the comparison is itself a result *(decided at the planning gate,
2026-06-09)*:

- **Arm A — G in (all 8 axes): "severity × profile."** Strata may differ in overall burden *and* profile
  shape (e.g. "severe, high-immunometabolic" vs "mild, high-immunometabolic"). The honest full-coordinate
  object; captures severity×profile interactions. **Risk:** G is the dominant, best-measured, highest-variance
  axis (and the depression/anxiety windows load on it), so it can dominate → strata collapse toward **severity
  tiers** (the Q2 failure mode).
- **Arm B — G out (7 specifics): "pure profile."** Strata defined by profile shape *independent of overall
  severity* (e.g. "high-immunometabolic" at any severity) — targeting the value proposition (heterogeneity
  severity misses). **Because M1's specific coordinates come from the bifactor identification (orthogonal to
  G by construction), dropping G from the feature set *is* the G-residualized view** — no ad-hoc regression
  needed; the specifics already are the G-removed profile. (The immunometabolic axis is already ≈⊥G so it
  barely moves; cognition/sleep — which partly track G — change most.)

**What the comparison yields.** Same strata under A and B → severity is not driving the partition; the
profile structure is robust (a strong result). Different → we quantify how much of the stratification is
severity vs profile and report **both** maps (a severity×profile map and a pure-phenotype map). For **M4**,
Arm B is the cleaner substrate for showing prognostic value *beyond* diagnosis + G (severity is a known
prognosticator; a pure-profile stratum that still predicts is the strongest claim).

---

## 4. Acceptance gates (per stage and for the milestone)

A modeling stage **certifies** only if: `R-hat ≤ 1.01`, `ESS ≥ 400`, `0` divergences (mixture); no
degenerate/empty-collapse components beyond those the sparsity prior intentionally prunes; stable component/
archetype recovery across seeds (no unexplained switching); posterior-predictive (mixture) / reconstruction
(AA) not grossly violated.

The **milestone** locks only if: (i) the **structure-discovery shape** is reported and the lead view chosen
by it (§3.1); and (ii) the strata/archetypes pass the four scientific gates (§7) — **existence** (mixture ≻
single Gaussian; `K`/`A` data-supported), **not-just-severity** (Q2), **transdiagnostic** (Q3), **stability
+ not-an-artefact** (Q4) — each passed or documented-partial, PI-signed.

---

## 5. Choosing K (regions) and A (archetypes) — both are estimands

Neither count is set by hand:

- **K (mixture) — sparse / overfitted finite mixture** (Malsiner-Walli): generous `K_max` (≈ 12–15), small
  `α` (`α/K ≪ 1`); superfluous components empty out, `K_eff` = components retaining mass in the posterior.
  (A Dirichlet-process mixture is the equivalent nonparametric alternative.) Cross-checks: **BIC** over
  fixed `K = 2 … K_max` (the classical/XD EM arm) + **stability** across seeds/resamples.
- **A (archetypes)** — explained-variance scree/elbow, stability of archetypes across resamples, and
  interpretability; reported with a sensitivity table over neighbouring `A`.
- **The null/continuum is allowed.** If evidence favours `K = 1`, or separation purely along G, or the §3.1
  gate says "continuum," M2 reports "no transdiagnostic strata beyond severity / a graded continuum" — a
  legitimate scientific outcome, with archetypes describing the continuum's extremes.

---

## 6. Estimation strategy — staged continuation (M1's discipline)

The dimension set is fixed at 8 (planning decision); staging deforms **model complexity**, with a discussion
gate after each stage. **Only the certified global models (M2.2 full-Σ mixture + M2.3 archetypes) are
interpreted** — intermediate rungs (diagonal-only, a provisional `K`) are checkpoints, never reported.

**Pipeline (data flow).** One engine (`src/face/strata/`) drives a linear, resumable pipeline; each stage
consumes the prior stage's artifact, emits a `reports/2x_*.md` + figures, and stops at a discussion gate.

```
   M1 artifacts                       M2 engine — src/face/strata/                    stage artifacts
─────────────────────         ────────────────────────────────────────         ───────────────────────────
patient_scores.parquet ─┐
s5_cert9_s1/idata.nc   ─┼─►  20 prep    (scoring, structure.prepare)      ─►  coordinates_full.parquet
data layer (covariates)─┘       full-N f_e projection · cov/draws · labels     validation_table.parquet
                                         │
                                         ▼
                            21 structure (Mapper · dip · Hopkins, on draws) ─►  reports/21 + shape figs
                                         │   shape verdict → chooses lead view
                         ┌───────────────┴───────────────┐
                         ▼                                ▼
             22 mixture  (Model A)               23 archetypes (Model B)
                diag → sparseK → fullΣ              AA over draws · select A
                Arm A (8) ∥ Arm B (7)              soft w_i · out-of-sample    ─►  reports/22,23 + figs
                         └───────────────┬───────────────┘
                                         ▼
                            24 validate  (Q1–Q4, both views · robustness)   ─►  reports/24 + figs
                                         │
                                         ▼
                            25 atlas  ·  26 score   ─►  patient_strata.parquet · STRATA_ATLAS.md · figures
```

| Stage | Adds | Gate before advancing |
|---|---|---|
| **M2.0 — prep** | full-N projection of the 3 explicit axes (suic/dev/substance) for all 9,013; **per-patient covariance/draws export** (§2); validation table | all 8 dims full-N with uncertainty (cov/draws); no-imputation handling of substance-in-DR set up |
| **M2.1 — structure gate** | Mapper / persistent homology + Hopkins + dip + GMM-vs-1, **uncertainty-aware** (§3.1) | the cloud-shape verdict (clustered / continuum / branched) is **reported**; the lead view is chosen |
| **M2.2 — mixture (Model A)** | measurement-error mixture; internal continuation **diagonal Σ → sparse-K → full Σ** | certifies; `K_eff` data-supported; mixture ≻ single-Gaussian; classical EM/BIC triangulates |
| **M2.3 — archetypes (Model B)** | archetypal analysis over draws; select `A`; soft membership + out-of-sample projection | `A` selected; archetypes stable across resamples; AA reconstruction adequate |
| **M2.4 — validation** | Q1–Q4 (§7) on **both** views + missingness-artefact + per-cohort invariance + graph/density shape-checks | the four gates pass or documented-partial |
| **M2.5 — atlas + score** | per-patient responsibilities **and** archetype weights → parquet; the stratum/archetype atlas; findings doc; PI gate | hand-off object + deliverable; M2 locked |

**M2.0 detail (the prerequisite).** The 5 continuous axes are already full-N (analytic conditional-Gaussian,
M1 §7). The 3 explicit axes are non-Gaussian latents `f_e`; full-N projection scores each patient's posterior
on `f_e` **conditional on the fixed 8-dim measurement parameters** (loadings, cutpoints/thresholds,
dispersion, Φ) given that patient's observed binary/ordinal/count indicators — a per-patient conditional
posterior (Laplace/variational or short conditional MCMC under fixed Λ,θ), **not** a re-fit of M1. DR
contributes no `substance` cell. Output extends `patient_scores.parquet` to full-N × 8 with (mean, SD, HDI,
covariance/draws, n_obs, reliability) — re-using `src/face/scoring` + `continuous_core.prepare_mixed`.

---

## 7. Validation — the four gates (the M2 §6/§8), on both views

Each gate restates an M1 design value at the strata layer; diagnosis/covariates are validation-only. The
battery is applied to the mixture and, where applicable, to the archetype representation.

- **Q1 — Existence & selection.** The §3.1 structure verdict is reported; the mixture decisively beats a
  single Gaussian (WAIC); `K_eff` / `A` are data-supported (§5) and stable. *Else:* report continuum/null.
- **Q2 — Not just severity (the headline test, descendant of biology⊥G).** Fit a **G-only 1-D mixture**;
  the full 8-D mixture must beat it decisively (WAIC). Decompose between-stratum (and between-archetype)
  separation **per axis**: regions must separate on the **specific / biological** axes, not only on G.
  Report how much of the partition is explained by G alone vs the specifics, with per-stratum/archetype
  profiles; the explicit test is the §3.5 **Arm A (G-in) vs Arm B (G-out)** comparison. *A stratification
  that only recovers mild/moderate/severe adds nothing beyond CGI-S and fails Q2.*
- **Q3 — Transdiagnostic (not just diagnosis), at TWO granularities.** Concordance with diagnosis is
  measured against **both** the 3 cohorts (BP/SZ/DR) **and the 7 DSM-5 subtypes** (`arm`): `ARI`, η² /
  Cramér's V **low**, within-stratum composition **mixes**. The subtype view is the sharper test and asks
  what a coarse label cannot: does a stratum **align with** a subtype (e.g. BP-I vs BP-II separating on the
  mania / suicidality axes), **cut across** subtypes (transdiagnostic grouping), or **split** one
  (within-DSM heterogeneity)? Where does the **schizoaffective** boundary group (mood × psychosis) land
  relative to SZ and BP? *If strata ≈ cohorts or ≈ subtypes, the map added nothing.*
- **Q4 — Stable & not an artefact.** (a) **Stability:** co-assignment (consensus) matrix across **seeds +
  bootstrap**, mean `ARI ≥` a stated guideline (≈ 0.6–0.7), and under **LOCO**, **site cluster-bootstrap**,
  **1/n_cohort weighting** (reusing M1's `08_robustness` scaffolding); archetype stability across resamples.
  (b) **Not-an-artefact-of-missingness:** stratum/archetype membership **not predictable from the
  coverage/reliability pattern** (a coverage→membership classifier is weak); the partition reproduces on a
  higher-coverage subset; the §3.2.2 measurement-error model + §3.3 masking are the primary defense.
  (c) **Per-cohort invariance:** per-cohort fits vs the joint — does the region structure hold within
  BP/SZ/DR? Documented partials allowed (parallel to M1's invariance partials).
- **Uncertainty honesty.** Report the assignment-entropy distribution and confidently- vs
  ambiguously-assigned fractions; every patient carries soft membership, not just a hard label.
- **Head-to-head vs DSM-5 — the "better description" test (§1.7).** Does a mixture *free* to find groups
  beat one *constrained* to the 7 DSM-5 subtypes (WAIC/BIC), and do the strata explain more coordinate /
  held-out-indicator variance (η², silhouette) than the DSM-5 partition? *(The validators that matter —
  predictive + treatment, head-to-head vs DSM-5 — are M4/M5 per §1.7; M2 establishes only this descriptive
  head-to-head plus the preconditions Q1–Q4.)*

**We will claim** (if supported): an internally-validated set of **probabilistic transdiagnostic strata +
archetypes** on the M1 coordinates, with per-patient soft membership + uncertainty, **not reducible** to
severity, diagnosis, site, or missingness. **We will not claim** (in M2): "biotypes"/natural kinds; any
predictive, prognostic, temporal, or treatment value (M3–M5); external validity. "Converged" ≠ validated.

---

## 8. The deliverable — per-patient membership + the atlas

**Per-patient hand-off** (`results/face/patient_strata.parquet`, gitignored): `patient_id · cohort ·
r_i1…r_iK (mixture soft membership) · MAP label · assignment confidence (max-r, entropy) · w_i1…w_iA
(archetype soft membership) · per-region/archetype posterior summaries`. Uncertainty preserved (optionally as
draws) so M3/M4 can propagate it.

**The atlas** (`docs/STRATA_ATLAS.md` + figures) — the scientific story, mirroring M1's dimension atlas.
Reports: the **structure-discovery shape verdict** (§3.1) with the Mapper figure; for each **stratum** —
size, **diagnostic composition at both granularities** (3 cohorts + 7 DSM-5 subtypes), the **mean 8-D profile with uncertainty** + defining
axes, separation, a **data-driven** label (e.g. "high-immunometabolic / average-G" — *not* theory-imposed),
centroid, coverage caveats (substance undefined for DR, mania low-information throughout); for each
**archetype** — its extreme-phenotype profile and the population's simplex spread; and the **A-vs-K
agreement** (§3.4) + classical-triangulation agreement + the `K`/`A` support.

### Figures (methods + results) — every claim gets a picture

Visuals are a first-class deliverable, generated by the stage scripts into `docs/figures/`. Embeddings are
**visualization-only, never a clustering input** (clustering is on the 8-D coordinates; UMAP distorts
density/distance, so a PCA companion guards against illusory clusters).

*Methods-explainer:*
- **F-M1 Pipeline / data-flow** schematic (§6).
- **F-M2 The measurement-error idea** — a 2-D toy: clustering *points* vs clustering *blobs* (how a wide
  `S_i` on a prior-dominated axis spreads a patient's responsibility instead of forcing assignment).
- **F-M3 Mapper graph** of the coordinate cloud (the §3.1 structure verdict), nodes coloured by G / a
  biological axis / cohort.
- **F-M4 Archetype simplex** schematic (what "a blend of extreme phenotypes" means).

*Results:*
- **F-R1 2-D embedding — UMAP (primary, local) + PCA (companion, variance-faithful)**, paneled by MAP
  stratum, **cohort** (Q3 mixing), **G** (Q2 — are strata just severity tiers?), and each biological axis.
  (PHATE optional if the structure gate finds a continuum/trajectory.)
- **F-R2 Stratum profile heatmap** — strata × 8 dims (posterior-mean coordinate + uncertainty) — the atlas
  core, mirroring M1's `empirical_atlas`.
- **F-R3 Archetype profiles** — archetypes × 8 dims (radar/heatmap) + the population's simplex spread.
- **F-R4 Diagnostic composition** — per-stratum stacked bars at **both** granularities (3 cohorts and the 7
  DSM-5 subtypes), highlighting where schizoaffective / BP-II land (Q3).
- **F-R5 Assignment uncertainty** — entropy distribution; embedding coloured by max-responsibility (boundary
  patients).
- **F-R6 Stability / consensus** — co-assignment heatmap across seeds + bootstrap (Q4).
- **F-R7 Not-just-severity** — strata in the G × biological-axis plane (biological spread at fixed G) + the
  WAIC G-only-vs-full bar (Q2).
- **F-R8 K / A selection** — WAIC/BIC over `K`; AA scree over `A`.
- **F-R9 G-arm comparison** — Arm A vs Arm B strata (cross-tab + aligned embedding): how much of the
  partition is severity vs profile (§3.5).

**Paper-facing synthesis** (`docs/STRATA_FINDINGS.md`): findings + discussion, the M2 sibling of
[`M1_FINDINGS.md`](M1_FINDINGS.md); every numeric claim backed by a committed `reports/2x_*.md`.

---

## 9. Open methods choices (flagged for the PI)

Defaults are set above; confirm or overrule before M2.1:

- **`K_max` and sparsity `α`** · **sparse-finite vs Dirichlet-process** K · **number of archetypes `A`**
  range.
- **Reported `Σ_k` geometry** — full (oriented, default) vs diagonal (the M2.2 checkpoint).
- **`S_i` fidelity** — diagonal (mean, SD) default vs full per-patient covariance / draws (the faithfulness
  arm, §2).
- **Prior-dominated handling** — mask (default) vs variance-inflate; whether they materially differ.
- **G in the clustering** — **decided (planning gate): run both arms** and compare (§3.5) — Arm A (all 8,
  severity×profile) ∥ Arm B (7 specifics = the bifactor G-residualized pure-profile view); the lead arm is
  chosen by what each yields, reported in the atlas (F-R9).
- **Mapper config** — lens(es), cover resolution/overlap, per-bin clusterer (the structure-gate knobs).
- **Visualization embedding** — UMAP (primary, local structure) + PCA (companion, variance-faithful),
  PHATE optional for continuum/trajectory views; **viz-only, never a clustering input** (§8 figures).
- **Reconciliation rule** — which view leads if the §3.1 gate is ambiguous or A and K disagree.
- **Stability / ARI thresholds** and the **relabeling** algorithm (ECR vs Stephens').
- **Whether mania participates meaningfully** — predicted to self-down-weight (partial for all); reported,
  not forced.

---

## 10. Position in the roadmap (named, not specified here)

M1 (the map) → **M2 strata** (this document — probabilistic decision regions on the V0 coordinates) →
**M3 temporal coherence** (do the strata + scores persist across V1–V4?) → **M4 prognosis** (do strata add
incremental predictive value beyond diagnosis + G? — the first test of "decision-relevant"; the natural
extension is **Bayesian profile regression**, clustering coordinates *jointly with* the outcome so strata are
decision-relevant by construction) → **M5 treatment** (do strata moderate treatment response? target-trial
emulation, only if the data support it). Each is a separate milestone with its own gate.

---

## 11. Repository and reproducibility (lean — same conventions as M1)

```
src/face/strata/{structure, mixture, archetypes, scoring, validation, viz}.py   # one engine: 3 views + Q-battery + figures
scripts/  20_prep_coordinates → 21_structure → 22_mixture → 23_archetypes → 24_validate → 25_atlas → 26_score   (each stage emits its figures → docs/figures/)
configs/  strata.yaml          # K_max, α, A range, Σ geometry, S_i fidelity, prior-dominated handling, Mapper/embedding cfg, seeds
results/face/patient_strata.parquet                                          # per-patient soft membership (gitignored)
reports/  2x_*.md per stage (+ figures in docs/figures/)
docs/     STRATIFICATION_MODEL.md (this) · STRATA_FINDINGS.md · STRATA_ATLAS.md
```

Reuse M1 infrastructure: `src/face/scoring` (conditional scores, reliability flags), the 8-factor copula fit,
and the robustness scaffolding (`scripts/08_robustness.py`,
`results/face/robust_cache`). Lean stack — no DVC/Hydra/MLflow; YAML configs; Parquet model-ready tables;
fixed seeds; long fits run detached under `caffeinate` with a per-seed cache (the M1 pattern). Every number
reproducible from `scripts/` → `reports/`.
