# RESULTS — empirical findings log (M1, the measurement map)

> Curated, article-grade record of the measurement-model results and their interpretation. Methods of
> record: [`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md); the prior (theory) map: [`PRIOR_ATLAS.md`](PRIOR_ATLAS.md);
> machine-generated per-stage tables: `reports/04_stage*_{report.md,loadings.csv}`. Every number here is
> reproducible from `scripts/01_build_data.py` → `scripts/04_fit.py`. Accumulates per stage (S1 → S5).
>
> **Scope discipline:** staged fits S1–S4 are convergence checkpoints + partial reads; the *reported map*
> is the global fit S5. S1 below is the **continuous backbone** only — read its boundaries (§S1.6).

---

## S1 — continuous core (G + cognition · metabolic · inflammatory · sleep)

**Headline.** On the full FACE V0 sample (N = 9,013, no completeness selection, no imputation), a certified
bifactor measurement model shows a clean **general factor G = transdiagnostic functional burden**, and —
the load-bearing result — **biological load (metabolic, inflammatory) is approximately orthogonal to G**,
whereas cognition and sleep partly track it.

### S1.1 Goal

Three objectives, deliberately narrow (S1 is the "first stable fit", not the final map):
1. Establish the **general factor G** (overall illness burden) on the continuous backbone.
2. Establish the four **continuous specific dimensions**: cognition, metabolic, inflammatory, sleep.
3. **Feasibility gate** — prove the model can be fit on the *entire* sample with *no imputation* and
   *certify*. If it could not, the whole no-imputation/full-sample approach would be in doubt.

### S1.2 Method

- **Data:** all 9,013 V0 patients (BP 6,252 · SZ 2,209 · DR 552); the 68 **continuous** indicators
  homing on G + the four specific factors; skip-logic structural-zeros decoded; **missing left missing**.
- **Model:** **bifactor** — each indicator's variance = (loading on G)² + (loading on its specific
  factor)² + noise. **G anchored only by functioning / global-severity measures** (FAST, EGF, EQ-5D,
  CGI-S) so it denotes "overall impairment/severity", not any single symptom. Independent specifics
  (Φ = I) and simple structure (no cross-loadings) — the minimal first fit; correlations + cross-loadings
  are S2.
- **Estimation:** observed-data likelihood (each patient contributes only observed cells — the
  no-imputation principle) on the **full sample** (no most-complete subsampling → no selection bias). Fit
  via the **marginalized (Woodbury, low-rank) parameterization** (latents integrated out) with NumPyro/JAX
  on the Mac M4 (CPU). The mathematically-equivalent **explicit-latent** parameterization was also run and
  **reproduced the loadings** — the result is not an artifact of one estimator.

### S1.3 Certification — **CERTIFIED**

`N = 9,013` · `J = 68` continuous indicators · `415,531` observed cells ·
**max R-hat 1.010 · min ESS 1,939 · 0 divergences · no Heywood** (gates: R-hat ≤ 1.01, ESS ≥ 400, div = 0).
Source: `reports/04_stage1_report.md`.

### S1.4 Loadings

**G — functional burden (anchors load on G only):**

| indicator | loading | | indicator | loading |
|---|---:|---|---|---:|
| FAST (total) | 1.04 | | CGI-S | 0.54 |
| FAST-25…30 (components) | 0.71–0.80 | | EQ-5D VAS | 0.53 |
| EGF (functioning) | 0.69 | | FAST-28 | 0.47 |
| EQ-5D | 0.58 | | subjective-illness (`lvsbjind`) | 0.01 |

**Specific factors — mean primary home loading:** cognition 0.57 · sleep 0.50 · inflammatory 0.38 ·
metabolic 0.32. (Biology pools are heterogeneous — e.g. BMI 0.92 but many labs 0.2–0.4 — so the factors
are identified but indicator quality varies.)

**Bifactor — mean |loading on G| of each specific factor's indicators (the orthogonality test):**

| domain | mean \|loading on G\| | reading |
|---|---:|---|
| **metabolic** | **0.08** | ≈ independent of overall burden |
| **inflammatory** | **0.07** | ≈ independent of overall burden |
| sleep | 0.22 | moderately tracks burden |
| cognition | 0.27 | moderately tracks burden |

### S1.5 Interpretation (in project context)

1. **Methodological result — the map can be built honestly, at scale.** A bifactor model certifies on
   *all* 9,013 patients with no imputation and no completeness selection. This removes the central
   validity threat (and the specific flaw behind the earlier engine's "no general factor" claim, which
   was fit on a completeness-selected subsample).
2. **G is a clean functional-burden axis.** It is anchored by impairment (FAST), functioning (EGF/EQ-5D),
   and clinician severity (CGI-S) — and *not* by any specific symptom. (`lvsbjind`, subjective illness,
   ≈ 0 — patients' subjective rating barely tracks the objective burden axis.) G is the map's principal
   axis: "how impaired/severe is this patient overall."
3. **Biology is a separate axis from clinical severity — the project's load-bearing premise.** Metabolic
   (0.08) and inflammatory (0.07) load ≈ 0 on G, so biological load varies *independently* of overall
   burden: two patients equally impaired clinically can differ sharply in metabolic/inflammatory load. If
   biology merely rose with severity it would be redundant and useless for stratification; because it is
   roughly orthogonal, **biological strata can capture heterogeneity that severity alone misses** — which
   is the whole point of the precision-psychiatry layer to come. It also means a patient **cannot be
   reduced to one severity number**; the multidimensional profile is the object the stratification layer
   will act on. Cognition (0.27) and sleep (0.22) partly *are* "being unwell" (they move with burden) —
   clinically sensible.

**Theory → data check.** The prior atlas (`PRIOR_ATLAS.md`) *hypothesised* biology on its own
metabolic/inflammatory factors with only a weak *possible* cross-link to G; the data **confirmed** biology
sits off the burden axis. On the continuous backbone, the hybrid model did its job — theory proposed, the
FACE data confirmed.

### S1.6 Boundaries — what S1 does **not** yet show

- **Continuous backbone only.** The symptom/behavioural dimensions — suicidality, the depression/anxiety
  composites (MADRS/QIDS/STAI, modelled as cross-loading "windows"), developmental-risk, mania, substance,
  anhedonia — are **not in S1**. So "biology ⊥ burden" concerns the *functional-burden G*, not symptom dimensions.
- **Independent specifics (Φ = I).** Inter-dimension correlations (e.g. metabolic ↔ inflammatory) are **not
  estimated yet** — that is S2. No between-dimension claims may be drawn from S1.
- **Simple structure.** No cross-loadings yet (S2 frees them).
- **V0 baseline, internal validity only** — no temporal persistence (V1–V4), no external cohort, no
  cross-cohort measurement-invariance test yet.
- It is a **measurement** result — not strata, not prognosis.

### S1.7 Position in the roadmap

```
cohorts → DIMENSIONS (M1, building) → strata (M2) → prognosis (M4) / treatment (M5)
               ▲
          S1 = G + continuous backbone, CERTIFIED   →   S2 cross-loadings + windows + Φ
          → S3 mixed-likelihood (suicidality, developmental) → S4 anhedonia
          → S5 GLOBAL = the reported map → FIML confirmation → adjudication → empirical atlas
```

S1 is the first certified piece of the dimensions layer, and it already delivers the project's
load-bearing hypothesis on the backbone: **biology is its own axis, not a proxy for severity** — exactly
what would make biological strata worth drawing later.

---

## S2 — inter-dimension correlations Φ + the MADRS/QIDS/STAI windows

**Headline.** Deforming the certified S1 backbone into a correlated-factors ESEM — adding the
inter-dimension correlation matrix **Φ**, freeing the **MADRS/QIDS/STAI** composites as cross-loading
windows, and re-checking identification — three results land on the full FACE V0 sample (N = 9,013, no
imputation): (1) the specific dimensions are only **weakly correlated** (mean |Φ off-diagonal| 0.09; the
largest, metabolic↔inflammatory, is **0.20**) — they are genuinely **distinct axes, not one collapsing
factor**; (2) the depression/anxiety composites are **windows onto the general burden axis** — MADRS,
QIDS and STAI load **0.66–0.80 on G** with only minor sleep/cognition side-loadings — empirically
vindicating the decision to model them as cross-loading windows rather than as an 11th "affective"
dimension; (3) the S1 structure is **robust under elaboration** — loadings barely move and **biology
stays orthogonal to G**, so "biology ⊥ burden" was not a bifactor artefact.

### S2.1 Goal

S2 is still a checkpoint, not the reported map. It adds the three pieces S1 deliberately omitted:
1. **Cross-loadings (ESEM):** free the theory-motivated `plausible_cross` cells so indicators can load on
   more than their home factor.
2. **The MADRS/QIDS/STAI windows:** the depression/anxiety composites have **no home factor** (they are
   not a dimension); they enter only as signed cross-loadings onto the axes they clinically touch
   (G, cognition, sleep), and the data place them.
3. **Inter-dimension correlations Φ:** replace S1's independence assumption (Φ = I) with an estimated
   correlation matrix over the specifics, so we can finally ask how the dimensions relate. **G is held
   orthogonal to the specifics** (bifactor identification); the correlated-G variant is reserved for S5.

### S2.2 Method

- **Data / estimator:** identical to S1 — full N = 9,013, observed-cell Gaussian likelihood, no
  imputation, the marginalized (Woodbury) parameterization. Φ enters analytically via
  `Σ = Λ Φ Λᵀ + Ψ = (Λ·chol Φ)(Λ·chol Φ)ᵀ + Ψ`, so the certified S1 kernel runs unchanged.
- **Continuation warm-start (§4.2):** every chain is initialised from the **certified S1 posterior**
  (loadings, residuals), so S2 *deforms* the S1 fit rather than re-deriving it cold.
- **Φ:** `LKJ(η = 2)` over the four specifics; G orthogonal. **Windows:** signed `Normal(0, 0.25)` cells.
- **One identification decision (S2.6):** the *specific↔specific* cross-loadings (all of them
  metabolic↔inflammatory) are **not freed in the reported fit** — they are rotationally aliased with
  Φ_{metab,inflam} and not separately identifiable; **Φ carries that association.** A ridge-guarded
  sensitivity arm that frees them exists in the engine but is not the reported model.

### S2.3 Certification — **CERTIFIED**

`N = 9,013` · `J = 71` continuous indicators (68 + the 3 windows) · `434,765` observed cells ·
**max R-hat 1.010 · min ESS 676 · 0 divergences · no Heywood** · on the Mac M4 (CPU).
Source: `reports/04_stage2_report.md` (+ `_loadings.csv`, `_phi.csv`).

> **Engine corrections (re-certified).** Two engine bugs found while building S3 were fixed and S2
> re-run; **the Φ and loadings below are unchanged to 2 decimals** (the fixes mattered for S3, not for
> S2's numbers): (i) `pm.LKJCorr` returns the Cholesky factor `L`, so the correlation is `Φ = L Lᵀ`
> (the earlier code symmetrized `L`'s lower triangle — indefinite at the 6-factor S3 scale, but
> coincidentally near-identical for S2's 4 specifics); (ii) a **grouped-GEMM Woodbury** (Cholesky once
> per unique observed-pattern, `A` as one BLAS GEMM) replaced the per-patient form — **2.75× faster**,
> verified log-likelihood-identical to the dense computation (diff 0.0000).

### S2.4 Inter-dimension correlations Φ (the new estimand)

Specific block (G orthogonal by construction); mean |off-diagonal| **0.09**:

|  | cognition | metabolic | inflammatory | sleep |
|---|---:|---:|---:|---:|
| **cognition** | 1 | 0.15 | 0.06 | −0.09 |
| **metabolic** | 0.15 | 1 | **0.20** | −0.03 |
| **inflammatory** | 0.06 | 0.20 | 1 | −0.01 |
| **sleep** | −0.09 | −0.03 | −0.01 | 1 |

The dimensions are **weakly correlated** — the strongest link, metabolic↔inflammatory (0.20), is a modest
**immunometabolic** coupling that leaves the two clearly **distinct** (far from collinear), supporting the
hypothesised *split* of candidate 5 into two factors. **Sleep is essentially orthogonal** to the
biological axes (≈ 0) and weakly negative with cognition. That the off-diagonals are small is itself a
result: the specifics are not redundant facets of one super-factor — the multidimensional profile carries
real information for the stratification layer.

### S2.5 The MADRS/QIDS/STAI windows — where depression/anxiety land

| window | → G | → sleep | → cognition |
|---|---:|---:|---:|
| **MADRS** (depression) | **0.80** | 0.14 | −0.06 |
| **QIDS** (depression) | **0.77** | 0.24 | −0.05 |
| **STAI** (anxiety) | **0.66** | 0.21 | — |

All three load **predominantly on G** (the functional-burden axis), with a smaller, sensible **sleep**
side-loading (QIDS/STAI carry sleep content) and a near-zero negative tap on cognition. **Read:**
depression and anxiety severity, as measured here, are largely expressions of **overall illness burden**,
not a separate latent — exactly the methods-doc rationale for treating them as windows, now confirmed by
the data. No 11th affective dimension is needed (its possible emergence is an S5 model-comparison question,
not an assumption).

### S2.6 An identification finding (not just an engineering note)

Freeing the metabolic↔inflammatory mutual cross-loadings **and** a free Φ_{metab,inflam} parameterizes the
*same* covariance two ways → a rotational ridge: chains agreed on every Φ entry except
metabolic↔inflammatory, which swung from −0.16 to +0.46. This is a **statement about the data**: the
metabolic/inflammatory shared variance is identifiable as a **factor correlation** (Φ ≈ 0.20), not as
separable item-level bridges. We therefore report Φ and leave the mutual cross-loadings at zero — the
standard ESEM resolution. (The same freeing also made the full-N fit intractable; the principled model is
also the tractable one.)

### S2.7 Robustness — S1 survives elaboration

Adding Φ + windows barely moves the S1 estimates — evidence the structure is stable, not fragile:

| quantity | S1 | S2 |
|---|---:|---:|
| mean primary loading — cognition / sleep / inflammatory / metabolic | 0.57 / 0.50 / 0.38 / 0.32 | 0.58 / 0.48 / 0.38 / 0.32 |
| mean \|loading on G\| — **metabolic / inflammatory** (biology ⊥ G) | **0.08 / 0.07** | **0.08 / 0.07** |
| mean \|loading on G\| — cognition / sleep | 0.27 / 0.22 | 0.27 / 0.26 |
| G anchors (FAST / EGF / CGI-S) | 1.04 / 0.69 / 0.54 | 0.91 / 0.73 / 0.62 |

**Biology ⊥ G is unchanged** with correlated specifics and the affective windows in the model — the
load-bearing S1 result is not an artefact of the independence/simple-structure constraints.

### S2.8 Boundaries — what S2 does **not** yet show

- **Still the continuous backbone.** Suicidality (binary/count), developmental-risk, mania, substance, and
  anhedonia are **not in S2** (S3–S4). Φ and the windows concern only G + cognition/metabolic/inflammatory/sleep.
- **Reported metabolic↔inflammatory association is Φ only** — item-level bridges between them are not
  claimed (S2.6).
- **V0, internal validity only;** no invariance test, no temporal/external validation yet.
- **Checkpoint, not the map.** Only the global fit **S5** is interpreted/reported; S2's Φ and window
  loadings are provisional reads that the global fit may revise.

### S2.9 Position in the roadmap

```
cohorts → DIMENSIONS (M1, building) → strata (M2) → prognosis (M4) / treatment (M5)
               ▲
   S1 (G + backbone, Φ=I) ✓ → S2 (Φ + windows, cross-loadings) ✓
   → S3 mixed-likelihood (suicidality, developmental) → S4 anhedonia
   → S5 GLOBAL = the reported map → FIML confirmation → adjudication → empirical atlas
```

S2 adds the first picture of **how the dimensions relate** (weakly — they are distinct axes) and shows the
**depression/anxiety instruments are burden windows, not a new axis** — while confirming S1's backbone and
its biology⊥burden headline survive a richer model.

---

## S3 — developmental-risk + mixed-likelihood suicidality

**Headline.** S3 brings the two remaining 3-cohort candidate dimensions onto the map. **Developmental-risk**
(childhood trauma, perinatal, family liability) enters cleanly as a **distinct axis** (S3a, certified).
**Suicidality** — which has essentially no continuous content (its lone continuous item, isf07, cannot
identify it) — is established from its **binary/count ISF ideation & attempt items via a mixed-likelihood
block** (S3b), answering the methods-doc question affirmatively: **non-Gaussian indicators do compose with
the shared Φ** (0 divergences; suicidality solidly identified). The two new dimensions are weakly related to
the rest and most related to **each other** (suicidality~developmental ≈ +0.22) — childhood adversity and
suicidality travel together, as expected.

### S3.1 Compute note (read first)

S3 is the **mixed-likelihood frontier** the methods doc flags as exceeding the Mac ceiling (§3.6): the
7-factor marginalized fit is ~2.7× heavier per step than S2, and S3b's explicit non-Gaussian latents are
heavier still. Per §3.6/§4.3 — *only the global fit S5 is interpreted; S1–S4 are convergence checkpoints* —
the S3 checkpoints are run on a **random N = 4,000 subsample** (realistic missingness, **not**
completeness-selected), with full N reserved for the reported S5 map. Engine: the corrected Φ (`L Lᵀ`) +
grouped-GEMM Woodbury + tree-depth cap 8 + `ta` 0.85 (all validated; see §S2.3).

### S3a — developmental-risk (continuous-anchored) — **CERTIFIED**

6 factors (G + cognition/metabolic/inflammatory/sleep + **developmental-risk**, anchored by CTQ×6,
age-of-onset, WURS, perinatal). `N = 4,000` · **R-hat 1.010 · ESS 832 · 0 div · no Heywood**.
*(Suicidality is deliberately excluded here — its only continuous indicator isf07 is too thin and left the
factor unidentified, R-hat 1.55; it is established at S3b instead.)*

- **Developmental-risk is its own axis:** mean primary loading 0.41; **≈ orthogonal to biology and G**
  (loading on G 0.14; Φ with metabolic/inflammatory ≈ 0), weakly tied to **sleep (+0.16)**.
- **The continuous core is unchanged from S2** (biology ⊥ G: metabolic 0.09 / inflammatory 0.07;
  metabolic~inflammatory 0.21; windows → G 0.81/0.76/0.65) — adding a factor did not disturb the backbone.
- **Resample-stable:** an independent random N = 4,000 draw (seed B) reproduces Φ to **|Δ| ≤ 0.035** and
  loadings to **|Δ| ≤ 0.012** (the §3.6 stability check); and the continuous-core Φ matches the **full-N S2**
  to 2 decimals — subsample and full sample agree.

### S3b — mixed-likelihood suicidality + developmental — **provisional**

7 factors; the binary/ordinal/count indicators (14 binary + 3 ordinal + 1 count) enter via **explicit
latents `f_e = (G, suicidality, developmental)`**, with the 4 continuous specifics marginalized and coupled
to `f_e` through Φ (the conditional decomposition `f_m | f_e`). `N = 4,000` · **0 divergences** ·
R-hat 1.06 · structural ESS 58 — **not fully certified** (see S3.4).

**Suicidality is now solidly identified** (each binary item loads strongly on the logit scale; all
**R-hat 1.00, ESS 0.8–2.3k** — the well-mixed part of the fit):

| ISF indicator | loads on suicidality | on G (bifactor) |
|---|---:|---:|
| isf01–05 (ideation) | **+2.5 to +3.3** | +0.39 to +0.56 |
| isf08 / isf09 (attempt) | +1.8 / +1.9 | +0.00 / +0.28 |
| isf09a (attempt count, NegBin) | +1.54 | +0.22 |
| isf08a (attempt ordinal) | +1.70 | −0.01 |

Suicidal ideation/attempt items load **both** on the suicidality factor **and** on the general burden axis G
(+0.4–0.56) — clinically coherent (suicidality reflects overall illness severity *and* a specific axis).
**Developmental non-Gaussian** indicators: family psychiatric history (mère/père structure +0.30/+0.21) and
childhood trauma (+0.21) carry the factor; perinatal flags are weak.

**Φ — the 6-specific correlations (provisional for the suicidality row):**

| | cog | met | inf | sleep | suic | dev |
|---|---:|---:|---:|---:|---:|---:|
| suicidality | −0.13 | −0.09 | −0.01 | +0.11 | 1 | **+0.22** |
| developmental | −0.06 | −0.02 | −0.01 | +0.16 | +0.22 | 1 |

The strongest new link is **suicidality~developmental +0.22** (childhood adversity ↔ suicidality); both are
otherwise weakly/negatively related to cognition and biology. mean |off-diagonal| 0.10.

### S3.4 Boundaries — what S3 does **not** yet show

- **S3b is provisional, not certified.** The slow mixing (R-hat 1.06, ESS 58) is **not** in the suicidality
  block (which mixes excellently) — it is in the **continuous cross-loadings** (window/bifactor cells) and the
  suicidality~developmental Φ cell, which couple to the explicit `f_e` through the conditional decomposition.
  So the **suicidality loadings are trustworthy**; the **Φ_suicidality correlations and the cross-loading
  refinements are provisional** and will be re-estimated with more compute at the global S5 fit (GPU).
- **Checkpoints on a random subsample** (N = 4,000), not full N — by design (§3.1); the reported map is S5.
- **Anhedonia (S4), the correlated-G variant, FIML confirmation, and adjudication are still ahead.**
- Internal validity only; no invariance/temporal/external validation yet.

### S3.5 Position in the roadmap

```
cohorts → DIMENSIONS (M1, building) → strata (M2) → prognosis (M4) / treatment (M5)
               ▲
   S1 (G + backbone) ✓ → S2 (Φ + windows) ✓ → S3a (+developmental) ✓ · S3b (+suicidality, mixed) ~prov
   → S4 anhedonia → S5 GLOBAL = the reported map (full N) → FIML → adjudication → empirical atlas
```

With S3, the transdiagnostic map now spans **seven dimensions** — G + cognition, metabolic, inflammatory,
sleep, developmental-risk, suicidality — and the mixed-likelihood machinery (the methods doc's hardest
engineering step) is shown to work: binary/count psychopathology indicators compose with the shared
continuous Φ without breaking identification.

---

## S4 — anhedonia: adjudicated **not a distinct dimension** (rejected)

**Headline.** Candidate dimension #4 (anhedonia) is **thin** — one dedicated indicator
(`qids_anhedonia_interest`, QIDS item 13, BP/DR only; SZ has no QIDS). Tested on top of the certified S3a
map, it **does not form a stable, distinct factor**: R-hat **1.54** (ESS 7) at *both* the N=1,500 smoke and
the N=4,000 fair test, with a reflection/collapse instability across chains. When it does form, it is
**redundant with general burden and the depression composite** — its indicator loads **0.61 on G** and the
QIDS-total window loads **0.365** onto it. **Verdict (§6): rejected as a standalone dimension** — its variance
is absorbed by **G + the MADRS/QIDS/STAI windows**. This is the methods-doc's anticipated outcome (anhedonia
"thin; may merge into G or be rejected"), and a *result*, not a failure: theory proposed it, the FACE common
data declined it.

### S4.1 Evidence

- **Non-identification is robust** — R-hat 1.54 / ESS 7 at N=1,500 (tune 400) *and* N=4,000 (tune 1,000),
  0 divergences. Per-chain: the anhedonia loading swings 0.00↔0.56 (3 chains form it, 1 collapses) — a thin
  factor with no stable orientation.
- **Redundant with G + depression** — `qids_anhedonia_interest` loads **0.61 on G** (mostly a general-burden
  signal); the QIDS depression-total window (`qidsr120`) cross-loads **0.365** onto the anhedonia factor. So
  the "anhedonia factor" largely re-measures depression severity, not a separable anhedonic axis.
- **Near-collinear by construction** — the anhedonia anchor (QIDS *item 13*) is a component of the QIDS
  *total* (`qidsr120`, a window), so the two cannot be separated.
- **The rest of the map is undisturbed** — adding anhedonia leaves S3a intact (biology⊥G 0.09/0.07,
  developmental~sleep 0.17, metab~inflam 0.20, windows→G ≈0.79/0.74/0.63); the non-convergence is *isolated*
  to the anhedonia cells. mean |Φ off-diagonal| 0.07.

### S4.2 Consolidated candidate adjudication (after S1–S4)

The continuous/mixed measurement layer is now adjudicated:

| Candidate | Verdict | Basis |
|---|---|---|
| Overall severity → **G** | **confirmed** | clean functional-burden axis (S1) |
| Cognitive flexibility → **cognition** | **confirmed** | S1 |
| Metabolism/immuno → **metabolic** + **inflammatory** | **confirmed (split)** | distinct, Φ 0.20 (S1–S2) |
| Sleep/circadian → **sleep** | **confirmed** | S1; ~orthogonal axis (S2) |
| Neurodevelopment → **developmental-risk** | **confirmed (proxy)** | own axis (S3a) |
| Suicidality | **confirmed (mixed-likelihood)** | binary ISF items (S3b) |
| **Anhedonia** | **rejected / not distinct** | **thin; merges into G + depression (S4)** |
| Impulsivity · Negative symptoms · Sensory | **dropped (pre-modeling)** | no indicators (§2) |

→ **the empirical map carries 7 dimensions** (G + cognition, metabolic, inflammatory, sleep,
developmental-risk, suicidality); the depression/anxiety instruments remain **cross-loading windows**, not a
dimension — confirmed consistently across S2–S4.

### S4.3 Position in the roadmap

```
S1 (G + backbone) ✓ → S2 (Φ + windows) ✓ → S3a (+developmental) ✓ · S3b (+suicidality, mixed) ~prov
   → S4 (anhedonia → rejected) ✓ → S5 GLOBAL = the reported map (full N) → FIML → adjudication → atlas
```

S1–S4 (the staged checkpoints) are complete: the eligible dimension set is adjudicated, the mixed-likelihood
machinery works, and the engine is fast + correct. **Next is S5** — the single global fit on the full sample
that combines all surviving dimensions and yields the *reported* map (loadings, Φ_full, the correlated-G
sensitivity, per-patient scores), followed by FIML confirmation and the prior→posterior atlas.
