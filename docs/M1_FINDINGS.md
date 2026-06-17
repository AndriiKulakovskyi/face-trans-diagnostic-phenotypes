# M1 — Findings & Discussion

> **The paper-facing synthesis of Milestone 1**: what we did, what we observed, what we found, and what it
> means. This is the canonical *findings + discussion* record for traceability, PI review, and the manuscript.
> Companions: methods of record → [`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md); per-candidate verdict →
> [`ADJUDICATION.md`](ADJUDICATION.md); per-stage development detail → [`RESULTS.md`](RESULTS.md); data
> contract → [`DATA.md`](DATA.md); live status → [`STATE.md`](STATE.md). Every numeric claim here is backed
> by a committed `reports/NN_*.md`. Updated 2026-06-09.

---

## 1. Summary

On the harmonized 3-cohort FACE **V0 baseline** (N = 9,013 = BP 6,252 · SZ 2,209 · DR 552), one global,
missingness-aware Bayesian sparse **bifactor/ESEM** model — estimated from each patient's *observed cells
only* (no imputation) — yields a **certified 9-dimension transdiagnostic map**: a general factor **G
(functional burden)** plus eight specific axes — **cognition, metabolic, inflammatory, sleep,
developmental-risk, suicidality, mania, substance**. The map is hardened end-to-end (estimator/prior
robustness, measurement invariance, resample robustness, absolute-fit PPC across both likelihood blocks),
certified at the largest N that mixes (cross-seed Tucker φ 0.993), and projected to per-patient coordinates
with uncertainty and reliability flags. The headline scientific finding is that **biology is the least
severity-entangled domain** — metabolic and inflammatory burden are nearly orthogonal to the general
functional-impairment factor, whereas cognition and sleep partly track it.

---

## 2. The empirical map

| # | Dimension | Block | Anchoring indicators | Mean primary \|λ\| |
|---|---|---|---|---|
| 0 | **G — functional burden** | explicit (Gaussian) | FAST 0.90, EGF 0.73, EQ-5D; **no symptom content** | — (general) |
| 1 | cognition | marginalized | executive/processing-speed tasks | 0.57 |
| 2 | metabolic | marginalized | BMI, lipids, glycemia, BP | 0.32 |
| 3 | inflammatory | marginalized | CRP, leukocyte subsets | 0.39 |
| 4 | sleep | marginalized | sleep/circadian items | 0.48 |
| 5 | developmental-risk | explicit | childhood adversity (CTQ), birth/parental history | 0.42 |
| 6 | suicidality | explicit (binary) | ISF ideation/attempt items (+2.7…+3.4 logit) | strong |
| 7 | **mania** | marginalized | YMRS (0.57), Altman (0.76 in BP) | 0.49–0.76 |
| 8 | **substance** | explicit (mixed) | alcohol/cannabis lifetime SUD, nicotine | +0.38…+0.83 |

Inter-dimension correlations Φ are **weak** (mean \|off-diagonal\| ≈ 0.10): the specifics are genuinely
distinct axes, not a single collapsed factor. Depression/anxiety (MADRS/QIDS/STAI) are **not a dimension** —
they load 0.66–0.80 on **G** as cross-loading "windows" (burden surfaces, no separable affective factor).

---

## 3. Principal findings

Each finding is stated as *observation → result → interpretation*.

### F1 — A clean general factor is functional burden, not a "p-factor"
G is anchored only by **functioning/severity** items (FAST, EGF, EQ-5D) and carries **no symptom content**
(`lvsbjind` ≈ 0). G is therefore best read as a **transdiagnostic impairment/distress axis**, not a latent
liability to psychopathology — a deliberately conservative reading that avoids the bifactor "p-factor"
over-claim.

### F2 — Biology ⊥ G (the load-bearing refinement)
Under a sensitivity arm that *frees* G to correlate with the specifics (correlated-G, §3.1), G correlates
**+0.06 with inflammatory** and **+0.14 with metabolic**, versus **+0.39 cognition** and **+0.44 sleep**.
**Result:** metabolic and inflammatory burden are the **least severity-entangled** domains — a patient's
metabolic/immune load is almost independent of how impaired they are overall, while cognitive and sleep
burden partly track impairment. **Interpretation:** biological risk is carried on axes the clinical severity
picture does not see — exactly the kind of orthogonal signal a stratification (M2) can exploit. *(The clean
continuous-backbone estimate, metabolic~G 0.12–0.14, supersedes an earlier provisional mixed-fit read of
0.28; both agree on the ordering.)*

### F3 — Theory's single "biology" candidate splits into two
The prior ontology posited one metabolic/immuno candidate; the data **split** it into **metabolic** and
**inflammatory** factors that correlate only Φ ≈ 0.19 — not collinear. The map earns a finer biological
resolution than theory specified.

### F4 — Developmental-risk is a *proxy*, not measured neurodevelopment
The developmental axis is real and distinct (loading 0.42) but is anchored by **childhood adversity / birth
/ parental** indicators — it indexes **early-adversity/liability**, and is named and interpreted as a
*proxy*, not as measured neurodevelopment.

### F5 — Suicidality composes as a mixed-likelihood axis
Binary ISF ideation/attempt items load **+2.7…+3.4 on the logit scale** and compose with the shared Φ under
the proper Bernoulli likelihood — demonstrating that non-Gaussian psychopathology indicators integrate into
the same correlated-factor space as the continuous biology without breaking identification.

### F6 — The map is larger than seven: mania and substance are real axes
Mania and substance were **added after the original ten**, once their indicators were ingested into the harmonized dataset; the joint refit then **confirmed
both**: **mania** (YMRS/Altman, primary \|λ\| 0.49–0.76, \|G\| 0.15) and **substance** (alcohol/cannabis SUD
+ nicotine under the proper Bernoulli/NegBinom likelihoods, \|G\| 0.13). Re-certifying the **joint 9-dim**
model integrated them under one shared Φ (R-hat ≤ 1.04, ESS ≥ 112, 0 div, cross-seed Tucker φ 0.993).
**Interpretation:** manic activation and substance use are distinct, low-G transdiagnostic axes — not
reducible to severity, and worth carrying into stratification.

### F7 — Measurement invariance: largely invariant, with documented partials
The loadings are **largely invariant** across BP/SZ/DR. The **honest exceptions**, each documented rather
than hidden:
- **G** — partial BP–SZ (SZ lacks the FAST anchor; G re-anchored on CGI-S/EGF/EQ-5D there).
- **inflammatory in DR** — neutrophils load ≈ 0 in DR while eosinophils dominate (a real biological
  re-weighting of the immune axis in the DR cohort).
- **mania-Altman in DR** — YMRS holds BP–DR (0.57/0.41) but the **self-rated Altman does not transfer**
  (0.76 → 0.10; Tucker φ 0.764). Self-reported manic activation is a near-floor signal in a
  DR cohort; clinician-rated YMRS carries mania there.
- **substance is invariant BP–SZ** (Tucker φ **0.997**) and is declared a **2-cohort axis** (its
  alcohol/cannabis SUD are BP/SZ-only; not claimed for DR).

### F8 — The structure is not an artefact of estimator, prior, resampling, cohort, or site
- **Not a prior/estimator artefact** (§5): a flat-prior refit reproduces the loadings/Φ **exactly** (Tucker
  φ = 1.00); WAIC decisively prefers the bifactor over unidimensional and correlated-no-G alternatives.
- **Resample-robust** (§8): under leave-one-cohort-out, diagnosis-balanced subsampling, site cluster-
  bootstrap, and 1/n_cohort weighting, the minimum Tucker φ vs the full-sample reference is **0.958**.
  The map is not driven by cohort imbalance, any single cohort, or recruitment-site clustering.

### F9 — Absolute fit holds across both likelihood blocks — with one localized item caveat
Posterior-predictive checks cover both blocks: continuous **SRMR ≈ 0.07**, and the **non-Gaussian block
reproduces 21/22 indicators'** observed endorsement rates / means within the 90% predictive interval
(Bayesian p ≈ 0.5). The **single exception is `isf09a` (suicide-attempt count)** — a 90.8%-zero hurdle count
that a plain NegBinom over-predicts in the high-suicidality tail. This is an **item-level** mis-fit, *not* a
factor-level one: the suicidality dimension is carried by its 7 binary ISF items, all of which reproduce.

### F10 — Per-patient coordinates carry honest uncertainty
All 9,013 patients are scored on the six continuous-anchored dimensions (and the explicit dimensions on the
fit subsample), each coordinate carrying a posterior **mean / SD / HDI** plus a **reliability tier**
(well-characterised / partial / prior-dominated) by observed-indicator count. The flags correctly expose
coverage: cognition is prior-dominated for the 2,506 patients without cognitive testing; **mania is
*partial* for everyone** (only 2 indicators). Downstream strata must propagate this uncertainty, not treat
all coordinates as equally measured.

---

## 4. Methodological contributions & work log

The methodology is itself a contribution, and the path matters for reproducibility:

- **Full-sample, no-imputation estimation via a marginalized Woodbury parameterization.** Integrating the
  Gaussian latents out analytically (matrix-determinant lemma + Woodbury, grouped per observed-missingness
  pattern) turns a funnel-prone explicit-latent model into one that **certifies on the full N = 9,013 on a
  Mac (M4 Pro, 24 GB) — no GPU**. This is what lets the project *refuse* the completeness-selected subsample
  that would otherwise bias the map.
- **In-engine confirmation replaces standalone FIML.** Because the marginalized Bayesian model and FIML
  optimize the *same* observed-data objective (§3.5), a separate FIML arm is redundant; confirmation is done
  *in-engine* (flat-prior refit + posterior-predictive checks + WAIC). A trial standalone semopy FIML was
  intractable and produced inconsistent fit indices on the high-missingness backbone — documented and
  dropped.
- **A reparameterization ladder, not hardware, resolved the mixing frontier.** The S3+ mixed-likelihood
  stages were weak-identification *ridges* (slow mixing with 0 divergences), fixed by targeted cross-loading
  shrinkage (tightening every explicit specific's →G loading) and a unit-row-Cholesky Φ — the explicit-latent
  block remains the documented mixing limit, so the map is certified at the **largest N that mixes** with a
  cross-seed resample-stability guard (§3.6).
- **A resilient long-run compute pattern.** Long mixed fits are run **detached (`nohup`+`disown`) under
  `caffeinate` with a per-seed disk cache** — defeating macOS sleep and background-task reaping, and making
  every multi-hour certification resumable.
- **The 7→9 discovery loop.** Ingesting the mania and substance indicators revealed the map is
  larger than first reported; the response was to re-certify the *joint* 9-dim model rather than bolt the new
  axes on — keeping a single, internally consistent reported map.

Chronology of the M1 hardening (each a committed stage + report): estimator/prior confirmation (§5) →
per-cohort invariance (§8) → full-N S5 certification + correlated-G (§4) → resample robustness (§8/§3.6) →
per-patient scoring (§7) → prior→posterior atlas + formal adjudication (§6) → **9-dim joint integration** →
mixed-model PPC (§8) → mania/substance invariance (§8).

---

## 5. Discussion

**What the map is for.** M1 deliberately stops at *measurement*: it converts three diagnostic cohorts into a
shared, validated coordinate system without ever using diagnosis as a modelling feature (diagnosis is
covariate/validation only). The product is the substrate the later milestones act on — **M2 will discover
validated strata in this 9-dimensional space**, then M3–M5 add temporal coherence, prognosis, and treatment.

**Why biology⊥G is the most consequential finding.** If biological burden tracked overall severity, it would
add little beyond a clinician's global impression. The data say the opposite: metabolic and inflammatory load
are **carried on axes severity does not see**. A stratification that uses these axes can therefore separate
patients who look equally ill clinically but differ biologically — the precise value proposition of a
transdiagnostic, biology-aware map.

**Where the cohorts differ, and why it's informative.** The invariance partials are not failures; they are
*findings*. Inflammatory re-weights toward eosinophils in DR; self-rated mania floors out in DR; G loses its
behavioural anchor in SZ. Each tells us where a single instrument means different things across populations —
essential to interpret any cross-cohort stratum and to choose the right per-cohort score (e.g., lean on YMRS
for mania in DR).

**Honesty as a design principle.** The pipeline is built to *let the data overturn the theory*: anhedonia was
rejected, the biology candidate was split, neurodevelopment was demoted to a proxy, two candidates were
added and confirmed once their indicators were ingested, and one count item (`isf09a`) was flagged as mis-specified. The map is
what survived adversarial checking, not what was assumed.

---

## 6. Limitations

1. **Internal validity only.** V0 baseline; no temporal (V1–V4) persistence and no external-cohort validation
   — by design, deferred to later milestones.
2. **Documented invariance partials** (F7): G (BP–SZ), inflammatory (DR), mania-Altman (DR). Cross-cohort
   comparisons on these must carry the caveat.
3. **`isf09a` item-level mis-fit** (F9): the suicide-attempt *count* needs a hurdle/zero-inflated likelihood
   if its count precision is ever required; the suicidality *factor* is unaffected.
4. **Non-Gaussian per-patient scores are on the fit subsample** (suicidality/developmental/substance);
   full-N projection (a logistic/count conditional) is an M2-prep follow-on.
5. **Mania is a 2-indicator, lower-reliability axis** (flagged *partial* for every patient); **substance is a
   2-cohort axis** (no DR SUD).
6. **Bootstrap-robustness and correlated-G have not been separately extended to mania/substance** (they carry
   the 9-dim cross-seed φ 0.993 and low bifactor-G loadings; a small follow-on).
7. **Secondary-check deferrals** (principled): Student-t continuous likelihood (§3.2, the marginalization
   requires Gaussian; outliers handled upstream by sanity bounds) and an MNAR selection-model arm (§3.4,
   missingness is largely structural/by-design, i.e. MCAR-by-design).

---

## 7. Future work

- **M2 — validated strata** on these 9-dimensional coordinates (with uncertainty propagated from §7).
- Fold the small follow-ons as M2-prep: full-N non-Gaussian scoring; mania/substance bootstrap + correlated-G;
  a hurdle likelihood for `isf09a` if needed.
- **M3 — temporal coherence** (V1–V4): do the dimensions and strata persist longitudinally?
- **M4 / M5 — prognosis and treatment** decision models on the validated strata.
