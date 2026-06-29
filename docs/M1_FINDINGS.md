# M1 — Findings & Discussion

> **Map of record (read first).** The measurement map is the **8-factor immunometabolic map**: a general
> factor **G (overall burden)** + 7 specific axes — cognition, **immunometabolic** (one biology factor:
> cardiometabolic + inflammatory markers together), sleep, mania/activation, suicidality, developmental-risk,
> and **substance** (pinned orthogonal). The map is otherwise simple-structure with **3 earned cross-loadings**
> (CTQ-37 → cognition, PSQI-latency → cognition, PSQI-daytime → cognition). On this map the strata reading lens
> is **A = 5 archetypes (A0–A4)**. Canonical findings: [`HORSESHOE_ESEM.md`](HORSESHOE_ESEM.md) (map),
> [`STRATA_OOP_FINDINGS.md`](STRATA_OOP_FINDINGS.md) (archetypes). diagnosis is validation-only.

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
only* (no imputation) — yields an **8-factor transdiagnostic map**: a general factor **G
(overall burden)** plus 7 specific axes — **cognition, immunometabolic, sleep, mania/activation,
suicidality, developmental-risk, substance**. The map is hardened end-to-end (estimator/prior
robustness, measurement invariance, resample robustness, absolute-fit PPC across both likelihood blocks)
and projected to per-patient coordinates with uncertainty and reliability flags. The headline scientific
finding is that **biology is the least severity-entangled domain** — the **immunometabolic** axis
(cardiometabolic + inflammatory load on one factor) is largely independent of the general functional-impairment
factor, whereas cognition and sleep partly track it.

---

## 2. The empirical map

**Table 1 — factors and primary loadings.**

| # | Dimension | Block | Anchoring indicators | Mean primary \|λ\| |
|---|---|---|---|---|
| 0 | **G — functional burden** | explicit (Gaussian) | FAST 0.78, EGF 0.60, EQ-5D 0.60, CGI-S 0.57; **no symptom content** | 0.51 (anchors) |
| 1 | cognition | marginalized | CVLT 0.85–0.89, WAIS, TMT (executive / processing-speed / memory) | 0.58 |
| 2 | **immunometabolic** | marginalized | BMI 0.95, weight/waist, lipids, glycemia, BP, CRP 0.37, leukocyte subsets | 0.24 |
| 3 | sleep | marginalized | PSQI items (PSQI-total 0.88) | 0.40 |
| 4 | developmental-risk | explicit | childhood adversity (CTQ 0.50–0.93), birth/parental history | 0.60 (CTQ) / 0.24 (all) |
| 5 | suicidality | explicit (binary/ordinal) | ISF ideation/attempt items (+2.2…+4.5 logit) | logit scale |
| 6 | **mania/activation** | marginalized | YMRS 0.56, Altman 0.37 (0.76 BP-only) | 0.46 |
| 7 | **substance** | explicit (mixed) | nicotine (cigarettes 0.72, Fagerström 0.41), alcohol/cannabis lifetime SUD | 0.39 |

*(Loadings are not comparable across blocks: the explicit suicidality/substance items are on a logit/threshold
scale, and developmental-risk is heterogeneous — strong continuous CTQ items plus weak explicit birth-history
flags.)* The biology axis is a **single immunometabolic factor** carrying cardiometabolic and inflammatory load
together (BMI, lipids, glycemia, blood pressure, CRP, leukocyte subsets), kept simple-structure by the
sparsity prior on its off-home cells.

**Table 2 — how each domain relates to G (direct loadings).** Because the map is identified as a bifactor (G
orthogonal to the specifics *by construction*), a domain's relationship to general burden is read from its
indicators' **direct loadings on G**, not from Φ. Mean \|G-loading\| per domain:

| domain | mean \|G-loading\| |
|---|---|
| cognition | 0.20 |
| sleep | 0.20 |
| mania/activation | 0.13 |
| suicidality | 0.11 |
| developmental-risk | 0.07 |
| substance | 0.07 |
| immunometabolic | 0.06 |

**Table 3 — factor correlations Φ.** Φ is the **latent factor-correlation matrix** (correlations between the
factor scores themselves). Two rows are fixed by design — **G** is orthogonal to every specific (bifactor
identification) and **substance** is pinned orthogonal (its cross-factor correlations are non-identifiable) — so
Φ describes the six freely-correlating specifics, where it is **near-simple-structure** (specific–specific mean
\|off-diagonal\| ≈ 0.08): the specifics are genuinely distinct axes, not a single collapsed factor. The
non-trivial couplings are sleep–mania +0.23, sleep–developmental +0.20, suicidality–developmental +0.20, and
cognition–sleep −0.16; **immunometabolic's largest tie is only +0.08**. The map is otherwise simple-structure
with **3 earned cross-loadings** into cognition (CTQ-37 −0.08, PSQI-latency +0.05, PSQI-daytime −0.05, each 95%
CI excluding 0 — childhood adversity and poor sleep load weakly on cognition). Depression/anxiety
(MADRS/QIDS/STAI) are **not a dimension** — they load on **G** as cross-loading "windows" (burden surfaces, no
separable affective factor).

---

## 3. Principal findings

Each finding is stated as *observation → result → interpretation*.

### F1 — A clean general factor is functional burden, not a "p-factor"
G is anchored only by **functioning/severity** items (FAST, EGF, EQ-5D) and carries **no symptom content**
(`lvsbjind` ≈ 0). G is therefore best read as a **transdiagnostic impairment/distress axis**, not a latent
liability to psychopathology — a deliberately conservative reading that avoids the bifactor "p-factor"
over-claim.

### F2 — A periphery of three weakly-G axes, of which immunometabolic is the *earned island*
**Observation.** Three axes load only weakly on general burden — immunometabolic (0.06), substance (0.07) and
developmental-risk (0.07) — a near-tie on the direct G-loading (Table 2). On this metric alone they are
indistinguishable; the differences below come from *how* each is decoupled, not from the G-loading.

**Result.** They are peripheral for three different reasons. **Substance** is orthogonal *by construction*
(pinned; thin, two-cohort, DR = 0) — its independence is imposed, not estimated. **Developmental-risk** is weak
on G but **couples to the symptom axes** (sleep / suicidality, Φ ≈ 0.20) and is a *historical antecedent* rather
than a current state (the least temporally durable axis, M3, ICC 0.39). **Immunometabolic load** is the only
domain weakly tied to the *entire* clinical picture — both general burden (direct loading 0.06; correlated-G
≈ +0.10, versus **+0.39 cognition** and **+0.42 sleep**) **and** the symptom axes (max Φ 0.076, versus ≥ 0.16
for every other freely-correlating factor). It is, uniquely, an **earned island** in the map: weakly tied to
everything else on a freely-estimated basis.

> **Caveat (correlated-G does not separate biology from developmental).** The freely-correlated-G arm cleanly
> separates immunometabolic (≈ 0.10) from cognition / sleep (0.39 / 0.42), but it does **not** cleanly separate
> it from developmental-risk: the one fit that included both put them at a comparable ≈ 0.28, and biology's value
> was only later refined down to ≈ 0.10. The immunometabolic-vs-developmental distinction therefore rests on the
> **Φ symptom-decoupling** (0.08 vs 0.20), not on the correlated-G arm.

**Interpretation.** Because immunometabolic load is the only earned island — and is also measurable, modifiable,
and overturns the clinical prior that more severely ill patients carry worse cardiometabolic/inflammatory load —
it is the axis carried forward as the biological stratification target (M2). The robustness of its low coupling
(unchanged under age/sex/education/site and antipsychotic adjustment) is the load-bearing premise downstream.

### F3 — Theory's "biology" candidate is confirmed as one immunometabolic factor
The prior ontology posited a metabolic/immuno candidate; the data **confirm it as a single immunometabolic
axis** on which cardiometabolic and inflammatory markers cohere together (BMI → immunometabolic ≈ 0.95,
CRP → immunometabolic ≈ 0.37). Cardiometabolic and immune load are not separable axes here — they are two
facets of one biological dimension — which is exactly why this axis behaves as a unit in stratification (M2)
and persists as the single most durable trait longitudinally (M3, ICC 0.91).

### F4 — Developmental-risk is a *proxy*, not measured neurodevelopment
The developmental axis is real and distinct (loading 0.42) but is anchored by **childhood adversity / birth
/ parental** indicators — it indexes **early-adversity/liability**, and is named and interpreted as a
*proxy*, not as measured neurodevelopment.

### F5 — Suicidality composes as a mixed-likelihood axis
Binary ISF ideation/attempt items load **+2.2…+2.7 on the logit scale** and compose with the shared Φ under
the proper Bernoulli likelihood — demonstrating that non-Gaussian psychopathology indicators integrate into
the same correlated-factor space as the continuous biology without breaking identification.

### F6 — Mania and substance are real, distinct axes
**Mania/activation** (YMRS/Altman, primary \|λ\| 0.49–0.76, \|G\| 0.15) and **substance** (alcohol/cannabis
SUD + nicotine under the proper Bernoulli/NegBinom likelihoods, \|G\| 0.13) are confirmed as two of the
eight axes. The global model integrates them under the one shared Φ — **substance pinned orthogonal** to the
correlated block (its cross-factor correlations are non-identifiable) — and certifies cleanly (R-hat ≤ 1.04,
ESS ≥ 112, 0 div, cross-seed Tucker φ 0.993). **Interpretation:** manic activation and substance use are
distinct, low-G transdiagnostic axes — not reducible to severity, and worth carrying into stratification.

### F7 — Measurement invariance: largely invariant, with documented partials
The loadings are **largely invariant** across BP/SZ/DR — the **immunometabolic** axis is fully invariant
(Tucker φ 0.987), and the merged biology factor is one of the durable backbone axes. The **honest
exceptions**, each documented rather than hidden:
- **G** — partial BP–SZ (SZ lacks the FAST anchor; G re-anchored on CGI-S/EGF/EQ-5D there).
- **mania-Altman in DR** — YMRS holds BP–DR (0.57/0.41) but the **self-rated Altman does not transfer**
  (0.76 → 0.10; Tucker φ 0.764). Self-reported manic activation is a near-floor signal in a
  DR cohort; clinician-rated YMRS carries mania there.
- **substance is invariant BP–SZ** (Tucker φ **0.997**) and is declared a **2-cohort axis** (its
  alcohol/cannabis SUD are BP/SZ-only; not claimed for DR).

### F8 — The structure is not an artefact of estimator, prior, resampling, cohort, or site
- **Not a prior/estimator artefact** (§5): an independent flat-prior refit (fresh seed, no warm-start)
  reproduces the loadings/Φ **to 3 d.p.** (Tucker φ = 1.00, max |ΔΦ| = 0.00); WAIC decisively prefers the
  bifactor over unidimensional and correlated-no-G alternatives.
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
All 9,013 patients are scored on the continuous-anchored backbone axes (G, cognition, immunometabolic, sleep,
mania) — with the explicit dimensions on the fit subsample — each coordinate carrying a posterior **mean / SD
/ HDI** plus a **reliability tier** (well-characterised / partial / prior-dominated) by observed-indicator
count. The flags correctly expose coverage: cognition is prior-dominated for the 2,506 patients without
cognitive testing; **mania is *partial* for everyone** (only 2 indicators). Downstream strata must propagate
this uncertainty, not treat all coordinates as equally measured.

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
  block remains the documented mixing limit, so the map is documented at the **largest N that mixes** with a
  cross-seed resample-stability guard (§3.6).
- **A resilient long-run compute pattern.** Long mixed fits are run **detached (`nohup`+`disown`) under
  `caffeinate` with a per-seed disk cache** — defeating macOS sleep and background-task reaping, and making
  every multi-hour certification resumable.
- **One joint global fit, never bolted-on axes.** All eight axes — including the mixed-likelihood mania and
  substance indicators — are estimated in a single *joint* model under one shared Φ rather than fitted
  separately and merged, keeping a single, internally consistent reported map.

Chronology of the M1 hardening (each a committed stage + report): estimator/prior confirmation (§5) →
per-cohort invariance (§8) → full-N S5 certification + correlated-G (§4) → resample robustness (§8/§3.6) →
per-patient scoring (§7) → prior→posterior atlas + formal adjudication (§6) → **joint 8-factor integration** →
mixed-model PPC (§8) → mania/substance invariance (§8).

---

## 5. Discussion

**What the map is for.** M1 deliberately stops at *measurement*: it converts three diagnostic cohorts into a
shared, validated coordinate system without ever using diagnosis as a modelling feature (diagnosis is
covariate/validation only). The product is the substrate the later milestones act on — **M2 validates the continuum
(continuous coordinates + a stable A = 5 archetype simplex) in this 8-factor space**, then M3–M5 add
temporal coherence, prognosis, and treatment.

**Why biology⊥G is the most consequential finding.** If biological burden tracked overall severity, it would
add little beyond a clinician's global impression. The data say the opposite: **immunometabolic load**
is **carried on an axis severity does not see**. A stratification that uses these axes can therefore separate
patients who look equally ill clinically but differ biologically — the precise value proposition of a
transdiagnostic, biology-aware map.

**Where the cohorts differ, and why it's informative.** The invariance partials are not failures; they are
*findings*. Self-rated mania floors out in DR; G loses its behavioural anchor in SZ. Each tells us where a
single instrument means different things across populations — essential to interpret any cross-cohort stratum
and to choose the right per-cohort score (e.g., lean on YMRS for mania in DR).

**Honesty as a design principle.** The pipeline is built to *let the data overturn the theory*: anhedonia was
rejected, the biology candidate was confirmed as a single immunometabolic axis, neurodevelopment was demoted
to a proxy, mania and substance were confirmed as distinct axes, and one count item (`isf09a`) was flagged as
mis-specified. The map is what survived adversarial checking, not what was assumed.

---

## 6. Limitations

1. **Internal validity only.** V0 baseline; no temporal (V1–V4) persistence and no external-cohort validation
   — by design, deferred to later milestones.
2. **Documented invariance exceptions** (F7): G (BP–SZ partial) and mania-Altman (DR partial); the
   immunometabolic axis is fully invariant. Cross-cohort comparisons on the partials must carry the caveat.
3. **`isf09a` item-level mis-fit** (F9): the suicide-attempt *count* needs a hurdle/zero-inflated likelihood
   if its count precision is ever required; the suicidality *factor* is unaffected.
4. **Non-Gaussian per-patient scores are on the fit subsample** (suicidality/developmental/substance);
   full-N projection (a logistic/count conditional) is an M2-prep follow-on.
5. **Mania is a 2-indicator, lower-reliability axis** (flagged *partial* for every patient); **substance is a
   2-cohort axis** (no DR SUD).
6. **Bootstrap-robustness and correlated-G have not been separately extended to mania/substance** (they carry
   the joint cross-seed φ 0.993 and low bifactor-G loadings; a small follow-on).
7. **Secondary-check deferrals** (principled): Student-t continuous likelihood (§3.2, the marginalization
   requires Gaussian; outliers handled upstream by sanity bounds) and an MNAR selection-model arm (§3.4,
   missingness is largely structural/by-design, i.e. MCAR-by-design).

---

## 7. Future work

- **M2 — the validated continuum** (continuous coordinates + a stable A = 5 archetype simplex; no privileged K)
  on these 8-factor coordinates (with uncertainty propagated from §7).
- Fold the small follow-ons as M2-prep: full-N non-Gaussian scoring; mania/substance bootstrap + correlated-G;
  a hurdle likelihood for `isf09a` if needed.
- **M3 — temporal coherence** (V1–V4): do the coordinates and archetypes persist longitudinally?
- **M4 / M5 — prognosis and treatment** decision models on the continuous map + archetypes.
