# ADJUDICATION — the M1 empirical dimension atlas (§6)

> **Map of record (read first).** The measurement map is the **8-factor immunometabolic map**: G (overall
> burden) + 7 specific axes — cognition, **immunometabolic** (a single biology factor carrying cardiometabolic
> + inflammatory markers together), sleep, mania/activation, suicidality, developmental-risk, and **substance**
> (pinned orthogonal). Otherwise simple-structure with **3 earned cross-loadings** into cognition. The strata
> reading lens is **A = 5 archetypes**. Canonical: [`HORSESHOE_ESEM.md`](HORSESHOE_ESEM.md).

> The formal verdict on each candidate construct, synthesizing the whole M1 evidence chain
> (confirmation §5 · invariance §8 · S5 certification §4 · correlated-G §3.1 · robustness §8 ·
> scoring §7). Methods of record: [`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md); **findings + discussion:
> [`M1_FINDINGS.md`](M1_FINDINGS.md)**; per-stage detail: [`RESULTS.md`](RESULTS.md); the prior (theory)
> map: [`PRIOR_ATLAS.md`](PRIOR_ATLAS.md).

## The empirical map

On the harmonized 3-cohort FACE **V0** baseline (N = 9,013), the hybrid Bayesian sparse bifactor/ESEM
yields an **8-factor transdiagnostic map**: a general factor **G (overall burden)** + 7 specific
axes — **cognition, immunometabolic, sleep, mania/activation, suicidality, developmental-risk, substance** —
weakly correlated, each estimated from observed cells only (no imputation). The biology block
(cardiometabolic + inflammatory markers) is the single **immunometabolic** axis; continuous-anchored axes are
marginalized (cognition/immunometabolic/sleep/mania), and three carry non-Gaussian indicators
(explicit latents: suicidality, developmental, substance) alongside G. The map is otherwise
simple-structure, with **3 earned cross-loadings** into cognition (CTQ-37 −0.094, PSQI-latency +0.057,
PSQI-daytime −0.070, each 95% CI excluding 0). **Substance is pinned orthogonal** to the correlated block.

## Per-candidate verdict

| # | Candidate (theory) | Verdict | Empirical factor | Evidence |
|---|---|---|---|---|
| 1 | Overall severity | **confirmed** | **G — functional burden** | clean functioning/severity anchor (FAST 0.90, EGF 0.73); no symptom content (`lvsbjind`≈0) |
| 2 | Cognitive flexibility | **confirmed** | **cognition** | mean primary loading 0.57; invariant across cohorts |
| 3+5 | Metabolism / immuno | **confirmed (merged)** | **immunometabolic** | cardiometabolic + inflammatory markers compose a single biology axis (bmi→immunometabolic ≈ 0.95, crp→immunometabolic ≈ 0.37) — theory's single "biology" candidate is one factor |
| 6 | Sleep / circadian | **confirmed** | **sleep** | mean loading 0.48; the most invariant axis (φ 0.99 all cohorts) |
| 7 | Neurodevelopment | **confirmed (proxy)** | **developmental-risk** | own axis (loading 0.42); a *proxy* for early-adversity/liability, not measured neurodevelopment |
| 8 | Suicidality | **confirmed (mixed-likelihood)** | **suicidality** | binary ISF ideation/attempt items load +2.2…+2.7 (logit); composes with the shared Φ |
| 4 | Anhedonia | **rejected** | — (absorbed by G + depression windows) | 1 thin indicator (BP/DR only); non-identified (R-hat 1.54); loads 0.61 on G |
| 9 | Mania / activation | **confirmed** | **mania/activation** (marginalized) | YMRS/Altman load 0.49–0.73, \|G\| 0.15, distinct — jointly modelled in the 8-factor map |
| 10 | Substance use | **confirmed** | **substance** (explicit, orthogonal) | alcohol/cannabis SUD + cigarettes load +0.38…+0.69 (logit) under the **proper Bernoulli/NB likelihood** in the joint mixed model — jointly modelled in the 8-factor map, pinned orthogonal |
| — | Impulsivity | **not_testable** | — | no common indicators (dropped pre-matrix) |
| — | Negative symptoms | **not_testable** | — | no common indicators (dropped pre-matrix) |
| — | Sensory abnormalities | **not_testable** | — | no common indicators (dropped pre-matrix) |
| — | Depression / anxiety | **not a dimension** | cross-loading **windows** onto G | MADRS/QIDS/STAI load 0.66–0.80 on G — burden windows, no separate affective factor |

**Tally:** of the 10 prior-matrix candidates — **9 confirmed** (with metabolism/immuno adjudicated into the
single **immunometabolic** biology axis), **1 rejected** (anhedonia); 3 pre-matrix constructs **not_testable**;
depression/anxiety are **windows, not a dimension**. The confirmed candidates resolve to **8 latent
dimensions** — G + the 7 specifics {cognition, immunometabolic, sleep, mania/activation, suicidality,
developmental-risk, substance} — jointly modelled in the **8-factor map** (R-hat ≤ 1.04 · ESS ≥ 112 · 0
div · BFMI ≥ 0.41 · cross-seed Tucker φ 0.993). No candidate remains deferred.

## §6 confirmation criteria — evidence for the 8 confirmed dimensions

| Criterion | Result |
|---|---|
| ≥3 meaningful indicators | yes 7 of 8 (cognition 11 · immunometabolic 46 · G 14 · sleep 9 · developmental 23 · suicidality 30 · **substance 4**);  **mania 2** (YMRS/Altman — just-identified, below the ≥3 guideline; flagged *partial*, never *well-characterised*, in scoring) |
| Primary \|λ\| ≥ 0.30, CI away from 0 | yes all surviving primaries (home loadings: continuous 0.32–0.95; mania 0.49–0.73; substance SUD +0.38…+0.69 logit) |
| **Not reducible to G** | yes §3.1 correlated-G: G correlates +0.12 immunometabolic (≪ +0.39 cognition, +0.42 sleep) — **biology least severity-entangled**; new axes low on G (mania \|G\| 0.15, substance \|G\| 0.13); specifics distinct |
| **Not a Bayesian-prior artefact** (§5) | yes flat-prior refit reproduces loadings/Φ **exactly** (Tucker φ = 1.00); WAIC decisively prefers the bifactor |
| **Absolute fit — both blocks reproduce the data** (PPC) | yes continuous SRMR ≈ 0.07 (§5); **non-Gaussian block 21/22 endorsement rates/means within the 90% posterior-predictive interval, Bayesian p ≈ 0.5** (§8/`12_mixed_ppc`) — lone exception `isf09a` (zero-inflated attempt count), an item-level caveat below |
| **Stable under resampling** (φ ≥ 0.85) | yes §8 robustness (continuous backbone): min Tucker φ 0.958 under LOCO, diagnosis-balanced subsampling, site cluster-bootstrap, 1/n_cohort weighting; **mania/substance** carry the **cross-seed** φ 0.993 from the 8-factor cert (their bootstrap extension is a follow-on) |
| **Measurement invariance** across BP/SZ/DR | yes largely invariant; **substance invariant BP–SZ** (φ 0.997, §8/`13_invariance`); **partial**: G (BP–SZ, no FAST in SZ), **immunometabolic in DR** (partial; inflammatory markers eosinophil- vs neutrophil-leaning), **mania-Altman in DR** (φ 0.764 — YMRS holds 0.57/0.41, self-rated Altman is a near-floor signal 0.76→0.10); substance declared a **2-cohort axis** (no DR SUD) — all documented, not hidden |
| Acceptable score reliability | yes §7 per-patient scores carry mean/SD/HDI + a reliability tier (well/partial/prior-dominated) by observed-indicator count |

## Documented M1 boundaries (honest gaps)

- **Mania & substance are jointly modelled** — the reported map is the **8-factor joint fit**, with substance's
  binary SUD under the proper Bernoulli likelihood and substance **pinned orthogonal** to the correlated block.
  Substance is a 2-cohort axis (alcohol/cannabis SUD are BP/SZ-only, DR-absent — observed-likelihood handles
  the missing cohort), declared as such, not claimed invariant in DR.
- **Per-cohort invariance covers the thin axes** (§8/`13_invariance`): **substance is invariant
  BP–SZ** (φ 0.997); **mania is partially invariant** — YMRS holds BP–DR but the self-rated Altman is a
  near-floor signal in DR (φ 0.764, a documented partial like G-in-SZ / immunometabolic-in-DR). The
  bootstrap-robustness and correlated-G arms for mania/substance remain a small follow-on (they already
  carry the 8-factor cross-seed φ 0.993 and low bifactor-G loadings).
- **One item-level PPC mis-fit — `isf09a` (suicide-attempt count).** The mixed-model PPC (§8) shows 21/22
  non-Gaussian items reproduce their observed rates; the exception is the attempt-*count* item, which is
  90.8% zeros — a hurdle count the plain NegBinom over-predicts in the high-suicidality tail. The
  **suicidality factor is unaffected** (its 7 binary ISF items all reproduce, Bayesian p 0.48–0.59); a
  hurdle/zero-inflated likelihood for that one item is the fix if its count precision is ever needed.
- **Suicidality/developmental per-patient scoring** is on the S5 subsample; full-N projection of the
  non-Gaussian block is an M2 follow-on (§7).
- **Internal validity only** — V0 baseline; no temporal (V1–V4) persistence or external-cohort validation
  (by design, later milestones).
- **Biology~G entanglement:** the correlated-G immunometabolic~G value from the continuous-backbone fit is
  +0.12 — biology is the least severity-entangled specific axis (≪ cognition +0.39, sleep +0.42).

## M1 status

The measurement layer is **adjudicated and complete**: a jointly-modelled, **8-dimension**
transdiagnostic map — earned from the cohort data, estimator- and prior-robust, largely invariant across
cohorts, resample-stable, with per-patient coordinates + uncertainty. Every candidate has a verdict (none
deferred). This is the object the **M2 stratification** layer acts on, where the continuous coordinates and a
stable **A = 5 archetype simplex** are the load-bearing reading lens (a nested K-family 2/3/4 is exported as a
convention, with no privileged K). *PI sign-off on this adjudication + the prior→posterior atlas locks M1.*
