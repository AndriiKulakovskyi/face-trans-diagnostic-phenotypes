# ADJUDICATION — the M1 empirical dimension atlas (§6)

> **Map of record (read first).** The measurement map is the **8-factor immunometabolic map**: G (overall
> burden) + 7 specific axes — cognition, **immunometabolic** (a single biology factor: cardiometabolic +
> inflammatory markers together), sleep, mania/activation, suicidality, developmental-risk, and **substance**
> (orthogonal). Otherwise simple-structure with **3 earned cross-loadings** into cognition. The strata reading
> lens is **A = 5 archetypes**. Canonical: [`HORSESHOE_ESEM.md`](HORSESHOE_ESEM.md). The per-candidate
> adjudication below stands as written; where it records the biology candidate as a "split," that
> cardiometabolic + inflammatory variance is carried by the single **immunometabolic** factor.

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
(explicit latents: suicidality, developmental, substance) alongside G.

## Per-candidate verdict

| # | Candidate (theory) | Verdict | Empirical factor | Evidence |
|---|---|---|---|---|
| 1 | Overall severity | **confirmed** | **G — functional burden** | clean functioning/severity anchor (FAST 0.90, EGF 0.73); no symptom content (`lvsbjind`≈0) |
| 2 | Cognitive flexibility | **confirmed** | **cognition** | mean primary loading 0.57; invariant across cohorts |
| 3+5 | Metabolism / immuno | **confirmed (split)** | **metabolic + inflammatory** | two distinct factors, Φ≈0.19 (not collinear) — theory's single "biology" candidate split |
| 6 | Sleep / circadian | **confirmed** | **sleep** | mean loading 0.48; the most invariant axis (φ 0.99 all cohorts) |
| 7 | Neurodevelopment | **confirmed (proxy)** | **developmental-risk** | own axis (loading 0.42); a *proxy* for early-adversity/liability, not measured neurodevelopment |
| 8 | Suicidality | **confirmed (mixed-likelihood)** | **suicidality** | binary ISF ideation/attempt items load +2.2…+2.7 (logit); composes with the shared Φ |
| 4 | Anhedonia | **rejected** | — (absorbed by G + depression windows) | 1 thin indicator (BP/DR only); non-identified (R-hat 1.54); loads 0.61 on G |
| 9 | Mania / activation | **confirmed** | **mania** (marginalized) | YMRS/Altman load 0.49–0.73, \|G\| 0.15, distinct — **integrated into the largest-N-documented 9-dim joint map** |
| 10 | Substance use | **confirmed** | **substance** (explicit) | alcohol/cannabis SUD + cigarettes load +0.38…+0.69 (logit) under the **proper Bernoulli/NB likelihood** in the joint mixed model (no longer an approximation) — integrated into the largest-N-documented 9-dim map |
| — | Impulsivity | **not_testable** | — | no common indicators (dropped pre-matrix) |
| — | Negative symptoms | **not_testable** | — | no common indicators (dropped pre-matrix) |
| — | Sensory abnormalities | **not_testable** | — | no common indicators (dropped pre-matrix) |
| — | Depression / anxiety | **not a dimension** | cross-loading **windows** onto G | MADRS/QIDS/STAI load 0.66–0.80 on G — burden windows, no separate affective factor |

**Tally:** of the 10 prior-matrix candidates — **9 confirmed** (incl. one split: metabolic/inflammatory),
**1 rejected** (anhedonia); 3 pre-matrix constructs **not_testable**; depression/anxiety are **windows, not
a dimension**. All 9 are jointly modelled in the **9-dimension S5** (R-hat ≤ 1.04 · ESS ≥ 112 · 0
div · BFMI ≥ 0.41 · cross-seed Tucker φ 0.993). No candidate remains deferred.

## §6 confirmation criteria — evidence for the 9 confirmed dimensions

| Criterion | Result |
|---|---|
| ≥3 meaningful indicators | yes 8 of 9 (cognition 11 · metabolic 32 · inflammatory 14 · G 14 · sleep 9 · developmental 23 · suicidality 30 · **substance 4**);  **mania 2** (YMRS/Altman — just-identified, below the ≥3 guideline; flagged *partial*, never *well-characterised*, in scoring) |
| Primary \|λ\| ≥ 0.30, CI away from 0 | yes all surviving primaries (home loadings: continuous 0.32–0.90; mania 0.49–0.73; substance SUD +0.38…+0.69 logit) |
| **Not reducible to G** | yes §3.1 correlated-G: G correlates +0.07 inflammatory / +0.12 metabolic (≪ +0.39 cognition, +0.42 sleep) — **biology least severity-entangled**; new axes low on G (mania \|G\| 0.15, substance \|G\| 0.13); specifics distinct |
| **Not a Bayesian-prior artefact** (§5) | yes flat-prior refit reproduces loadings/Φ **exactly** (Tucker φ = 1.00); WAIC decisively prefers the bifactor |
| **Absolute fit — both blocks reproduce the data** (PPC) | yes continuous SRMR ≈ 0.07 (§5); **non-Gaussian block 21/22 endorsement rates/means within the 90% posterior-predictive interval, Bayesian p ≈ 0.5** (§8/`12_mixed_ppc`) — lone exception `isf09a` (zero-inflated attempt count), an item-level caveat below |
| **Stable under resampling** (φ ≥ 0.85) | yes §8 robustness (7-factor backbone): min Tucker φ 0.958 under LOCO, diagnosis-balanced subsampling, site cluster-bootstrap, 1/n_cohort weighting; **mania/substance** carry the **cross-seed** φ 0.993 from the 9-dim cert (their bootstrap extension is a follow-on) |
| **Measurement invariance** across BP/SZ/DR | yes largely invariant; **substance invariant BP–SZ** (φ 0.997, §8/`13_invariance9`); **partial**: G (BP–SZ, no FAST in SZ), **inflammatory in DR** (non-invariant; eosinophil- vs neutrophil-leaning), **mania-Altman in DR** (φ 0.764 — YMRS holds 0.57/0.41, self-rated Altman is a near-floor signal 0.76→0.10); substance declared a **2-cohort axis** (no DR SUD) — all documented, not hidden |
| Acceptable score reliability | yes §7 per-patient scores carry mean/SD/HDI + a reliability tier (well/partial/prior-dominated) by observed-indicator count |

## Documented M1 boundaries (honest gaps)

- **Mania & substance are fully integrated** — the reported map is now the **largest-N-documented 9-dim joint S5**
  (`scripts/s5_certify9.py` → `reports/11_s5_9dim_report.md`), with substance's binary SUD under the proper
  Bernoulli likelihood. Substance is a 2-cohort axis (alcohol/cannabis SUD are BP/SZ-only, DR-absent —
  observed-likelihood handles the missing cohort), declared as such, not claimed invariant in DR.
- **Per-cohort invariance now extends to the two new axes** (§8/`13_invariance9`): **substance is invariant
  BP–SZ** (φ 0.997); **mania is partially invariant** — YMRS holds BP–DR but the self-rated Altman is a
  near-floor signal in DR (φ 0.764, a documented partial like G-in-SZ / inflammatory-in-DR). The
  bootstrap-robustness and correlated-G arms for mania/substance remain a small follow-on (they already
  carry the 9-dim cross-seed φ 0.993 and low bifactor-G loadings).
- **One item-level PPC mis-fit — `isf09a` (suicide-attempt count).** The mixed-model PPC (§8) shows 21/22
  non-Gaussian items reproduce their observed rates; the exception is the attempt-*count* item, which is
  90.8% zeros — a hurdle count the plain NegBinom over-predicts in the high-suicidality tail. The
  **suicidality factor is unaffected** (its 7 binary ISF items all reproduce, Bayesian p 0.48–0.59); a
  hurdle/zero-inflated likelihood for that one item is the fix if its count precision is ever needed.
- **Suicidality/developmental per-patient scoring** is on the S5 subsample; full-N projection of the
  non-Gaussian block is an M2 follow-on (§7).
- **Internal validity only** — V0 baseline; no temporal (V1–V4) persistence or external-cohort validation
  (by design, later milestones).
- **Magnitude reconciliation:** the correlated-G biology~G values from the clean continuous-backbone fit
  (metabolic 0.12) is lower than the provisional full-mixed read (0.28); both agree biology is least
  entangled. The continuous-backbone value is the cleaner estimate.

## M1 status

The measurement layer is **adjudicated and complete**: a jointly-modelled, **largest-N-documented 9-dimension**
transdiagnostic map — earned from the cohort data, estimator- and prior-robust, largely invariant across
cohorts, resample-stable, with per-patient coordinates + uncertainty. Every candidate has a verdict (none
deferred). This is the object the **M2 stratification** layer will act on. *PI sign-off on this adjudication
+ the prior→posterior atlas locks M1.*
