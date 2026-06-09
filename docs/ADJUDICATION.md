# ADJUDICATION — the M1 empirical dimension atlas (§6)

> The formal verdict on each candidate construct, synthesizing the whole M1 evidence chain
> (confirmation §5 · invariance §8 · S5 certification §4 · correlated-G §3.1 · robustness §8 ·
> scoring §7). Methods of record: [`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md); findings:
> [`RESULTS.md`](RESULTS.md); the prior (theory) map: [`PRIOR_ATLAS.md`](PRIOR_ATLAS.md).

## The empirical map

On the harmonized 3-cohort FACE **V0** baseline (N = 9,013), the hybrid Bayesian sparse bifactor/ESEM
yields a **7-dimension transdiagnostic map**: a general factor **G (functional burden)** + six specific
axes — **cognition, metabolic, inflammatory, sleep, developmental-risk, suicidality** — weakly correlated
(mean |Φ| ≈ 0.10), each estimated from observed cells only (no imputation).

## Per-candidate verdict

| # | Candidate (theory) | Verdict | Empirical factor | Evidence |
|---|---|---|---|---|
| 1 | Overall severity | **confirmed** | **G — functional burden** | clean functioning/severity anchor (FAST 0.90, EGF 0.73); no symptom content (`lvsbjind`≈0) |
| 2 | Cognitive flexibility | **confirmed** | **cognition** | mean primary loading 0.57; invariant across cohorts |
| 3+5 | Metabolism / immuno | **confirmed (split)** | **metabolic + inflammatory** | two distinct factors, Φ≈0.19 (not collinear) — theory's single "biology" candidate split |
| 6 | Sleep / circadian | **confirmed** | **sleep** | mean loading 0.48; the most invariant axis (φ 0.99 all cohorts) |
| 7 | Neurodevelopment | **confirmed (proxy)** | **developmental-risk** | own axis (loading 0.42); a *proxy* for early-adversity/liability, not measured neurodevelopment |
| 8 | Suicidality | **confirmed (mixed-likelihood)** | **suicidality** | binary ISF ideation/attempt items load +2.7…+3.4 (logit); composes with the shared Φ |
| 4 | Anhedonia | **rejected** | — (absorbed by G + depression windows) | 1 thin indicator (BP/DR only); non-identified (R-hat 1.54); loads 0.61 on G |
| 9 | Mania / activation | **confirmed** | **mania-activation** | adjudicated post-hoc (2 seeds, R-hat 1.01): identifies cleanly, primary \|λ\| **0.61**, \|G\| 0.15, distinct (max \|Φ\| 0.18). Not yet in the *joint* reported map |
| 10 | Substance use | **confirmed (provisional)** | **substance** | primary \|λ\| **0.34**, \|G\| 0.13 — but binary SUD modelled as continuous (approximation) + one weak nicotine item → a mixed-likelihood re-test is the proper confirmation |
| — | Impulsivity | **not_testable** | — | no common indicators (dropped pre-matrix) |
| — | Negative symptoms | **not_testable** | — | no common indicators (dropped pre-matrix) |
| — | Sensory abnormalities | **not_testable** | — | no common indicators (dropped pre-matrix) |
| — | Depression / anxiety | **not a dimension** | cross-loading **windows** onto G | MADRS/QIDS/STAI load 0.66–0.80 on G — burden windows, no separate affective factor |

**Tally:** of the 10 prior-matrix candidates — **8 confirmed** (incl. one split) + **1 confirmed-provisional**
(substance), **1 rejected** (anhedonia); 3 pre-matrix constructs **not_testable**; depression/anxiety are
**windows, not a dimension**. The reported S5 map jointly modelled **7**; **mania (confirmed) and substance
(provisional)** were adjudicated post-hoc — folding them into the joint reported map (a re-certified 9-dim
S5 + scoring) is the immediate next increment, not an M1 result.

## §6 confirmation criteria — evidence for the 7 confirmed dimensions

| Criterion | Result |
|---|---|
| ≥3 meaningful indicators | ✅ all 7 (cognition 11 · metabolic 32 · inflammatory 14 · G 14 · sleep 9 · developmental 23 · suicidality 30) |
| Primary \|λ\| ≥ 0.30, CI away from 0 | ✅ all surviving primaries (mean home loadings 0.32–0.90) |
| **Not reducible to G** | ✅ §3.1 correlated-G: G correlates +0.06 inflammatory / +0.14 metabolic (≪ +0.39 cognition, +0.44 sleep) — **biology least severity-entangled**; specifics distinct (mean \|Φ\| 0.10) |
| **Not a Bayesian-prior artefact** (§5) | ✅ flat-prior refit reproduces loadings/Φ **exactly** (Tucker φ = 1.00); WAIC decisively prefers the bifactor; PPC SRMR ≈ 0.07 |
| **Stable under resampling** (φ ≥ 0.85) | ✅ §8 robustness: min Tucker φ ≥ 0.85 under leave-one-cohort-out, diagnosis-balanced subsampling, site cluster-bootstrap, and 1/n_cohort weighting |
| **Measurement invariance** across BP/SZ/DR | ✅ largely invariant; **partial**: G (BP–SZ, no FAST in SZ) and **inflammatory in DR** (eosinophil- vs neutrophil-leaning) — documented, not hidden |
| Acceptable score reliability | ✅ §7 per-patient scores carry mean/SD/HDI + a reliability tier (well/partial/prior-dominated) by observed-indicator count |

## Documented M1 boundaries (honest gaps)

- **Mania & substance are now adjudicated (confirmed / confirmed-provisional)** but not yet folded into
  the *joint* reported map. Mania identifies cleanly; substance survives a continuous-approximation test
  and needs a mixed-likelihood re-test (its SUD indicators are binary, BP/SZ-only). Re-certifying a 9-dim
  S5 and re-scoring with them is the immediate next increment (`scripts/s5_mania_substance.py` →
  `reports/10_mania_substance_report.md`).
- **Suicidality/developmental per-patient scoring** is on the S5 subsample; full-N projection of the
  non-Gaussian block is an M2 follow-on (§7).
- **Internal validity only** — V0 baseline; no temporal (V1–V4) persistence or external-cohort validation
  (by design, later milestones).
- **Magnitude reconciliation:** the correlated-G biology~G values from the clean continuous-backbone fit
  (metabolic 0.12–0.14) are lower than the provisional full-mixed read (0.28); both agree biology is least
  entangled. The continuous-backbone value is the cleaner estimate.

## M1 status

The measurement layer is **adjudicated**: a jointly-modelled **7-dimension** transdiagnostic map (earned
from the cohort data, estimator- and prior-robust, largely invariant across cohorts, resample-stable, with
per-patient coordinates + uncertainty) **plus two post-hoc-confirmed axes — mania (solid) and substance
(provisional)**. Every candidate now has a verdict (no remaining "deferred"). This is the object the **M2
stratification** layer will act on. *Two things finalize M1: PI sign-off on this adjudication + the
prior→posterior atlas, and (optionally) folding mania/substance into a re-certified 9-dim joint map.*
