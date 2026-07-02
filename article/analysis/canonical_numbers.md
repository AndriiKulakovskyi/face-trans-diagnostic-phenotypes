# FACE-ATLAS — canonical numbers (single source of truth for the rewrite)

**Every track must use these exact values.** If a value you are about to write disagrees with this table, this table wins. Do not "correct" or re-round. All values verified this session against the fitted model and results tree.

## Cohort
- N = 9,013 total (BP 6,252; SZ 2,209; DR 552); 21 sites; diagnosis withheld from fitting.
- LOSO primary analysis: 15 sites with N ≥ 100.

## Immunometabolic dissociation (the existence proof)
- Correlation with general burden G: **≈0.10** (loading-level, the paper's headline number). Score-level is 0.08 — use ≈0.10 in prose to match the existing figures/tables.
- Comparison specifics: cognition **0.39**, sleep **0.42**.
- Test–retest ICC: **0.91** (most durable axis).
- BMI loading 0.946 (table value); specific-factor loading 0.908; CRP 0.366; triglycerides 0.382; waist 0.838; weight 0.875.

## Prognosis (Fig 6)
- **Lead with WITHIN-cohort gradient:** BP A2=27% / A4=73%; SZ A2=9% / A4=25%; DR A2=31% / A4=72%. A2 (immunometabolic pole) is lowest-remitting in every diagnosis.
- Pooled-across-cohort range = 22%–63% (this is the marginal; keep only as a secondary framing, do not lead with it).
- η² (compactness on the 8 map axes): archetype **0.256**, DSM-5 **0.026**, cohort **0.018**; ratio **9.7×**. This is compactness, NOT outcome variance — say so.
- ΔELPD held-out functioning: A=5 archetypes **+62.8**; continuous 8-axis **+38.1**; A=5 (G-free) **+33.5**; tessellation K=3 **+19.6**; durable biology axis **+2.3**.
- Co-informativeness: map **+17.3**, DSM-5 **+29.0**, both together **+62.6** (these three are a distinct triple from the +62.8 above — do NOT merge them).
- Individual-level remission AUC gain over diagnosis: **+0.010** (demote to a sentence; the value is in stratification + durable measurement, not point prediction).

## LOSO external validity (new ED figure, patch P10, addresses B-2/R01)
- 15 folds; the GLLVM (VI path, NOT NUTS) refit per fold.
- Immunometabolic loading congruence Tucker φ **0.9993–1.0000** (min 0.9993, median 0.9998).
- **All eight factors cleared their pre-set congruence bar in all 15 folds.**
- Decoupling |corr(immuno, G)| per fold **0.073–0.082** (vs ≈0.08 full sample).
- Immunometabolic is the least-severity-correlated specific axis in **15/15** folds.
- Weakest single factor value: φ=0.917 (Monaco, n=237), a thin 2–4-item axis, still above the 0.85 thin-factor bar. Congruence bars: 0.95 major axes, 0.85 thin axes.
- VI vs NUTS agreement: Tucker 0.957–0.999.

## Adaptive assessment / exact Fisher (all 8 axes)
- Immunometabolic anthropometric triad (BMI, weight, waist) → reliability **0.85 in 3 items**, plateau **~0.88** (does NOT reach 0.90). Reliability 0.80 at 2 items.
- Suicidality reaches 0.90 at 11 items (well-behaved).
- Mania/activation: only **2 items**, max reliability **0.408**. Substance: **4 items**, max reliability **0.429**. Both are BANK limitations, not missingness (95–96% of patients have ≥1 item).
- Method note: exact Fisher information per likelihood family (Gaussian λ²/σ², Bernoulli λ²p(1−p), graded-response cumulative-logit, NB λ²μr/(r+μ)). This SUPERSEDES the earlier λ²/ψ approximation.

## Value of information (all 8 axes)
- Shared cross-axis battery: **27 items → mean reliability 0.70**, all 8 axes online.
- Cohort collection gaps (mean items/axis): SZ under-measured on immunometabolic (20.7 vs BP 34.9) and sleep (3.9 vs 8.4); DR thinnest on substance.

## Loading ≠ information (new supporting figure)
- Alcohol flag: loading 0.957, endorsement ~1.4%, Fisher info **0.013**. Cannabis flag: loading 0.923, ~1.2%, info **0.010**.
- BMI (Gaussian): info **2.8** — ~100× the rare flags despite similar/lower loading magnitude comparability caveat (cross-family loadings on different link scales; information is the comparable quantity).

## Worked patient (numbered Fig 2)
- Patient BP-62162: core-tier A2, **86%** immunometabolic-archetype weight; immunometabolic score +3.57 SD (posterior SD 0.22, from 6 items); cognition & suicidality prior-dominated (0 items, SD ~0.99).

## Structure test (Hopkins) — B-1/B-3 fix
- Hopkins statistic real = **≈0.79–0.81** (arm A 0.794, arm B 0.809) — NOT "→0.5". The glossary text saying the continuum verdict requires H→½ is the RULE; the OBSERVED value is ≈0.79–0.81.
- Hopkins null 0.756 ± 0.004; **z = 8.71** (this large, significant Hopkins z must be DISCLOSED, not hidden — the honest framing is that Hopkins rejects uniformity but the silhouette test, the relevant one for cluster separation, is n.s.).
- Silhouette real 0.140, null 0.137 ± 0.002, z = **1.13** (n.s.) — this is the load-bearing "no discrete clusters" evidence, peak silhouette 0.146 < 0.15.

## Unlikely-prior equation — M-1/K01 fix (methods)
- The reported production map uses `soft_unlikely=False`: the ~980 "unlikely" cross-loading cells are **hard zeros (δ₀ point mass at 0)**.
- The N(0, 0.05²) "unlikely" prior in Eq.(softprior) is the **soft SENSITIVITY arm**, not the reported model. Present it as such: reported = hard-zero; sensitivity = soft N(0,0.05²).

## Convergence gate — M-3 fix (methods)
- Reported global R̂ = 1.03. State the **two-tier gate up front**: backbone/major axes held to R̂ ≤ 1.01; thin factors run warmer (up to ~1.03) and are validated by congruence/VI-NUTS agreement rather than R̂ alone. Do not claim a uniform R̂ ≤ 1.01 gate the headline number violates.

## Novelty anchors (verified in references.bib)
- Nearest prior art: **Lamers et al. 2020, Brain Behav Immun, DOI 10.1016/j.bbi.2020.04.002** (within-MDD immuno-metabolic profiles; symptom-anchored, biology as correlated outcome — NOT a latent axis with diagnosis held out). Also BSNIP biotypes (Clementz 2016), HiTOP, normative modelling (Marquand 2016 / Wolfers 2018).
- Three novel-in-combination pillars: (1) missingness-native measurement (never imputes; one global observed-cell likelihood; ~40% mean cell-missingness), (2) uncertainty propagated end-to-end, (3) fixed-then-validated map with diagnosis held out of fitting.
