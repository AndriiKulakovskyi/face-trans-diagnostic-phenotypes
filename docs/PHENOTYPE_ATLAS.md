# PHENOTYPE_ATLAS — named factors as predictive features (v2)

> The **feature** view of the v2 dimensional analysis. The manuscript's *structural* model is **3
> weakly-correlated trans-diagnostic axes**; this atlas adds the reproducible **orthogonal standalone**
> dimensions so the whole, non-redundant feature set is documented in one place.
> Code: [`src/trans_diag/phenotype.py`](../src/trans_diag/phenotype.py) ·
> Export: [`scripts/export_phenotype_features.py`](../scripts/export_phenotype_features.py) →
> `results/hfa/phenotype_features.csv`. Provenance: `docs/LABBOOK.md` **V2-23**.

## Why this exists (and why it isn't "K")

A bootstrap robustness analysis (50 cohort-stratified resamples) showed two different, easily-confused
counts:

- **How many reproducible factors are there?** ≥ 6 — every multi-construct factor below recovers at
  **98–100 %** across resamples (mean Tucker congruence ≥ 0.97).
- **How many form a single *correlated* trans-diagnostic backbone?** **3** (internalizing, cognition,
  cardiometabolic) — eigengaps 1–2 are bounded well away from 0; gap 4 ≈ 0 (a degenerate eigenpair).

The split-half "first-collapse-minus-1" rule gives a *noisy* scalar K (bootstrap: K=2 26 %, K=3 60 %,
K≥5 14 %) because it tries to compress "3 correlated + several orthogonal" into one integer. The
**factors** are robust; the **count** is not. So we report factors, not a K. (Pushing K higher just
peels off progressively narrower clusters — ECG `RR/QTc`, then incoherent grab-bags, then improper
Heywood solutions at K≥12 — not new structure.)

## How a factor is scored

Each factor = a **curated cluster of first-order constructs** (members derived from a K=7 varimax
solution, |loading| > 0.40, lightly curated for clinical coherence). Its score is the **masked mean of
the sign-oriented, standardized member construct scores** — observed support only, **no imputation**
(a patient observing none of the members gets `NaN`). Single-construct standalones pass through their
construct score. Members are oriented so the **stated direction is the high end**.

Each exported feature ships with a `<factor>__cov` column: the **fraction of the factor's member
constructs the patient actually observed**. *Gate every feature on it* (e.g. require ≥ 0.5) — coverage
is the dominant practical constraint here.

## The atlas

`✓50%` = fraction of that cohort with ≥ 50 % member coverage (i.e. where the feature is usable).

| Factor | Kind | Direction (high =) | ✓50% BP / SZ / DR | Temporal | Stable |
|---|---|---|---|---|---|
| **internalizing** | axis | more distress / severity | 0.96 / **0.41** / 0.89 | state | 100 % |
| **cognition** | axis | better cognition | 0.68 / 0.77 / **0.54** | baseline-anchored | 100 % |
| **cardiometabolic** | axis | worse cardiometabolic load | 0.82 / 0.79 / 0.77 | trait | 100 % |
| **illness_course** | standalone | later onset / milder course | 0.77 / 0.81 / **0.48** | fixed-historical | 100 % |
| **substance_use** | standalone | lifetime alcohol/cannabis SUD | 1.00 / 1.00 / **0.00** | lifetime | (construct) |
| **mania** | standalone | more activation / mania | 0.98 / 0.92 / 0.76 | state | (construct) |
| **suicidality** | standalone | more suicidal ideation | 0.91 / 0.95 / 0.74 | state | (construct) |
| **childhood_adversity** | standalone | more childhood adversity | 0.95 / 0.83 / 0.83 | fixed-historical | 98 %, **weak** |

The 3 axes reproduce the pipeline's Stage-3 dimensions (|r| = **0.97 / 0.87 / 0.81** vs dim1/dim2/dim3).
The whole feature set is near-orthogonal: mean inter-feature |r| = **0.09** (max 0.36, illness-course ↔
cardiometabolic via residual hospitalization burden).

## Per-factor meaning

- **internalizing** — transdiagnostic distress: depression (QIDS/MADRS) + anxiety (STAI) + poor
  functioning/QoL (FAST, EQ-5D⁻, GAF⁻, CGI) + anhedonia + poor sleep, all moving together. ⚠️ the mood
  scales are **0 % observed in FACE-SZ**; SZ rests on the QoL/functioning proxies only, so in SZ the
  score reads as "poor functioning", not "depression". Treat as a **BP/DR mood axis**.
- **cognition** — a general performance factor led by **verbal episodic memory** (CVLT), with
  executive (TMT-B⁻), processing speed, fluency and education. ⚠️ memory-anchored in BP/SZ but
  **executive/fluency-based in DR** (no CVLT there) — the feature shifts meaning by cohort.
- **cardiometabolic** — inflammatory + metabolic + autonomic load (CRP/WBC, HDL/lipids, cholesterol,
  adiposity, glucose, BP, heart rate). The cleanest **pan-cohort biological** feature; **trait-stable**.
- **illness_course** — a staging/chronicity axis: later age at onset/first-treatment/first-hospitalization
  vs. heavier lifetime hospitalization burden. **Historical** (baseline-only), so a stratifier, never an
  outcome; only ~half-covered in DR.
- **substance_use** — lifetime alcohol/cannabis use disorder (MINI). **BP/SZ only** (never measured in DR).
- **mania** — Altman + YMRS activation; orthogonal to internalizing (its independence is itself a finding).
- **suicidality** — ISF ideation; orthogonal even after skip-logic recovered its coverage.
- **childhood_adversity** — childhood ADHD (WURS) + trauma (CTQ). Real but **weak** (λ≈0.45); exploratory.

**Not phenotype features** (do not export as named dimensions): the FA grab-bag
`substance + cardiac-hx + ulcer + neonatal` (use the `substance_use_disorder` construct instead) and the
ECG cluster `RR/QTc/QT` (physiological measurement structure, improper Heywood loading). Use the
underlying construct scores if you want their content.

## Using them as predictive features

1. **Coverage decides which model fits where** — the binding constraint.
   - **Pan-diagnostic model:** lean on **cardiometabolic + cognition** (the only well-covered, 3-cohort
     features). Add others as **cohort-conditional** terms.
   - **internalizing** is strong *within BP/DR*; pooling it across all three mixes mood (BP/DR) with
     functioning (SZ proxy) — keep it BP/DR or model the cohort interaction.
   - **substance_use** is a **BP/SZ-only** feature; **illness_course** is ~half-missing in DR.
2. **Orthogonality is a gift** — mean inter-feature |r| = 0.09 → **non-redundant**. Drop them all into a
   multivariable model with no collinearity; each adds independent variance.
3. **Temporal character picks the task** (longitudinal arm, Study C):
   - **cardiometabolic = trait** (ρ≈0.61) → best **baseline** predictor / stable risk marker.
   - **internalizing = state** (ρ≈0.58) → concurrent severity; weaker over long horizons.
   - **cognition = baseline-anchored**, **illness_course = fixed-historical** → baseline covariates /
     stratifiers, never trackers or outcomes.
4. **What they predicted** (Study D): functioning (GAF/FAST) was carried by **internalizing +
   cardiometabolic**; cognition added little incremental; de-confounded relapse by residual
   **internalizing**. For prognosis, internalizing + cardiometabolic are the workhorses; cognition and
   illness-course are better as descriptors/stratifiers.
5. **Trust only the stable factors** — axes + illness-course are bootstrap-rock-solid; treat
   childhood-adversity as exploratory; never expose the grab-bag/ECG factors.

## Reproduce

```bash
python3 scripts/00_run_all.py                 # (re)builds results/hfa/stage2_scores.pkl
python3 scripts/export_phenotype_features.py  # -> results/hfa/phenotype_features.csv (+ coverage cols)
python3 -m pytest tests/test_phenotype.py -q
```
