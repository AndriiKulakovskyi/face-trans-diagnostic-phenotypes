# V3 results — the certified transdiagnostic measurement model

> Living results page for the V3 **measurement layer** (Phases F–G). Plan: [`V3_PLAN.md`](V3_PLAN.md) ·
> step journal: [`LABBOOK_V3.md`](LABBOOK_V3.md) · numbers log: [`FINDINGS.md`](FINDINGS.md). Figures are
> aggregate (no per-patient data); regenerate with `scripts/v3/04_extended_model.py` → `05_visualize.py`.

## The model

A patient-level, **no-imputation** latent model on the V0 baseline, **cohort-balanced** (500
most-complete patients per cohort — BP/SZ/DR equal, removing the BP≫SZ≫DR dominance):

- **5 Gaussian factors** estimated by a **marginalized** observed-data likelihood (factors integrated
  out → certifiable geometry): **cognition · metabolic · inflammatory · sleep · affective**. Affective
  (MADRS/QIDS/STAI/anhedonia) is BP/DR-measured; SZ contributes no affective cells (handled by the
  observed-data likelihood, never imputed).
- **Suicidality** as an **explicit-latent module with mixed likelihoods** — 7 ISF items (Bernoulli) +
  attempt count (negative-binomial), 3-cohort.

**Certified:** max R-hat **1.020**, min ESS **1,066**, **0 divergences** (N=1,500, 4 chains; 15
observed-patterns, ~23% rare-pattern tail dropped — see Caveats).

## 1 · Correlation structure — no general factor, one strong coupling

![Factor correlation matrix](figures/v3/phi_heatmap.png)

![Dimension correlation network](figures/v3/correlation_network.png)

**Mean |off-diagonal| ≈ 0.18 (0.12 excluding sleep↔affective) — there is no general psychopathology
factor.** The dimensions are a *weakly-correlated coordinate system*, not one axis of "severity,"
established under a certified, cohort-balanced, patient-level estimator.

What the off-diagonals say, and how we read them:

- **Symptoms (affective) ≈ orthogonal to biology.** affective×inflammatory **0.07**, affective×metabolic
  **0.15**. With the *symptom* factor inside the same latent model, mood/anxiety is nearly uncorrelated
  with cardiometabolic and inflammatory load. Because this is a *within-model* factor correlation rather
  than a between-block average, it is a strong form of the claim.
- **Sleep ↔ affective = 0.68 — the one strong edge, and a finding to interrogate.** In BP/DR, the PSQI
  sleep factor and the affective factor are tightly coupled. Clinically unsurprising (insomnia *is* a
  depression symptom), but the magnitude is large enough that PSQI may be partly indexing
  depression-driven sleep complaints rather than a fully separable sleep axis. It is BP/DR-specific (SZ
  has no affective measurement), so we treat "sleep" as a **partly affective-entangled** dimension in
  the affective cohorts and flag a sensitivity analysis (sleep factor with affective residualized).
- **Metabolic ↔ inflammatory = 0.20** — separable but related: a single "cardiometabolic-inflammatory"
  axis resolves into two correlated-but-distinct loads.
- **Cognition** is modestly tied to **metabolic (0.26)** and **affective (0.23)**, ≈ orthogonal to
  inflammation/sleep — i.e. cognitive burden tracks metabolic load and distress a little, biology-of-
  inflammation not at all.

## 2 · Measurement model — what loads where

![Loadings by dimension](figures/v3/loadings.png)

Loadings are clean and clinically coherent (oriented so higher = more burden): adiposity (BMI/waist)
and WBC/neutrophils anchor metabolic/inflammatory; PSQI anchors sleep; MADRS/QIDS/STAI anchor affective;
verbal memory/processing-speed/TMT-B anchor cognition; the ISF items anchor suicidality. No Heywood
blow-ups under the marginalized estimator.

## 3 · Suicidality (mixed-likelihood) — a distress-linked near-standalone

Suicidality (Bernoulli + negative-binomial indicators) correlates with **affective 0.14** and **sleep
0.17**, and is ≈ orthogonal to cognition (−0.08), metabolic (−0.05), inflammatory (0.00). So it is
**not** a biology- or cognition-linked dimension; it sits closest to affective distress and disturbed
sleep, but only weakly — a standalone risk dimension with a modest, sensible distress link rather than
full independence.

## 4 · Dimension scores by cohort — clinical validation (diagnosis as a *check*, not a feature)

![Dimension scores by cohort](figures/v3/scores_by_cohort.png)

Diagnosis was **never** used to fit the dimensions; here we use it only to validate them. The scores
behave exactly as clinical knowledge predicts:

- **Cognition** — worst in **SZ** (highest burden), best in BP. Cognitive impairment is a schizophrenia
  hallmark → the cognition factor is measuring the right thing.
- **Sleep & affective** — worst in **DR** (depression), as expected.
- **Metabolic / inflammatory** — broadly **flat across cohorts** → biological load genuinely cuts across
  diagnoses (truly transdiagnostic), not a cohort marker.
- **Suicidality** — somewhat higher in **BP**.

⚠️ **Read the SZ affective box with care:** SZ has no affective indicators, so its affective score is a
*proxy* derived through the sleep↔affective correlation, not a measurement — it should not be compared
to BP/DR affective at face value.

## 5 · Informative missingness (MNAR)

The in-model MNAR arm is **not identifiable on the most-complete subsample** (cognition/suicide items are
~fully observed there by construction). The real MNAR result is the **full-sample atlas** (V3-2/V3-4):
cognition's informative missingness was partly BP-driven, and the robust informative-missingness sits in
**suicidality + self-report** completion (sicker patients skip them). A de-biasing joint selection model
on the full sample is future work.

## Caveats

- **~23% rare-pattern tail dropped** (`--min-group 10`) to keep the marginalized likelihood tractable —
  a mild extra completeness selection on top of the most-complete subsample. Lowering it (more Cholesky
  ops) is the next robustness step.
- **Affective + suicidality are BP/DR-anchored / cohort-heterogeneous**; SZ affective is a proxy.
- **Single visit (V0).** Temporal coherence (do these scores persist V1–V4?) is Phase H/C, not yet run.
- The suicidality↔factor correlations are read **post-hoc** from scores (mild attenuation toward 0).

## Bottom line

A certified, cohort-balanced, no-imputation transdiagnostic measurement model of BP/SZ/DR says:
**no general factor; symptoms (affective) ≈ orthogonal to biology; metabolic and inflammatory are
separable; cognition tracks metabolic/affective weakly; suicidality is a distress-linked near-standalone;
and sleep is strongly entangled with affect in the mood cohorts.** Next: temporal coherence + measurement
invariance (Phase H), then probabilistic strata (Phase J).
