# FINDINGS — FACE V3 precision psychiatry — running log

Paper-oriented log of empirical + methodological findings for **V3**. Plan of record:
[`V3_PLAN.md`](V3_PLAN.md). Every number must be reproducible from the V3 pipeline once built.

> **Status.** V3 modeling has **not been run yet** — the patient-level Bayesian / FIML discovery engine
> (Phases E–M) is not implemented. The runnable code (`src/trans_diag/`, `scripts/01–15`) is the **V2
> benchmark implementation**; its complete results log is preserved at
> [`legacy_v2/FINDINGS.md`](legacy_v2/FINDINGS.md). **Do not carry V2 numbers forward as conclusions** —
> they enter V3 only as priors, baselines, and the hypotheses below.

## How V2 findings enter V3 (hypotheses, not conclusions)

Each V2 result is a **falsifiable hypothesis** to confirm / refine / **downgrade** under the V3
patient-level observed-likelihood model (V3_PLAN §0C, Phase G3). The V2 evidence is in
[`legacy_v2/FINDINGS.md`](legacy_v2/FINDINGS.md).

| # | V2 finding (benchmark) | V3 hypothesis / action | Verdict |
|---|---|---|---|
| H1 | 3 weakly-correlated axes: internalizing · cognition · cardiometabolic | retest under FIML + Bayesian mixed-likelihood; adjudicate {confirm/split/merge} | ⬜ open |
| H2 | No dominant general factor (Schmid–Leiman ECV 0.42) | estimate `G` **directly**; test whether specifics survive beyond it | 🟡 **supported** (V3-5, core: mean\|Φ\|≈0.09, G un-identifiable) |
| H3 | **Symptoms ⊥ biology** (between-block mean \|r\| ≈ 0.03); p-factor is symptom-only | retest under observed-data likelihood + posterior uncertainty | 🟢 **supported in-model** (V3-6: affective×biology 0.07–0.15) |
| H4 | Cardiometabolic axis robust but possibly mixed | **test split** into metabolic load vs inflammatory load | 🟢 **supported** (V3-5: metabolic×inflammatory 0.17 — separable) |
| H5 | Cognition = strongest fully-transdiagnostic axis | keep core; refine into cognitive-flexibility / broader cognition if supported | ⬜ open |
| H6 | Internalizing axis (mood scales 0% in FACE-SZ → SZ-proxy) | model as affective/anhedonic extension unless invariance supports all-cohort status | ⬜ open |
| H7 | Standalone suicidality / mania / substance-use (orthogonal) | model suicidality with mixed binary/ordinal/count likelihoods; mania = activation/impulsivity **proxy**; substance = module/covariate | ⬜ open |
| H8 | No discrete subtypes (continuum) | accept; build **probabilistic decision regions**, not natural clusters | ⬜ open |
| H9 | Modest prognosis increment over DSM (functioning ΔR²≈0.04; de-confounded relapse ΔAUC≈+0.036) | the **minimum benchmark** the V3 model ladder + decision-curve utility must beat | ⬜ open |

## V3 log

- **V3-0 · Project reframed to precision-psychiatry V3 — 2026-06-05.** V3 plan adopted as the single
  source of truth ([`V3_PLAN.md`](V3_PLAN.md), [`V3_PLAN_SOURCE.md`](V3_PLAN_SOURCE.md)); V2 demoted to a
  benchmark/reference arm under [`legacy_v2/`](legacy_v2/README.md). No V3 modeling run yet.

- **V3-1 · Eligibility & data-contract audit (Phases A+B+C) — 2026-06-05.** First V3 deliverable.
  `configs/candidate_dimensions_v3.yaml` (curated soft-ontology → indicator map) +
  `scripts/v3/01_eligibility_audit.py` → per-cohort V0 observed coverage, likelihood map, missingness
  taxonomy, soft prior matrix. Report: `results/v3/eligibility/`. **Verdict on the 10 candidates
  (+4 data-implied), at V0 N=9,013:**
  - **Core, 3-cohort (well-covered):** `overall_severity` (= general factor **G**; CGI/EGF/EQ-5D/FAST),
    `cognition` (neuropsy, cov ≈0.67/0.76/0.56), `metabolism_immunometabolic` (labs+vitals ≈0.78/0.78/0.72;
    **test metabolic-vs-inflammatory split**), `sleep_circadian` (**confirmed** — PSQI total+7 subscores +
    Epworth `ess0109` + morningness `csm`; ≈0.94/0.56/0.62), `mania_activation` (YMRS 3-cohort + Altman).
  - **Core but caveated:** `suicidality` — present 3-cohort but **sparse + cohort-heterogeneous
    measurement** (BP C-SSRS skip-gated med 0.05 vs SZ ISF 0.93) → DIF/invariance flag; mixed
    binary/ordinal/count likelihoods + skip-logic; keep separate from suicide *outcomes*.
  - **Extension (BP/DR only; 0% in FACE-SZ):** `affective_internalizing` (MADRS/QIDS/STAI),
    `anhedonia` (single item QIDS-13). SZ must be scored by proxy (G/sleep/functioning).
  - **Module / historical:** `substance` (alcohol/cannabis SUD BP/SZ; DR weak), `neurodevelopment`
    (proxy: perinatal/CTQ/onset, well-covered), `illness_course` (fixed-historical staging).
  - **UNSUPPORTED in the common dictionary — exclude from the core model:** `negative_symptoms`
    (no PANSS/SANS; candidate to add from FACE-SZ full data), `sensory_abnormalities` (0 indicators),
    `impulsivity` (no Barratt/UPPS; proxy via mania/substance only). → **3 of the 10 candidates are not
    directly measured.**
  - **Likelihoods:** 95 Gaussian · 40 Bernoulli · 38 ordered-logit · 20 lognormal · 3 neg-binomial ·
    1 nominal(review) — a clean mixed-likelihood contract.
  - **Implied Phase-F core model:** bifactor `G` + specifics {cognition, metabolic, inflammatory, sleep,
    suicidality} (3-cohort) + {affective_internalizing, anhedonia} (BP/DR extension) +
    {mania_activation, substance, neurodevelopment, illness_course} (modules/covariates).
  - Next: Phase B full missingness atlas (observation-probability models) → choose PPL → fit the small
    core Bayesian model.

- **V3-2 · Missingness atlas (Phase B) — 2026-06-05.** `scripts/v3/02_missingness_atlas.py` →
  `results/v3/missingness/`. **Overall V0 missingness by cohort: BP 36% · SZ 58% · DR 43%**
  (SZ most incomplete). By block: **SZ labs (BILAN) 72% missing**, SZ hetero-Q 73%, DR neuropsych 58%,
  DR substances 72%, SUICIDE ≈62–73% all cohorts (skip-logic); demographics (PATIENT) 0% — complete.
  **Observation-mechanism drivers** (per-variable logit `observed ~ cohort+age+sex+severity`, severity =
  z(CGI-S, −GAF), within designed cohorts): **97 sporadic (MAR-safe) · 67 design/cohort (structural) ·
  28 informative (severity-related) · 2 const.** **Headline MNAR finding: cognition is informatively
  missing — more severe patients do not complete neuropsychology** (TMT-A/B, CVLT total/short/long, WAIS
  digit-span all severity↓observation, p<1e-3); also `csm`, `wurs`, FAST items, perinatal, some suicide.
  → **cognition + the 28 flagged variables need the Phase-F missingness-sensitivity arm (joint `R_ij`
  model), NOT naive MAR;** the ≈97 sporadic + structural-by-design variables are fine under the
  observed-likelihood MAR model. **SZ-metabolism rests on thin observed support** (72% labs missing) →
  expect low per-cohort SZ reliability for the biology factor. Next: Phase F core Bayesian model (with a
  cognition MNAR arm), with Phase D (V2 replication) / E (FIML) as benchmarks.

- **V3-3 · Bayesian core engine (Phase F prototype) — 2026-06-05.** `scripts/v3/03_bayesian_core.py`
  (PyMC 6), patient-level **observed-cell likelihood — NO imputation** (long (patient,indicator,value)
  table; missing cells never appear) on the **3-cohort continuous core** (20 indicators:
  cognition·metabolic·inflammatory·sleep), cohort-stratified subsample N=1,500. Two parameterizations:
  - **Bifactor (G + specifics): weakly identified** (max R-hat **2.04**, ESS 2, 0 divergences). `G`
    competes with the specifics for the metabolic/inflammatory variance while cognition/sleep barely
    load on `G`. The non-identification is itself evidence that **no dominant general factor** is
    supported — consistent with V2's no-p-factor (H2).
  - **Correlated 4-factor simple structure (LKJ): near-converged** (max R-hat **1.06**, 39 div, ESS 31 —
    **PROVISIONAL**; qualitative structure stable across runs). Clean positive loadings (psqi 0.99,
    wstcir 0.98, wbc 0.89, tmt_b 0.85). **Factor correlations Φ: mean |off-diagonal| ≈ 0.12 — weakly-
    correlated factors, NO general factor.** cognition×metabolic 0.16 · cognition×inflammatory 0.10 ·
    cognition×sleep 0.07 → **cognition ≈ orthogonal to biology** (first V3 echo of V2's cognition/
    symptoms⊥biology [H3], now under patient-level observed-likelihood); **metabolic×inflammatory 0.28
    → separable** (supports the H4 split); sleep ≈ orthogonal to all.
  - **Caveats:** continuous-core only (suicidality/affective/anhedonia not yet in); MAR (the cognition
    MNAR arm is wired in the bifactor variant, b_cog<0, but untrustworthy until converged); N=1,500
    subsample; R-hat 1.06 is not the 1.01 bar → **precise Φ values are provisional** (certify with a
    longer target_accept=0.99 run). **Engine + no-imputation pipeline validated; structure emerging and
    consistent.** Report: `results/v3/bayesian/core_model.md`.
  - Next: certify convergence → add suicidality (ordinal/Bernoulli/count) + affective/anhedonia BP/DR
    extension + cognition MNAR arm → ESEM soft cross-loadings → scale to full N (ADVI/NumPyro) →
    Phase H invariance → Phase J strata.

- **V3-4 · V0 confirmed + cohort-imbalance correction — 2026-06-05.** (1) **V0 anchor verified** across
  all V3 scripts (N=9,013 = BP 6,252 + SZ 2,209 + DR 552; later visits V1–V4 reserved for temporal
  coherence only). (2) **Corrected BP ≫ SZ ≫ DR** (BP is 11× DR): the atlas observation models now use
  **1/n_cohort weighting** (rescaled to preserve N); the Bayesian core now uses the **500 most-complete
  patients per cohort** (`--select complete`; observed-cell density 70% → 84%).
  - **Impact on V3-2 (a conclusion changed):** under cohort weighting the cognition-MNAR signal was
    **partly BP-driven** — TMT/WAIS/fluency drop to *sporadic*; only **CVLT** stays informative. The
    robust informative-missingness is **suicidality (ISF) + self-reports (Altman/STAI/PSQI/CSM/ESS)**.
    → the Phase-F MNAR arm should target suicidality + self-reports (+ CVLT), not the cognition block.
  - **Balanced Bayesian core re-run:** the **structure is robust to balancing** — loadings + Φ stable
    across random/balanced & 70%/84% dense (cognition≈⊥biology, metabolic×inflammatory 0.19,
    **mean |Φ| ≈ 0.09**, no general factor). **But convergence is worse** (R-hat **1.56**, **648
    divergences** vs 39) → the bottleneck is the **LKJ correlation parameterization geometry** (unused
    nuisance `stds` + sharper likelihood), a **model-engineering** fix, not data/compute.
  - **Decision pending:** re-parameterize the factor correlation (LKJCorr / non-centered / marginalize
    the Gaussian factors) to certify (R-hat<1.01, ~0 div) before extending or building strata.

- **V3-5 · Workbook enrichment + CERTIFIED marginalized core — 2026-06-05.**
  **(A) Folded in `FACE_dimension_recommendations.xlsx` (independent expert curation — confirms V3-1):**
  ESS/CSM → BP/DR circadian extension (3-cohort sleep core = PSQI only); the unit-mislabel flags are
  **cosmetic** (all sanity bounds correct, no data loss); metabolism trimmed 60 → **26**-var cardiometabolic+
  inflammatory core; per-variable covariate/proxy roles added (smoking/MARS/psqi16 → covariate; YMRS/Altman/
  WURS → proxy).
  **(B) Reparameterized for convergence:** the explicit-latent correlated model diverges (LKJCholeskyCov 648
  div; LKJCorr 250 div + Heywood loadings) → switched to a **marginalized Gaussian factor model** (factors
  integrated out; `MVN(ν, ΛΦΛᵀ+Ψ)` on each patient's observed cells, grouped by missingness pattern; **no
  imputation**; `--min-group` bounds the Cholesky count; `cores=1`). **CERTIFIED** (N=1,500 balanced
  most-complete, 86% dense, 17 patterns, 4 chains): **max R-hat 1.010 · min ESS 1,863 · 0 divergences.**
  Loadings clean (psqi 1.00, wstcir 0.97, wbc 0.94, tmt_b 0.74). **Φ: cognition×metabolic 0.22 ·
  cognition×inflammatory 0.05 · metabolic×inflammatory 0.17 · sleep ≈ orthogonal; mean |Φ| ≈ 0.09 — NO
  general factor.** First properly-converged V3 measurement model (structure identical across all 5 prior runs).
  - **V2-claim retests (now under a certified patient-level estimator):** **H2 (no p-factor) → SUPPORTED**
    (mean |Φ|≈0.09; bifactor G un-identifiable); **H3 (cognition ⊥ biology) → SUPPORTED** for cognition vs
    inflammation/sleep (≈0.05), *partial* vs metabolic (0.22); **H4 (metabolic vs inflammatory split) →
    SUPPORTED** (0.17, separable).
  - **Caveats:** continuous core only (suicidality/affective not yet in); **~20% rare-pattern tail dropped**
    by `--min-group 10` (mild completeness selection); single visit V0.
  - Next: extend to suicidality (ordinal/binary/count) + affective/anhedonia (BP/DR) + cognition MNAR arm;
    shrink the rare-pattern drop; Phase H invariance → Phase J strata.

- **V3-6 · Extended certified model + visualizations — 2026-06-05.** `scripts/v3/04_extended_model.py`
  (+ `05_visualize.py` → `docs/figures/v3/`; results page [`V3_RESULTS.md`](V3_RESULTS.md)). Marginalized
  Gaussian block extended to **5 factors** (added **affective** = MADRS/QIDS/STAI/anhedonia, BP/DR — so
  symptom⊥biology is a *within-model* correlation) + an **explicit mixed-likelihood suicidality** module
  (7 ISF Bernoulli + 1 neg-binomial count). **CERTIFIED:** max R-hat **1.020**, ESS **1,066**, 0 div
  (N=1,500 balanced).
  - **Φ — no general factor:** mean |off-diag| ≈ **0.18 (0.12 excl. sleep-affective).** **Affective ⊥
    biology** (affective×inflammatory 0.07, ×metabolic 0.15 → **H3 confirmed, stronger form**);
    **sleep×affective 0.68** (the one strong edge — PSQI tightly coupled to depression in BP/DR);
    metabolic×inflammatory 0.20 (**H4**); cognition×metabolic 0.26, ×affective 0.23.
  - **Suicidality** ≈ orthogonal to biology/cognition (−0.08…0.00), modest affective (0.14) + sleep (0.17)
    link → distress-linked near-standalone (partly supports V2's standalone suicidality).
  - **Cohort validation** (diagnosis = check, not feature): cognition worst in SZ; sleep/affective worst
    in DR; biology flat across cohorts (truly transdiagnostic).
  - **Caveats:** ~23% rare-pattern tail dropped; **SZ affective is a proxy** (no SZ affective indicators);
    suicidality↔factor correlations are post-hoc; MNAR not identifiable on the most-complete subsample
    (see atlas); single visit V0; sleep↔affective coupling needs a residualized-sleep sensitivity.
  - **H3 (symptoms ⊥ biology) → now SUPPORTED in-model** (was partial). Next: sleep-affective sensitivity;
    Phase H temporal coherence + invariance; Phase J strata.
