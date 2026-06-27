# M4 — Prognosis: methods of record

> **The methods + math of record for Milestone 4.** Logic, estimand, the nested model ladder, the
> errors-in-variables engine, the clinical-endpoint layer, the acceptance gates, and the staged
> pipeline. Findings: [`PROGNOSIS_OOP_FINDINGS.md`](PROGNOSIS_OOP_FINDINGS.md); clinician atlas:
> [`PROGNOSIS_OOP_FINDINGS.md`](PROGNOSIS_OOP_FINDINGS.md).
> Read before any M4 modelling work. *Status: COMPLETE, pending PI sign-off. 2026-06-11.*
>
> **8-factor rebuild note (2026-06-27; pending PI sign-off).** The M4 method below is unchanged, but it is now
> run on the **8-factor** M1/M2/M3 objects (immunometabolic merge + substance orthogonal; **A = 5** archetypes).
> The engine already discovers the archetype/K-family encodings dynamically, so A=4→5 needed no live-path change;
> the only functional edit is `DURABLE = (cognition, immunometabolic)` (the merged durable biology, M3 ICC 0.91).
> Result **replays and is slightly stronger**: the **A = 5 archetypes predict 2-year functioning** (ΔELPD
> **+62.8** beyond DSM-5+severity+baseline, was +59; IPW-robust +54.4, permutation-null, co-informative with
> DSM-5, course-dependent/BP-led), **operative K = none**, **not** severity, and the archetype atlas is a
> **17→52%** functional-remission gradient with the **immunometabolic biology corner (A2) worst**. The
> durable-*pair*-alone EIV stays ambiguous (the archetype representation is the carrier). Both follow-up
> sub-analyses also re-ran on the 8-factor map: the within-cohort de-confounding (composition only ~4%, OR 6.3,
> BP-carried) and the raw-vs-map representation benchmark (sufficient for deterioration, ≥92% within-factor for
> recovery; xgboost under `OMP_NUM_THREADS=1`). Findings: [`PROGNOSIS_OOP_FINDINGS.md`](PROGNOSIS_OOP_FINDINGS.md).

## 1. Estimand and invariants

**The question.** Does a baseline coordinate or stratum on the transdiagnostic map predict a future
clinical outcome **incrementally beyond DSM-5 diagnosis + baseline severity + the baseline value of
that outcome**, built on the durable axes M3 certified (cognition, metabolic, inflammatory)? The sharp
hypothesis (the M3 → M4 throughline): *the durable TRAIT axes predict the future value of the moving
STATE outcomes.*

**Reframe (2026-06-10).** M4 is a clinician-facing **stratification-prognosis** demonstration, not only
an ΔELPD test. We forecast every clinically-useful state outcome; a predictor that shares items with
its outcome (sleep→PSQI, G→functioning) is the **autoregressive bar to beat**, not a credited finding.
The clean added-value test is cross-construct: the ⊥G durable biology / the archetypes predicting the
future functional state.

**House invariants.** V0 defines, follow-up validates (the map is *fixed* — M4 never re-discovers or
re-scores; it consumes the M3 panel). Observed-cell likelihood, **no imputation** (a V2-absent patient
contributes nothing; IPW, not fill-in, corrects selection). Diagnosis is metadata — here a *comparator*
and *validation grouping*, never a feature under test. A signal counts only if it clears its
uncertainty band. Internal validity only ("predicts" ≠ "causes"). Outcomes are repeated clinical
scales, **not** incident events.

## 2. Outcomes (configs/m4_outcomes.yaml)

Read native-scale from `data/processed/baseline_v{0,1,2}.parquet`, NaN-honest, cohort-scope-masked.
**Primary (3-cohort, PI-locked):** EGF (functioning) + CGI-S (severity), horizon **V2** (2-yr), V1 as
the secondary/replication horizon. Secondary: FAST, EQ-5D-VAS, MADRS, YMRS, PSQI, C-SSRS. The
predictor↔outcome item-overlap audit (M4.0) quarantines self-prediction from the *added-value* claim
(the durable trio loads on functioning/severity only as `g_anchor_on_specific`, a soft-zero — so the
durable → functioning test is clean).

**Clinical endpoints (M4.5, `src/face/prognosis/endpoints.py`)** — binary state transitions from the
V0→V1→V2 scales (GAF/CGI-S anchors): functional remission (GAF≥71), recovery, deterioration (GAF drop
≥10), sustained impairment (GAF<61 at V1 & V2), CGI-S remission (≤2), relapse surrogate (CGI-S rise ≥2),
sustained illness (CGI-S≥4 at V1 & V2).

## 3. The nested model ladder

Per outcome `Y`, horizon `T`, on one complete-case sample so held-out ELPD is comparable:

```
R0  age + sex + site(random intercept)
R1  + DSM-5 arm (7 subtypes)
R2  + severity        [CGI-S ; and the error-corrected G coordinate]
R3y + baseline Y      ← THE BAR (ANCOVA autoregression term; makes "incremental" honest, dodges RTM)
Tc  + durable coords {cognition, metabolic, inflammatory}   (EIV)   ← representation 1
Ta  + 8 archetypes  (Arm A full ; Arm B = G-residualized, the clean ⊥G)
Tt  + 4-region tessellation
Tf  + 8 specific coords (ceiling)
```

Win = held-out ΔELPD vs R3y with SE excluding 0 **and** the coefficient 94% HDI excluding 0, under
**both** severity operationalizations (Q2). The head-to-head vs DSM-5 (Q3) fits D / +DSM-5 / +map /
+both on a shared foundation and reads the dominance asymmetry (map-beyond-DSM-5 vs DSM-5-beyond-map).

## 4. The engine — errors-in-variables Bayesian GLM (`src/face/prognosis/glm.py`)

A bespoke NumPyro model (the M3 `variance.py` idiom): `g(E[Y]) = α + Xβ + u_site + ξ·β_eiv`, with
Gaussian / ordered-logit / Bernoulli likelihoods, a **non-centred** site random intercept, and
**errors-in-variables** predictors — the latent true coordinate `ξ_i ~ N(μ, τ)` with the **known M1
per-patient SD plugged** (`z_obs ~ N(ξ_i, S_i)`, non-centred), so wide-posterior coordinates
self-down-weight and β_eiv is attenuation-corrected. Per-stage gate: R-hat ≤ 1.01, ESS ≥ 400, 0
divergences, Pareto-k < 0.7. InferenceData is built via `az.from_dict` from raw samples (arviz≥1.1
`from_numpyro` re-traces the model and leaks a JAX tracer under the IPW scale handler). LOO/ΔELPD via
`compare.py`.

**Clinical-value layer (M4.6, `clinical_value.py`)** — patient-level 5-fold cross-validated logistic
models give the clinician's currency: AUC (paired bootstrap CI on the map's gain), calibration (Brier),
and decision-curve net benefit. Frequentist CV is the field standard (TRIPOD) and fast; the Bayesian
EIV ladder is the uncertainty-aware backbone.

## 5. Acceptance gates (Q1–Q4)

- **Q1 incremental validity** — held-out ΔELPD(map) vs R3y, SE excluding 0; coefficient HDI excluding 0.
- **Q2 beyond severity** — survives both manifest CGI-S and the error-corrected G severity.
- **Q3 transdiagnostic / vs DSM-5** — within-cohort consistency + the head-to-head dominance; "better"
  = outcome ELPD, never agreement with DSM-5.
- **Q4 robust / not artefact** — IPW (attrition), reliability-stratified (not prior-dominated),
  leave-one-cohort-out, permutation null. The measurement-error-in-baseline (Lord/RTM) concern is met
  by Q2's error-corrected-severity survival.

The milestone locks when, per primary outcome, Q1+Q2 pass and Q3+Q4 pass-or-documented-partial.

## 6. Pipeline (`scripts/40–48`, **RETIRED 2026-06-24**) and engine (`src/face/prognosis/`)

> The native driver pipeline below was **retired** — the canonical M4 is the copula OOP engine
> (`prognosis_model_oop.py`, [`PROGNOSIS_OOP_FINDINGS.md`](PROGNOSIS_OOP_FINDINGS.md)) plus the representation
> benchmark ([`M4_REPRESENTATION_BENCHMARK.md`](M4_REPRESENTATION_BENCHMARK.md)). The shared array-in/array-out
> kernels in `src/face/prognosis/` (glm · compare · clinical_value · frame · endpoints · reference ·
> transdiagnostic · robustness) and `tests/m4/` remain. The script list is kept as the native methods record.

`40_inventory` (feasibility + circularity audit) · `41_frame` (the EIV analysis frame + predictor draw
tensor) · `42_reference` (the R0–R3y bar) · `43_incremental` (Tc/Ta/Tt/Tf vs R3y; +Arm-B refinement) ·
`44_transdiagnostic` (head-to-head vs DSM-5 + course-dependence) · `45_endpoints` (clinical endpoints +
the archetype prognostic atlas) · `46_clinical_value` (AUC / calibration / net benefit) · `47_robustness`
(Q4 sweep) · `48_consolidate` (M5 hand-off). Engine: `frame · reference · glm · compare · endpoints ·
clinical_value · transdiagnostic · robustness`. Consumes the fixed `patient_panel`, `panel_draws`,
`patient_strata`, `ipw_weights`, `archetype_profiles`; nothing re-scored. Tests: `tests/m4/` (33).

## 7. Roadmap

M4 prognosis (this) → **M5 treatment** (does a stratum *moderate treatment response* — stratum ×
treatment interaction — the point at which a phenotype changes management). Hand-off:
`results/face/m4/{prognosis_summary.csv, prognosis_patient_risk.parquet}`.
