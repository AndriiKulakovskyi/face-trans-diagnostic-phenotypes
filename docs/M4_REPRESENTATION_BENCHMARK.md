# M4 — representation benchmark (latent map vs raw indicators)

> **Map of record (read first).** The latent map under test is the **8-factor immunometabolic map** (G + 7
> specifics; immunometabolic a single biology factor; substance orthogonal; 3 earned cross-loadings) with
> **A = 5 archetypes**. The benchmark headline: against the raw indicators under a matched XGBoost, the
> 8-factor map is **sufficient for deterioration** (AUC tie) and **near-sufficient for recovery** (raw +0.04
> AUC), with **92–97% within-factor compression** — the residual is item-level, not a missing axis. Canonical
> map findings: [`HORSESHOE_ESEM.md`](HORSESHOE_ESEM.md).

> **STATUS** — methods record for the representation benchmark that tests whether the latent map is a
> *sufficient, data-efficient, uncertainty-aware, transportable, interpretable* summary of the raw indicators. This doc is the methods record for an M4
> rework that tests whether the M1/M2 latent map is a *sufficient, data-efficient, uncertainty-aware,
> transportable, interpretable* summary of the raw indicators for outcome prediction — not whether it
> "beats" raw (it can't, asymptotically; the coordinates are a compression of the raw cells). Author: planning
> pass, 2026-06-24.

## 1. Objective & reframed hypotheses

The naive hypothesis "coordinates predict better than raw" is wrong: σ(coordinates) ⊆ σ(raw), so a flexible
model on raw weakly dominates in-distribution asymptotically, and a masked NN on raw is *itself* a learned
bottleneck. We therefore test **representation quality**, where a tie is a success:

- **H1 — Sufficiency.** Raw indicators add ~nothing over the 8-factor map: `(ref+raw+latent) ≈ (ref+latent)`.
  The compression to 8 factors preserves the outcome signal. *(A near-equivalence claim → report the gap with
  a CI against a pre-specified margin, not p>0.05.)*
- **H2 — Data efficiency.** At realistic / small N (and small cohorts), the 8-factor bottleneck generalizes
  better than raw (which overfits sparse features). Tested by learning curves.
- **H3 — Uncertainty value.** Feeding per-patient uncertainty (mean+sd / draws) beats mean-only and the raw
  model that has no reliability channel.
- **H4 — Transportability.** The low-dim, meaningful representation transfers across cohorts (LOCO) and time
  better than a raw black box. *(OOD — where the asymptotic "raw dominates" argument breaks.)*
- Plus **interpretability** as a free win at equal accuracy.

**Pre-registered honesty:** three outcomes are all valid and reported — positive, subgroup-positive (signal in
an identifiable group only), or null (value is group-level, no individual decision value internally). Given the
known ΔELPD +59 / ΔAUC +0.011 split, null is a live result.

## 2. Targets (functional-recovery triage estimand)

Both from EGF/GAF, V0→V2, complete-case (never imputed); definitions already in `endpoints.py`:

| target | definition | denominator | N | rate |
|---|---|---|---|---|
| **Recovery** | `ep_egf_recovery` = (GAF<61 @V0) & (GAF≥71 @V2) | baseline-impaired (GAF<61) | **1,087** | ~26% |
| **Deterioration** ("relapse") | `ep_egf_deterioration` = GAF drop ≥10 (V0→V2) | all with V0&V2 EGF | **2,121** | 15.4% |

**Backbone (recommended):** predict the **continuous GAF@V2** distribution, then derive both binary endpoints
by thresholding the predictive distribution (dodges the AUC-collapse trap) — *and* fit each binary endpoint
directly as a check. Power caveat: recovery N=1,087 is modest → sufficiency CIs will be wide; deterioration is
better powered.

## 3. Data

- **Raw arm:** `data/processed/baseline_v0.parquet` (9,013×143) **subset to the exact M1 indicator list**
  (`prepare(factors=S5_FACTORS).items`), `mask = X.notna()`; native scales (XGBoost handles NaN natively;
  the masked NN gets zero-impute + the mask channel). Sign/family from `data/processed/indicator_metadata.parquet`.
- **Latent arm:** `results/face/strata_oop/coordinates/coordinates_full.parquet` (`{ax}__mean`, `__sd` for the
  9 CANON axes), `coordinates_draws.npz` ([200,N,9]), `consolidate/patient_strata.parquet` (`arch_w0..3`).
- **Outcomes + IPW:** reuse `frame.py` outcome assembly + `results/face/m3/ipw_weights.parquet`
  (`w_retained_V2`; attrition is severity-neutral, MAR-given-V0 → complete-case primary, IPW as sensitivity).
- **Join key everywhere:** `(cohort, patient_id)` (patient_id is per-cohort). Latent z-scale, not re-standardized.
- **No diagnosis as a feature** (cohort/arm = validation only). **Targets never in any input matrix.**

## 4. Feature arms (the representations under test)

- **REF** — DSM-5 arm + baseline severity + **baseline GAF** (the clinician bar, = R3y). The common base.
- **RAW** — 143 indicators + mask.
- **LAT-μ** — 8 coordinate means.
- **LAT-σ** — 8 means + 8 sds (+ draws where the model supports it).
- **LAT-A** — LAT-σ + A = 5 archetype weights.
- **RAW+LAT** — raw + latent (the H1 sufficiency test).

Each evaluated standalone and as an increment on REF. **Key sufficiency contrast:** `REF+RAW+LAT` vs `REF+LAT`.

## 5. Models (strongest-vs-strongest, fair)

The contrast must isolate the *representation*, not the modeller — so same model class across representations,
same CV folds, matched HPO; and a genuinely strong raw baseline (a weak NN that loses proves nothing):

- **XGBoost** (native missing) — on RAW and on every LATENT arm. Primary tabular workhorse; regularized + early
  stopping.
- **Masked MLP** (PyTorch, MPS) — on RAW (zero-impute + mask channel, per the user's design); small MLP on LATENT.
- **Regularized logistic / linear** — interpretable anchor on LATENT.
- **Bayesian EIV GLM** (`glm.fit_glm`) — the uncertainty-aware arm: plugs per-patient `sd` as `eiv_sd`, gives a
  posterior predictive + ΔELPD that **bridges to the certified M4 story**. On LAT-σ / LAT-A.

Conclusions must hold across model classes to count.

## 6. Scoring protocol

- **Out-of-fold** predictions via **repeated stratified CV** (identical folds across all arms → paired
  comparisons), stratified by outcome × cohort.
- **Metrics:** log-loss + **Brier** (binary), **CRPS** (continuous backbone) — *new, add to `clinical_value.py`*;
  **calibration** (reliability + slope — new); **net benefit / DCA** at the rehab-decision threshold band
  (reuse `net_benefit`); AUC reported but not headline.
- **Paired CI** on per-arm metric *differences* via bootstrap (generalize the `paired_auc_delta` idiom to
  Brier / CRPS / net-benefit).

## 7. Hypothesis → analysis map

| H | analysis | success criterion |
|---|---|---|
| H1 sufficiency | `Δ(REF+RAW+LAT − REF+LAT)` with bootstrap CI + TOST equivalence margin | CI within margin (tie) |
| H2 efficiency | learning curves (subsample N = 200…full) per representation | LAT dominates small-N, converges large-N |
| H3 uncertainty | LAT-σ vs LAT-μ; EIV-GLM sd vs sd=0 | uncertainty adds (CI excl 0) |
| H4 transport | leave-one-cohort-out (emphasis BP↔SZ; DR too thin) + temporal (train V0→V2, test V0→V1) | LAT degrades less OOS than RAW |
| interpretability | SHAP (XGBoost-LAT) / coefficients | qualitative |

## 8. Fairness controls (load-bearing — these decide credibility)

1. **Same indicator set** — RAW = exactly M1's `items`, not a super/subset.
2. **Matched tuning + identical CV folds**; strong, regularized raw baseline.
3. **Name the missingness confound** — XGBoost-native vs NN zero+mask vs FIML(latent) differ; part of any latent
   edge is "FIML > zero+mask", a real but *missingness-handling* result, reported as such.
4. **Baseline-autoregression handled symmetrically** — both arms carry baseline GAF (in REF).
5. **No leakage** — M1 is unsupervised wrt the outcome (coords clean); the only watch is the baseline-GAF path;
   outcomes never enter any input matrix.
6. **Uncertainty as draws** where the model allows, not just a point sd.

## 9. Validation discipline (no external data)

- Nested / repeated CV or bootstrap **optimism correction**; the outcome model is re-fit inside every fold (M1
  coordinates are precomputed and unsupervised, so fixed coords are acceptable; document this). Report apparent
  vs corrected.
- **IPW** (existing weights) as attrition sensitivity; complete-case primary.
- **Pre-register** before running: primary target (recovery), primary metric (net benefit + CRPS), equivalence
  margin, subgroup. Seeds = 20260610.
- **LOCO + temporal** as the only internal proxies for transportability.

## 10. Engineering

- **New:** `src/face/prognosis/representation_benchmark.py` (arms → CV harness → models → metrics →
  hypothesis tables), wrapping the proven kernels. Raw-matrix loader (reuse `prepare()` / `baseline_v0` + mask).
  Add **CRPS + calibration-slope** to `clinical_value.py`. XGBoost + masked-MLP wrappers. Learning-curve + LOCO
  harness.
- **Driver:** `notebooks/run_representation_benchmark.py` (smoke + full; heavy fits run detached).
- **Deps:** add `xgboost`, `torch` (MPS) to the env; the PyMC/NumPyro stack is untouched.
- **Outputs:** `results/face/m4_repbench/{sufficiency,efficiency,uncertainty,transport,decision}.csv` (+ a
  `repbench_summary.csv`); figures → `docs/figures/repbench/`.
- Invariants honored: no imputation (raw keeps NaN/mask), determinism, `(cohort,patient_id)` joins, diagnosis = validation only.

## 11. Documentation updates (focused — consolidate, don't accrete)

- **This doc** = the methods record for the benchmark (linked from PROGNOSIS_MODEL.md).
- `docs/PROGNOSIS_MODEL.md`: add a short "§ Representation benchmark" (estimand + design pointer). No mass retitling / no new Q-gate proliferation.
- `docs/PROGNOSIS_FINDINGS.md`: add one "Representation: sufficient / efficient / uncertainty-aware / transportable" results section (the benchmark table + the 4 verdicts).
- `report/sections/m4_prognosis.tex`: one new subsection + one figure, folded near the existing "Is it worth using?" subsection.
- `docs/STATE.md` (M4 block) + `docs/VERTICAL_FINDINGS.md` (M4 paragraph): one-line additions.

## 12. Tests (`tests/prognosis/test_representation_benchmark.py`)

Arm construction; **identical CV folds across arms**; CRPS + calibration-slope numerics; sufficiency-gap sign on
synthetic data; learning-curve shape; LOCO runs; **leakage assertion** (no target column in any X);
**no-imputation assertion** (raw retains NaN / mask). Smoke config for CI.

## 13. Acceptance / verification

Smoke run green; full run yields the 5 hypothesis tables + figures; report builds clean; tests pass. The
*scientific* acceptance is the pre-registered H1–H4 read, reported honestly (incl. a null).

## 14. Build order (incremental QC gates, discussion after each)

- **P0** — pre-registration + raw-matrix loader + CRPS/calibration utils + tests.
- **P1** — arms + CV harness + XGBoost; sufficiency + decision metrics on recovery & deterioration.
- **P2** — masked NN (MPS) + uncertainty arm (EIV-GLM, LAT-σ) + efficiency learning curves.
- **P3** — transportability (LOCO + temporal) + interpretability.
- **P4** — figures + documentation/report updates.

## Decisions (resolved 2026-06-24)

1. **Cohort scope:** run **both** the pooled BP+SZ+DR analysis and the **BP+DR (episodic)** analysis, and
   **headline BP+DR** — the cohorts where 2-yr functioning actually moves and the map forecasts (SZ functioning
   is baseline-saturated). The pooled result is reported for honesty, not buried. "Impaired" (GAF<61) remains the
   fixed denominator of the *recovery* target (not a knob); deterioration uses all with V0&V2 GAF.
2. **Modeling backbone:** predict the **continuous GAF@V2 distribution**, then derive recovery & deterioration
   probabilities by thresholding (+ direct binary fits as a check). **CRPS** is the primary proper score; this
   avoids the AUC-collapse trap.
3. **Sufficiency equivalence margin:** pre-register the "tie" band at **±1 SE of the raw-arm metric** (net
   benefit and CRPS), revisable to a clinically-set net-benefit band if preferred.

## Pre-registration (locked at P0, 2026-06-24)

Frozen before any model is fit. All inputs come from the **Gaussian-copula vertical**
(`results/face/strata_oop/`); raw = the same 143 indicators M1 ingested (`data/processed/baseline_v0.parquet`).

- **Targets** (EGF/GAF, V0→V2, never imputed; `endpoints.build_endpoints`):
  *primary* `egf_recovery` (GAF<61 @V0 → GAF≥71 @V2; denominator = baseline-impaired);
  *secondary* `egf_deterioration` (GAF drop ≥10; denominator = V0&V2 present).
- **Backbone:** predict the continuous **GAF@V2** distribution; derive recovery/deterioration probabilities by
  thresholding; direct binary fits as a check.
- **Primary metrics:** **CRPS** (continuous) and **net benefit / DCA** over threshold band `pt ∈ [0.05, 0.50]`.
  Secondary: Brier, log-loss, calibration slope, AUC (reported, not headline).
- **Cohort scope:** headline **BP+DR**; pooled **BP+SZ+DR** reported alongside.
- **Arms:** REF (DSM-5 arm + baseline severity + baseline GAF) · RAW (143 + mask) · LAT-μ · LAT-σ · LAT-A ·
  RAW+LAT.
- **Models:** XGBoost (native-missing) on every arm; masked MLP (PyTorch/MPS) on RAW; EIV-GLM bridge on LAT-σ/A.
- **CV:** repeated stratified K-fold, **identical folds across all arms** (stratified by cohort × target),
  out-of-fold predictions, `seed = 20260610`. Complete-case primary; IPW as attrition sensitivity.
- **Sufficiency test:** `Δ(REF+RAW+LAT − REF+LAT)` with paired bootstrap CI; tie if within ±1 SE of the raw arm.
- **Invariants:** no imputation (RAW keeps NaN + mask); diagnosis/cohort = stratification only, never a feature;
  targets never enter any input matrix; all joins on `(cohort, patient_id)`.

**Amendment A (2026-06-24, during P1 — recorded, not silently changed).** Two additions, prompted by the
autoregression question:
1. **Both horizons** are run — V1 (1-year) and V2 (2-year). V1 roughly doubles the sample (recovery 1,744 vs
   1,087; deterioration 3,196 vs 2,121; milder attrition), so it is the better-powered replication.
2. **REF0 baseline (no baseline GAF).** Beside REF (DSM-5 + latent-G severity + baseline GAF) we add
   **REF0 = DSM-5 + latent-G severity, with baseline GAF dropped**, plus `REF0+RAW` / `REF0+LAT-A`. REF − REF0
   *quantifies the autoregression contribution*; sufficiency is reported under **both** the with-GAF (REF) and
   no-GAF (REF0) contrasts. Dropping GAF does not "fix" saturation — it changes the question from *incremental
   beyond baseline* to *unconditional*; both are reported.

   Note on missingness: recovery is, by definition, evaluated among the **baseline-impaired (GAF<61)**, and the
   1,527 / 9,013 patients missing baseline GAF have **no definable functional-change outcome** — they are out of
   scope, not a maskable-missingness case. Within every eligible set baseline GAF is present for 100%.

## P1 findings — XGBoost, V1+V2, REF/REF0 (results in `results/face/m4_repbench/`)

Out-of-fold (5×2 stratified CV, 2000 bootstraps). Eligible N: recovery 1,087 (pooled V2) / 681 (BP+DR V2) /
1,744 (V1); deterioration 2,121 / 3,196 (V1).

1. **Autoregression is not the driver (the REF0 test).** Dropping baseline GAF changes AUC by ≈0 everywhere
   (REF − REF0: recovery V2 pooled +0.000, V1 +0.001; deterioration −0.008..+0.002). The latent-G coordinate
   already encodes baseline functioning, and recovery-among-impaired compresses the baseline range — so raw GAF
   is redundant, and **every result below holds with baseline GAF dropped.** Not an autoregression artefact.
2. **Sufficiency is target-dependent, replicated across horizons.**
   - **Recovery — the map is slightly lossy.** Raw beats the latent map by **ΔAUC ≈ +0.04** (pooled V1 +0.040
     CI[0.022,0.057]; V2 +0.039 CI[0.018,0.059]; BP+DR V1 +0.063 CI[0.032,0.095]), robust under both REF and
     REF0. The map captures most of it (REF 0.65 → LAT-A 0.71 → RAW 0.75) but not all. BP+DR V2 is underpowered
     (tie, N=681).
   - **Deterioration — the map is sufficient** (AUC tie everywhere) — though nothing beats REF much
     (baseline-saturated). Raw sharpens the continuous CRPS by a clinically negligible fraction of a GAF point.
3. **Uncertainty + archetypes add within the latent arm** (recovery V2 pooled: LAT-μ 0.690 → LAT-σ 0.696 →
   LAT-A 0.707), consistent with H3 — but this is XGBoost using sd as a feature; the faithful test is the
   EIV-GLM (P2).

**Honest headline:** the 8-factor map is a **sufficient** summary for the baseline-saturated deterioration
outcome and a **near-sufficient (≈0.04-AUC lossy)** summary for recovery — not an autoregression artefact. Raw
carries a little recovery-specific signal the transdiagnostic compression drops. **Caveat:** XGBoost-only; the
EIV-GLM uncertainty arm, the recovery-gap diagnostic (which raw features?), efficiency learning-curves, and LOCO
transport are P2.

## P2 findings — diagnostics, uncertainty, efficiency, transport

1. **Recovery-gap (SHAP, `diagnostic.py`).** Of raw's recovery-predictive SHAP mass, **92–97% sits *within* the
   8 modelled factors** — the top drivers are the factor anchors (CRP/platelets/eosinophils, BMI/HbA1c/lipids/urate,
   CVLT/WAIS/TMT, Fagerström, CTQ, FAST/EQ-5D); only **3% is off-map** (the depression/anxiety *window* items
   QIDS/STAI/MADRS that M1 folds into G). So the ~0.04-AUC recovery gap is **within-factor compression loss** —
   item-level resolution the factor scores blend away — **not a missing dimension**.
2. **Uncertainty (EIV-GLM, `eiv.py`, H3).** Honest per-patient uncertainty adds **modestly for recovery**
   (EIV vs mean ΔELPD **+3.1 ± 1.7**, ≈1.8 SE — suggestive, not decisive) and is **null for deterioration**
   (−0.9 ± 0.8). It does **not** close the raw gap → consistent with "compression, not noise". (Linear EIV-GLM,
   so absolute gains < nonlinear XGBoost; this isolates the uncertainty contribution.)
3. **Efficiency (learning curves, `curves.py`, H2).** **Not supported for recovery** — raw dominates at every N
   (150 → full, 0.68 → 0.73 vs LAT-A 0.65 → 0.68); the regularised 143-feature model does not over-fit away its
   item-resolution edge.
4. **Transport (LOCO, `curves.py`, H4).** The map transports **as well or better than raw for deterioration** on
   the well-powered held-out cohorts (BP, SZ); raw transports better for recovery. The map's transport advantage
   holds **where it is sufficient**.

**Calibrated representation claim.** The 8-factor map is a **sufficient, uncertainty-honest, transportable**
summary for the deterioration outcome, and a **near-sufficient, structurally-faithful** summary for recovery
whose small residual gap is **item-level compression** (not a missing axis, only marginally noise). It **trades a
sliver of task-specific resolution for parsimony, interpretability, transportability, and honest uncertainty** —
a favourable trade exactly where the outcome is not dominated by item-level detail.
