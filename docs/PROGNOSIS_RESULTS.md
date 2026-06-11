# M4 — Prognosis: development record

> Per-stage dev log, methods choices, refinements, and engineering lessons. Paper-facing synthesis:
> [`PROGNOSIS_FINDINGS.md`](PROGNOSIS_FINDINGS.md); methods: [`PROGNOSIS_MODEL.md`](PROGNOSIS_MODEL.md).
> Aggregates `reports/40–48`. *2026-06-11.*

## Per-stage record

- **M4.0 inventory** (`reports/40`). Verified the data contract: outcomes live in
  `baseline_v{0,1,2}.parquet`, native scale, `(cohort, patient_id)`-indexed. Paired V0→V2 N: EGF 2,121,
  CGI-S 2,345 (3-cohort); FAST/MADRS/C-SSRS BP/DR-only. Circularity audit (data-derived): durable trio
  → functioning/severity loads only as `g_anchor_on_specific` (soft-zero) → clean; G + same-construct
  pairs are the autoregressive bar. **No incident-event register exists** → de-scope to scale
  trajectories (flagged).
- **M4.1 frame** (`reports/41`). One row per V0 patient: coords+SD, archetypes A/B + tessellation,
  covariates (from `m2/validation_table`), native outcomes + derived endpoints, IPW. Predictor draw
  tensor aligned to the frame (corr(draw-mean, panel-mean) = 0.9955; the 0.26 max gap was the 200-draw
  Monte-Carlo error, not misalignment — QC corrected to report the correlation).
- **M4.2 reference** (`reports/42`). R0→R3y bar, all R-hat 1.0, Pareto-k < 0.52. EGF bar ELPD −2566,
  CGI-S −3022; diagnosis + severity + baseline already predict strongly → the increment must beat R3y.
- **M4.3 incremental** (`reports/43`). Headline: EGF +archetypesB ΔELPD **+46**, +tessellation +47,
  durable coords +7 (ambiguous); metabolic β −0.062 and inflammatory −0.060 exclude 0; metabolic
  survives the G severity (Q2). CGI-S: map adds ~0 (autoregression-saturated). **Refinement:** added
  Arm-B (G-residualized) archetypes (≈80% of Arm-A's gain is ⊥G), and non-centred the EIV latent to
  cure the +specifics8 funnel (R-hat 1.17 → still partial for specifics8, which is the redundant
  ceiling; the headline rests on the converged Arm-B/tessellation).
- **M4.4 head-to-head** (`reports/44`). Dominance = **co-informative** (EGF B−A +47, B−C +40) — the map
  complements DSM-5. Within-cohort: BP +43, SZ −1.8, DR thin → **course-dependent**. The SZ-null probe
  (OLS): foundation R² 0.26 (SZ) vs 0.17 (BP), equal outcome variance / archetype spread → **foundation
  saturation**, not map failure. Corrected the report wording: the cohort×map interaction null is
  underpowered, not proof of homogeneity — the within-cohort fits are the evidence.
- **M4.5 endpoints + atlas** (`reports/45`). Functional remission 14%→60% across archetypes, all
  transdiagnostic. Archetypes separate the dynamic transitions (remission/deterioration/relapse) better
  than DSM-5; DSM-5 the severity-level/sustained outcomes. Heatmap re-oriented by adversity
  (polarity-aware) so green=favourable uniformly.
- **M4.6 clinical value** (`reports/46`). Reference AUC 0.73–0.87; map adds reliably only for functional
  remission (+0.017 [.009,.026]); decision curves overlap for adverse endpoints. The +46-ELPD /
  +0.017-AUC gap = continuous→binary collapse.
- **M4.7 robustness** (`reports/47`). Durable β survives IPW + reliability; permutation null p=0.001;
  weakens dropping BP (course-dependent). Remission AUC gain survives IPW (+0.016), vanishes LOCO.
- **M4.8 consolidation** (`reports/48`). `prognosis_summary.csv` + per-patient
  `prognosis_patient_risk.parquet` (archetype + CV remission/deterioration risk, N=2,114).

## Engineering lessons

- **arviz ≥ 1.1 is a different API.** `az.summary` uses `ci_prob` + ETI columns; `from_numpyro` returns
  a `DataTree` and does **not** attach `log_likelihood`; `ELPDData` renamed `elpd_loo→elpd`,
  `loo_i→elpd_i`, `p_loo→p`. Decisive fix: build InferenceData via **`az.from_dict` from raw samples**
  (posterior + manually-computed per-obs log-likelihood + divergences) — `from_numpyro` re-traces the
  model and **leaks a JAX tracer** under the IPW `scale` handler. The from_dict path is equivalent for
  unweighted fits and handles weights.
- **Non-centre the EIV latent** (`ξ = μ + τ·raw`) — the centred parameterization funnels at high K.
- **A stratification's value is group-level.** The continuous ΔELPD / per-archetype outcome rates make
  the signal visible where the individual binary ΔAUC looked modest — same signal, decision-relevant
  granularity. The demo-layer reframe (endpoints + atlas + clinical value) was the right call.
- **Frequentist CV for the clinical metrics, Bayesian EIV for the inference** — each tool where it fits.

## Test + provenance

`tests/m4/` (33): config, frame, reference, GLM (incl. EIV de-attenuation), compare, transdiagnostic,
endpoints, clinical-value, robustness. Branch `m4-prognosis`, commits `c5d789e … 48`. Results gitignored
under `results/face/m4/`.
