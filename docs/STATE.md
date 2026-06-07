# STATE — where the project is right now

> **Read this first.** Updated 2026-06-07.

## TL;DR

The project has been **replanned** around **Milestone 1 (M1): the transdiagnostic dimensional map** on the
FACE **V0** baseline. The methods and mathematics are **fixed** in
[`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md) (the single methods/plan of record). The previous
"Engine A" stage results (the old `03/04` Bayesian engine and its "no general factor" headline) are
**discarded** — superseded by the global, full-sample, explicit-latent approach in the methods doc. The
repository is on a **clean base**, and M1 implementation is underway. The data layer + the marginalized
measurement engine are built; **S1 (continuous core) and S2 (inter-dimension Φ + MADRS/QIDS/STAI windows)
are certified at full N (9,013)**, and **S3 is done** — S3a (+developmental-risk) certified and
resample-stable, S3b (+suicidality via a mixed-likelihood block) provisional — on a random N = 4,000
subsample (the §3.6 frontier fallback; full N reserved for the reported S5 fit). Updated 2026-06-07.

## What's decided

- **Model:** one **global** Bayesian sparse bifactor / ESEM — mixed likelihoods, soft priors,
  observed-cell likelihood (no imputation), **full V0 sample**. Estimated via a **staged continuation**
  (S1→S5); **only the global fit (S5) is interpreted.**
- **Confirmation:** **FIML** on the continuous backbone (masked-PAF dropped).
- **Dimension set (V0):** `G(severity)` · `cognition` · `metabolic` · `inflammatory` · `sleep` ·
  `suicidality` · `developmental-risk` (3-cohort) + `anhedonia` (BP/DR, thin). Dropped: impulsivity,
  negative symptoms, sensory.
- **Stack:** lean — PyMC + **NumPyro/JAX**. The marginalized (Woodbury) engine **certifies on the Mac M4
  (CPU)**; the RTX 4090 is optional (faster for later mixed-likelihood stages). YAML configs; Parquet
  model-ready persistence (raw stays CSV); per-stage reports; notebook later.
- **Repo:** package **`src/face/…`** (renamed from `src/v3`, tests green). Pipeline built so far:
  `scripts/01_build_data` (full-N V0 → Parquet) · `scripts/04_fit --stage {1,2}` (one canonical engine,
  `src/face/models/bayesian/continuous_core`: marginalized Woodbury default, explicit-latent + `--gpu`
  optional). S2 stage flags (`correlated`/`windows`/`specific_cross`) live in `scripts/04_fit`.

## What exists vs. not

- **Exists:** `src/face/data` (harmonization + skip-logic, no imputation); `configs/` ontology +
  `prior_loading_matrix_v3.csv` (143 indicators × 10 factors) + the **prior atlas**
  (`docs/PRIOR_ATLAS.md`); `scripts/01_build_data` (Parquet persistence) + `scripts/04_fit` + the
  single marginalized/explicit engine (`continuous_core`; the parallel config-first engine + its
  `bayesian_model.yaml` were retired — one canonical engine now); tests (`tests/v3/`, **88 passing**).
- **Results so far — S1 + S2 CERTIFIED (full N); S3 done (subsample):** see "S1/S2/S3 result" below.
- **Next (M1 build):** S4 (anhedonia, BP/DR thin) → **S5 global = the reported fit (full N)** → FIML
  confirmation → adjudication → scoring → the **empirical atlas + prior→posterior comparison**.
- **Compute lesson (this session):** full-N S1/S2 ≈ 1 h; the S3+ mixed-likelihood frontier is heavier, so
  S3 checkpoints use a random N=4,000 subsample (§3.6). Engine perf fixes: grouped-GEMM Woodbury (Cholesky
  per observed-pattern, 2.75×), tree-depth cap 8 + ta 0.85 (2.7× at 7 factors). Φ bug fixed (LKJCorr=Cholesky
  → Φ = L Lᵀ). The reported **S5** map targets full N (GPU per §4.5 if the Mac can't hold the mixed block).
- **Later milestones (not started):** strata (M2) · temporal coherence V1–V4 (M3) · prognosis (M4) ·
  treatment (M5).

## S1 result — continuous core (CERTIFIED, full N = 9,013, no imputation)

Marginalized (Woodbury) bifactor, NumPyro/JAX-CPU: **R-hat 1.010 · ESS 1,939 · 0 divergences**
(415,531 observed cells, ~72 min on the Mac). Factors: G + cognition/metabolic/inflammatory/sleep
(continuous block).
- **G = functional burden / illness severity**, anchored cleanly by functioning + global severity only
  (FAST 1.04, EGF 0.69, EQ-5D 0.58, CGI-S 0.54 — no symptom content by design).
- **Biology ⊥ G:** mean |loading on G| = metabolic **0.08**, inflammatory **0.07** vs cognition 0.27,
  sleep 0.22 — biology sits *off* the general-burden axis; cognition/sleep partly track it. The earlier
  explicit-latent run reproduced these loadings (the two parameterizations triangulate).
- *Continuous backbone only* (independent-specifics bifactor, Φ = I); cross-loadings, the symptom blocks,
  and inter-factor correlations come at S2–S5. **Full writeup + interpretation:
  [`RESULTS.md`](RESULTS.md) §S1.** Artifacts: `reports/04_stage1_report.md` + `_loadings.csv`.

## S2 result — inter-dimension Φ + MADRS/QIDS/STAI windows (CERTIFIED, full N = 9,013, no imputation)

Marginalized (Woodbury) ESEM, warm-started from S1: **R-hat 1.010 · ESS 676 · 0 divergences**
(J = 71 = 68 + 3 windows, 434,765 cells, on the Mac). Adds Φ (LKJ over specifics, G orthogonal) +
the depression/anxiety windows. **Re-certified after two engine fixes** (Φ = L Lᵀ from the LKJCorr
Cholesky; grouped-GEMM Woodbury, 2.75× faster, logp-identical) — Φ/loadings unchanged to 2 decimals.
- **Φ — specifics are weakly correlated** (mean |off-diag| 0.09): largest is metabolic↔inflammatory
  **0.20** (immunometabolic coupling, but distinct — supports the candidate-5 *split*); sleep ≈ orthogonal
  to biology. Distinct axes, not one factor.
- **Windows load on G** (functional burden): MADRS **0.80**, QIDS **0.77**, STAI **0.66**, with minor
  sleep side-loadings (QIDS 0.24, STAI 0.21) — **depression/anxiety are burden windows, not an 11th
  dimension** (as the methods doc hypothesised). No separate affective factor.
- **S1 survives elaboration:** primary loadings barely move and **biology ⊥ G holds** (metabolic 0.08,
  inflammatory 0.07 on G — identical to S1) — the headline was not a bifactor/independence artefact.
- **Identification finding:** metabolic↔inflammatory *mutual cross-loadings* are rotationally aliased with
  Φ_{metab,inflam} (not separately identifiable; freeing both ways also made full-N intractable) → **Φ
  carries that association**, mutual crosses left at 0 (standard ESEM resolution; a ridge-guarded
  sensitivity arm exists in the engine). **Full writeup: [`RESULTS.md`](RESULTS.md) §S2.** Artifacts:
  `reports/04_stage2_report.md` + `_loadings.csv` + `_phi.csv`.

## S3 result — developmental-risk (certified) + mixed-likelihood suicidality (provisional)

Random N = 4,000 subsample (§3.6 frontier fallback), grouped-GEMM + tree-cap8 + ta 0.85.
- **S3a — +developmental-risk, CERTIFIED** (6 factors; R-hat 1.010 · ESS 832 · 0 div). Developmental is its
  **own axis** (loading 0.41; ≈ orthogonal to biology and G; weakly tied to sleep +0.16). Continuous core
  unchanged from S2 (biology⊥G 0.09/0.07; metab~inflam 0.21; windows→G 0.81/0.76/0.65). **Resample-stable**
  (seed A vs B: |ΔΦ| ≤ 0.035, |Δloading| ≤ 0.012; continuous-core Φ matches full-N S2).
- **S3b — +suicidality via mixed-likelihood block, PROVISIONAL** (7 factors; explicit f_e=(G,suic,dev), the
  4 continuous specifics marginalized + coupled through Φ; 14 binary + 3 ordinal + 1 count; **0 divergences**;
  R-hat 1.06 · structural ESS 58 — not fully certified). **Suicidality is solidly identified** by its binary
  ISF items (ideation isf01–05 load +2.5…+3.3 on the logit scale **and** +0.4–0.56 on G; all R-hat 1.00,
  ESS 0.8–2.3k). **suicidality~developmental +0.22** (childhood adversity ↔ suicidality). The slow mixing is
  in the **continuous cross-loadings + the suic~dev Φ cell** (the conditional-coupling part), **not** the
  suicidality block — so suicidality loadings are trustworthy, Φ_suicidality is provisional (→ resolve at S5).
  **Answered the methods-doc S3 question: non-Gaussian indicators DO compose with the shared Φ.**
  **Full writeup: [`RESULTS.md`](RESULTS.md) §S3.** Artifacts: `reports/04_stage3{,b}_*`.

## Open methods choices (flagged for the PI)

Sparsity prior (soft-normal vs horseshoe) · Student-t vs Gaussian continuous default · item- vs
factor-level covariates · acceptance-gate numbers. Defaults are set in
[`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md); confirm or overrule before S1.

## What to read

[`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md) (methods + math) · [`RESULTS.md`](RESULTS.md) (findings log)
· [`PRIOR_ATLAS.md`](PRIOR_ATLAS.md) (prior map) · [`../README.md`](../README.md) (overview) ·
[`../CLAUDE.md`](../CLAUDE.md) (guide) · [`DATA.md`](DATA.md) (data contract).
