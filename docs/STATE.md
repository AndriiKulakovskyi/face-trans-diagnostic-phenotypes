# STATE — where the project is right now

> **Read this first.** Updated 2026-06-07.

## TL;DR

**Milestone 1 (M1) — the transdiagnostic dimensional map — is COMPLETE** (pending PI sign-off), on the FACE
**V0** baseline (N = 9,013). Methods of record: [`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md); verdict:
[`ADJUDICATION.md`](ADJUDICATION.md). The map is **9 transdiagnostic dimensions** — a general factor **G
(functional burden)** + **cognition, metabolic, inflammatory, sleep, developmental-risk, suicidality,
mania, substance** — estimated from observed cells only (no imputation), via one global Bayesian sparse
bifactor/ESEM (marginalized continuous core + explicit non-Gaussian block). It is **hardened end-to-end**:
not a prior/estimator artefact (flat-prior φ=1.00, WAIC, PPC §5); largely invariant across BP/SZ/DR (§8);
**certified** at largest-N with cross-seed Tucker φ 0.993 (§4); biology is the least severity-entangled
domain (correlated-G §3.1); resample-robust (min φ ≥ 0.85 under LOCO + site-bootstrap + weighting, §8);
with per-patient coordinates + uncertainty + reliability flags (§7). Anhedonia **rejected**;
impulsivity/negative-symptoms/sensory **not_testable**; depression/anxiety are cross-loading **windows**.
Engine in `src/face/{models/bayesian,confirm,runner,scoring}.py`; pipeline `scripts/01,04–09,s5_*`; results
in `reports/01,04–11`. Next: **M2 strata** on these coordinates. Updated 2026-06-09.

## What's decided

- **Model:** one **global** Bayesian sparse bifactor / ESEM — mixed likelihoods, soft priors,
  observed-cell likelihood (no imputation), **full V0 sample**. Estimated via a **staged continuation**
  (S1→S5); **only the global fit (S5) is interpreted.**
- **Confirmation:** **in-engine** — prior-free refit + PPC + WAIC (standalone FIML dropped, §5; semopy
  intractable/unreliable on the full backbone, and §3.5 makes the marginal = FIML). **Done** (see below).
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
  `bayesian_model.yaml` were retired — one canonical engine now); tests (`tests/v3/`, **90 passing**).
- **The map (FINAL):** **9 dimensions** — G + cognition/metabolic/inflammatory/sleep/developmental-risk/
  suicidality **+ mania + substance** — certified jointly (see the "DONE" bullets below). anhedonia
  **rejected**; impulsivity/negative-symptoms/sensory dropped pre-modeling; depression/anxiety = cross-loading
  windows. Biology is *least severity-entangled* (not strictly ⊥) via the correlated-G test. *(The dated
  "S1 result → S5 result" sections lower down are the **7-dim development record**, superseded by the
  9-dim certification — kept for provenance, not current status.)*
- **Confirmation result (§5, DONE):** the continuous backbone is **estimator- and prior-robust**. A
  **prior-free** (flat-prior) refit at full N reproduces the soft-prior loadings/Φ **exactly** (Tucker φ =
  1.00 every factor; max |ΔΦ| = 0.00) → not a Bayesian-prior artefact; **PPC** absolute fit SRMR ≈ 0.07
  (misfit only in repeated-measure item clusters); **WAIC** decisively prefers the bifactor over
  unidimensional (Δ≈53k) and correlated-factors (Δ≈2.7k). Artifacts: `reports/05_confirmation_report.md`
  (+ `05_waic.csv`, `05_residual_correlations.csv`); engine `src/face/confirm.py` · `scripts/05_confirm.py`.
- **Invariance result (§8, DONE):** in-engine, per-cohort **simple-structure** fits (the bifactor G is
  multimodal in SZ without FAST), N≈600/cohort × 3 seeds, **9/9 converged**. The map is **largely invariant**
  across BP/SZ/DR (12/15 factor×pair φ ≥ 0.95): **cognition · metabolic · sleep invariant** everywhere; **G
  invariant** except BP–SZ (partial, φ 0.92 — few anchors, no FAST in SZ); **inflammatory non-invariant in
  DR** (φ 0.71/0.75) — **neutrophils load ≈0 in DR** (0.07 vs 0.88), eosinophils high (0.59 vs 0.23): DR's
  inflammatory axis is compositionally different → a **documented partial-invariance caveat** for DR
  inflammatory scores. Artifacts: `reports/06_invariance_report.md` (+ `06_congruence.csv`, `06_dif_items.csv`);
  `scripts/06_invariance.py` · `src/face/runner.py`. Working pattern: subsample ≈2k + multi-seed + resumable
  cache + progress (§3.6).
- **S5 certification (§3.6/§4.5, DONE — largest-N documented):** the reported 7-dim map, multi-seed at
  N≈2,000 cohort-balanced (tune 2000 · draws 1500 · ta 0.9, 2 seeds). **§4.4 rung-3 reparam:** diagnosed
  the slow locus as the **CTQ→G bifactor loadings** (dev is explicit ⇒ 2-explicit-factor ridge, ESS 30);
  the `bifactor_g_sd` knob tightens dev/suic→G toward 0 (they're ≈⊥G) **leaving the biology→G estimand
  free** — cross-loadings ESS 30→85. Result: R-hat **1.03** · struct ESS **114–158** · **0 div** · BFMI
  **0.40** (healthy — no funnel; the limit is ESS-autocorrelation on the **suic~dev Φ + explicit-latent
  coupling**, not geometry). **Cross-seed resample-stability: Tucker φ 0.993**, max |ΔΦ| 0.05 — the reported
  loadings/Φ are stable; suic~dev Φ *precision* is the documented limit (point estimates solid). Artifacts:
  `reports/07_s5_certification_report.md`; `scripts/s5_certify.py` (per-seed resumable cache). Run under
  `caffeinate` + detached (`nohup`/`disown`) — the fix for macOS-sleep/harness interruptions on long fits.
- **Correlated-G sensitivity (§3.1, DONE — biology⊥G refined):** relaxing G⊥specifics (all factors freely
  correlated, simple-structure marginalized model, clean R-hat 1.01 · ESS 421 · **0 div**) → G correlates
  **+0.06 inflammatory · +0.14 metabolic** vs **+0.39 cognition · +0.44 sleep**: biology is the **least
  severity-entangled** domain (not strictly ⊥, but lowest by far). Engine: `g_correlated` Φ is now a
  **unit-row Cholesky** (`pm.LKJCorr(n≥5)` breaks jitter-init; `LKJCholeskyCov` sd funnels → divergences).
  `scripts/s5_corrg.py` → `reports/07_corrG_report.md`.
- **Robustness (§8, DONE):** Tucker congruence φ of the loadings vs the certified S2 reference under
  leave-one-cohort-out + diagnosis-balanced subsampling + **site cluster-bootstrap** + **1/n_cohort-weighted
  fit** (§3.6) — **min φ ≥ 0.85** (map not an artefact of cohort imbalance, any single cohort, or site
  clustering). `scripts/08_robustness.py` → `reports/08_robustness_report.md`.
- **9-dim joint integration (DONE):** mania + substance closed the deferred gap as **real dimensions**, so
  the reported map was re-certified at **9 dimensions** — 5 marginalized (cognition/metabolic/inflammatory/
  sleep/**mania**) + 4 explicit (G/suicidality/developmental/**substance**, substance's binary SUD under the
  proper Bernoulli likelihood). Certified: R-hat ≤ 1.04 · ESS ≥ 112 · 0 div · BFMI ≥ 0.41 · cross-seed Tucker
  φ **0.993**. `scripts/s5_certify9.py` → `reports/11_s5_9dim_report.md` (engine: `prepare_mixed` gained
  `explicit_factors`/`min_cohorts`; `S5_FACTORS`).
- **Scoring (§7, DONE):** per-patient coordinates for all 9,013 — 6 continuous-anchored dims (incl. mania)
  via conditional-Gaussian from the certified 9-dim loadings + 3 explicit (suic/dev/substance) via f_e —
  each with mean/SD/HDI + reliability tier. `scripts/07_score.py` → `results/face/patient_scores.parquet`.
- **Atlas + adjudication (§2.3/§6, DONE — M1 LOCK pending PI review):** prior→posterior heatmap at 9 factors
  (`docs/figures/empirical_atlas.png`) + `docs/ADJUDICATION.md`: **9 confirmed** (G + cognition, metabolic/
  inflammatory split, sleep, developmental-proxy, suicidality, mania, substance), **anhedonia rejected**,
  impulsivity/negative/sensory **not_testable**, depression/anxiety = windows. No candidate deferred.
- **Mixed-model PPC (§8, DONE):** absolute-fit check for the non-Gaussian block (the continuous block was
  §5, SRMR 0.07). True posterior-predictive on the 9-dim cert — **21/22 indicators reproduce** their
  observed endorsement rates/means (Bayesian p ≈ 0.5); lone flag `isf09a` (zero-inflated attempt count,
  item-level — the suicidality factor's 7 binary items all reproduce). `scripts/12_mixed_ppc.py` →
  `reports/12_mixed_ppc_report.md`, `docs/figures/mixed_ppc.png`.
- **M1 complete** — the measurement layer is built, hardened (confirmation/invariance/robustness/PPC),
  **certified at 9 dims**, scored, and adjudicated. PI sign-off on the adjudication + atlas locks it; then
  **M2 strata**. *Small follow-ons: extend invariance/robustness/corr-G to mania/substance; full-N
  non-Gaussian scoring; hurdle likelihood for isf09a if its count precision is needed.*
- **Compute lesson (this session):** full-N S1/S2 ≈ 1 h; the S3+ mixed-likelihood frontier is heavier, so
  S3 checkpoints use a random N=4,000 subsample (§3.6). Engine perf fixes: grouped-GEMM Woodbury (Cholesky
  per observed-pattern, 2.75×), tree-depth cap 8 + ta 0.85 (2.7× at 7 factors). Φ bug fixed (LKJCorr=Cholesky
  → Φ = L Lᵀ). **No GPU was needed** — the reparam ladder (marginalization + rung-3 tightening) certified the
  mixed S5 (7-dim and 9-dim) on the Mac via the detached + caffeinate + per-seed-cache pattern.
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

## S4 result — anhedonia REJECTED (not a distinct dimension)

Thin BP/DR candidate (1 dedicated indicator, `qids_anhedonia_interest`; SZ has no QIDS). Tested on the S3a
map (random N=4,000). **Does not form a stable factor:** R-hat **1.54** · ESS 7 · 0 div at *both* the smoke
and the N=4,000 fair test (reflection/collapse across chains). When it forms it is **redundant with G +
depression** — its indicator loads **0.61 on G**, and the QIDS-total window loads **0.365** onto it (the
anhedonia anchor, QIDS item 13, is part of QIDS total → near-collinear). The rest of the map is undisturbed
(non-convergence isolated to the anhedonia cells). **Verdict (§6): rejected as a standalone dimension** —
variance absorbed by G + the depression windows. Matches the methods-doc prior. **Full writeup:
[`RESULTS.md`](RESULTS.md) §S4** (incl. the consolidated S1–S4 candidate-adjudication table). Artifacts:
`reports/04_stage4_*`.

## S5 result — the 7-dimension map (development record — SUPERSEDED by the certified 9-dim, above)

> ⚠️ **Historical.** This was the first global fit (provisional, N=5,000). It has since been **certified at
> largest-N** and **extended to 9 dimensions** (mania + substance) — see the "DONE" bullets at the top. Kept
> for provenance. No GPU was used.

Global mixed-likelihood fit over all 7 dimensions (continuous core marginalized + suicidality/developmental
explicit, one shared Φ), random N=5,000 subsample (Mac best-effort): **R-hat 1.040 · 0 div** (provisional —
slow-mixing continuous cross-loadings).
- **G = functional burden** (FAST 0.90, EGF 0.73); specifics: cognition 0.57 · sleep 0.48 · developmental
  0.42 · inflammatory 0.39 · metabolic 0.32 · suicidality (isf07 0.60 + binary ISF 2.7–3.4 logit).
- **Φ weak** (mean |off-diag| 0.10): metab~inflam 0.19, suic~dev 0.23, sleep~dev 0.19 — distinct axes.
- **G⊥biology refined (the load-bearing check, both identifications):** bifactor direct G-loadings
  metabolic 0.08 / inflammatory 0.07; **correlated-G** factor correlations metabolic **0.28** / inflammatory
  **0.14** (vs cognition 0.35, sleep 0.47). → biology is **not strictly orthogonal** to severity (the
  bifactor overstated it) but **is the least severity-entangled domain** (~92–98% of its variance independent
  of G) — biological strata remain worthwhile; "largely severity-independent" replaces "orthogonal".
  **Full writeup: [`RESULTS.md`](RESULTS.md) §S5.** Artifacts: `reports/04_stage5_*` + `04_stage5_corrG_phi.csv`.

## Open methods choices (flagged for the PI)

Sparsity prior (soft-normal vs horseshoe) · Student-t vs Gaussian continuous default · item- vs
factor-level covariates · acceptance-gate numbers. Defaults are set in
[`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md); confirm or overrule before S1.

## What to read

[`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md) (methods + math) · [`RESULTS.md`](RESULTS.md) (findings log)
· [`MEASUREMENT_MAP_EXPLAINED.md`](MEASUREMENT_MAP_EXPLAINED.md) (teaching companion: intuition, the
marginalized Woodbury engine, NUTS, + Part C as-built results) · [`PRIOR_ATLAS.md`](PRIOR_ATLAS.md) (prior
map) · [`../README.md`](../README.md) (overview) · [`../CLAUDE.md`](../CLAUDE.md) (guide) ·
[`DATA.md`](DATA.md) (data contract).
