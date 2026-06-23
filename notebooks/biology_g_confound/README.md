# biology⊥G confound sensitivity

A reviewer-driven robustness arm for the FACE‑ATLAS **load‑bearing claim**: metabolic/inflammatory load is
the *least severity‑entangled* domain (Φ(G, metabolic) ≈ 0.12, Φ(G, inflammatory) ≈ 0.07, vs 0.39 cognition /
0.42 sleep). The hardest critique is that this "biology axis" is really a proxy for **medication** (antipsychotics
cause metabolic syndrome), **adiposity** (BMI), or **site/assay batch**. This folder re‑derives Φ(G, ·) under a
ladder of adjustments to test whether the claim is a real biological signal or an artifact.

## Arms (FWL‑partial each continuous item before the correlated‑G marginalized model)

| arm | adjusts for | role |
|---|---|---|
| `A0_unadjusted` | nothing | the reported 0.12 / 0.07 |
| `A1_demo_site` | age(spline) + sex + edu + site | reproduces `scripts/10` |
| `A2_antipsychotic` | A1 + antipsychotic exposure | **conservative headline** — biology vs medication |
| `A3_bmi` | A2 + BMI as a covariate | **exploratory / partly circular** (BMI is a metabolic indicator) |

**Verdict (on A2):** if metabolic & inflammatory stay small and below cognition/sleep → biology⊥G is
confound‑robust; if they inflate → downgrade honestly to "a medication/adiposity‑linked biological axis."

## Run

```bash
# 1. build the extended covariates (adds on_antipsychotic + bmi to covariates_v0.parquet)
PYTHONPATH=$PWD/src python notebooks/biology_g_confound/build_covariates.py

# 2a. smoke (fast end-to-end check, N≈500, 1 seed)
PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE \
  python notebooks/biology_g_confound/run_confound_sensitivity.py --smoke

# 2b. real run, detached (N≈2000 balanced, 2 seeds, 4 arms)
python scripts/run_job.py biology_g_confound -- \
  env HDF5_USE_FILE_LOCKING=FALSE python -u notebooks/biology_g_confound/run_confound_sensitivity.py
python scripts/status.py --watch        # monitor
```

## Outputs
- `reports/12_biology_g_confound.csv` — Φ(G, domain) across the four arms.
- `reports/12_biology_g_confound_report.md` — table + convergence + verdict + honest limits.
- `results/face/biology_g_confound/<arm>/` — cached idata + manifests (gitignored).

## Engine note
Reuses the proven correlated‑G + covariate machinery (`continuous_core.prepare` → `confirm.corr_no_g_prep` →
`runner.sample_marginalized`). The only kernel change is a **backward‑compatible** `covariate_extra_cols`
parameter on `prepare` / `_covariate_design` / `_residualize_on_covariates` (default empty → the primary
engine is byte‑for‑byte unchanged); arms A2/A3 pass `("on_antipsychotic", ...)` / `(..., "bmi")`.

## Honest limits
- Antipsychotic coverage ~54 % (NaN mean‑imputed for the design; **BP lifetime** vs **SZ/DR current**).
- Antipsychotic is on the causal path to metabolic load → adjusting for it is conservative‑to‑over‑conservative.
- A site dummy is coarser than full cross‑platform assay/batch harmonization.
- Internal sensitivity on the correlated‑G measurement structure; **not** external validation.
