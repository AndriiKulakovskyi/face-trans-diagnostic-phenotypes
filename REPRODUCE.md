# REPRODUCE — data loading + the measurement model (M1)

This guide reproduces **two things only**: (1) building the model-ready V0 data from the harmonized
dictionary, and (2) fitting the **measurement model** (the global Bayesian sparse bifactor / ESEM map).
The downstream milestones (M2 strata, M3 temporal, M4 prognosis, M5 treatment) are **not** covered here —
see [docs/STATE.md](docs/STATE.md) and the per-milestone methods docs for those.

Methods + math of record: **[docs/MEASUREMENT_MODEL.md](docs/MEASUREMENT_MODEL.md)**. Data contract:
**[docs/DATA.md](docs/DATA.md)**.

---

## 0. Prerequisites

```bash
# Python >= 3.11. Install the engine + Bayesian backend (PyMC / NumPyro):
pip install -e ".[full,bayesian]"
```

**Inputs** (in `data/`):

| file | tracked? | what |
|---|---|---|
| `data/face-common-vars.xlsx` | ✅ tracked | the variable dictionary (harmonization rules, bounds, likelihood family, skip-logic) |
| `data/site_lookup.csv` | ✅ tracked | recruitment-site code lookup |
| `data/{bipolar,schizophrenia,depression}.csv` | ❌ **confidential, gitignored** | the raw per-cohort exports |
| `configs/prior_loading_matrix_v3.csv` | ✅ tracked | the item × factor prior matrix (the soft-prior ontology) |

The raw per-cohort CSVs are confidential (Fondation FondaMental) and are **not** in the repository. The
data-build steps below require them locally; without them you can still run the smoke wiring check on the
synthetic generator (`synthetic/generate_face_like.py`), which is what the golden tests use.

---

## 1. Build the model-ready data (no imputation)

```bash
cd <repo-root>
python3 scripts/01_build_data.py          # baseline + indicator metadata + site side-table
python3 scripts/02_build_covariates.py     # covariates (needed for the residualized fit, see §2)
```

`01_build_data.py` loads the harmonized **full V0 sample** (all 3 cohorts, no completeness selection),
applies deterministic **skip-logic structural-zero decoding**, restricts to the indicators declared in the
prior matrix, and persists — **NaN preserved, never imputed**:

| output | tracked? | content |
|---|---|---|
| `data/processed/baseline_v0.parquet` | ❌ gitignored | ~9,013 patients × ~143 indicators (raw harmonized; NaN = missing) |
| `data/processed/indicator_metadata.parquet` | ❌ gitignored | per indicator: home factor, likelihood family, modeling block, burden sign |
| `data/processed/site_v0.parquet` | ❌ gitignored | `siteid_city` (administrative; never modeled) |
| `reports/01_build_data.md`, `reports/01_coverage_by_indicator.csv` | ✅ tracked | aggregate QC (counts/fractions only) |

`02_build_covariates.py` writes `data/processed/covariates_v0.parquet` (age, sex, education) aligned to the
same `(cohort, patient_id)` index. The measurement model's default covariate mode (residualize) reads it; if
it is absent the fit still runs (covariates fall back to mean-only ≈ de-meaning), but for the proper
covariate-adjusted map you should build it.

**Expected** (full FACE data): N ≈ 9,013 (BP 6,252 · SZ 2,209 · DR 552), ~143 modeled indicators
(~continuous + ~explicit), mean cell missingness ≈ 40%, skip-logic applied. The script prints these and
writes `reports/01_build_data.md`.

> Invariant: **no naive imputation, ever.** Missing cells stay `NaN` and are dropped from each patient's
> likelihood (FIML / observed-data likelihood). Diagnosis (`cohort`) is metadata, never an indicator.

### QA: inspect the indicator distributions

```bash
PYTHONPATH=$PWD/src python notebooks/run_distribution_report.py
# -> results/reports/qa_distributions.html  (per-indicator raw + rank-INT-Gaussianized panels,
#    NAMED empirical form, declared family, recommended copula tier)
# -> reports/qa_distributions_summary.csv   (committable aggregate; drives the copula tiering)
```

Use this to see each indicator's true distribution form (e.g. ~60% of the "continuous" block is
empirically heavy-tailed / count-like / zero-inflated) and which items the `gaussian_copula` vertical
(§2) will Gaussianize vs keep as native discrete likelihoods.

---

## 2. Fit the measurement model (OOP engine)

The self-contained OOP engine is `src/face/models/bayesian/measurement_model_oop.py`, driven by
`notebooks/run_measurement_model_oop.py`. Its **defaults are the validated, convergence-tested
configuration** — no flags needed for the canonical map:

- **hard-zero** unlikely cells (`soft_unlikely=False`) — keeps thin factors identified;
- **residualized covariates** (`covariate_mode="residualize"`) — FWL-equivalent, zero added parameters;
- **warm-start continuation** across the staged ladder S1 → S2 → S3 → S5;
- **`max_tree_depth=8`**; mixed stage **S5** at `tune=2000, target_accept=0.95, 4 chains, N≈2000 balanced`.

### Environment (required)

```bash
cd <repo-root>
export PYTHONPATH=$PWD/src           # required — else `import face` resolves to the wrong package
export HDF5_USE_FILE_LOCKING=FALSE   # required on macOS — avoids the netCDF write lock error
# do NOT set XLA_FLAGS host-device-count — it oversubscribes CPU cores and slows numpyro ~4x
mkdir -p /tmp/oop_fit_logs
```

### Run

```bash
# (a) wiring check — seconds to ~1 min; validates the path, NOT convergence
python notebooks/run_measurement_model_oop.py --mode mixed-smoke --overwrite

# (b) diagnostic ladder — N≈2000 balanced, S1→S2→S3→S5 warm-started (the usual manual run)
python notebooks/run_measurement_model_oop.py --mode medium --overwrite

# (c) full production map — full-N continuous + N≈2000 mixed; long, run detached
nohup python notebooks/run_measurement_model_oop.py --mode production --overwrite \
      > /tmp/oop_fit_logs/production.log 2>&1 &
tail -f /tmp/oop_fit_logs/production.log
```

Useful flags: `--no-plots` (skip figures), `--soft` (the soft-unlikely **sensitivity** arm — writes to a
separate `soft/` subdir so it never collides with the primary), `--output-dir` / `--figure-dir`.

Each stage caches to `results/face/oop_measurement/<stage>/{idata.nc, manifest.json}` (gitignored). A stage
is reused unless `--overwrite` is set or the model version / stage recipe / config signature changed.

### Figures + patient projections

```bash
python notebooks/oop_make_figures.py --balanced --n-subsample 2000
# -> docs/figures/oop_measurement/ : loading atlas, factor-correlation (Phi), reliability,
#    per-patient 94% HDI forests, and a 2-D patient map with uncertainty crosses
```

### Expected convergence

- Continuous rungs **S1/S2/S3**: R-hat ≤ 1.01, ESS ≈ 800–1000+, **0 divergences**; Φ shows
  **biology ⊥ G** (metabolic↔inflammatory correlated, G orthogonal to all specifics).
- Mixed **S5** (9-dim): backbone + substance clean (substance loadings well-mixed, R-hat ~1.00–1.01); the
  recall-noisy **developmental_risk** periphery keeps the headline R-hat ~1.06–1.14 (0 divergences). Its
  point estimates are reproducible across resamples — confirm with
  `python notebooks/confirm_s5_multiseed.py` (multi-seed cross-seed stability) — the criterion the certified
  pipeline uses for the explicit factors.

### Alternative likelihood vertical: Gaussian copula (acceleration)

```bash
PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE \
  python notebooks/run_measurement_model_oop.py --mode medium --likelihood-mode gaussian_copula
# writes to results/face/oop_measurement/copula/ (kept separate from the native primary)
```

`gaussian_copula` maps each Gaussianizable indicator through its empirical CDF (rank-INT,
`z = Φ⁻¹(F_j(y))`) and runs the **same** marginalized Woodbury model on `z` — a semiparametric
Gaussian copula factor model. Tiering (auditable in `qa_distributions_summary.csv`): continuous always
Gaussianized; ordinal/count promoted to the marginalized block iff high-cardinality + not point-mass
dominated; binary + low-cardinality ordinal stay native. The transform is invertible
(`face.models.bayesian.measurement_model_oop.copula_invert`, `y = F_j⁻¹(Φ(z))`) for synthetic
generation. Marginal Gaussianity is necessary-not-sufficient — the joint Gaussian-copula assumption is
validated post-fit by residual/PPC checks.

**Convergence + speed.** The copula transform converts the native model's *multimodal* mixed-stage
residual (budget-proof) into a single *slow-mixing-unimodal* weak correlation, so `--mode medium
--likelihood-mode gaussian_copula` uses the fuller S5 budget (4 chains / tune 2000 / 1500 draws) and
reaches **max structural R-hat 1.04, 0 cells > 1.05, 0 divergences** — which the native mixed stage does
not at any budget. The continuous rungs also mix 2–3× better (higher ESS) than native at the same
budget, and biology⊥G is preserved. So copula = better-specified marginals (accuracy) + per-effective-
sample faster + a clean sub-1.05 9-dim map.

**Maximum precision (cohort-weighted full-N).** `--likelihood-mode gaussian_copula --cohort-weighted` fits
ALL 9013 patients with §3.6 cohort weights (transdiagnostic estimand, single posterior); writes to
`copula/weighted/`. The mixed stage at full-N is heavy (~4 h, run detached). It roughly **doubles the
precision** on the weak dev↔suicidality correlation (posterior SD 0.027→0.013) and is congruent with the
balanced map (Tucker φ ≥ 0.994). Caveat: a weighted likelihood is composite/pseudo (point estimate =
balanced estimand; posterior SD order-correct, validated by the congruence), and the **substance** factor's
correlations are weighting-sensitive (its SUD indicators are BP/SZ-only) — see substance handling below.

**Substance handling (recommended).** Add `--substance-orthogonal` to pin the substance factor orthogonal
to the other specifics (writes to a `subortho/` subdir). Substance's cross-factor correlations are
non-identifiable and unstable (e.g. substance↔inflammatory swings 0.33↔0.82 across samples/weightings,
because its SUD indicators are BP/SZ-only + rare), so it is modeled as an independent axis defined by its own
indicators — justified by non-identifiability, not asserted as an empirical zero. The other factors are
unchanged (Tucker φ ≈ 1).

### Store params + generate synthetic patients

```bash
PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_synthetic_check.py
# -> results/face/oop_measurement/copula/weighted/fitted_model/  (portable Lambda/Phi/sigma + copula maps
#    + explicit-item GLM params: arrays.npz + model.json)
# -> results/reports/synthetic_vs_real.html      (per-indicator real-vs-synthetic overlays + dependence match)
# -> reports/synthetic_vs_real_summary.csv       (per-item moments, committable)
```

The copula model is invertible, so it doubles as a generator (`face.models.bayesian.synthetic`):
`eta~N(0,Phi); z=Lambda eta+eps; continuous y=F_j^-1(Phi(z)); explicit y~fitted GLM`. The synthetic cohort
reproduces the real marginals (median SD error ~1%; a few ultra-heavy-tailed labs' extreme tails are
imperfect) and the dependence structure (mean |Δcorr| ≈ 0.05).

**Native (pre-copula) generator — the contrast.** `run_synthetic_check.py --likelihood-mode native` generates
from the previous-best native model (continuous block = `log+z-score`), inverting parametrically
(`y = inv_log(z*sd + mu)*sign`) → `results/reports/synthetic_vs_real_native.html`. It reproduces the bulk
(continuous rel-SD median ~0.8%) but **cannot match the heavy-tailed lab block** (rel-SD p90 ~40% vs copula's
~7%; |skew| error median ~0.8 vs ~0.03 — bilirubin real skew 36 → native 1.9) because the encoding forces a
(log)normal marginal, and the native 9-dim mixed fit never converged (R-hat 2.7). This is the concrete
motivation for the copula. *(Native uses `include_covariates=False` so the stored moments are original-scale —
the certified fit residualizes covariates, which would otherwise center the recovered marginal at 0.)*

### Sensitivity arms (optional)

- Soft-unlikely (free the ~980 unlikely cells): `--mode medium --soft` — congruent with the hard-zero map
  for the well-anchored backbone; dilutes thin factors (why hard-zero is primary).
- In-likelihood covariates: set `covariate_mode="in_likelihood"` on `MeasurementConfig` (samples per-item
  `alpha`/`beta`; heavier, opt-in).

---

## 3. Tests

```bash
PYTHONPATH=$PWD/src python -m pytest tests/golden/test_measurement_model_oop.py -q   # OOP engine (incl. copula vertical)
PYTHONPATH=$PWD/src python -m pytest tests/golden/test_distribution_report.py -q     # QA distribution report
PYTHONPATH=$PWD/src python -m pytest tests/ -q                                       # full suite (data layer + engines)
```

The golden tests run on the synthetic generator, so they pass **without** the confidential data.

---

## Notes

- This OOP engine is the clean, self-contained reimplementation. The original **certified** production engine
  for the canonical M1 is `src/face/models/bayesian/continuous_core.py` driven by `scripts/04_fit.py`,
  `scripts/s5_certify9.py`, etc. (see [docs/MEASUREMENT_MODEL.md](docs/MEASUREMENT_MODEL.md)); the two produce
  congruent maps.
- Determinism: fixed seeds (default `20260605`). Re-runs with the same config reproduce the cached fits.
- Compute: developed/tested on a Mac (Apple Silicon) with PyMC + NumPyro; the marginalized Woodbury
  likelihood is what makes the continuous stages tractable at full N ≈ 9,013.
