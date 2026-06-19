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

### Sensitivity arms (optional)

- Soft-unlikely (free the ~980 unlikely cells): `--mode medium --soft` — congruent with the hard-zero map
  for the well-anchored backbone; dilutes thin factors (why hard-zero is primary).
- In-likelihood covariates: set `covariate_mode="in_likelihood"` on `MeasurementConfig` (samples per-item
  `alpha`/`beta`; heavier, opt-in).

---

## 3. Tests

```bash
PYTHONPATH=$PWD/src python -m pytest tests/golden/test_measurement_model_oop.py -q   # OOP engine
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
