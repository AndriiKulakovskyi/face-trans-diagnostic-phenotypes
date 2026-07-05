# REPRODUCE — the full FACE vertical (M1 → M5) from raw sources

One entry point, one command per milestone. The heavy Bayesian fits run **detached** (they survive the
shell, the harness turn-limit, and Mac sleep); everything else is fast. Methods of record:
[docs/methods/](docs/methods/); data contract: [docs/DATA.md](docs/DATA.md).

## 0. Environment

```bash
pip install -e ".[bayesian,strata,dev]"      # core + NUTS backend + strata engine + dev tools
export PYTHONPATH=$PWD/src
export HDF5_USE_FILE_LOCKING=FALSE            # required on macOS (netCDF write lock)
# do NOT set XLA_FLAGS host-device-count — it oversubscribes cores and slows numpyro ~4×
```

**Inputs** (in `data/`): the confidential per-cohort exports `data/{bipolar,schizophrenia,depression}.csv`
(Fondation FondaMental — not in the repo), the harmonization dictionary `data/face-common-vars.xlsx`, and
`data/site_lookup.csv`. Without the CSVs you can still run the golden tests (they use the synthetic
generator, `synthetic/generate_face_like.py`).

## 1. Build the model-ready data (no imputation)

```bash
face build-data          # → data/processed/baseline_v{0,1,2}.parquet + indicator_metadata + site_v0
face build-covariates    # → data/processed/covariates_v0.parquet
```

Loads the full V0 sample (all 3 cohorts, no completeness selection), applies deterministic skip-logic
structural-zero decoding, restricts to the modeled indicators, and persists — **NaN preserved, never
imputed**. Expected: **N = 9,013** (BP 6,252 · SZ 2,209 · DR 552), 143 indicators, ~40% cell missingness;
follow-up V1 (4,270) / V2 (2,958) tables for M3/M4. QC → `reports/01_build_data.md`.

## 2. Fit the milestones

Each milestone is one clean engine (`face.<milestone>`) driven by `face fit`. Use `--detach` for the long
fits and watch them with `face status`.

```bash
# M1 — the transdiagnostic map (the long pole, ~4–8 h): the canonical two-phase recipe
#   (balanced horseshoe immunometabolic-merge ladder → full-N cohort-weighted substance-orthogonal salvage)
face fit m1 --mode production --detach
face status --watch                       # per-chain progress + ETA;  face status --logs m1_fit

# M2–M5 — each consumes only the FIXED prior milestones (never re-discovers):
face fit m2 --detach     # strata: A=5 archetype simplex + nested K-family        (deterministic, fast)
face fit m3 --detach     # temporal: score V1/V2 under the fixed M1/M2            (projection MCMC)
face fit m4 --detach     # prognosis: errors-in-variables Bayesian GLM
face fit m5 --detach     # treatment: overlap → propensity → DR moderation + E-value

face fit all             # or run the whole chain in dependency order
```

Smoke first if wiring is uncertain: `face fit m1 --mode smoke` (a fast, non-scientific end-to-end check).
Each stage caches under `results/<mN_name>/…`; re-runs reuse a stage unless its config/data/code fingerprint
changed. Hand-offs: `results/m1_measurement/primary/` (map) → `results/m2_strata/consolidate/` (strata) →
`results/m3_temporal/{consolidate,attrition}/` → `results/m4_prognosis/consolidate/` → `results/m5_treatment/consolidate/`.

## 3. Expected convergence (M1)

- Continuous rungs (`hs_s1_merged`, `hs_s3_merged`): R-hat ≤ 1.01, ESS 800+, **0 divergences**.
- Mixed 8-factor stage: R-hat ~1.06 balanced / **1.03 full-N weighted primary**, **0 divergences**.
- The map is 8 dimensions — G (overall burden) ⊥ 7 specifics {cognition, immunometabolic, sleep,
  mania/activation, suicidality, developmental-risk, substance} — with 3 earned cross-loadings on cognition.

## 4. Tests

```bash
make golden                                   # numerical-kernel regression (synthetic data; no confidential data)
PYTHONPATH=src python -m pytest tests -q       # full suite (data-needing integration tests skip when absent)
```

## Notes

- **No naive imputation, ever** — observed-cell / FIML likelihood only.
- Determinism: fixed seeds (M1 20260605; each milestone has its config-of-record seed). Deterministic stages
  (data build, M2 strata) reproduce bit-identically; NUTS stages reproduce within-tolerance on the same env.
- Reproduction was verified end-to-end against a frozen oracle (`reference/oracle/`): M1 loadings/Φ
  bit-identical, M2 archetypes exact, M3–M5 within tolerance. See `reports/refactor/05_regen_m{1..5}.md`.
- Compute: developed on a Mac (Apple Silicon) with PyMC + NumPyro; the marginalized Woodbury likelihood makes
  the continuous stages tractable at full N ≈ 9,013 on CPU.
