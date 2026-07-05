# Refactor Step 7 — final validation + sign-off

**Branch:** `refactor/ground-up-2026-07` (off `main` @ `8cf4f3e`) · **Date:** 2026-07-05 · Tag: `pre-refactor-2026-07-04`.

The deep refactor + full reproduction is complete. Every gate is green.

## Deliverables (the user's asks)
- **Cleaned repo:** 22.7 G → 581 M working tree; logs/run-state/caches/exploratory cruft removed; raw sources
  preserved (byte-identical, verified) + backed up off-repo.
- **No legacy naming:** **zero** `_oop` / `v3` / `weighted_8d` / `hs_s5_merged_xc` / timestamped-`MODEL_VERSION`
  in code, configs, results, tests, or tracked filenames (the frozen `reference/oracle/` + this `reports/refactor/`
  record intentionally retain the old names for provenance).
- **No scripts/notebooks mixture:** one `face` CLI (`build-data | build-covariates | fit m1..m5 [--detach] |
  status | run`), a clean engine module + a detached, wake-locked, progress-tracked runner per milestone; the
  duplicate `.ipynb` and superseded per-milestone drivers deleted; sensitivity arms moved under `analyses/`.
- **Regenerated from sources:** all `data/processed/*` and `results/*` deleted and rebuilt from the raw cohort
  CSVs through the clean pipeline.
- **Articles preserved + rebuilt:** all three `article*/` folders re-pointed + compiled clean.

## Reproduction — reconciled to the frozen oracle (`reference/oracle/`)
| milestone | reconciliation |
|---|---|
| M1 measurement | **bit-identical** — loadings Tucker φ=1.0000 ×8, Φ Δ=0, 3 cross-loadings exact, R-hat 1.03 / 0 div |
| M2 strata | **exact** — archetype profiles Δ=0, K-family Δ=0 (A=5, K=2) |
| M3 temporal | invariance license **exact** (immunometabolic φ 0.987); ICCs match STATE.md copula values |
| M4 prognosis | within-tolerance — +archetypesA ΔELPD **+62.76** (=+62.8), operative_K none, top-ranked; cleaner convergence than oracle |
| M5 treatment | within-tolerance — E-values ±0.01, null-vs-signal verdicts preserved (lithium null, antipsychotic suggestive) |
| global fingerprint | M1/M2 coords rank-hash 8/8 exact; M2/M4 archetype dominant-hash EXACT; M3 panel within tol |

Deterministic layers (M1 map, M2/M4 archetypes) reproduce **bit-identically at the per-patient level**; the
stochastic NUTS layers reproduce within pre-declared tolerance on the env of record (Python 3.13.9, pymc 6.0.1,
numpyro 0.21.0, jax 0.10.1, seed 20260605). Per-milestone detail: `reports/refactor/05_regen_m{1..5}.md`.

## Green checks
- **`make golden`**: 35 passed (bit-level numerical kernels).
- **Full `pytest tests/`**: **265 passed, 0 failed** (integration tests run with data present).
- **`ruff check .`**: passes.
- **Working tree clean**; results tree = `{m1_measurement, m2_strata, m3_temporal, m4_prognosis, m5_treatment, analyses, manifests}`.
- **All 3 article PDFs compile clean** (article 69pp · article_methods 22pp · article_immunometabolic_burden),
  0 undefined refs/citations; every main-text figure regenerated from the reproduced pipeline — including the 3
  that needed the separated sensitivity arms (variational-GLLVM, representation-benchmark), now also regenerated.

## Converged?
**YES — the deep refactor is complete and the entire research program reproduces from raw sources through the
clean pipeline.** Recommended follow-up (out of scope here): merge `refactor/ground-up-2026-07` to `main` after
PI review; the `pre-refactor-2026-07-04` tag + `reference/oracle/` remain the historical baseline.
