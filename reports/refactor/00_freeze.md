# Refactor Step 0 — Freeze the oracle

**Branch:** `refactor/ground-up-2026-07` off `main` @ `8cf4f3e` · **Tag:** `pre-refactor-2026-07-04` · **Date:** 2026-07-04

Hard gate before any deletion (Step 1) or rewrite: an immutable restore point + an off-repo backup of the
irreplaceable raw sources + a committed numeric oracle that every regenerated number will be reconciled against.

## What ran
1. **Branch + tag.** `refactor/ground-up-2026-07` created; annotated tag `pre-refactor-2026-07-04` on `8cf4f3e`
   (freezes all tracked code, `reports/*`, `docs/figures/*`, `results/manifests/`, and the three article folders).
2. **Raw-source backup (off-repo).** 7 irreplaceable inputs → `~/FACE_raw_backup_2026-07-04` (117 MB, 11 files):
   `data/{bipolar,schizophrenia,depression}.csv`, `data/face-common-vars.xlsx`, `data/site_lookup.csv`,
   `data/thesaurus/*.xlsx`, `data/face_dimension_soft priors.xlsx`, `configs/prior_loading_matrix_v3.csv`.
   Checksum manifest `RAW_MANIFEST.sha256` (meta-hash `732c714f5210f59d2fc6a6d2c1013b80da68a9de7d139dd3a9acbfd3b972d6a3`).
   In-repo raw snapshot saved to scratchpad `face_raw_inrepo.sha256` for the Step-1 before/after diff.
3. **Numeric oracle** → tracked `reference/oracle/` (328 K, aggregate-only, no per-patient rows):
   - `m1/` loadings + Φ (`copula_8factor_*`), canonical fit manifest + `loadings_summary`, congruence, biology↔G confound, sparse-ESEM credible-cross, corrG Φ.
   - `m2/` archetype_profiles, k_family_menu, h2h_dsm5.
   - `m3/` trait_state, congruence, change_rates, transitions, informative_dropout, invariance license.
   - `m4/` incremental (ΔELPD), archetype_atlas, clinical_value, robustness, prognosis_summary.
   - `m5/` moderation (E-values), propensity_summary, severity_confound, overlap_audit, treatment_summary.
   - `diagnostics/` 8 stage `diagnostics.json`; `manifests/` 16 tracked run manifests; `INVENTORY_before.tsv` (512 result/processed files w/ sizes).
   - `fingerprint.py` (runnable recipe) + `FINGERPRINT.json` (4/4 per-patient arrays: coordinate moments + 8×8 correlation + per-factor rank/mean hashes; archetype weight moments + label-invariant dominant sizes + argmax hash). Sorted-canonical, rounded, tolerance-banded → safe to commit, sensitive to drift.
   - `MANIFEST.json` — env of record, seed 20260605, backup meta-hash, reconciliation tolerances.

## What was checked
- **Env of record identified:** `/opt/anaconda3/bin/python3` = **Python 3.13.9** with the exact package set that
  produced the canonical results (pymc 6.0.1, pytensor 3.0.4, numpyro 0.21.0, jax 0.10.1, arviz 1.1.0, pytest 9.0.2)
  — matches `results/manifests/*.json`. The local `.venv` (3.11.15, pymc 5.28.5, arviz 0.23.4, **no pytest**) is
  stale/incidental and will be deleted in Step 1 (rebuildable).
- **Green check (env of record):** the **core M1 fidelity kernels pass — 35 passed in 33 s**
  (`test_woodbury_likelihood`, `test_conditional_scoring`, `test_coherent_scoring`, `test_hurdle_logp`,
  `test_extreme_deconvolution`, `test_synthetic_recovery`, `test_measurement_model_oop`, `test_distribution_report`).
- **Known env crash (not a code defect):** the full `tests/golden` session **segfaults** — a native
  **multiple-OpenMP-runtime** crash from torch's bundled libomp coexisting with numpy/jax/pymc in one pytest
  process (pytest imports all test modules at collection, so torch loads alongside everything). The canonical
  results were produced by running engines as standalone scripts, never with torch+jax+pymc co-imported. **This
  validates the plan's Step-2/4 decision to move GLLVM/torch into `analyses/` and drop torch from the core wheel**
  — after which `make golden` (core) never imports torch and won't crash; the variational arm gets its own env.

## Reconciliation vs oracle
N/A for Step 0 — this step *creates* the reconciliation target. Every later step diffs against `reference/oracle/`
+ the `pre-refactor-2026-07-04` tag (tolerances in `MANIFEST.json`).

## Converged?
**YES.** Immutable tag exists; raw sources backed up off-repo + checksummed (meta-hash recorded); 328 K
aggregate-only oracle + fingerprint committed; env of record identified and its core kernels green.

**Open item for the gate (not blocking the freeze):** the env-of-record is anaconda 3.13.9 (present locally), but
torch cannot coexist with jax/pymc in one process — decided by the plan (separate `analyses/`), so no action now.
