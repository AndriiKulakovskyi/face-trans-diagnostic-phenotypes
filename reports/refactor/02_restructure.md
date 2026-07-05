# Refactor Step 2 — Package restructure + rename (behavior-preserving)

**Branch:** `refactor/ground-up-2026-07` · **Date:** 2026-07-04 · Preceded by Step 1 cleanup (`e5d4c92`).
**160 files changed, +639/−320**; all engine/config/test moves done via `git mv` (89–100% rename similarity).

## What ran
1. **New permanent scaffolding** (`src/face/config/`, `src/face/caching/`):
   - `config/paths.py` — one `REPO` resolver + milestone-keyed `results()/figures()` + `analysis_results()`
     (kills the per-file `parents[N]` constants, where N drifted 3/4 — a real bug: `measurement` was `[4]`).
   - `config/registry.py` + `loader.py` — logical-name → config-file table + typed accessors (`matrix()`, `ontology()`, `outcomes()`).
   - `caching/cache_key.py` — content-hash reuse keys (config+data+code+stage); `caching/manifest.py` ← `io/manifest.py`.
2. **Engine renames (`_oop` killed):** `models/bayesian/measurement_model_oop.py`→`measurement/engine.py`,
   `continuous_core.py`→`measurement/kernel.py`, `confirm.py`/`runner.py`→`measurement/{confirm,sampling}.py`;
   `strata|temporal|prognosis|treatment/*_model_oop.py`→`<pkg>/engine.py`; `prognosis/repbench/`→`benchmark/`.
3. **Sensitivity arm separated:** `models/variational/*` (torch) → **`analyses/variational_gllvm/`** (out of the
   core wheel + out of `tests/golden/`). This is what makes `make golden` torch-free — **it fixes the
   multiple-OpenMP-runtime segfault** the full golden session hit under the env of record.
4. **V3 stub packages deleted:** `models/{,bayesian}/__init__.py`, `adjudication/`, `missingness/` (all `"""V3 X module."""` one-liners).
5. **Configs renamed (`v3` dropped):** `prior_loading_matrix_v3{,_biomerge,_biomerge_xc}.csv` →
   `loading_matrix{,.immunometabolic,.immunometabolic_crossload}.csv`; `soft_loading_priors_v3`→`loading_priors`;
   `candidate_dimensions_v3`→`ontology_candidates`; `likelihood_map_v3`→`likelihood_map`; `m4/m5_outcomes`→`prognosis/treatment_outcomes`.
6. **Global rewrite:** 74 `.py` files — import dotted-paths + config-filename literals (longest-first). Engine
   path/version constants cleaned: `results/face/<x>_oop`→`results/m{1..5}_<name>`, `weighted_8d`/`hs_s5_merged_xc`
   folded to `primary`, timestamped `MODEL_VERSION` → clean stems (`"m1_measurement"`…).
7. **Test tree consolidated + de-legacied:** `tests/m4`→`tests/prognosis`, `tests/m5`→`tests/treatment`,
   `tests/v3`→`tests/data`; gllvm tests → `analyses/variational_gllvm/tests/`; `test_*_oop.py`→`test_*_engine.py`;
   `test_oop_*` fns → `test_*`. Added `--import-mode=importlib` (same-basename `test_endpoints.py` in two dirs).
8. **`.gitignore` / `pyproject` / `conftest`** updated to the milestone tree + new module homes; tracked GLLVM
   `model_state.pt` relocated `results/face/gllvm_oop/…` → `results/analyses/variational_gllvm/…` (empty `results/face/` removed).

## What was checked (gate)
- **Imports:** 21/21 core modules import cleanly under the new structure; **265 tests collect with 0 errors** (was 3).
- **Config:** `registry.path()` resolves every renamed file (exists=True); `loader.matrix()` loads (1430×11); `paths.results('m1')` → `results/m1_measurement`.
- **Golden:** **30 passed, 5 skipped** (torch-free — no OpenMP crash; the 5 skips are integration tests needing confidential data).
- **Lint:** `ruff check .` **passes clean** (137 import-order autofixes across the repo).
- **`src/face` imports torch:** none.

## Reconciliation vs oracle
No numeric outputs produced (behavior-preserving structural change; results were deleted in Step 1 and are
regenerated in Step 5). The oracle (`reference/oracle/` + tag) is untouched.

## Converged?
**YES.** Clean package skeleton (`config/`, `caching/`, `measurement/`, `benchmark/`, `analyses/`); `_oop`/`v3`/
timestamped-version naming removed from all **code** (paths, config filenames, `MODEL_VERSION`, module/file/test
names); imports resolve; golden green; ruff clean; CI-shaped suite collects.

**Deferred to Step 3 (engine rewrite):** (a) docstring/comment prose still names old modules (e.g. `measurement_model_oop`,
`strata_oop`) — rewritten with each engine; (b) full content-hash `cache_key` adoption inside the engines (the
`MODEL_VERSION` strings are now clean stems, and `caching/cache_key.py` is scaffolded, ready to wire in). The
`face` CLI entry point + detached runners land in Step 4.
