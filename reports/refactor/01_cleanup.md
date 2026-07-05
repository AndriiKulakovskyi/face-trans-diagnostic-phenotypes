# Refactor Step 1 — Cleanup deletions (cache/cruft only)

**Branch:** `refactor/ground-up-2026-07` · **Date:** 2026-07-04 · Preceded by Step 0 freeze (`c6a8849`).

Delete only gitignored, regenerable content. No tracked file removed; raw sources provably untouched.

## What ran (staged, increasing blast radius)
1. **Caches + cruft:** `.DS_Store` (recursive), `.ruff_cache`, `.pytest_cache`, `.mypy_cache`, `__pycache__` (recursive), `.cursor`, `.gstack`.
2. **Job-orchestration state:** `run/`, `logs/`, `RUN_STATE.md`.
3. **Legacy + derived data:** `results/v3/` (legacy), `data/processed/`, `data/interim/`, `data/artifacts/`.
4. **The 22 GB:** `results/reports/`, `results/face/` — then `git checkout -- results/face` to **restore the one
   tracked file** under it (`gllvm_oop/s8_full/model_state.pt`, cited by the articles). `results/manifests/`
   (16 tracked run manifests) and `results/.gitkeep` were never targeted.
5. **Stale env:** `.venv/` (Python 3.11, wrong line vs the 3.13.9 env of record; rebuildable from `requirements.lock`).

## What was checked (gate)
- **Pre-delete gate:** raw-source checksums identical to the Step-0 in-repo snapshot → `✓ RAW INTACT` before any `rm`.
- **(a) Raw byte-identical pre/post:** `✓ RAW UNTOUCHED` (10 raw files: 3 cohort CSVs + `face-common-vars.xlsx` + `face_dimension_soft priors.xlsx` + `site_lookup.csv` + 4 thesaurus xlsx).
- **(b) No tracked deletions:** `git status --short` is **empty** — the working tree matches HEAD; every deleted byte was gitignored.
- **(c) Tracked survivors:** 16 `results/manifests/*.json` intact; `results/face/gllvm_oop/s8_full/model_state.pt` restored; `results/.gitkeep` present.

## Reconciliation vs oracle
Reclaimed **`results/` 22 G → 660 K**; **total repo ~22.7 G → 581 M**. 18 files remain under `results/`+`data/processed`
(all tracked manifests + `model_state.pt` + `.gitkeep`; `data/processed/` now empty, to be regenerated in Step 5).
The oracle (`reference/oracle/` + tag `pre-refactor-2026-07-04`) is untouched and remains the reconciliation target.

## Converged?
**YES.** Only gitignored/regenerable content removed; raw sources byte-identical; zero tracked deletions; oracle + tracked
manifests intact. Ready for Step 2 (package restructure + rename).
