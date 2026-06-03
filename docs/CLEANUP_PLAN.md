# CLEANUP & REFACTORING PLAN — FACE v2

> Goal: make the repository present as a **crisp, self-contained, current v2 study** — one pipeline,
> one axis system, one manuscript, numbers consistent across paper ↔ docs ↔ tests. Nothing should
> expose the superseded v1 generation except the intentional archive at tag `v1-archive-2026-05-30`.
>
> **Status legend:** ✅ done · 🔜 in progress · ⬜ not started.
> The v1 study is fully recoverable at git tag `v1-archive-2026-05-30` (branch `archive/v1-research`),
> so deletions on the v2 line lose nothing.

## Bottom line

The v2 science is done and high quality (`scripts/30–48_*_v2`, `docs/PIPELINE.md`, `docs/LABBOOK.md`,
99 passing tests, `results/manuscript/manuscript.md` → `.docx`). The problem is that the **v1 layer
was never cleared out** and the **top-level surfaces were never re-pointed at v2**, so a reviewer
cannot tell what is current. Five concrete symptoms:

1. The entire v1 machinery still sits in the tree as if active (`01–22` + `00_run_all` + 2 legacy
   sensitivity scripts + a v1 notebook + `axes.py`/`outcomes.py`), ~5,000 lines.
2. `README.md`, root `MANUSCRIPT.md`, `AGENTS.md`, and parts of `CLAUDE.md` say *"the analysis has
   NOT been run yet / No v2 results are written yet"* — false; `CLAUDE.md` contradicts itself.
3. The package default export `from trans_diag import AXIS_NAMES` returns the **v1 K=6** names; the
   live v2 K=4 names are only reachable via `axes_v2`.
4. The 2026-06-03 dictionary review (CVLT/fluency/QIDS-13 anhedonia + suicide skip-logic) updated the
   manuscript + tests to **194 items / 94 constructs / ECV 0.34 / memory-anchored cognition**, but the
   docs still say **188 / 88 / 0.36**.
5. `00_run_all.py` runs only the v1 pipeline, yet the golden-test docstring tells a fresh clone to run
   it to regenerate the **v2** `results/hfa/` artifacts. There is **no v2 orchestrator**.

## Current-vs-legacy map

| Layer | Current (v2 — keep) | Legacy (v1 — remove) |
|---|---|---|
| Pipeline | `30–35_*_v2` (stages 0–4), `40–48_*_v2`, `sensitivity_{aggregation,comorbidity,polychoric}_v2` | `00_run_all`, `01`–`13`, `15`–`22`, `sensitivity_masked_fa{,_mechanism}` |
| Utilities | `qa_harmonization`, `verify`, `audit`, `figures_manuscript_v2`, `build_manuscript_v2`, `build_dr_neuropsych_mapping`* | `build_notebook`, `qa_missingness` |
| `src/trans_diag` | `axes_v2`→`axes`, `skip_logic`, `masked_fa`, `domains`, `variable/rules/loader/filters/schema_gen/adapter`, `engine/*` | `axes.py` (v1 K=6), `outcomes.py` |
| Outputs | `results/hfa/*` | top-level `results/*` (v1) |
| Notebook | — | `notebooks/FACE_reproduction.ipynb` |
| Manuscript | `results/manuscript/manuscript.md` | root `MANUSCRIPT.md` (skeleton) |

\* `build_dr_neuropsych_mapping.py` is a one-time dictionary-prep tool whose job is done; verify it is
a no-op against the locked dictionary, then archive.

**Dependency check (verified):** every v2 script imports only from the surviving `trans_diag` core or
other `_v2` scripts; `outcomes.py`/`axes.py` are imported only by the legacy scripts. Removal is clean.

---

## P0 — Correctness & coherence (cheap, high-impact)

- ✅ **P0.1** Kill the "not yet run" contradiction in `README.md`, `AGENTS.md`, `CLAUDE.md` (status
  header, repo-layout `scripts/` line, the `## Pipeline … NOT yet re-run on v2` section), and repoint
  every `MANUSCRIPT.md` link to `results/manuscript/manuscript.md`.
- ✅ **P0.2** Make the package default the v2 model: delete v1 `axes.py`, rename `axes_v2.py → axes.py`,
  export `ORTHOGONAL_DIMENSIONS`, repoint the 4 importers + `test_axes.py`.
- ✅ **P0.3** Fix regeneration instructions that claim `00_run_all.py` rebuilds `results/hfa/`
  (`test_golden_numbers.py` docstring + skip message).
- ⬜ **P0.4** Reconcile headline numbers to one script-computed source of truth and propagate. The
  variable count disagrees across sources (199 live READY+PARTIAL / 214 docs / ~216 `DATA.md` / 220
  manuscript) — **needs a decision on what "usable variable" counts**; `n_items = 194` is the
  test-pinned truth. Items 188→194, constructs 88→94, ECV 0.36→0.34, canonical r `0.99/0.93/0.77`→
  `0.99/0.90/0.79`. *(Deferred to the P2 documentation pass; verify each against `results/hfa/`.)*

## P1 — Remove the v1 layer

- ✅ Delete scripts: `00_run_all`, `01`–`13`, `15`–`22`, `sensitivity_masked_fa`,
  `sensitivity_masked_fa_mechanism`, `build_notebook`, `qa_missingness`.
- ✅ Delete source: `src/trans_diag/axes.py` (v1), `src/trans_diag/outcomes.py`.
- ✅ Delete `notebooks/FACE_reproduction.ipynb` and root `MANUSCRIPT.md`.

## P2 — Documentation rewrite (point everything at "what it is now")

- ⬜ **P2.1** Rewrite entry docs fully; add the (currently unlinked) `docs/PIPELINE.md` to `README`.
- ⬜ **P2.2** Sync numbers (P0.4) in `FINDINGS.md`, `PIPELINE.md`, `ROADMAP.md`, `DATA.md`; fix
  `FINDINGS.md`'s false opening "No analysis results yet …" and `ROADMAP.md`'s "91 passed" (now 99).
- ⬜ **P2.3** Archive the executed pre-registration plans (`HIERARCHICAL_FA_PLAN.md`,
  `VALIDATION_PLAN_v2.md`, `MANUSCRIPT_PLAN.md`) to `docs/planning/` with an "executed" banner — they
  are future-tense and have drifted (propose a `hierarchical_fa.py` module never built; planned output
  paths `results/hfa_items.csv` ≠ actual `results/hfa/stage0_items_v2.csv`).
- ⬜ **P2.4** Add the missing final `LABBOOK.md` entry (V2-21) for the 2026-06-03 dictionary review.
- ⬜ **P2.5** Remove/relocate root `todo_data_cleaning.md` (17KB working notes, mostly executed,
  references a stale `face-common-vars-v2.xlsx` filename); fold open caveats into `DATA.md`.
- ⬜ Rewrite `CLAUDE.md`'s "three stages / why aggregate" section: it still describes the **flat
  masked-mean** domain scores (~69), which v2 **replaced** with the hierarchical/bifactor model
  (88/94 constructs).

## P3 — Structure & naming

- ⬜ **P3.1** Drop the now-redundant `_v2` suffix from scripts, the `axes` module, and `results/hfa/*`
  artifacts — in lockstep with the golden tests' paths, `PIPELINE.md`'s script→artifact map, and the
  figures/manuscript builders.
- ⬜ **P3.2** Add a real v2 orchestrator (`00_run_all`) that runs stage0→stage4 → stratify → inventory
  → Studies A–D → sensitivity → figures → manuscript and writes `results/hfa/`.
- ⬜ **P3.3** Renumber to a contiguous, phase-grouped sequence so the filename prefix is the run order.
- ⬜ **P3.4** Scrub external-project provenance from `src` docstrings (~15 `face_stratification` /
  `face_rlvr` refs; the misleading `__init__.py` "imports face_stratification" comment; dead
  `config/face_stratification/…` / `docs/face_stratification/…` path lookups).
- ⬜ **P3.5** Prune dead `__init__.py` exports (`PatientFilterReport`, `COGNITION_SECTIONS`; review
  `DEFAULT_SCHEMA_VERSION`, `identity_cast`).

## P4 — Polish

- ⬜ Delete `.env.example` (unused API keys). Make `manuscript.md` figure paths relative (currently
  absolute `/Users/…`). Regenerate `requirements.lock` on Python 3.11 (CI pins 3.11; lock built on
  3.12). Update CI branch triggers. Confirm `neuropsy_features.yaml` lists the new CVLT/fluency
  features. Drop the now-unused `notebook` extra from `pyproject.toml`.
- ⬜ **CI lint gate is red.** `ruff check .` reports **143 errors** (was **150** before this cleanup);
  `[tool.ruff.lint]` in `pyproject.toml` is empty, so ruff runs its full default rule set with no
  ignores. Dominant: **125 × E402** (module-import-not-at-top) from the `sys.path.insert(0, "src")`-
  before-`import trans_diag` bootstrap in every script, plus a few F401/F541/B905. CI's `ruff check .`
  step therefore fails. Decide: add `# noqa: E402` on the bootstrap imports, configure
  `[tool.ruff.lint.per-file-ignores]` for `scripts/*`, or move the path bootstrap into
  `tests/conftest.py` + a `scripts/_bootstrap.py`. (Not a P0/P1 regression — pre-existing.)

---

## Verification (run after each phase)

```bash
python3 -m pytest tests/ -q     # expect 99 passed (golden tests need a local results/hfa/)
ruff check .
python3 scripts/verify.py       # harmonization smoke test
```

## Reviewed and intentionally KEPT as-is

`docs/PIPELINE.md` (excellent end-to-end map — number-sync only), `docs/LABBOOK.md` (one missing
entry), the `engine/` subpackage (all 7 modules used), `masked_fa.py`/`domains.py`/`skip_logic.py`,
the test-suite design, the manuscript content, the CI workflow file (its lint *gate* is red — see
P4), and `pyproject.toml` (deps are clean; the empty `[tool.ruff.lint]` is the lint-gate issue).
