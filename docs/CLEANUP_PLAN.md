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
- ✅ **P0.4** Reconciled headline numbers against `results/hfa/` and propagated to the docs + golden
  test: **194** items, **94** constructs, **81** Stage-3 inputs, **ECV 0.34**, **56** eig>1, canonical
  r **0.99/0.90/0.79**; dictionary **199 usable** (READY+PARTIAL) of **223** entries.
  **Flagged for the scientist (not doc bugs):** (a) the manuscript says "220-variable dictionary" —
  it counts curated entries minus identifiers, a *different denominator* than the 199 usable; reconcile
  the wording in the paper. (b) **RESOLVED:** `40_phase5`'s arm-B "75" was a stale hardcoded label —
  the actual set is the same coverage≥0.30 filter (**81**, confirmed by Study-D3's `n_constructs: 81`);
  the label is now computed dynamically and the docs all say 81.

## P1 — Remove the v1 layer

- ✅ Delete scripts: `00_run_all`, `01`–`13`, `15`–`22`, `sensitivity_masked_fa`,
  `sensitivity_masked_fa_mechanism`, `build_notebook`, `qa_missingness`.
- ✅ Delete source: `src/trans_diag/axes.py` (v1), `src/trans_diag/outcomes.py`.
- ✅ Delete `notebooks/FACE_reproduction.ipynb` and root `MANUSCRIPT.md`.

## P2 — Documentation rewrite (point everything at "what it is now")

- ✅ **P2.1** Entry docs rewritten (P0.1); `docs/PIPELINE.md` linked from `README`; the stale flat-
  domain methodology ("~69 domain scores … feed the model") corrected to the hierarchical constructs
  in `CLAUDE.md` + `README.md`.
- ✅ **P2.2** Numbers synced in `FINDINGS`/`PIPELINE`/`ROADMAP`/`DATA`/`CLAUDE` + golden test; fixed
  `FINDINGS.md`'s false opening "No analysis results yet …" and `ROADMAP.md`'s "91 passed" (→99).
- ✅ **P2.3** Archived the executed pre-registration plans (`HIERARCHICAL_FA_PLAN.md`,
  `VALIDATION_PLAN_v2.md`, `MANUSCRIPT_PLAN.md`) to `docs/planning/` with an "ARCHIVED — executed"
  banner + a `docs/planning/README.md`; repointed all inbound links.
- ✅ **P2.4** Added `LABBOOK.md` V2-21 (the 2026-06-03 dictionary review: 188→194 items, 88→94
  constructs, ECV 0.36→0.34, memory-anchored cognition, suicide skip-logic).
- ✅ **P2.5** Removed root `todo_data_cleaning.md` (working notes; all actioned decisions are encoded
  in the dictionary + `rules.py`); folded the one open caveat (`ltsg07`) into `DATA.md`.
- ⬜ Rewrite `CLAUDE.md`'s "three stages / why aggregate" section: it still describes the **flat
  masked-mean** domain scores (~69), which v2 **replaced** with the hierarchical/bifactor model
  (88/94 constructs).

## P3 — Structure & naming

- ⬜ **P3.1 — deferred (needs a verifying pipeline run).** Dropping `_v2` from script + artifact
  filenames is a repo-wide rename touching importlib script-loads + ~25 *gitignored* artifact
  read/write paths across ~20 scripts. The golden tests catch broken *read* paths, but a missed *write*
  path only surfaces on the next pipeline run — which needs the confidential CSVs (can't run here).
  Recommend: one rename commit immediately followed by `python3 scripts/00_run_all.py` to verify.
  (`axes_v2 → axes` was already done in P0.2.)
- ✅ **P3.2** Added `scripts/00_run_all.py` (v2): QA → Stages 0–4 → stratify → inventory → Studies A–D
  → sensitivity → figures, in `PIPELINE.md` order (dependency-checked: 41 before 45–48; 32 before 44;
  44 before 48). Golden-test + README/CLAUDE regeneration hints repointed to it. *(Structure verified —
  compiles, ruff-clean, all 20 steps exist; a full run needs the confidential CSVs.)*
- ⬜ **P3.3 — deferred (same reason as P3.1).** Renumbering also moves the importlib filename strings;
  pair it with the P3.1 rename + a verifying `00_run_all` run.
- ✅ **P3.4** Fixed the actively-misleading refs (`__init__.py` "imports", `schema_gen.py` +
  `feature_schema.py` external-module/config pointers, `enrichment.py` dead-doc link) and **deleted the
  dead `load_feature_schema` YAML loader** (never called; it carried the `config/face_stratification/`
  path) + its now-orphaned imports. The legitimate "trimmed/adapted from the FACE engine, external deps
  removed" provenance notes stay — they match the self-contained claim.
- ✅ **P3.5** Removed the dead `__init__.py` exports `PatientFilterReport`, `COGNITION_SECTIONS`.

## P4 — Polish

- 🔜 **P4** Done: deleted `.env.example`; removed dead `ai`/`combat`/`notebook` extras + trimmed
  `install.py`; **updated CI push triggers** (`main`, `v2-study`); confirmed `neuropsy_features.yaml`
  already lists the CVLT/fluency features. **Remaining (need a build/network unavailable here):**
  `manuscript.md`'s 6 figure paths are absolute `/Users/…` (broken on any other machine) — the fix is
  repo-root-relative paths + `cwd=ROOT` in `build_manuscript_v2`'s pandoc call, but the `.docx` build
  can't be verified here (no pandoc), so it's left for a tested pass; regenerate `requirements.lock` on
  Python 3.11 (it still pins the removed torch/neuro deps; `pip-compile` is unavailable here).
- ✅ **CI lint gate is now green.** `ruff check .` passes (was **150** errors). Fix: added **E402** to
  the curated `[tool.ruff.lint] ignore` (with rationale — scripts insert `src/` on `sys.path` before
  importing `trans_diag`), plus ruff autofixes (F541/B905/I001/unused-imports) and 2 manual edits
  (a `rules.py` `isinstance` PEP-604 modernization; a `# noqa: E712` preserving a NaN-safe filter).

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
