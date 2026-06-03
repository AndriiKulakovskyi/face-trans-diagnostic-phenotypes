# CLAUDE.md — FACE trans-diagnostic phenotyping (BP · SZ · DR) — v2 study

> Guide for collaborators and AI assistants. Keep it short. Paper: [results/manuscript/manuscript.md](results/manuscript/manuscript.md).
> Plan: [docs/ROADMAP.md](docs/ROADMAP.md). Dictionary: [docs/DATA.md](docs/DATA.md).
> Findings: [docs/FINDINGS.md](docs/FINDINGS.md) · Lab notebook: [docs/LABBOOK.md](docs/LABBOOK.md).
> **Pipeline diagram (end-to-end): [docs/PIPELINE.md](docs/PIPELINE.md).**
> Cognition include-list: [docs/neuropsy_features.yaml](docs/neuropsy_features.yaml).

## What this is

A **self-contained** project that harmonizes the 3-cohort FACE psychiatric data (Bipolar,
Schizophrenia, Depression; baseline **V0** → 4-year V4) and models **trans-diagnostic** structure
two ways: **(1) dimensional analysis** (latent symptom / biology / cognition dimensions) and
**(2) patient stratification** (clustering). **No imputation anywhere** — pervasive missingness is
handled by *masked* methods (pairwise-complete correlation, masked similarity, masked posterior
scores).

**This is the v2 study — a clean restart on a re-curated dictionary.** The earlier (v1) analysis is
archived at git tag **`v1-archive-2026-05-30`** (branch `archive/v1-research`); we stopped trusting
the v1 common-variables set and now **re-derive every result from zero** on v2 — no v1 finding is
assumed. Work happens on branch `v2-study`.

The stratification **engine** (masked similarity → multipartite-spectral embedding, enrichment) is
internalized in `src/trans_diag/engine/` — no external dependency on `face_stratification`/`face_rlvr`.

## Status (v2 — analysis complete)

- ✅ **v2 dictionary finalized + locked** — **199 usable variables** (READY + PARTIAL, of 223 entries),
  with structured sanity bounds + coverage. Cognition reconciled to `docs/neuropsy_features.yaml`
  (3-cohort WAIS/TMT + verbal memory/fluency features + covariates). `qa_harmonization`: all variables
  load + pass sanity, 0 fail.
- ✅ **Preprocessing debugged + ML-ready** — fixed the robust-z explosion (prolactin |z|≈106→5),
  added **type-aware bounded scaling to [−1,1]**, kept the masked / no-imputation design. QA report
  has the three parts described below.
- ✅ **Dimensional analysis + stratification re-derived on v2** (scripts `30–48_v2`): 4 trans-diagnostic
  axes (K=4, no p-factor), dimensional / no discrete subtypes, validation arm A–D. **Manuscript + 6
  figures delivered** (`results/manuscript/`, `scripts/figures_manuscript_v2.py`). **Golden-number tests
  + `verify.py` re-baselined to v2** — `tests/test_golden_numbers.py` pins the manuscript's headline
  numbers to `results/hfa/` (pass locally; skip on a clean clone since `results/hfa/` is gitignored).
  The legacy `01–22` pipeline and the old manuscript skeleton have been removed from the tree;
  they remain recoverable at git tag `v1-archive-2026-05-30`.

## Data processing — three stages (= the QA report's three Parts)

`scripts/qa_harmonization.py` → `results/reports/qa_harmonization.html` shows the data at each stage.
This is the debugging surface that must be clean *before* any analysis.

1. **Part 1 — Harmonized variables (native scale).** Each dictionary variable is read from its
   per-cohort source column, run through its harmonization rule (`rules.py`: text→code, unit fixes)
   and per-variable **sanity bounds** (out-of-range → NaN, **never imputed**), landing on its native
   clinical scale (TMT seconds, WAIS 1–19, Likert 0–3, binary 0/1, labs in clinical units). Verifies:
   every variable loads, is numeric (no text leaked through), within bounds, cross-cohort comparable.

2. **Part 2 — Post-processed variables (type-aware scaling → [−1,1]).** The *same* variables on a
   common, bounded, ML-ready scale, **by type**: binary/ordinal/Likert → min-max; continuous → log
   (if heavy right-skewed, e.g. prolactin) + winsorize(1/99) + robust-z (median/MAD) clipped ±5, ÷5.
   Puts a lab in the thousands and a 0/1 flag on the same footing and bounds outliers. Verifies:
   every feature lands in [−1,1] (`normalize_for_embedding`, `adapter.py`).

3. **Part 3 — Aggregated V0 domain scores (the QA view of construct-level features).**
   Items are aggregated into **construct-level domain scores** (each item robust-z'd + sign-oriented,
   then a **masked mean** within its instrument/composite; no imputation), at the **baseline V0**
   visit. The *actual model inputs* are the richer **hierarchical/bifactor constructs** (194 items →
   94 within-construct masked one-factor posteriors → 4 second-order axes; see
   [docs/PIPELINE.md](docs/PIPELINE.md) §5), which supersede flat masked means; these domain scores
   remain the interpretable QA cross-check.

   **Why aggregate into domain scores (what "Encoded modelling features (aggregated V0 domain scores)" means and why we need it):**
   - **Each construct counts once.** A construct measured by many items (e.g. a 30-item suicide
     instrument) would otherwise be ~30 % of the dimensions in a similarity/correlation computed over
     raw items, drowning out single-item constructs. Aggregating items → one score removes this
     **item-count weighting bias** (LABBOOK E-series rationale).
   - **More coverage, still no imputation.** A domain score needs only *some* of its items observed
     (masked mean with a min-item floor), so it is far better-covered than any single item.
   - **Interpretability.** Structure over ~90 *named* clinical constructs (depression, mania,
     metabolic, cognition…) is interpretable; over ~194 raw items it is noise.
   - **Comparable units.** Robust-z'ing each item before averaging lets members on different units
     (mmol/L, mmHg, Likert points) combine sensibly into one construct.
   - **V0 = the analysis anchor.** Dimensions/clusters are *defined* at baseline; later visits
     (V1, V2…) are used to test their **temporal coherence**, not to define the structure.

## Repository layout

```
face-common-bp-sz-dr/
├── CLAUDE.md  AGENTS.md  README.md    ← guides (root; paper at results/manuscript/manuscript.md)
├── data/                              ← inputs (read-only)
│   ├── face-common-vars.xlsx          ← v2 common-variables dictionary (tracked)
│   ├── thesaurus/                     ← per-cohort source dictionaries (tracked, reference)
│   └── {bipolar,schizophrenia,depression}.csv · site_lookup.csv   ← data (confidential; gitignored except site_lookup)
├── src/trans_diag/                    ← the package (all our code)
│   ├── variable·rules·loader·filters.py        ← harmonization + sanity bounds
│   ├── schema_gen·adapter·domains.py           ← matrix build, type-aware scaling, domain aggregation
│   ├── masked_fa·axes·skip_logic.py            ← imputation-free FA, axis names, suicide skip-logic
│   └── engine/                                 ← internalized stratification engine (masked, no imputation)
├── scripts/                           ← v2 pipeline (30–48_v2: hierarchical FA → stratify → validate) + qa_harmonization/verify/audit
├── tests/                             ← unit + v2 golden-number tests (pinned to results/hfa/; skip on a clean clone)
├── results/                           ← regenerated AGGREGATE artifacts: hfa/ · manuscript/ · reports/ (empty on a clean tree; .gitkeep)
│   └── reports/qa_harmonization.html  ← the 3-part QA report
├── docs/                              ← PIPELINE·ROADMAP·DATA·FINDINGS·LABBOOK·neuropsy_features.yaml
└── pyproject.toml
```

**Imports.** `trans_diag` resolves from `src/`. Scripts insert `src/` on `sys.path`; pytest uses
`pythonpath = ["src"]`. Or `pip install -e ".[full]"`.

## Core concepts

**`Variable`** (`variable.py`) — one per dictionary row; `source_col(cohort)` → CSV column; carries
`sanity_min/max` (v2). **Harmonization registry** (`rules.py`) — `@register(...)`; unregistered →
`identity_cast`. **`build_unified_dataframe(...)`** (`loader.py`) — `readiness=['READY','PARTIAL']`
(199 vars); auto-detects v2 (any sanity bound present) → applies sanity bounds + v2 rules + fondacode
site; `format='long'|'wide'`. **`to_harmonized_dataset(...)`** (`adapter.py`) — V0 numeric matrix,
MultiIndex `[cohort, patient_id]`, optional `residualize_on=('age','sex')`; `normalize_for_embedding`
= **type-aware scaling to [−1,1]**. **Domain aggregation** (`domains.py`) — items → construct-level
domain scores (masked mean of robust-z, clipped ±5) + curated biology/cognition composites.
**Identifiers (never modelled on):** `usubjid_patients`, `cohort`, `arm`, `visit`, `visitnum`,
`siteid_city` (kept loadable for site stratification, excluded from features via `ADMINISTRATIVE_FEATURES`).
**No imputation** anywhere — masked similarity, masked FA (pairwise-complete correlation → masked
posterior-mean scores), all on observed cells only.

## Quick start

```bash
pip install -e ".[full]"                 # core + torch + neuroHarmonize + kaleido
python3 scripts/qa_harmonization.py      # the 3-part data-processing QA report (must be clean first)
python3 -m pytest tests/ -q              # unit + v2 golden tests (golden skip on a clean clone — results/hfa/ gitignored)
python3 scripts/verify.py                # harmonization smoke test (v2-calibrated)
```

```python
from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset
df = build_unified_dataframe("data", "data/face-common-vars.xlsx",
                             readiness=["READY", "PARTIAL"], format="long")
ds = to_harmonized_dataset(df, load_variables("data/face-common-vars.xlsx"), visit="V0")
# ds.X: MultiIndex[cohort, patient_id] × numeric features (NaN = missing, never imputed)
```

## Pipeline (`scripts/`) — the v2 analysis, in execution order

The hierarchical/bifactor measurement model + validation arm (all masked / no-imputation), writing
aggregate artifacts to `results/hfa/`:

- **Stages 0–4** — `30_hfa_stage0_itemset_v2` (freeze the V0 item set) → `31_hfa_stage1_efa_v2`
  (exploratory first-order) → `32_hfa_stage2_v2` (hybrid first-order constructs) → `33_hfa_stage3_v2`
  (second-order: **K=4** axes; general factor tested via Schmid–Leiman ECV) → `34_hfa_kselect_v2`
  (per-factor split-half K) → `35_hfa_stage4_v2` (confound / leave-cohort-out / granularity validation).
- **Stratification** — `40_phase5_stratify_v2` (discrete-vs-continuum battery → **dimensional**).
- **Validation A–D** — `41_v1v4_inventory_v2` (relapse derivation) → `42_cohort_confound_v2` (A) ·
  `43_orthogonality_pfactor_v2` (B, the headline) · `44_longitudinal_coherence_v2` (C) ·
  `45`–`48` predictive (D: prognosis vs DSM).
- **Sensitivity** — `sensitivity_{aggregation,comorbidity,polychoric}_v2`.
- **Outputs** — `figures_manuscript_v2` (6 figures) · `build_manuscript_v2` (→ `.docx`).

End-to-end diagram + mathematics: **[docs/PIPELINE.md](docs/PIPELINE.md)**.

## Conventions

- **Python ≥ 3.11.** Develop in `src/trans_diag` (incl. `engine/`).
- **No imputation, ever.** Missingness is handled by masked methods (no hard missingness drop).
- **Output**: scripts write aggregates to `results/`, HTML/figures to `results/reports/`.
- **Determinism**: fixed seeds; CV folds shuffled (the patient matrix is cohort-ordered).
- **Recoverability**: the full v1 study is at tag `v1-archive-2026-05-30`.

## Where to read next

- **Pipeline (end-to-end diagram + math)** → [docs/PIPELINE.md](docs/PIPELINE.md)
- **Plan** → [docs/ROADMAP.md](docs/ROADMAP.md) · **Dictionary columns** → [docs/DATA.md](docs/DATA.md)
- **Findings (v2 log)** → [docs/FINDINGS.md](docs/FINDINGS.md) · **Lab notebook** → [docs/LABBOOK.md](docs/LABBOOK.md)
- **Manuscript** → [results/manuscript/manuscript.md](results/manuscript/manuscript.md) · **Engine internals** → `src/trans_diag/engine/`
