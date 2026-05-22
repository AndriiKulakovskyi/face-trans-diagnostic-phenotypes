# CLAUDE.md — FACE trans-diagnostic clustering (BP · SZ · DR)

> Guide for collaborators and AI assistants. Keep it short. Full research plan:
> [ROADMAP.md](ROADMAP.md). Dictionary reading guide: [DATA.md](DATA.md).

## What this is

A **merged project** for data-driven **trans-diagnostic clustering** of the FACE
psychiatric cohort (Bipolar, Schizophrenia, Depression). Two halves:

1. **Our pipeline** (`src/face_common`) — harmonizes the 3-cohort **longitudinal**
   data (V0 baseline → V4) from the common-variables dictionary into a unified
   patient × feature matrix. **This is the feature/data source.**
2. **The vendored engine** (`archive/face_stratification`) — a sister project's
   modelling algorithms (masked similarity → multipartite-spectral embedding →
   consensus clustering → validation), **reused, not developed**. We feed it our
   matrix; its original 4-cohort clusters are a comparison **reference**.

Goal: discover clusters that cut across DSM-5, track their **temporal coherence**
V0→V4, and relate them to outcomes (the "FACE Score").

## Repository layout

```
face-common-bp-sz-dr/
├── CLAUDE.md  ROADMAP.md  DATA.md   ← docs (ours)
├── face-common-vars.xlsx            ← the common-variables dictionary (input)
├── src/
│   └── face_common/                 ← OUR development base (the only code we write)
│       ├── variable.py  rules.py  loader.py  filters.py
│       ├── schema_gen.py  adapter.py  ← engine bridge (dictionary→schema, frame→HarmonizedDataset)
├── archive/                         ← copied sister code — VENDORED, do not develop here
│   ├── face_stratification/         ← the reused engine (importable)
│   ├── face_rlvr/                   ← engine's patient extractors + glossary loader
│   ├── data/ scripts/ notebooks/ tests_face_stratification/ docs/ output/
├── config/                          ← engine config (feature schema + glossary; vendored, kept at root)
├── data/                            ← OUR 3-cohort longitudinal CSVs + data/external (engine reference artifacts)
├── scripts/                         ← OUR runnable scripts (verify, audit, qa_missingness, v0_anchor, phase2*, reproduce_v0_clusters)
├── tests/                           ← OUR tests (test_filters.py, test_adapter.py)
├── results/  reports/               ← our outputs
└── pyproject.toml                   ← packages: src/face_common + archive engine; pytest pythonpath = [src, archive]
```

**Imports.** `face_common` resolves from `src/`; `face_stratification` /
`face_rlvr` from `archive/`. pytest is configured for both; scripts insert
`src/` + `archive/` on `sys.path`. (Or `pip install -e .`.)

## Data inputs (read-only)

- `face-common-vars.xlsx` — Sheet1 dictionary (379 rows × 20 cols). Each row =
  one harmonized variable: per-cohort source columns, `dtype`, value set,
  `section` (13 clinical blocks), `cluster_readiness`, rationale, rule.
- `data/bipolar.csv` (6,252 patients), `data/schizophrenia.csv` (2,209),
  `data/depression.csv` (552) — visit-level rows (V0–V4).
- `archive/data/{BP,SZ,DR,ASP}.csv` — the sister's V0-only extracts (reference).

## Core concepts (our pipeline)

**`Variable`** (`face_common/variable.py`) — one per dictionary row;
`source_col(cohort)` returns the right CSV column or `None`.

**Harmonization registry** (`face_common/rules.py`) — `RULES: {canonical_name →
callable(series, cohort)}`. Register with `@register(...)`; unregistered falls
to `identity_cast` (dtype coercion + value-set warnings).

**`build_unified_dataframe(...)`** (`face_common/loader.py`) — required
`readiness=[...]` (`['READY']` = 130 vars, `['READY','PARTIAL']` = 351);
`format='long'|'wide'`. Keeps only yearly visits, recoded `V0..V10`.

**Filters** (`face_common/filters.py`) — `filter_variables`, `filter_patients`,
`V0Anchor`, `select_v0_anchor` (completeness floors, visit-scoped).

**Engine bridge** (`face_common/schema_gen.py` + `adapter.py`) —
`to_harmonized_dataset(df, variables, visit='V0')` reshapes our matrix into the
engine's `HarmonizedDataset` (numeric float `X`, MultiIndex `[cohort,
patient_id]`); `build_feature_schema(...)` turns the dictionary into the engine's
`FeatureSchema` (`section`→block, `dtype`→type). Drives `scripts/cluster_v0.py`.

**`patient_uid = cohort::usubjid_patients`** — the globally-unique key
(`usubjid_patients` is reused across cohorts; 970 collisions). All patient-level
ops key on it.

**Identifiers (never clustered on):** `patient_uid`, `usubjid_patients`,
`cohort`, `arm`, `visit`, `visitnum`.

**No imputation** — the engine uses masked pairwise-complete similarity; missing
values are never filled (optional KNN/MICE imputers exist but are off).

## Quick start

```bash
python3 scripts/verify.py          # end-to-end smoke test (~30 s)
python3 scripts/audit.py           # per-variable correctness audit → results/
python3 scripts/qa_missingness.py  # interactive HTML report → reports/
python3 scripts/cluster_v0.py      # V0 3-cohort clustering via engine → results/cluster_v0_*
python3 -m pytest tests/ -q        # unit tests
```

```python
from face_common import build_unified_dataframe
df = build_unified_dataframe("data", "face-common-vars.xlsx",
                             readiness=["READY", "PARTIAL"], format="long")
features = df.drop(columns=["patient_uid", "usubjid_patients", "cohort",
                            "arm", "visitnum", "visit"])

# Feed our V0 matrix into the vendored engine (no imputation):
from face_common import load_variables, to_harmonized_dataset
ds = to_harmonized_dataset(df, load_variables("face-common-vars.xlsx"), visit="V0")
# ds.X is MultiIndex[cohort, patient_id] × numeric features; ds.schema is a
# face_stratification FeatureSchema generated from our dictionary.
```

## Conventions

- **Python ≥ 3.11**; pandas, numpy, scikit-learn, scipy, networkx, plotly,
  matplotlib, openpyxl. Engine extras (torch, leidenalg…) via
  `pip install -e ".[stratification]"`.
- **Develop only in `src/face_common`.** `archive/` is vendored copied code —
  reuse by import, do not edit it.
- **Adding a harmonization rule**: `@register(...)` in `rules.py`, then re-run
  `scripts/audit.py`.
- **Output paths**: scripts write to `results/` (data) or `reports/` (HTML).
- The clustering runs on **our common-variables features** (block = `section`,
  metric by `dtype`); the engine's 184-feature schema is the reference only.

## Status

- Harmonization: 348/348 feature variables PASS the audit (0 FAIL, 45 WARN).
- Merge done; engine reproduces the sister 4-cohort clusters exactly
  (`results/v0_clusters_anchor.csv`); **no imputation** confirmed.
- **Phase 3 (3-cohort recovery) — first result done.** `schema_gen.py` +
  `adapter.py` bridge our V0 matrix into the engine; `scripts/cluster_v0.py`
  embeds + clusters it. At k=6: bootstrap ARI **0.96** (stable), recovers a
  clean SZ cluster + a BP–DR mood bridge (all 552 DR co-cluster with BP);
  moderate match to the 4-cohort reference (ARI **0.31**). 40 tests pass.
- DR has a V3 attrition cliff (3 patient×visit rows) — exclude from V3.
- Next: name clusters (feature enrichment + Cohen's d), verify metabolic
  direction, ablations (READY-only / core-67), then Phase 4 longitudinal.

## Where to read next

- **Dictionary columns** → [DATA.md](DATA.md)
- **Plan, methods, paper framing, course-corrections** → [ROADMAP.md](ROADMAP.md)
- **Engine internals** → `archive/docs/` (sister stage docs)
