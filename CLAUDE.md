# CLAUDE.md — FACE Common (BP · SZ · DR)

> Project guide for human collaborators and AI assistants. Keep this short. For
> the full research plan see [ROADMAP.md](ROADMAP.md); for the data dictionary
> see [DATA.md](DATA.md).

## What this is

A harmonization + clustering pipeline for the FACE multi-cohort psychiatric
dataset (Bipolar, Schizophrenia, Depression). The goal is **data-driven
trans-diagnostic clustering** of patients across the three DSM-5 categories
using a common variable dictionary, then comparing the discovered clusters
against DSM-5 labels and tracking their stability across annual follow-up
visits (V0 baseline → V4).

## Data inputs (read-only)

- `face-common-vars.xlsx` — Sheet1 dictionary (379 rows × 20 cols). Each row =
  one harmonized variable, with per-cohort source-column names, target dtype,
  value set, clinical rationale, and harmonization rule. **Authoritative.**
- `data/bipolar.csv` — 21,343 visit-level rows, 6,252 patients (BP).
- `data/schizophrenia.csv` — 6,203 rows, 2,209 patients (SZ).
- `data/depression.csv` — 1,953 rows, 552 patients (DR).
- `thesaurus/*.xlsx` — original per-cohort thesauri; reference only.

## Repository layout

```
face-common-bp-sz-dr/
├── CLAUDE.md                  ← this file
├── DATA.md                    ← dictionary reading guide
├── ROADMAP.md                 ← research plan, milestones, open questions
├── face-common-vars.xlsx      ← dictionary (input)
├── data/                      ← raw CSVs (input, large)
├── thesaurus/                 ← original per-cohort xlsx (reference)
├── face_common/               ← the library
│   ├── __init__.py
│   ├── variable.py            ← Variable dataclass + load_variables()
│   ├── rules.py               ← harmonization registry (RULES dict)
│   └── loader.py              ← build_unified_dataframe()
├── scripts/                   ← runnable analysis scripts
│   ├── verify.py              ← end-to-end smoke test
│   ├── audit.py               ← per-variable correctness audit
│   └── qa_missingness.py      ← interactive HTML missingness report
├── results/                   ← CSV/JSON outputs (audit_report.csv, qa_missingness.csv …)
├── reports/                   ← HTML / figure artifacts (qa_missingness.html …)
└── tests/                     ← unit tests (currently empty, see roadmap)
```

## Core concepts

**`Variable`** (`face_common/variable.py`) — one instance per dictionary row.
Carries `canonical_name`, `bp_csv_col` / `sz_csv_col` / `dr_csv_col`, `dtype`,
`unit_or_value_set`, `cluster_readiness`, `clinical_rationale`, `rule`,
`section`, `label`, `findings`. Method `source_col(cohort)` returns the right
CSV column for a cohort or `None`.

**Harmonization registry** (`face_common/rules.py`) — module-level dict
`RULES: {canonical_name → callable(series, cohort) → series}`. Register custom
transformers with `@register("canonical_name")`. Anything unregistered falls
through to `identity_cast`, which handles standard dtype coercion and warns
when produced values fall outside the dictionary's declared value set.

**Visits** — CSV column `visit` holds labels like `V0`, `V1_an`, `V2_ans`,
`V6_mois`, `screening`. The loader keeps **only yearly visits** and recodes
them to `V0..V10`. `visitnum` is a global row id (not a sequence) and is
preserved as metadata only.

**Identifiers** (never clustered on) — `patient_uid`, `usubjid_patients`,
`cohort`, `arm` (DSM-5 text label), `visit`, `visitnum`. Always present
unmodified in the output. `arm` is reserved for post-clustering evaluation
(ARI vs clusters). **`patient_uid = cohort::usubjid_patients`** is the
globally-unique patient key — `usubjid_patients` alone is only unique within
a cohort (970 ids are reused across BP/SZ/DR), so all patient-level
operations must key on `patient_uid`.

**Readiness filter** — `build_unified_dataframe(..., readiness=[...])` is
**required**. Use `['READY']` for the 130-variable clean set or `['READY',
'PARTIAL']` for the 351-variable richer set. Prefix-matched on the dictionary's
multi-word readiness strings.

## Quick start

```bash
# 1. Smoke test (≈ 30 s) — confirms the pipeline runs end-to-end
python3 scripts/verify.py

# 2. Per-variable correctness audit (writes results/audit_report.csv)
python3 scripts/audit.py

# 3. Interactive missingness QA report (writes reports/qa_missingness.html)
python3 scripts/qa_missingness.py
open reports/qa_missingness.html
```

```python
# Use the library
from face_common import build_unified_dataframe

df = build_unified_dataframe(
    data_dir="data",
    dictionary_path="face-common-vars.xlsx",
    readiness=["READY", "PARTIAL"],     # required
    format="long",                       # 'long' or 'wide'
)
# To get the feature matrix for clustering:
features = df.drop(columns=["patient_uid", "usubjid_patients", "cohort",
                            "arm", "visitnum", "visit"])
```

## Conventions

- **Python ≥ 3.10**, pandas, numpy, plotly, matplotlib, openpyxl.
- **Editing the dictionary** (`face-common-vars.xlsx`) is allowed but logged
  carefully — the audit and QA scripts use it as the source of truth.
- **Adding a new harmonization rule**: append a `@register("canonical_name")`
  function in `face_common/rules.py`. Re-run `python3 scripts/audit.py` to
  verify it cleared the relevant value-set warning.
- **Output paths**: scripts always write to `results/` (data) or `reports/`
  (HTML / figures). Do not write to repo root.
- **No notebooks in the main path** — keep analysis reproducible as scripts.
  Ad-hoc exploration is fine in a personal scratch dir but not committed.

## Status (as of last update)

- Pipeline: 348/348 feature variables passing the audit (0 FAIL, 45 WARN — all
  legitimate clinical heterogeneity, not bugs).
- 31 custom harmonization rules registered; 317 fall through to identity_cast.
- DR cohort has a known V3 attrition cliff (only 3 patient×visit rows). Flag
  prominently in any V3-dependent analysis.
- See `reports/qa_missingness.html` for the current per-variable missingness
  snapshot at V0..V4.

## Where to read next

- **What the dictionary columns mean** → [DATA.md](DATA.md)
- **What we're building toward and why** → [ROADMAP.md](ROADMAP.md)
- **The library API** → `face_common/__init__.py` (re-exports the public surface)
