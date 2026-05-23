# CLAUDE.md — FACE trans-diagnostic dimensional phenotyping (BP · SZ · DR)

> Guide for collaborators and AI assistants. Keep it short. Paper draft:
> [MANUSCRIPT.md](MANUSCRIPT.md). Plan: [ROADMAP.md](ROADMAP.md). Dictionary guide:
> [DATA.md](DATA.md). Findings log: [FINDINGS.md](FINDINGS.md) · [LABBOOK.md](LABBOOK.md).

## What this is

A **self-contained** project that harmonizes the 3-cohort FACE psychiatric data
(Bipolar, Schizophrenia, Depression; baseline V0 → 4-year V4) and models
**trans-diagnostic** structure across DSM-5. Headline result: trans-diagnostic
variation is **dimensional, not categorical** — six reproducible, confound-controlled
symptom dimensions that complement/outperform DSM diagnosis for patient-reported
outcomes (full write-up in `MANUSCRIPT.md`).

The stratification **engine** (masked similarity → multipartite-spectral embedding,
cluster enrichment, factor scaffolding) was originally a sister project; the pieces we
use are now **internalized** in `src/trans_diag/engine/` — the repo has **no external
dependency on `face_stratification`/`face_rlvr`**.

## Repository layout

```
face-common-bp-sz-dr/
├── MANUSCRIPT.md  CLAUDE.md  ROADMAP.md  DATA.md  FINDINGS.md  LABBOOK.md  README.md
├── face-common-vars.xlsx            ← the common-variables dictionary (input)
├── data/                            ← 3-cohort longitudinal CSVs (confidential; gitignored)
├── src/trans_diag/                 ← the package (all our code)
│   ├── variable.py  rules.py  loader.py  filters.py   ← harmonization
│   ├── schema_gen.py  adapter.py  domains.py          ← matrix build + domain aggregation
│   └── engine/                      ← internalized stratification engine
│       ├── feature_schema.py  harmonized_dataset.py   ← data contracts
│       ├── masked_similarity.py  spectral_base.py  multipartite.py  ← embedding (no imputation)
│       ├── enrichment.py  clustering.py               ← enrichment + kmeans/bootstrap
├── scripts/                         ← pipeline (00_run_all.py orchestrates) + verify/audit/qa infra
├── tests/                           ← unit tests (filters, adapter, domains)
├── results/  reports/               ← reproducible artifacts (CSV/JSON/parquet) + HTML + figures
└── pyproject.toml                   ← packages = src/trans_diag; deps; [full] extras
```

**Imports.** `trans_diag` resolves from `src/`. Scripts insert `src/` on `sys.path`;
pytest uses `pythonpath = ["src"]`. Or `pip install -e ".[full]"`.

## Data inputs (read-only, confidential)

- `face-common-vars.xlsx` — dictionary: one row per harmonized variable (per-cohort
  source columns, `dtype`, value set, `section`, `cluster_readiness`, rule).
- `data/{bipolar,schizophrenia,depression}.csv` — 6,252 / 2,209 / 552 patients,
  visit-level rows (V0–V4). **Confidential.** ⚠️ Currently *tracked* in git — must be
  removed (gitignore + history scrub) before sharing the repo externally.
- `results/v0_clusters_anchor.csv` — the sister 4-cohort clusters projected onto our
  ids; a reference used only by `02_confound_ladder.py` (ARI-vs-sister in §3.1).

## Core concepts

**`Variable`** (`variable.py`) — one per dictionary row; `source_col(cohort)` → CSV column.
**Harmonization registry** (`rules.py`) — `@register(...)`; unregistered → `identity_cast`.
**`build_unified_dataframe(...)`** (`loader.py`) — `readiness=['READY','PARTIAL']` (351 vars),
`format='long'|'wide'`; yearly visits recoded `V0..V10`.
**Filters** (`filters.py`) — variable/patient filters, V0 anchoring.
**Engine bridge** (`schema_gen.py`+`adapter.py`) — `to_harmonized_dataset(df, variables,
visit='V0', sections=…, residualize_on=('age','sex'), normalize=…, exclude=…)` → a
`trans_diag.engine.HarmonizedDataset` (numeric `X`, MultiIndex `[cohort, patient_id]`);
`residualize_features` (spline + cross-fit) and `normalize_for_embedding` (robust z) live here.
**Domain aggregation** (`domains.py`) — items → construct-level domain scores (masked mean
of robust-z, min-item floor) + curated biology composites.
**`patient_uid = cohort::usubjid_patients`** — globally-unique key (usubjid collides across
cohorts; 970 collisions). **Identifiers (never modelled on):** `patient_uid`, `usubjid_patients`,
`cohort`, `arm`, `visit`, `visitnum`.
**No imputation** on the masked-similarity embedding and the autoencoder objective; the
factor-analysis input is the one place gaps are mean-filled (the residual matrix is ~65%
observed — see MANUSCRIPT §2.1).

## Quick start

```bash
pip install -e ".[full]"             # core + torch (AE) + neuroHarmonize (ComBat) + kaleido (figures)
python3 scripts/00_run_all.py           # reproduce the whole manuscript pipeline (~5 min)
python3 -m pytest tests/ -q          # unit tests (54)
python3 scripts/verify.py            # end-to-end harmonization smoke test
python3 scripts/02_confound_ladder.py   # reproduce the §3.1 confound ladder
```

```python
from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset
df = build_unified_dataframe("data", "face-common-vars.xlsx",
                             readiness=["READY", "PARTIAL"], format="long")
ds = to_harmonized_dataset(df, load_variables("face-common-vars.xlsx"), visit="V0")
# ds.X: MultiIndex[cohort, patient_id] × numeric features (NaN = missing, never imputed here);
# ds.schema: a trans_diag.engine.FeatureSchema generated from our dictionary.
```

## Pipeline order (`scripts/00_run_all.py`)

Scripts are numbered in execution order — read or run them top-to-bottom:
`01_manuscript_table1` (Table 1) → `02_confound_ladder` (§3.1 confound trap) →
`03_cluster_domains` (residualized scores + embedding) → `04_structure_test`
(discrete-vs-dimensional) → `05_dimensional_axes` (varimax FA) → `06_dimensional_ae`
(autoencoder cross-check) → `07_dimensional_refine` (locked K=6 axes) →
`08_longitudinal_axes` → `09_longitudinal_coherence` → `10_phase5_outcomes` (V1 then V2)
→ `11_phase5_ci` → `12_phase5_decircularized` → `13_robustness_site` (ComBat) →
`14_cognition_bpsz` → `15_review_checks` → `16_manuscript_figures` →
`17_export_longitudinal_figure` (Suppl. Fig S1) → `18_export_dimensional_flow` (Fig 6).
`00_run_all.py` runs all 18 in this order; the four unnumbered scripts (`verify`,
`audit`, `qa_missingness`, `build_notebook`) are utilities, not pipeline steps.

## Conventions

- **Python ≥ 3.11.** Core deps in `pyproject.toml`; full reproduction needs `".[full]"`.
- **Develop in `src/trans_diag`** (incl. `engine/` — vendored but now ours to maintain).
- **Output paths**: scripts write to `results/` (data) or `reports/` (HTML/figures).
- **Determinism**: fixed seeds throughout; reproduces to ≤1e-12 (BLAS round-off only).
  All CV folds are shuffled (the patient matrix is cohort-ordered).

## Status (manuscript-complete)

- **Trans-diagnostic structure is DIMENSIONAL, not discrete** (structure test: no eigengap,
  monotone gap, HDBSCAN≈cohort ARI 0.70; the only discrete structure is diagnosis). The 7
  DSM subtypes order on a mood↔psychosis continuum (ρ 0.79 [0.75,0.86]).
- **Final model: K=6 confound-free axes** (`07_dimensional_refine.py`; FA + PyTorch AE agree,
  CCA 0.93 vs permutation null 0.06). Diagnosis-independent (cohort η²≤0.10, site ≤0.05).
- **Outcomes (shuffled CV + repeated-CV CIs):** axes beat DSM on QoL (+0.036 [+0.033,+0.039]),
  complement functioning (combined +0.029), DSM dominates hospitalization. Robust to
  de-circularization, ComBat, and V2 (same cohort).
- **Discrete clustering = negative result** (~38% persistence, DSM-ARI 0.006) — slices of a
  continuum; supplement only.
- **Repo independence:** engine internalized (`engine/`); full pipeline reproduces the
  manuscript to ≤1e-12; 54 tests pass with no `archive/` dependency.

## Where to read next

- **The paper** → [MANUSCRIPT.md](MANUSCRIPT.md)
- **Dictionary columns** → [DATA.md](DATA.md) · **Plan/framing** → [ROADMAP.md](ROADMAP.md)
- **Findings (paper log)** → [FINDINGS.md](FINDINGS.md) · **Lab notebook** → [LABBOOK.md](LABBOOK.md)
- **Engine internals** → `src/trans_diag/engine/` (module docstrings)
