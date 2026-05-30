# AGENTS.md — FACE trans-diagnostic dimensional phenotyping (BP · SZ · DR)

> Guide for collaborators and AI assistants. Keep it short. Paper draft:
> [MANUSCRIPT.md](MANUSCRIPT.md). Plan: [docs/ROADMAP.md](docs/ROADMAP.md). Dictionary guide:
> [docs/DATA.md](docs/DATA.md). Findings log: [docs/FINDINGS.md](docs/FINDINGS.md) · [docs/LABBOOK.md](docs/LABBOOK.md).

## What this is

A **self-contained** project that harmonizes the 3-cohort FACE psychiatric data
(Bipolar, Schizophrenia, Depression; baseline V0 → 4-year V4) and models
**trans-diagnostic** structure across DSM-5. Headline result: trans-diagnostic
variation is **dimensional, not categorical** — six reproducible, confound-controlled
trans-diagnostic dimensions (five symptom/biology axes plus one cognitive axis, after the DR
neuropsychology extraction gap was closed) that complement/outperform DSM diagnosis for
patient-reported outcomes (full write-up in `MANUSCRIPT.md`).

The stratification **engine** (masked similarity → multipartite-spectral embedding,
cluster enrichment, factor scaffolding) was originally a sister project; the pieces we
use are now **internalized** in `src/trans_diag/engine/` — the repo has **no external
dependency on `face_stratification`/`face_rlvr`**.

## Repository layout

```
face-common-bp-sz-dr/
├── MANUSCRIPT.md  AGENTS.md  README.md   ← paper + guides (kept at root)
├── data/                            ← inputs (read-only)
│   ├── face-common-vars.xlsx        ← common-variables dictionary (tracked)
│   ├── thesaurus/                   ← per-cohort source dictionaries (tracked, reference)
│   └── {bipolar,schizophrenia,depression}.csv  ← longitudinal data (confidential; gitignored)
├── src/trans_diag/                 ← the package (all our code)
│   ├── variable.py  rules.py  loader.py  filters.py   ← harmonization
│   ├── schema_gen.py  adapter.py  domains.py          ← matrix build + domain aggregation
│   ├── masked_fa.py  axes.py  outcomes.py             ← imputation-free FA, axis names, outcome models
│   └── engine/                      ← internalized stratification engine
│       ├── feature_schema.py  harmonized_dataset.py   ← data contracts
│       ├── masked_similarity.py  spectral_base.py  multipartite.py  ← embedding (no imputation)
│       ├── enrichment.py  clustering.py               ← enrichment + kmeans/bootstrap
├── scripts/                         ← pipeline (00_run_all.py orchestrates 01–22) + verify/audit/qa infra
├── tests/                           ← unit + golden-number regression tests (76)
├── results/                         ← reproducible AGGREGATE artifacts (CSV/JSON; tracked)
│   └── reports/                     ← rendered HTML + figures/ (PNG/SVG; tracked)
├── notebooks/                       ← FACE_reproduction.ipynb
├── docs/                            ← ROADMAP · DATA · FINDINGS · LABBOOK
└── pyproject.toml                   ← packages = src/trans_diag; deps; [full] extras
```

**Imports.** `trans_diag` resolves from `src/`. Scripts insert `src/` on `sys.path`;
pytest uses `pythonpath = ["src"]`. Or `pip install -e ".[full]"`.

## Data inputs (read-only, confidential)

- `data/face-common-vars.xlsx` — dictionary: one row per harmonized variable (per-cohort
  source columns, `dtype`, value set, `section`, `cluster_readiness`, rule). Small (103 KB);
  tracked and safe to share. `data/thesaurus/` holds the per-cohort source dictionaries (reference).
- `data/{bipolar,schizophrenia,depression}.csv` — 6,252 / 2,209 / 552 patients,
  visit-level rows (V0–V4). **Confidential.** ✅ gitignored and **never committed**
  (`git log --all --full-history -- 'data/*.csv'` is empty) — a local working copy only,
  safe to share the repo as-is.
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
**No imputation** anywhere in the final model: the masked-similarity embedding, the
autoencoder objective, AND the factor analysis all operate on observed cells only
(`masked_fa.py`: pairwise-complete correlation → masked posterior-mean scores; the residual
matrix is ~65% observed — see MANUSCRIPT §2.1, §3.8). The superseded FA mean-fill survives
only as an ablation (`scripts/sensitivity_masked_fa*.py`).

## Quick start

```bash
pip install -e ".[full]"             # core + torch (AE) + neuroHarmonize (ComBat) + kaleido (figures)
python3 scripts/00_run_all.py           # reproduce the whole manuscript pipeline (~5 min, steps 01–22)
python3 -m pytest tests/ -q          # unit + golden-number tests (76)
python3 scripts/verify.py            # end-to-end harmonization smoke test
python3 scripts/02_confound_ladder.py   # reproduce the §3.1 confound ladder
```

```python
from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset
df = build_unified_dataframe("data", "data/face-common-vars.xlsx",
                             readiness=["READY", "PARTIAL"], format="long")
ds = to_harmonized_dataset(df, load_variables("data/face-common-vars.xlsx"), visit="V0")
# ds.X: MultiIndex[cohort, patient_id] × numeric features (NaN = missing, never imputed here);
# ds.schema: a trans_diag.engine.FeatureSchema generated from our dictionary.
```

## Pipeline order (`scripts/00_run_all.py`)

Scripts are numbered in execution order — read or run them top-to-bottom:
`01_manuscript_table1` (Table 1) → `02_confound_ladder` (§3.1 confound trap) →
`03_cluster_domains` (residualized scores + embedding) → `04_structure_test`
(discrete-vs-dimensional) → `05_dimensional_axes` (varimax FA) → `06_dimensional_ae`
(autoencoder cross-check) → `07_dimensional_refine` (locked K=6 axes, incl. cognition) →
`08_longitudinal_axes` → `09_longitudinal_coherence` → `10_phase5_outcomes` (V1 then V2)
→ `11_phase5_ci` → `12_phase5_decircularized` → `13_robustness_site` (ComBat) →
`15_review_checks` (incl. the cognition confound battery #10) → `16_manuscript_figures` →
`17_export_longitudinal_figure` (Suppl. Fig S1) → `18_export_dimensional_flow` (Fig 6) →
`19_pfactor` (§4.6 general-factor 'p' check) → `20_robustness_cvrefit` (Limitation 10:
axes re-fit inside CV folds) → `21_replication_holdout` (Limitation 9: within-FACE
leave-one-cohort / leave-one-site replication) → `22_screening_panel` (§4.5 parsimonious
screening panel: sparse item→axis distillation).
`00_run_all.py` runs all 21 in this order (step 14, the old BP/SZ-only cognition analysis,
was removed when neuropsychology was folded into the main model); the unnumbered scripts
(`verify`, `audit`, `qa_missingness`, `build_notebook`, `build_dr_neuropsych_mapping`,
`sensitivity_masked_fa{,_mechanism}`) are utilities, not pipeline steps.

## Conventions

- **Python ≥ 3.11.** Core deps in `pyproject.toml`; full reproduction needs `".[full]"`.
- **Develop in `src/trans_diag`** (incl. `engine/` — vendored but now ours to maintain).
- **Output paths**: scripts write aggregates to `results/` and HTML/figures to `results/reports/`.
- **Determinism**: fixed seeds throughout; reproduces to ≤1e-12 (BLAS round-off only).
  All CV folds are shuffled (the patient matrix is cohort-ordered).

## Status (manuscript draft; imputation-free model)

- **Trans-diagnostic structure is DIMENSIONAL, not discrete** (structure test: no eigengap,
  monotone gap, HDBSCAN≈cohort ARI 0.64; the only discrete structure is diagnosis). The 7
  DSM subtypes order on a mood↔psychosis continuum (ρ 0.50 [0.36,0.61]; down from ρ 0.79 in the
  symptom-only embedding — cognition now shares the leading embedding dimension).
- **Final model: K=6 imputation-free confound-free axes** (`07_dimensional_refine.py`: masked
  pairwise-complete correlation → PAF+varimax → masked posterior-mean scores, NO cell filled;
  K=6 = max reproducible dimensionality before collapse by deterministic single-split half
  congruence, K≥7 collapse; a 25-split robustness curve is reported as a caveat; FA + PyTorch AE
  agree, leading CCA 0.94 vs perm-null 0.05). Diagnosis-independent (cohort η²≤0.106, site ≤0.049).
  Axes: depression, later-onset, mania/activation (+externalizing: impulsivity/childhood-ADHD),
  illness-burden, **cognition (verbal reasoning + working memory)**, metabolic.
- **Cognition integrated (DR neuropsych extraction gap closed, 2026-05).** The old "neuropsychology
  absent in DR by design (0% vs BP 71% / SZ 86%)" claim was an extraction artifact; a full DR export
  recovered it (V0 coverage ~57%, vs BP ~68% / SZ ~80%). DR neuropsych columns were mapped into the
  dictionary (`scripts/build_dr_neuropsych_mapping.py`) and curated cognitive constructs now feed the
  main masked FA. A confound battery (`15_review_checks` #10 + `21`) admitted ONE genuine cognitive
  axis (verbal reasoning + working memory; cohort η² 0.072, transports leave-DR-out 1.0) and rejected
  the rest: processing speed/executive (incoherent across cohorts, ~0 communality), verbal fluency
  (cohort artifact — η² 0.46, survives within-cohort permutation), CVLT memory + matrix reasoning
  (BP/SZ-only). The standalone BP/SZ cognition analysis (old `14_cognition_bpsz.py`) was removed; vs
  the prior symptom-only K=7, the pure-mania and externalizing axes re-merge and work-disability is
  no longer separately resolved.
- **Outcomes (shuffled CV + repeated-CV CIs) — UNCHANGED by the cognition integration:** axes beat
  DSM on QoL (+0.037 [+0.034,+0.040]), complement functioning (combined +0.035), DSM dominates
  hospitalization (−0.141). Robust to de-circularization, ComBat, fold-honest re-fit, and V2.
  Trait-state: metabolic (0.63) & depression (0.58) most trait-like; cognition moderately
  trait-like (0.31).
- **Discrete clustering = negative result** — slices of a continuum; supplement only.
- **Repo independence:** engine internalized (`engine/`); full pipeline reproduces to ≤1e-12;
  76 tests pass with no `archive/` dependency.

## Where to read next

- **The paper** → [MANUSCRIPT.md](MANUSCRIPT.md)
- **Dictionary columns** → [docs/DATA.md](docs/DATA.md) · **Plan/framing** → [docs/ROADMAP.md](docs/ROADMAP.md)
- **Findings (paper log)** → [docs/FINDINGS.md](docs/FINDINGS.md) · **Lab notebook** → [docs/LABBOOK.md](docs/LABBOOK.md)
- **Engine internals** → `src/trans_diag/engine/` (module docstrings)
