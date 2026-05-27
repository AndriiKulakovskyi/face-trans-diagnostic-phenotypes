# FACE — Trans-diagnostic Dimensional Phenotyping (BP · SZ · DR)

Across bipolar disorder, schizophrenia and major depression in the FACE cohort,
the only categorical boundary the data support is **diagnosis itself** — every
*trans-diagnostic* axis of variation is **continuous**. We harmonize the 3-cohort
longitudinal data (baseline V0 → 4-year V4) and fit a confound-controlled,
imputation-free **seven-dimension** model that complements or outperforms DSM
diagnosis for patient-reported outcomes, and distill it into a parsimonious
(≤15-item) screening panel. Full write-up: **[MANUSCRIPT.md](MANUSCRIPT.md)**.

The repo is **self-contained** — the stratification engine (masked similarity →
multipartite-spectral embedding, enrichment, factor scaffolding) is internalized in
`src/trans_diag/engine/`; there is no external dependency on the sister
`face_stratification` / `face_rlvr` projects.

## Repository structure

```
face-common-bp-sz-dr/
├── MANUSCRIPT.md             ← the paper (methods, results, figures, references)
├── README.md  CLAUDE.md      ← this file + the concise project/AI-assistant guide
├── install.py                ← one-step setup: creates .venv and installs .[full]
├── pyproject.toml            ← packages = src/trans_diag; deps; ".[full]" extras
│
├── data/                     ← inputs (read-only)
│   ├── face-common-vars.xlsx     • common-variables dictionary (tracked, 103 KB)
│   ├── thesaurus/                • per-cohort source dictionaries (tracked, reference)
│   └── {bipolar,schizophrenia,depression}.csv
│                                 • 3-cohort longitudinal data — CONFIDENTIAL, gitignored
│
├── src/trans_diag/           ← the package (all our code)
│   ├── variable·rules·loader·filters.py      • harmonization
│   ├── schema_gen·adapter·domains.py         • matrix build + domain aggregation
│   ├── masked_fa·axes·outcomes.py            • imputation-free FA, axis names, outcome models
│   └── engine/                               • internalized stratification engine
│       ├── feature_schema·harmonized_dataset.py   – data contracts
│       ├── masked_similarity·spectral_base·multipartite.py  – embedding (no imputation)
│       └── enrichment·clustering.py               – enrichment + kmeans/bootstrap
│
├── scripts/                  ← the pipeline; 00_run_all.py orchestrates steps 01–22
│   └── verify·audit·qa_missingness·build_notebook.py   • utilities (not pipeline steps)
│
├── results/                  ← reproducible AGGREGATE artifacts (CSV/JSON; tracked)
│   └── reports/              ← rendered HTML + figures/ (PNG/SVG; tracked)
│
├── tests/                    ← unit + golden-number regression tests (76)
├── notebooks/                ← FACE_reproduction.ipynb (narrative walk-through)
└── docs/                     ← ROADMAP · DATA · FINDINGS · LABBOOK
```

**Imports.** `trans_diag` resolves from `src/`. Scripts insert `src/` on `sys.path`;
pytest uses `pythonpath = ["src"]`. Or install editable: `pip install -e ".[full]"`.

## Quick start

```bash
python3 install.py                   # create .venv and install .[full] + dev tools
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python3 scripts/00_run_all.py        # reproduce the whole manuscript pipeline (~5 min, steps 01–22)
```

`install.py` requires Python ≥ 3.11 and creates an isolated `.venv` in the project root.
Pass `--no-dev` to skip pytest/ruff, `--check` to verify every import after install,
or `--force` to wipe the existing `.venv` and reinstall from scratch.

<details>
<summary>Manual install (no .venv)</summary>

```bash
pip install -e ".[full]"             # core + torch + neuroHarmonize (ComBat) + kaleido + ipython + nbformat
```
</details>

```python
from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset

df = build_unified_dataframe("data", "data/face-common-vars.xlsx",
                             readiness=["READY", "PARTIAL"], format="long")
ds = to_harmonized_dataset(df, load_variables("data/face-common-vars.xlsx"), visit="V0")
# ds.X: MultiIndex[cohort, patient_id] × numeric features (NaN = missing, never imputed here)
```

The pipeline is numbered in execution order — read or run the scripts top-to-bottom.
`00_run_all.py` runs steps **01–22**: Table 1 → confound ladder (§3.1) → residualized
domain scores + embedding → discrete-vs-dimensional structure test → varimax FA →
autoencoder cross-check → **locked K=7 axes** → longitudinal stability → outcome
head-to-heads + CIs + de-circularization → ComBat/site robustness → cognition → review
checks → manuscript figures → general-factor ('p') check → fold-honest CV re-fit →
within-FACE held-out replication → **parsimonious screening panel**.

## Verify / reproduce

```bash
python3 -m pytest tests/ -q          # 76 tests: unit (filters, adapter, domains, masked-FA, axes)
                                     #           + golden-numbers regression against results/
python3 scripts/verify.py            # end-to-end harmonization smoke test
python3 scripts/00_run_all.py        # full reproduction → writes results/ + results/reports/
python3 install.py --check           # re-verify every import in the .venv
```

- **Golden numbers** (`tests/test_golden_numbers.py`) pin every headline value in the
  manuscript (with its §/Table) to the committed aggregate in `results/`. A pipeline re-run
  that changes a result changes the artifact and fails the matching assertion — forcing a
  synchronized update of both test and manuscript. On a fresh clone without the confidential
  cohort, these tests **skip** (they read aggregates, never patient data).
- **Determinism.** Fixed seeds throughout; reproduces to ≤1e-12 (BLAS round-off only). All
  CV folds are shuffled (the patient matrix is cohort-ordered). Re-running `00_run_all.py`
  with the source data present yields byte-identical `results/` aggregates; only the rendered
  `results/reports/*.{html,svg}` carry cosmetic render-metadata diffs.

## Documentation

- **[MANUSCRIPT.md](MANUSCRIPT.md)** — the paper (methods, results, figures, references).
- **[CLAUDE.md](CLAUDE.md)** — concise project + repo guide (for collaborators and AI assistants).
- **[docs/ROADMAP.md](docs/ROADMAP.md)** — research question, methods, phased plan, course-corrections.
- **[docs/FINDINGS.md](docs/FINDINGS.md)** — running research log (paper-oriented).
- **[docs/LABBOOK.md](docs/LABBOOK.md)** — chronological lab notebook (full traceability).
- **[docs/DATA.md](docs/DATA.md)** — how to read the common-variables dictionary.
- Engine internals → `src/trans_diag/engine/` module docstrings.

## Confidentiality & what is shared

> ⚠️ The FACE database contains **confidential** clinical data (Fondation FondaMental).
> The per-cohort `data/*.csv` files and all per-patient derived artifacts
> (`results/*_scores.parquet`, embeddings, cluster/longitudinal assignments) are
> **gitignored and have never been committed** (`git log --all --full-history -- 'data/*.csv'`
> is empty) — they stay on disk as a local working copy only.
>
> What **is** tracked and safe to share: the code (`src/`, `scripts/`, `tests/`), the small
> input dictionaries (`data/face-common-vars.xlsx`, `data/thesaurus/`), the **aggregate**
> results (`results/`: 26 CSV + 16 JSON, no per-patient rows), and the rendered reports and
> figures (`results/reports/`). Reviewers can read every result and reproduce all figures
> without the confidential cohort; full numerical reproduction requires the `data/*.csv` files.
