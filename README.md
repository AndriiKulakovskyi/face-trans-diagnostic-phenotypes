# FACE — Trans-diagnostic Phenotyping (BP · SZ · DR) — v2 study

Across bipolar disorder, schizophrenia and major depression in the FACE cohort, we ask whether
trans-diagnostic variation is **dimensional** (latent symptom / biology / cognition dimensions)
and/or **categorical** (patient strata). We harmonize the 3-cohort longitudinal data (baseline V0
→ 4-year V4) with **no imputation** (masked methods) and re-derive every result from zero on a
re-curated ("v2") common-variables dictionary.

> **Status — v2 restart.** The dictionary is finalized (**214 usable variables**) and the
> preprocessing is debugged + ML-ready (**type-aware scaling to [−1, 1]**, no imputation); the
> dimensional analysis and patient stratification have **not** yet been re-run on v2. The prior
> v1 study is archived at git tag `v1-archive-2026-05-30`. Project guide: **[CLAUDE.md](CLAUDE.md)**.

The repo is **self-contained** — the stratification engine (masked similarity →
multipartite-spectral embedding, enrichment) is internalized in `src/trans_diag/engine/`; there is
no external dependency on the sister `face_stratification` / `face_rlvr` projects.

## Data processing — three stages
1. **Harmonized variables** (native scale) — per-cohort source → harmonization rules + sanity
   bounds (out-of-range → NaN, never imputed).
2. **Type-aware scaling to [−1, 1]** — binary/ordinal → min-max; continuous → log-if-skewed +
   winsorize + robust-z clipped ±5.
3. **Aggregated V0 domain scores** — items → construct-level scores (masked mean of robust-z; no
   imputation) — the ~69 features that actually enter the models.

The QA report (`scripts/qa_harmonization.py` → `results/reports/qa_harmonization.html`) shows all
three, per variable. **Why aggregate to domain scores?** so each construct counts once (no
item-count weighting bias), to raise coverage without imputing, and to keep dimensions/clusters
interpretable over named constructs. Full rationale in [CLAUDE.md](CLAUDE.md).

## Repository structure
```
├── MANUSCRIPT.md  CLAUDE.md  AGENTS.md  README.md   ← paper skeleton + guides
├── data/            face-common-vars.xlsx (v2 dict) · thesaurus/ · *.csv (confidential) · site_lookup.csv
├── src/trans_diag/  variable·rules·loader·filters · schema_gen·adapter·domains · masked_fa·axes·outcomes · engine/
├── scripts/         method pipeline 01–22 (00_run_all) + qa_harmonization · verify · audit
├── tests/           unit + golden-number tests (golden skip until v2 results exist)
├── results/         regenerated aggregates + reports/qa_harmonization.html (empty on a clean tree)
└── docs/            ROADMAP · DATA · FINDINGS · LABBOOK · neuropsy_features.yaml
```

## Quick start
```bash
pip install -e ".[full]"                 # core + torch + neuroHarmonize + kaleido
python3 scripts/qa_harmonization.py      # the 3-part data-processing QA report (clean this first)
python3 -m pytest tests/ -q              # unit tests (golden tests skip until v2 results exist)
```
```python
from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset
df = build_unified_dataframe("data", "data/face-common-vars.xlsx",
                             readiness=["READY", "PARTIAL"], format="long")
ds = to_harmonized_dataset(df, load_variables("data/face-common-vars.xlsx"), visit="V0")
# ds.X: MultiIndex[cohort, patient_id] × numeric features (NaN = missing, never imputed)
```

## Documentation
- **[CLAUDE.md](CLAUDE.md)** — project guide + the data-processing pipeline (the central read).
- **[docs/ROADMAP.md](docs/ROADMAP.md)** · **[docs/FINDINGS.md](docs/FINDINGS.md)** ·
  **[docs/LABBOOK.md](docs/LABBOOK.md)** · **[docs/DATA.md](docs/DATA.md)** ·
  **[docs/neuropsy_features.yaml](docs/neuropsy_features.yaml)** (cognition include-list).

## Confidentiality
The FACE database is **confidential** (Fondation FondaMental). The per-cohort `data/*.csv` and all
per-patient artifacts are **gitignored and never committed**. Tracked + shareable: the code
(`src/`, `scripts/`, `tests/`), the small input dictionaries (`data/face-common-vars.xlsx`,
`data/thesaurus/`, `data/site_lookup.csv`), and regenerated **aggregate** results.
