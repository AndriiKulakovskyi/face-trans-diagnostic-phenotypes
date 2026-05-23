# FACE — Trans-diagnostic Dimensional Phenotyping (BP · SZ · DR)

Across bipolar disorder, schizophrenia and major depression in the FACE cohort,
the categorical boundary the data support is **diagnosis itself** — every
*trans-diagnostic* axis of variation is **continuous**. We harmonize the 3-cohort
longitudinal data (baseline V0 → 4-year V4) and fit a confound-controlled
**six-dimension** model that complements/outperforms DSM diagnosis for
patient-reported outcomes. Full write-up: **[MANUSCRIPT.md](MANUSCRIPT.md)**.

The repo is **self-contained** — the stratification engine (masked similarity →
multipartite-spectral embedding, enrichment) is internalized in
`src/face_common/engine/`; there is no external dependency on the sister
`face_stratification`/`face_rlvr` projects.

## Layout

- `src/face_common/` — the package: harmonization (`variable`/`rules`/`loader`/`filters`),
  matrix build + domain aggregation (`schema_gen`/`adapter`/`domains`), and the
  internalized `engine/`.
- `scripts/` — the pipeline (`run_all.py` orchestrates 18 steps) + infra
  (`verify`, `audit`, `qa_missingness`) + `confound_ladder.py` (reproduces §3.1).
- `tests/` — unit tests. `results/`, `reports/` — reproducible artifacts + HTML + figures.
- `face-common-vars.xlsx` — the common-variables dictionary (input).
- `data/` — the 3-cohort longitudinal CSVs (**confidential**; see note below).

## Quick start

```bash
pip install -e ".[full]"           # core + torch (AE) + neuroHarmonize (ComBat) + kaleido
python3 scripts/run_all.py         # reproduce the whole manuscript pipeline (~5 min)
python3 -m pytest tests/ -q        # 54 unit tests
python3 scripts/verify.py          # end-to-end harmonization smoke test
```

```python
from face_common import build_unified_dataframe, load_variables, to_harmonized_dataset
df = build_unified_dataframe("data", "face-common-vars.xlsx",
                             readiness=["READY", "PARTIAL"], format="long")
ds = to_harmonized_dataset(df, load_variables("face-common-vars.xlsx"), visit="V0")
```

`face_common` imports from `src/` (scripts add it to `sys.path`; or `pip install -e .`).
The pipeline is deterministic (fixed seeds) and reproduces the manuscript to ≤1e-12.

## Documentation

- **[MANUSCRIPT.md](MANUSCRIPT.md)** — the paper (methods, results, figures, references).
- **[CLAUDE.md](CLAUDE.md)** — concise project + repo guide.
- **[ROADMAP.md](ROADMAP.md)** — research question, methods, phased plan, course-corrections.
- **[FINDINGS.md](FINDINGS.md)** — running research log (paper-oriented).
- **[LABBOOK.md](LABBOOK.md)** — chronological lab notebook (full traceability).
- **[DATA.md](DATA.md)** — how to read the common-variables dictionary.
- Engine internals → `src/face_common/engine/` module docstrings.

> ⚠️ Internal research use only. The FACE database contains **confidential**
> clinical data (Fondation FondaMental). The `data/*.csv` files are currently
> tracked in git — remove them (gitignore + history scrub) before any external share.
