# FACE — Trans-diagnostic Clustering (BP · SZ · DR)

Data-driven discovery of clinical phenotypes that cut **across** DSM-5 diagnostic
boundaries in the FACE psychiatric cohort, validated for **temporal coherence**
across annual follow-up visits.

## The two halves

| | what | where |
|---|---|---|
| **Our pipeline** | harmonizes the 3-cohort **longitudinal** data (V0→V4) from a common-variables dictionary into a patient × feature matrix — the **feature source** | `src/face_common/` |
| **Vendored engine** | a sister project's modelling algorithms (masked similarity → multipartite-spectral embedding → consensus clustering → validation), **reused, not developed** | `archive/face_stratification/` |

We feed our matrix into the engine; the engine's original 4-cohort clusters are
a comparison **reference**. No imputation anywhere (masked pairwise-complete
similarity).

## Layout

- `src/face_common/` — the only code we develop (harmonization + filters).
- `archive/` — copied sister code, vendored (engine, extractors, their scripts/
  notebooks/data/tests). Import it; don't edit it.
- `config/` — engine config (feature schema + clinical glossary), kept at root.
- `data/` — our 3-cohort longitudinal CSVs (`bipolar/schizophrenia/depression.csv`)
  + `data/external/` (engine reference artifacts).
- `scripts/` — our runnable scripts. `tests/` — our tests. `results/`, `reports/` — outputs.
- `face-common-vars.xlsx` — the common-variables dictionary (input).

## Quick start

```bash
python3 -m pytest tests/ -q        # unit tests
python3 scripts/verify.py          # end-to-end smoke test (~30 s)
python3 scripts/qa_missingness.py  # interactive HTML missingness report
```

```python
from face_common import build_unified_dataframe
df = build_unified_dataframe("data", "face-common-vars.xlsx",
                             readiness=["READY", "PARTIAL"], format="long")
```

`face_common` imports from `src/`; the engine (`face_stratification`,
`face_rlvr`) from `archive/`. pytest is configured for both; or
`pip install -e ".[stratification]"`.

## Documentation

- **[CLAUDE.md](CLAUDE.md)** — concise project + repo guide.
- **[ROADMAP.md](ROADMAP.md)** — research question, methods, paper framing,
  phased plan, course-correction log.
- **[DATA.md](DATA.md)** — how to read the common-variables dictionary.
- **`archive/docs/`** — the sister engine's stage documentation.

> Internal research use only. The FACE database contains confidential clinical
> data (Fondation FondaMental).
