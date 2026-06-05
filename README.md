# FACE — Precision Psychiatry (BP · SZ · DR) — **V3**

Across bipolar disorder, schizophrenia and major depression in the FACE cohorts, V3 turns the
harmonized 3-cohort data into a **precision-psychiatry stratification and decision-modeling
framework**, in four layers:

```text
diagnostic cohorts (BP · SZ · DR)          → transdiagnostic dimension discovery
  → validated patient strata               → prognosis / treatment decision models
```

Dimensions are discovered with a **patient-level, missingness-aware latent model** (the primary engine
is a **Bayesian sparse bifactor / ESEM-like** model with **mixed likelihoods** and **soft loading
priors**; **FIML/SEM** is the confirmatory benchmark; the **V2 masked-correlation** factors are the
reproducibility baseline). Strata are **probabilistic decision regions**, not natural subtypes. **No
naive imputation anywhere** — structure is estimated from each patient's observed cells via
observed-data likelihood, never by filling cells.

> **Status.** The **V3 plan is the single source of truth** — direction, framing, and the estimator
> hierarchy are fixed by **[docs/V3_PLAN.md](docs/V3_PLAN.md)**. The completed **V2** dimensional study
> is a **benchmark / reference arm only** ([docs/legacy_v2/](docs/legacy_v2/README.md)); the V3
> discovery engine (Phases E–M) is **not yet built**. The runnable code in `src/` + `scripts/01–15` is
> the V2 benchmark implementation. Project guide: **[CLAUDE.md](CLAUDE.md)**.

The repo is **self-contained** — the (V2) stratification engine (masked similarity →
multipartite-spectral embedding, enrichment) is internalized in `src/trans_diag/engine/`; there is no
external dependency on the sister `face_stratification` / `face_rlvr` projects.

## The 10 candidate dimensions are a *soft starting ontology*

Impulsivity · Cognitive flexibility · Negative symptoms · Anhedonia · Metabolism/immunometabolism ·
Sleep/circadian · Overall clinical severity · Sensory abnormalities · Neurodevelopment · Suicidality.

They seed **soft priors**, not hand-tagged scores — the data may **confirm, split, merge, reject,
downgrade, or cross-load** any of them through the missingness-aware hybrid latent model, with explicit
validation. Diagnosis (BP/SZ/DR) is an entry + **validation/covariate** label, **never** a clustering
feature.

## Data foundation (no-imputation, load-bearing for V3)

Each dictionary variable is read from its per-cohort source column → harmonization rule + per-variable
**sanity bounds** (out-of-range → NaN, never imputed) → native clinical scale, with deterministic
**skip-logic** structural-zero decoding. V3 extends the dictionary into a **data contract** (per-variable
likelihood family, missingness type, soft prior loading, covariate/outcome status, modeling role — see
[docs/DATA.md](docs/DATA.md)). The QA report (`scripts/qa_harmonization.py` →
`results/reports/qa_harmonization.html`) checks the harmonization; its `[−1,1]` scaling stage is the
benchmark-arm encoding (V3 lets the observation likelihood carry each variable's type).

## Repository structure
```
├── CLAUDE.md  AGENTS.md  README.md             ← guides (V3); plan of record: docs/V3_PLAN.md
├── data/            face-common-vars.xlsx (dictionary) · thesaurus/ · *.csv (confidential) · site_lookup.csv
├── src/trans_diag/  variable·rules·loader·filters · schema_gen·adapter·domains · masked_fa·axes·phenotype · engine/  (V2 benchmark; add V3 sub-packages)
├── scripts/         V2 benchmark pipeline 01–15 + qa_harmonization · verify · audit
├── tests/           unit + V2 golden-number tests (pinned to results/hfa/; skip on a clean clone)
├── results/         regenerated aggregates: hfa/ · manuscript/ (V2 paper) · reports/ (gitignored; empty on a clean tree)
└── docs/            V3_PLAN · ROADMAP · PIPELINE · DATA · FINDINGS · neuropsy_features.yaml · legacy_v2/ (the V2 arm)
```

## Quick start
```bash
pip install -e ".[full]"                 # core + kaleido (static figure export)
python3 scripts/qa_harmonization.py      # harmonization QA report (load-bearing for V3; [-1,1] stage is benchmark-only)
python3 scripts/00_run_all.py            # regenerate the V2-benchmark results/hfa/ artifacts (needs the cohort CSVs)
python3 -m pytest tests/ -q              # unit + V2 golden tests (golden tests skip on a clean clone)
```
```python
from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset
df = build_unified_dataframe("data", "data/face-common-vars.xlsx",
                             readiness=["READY", "PARTIAL"], format="long")
ds = to_harmonized_dataset(df, load_variables("data/face-common-vars.xlsx"), visit="V0")
# ds.X: MultiIndex[cohort, patient_id] × numeric features (NaN = missing, never imputed)
```

## Documentation
- **[CLAUDE.md](CLAUDE.md)** — project guide (the central read; includes instructions for future agents).
- **[docs/V3_PLAN.md](docs/V3_PLAN.md)** — the V3 plan of record (verbatim source: [docs/V3_PLAN_SOURCE.md](docs/V3_PLAN_SOURCE.md)).
- **[docs/ROADMAP.md](docs/ROADMAP.md)** (what/why) · **[docs/PIPELINE.md](docs/PIPELINE.md)** (target architecture + missing-data doctrine).
- **[docs/DATA.md](docs/DATA.md)** (data contract + dictionary) · **[docs/FINDINGS.md](docs/FINDINGS.md)** (V3 log; V2 findings as hypotheses).
- **[docs/legacy_v2/](docs/legacy_v2/README.md)** — the V2 benchmark/reference arm · **V2 manuscript** — [results/manuscript/manuscript.md](results/manuscript/manuscript.md).

## Confidentiality
The FACE database is **confidential** (Fondation FondaMental). The per-cohort `data/*.csv` and all
per-patient artifacts are **gitignored and never committed**. Tracked + shareable: the code
(`src/`, `scripts/`, `tests/`), the small input dictionaries (`data/face-common-vars.xlsx`,
`data/thesaurus/`, `data/site_lookup.csv`), and regenerated **aggregate** results.
