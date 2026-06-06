# FACE — Precision Psychiatry (BP · SZ · DR) — **V3**

Across bipolar disorder, schizophrenia and major depression in the FACE cohorts, V3 turns the
harmonized 3-cohort data into a **precision-psychiatry stratification and decision-modeling
framework**, in four layers:

```text
diagnostic cohorts (BP · SZ · DR)          → transdiagnostic dimension discovery
  → validated patient strata               → prognosis / treatment decision models
```

Dimensions are discovered with a **patient-level, missingness-aware latent model** (the primary engine
is a **marginalized Bayesian sparse bifactor / ESEM-like** model with **mixed likelihoods** and **soft
loading priors**; **FIML/SEM** is the confirmatory follow-up). Strata are **probabilistic decision
regions**, not natural subtypes. **No naive imputation anywhere** — structure is estimated from each
patient's observed cells via observed-data likelihood, never by filling cells.

> **Status.** Direction is fixed by **[docs/V3_PLAN.md](docs/V3_PLAN.md)**; **current state** lives in
> **[docs/STATE.md](docs/STATE.md)** — read it first. The config-first measurement engine
> (`src/v3/latent_models/bayesian/`, run via `scripts/v3/03–04`) is **converged through Stage 2**: a
> general factor `G` (functional impairment / distress) identifies, **orthogonal to
> metabolic/inflammatory biology**. The downstream strata / prognosis / treatment layers are **not yet
> built**. Project guide: **[CLAUDE.md](CLAUDE.md)**.

The repo is **self-contained** — the data layer (`src/v3/data/`) and the full V3 pipeline live in-tree;
there is no external dependency on the sister `face_stratification` / `face_rlvr` projects.

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
[docs/DATA.md](docs/DATA.md)). The data layer lives in `src/v3/data/`, and the V3 observation likelihood
carries each variable's native type (no shared rescaling).

## Repository structure
```
├── CLAUDE.md  AGENTS.md  README.md             ← guides (V3); plan of record: docs/V3_PLAN.md
├── data/            face-common-vars.xlsx (dictionary) · thesaurus/ · *.csv (confidential) · site_lookup.csv
├── src/v3/          data/ (harmonization, no-imputation) · latent_models/bayesian/ (the engine) · priors/ (prior-matrix builder)
├── configs/         ontology + contract: dimensions · priors · likelihoods · bayesian_model · prior_loading_matrix_v3
├── scripts/v3/      01_eligibility_audit · 02_missingness_atlas · 03_build_prior_matrix · 04_fit_measurement
├── tests/           unit tests
├── results/v3/      regenerated aggregates: eligibility/ · missingness/ · bayesian/ · bayesian_ext/ · sleep_sensitivity/ (gitignored; empty on a clean tree)
└── docs/            V3_PLAN · ROADMAP · PIPELINE · DATA · FINDINGS · LABBOOK_V3 · V3_RESULTS · neuropsy_features.yaml · figures/v3/
```

## Quick start
```bash
pip install -e ".[full]"                       # core + kaleido (static figure export)
python3 scripts/v3/01_eligibility_audit.py          # eligibility + V0 coverage      → results/v3/eligibility/
python3 scripts/v3/02_missingness_atlas.py          # missingness mechanism          → results/v3/missingness/
python3 scripts/v3/03_build_prior_matrix.py         # config ontology → prior matrix  → configs/
python3 scripts/v3/04_fit_measurement.py --stage 1  # staged measurement model       → results/v3/bayesian/stage1/
python3 -m pytest tests/ -q                         # unit tests
```
```python
from v3.data import build_unified_dataframe, load_variables, to_harmonized_dataset
df = build_unified_dataframe("data", "data/face-common-vars.xlsx",
                             readiness=["READY", "PARTIAL"], format="long")
ds = to_harmonized_dataset(df, load_variables("data/face-common-vars.xlsx"), visit="V0")
# ds.X: MultiIndex[cohort, patient_id] × numeric features (NaN = missing, never imputed)
```

## Documentation
- **[CLAUDE.md](CLAUDE.md)** — project guide (the central read; includes instructions for future agents).
- **[docs/STATE.md](docs/STATE.md)** — where V3 actually is right now (read first).
- **[docs/V3_PLAN.md](docs/V3_PLAN.md)** — the V3 plan of record (direction).
- **[docs/ROADMAP.md](docs/ROADMAP.md)** (what/why) · **[docs/PIPELINE.md](docs/PIPELINE.md)** (target architecture + missing-data doctrine).
- **[docs/DATA.md](docs/DATA.md)** (data contract + dictionary) · **[docs/FINDINGS.md](docs/FINDINGS.md)** (V3 log) · **[docs/LABBOOK_V3.md](docs/LABBOOK_V3.md)** (lab notebook).
- **[docs/V3_RESULTS.md](docs/V3_RESULTS.md)** — ⚠️ first-generation engine results, superseded (see STATE.md).

## Confidentiality
The FACE database is **confidential** (Fondation FondaMental). The per-cohort `data/*.csv` and all
per-patient artifacts are **gitignored and never committed**. Tracked + shareable: the code
(`src/`, `scripts/`, `tests/`), the small input dictionaries (`data/face-common-vars.xlsx`,
`data/thesaurus/`, `data/site_lookup.csv`), and regenerated **aggregate** results.
