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

> **Status.** The **V3 plan is the single source of truth** — direction, framing, and the estimator
> hierarchy are fixed by **[docs/V3_PLAN.md](docs/V3_PLAN.md)**. The **certified measurement model** is
> built (data layer `src/v3/data/`, pipeline `scripts/v3/`, outputs `results/v3/`, figures
> `docs/figures/v3/`; see **[docs/V3_RESULTS.md](docs/V3_RESULTS.md)**); the downstream
> strata / prognosis / treatment layers (Phases E–M) are **not yet built**. Project guide:
> **[CLAUDE.md](CLAUDE.md)**.

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
├── src/v3/data/     variable·rules·loader·filters · adapter·harmonized_dataset · schema_gen·feature_schema · skip_logic  (self-contained data layer; add V3 sub-packages)
├── configs/         V3 data contract: candidate_dimensions_v3 · likelihood_map_v3 · soft_loading_priors_v3
├── scripts/v3/      01_eligibility_audit · 02_missingness_atlas · 03_bayesian_core · 04_extended_model · 05_visualize · 06_sleep_affect_sensitivity
├── tests/           unit tests
├── results/v3/      regenerated aggregates: eligibility/ · missingness/ · bayesian/ · bayesian_ext/ · sleep_sensitivity/ (gitignored; empty on a clean tree)
└── docs/            V3_PLAN · ROADMAP · PIPELINE · DATA · FINDINGS · LABBOOK_V3 · V3_RESULTS · neuropsy_features.yaml · figures/v3/
```

## Quick start
```bash
pip install -e ".[full]"                       # core + kaleido (static figure export)
python3 scripts/v3/01_eligibility_audit.py     # eligibility audit            → results/v3/eligibility/
python3 scripts/v3/02_missingness_atlas.py     # missingness atlas            → results/v3/missingness/
python3 scripts/v3/03_bayesian_core.py         # certified core latent model  → results/v3/bayesian/
python3 scripts/v3/04_extended_model.py        # extended model              → results/v3/bayesian_ext/
python3 scripts/v3/05_visualize.py             # figures                      → docs/figures/v3/
python3 scripts/v3/06_sleep_affect_sensitivity.py  # sleep↔affect sensitivity → results/v3/sleep_sensitivity/
python3 -m pytest tests/ -q                    # unit tests
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
- **[docs/V3_PLAN.md](docs/V3_PLAN.md)** — the V3 plan of record.
- **[docs/ROADMAP.md](docs/ROADMAP.md)** (what/why) · **[docs/PIPELINE.md](docs/PIPELINE.md)** (target architecture + missing-data doctrine).
- **[docs/DATA.md](docs/DATA.md)** (data contract + dictionary) · **[docs/FINDINGS.md](docs/FINDINGS.md)** (V3 log) · **[docs/LABBOOK_V3.md](docs/LABBOOK_V3.md)** (lab notebook).
- **[docs/V3_RESULTS.md](docs/V3_RESULTS.md)** — certified measurement model: Φ heatmap/network, loadings, cohort scores.

## Confidentiality
The FACE database is **confidential** (Fondation FondaMental). The per-cohort `data/*.csv` and all
per-patient artifacts are **gitignored and never committed**. Tracked + shareable: the code
(`src/`, `scripts/`, `tests/`), the small input dictionaries (`data/face-common-vars.xlsx`,
`data/thesaurus/`, `data/site_lookup.csv`), and regenerated **aggregate** results.
