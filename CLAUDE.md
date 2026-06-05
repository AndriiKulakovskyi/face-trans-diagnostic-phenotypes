# CLAUDE.md — FACE precision psychiatry (BP · SZ · DR) — **V3**

> Guide for collaborators and AI assistants. Keep it short.
> **Plan of record: [docs/V3_PLAN.md](docs/V3_PLAN.md)**.
> What/why: [docs/ROADMAP.md](docs/ROADMAP.md) · Target pipeline: [docs/PIPELINE.md](docs/PIPELINE.md) ·
> Data contract: [docs/DATA.md](docs/DATA.md) · V3 log: [docs/FINDINGS.md](docs/FINDINGS.md).

## What this is

A **self-contained** project that harmonizes the 3-cohort FACE psychiatric data (Bipolar,
Schizophrenia, Depression; baseline **V0** → 4-year V4) and turns it into a **precision-psychiatry
stratification and decision-modeling framework**, in four layers that must not be collapsed:

```text
diagnostic cohorts (BP · SZ · DR)          ← entry + validation metadata, NEVER clustering features
  → transdiagnostic dimension discovery      ← patient-level, missingness-aware latent measurement
  → validated patient strata                  ← probabilistic decision regions, not natural subtypes
  → prognosis / treatment decision models     ← the precision-psychiatry objective
```

The **primary discovery engine** is a patient-level **marginalized Bayesian sparse bifactor / ESEM-like
latent model** with **mixed likelihoods** and **soft loading priors**; **FIML/SEM** is the confirmatory
follow-up. **No naive imputation anywhere** — missingness is handled by observed-data likelihood (and
explicit missingness models when informative), never by filling cells.

### The 10 candidate dimensions are a *soft starting ontology*, not fixed scores

Impulsivity · Cognitive flexibility · Negative symptoms · Anhedonia · Metabolism/immunometabolism ·
Sleep/circadian · Overall clinical severity · Sensory abnormalities · Neurodevelopment · Suicidality.

They seed **soft priors**, not hand-tagged composite scores. The data is explicitly allowed to
**confirm, split, merge, reject, downgrade, or cross-load** any of them; the final dimension set may be
smaller or different from the initial 10 (that is the expected output, not a failure). Starting
eligibility + the full A–T plan: [docs/V3_PLAN.md](docs/V3_PLAN.md).

## Status

- 🟢 **V3 plan adopted as the single source of truth** (2026-06-05). Direction, framing, and the
  estimator hierarchy are fixed by [docs/V3_PLAN.md](docs/V3_PLAN.md).
- 🟢 **Certified measurement model built.** The self-contained data layer (`src/v3/data/`), the
  pipeline (`scripts/v3/`: 01 eligibility audit · 02 missingness atlas · 03 Bayesian core · 04 extended
  model · 05 visualize · 06 sleep↔affect sensitivity), and its outputs (`results/v3/`, figures under
  `docs/figures/v3/`) are in place; results are logged in [docs/V3_RESULTS.md](docs/V3_RESULTS.md).
- ⬜ **Downstream decision layers NOT yet built.** The strata / prognosis / treatment layers
  (Phases E–M) are unimplemented. Do not describe them as done.

## ⚠️ Instructions for future agents / developers (read before doing V3 work)

1. **Preserve the no-naive-imputation principle.** Never build a mean/KNN/MICE-filled matrix for
   discovery or clustering, and never feed completed vectors to a model that requires them. Use
   **observed-data likelihood** (marginalized Bayesian / FIML) over each patient's observed cells; keep
   deterministic **skip-logic** structural-zero decoding (that is not imputation). Doctrine:
   [docs/PIPELINE.md](docs/PIPELINE.md) §1.
2. **Use observed-likelihood modeling where possible**, with **mixed likelihoods** that carry each
   variable's type (Gaussian/Student-t, lognormal, ordered-logit, Bernoulli, neg-binomial) and
   **posterior uncertainty** on dimension scores. Do **not** force everything onto a single shared metric.
3. **Keep diagnosis as a covariate / validation target, never a clustering feature.** Derive dimensions
   and strata without DSM labels; use BP/SZ/DR only to adjust indicator means and to *validate* (η²,
   ARI, confounding, invariance, coverage).
4. **Produce outputs aligned with the V3 decision-modeling framework.** Dimensions → **probabilistic
   strata as validated decision regions** → prognosis model ladder (M0→M6, with calibration + decision
   curves) → target-trial treatment modeling. Acceptance is **utility, not elegance**: every accepted
   dimension/stratum must show a downstream value.

## Data — harmonization, no-imputation, skip-logic (the load-bearing foundation)

The self-contained data layer is `src/v3/data/`; its **harmonization + sanity-bounds + skip-logic
decoding** produce each patient's observed-data matrix on native clinical scales.

1. **Harmonized variables (native scale).** Each dictionary variable is read from its per-cohort source
   column, run through its harmonization rule (`rules.py`: text→code, unit fixes) and per-variable
   **sanity bounds** (out-of-range → NaN, **never imputed**), landing on its native clinical scale.
2. **Skip-logic structural zeros.** Gated/branching items (e.g. suicide-attempt details) are decoded to
   structural zeros where the gate is negative — **never** where the gate is unknown, and **never**
   overwriting an observed value (`skip_logic.py`).
3. **V3 data contract.** The dictionary is extended with per-variable **likelihood family, missingness
   type, soft prior loading on the 10-candidate ontology, covariate/outcome status, and modeling role**
   ([docs/DATA.md](docs/DATA.md) → "V3 data contract"). These drive the patient-level latent model.

> **No imputation, ever.** All structure is estimated from observed cells via a patient-level
> **observed-data likelihood** (marginalized Bayesian / FIML), with deterministic skip-logic
> structural-zero decoding (which is not imputation).

## Repository layout

```
face-common-bp-sz-dr/
├── CLAUDE.md  AGENTS.md  README.md     ← guides (V3); plan of record: docs/V3_PLAN.md
├── data/                               ← inputs (read-only)
│   ├── face-common-vars.xlsx           ← harmonized common-variables dictionary (tracked)
│   ├── thesaurus/                      ← per-cohort source dictionaries (tracked, reference)
│   └── {bipolar,schizophrenia,depression}.csv · site_lookup.csv   ← data (confidential; gitignored except site_lookup)
├── src/v3/data/                        ← self-contained data layer (harmonization + no-imputation loading)
│   ├── variable·rules·loader·filters.py        ← harmonization + sanity bounds + skip-logic
│   ├── adapter·harmonized_dataset.py           ← observed-data V0 matrix builder (NaN = missing, never imputed)
│   ├── schema_gen·feature_schema.py            ← feature schema / data-contract wiring
│   └── skip_logic.py                           ← structural-zero decoding
│       ←  V3 modules to ADD: bayesian/ · fiml/ · missingness/ · priors/ · strata/ · prognosis/ · treatment/
├── configs/                            ← V3 data contract: candidate_dimensions_v3 · likelihood_map_v3 · soft_loading_priors_v3
├── scripts/v3/                         ← V3 pipeline: 01_eligibility_audit · 02_missingness_atlas · 03_bayesian_core · 04_extended_model · 05_visualize · 06_sleep_affect_sensitivity
├── tests/                              ← unit tests
├── results/v3/                         ← regenerated AGGREGATE artifacts (gitignored): eligibility/ · missingness/ · bayesian/ · bayesian_ext/ · sleep_sensitivity/
├── docs/                               ← V3_PLAN · ROADMAP · PIPELINE · DATA · FINDINGS · LABBOOK_V3 · V3_RESULTS · neuropsy_features.yaml
│   └── figures/v3/                      ← certified-model figures (Φ heatmap · correlation network · loadings · scores by cohort · sleep↔affect)
└── pyproject.toml
```

**Imports.** `v3` resolves from `src/`. Scripts insert `src/` on `sys.path`; pytest uses
`pythonpath = ["src"]`. Or `pip install -e ".[full]"`.

## Core concepts (the data layer)

**`Variable`** (`variable.py`) — one per dictionary row; `source_col(cohort)` → CSV column; carries
`sanity_min/max`. **Harmonization registry** (`rules.py`) — `@register(...)`; unregistered →
`identity_cast`. **`build_unified_dataframe(...)`** (`loader.py`) — `readiness=['READY','PARTIAL']`
(199 vars); applies sanity bounds + rules + fondacode site; `format='long'|'wide'`.
**`to_harmonized_dataset(...)`** (`adapter.py`) — V0 numeric matrix, MultiIndex `[cohort, patient_id]`,
optional `residualize_on=('age','sex')`. **Skip-logic** (`skip_logic.py`) — structural-zero decoding.
**Identifiers (never modelled on):** `usubjid_patients`, `cohort`, `arm`, `visit`, `visitnum`,
`siteid_city` (excluded from features via `ADMINISTRATIVE_FEATURES`; `cohort`/`arm` are **covariates /
validation labels**, not features). **No imputation** anywhere.

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

## Where to read next

- **Plan of record (A–T)** → [docs/V3_PLAN.md](docs/V3_PLAN.md)
- **What/why** → [docs/ROADMAP.md](docs/ROADMAP.md) · **Target pipeline + missing-data doctrine** → [docs/PIPELINE.md](docs/PIPELINE.md)
- **Data contract + dictionary** → [docs/DATA.md](docs/DATA.md) · **V3 results log** → [docs/FINDINGS.md](docs/FINDINGS.md) · **step-by-step lab notebook** → [docs/LABBOOK_V3.md](docs/LABBOOK_V3.md)
- **V3 results + figures** → [docs/V3_RESULTS.md](docs/V3_RESULTS.md) (certified measurement model: Φ heatmap/network, loadings, cohort scores)
- **V3 code/outputs** → `scripts/v3/` (01 eligibility · 02 missingness · 03 Bayesian core · 04 extended model · 05 visualize · 06 sleep↔affect sensitivity) → `results/v3/`; **data contract** → `configs/`
- **Cognition include-list** → [docs/neuropsy_features.yaml](docs/neuropsy_features.yaml)

## Conventions

- **Python ≥ 3.11.** Develop in `src/v3/`; add V3 sub-packages as above.
- **No naive imputation, ever.** Observed-data likelihood (marginalized Bayesian / FIML) only.
- **Diagnosis is a covariate / validation target**, never a clustering feature.
- **Determinism**: fixed seeds; CV folds shuffled (the patient matrix is cohort-ordered).
- **Output**: scripts write aggregates to `results/v3/`, figures to `docs/figures/v3/`.
