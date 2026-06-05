# CLAUDE.md — FACE precision psychiatry (BP · SZ · DR) — **V3**

> Guide for collaborators and AI assistants. Keep it short.
> **Plan of record: [docs/V3_PLAN.md](docs/V3_PLAN.md)** (verbatim source: [docs/V3_PLAN_SOURCE.md](docs/V3_PLAN_SOURCE.md)).
> What/why: [docs/ROADMAP.md](docs/ROADMAP.md) · Target pipeline: [docs/PIPELINE.md](docs/PIPELINE.md) ·
> Data contract: [docs/DATA.md](docs/DATA.md) · V3 log: [docs/FINDINGS.md](docs/FINDINGS.md).
> The completed **V2** dimensional study is a **benchmark/reference arm only**: [docs/legacy_v2/](docs/legacy_v2/README.md).

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

The **primary discovery engine** is a patient-level **Bayesian sparse bifactor / ESEM-like latent
model** with **mixed likelihoods** and **soft loading priors**; **FIML/SEM** is the confirmatory
benchmark; the **V2 masked-correlation** factors are the reproducibility baseline. **No naive
imputation anywhere** — missingness is handled by observed-data likelihood (and explicit missingness
models when informative), never by filling cells.

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
- ◻ **V2 = completed benchmark arm.** The runnable code (`src/trans_diag/` incl. `engine/`,
  `scripts/01–15`, `qa_harmonization`, `sensitivity_*`, `figures_manuscript`, `build_manuscript`) **is
  the V2 masked-estimator implementation**. It remains reproducible and becomes V3's Phase-D
  reproducibility baseline. Its docs live under [docs/legacy_v2/](docs/legacy_v2/README.md); its
  manuscript is `results/manuscript/` (the *"symptoms ⊥ biology"* benchmark paper).
- ⬜ **V3 discovery engine NOT yet built.** The FIML benchmark + Bayesian mixed-likelihood latent model
  + strata/prognosis/treatment layers (Phases E–M) are unimplemented. Do not describe them as done.

## ⚠️ Instructions for future agents / developers (read before doing V3 work)

1. **Ignore the V2 roadmap assumptions.** V2's conclusions — *K=3 axes, "no p-factor," "symptoms
   orthogonal to biology," "no discrete subtypes," the V2 manuscript framing* — are **hypotheses to
   retest** ([docs/FINDINGS.md](docs/FINDINGS.md)), **not** the project's objective or settled facts.
   Do not lead new work with the masked pairwise-correlation / PAF / Schmid–Leiman pipeline; it is the
   benchmark, not the primary model.
2. **Preserve the no-naive-imputation principle.** Never build a mean/KNN/MICE-filled matrix for
   discovery or clustering, and never feed completed vectors to a model that requires them. Use
   **observed-data likelihood** (FIML / Bayesian) over each patient's observed cells; keep deterministic
   **skip-logic** structural-zero decoding (that is not imputation). Doctrine: [docs/PIPELINE.md](docs/PIPELINE.md) §1.
3. **Use observed-likelihood modeling where possible**, with **mixed likelihoods** that carry each
   variable's type (Gaussian/Student-t, lognormal, ordered-logit, Bernoulli, neg-binomial) and
   **posterior uncertainty** on dimension scores. Do **not** force everything onto one `[−1,1]` metric
   (that is the benchmark-arm encoding).
4. **Keep diagnosis as a covariate / validation target, never a clustering feature.** Derive dimensions
   and strata without DSM labels; use BP/SZ/DR only to adjust indicator means and to *validate* (η²,
   ARI, confounding, invariance, coverage).
5. **Produce outputs aligned with the V3 decision-modeling framework.** Dimensions → **probabilistic
   strata as validated decision regions** → prognosis model ladder (M0→M6, with calibration + decision
   curves) → target-trial treatment modeling. Acceptance is **utility, not elegance**: every accepted
   dimension/stratum must show a downstream value. A V3 result must beat or defensibly refine the V2
   benchmark — reproducing V2 with heavier machinery is not success.

## Data — harmonization, no-imputation, skip-logic (the load-bearing foundation)

`scripts/qa_harmonization.py` → `results/reports/qa_harmonization.html` is the V2-arm QA surface; its
**harmonization + sanity-bounds + skip-logic decoding** remain load-bearing for V3, while the
`[−1,1]` post-processing stage is benchmark-only.

1. **Harmonized variables (native scale).** Each dictionary variable is read from its per-cohort source
   column, run through its harmonization rule (`rules.py`: text→code, unit fixes) and per-variable
   **sanity bounds** (out-of-range → NaN, **never imputed**), landing on its native clinical scale.
2. **Skip-logic structural zeros.** Gated/branching items (e.g. suicide-attempt details) are decoded to
   structural zeros where the gate is negative — **never** where the gate is unknown, and **never**
   overwriting an observed value (`skip_logic.py`).
3. **V3 data contract.** The dictionary is extended with per-variable **likelihood family, missingness
   type, soft prior loading on the 10-candidate ontology, covariate/outcome status, and modeling role**
   ([docs/DATA.md](docs/DATA.md) → "V3 data contract"). These drive the patient-level latent model.

> **No imputation, ever.** All structure is estimated from observed cells. The V2 arm did this with a
> *masked* pairwise-complete correlation; V3 does it with a patient-level **observed-data likelihood**
> (FIML / Bayesian), which preserves the principle while escaping pairwise-correlation limitations.

## Repository layout

```
face-common-bp-sz-dr/
├── CLAUDE.md  AGENTS.md  README.md     ← guides (V3); plan of record: docs/V3_PLAN.md
├── data/                               ← inputs (read-only)
│   ├── face-common-vars.xlsx           ← harmonized common-variables dictionary (tracked)
│   ├── thesaurus/                      ← per-cohort source dictionaries (tracked, reference)
│   └── {bipolar,schizophrenia,depression}.csv · site_lookup.csv   ← data (confidential; gitignored except site_lookup)
├── src/trans_diag/                     ← the package (V2 benchmark implementation today)
│   ├── variable·rules·loader·filters.py        ← harmonization + sanity bounds + skip-logic (V3-load-bearing)
│   ├── schema_gen·adapter·domains.py           ← matrix build, type-aware scaling, domain aggregation (benchmark encoding)
│   ├── masked_fa·axes·phenotype.py             ← V2 masked estimator, axis names, phenotype features (→ V3 priors)
│   └── engine/                                 ← masked-similarity stratification (→ V3 secondary control)
│       ←  V3 modules to ADD: bayesian/ · fiml/ · missingness/ · priors/ · strata/ · prognosis/ · treatment/
├── configs/                            ← V3 data contract: candidate_dimensions_v3 · likelihood_map_v3 · soft_loading_priors_v3
├── scripts/                            ← V2 benchmark pipeline (01–15) + qa_harmonization/verify/audit
│   └── v3/                              ← V3 pipeline: 01_eligibility_audit · 02_missingness_atlas · 03_bayesian_core
├── tests/                              ← unit + V2 golden-number tests (pinned to results/hfa/; skip on a clean clone)
├── results/                            ← regenerated AGGREGATE artifacts (gitignored): hfa/ · manuscript/ (V2 paper) · reports/
│   └── v3/                              ← V3 outputs: eligibility/ · missingness/ · bayesian/
├── docs/                               ← V3_PLAN · V3_PLAN_SOURCE · ROADMAP · PIPELINE · DATA · FINDINGS · LABBOOK_V3 · neuropsy_features.yaml
│   └── legacy_v2/                       ← the V2 benchmark/reference arm (PIPELINE · FINDINGS · ATLAS · LABBOOK · planning · …)
└── pyproject.toml
```

**Imports.** `trans_diag` resolves from `src/`. Scripts insert `src/` on `sys.path`; pytest uses
`pythonpath = ["src"]`. Or `pip install -e ".[full]"`.

## Core concepts (the data layer — shared by V2 and V3)

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
pip install -e ".[full]"                 # core + kaleido (static figure export)
python3 scripts/qa_harmonization.py      # harmonization QA report (load-bearing for V3; [-1,1] stage is benchmark-only)
python3 scripts/00_run_all.py            # regenerate the V2-benchmark results/hfa/ artifacts (needs the cohort CSVs)
python3 -m pytest tests/ -q              # unit + V2 golden tests (golden skip on a clean clone — results/hfa/ gitignored)
```

```python
from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset
df = build_unified_dataframe("data", "data/face-common-vars.xlsx",
                             readiness=["READY", "PARTIAL"], format="long")
ds = to_harmonized_dataset(df, load_variables("data/face-common-vars.xlsx"), visit="V0")
# ds.X: MultiIndex[cohort, patient_id] × numeric features (NaN = missing, never imputed)
```

## Where to read next

- **Plan of record (A–T)** → [docs/V3_PLAN.md](docs/V3_PLAN.md) · verbatim source [docs/V3_PLAN_SOURCE.md](docs/V3_PLAN_SOURCE.md)
- **What/why** → [docs/ROADMAP.md](docs/ROADMAP.md) · **Target pipeline + missing-data doctrine** → [docs/PIPELINE.md](docs/PIPELINE.md)
- **Data contract + dictionary** → [docs/DATA.md](docs/DATA.md) · **V3 results log** → [docs/FINDINGS.md](docs/FINDINGS.md) · **step-by-step lab notebook** → [docs/LABBOOK_V3.md](docs/LABBOOK_V3.md)
- **V3 code/outputs** → `scripts/v3/` (01 eligibility · 02 missingness · 03 Bayesian core) → `results/v3/`; **data contract** → `configs/`
- **Cognition include-list** → [docs/neuropsy_features.yaml](docs/neuropsy_features.yaml)
- **V2 benchmark/reference arm** → [docs/legacy_v2/](docs/legacy_v2/README.md) (pipeline · findings · phenotype atlas · lab notebook · planning)
- **V2 manuscript (benchmark paper)** → [results/manuscript/manuscript.md](results/manuscript/manuscript.md)

## Conventions

- **Python ≥ 3.11.** Develop in `src/trans_diag` (incl. `engine/`); add V3 sub-packages as above.
- **No naive imputation, ever.** Observed-data likelihood + masked methods only.
- **Diagnosis is a covariate / validation target**, never a clustering feature.
- **Determinism**: fixed seeds; CV folds shuffled (the patient matrix is cohort-ordered).
- **Output**: scripts write aggregates to `results/`, HTML/figures to `results/reports/`.
- **Recoverability**: the V2 arm is `legacy_v2/` + `results/manuscript/`; the full v1 study is at git
  tag `v1-archive-2026-05-30`.
