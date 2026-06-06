# FACE — Clinical-biological transdiagnostic stratification (BP · SZ · DR)

Across bipolar disorder, schizophrenia, and major depression in the FACE cohorts, this project turns the
harmonized 3-cohort **baseline (V0)** data into a **transdiagnostic dimensional map** — continuous,
diagnosis-agnostic axes of clinical and biological variation — and then, on that map, into **validated
patient strata** and **prognosis / treatment decision models**.

```text
diagnostic cohorts (BP · SZ · DR)  →  transdiagnostic dimensions  →  validated strata  →  prognosis / treatment
        (entry metadata)               (M1 — current work)             (later)              (later)
```

**Method — hybrid discovery.** The 10 candidate dimensions are a **soft prior ontology**, not fixed
scores. One global, missingness-aware **Bayesian sparse bifactor / ESEM** model with **mixed likelihoods**
estimates the actual factor structure, and the data **confirm, split, merge, downgrade, or reject** each
candidate; **FIML** is the confirmatory estimator. Three invariants: **no naive imputation** (structure is
estimated from each patient's observed cells); **diagnosis is metadata**, never an indicator; dimensions
are discovered on **V0**, later visits validate them.

> **Status — Milestone 1 (the measurement map).** The methods and mathematics are fixed in
> **[docs/MEASUREMENT_MODEL.md](docs/MEASUREMENT_MODEL.md)** (the methods-of-record). Implementation — the
> global staged fit, FIML confirmation, scoring — is the current build. Strata / prognosis / treatment are
> later milestones. Where the project stands right now: **[docs/STATE.md](docs/STATE.md)**.

## The map being estimated (V0)

3-cohort core: **G (overall severity)** · **cognition** · **metabolic** · **inflammatory** · **sleep** ·
**suicidality** · **developmental-risk**. BP/DR extension: **anhedonia** (thin). Dropped for lack of
indicators in the common variables: **impulsivity, negative symptoms, sensory** — *and stating so is a
result of the analysis, not a failure.*

## Data foundation (no imputation)

Each dictionary variable → harmonization rule + per-variable **sanity bounds** (out-of-range → NaN, never
imputed) → native clinical scale, with deterministic **skip-logic** structural-zero decoding. Raw data are
confidential per-cohort **CSV**; the model-ready tables are persisted as **Parquet**. The data layer
carries each variable's likelihood family and missingness type — the data contract, see
[docs/DATA.md](docs/DATA.md).

## Quick start

```bash
pip install -e ".[full]"        # core + figure export; add ".[bayesian]" for PyMC / NumPyro
python3 -m pytest tests/ -q     # data-layer + contract tests
```

The measurement pipeline (build → missingness → priors → staged global fit → FIML → adjudicate → score) is
specified in [docs/MEASUREMENT_MODEL.md](docs/MEASUREMENT_MODEL.md) §11 and being implemented.

## Documentation

- **[docs/MEASUREMENT_MODEL.md](docs/MEASUREMENT_MODEL.md)** — methods + mathematics + staged estimation (**canonical**).
- **[docs/STATE.md](docs/STATE.md)** — where the project is right now (read first).
- **[CLAUDE.md](CLAUDE.md)** — guide for collaborators / AI assistants.
- **[docs/DATA.md](docs/DATA.md)** — data contract + dictionary.

## Confidentiality

The FACE database is confidential (Fondation FondaMental). The per-cohort `data/*.csv` and all per-patient
artifacts are **gitignored and never committed**. Tracked + shareable: the code (`src/`, `scripts/`,
`tests/`), the small input dictionaries, and regenerated **aggregate** results.
