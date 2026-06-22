# FACE — Clinical-biological transdiagnostic stratification (BP · SZ · DR)

Across bipolar disorder, schizophrenia, and major depression in the FACE cohorts, this project turns the
harmonized 3-cohort **baseline (V0)** data into a **transdiagnostic dimensional map** — continuous,
diagnosis-agnostic axes of clinical and biological variation — and then, on that map, into **validated
patient strata**, a test of their **temporal coherence**, and **prognosis / treatment** decision models.
Four layers that are never collapsed:

```text
diagnostic cohorts  →  transdiagnostic dimensions  →  validated strata  →  prognosis  →  treatment
  (entry metadata)        (M1 ✓ certified 9-dim)      (M2 ✓ continuum)    (M4 ✓)       (M5 ✓ boundary)
                                                       (M3 ✓ temporal coherence: trait/state)
```

> **Status — Milestones 1–5 complete** (pending PI sign-off). **The calibrated bottom line:** a real,
> stable, **continuum** (not biotype) transdiagnostic map, with biology orthogonal to severity, that
> carries a **small, group-level** incremental prognostic signal for functioning and shows **no
> demonstrable treatment moderation** in observational data. *Scientific validity is demonstrated; clinical
> utility in the strong sense (individual prediction, treatment guidance) is not* — and the modest and null
> results are reported as the contribution, a calibrated counterweight to biotype/biomarker over-claiming.
> Read first: **[docs/STATE.md](docs/STATE.md)**.

## What the program found (the arc)

- **M1 — the map *exists*.** One global, missingness-aware Bayesian sparse bifactor/ESEM on observed cells
  yields a **certified 9-dimension map**: a general functional-burden factor **G** + eight weakly-correlated
  specific axes. Biology (metabolic, inflammatory) is the **least severity-entangled** domain.
  → [docs/M1_FINDINGS.md](docs/M1_FINDINGS.md)
- **M2 — it *organizes*.** A structure-discovery gate finds **no biotypes**: the space is a graded
  **continuum** — 8 soft archetypes + a 4-region tessellation — transdiagnostic (adjusted Rand ≈ 0 vs the
  DSM-5 subtypes) and a tighter description of the coordinates than diagnosis.
  → [docs/STRATA_OOP_FINDINGS.md](docs/STRATA_OOP_FINDINGS.md)
- **M3 — it *persists*.** Scored forward onto follow-up (V0→V1→V2, never re-estimated), the measurement is
  invariant and the geometry replays: **durable biology, moving symptoms** (trait/state).
  → [docs/TEMPORAL_OOP_FINDINGS.md](docs/TEMPORAL_OOP_FINDINGS.md)
- **M4 — it *predicts*.** An errors-in-variables Bayesian GLM shows a baseline coordinate forecasts 2-year
  **functioning** incrementally beyond diagnosis + severity + baseline — **modestly, and group-level**
  (remission ΔAUC +0.017; archetype atlas 14%→60%, partly severity), co-informative with DSM-5,
  course-dependent. → [docs/PROGNOSIS_OOP_FINDINGS.md](docs/PROGNOSIS_OOP_FINDINGS.md)
- **M5 — it does *not* (yet) *prescribe*.** Treatment data, found in the per-cohort source dictionaries and
  harmonized, is run through a causal pipeline (overlap → propensity → doubly-robust EIV moderation +
  E-value): on observational treatment-as-usual the map does **not** reliably moderate response (lithium-BP
  a well-identified null; antipsychotic suggestive-unconfirmed; clozapine channeled). The prognosis
  **survives** treatment adjustment. The boundary is **earned, not assumed**.
  → [docs/TREATMENT_OOP_FINDINGS.md](docs/TREATMENT_OOP_FINDINGS.md)

## Method — the discipline
flowchart TD
  dims["dimensions.yaml<br/>(ontology: factors, anchors, windows)"] --> bm["priors/build_matrix.py"]
  pri["priors.yaml<br/>(soft-prior tiers)"] --> bm
  lik["likelihood_map_v3.yaml<br/>(per-item family)"] --> bm
  bm --> mat["prior_loading_matrix_v3.csv<br/>(item x factor cells)"]

  xlsx["face-common-vars.xlsx<br/>(3 cohorts, harmonized)"] --> s01["01_build_data.py<br/>skip-logic, NaN preserved"]
  s01 --> pq["baseline_v0.parquet"]
  s02["02_build_covariates.py"] --> covpq["covariates_v0.parquet"]

  mat --> prep["continuous_core.prepare()<br/>encode + resolve priors"]
  pq --> prep
  prep --> fit["04_fit.py<br/>build_marginalized / build_mixed"]
  fit --> idata["idata.nc (Lam, Phi, sigma, f_e)"]

  idata --> conf["05 confirm (PPC/SRMR/WAIC/prior-free)"]
  idata --> inv["06,13 invariance"]
  idata --> score["07 score -> 20 full-N coords"]
  idata --> rob["08 robustness / 10,10b sensitivity"]
  idata --> atlas["09 atlas / 12 mixed PPC"]

One global Bayesian sparse **bifactor / ESEM** with **mixed likelihoods** estimates the factor structure;
the data **confirm, split, merge, downgrade, or reject** each candidate; confirmation is **in-engine**
(prior-free refit + posterior-predictive checks + WAIC — a standalone FIML proved redundant). Everything
downstream consumes the **fixed** M1/M2/M3 objects (coordinates, per-patient uncertainty, archetype
memberships, attrition weights) without re-estimation. Three load-bearing invariants:

- **No naive imputation** — structure is estimated from each patient's observed cells (FIML / observed-data
  likelihood); never a mean/KNN/MICE-filled matrix.
- **Diagnosis is metadata** — a covariate / invariance grouping / validation label, never a dimension
  indicator.
- **Baseline (V0) defines; later visits validate** — no discovery on V1–V4; uncertainty is propagated
  end-to-end.

## The map (V0) — the certified nine dimensions

**G (functional burden)** + **cognition · metabolic · inflammatory · sleep · developmental-risk ·
suicidality · mania · substance**. Depression/anxiety instruments (MADRS/QIDS/STAI) enter as **cross-loading
windows**, not a separate factor; **anhedonia** is **rejected**; **impulsivity / negative-symptoms /
sensory** are **`not_testable`** (no indicators in the common variables) — *and stating so is a result of
the analysis, not a failure.* Per-candidate verdicts: [docs/ADJUDICATION.md](docs/ADJUDICATION.md).

## Data foundation (no imputation)

Each dictionary variable → harmonization rule + per-variable **sanity bounds** (out-of-range → NaN, never
imputed) → native clinical scale, with deterministic **skip-logic** structural-zero decoding. Raw data are
confidential per-cohort **CSV**; model-ready tables are persisted as **Parquet**. The data layer carries
each variable's likelihood family and missingness type — the data contract, see [docs/DATA.md](docs/DATA.md).

## Engine & pipeline

The package is **`src/face/…`**: the measurement engine (`models/bayesian`, `confirm`, `runner`, `scoring`)
+ the milestone engines (`strata/`, `temporal/`, `prognosis/`, `treatment/`). The staged pipeline is
`scripts/01–57` (M1 `01–09,s5_*` · M2 `20–26` · M3 `30–37` · M4 `40–48` · M5 `50–57`); each stage writes a
`reports/NN_*.md` + figures. Hand-off artifacts (gitignored) live under `results/face/`.

## Quick start

```bash
pip install -e ".[full]"        # core + figure export; add ".[bayesian]" for PyMC / NumPyro
python3 -m pytest tests/ -q     # data-layer, engine, and milestone tests
```

To **rebuild the model-ready data and re-fit the measurement model (M1) from scratch** — data loading +
the Bayesian bifactor/ESEM map only — follow **[REPRODUCE.md](REPRODUCE.md)**.

The manuscript (the FACE Atlas, Milestones 1–5) builds from the project results:

```bash
python3 report/make_figures.py          # regenerate figures from tracked reports/*.csv
cd report && pdflatex FACE-ATLAS.tex    # ×2 for TOC/refs  (PDF is gitignored)
```

## Documentation

- **[docs/STATE.md](docs/STATE.md)** — where the project is right now (**read first**).
- **Paper-facing findings (read-first per milestone):** [M1](docs/M1_FINDINGS.md) ·
  [M2](docs/STRATA_OOP_FINDINGS.md) · [M3](docs/TEMPORAL_OOP_FINDINGS.md) · [M4](docs/PROGNOSIS_OOP_FINDINGS.md) ·
  [M5](docs/TREATMENT_OOP_FINDINGS.md).
- **Methods of record:** [measurement](docs/MEASUREMENT_MODEL.md) (canonical) ·
  [stratification](docs/STRATIFICATION_MODEL.md) · [temporal](docs/TEMPORAL_MODEL.md) ·
  [prognosis](docs/PROGNOSIS_MODEL.md) · [treatment](docs/TREATMENT_MODEL.md).
- **Clinician-facing atlases:** [strata](docs/STRATA_OOP_ATLAS.md) · [prognosis](docs/PROGNOSIS_OOP_FINDINGS.md).
- **[REPRODUCE.md](REPRODUCE.md)** — step-by-step: build the data + re-fit the measurement model (M1).
- **[docs/DATA.md](docs/DATA.md)** — data contract + dictionary · **[CLAUDE.md](CLAUDE.md)** — guide for
  collaborators / AI assistants · **`report/`** — the LaTeX manuscript (Milestones 1–5 + Conclusion).

## Confidentiality

The FACE database is confidential (Fondation FondaMental). The per-cohort `data/*.csv` and all per-patient
artifacts are **gitignored and never committed**. Tracked + shareable: the code (`src/`, `scripts/`,
`tests/`), the small input dictionaries, and regenerated **aggregate** results.
