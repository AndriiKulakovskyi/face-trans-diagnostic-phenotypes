# FACE — Clinical-biological transdiagnostic stratification (BP · SZ · DR)

Across bipolar disorder, schizophrenia, and major depression in the FACE cohorts, this project turns the
harmonized 3-cohort **baseline (V0)** data into a **transdiagnostic dimensional map** — continuous,
diagnosis-agnostic axes of clinical and biological variation — and then, on that map, into **validated
patient strata**, a test of their **temporal coherence**, and **prognosis / treatment** decision models.
Four layers that are never collapsed:

```text
diagnostic cohorts  →  transdiagnostic dimensions  →  validated strata  →  prognosis  →  treatment
  (entry metadata)        (M1 — 8-factor map)         (M2 — continuum)     (M4)         (M5 — boundary)
                                                       (M3 — temporal coherence: trait/state)
```

> **Status — Milestones 1–5 complete.** **The calibrated bottom line:** a real, stable, **continuum** (not
> biotype) transdiagnostic map, with biology least-entangled from severity, that carries a **small,
> group-level** incremental prognostic signal for functioning and shows **no demonstrable treatment
> moderation** in observational data. *Scientific validity is demonstrated; strong clinical utility
> (individual prediction, treatment guidance) is not* — the modest/null results are reported as the
> contribution, a calibrated counterweight to biotype/biomarker over-claiming. Read first:
> **[docs/STATE.md](docs/STATE.md)**.

## What the program found (the arc)

- **M1 — the map *exists*.** One global, missingness-aware Bayesian sparse bifactor/ESEM on observed cells
  yields the **8-dimension map**: a general functional-burden factor **G** ⊥ 7 specific axes {cognition,
  **immunometabolic**, sleep, mania/activation, suicidality, developmental-risk, substance}, with 3 earned
  cross-loadings on cognition. Biology (the immunometabolic axis) is the **least severity-entangled** domain.
- **M2 — it *organizes*.** A structure gate finds **no biotypes**: the space is a graded **continuum** —
  a stable **A=5 archetype simplex** + a nested K-family — transdiagnostic (ARI ≈ 0 vs DSM-5) and a tighter
  description of the coordinates than diagnosis.
- **M3 — it *persists*.** Scored forward onto follow-up (V0→V1→V2, never re-estimated), the measurement is
  invariant and the geometry replays: **durable biology, moving symptoms** (immunometabolic ICC 0.91).
- **M4 — it *predicts*.** An errors-in-variables Bayesian GLM shows the baseline map forecasts 2-year
  **functioning** incrementally beyond diagnosis + severity + baseline (archetypes ΔELPD +62.8) —
  **modestly, group-level**, co-informative with DSM-5, course-dependent.
- **M5 — it does *not* (yet) *prescribe*.** A causal pipeline (overlap → propensity → doubly-robust EIV
  moderation + E-value) finds the map does **not** reliably moderate response on observational treatment-as-
  usual (lithium-BP a well-identified null; antipsychotic suggestive-unconfirmed). The prognosis **survives**
  treatment adjustment. The boundary is **earned, not assumed**.

Per-milestone methods + findings: **[docs/](docs/)** (read `STATE.md` first).

## Method — the discipline

One global Bayesian sparse **bifactor / ESEM** with **mixed likelihoods**, a Gaussian-copula (rank-INT)
continuous block marginalized via the **Woodbury** identity, and a regularized-horseshoe prior on the
off-home cross-loadings, estimates the factor structure; the data **confirm, split, merge, downgrade, or
reject** each candidate; confirmation is **in-engine** (prior-free refit + PPC + WAIC). Everything downstream
consumes the **fixed** M1/M2/M3 objects (coordinates, per-patient uncertainty, archetype memberships,
attrition weights) without re-estimation. Three load-bearing invariants:

- **No naive imputation** — structure is estimated from each patient's observed cells (FIML / observed-data
  likelihood); never a mean/KNN/MICE-filled matrix.
- **Diagnosis is metadata** — a covariate / invariance grouping / validation label, never a dimension indicator.
- **Baseline (V0) defines; later visits validate** — no discovery on V1–V4; uncertainty propagated end-to-end.

## Engine & pipeline

The package is **`src/face/…`**, one clean engine per milestone — `face.measurement` (M1), `face.strata`
(M2), `face.temporal` (M3), `face.prognosis` (M4), `face.treatment` (M5) — plus the `config`, `caching`,
`data`, `benchmark`, and `reporting` layers. Sensitivity/exploration arms (variational GLLVM, representation
benchmark, …) live under **`analyses/`**, separate from the core wheel.

One entry point drives the whole vertical:

```bash
pip install -e ".[bayesian,strata,dev]"
export PYTHONPATH=$PWD/src ; export HDF5_USE_FILE_LOCKING=FALSE

face build-data && face build-covariates      # raw cohorts → data/processed/*.parquet
face fit m1 --mode production --detach         # the transdiagnostic map (long; wake-locked, survives sleep)
face fit m2 --detach ; face fit m3 --detach ; face fit m4 --detach ; face fit m5 --detach
face status --watch                            # detached-job dashboard
make golden                                    # numerical-kernel tests (no confidential data)
```

Hand-offs live under `results/<mN_name>/` (gitignored). Full reproduction: **[REPRODUCE.md](REPRODUCE.md)**.
The reproduction was verified end-to-end against a frozen oracle (`reference/oracle/`) — M1 loadings/Φ
bit-identical, M2 exact, M3–M5 within tolerance (see `reports/refactor/`).

## Data foundation (no imputation)

Each dictionary variable → harmonization rule + per-variable **sanity bounds** (out-of-range → NaN, never
imputed) → native clinical scale, with deterministic **skip-logic** structural-zero decoding. Raw data are
confidential per-cohort **CSV**; model-ready tables are **Parquet**. The data layer carries each variable's
likelihood family and missingness type — the data contract, see [docs/DATA.md](docs/DATA.md).

## Manuscripts

Three manuscripts build from the pipeline results: **`article/`** (the flagship FACE-Atlas), **`article_immunometabolic_burden/`**
(biology-forward reframing), and **`article_methods/`** (methods companion). Each has a `compile.sh` +
figure-generation scripts that read the regenerated `results/`.

## Documentation

- **[docs/STATE.md](docs/STATE.md)** — where the project is right now (**read first**).
- **[docs/](docs/)** — methods of record, paper-facing findings, and clinician atlases, per milestone.
- **[REPRODUCE.md](REPRODUCE.md)** — build the data + fit M1→M5 from sources.
- **[docs/DATA.md](docs/DATA.md)** — data contract + dictionary · **[CLAUDE.md](CLAUDE.md)** — collaborator / AI guide.

## Confidentiality

The FACE database is confidential (Fondation FondaMental). The per-cohort `data/*.csv` and all per-patient
artifacts are **gitignored and never committed**. Tracked + shareable: the code (`src/`, `scripts/`, `tests/`,
`analyses/`), the small input dictionaries, and regenerated **aggregate** results.
