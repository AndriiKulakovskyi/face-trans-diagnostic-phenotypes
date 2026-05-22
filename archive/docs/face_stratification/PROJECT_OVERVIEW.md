# CLAUDE.md — FACE Patient Vignette Builder

## Project Overview

This project turns the **FACE** (FondaMental Advanced Centers of Expertise) psychiatric cohort database into richly structured, clinically interpretable **patient vignettes**. FACE is a multi-site French research network that maintains deep-phenotyping cohorts for four pathologies, each shipped as a wide CSV where one row represents one patient:

| Cohort | CSV | Pathology | ~N patients | ~Columns |
|--------|-----|-----------|-------------|----------|
| **BP**  | `data/BP.csv`  | Bipolar Disorder                  | 5,400 | 2,229 |
| **SZ**  | `data/SZ.csv`  | Schizophrenia                     | 2,200 | 1,713 |
| **DR**  | `data/DR.csv`  | Treatment-Resistant Depression    |   350 | 2,217 |
| **ASP** | `data/ASP.csv` | Autism Spectrum Disorder (TSASDI) | 1,300 | 4,326 |

Each row mixes demographics, vitals, biology, dozens of rating-scale totals and sub-scales, cognitive test results, detailed psychiatric history, treatment history, family history, substance use, suicide history, comorbidities, and — for BP — a single-timepoint follow-up suffixed `_n1`. Column names are cohort-specific and often cryptic.

The codebase converts each patient row into a **`PatientProfile`** object containing:

1. A set of typed dataclasses (demographics, biology, cognitive profile, psychiatric history, treatment, suicide risk, family history, etc.) filled with normalized, clinically interpreted values (severity bands, z-scores, risk levels, French-language explanations).
2. A **`full_vignette`** — a long, human-readable French clinical narrative built from those dataclasses and suitable for downstream use by clinicians, researchers, or large language models.

Patient confidentiality: the CSVs under `data/` are confidential clinical research data. **Never commit raw patient data or expose identifiers.**

## High-Level Architecture

There are two main packages. `face_rlvr` provides deterministic patient vignette extraction and clinical interpretation. `face_stratification` implements transdiagnostic graph-based patient stratification on top of those profiles. Clinical knowledge lives in YAML files under `config/glossary/` and `config/face_stratification/`; Python code contains only *logic*.

```
psych-dataset/
├── config/
│   ├── glossary/                        # Clinical data for vignette extraction
│   │   ├── common/
│   │   │   ├── instruments.yaml            # Shared rating scales (MADRS, YMRS, ...)
│   │   │   ├── thresholds.yaml             # Reusable severity-band lists (PSQI, ESS, ...)
│   │   │   ├── clinical_constants.yaml     # BMI, metabolic syndrome, Framingham,
│   │   │   │                               # drug interactions, med–lab alerts,
│   │   │   │                               # cognitive norms
│   │   │   └── categorical_codes.yaml      # MARITAL / EDUCATION / EMPLOYMENT codes
│   │   ├── bp/
│   │   │   ├── column_map.yaml             # CSV column names used by bp_extractor
│   │   │   ├── instruments.yaml            # BP-specific scales + $registry (order + groups)
│   │   │   └── lab_ranges.yaml             # Ordered lab reference ranges
│   │   ├── sz/  (column_map, instruments, lab_ranges, categorical_codes)
│   │   ├── dr/  (column_map, instruments, lab_ranges, categorical_codes)
│   │   └── asp/ (column_map, instruments, lab_ranges, categorical_codes)
│   │
│   └── face_stratification/
│       └── feature_schema.yaml          # Unified feature definitions for stratification
│
├── src/face_rlvr/
│   └── profiles/
│       ├── glossary_schema.py           # Pydantic v2 models for every YAML file
│       ├── glossary_loader.py           # LRU-cached loader with cohort override logic
│       ├── common_instruments.py        # InstrumentDefinition / SeverityLevel dataclasses +
│       │                                #   interpret_score(); exports shared scales
│       ├── common_extractors.py         # BMI / metabolic syndrome / Framingham / drug
│       │                                #   interactions / med–lab alerts / cognitive norms
│       ├── {bp,sz,dr,asp}_instruments.py      # Per-cohort instrument registry
│       ├── {bp,sz,dr,asp}_extractor.py        # row → typed PatientData dataclass
│       └── {bp,sz,dr,asp}_profile_builder.py  # PatientData → PatientProfile + vignette
│
├── src/face_stratification/
│   ├── harmonization/                   # Stage A: feature harmonization
│   │   ├── feature_schema.py               # Pydantic models for feature_schema.yaml
│   │   ├── cohort_adapters.py              # PatientProfile → flat feature dict per cohort
│   │   ├── harmonizer.py                   # HarmonizedDataset builder (cross-cohort matrix)
│   │   ├── normalization.py                # z-score / robust normalization
│   │   ├── missingness.py                  # Missingness characterization, KNN/MICE imputation
│   │   └── dsm_subtypes.py                 # DSM subtype extraction for clinical validation
│   ├── graph/                           # Stage A: patient similarity graphs
│   │   ├── masked_similarity.py            # Pairwise-complete cosine / euclidean / Gower
│   │   ├── transdiagnostic.py              # Data-driven transdiagnostic feature selection
│   │   └── patient_similarity.py           # Block kNN, multiplex, balanced, mutual kNN
│   ├── models/                          # Stage B: embedding models
│   │   ├── base.py                         # BaseEmbeddingModel interface + PatientEmbedding
│   │   ├── baselines.py                    # PCA, UMAP, raw features
│   │   ├── raw_baseline.py                 # RawFeatureBaseline (no reduction)
│   │   ├── spectral.py                     # Spectral + multiplex spectral embeddings
│   │   ├── composite.py                    # ConcatenatedEmbedding (multi-view)
│   │   └── pipeline.py                     # fit_embedding / fit_and_save_embedding
│   ├── stage_b2/                        # Stage B2: deep GNN embeddings
│   │   ├── gcn.py                          # Sparse GCN layer (pure PyTorch, no PyG)
│   │   ├── gae.py                          # Graph Autoencoder (link prediction)
│   │   ├── contrastive.py                  # GraphCL contrastive SSL (NT-Xent)
│   │   └── sweep.py                        # Hyperparameter sweep + transdiagnostic scoring
│   ├── clustering/                      # Stage C: clustering + k selection
│   │   ├── algorithms.py                   # k-means, GMM soft, bootstrap stability
│   │   ├── metrics.py                      # ARI, NMI, silhouette, per-cohort entropy
│   │   └── k_selection.py                  # Dual-criterion k selection (gap + clinical)
│   ├── analysis/                        # Stage D: validation + interpretation
│   │   ├── enrichment.py                   # Per-cluster feature enrichment (BH-FDR)
│   │   ├── medoids.py                      # Medoid extraction + vignette retrieval
│   │   ├── ablation.py                     # Normalization ablation study
│   │   ├── meta_stability.py               # Meta-stability across embedding methods
│   │   ├── safety_analysis.py              # Suicide risk distribution safety checks
│   │   └── visualization.py                # t-SNE/UMAP projections, heatmaps, bar charts
│   ├── io/                              # I/O utilities
│   └── stage_c/                         # Stage C pipeline orchestration
│
├── notebooks/
│   ├── patient_vignette_explorer.ipynb  # End-to-end walkthrough for all 4 cohorts
│   └── stage_a_feature_inspection.ipynb # Feature engineering diagnostic plots
│
└── data/                                # FACE CSVs (gitignored, confidential)
    ├── BP.csv
    ├── SZ.csv
    ├── DR.csv
    └── ASP.csv
```

## How the Code Works

### 1. Glossary layer (YAML, validated by Pydantic)

All clinical knowledge — instrument definitions, severity thresholds, lab reference ranges, categorical code lookups, clinical computation constants, and **every CSV column name** — lives in YAML files under `config/glossary/`. `glossary_schema.py` declares a Pydantic v2 model for every file (`InstrumentsFileConfig`, `LabRangesFileConfig`, `CohortColumnMap`, `ClinicalConstantsConfig`, …), each with `extra="forbid"` so malformed YAML fails loudly at load time.

`glossary_loader.py` exposes a small read-only API backed by `functools.lru_cache`, so each YAML file is parsed once per process:

```python
get_cohort_instruments(cohort)        # dict[str, InstrumentDefinition]
get_cohort_instrument_groups(cohort)  # dict[str, list[str]] — e.g. MOOD_INSTRUMENTS
get_cohort_lab_ranges(cohort)         # list[LabRangeConfig], order preserved
get_cohort_column_map(cohort)         # CohortColumnMap — typed CSV column accessors
get_clinical_constants()              # ClinicalConstantsConfig
get_common_categorical_codes()        # dict[str, dict[str, str]]
```

Cohort instrument files declare only the scales that differ from `common/`. A cohort override (e.g. SZ redefines `CGI-S` with column `cgi0101`, ASP redefines `PSQI` with `psqi_score`) wins over the shared default. Each cohort instrument file ends with a `$registry` block that preserves display order and names reusable instrument groups (`MOOD_INSTRUMENTS`, `COGNITION_INSTRUMENTS`, …).

The loader converts the validated Pydantic models into pre-existing dataclasses (`InstrumentDefinition`, `SeverityLevel`) used across the codebase — this keeps the call sites trivially typed and makes the YAML migration fully transparent.

### 2. Extraction layer (`{cohort}_extractor.py`)

One extractor per cohort. Each takes a single pandas `Series` (one CSV row) and returns a typed cohort-specific dataclass tree (e.g. `BPPatientData`, `SZPatientData`, …) containing:

- `Demographics` — age, sex, marital status, education, employment
- `BiologicalPanel` — vitals (BMI, BP, HR), ECG (QT, QTc), labs (with interpreted ranges, flagged abnormalities, sex-specific thresholds)
- `CognitiveProfile` — TMT, Stroop, CVLT, fluency, WAIS, digit span, with z-scores against YAML-stored age-decade norms
- `PsychiatricHistory` — age at onset, lifetime and past-year episode counts, current episode DSM criteria, rapid cycling (BP)
- `TreatmentPanel` — current medication classes, plasma levels (Li/VPA/CBZ/LTG for BP, clozapine for SZ), adherence (MARS)
- `SuicideRiskPanel` — lifetime ISF items, C-SSRS or BDI-II item 9, MADRS item 10
- `FamilyHistory` — extended pedigree (grandparents, siblings, children) with per-relative psychiatric / substance / suicide flags
- `Comorbidities` — somatic + psychiatric, extracted from `_mhoccur` / `ceoccur*` flags
- Cohort-specific panels — PANSS + SUMD + AIMS + BARS (SZ), DIVA ADHD + circadian + V1 follow-up (BP), treatment resistance staging + C-SSRS (DR), autism diagnosis + developmental milestones + MCDD + learning disabilities (ASP)

**No column names are hardcoded in Python.** Every `row.get(...)` call routes through `cm: CohortColumnMap` which resolves to values loaded from `config/glossary/{cohort}/column_map.yaml`. Adapting to a future FACE data release where columns get renamed is a pure YAML edit — no Python changes, no redeploy.

### 3. Clinical interpretation layer (`common_extractors.py` + `interpret_score`)

Pure helper functions, all reading constants from YAML:

- `compute_bmi_category(bmi)` → walks `bmi_categories` list
- `detect_metabolic_syndrome(panel)` → IDF/ATP-III criteria from `metabolic_syndrome`
- `compute_framingham_risk(...)` → walks point tables and risk table
- `check_drug_interactions(treatments)` → iterates `drug_interactions` rules with severity prefixes
- `check_medication_lab_alerts(treatments, labs)` → per-drug alert rules with `{liver_name}` placeholder, absolute thresholds (e.g. clozapine < 3.5 × 10⁹/L), and `treatment_flags_any` matching
- `compute_cognitive_z_score(score, test, age)` → reads age-decade norms from `cognitive_norms.{tmt,stroop}.by_decade`
- `interpret_score(instrument, raw)` → looks up the instrument's severity bands (inline or via `severity_thresholds_ref`) and returns a `ScoreInterpretation` with level, label, clinical meaning, and French narrative

### 4. Profile builder layer (`{cohort}_profile_builder.py`)

Each `build_{cohort}_profile(patient_data)` function takes the extracted dataclass tree and produces a `PatientProfile` containing:

- Structured sections (already populated by the extractor)
- A French `synthesis` one-liner (e.g. *"Patient suivie pour Bipolaire de type 2, actuellement euthymique (MADRS = 2)…"*)
- A long `full_vignette` string assembled section-by-section, including demographics, psychiatric history, current DSM criteria, mood/psychotic/cognitive scales with interpretation, cognitive profile, somatic panel (vitals + labs + alerts), treatment, comorbidities, substance use, childhood trauma, family history, and risk factor summary

The vignette composition logic is procedural Python — straightforward string building — and is the place where clinical narrative style is tuned. Vignettes are always in **French**; international instrument abbreviations (MADRS, PANSS, YMRS, …) stay in their canonical form.

### 5. Usage surface

Only pandas + the `face_rlvr.profiles` namespace:

```python
import pandas as pd
from face_rlvr.profiles import (
    extract_bp_patient,  build_bp_profile,
    extract_sz_patient,  build_sz_profile,
    extract_dr_patient,  build_dr_profile,
    extract_asp_patient, build_asp_profile,
)

df = pd.read_csv("data/BP.csv", nrows=1, low_memory=False)
profile = build_bp_profile(extract_bp_patient(df.iloc[0]))
print(profile.synthesis)
print(profile.full_vignette)
```

`notebooks/patient_vignette_explorer.ipynb` runs this end-to-end across all four cohorts, inspects the loaded YAML glossary, compares multiple patients side-by-side, demonstrates that mutating a `column_map.yaml` entry propagates to extraction without any Python edit, and exports a full vignette to Markdown.

## How the Stratification Code Works

The `face_stratification` package implements the transdiagnostic patient stratification pipeline (Stage 1 from the roadmap). It takes `PatientProfile` objects from `face_rlvr.profiles` and produces data-driven patient clusters that cut across DSM diagnostic boundaries.

### 1. Harmonization layer (`harmonization/`)

All feature definitions live in `config/face_stratification/feature_schema.yaml`, validated by Pydantic models in `feature_schema.py`. Per-cohort adapters (`cohort_adapters.py`) turn each `PatientProfile` into a flat dictionary of unified features. `harmonizer.py` assembles these into a `HarmonizedDataset` — a cross-cohort feature matrix with per-feature type annotations (`FeatureType`: continuous, ordinal, binary, categorical) and temporal scope enforcement (`TemporalScope`: V1-only by default).

**Feature architecture**: The schema uses ~215 features across 21 clinical blocks. Features span three resolution levels:
1. **Instrument totals** — single summary scores (MADRS total, PANSS total, CTQ total)
2. **Clinical sub-scales** — the level at which clinicians interpret results, not raw items (CTQ 5 trauma domains, BIS-10 3 impulsivity facets, PANSS Wallwork 5 factors, BDHI 9 hostility dimensions, BRIEF 9 executive T-scores, RBS-R 6 repetitive behavior domains, ADI-R 4 autism domains, SUMD 9 insight items)
3. **Derived composites** — computed from existing data (cognitive domain z-score composites, illness burden index, polypharmacy index, metabolic syndrome flag, waist/height ratio, TMT-B/A executive ratio)

**Block structure**: Features are grouped into domain-specific blocks that each generate their own similarity graph layer. The 21 blocks are: demographics, mood, psychosis, anxiety_impulsivity, hostility_aggression, functioning, sleep_circadian, cognition, biology, treatment, substance, trauma, family_history, comorbidities, suicide_history, psychiatric_history, insight, autism_profile, treatment_resistance, neuropsych, personality. Each block declares its own distance metric (cosine, euclidean, gower) and minimum overlap constraints.

**Adapters** use a `_subscale(scores, instrument_key, subscale_name)` helper that reads `ScoreInterpretation.subscales` dicts populated by the `face_rlvr` extractors. Adding a new subscale feature is a 1-line Python addition + 1 YAML entry — no column parsing logic needed.

Supporting modules:
- `normalization.py` — Winsorize (1st/99th) + robust z-score (median, MAD) with `fit_normalization` / `transform_normalization`; sign-flip for `higher_is_better` features
- `missingness.py` — `characterize_missingness` for coverage analysis, `compute_missingness_mask`, block-level KNN and MICE imputation (optional, never on default path)
- `dsm_subtypes.py` — `extract_dsm_subtypes` for clinical validation against nosological labels

### 2. Graph layer (`graph/`)

Patient similarity is computed with **pairwise-complete masked similarity** — no imputation is performed in the default path. Two patients share an edge only if they have at least `min_shared_features` observed measurements in that block (semantic overlap constraint).

- `masked_similarity.py` — masked cosine, euclidean, Gower, and Manhattan distance kernels
- `transdiagnostic.py` — selects features with above-threshold coverage in every cohort, builds tiered transdiagnostic graphs at multiple coverage thresholds
- `patient_similarity.py` — block kNN graphs, multiplex graph assembly, `build_balanced_knn_graph` (cohort-balanced neighbor sampling), `build_mutual_knn_graph` (reciprocal neighbor filtering)

### 3. Embedding layer (`models/`)

All models implement `BaseEmbeddingModel` and produce `PatientEmbedding` objects. Pure numpy/scipy/sklearn baselines:

- `TransdiagnosticPCA`, `TransdiagnosticUMAP`, `TransdiagnosticRawFeatures` — baseline dimensionality reduction
- `RawFeatureBaseline` — identity embedding (no reduction)
- `TransdiagnosticSpectral`, `MultiplexSpectral` — spectral embeddings from the graph Laplacian
- `ConcatenatedEmbedding` — multi-view factory that concatenates multiple embedding views
- `pipeline.py` — `fit_embedding` / `fit_and_save_embedding` for model persistence

### 4. GNN layer (`stage_b2/`)

Deep graph neural network embeddings implemented in pure PyTorch (no torch-geometric dependency):

- `gcn.py` — sparse GCN layer, adjacency normalization, `get_device` for GPU detection
- `gae.py` — `StageB2GAE` graph autoencoder trained via link prediction (BCE loss)
- `contrastive.py` — `StageB2GraphContrastive` GraphCL-style contrastive SSL with edge-drop and feature-masking augmentations (NT-Xent loss)
- `sweep.py` — hyperparameter sweep with transdiagnostic scoring to select the best GNN config

### 5. Clustering layer (`clustering/`)

- `algorithms.py` — deterministic k-means (`run_kmeans`), k-means sweep, GMM soft clustering (`run_gmm_soft`), bootstrap stability, assignment entropy, boundary patient identification
- `metrics.py` — ARI, NMI, AMI, V-measure, homogeneity, completeness, silhouette, per-cohort entropy
- `k_selection.py` — `run_dual_criterion_k_selection` combining gap statistic with clinical metrics for principled k choice

### 6. Analysis layer (`analysis/`)

Validation, interpretation, and safety checking of discovered clusters:

- `enrichment.py` — per-cluster feature enrichment with Benjamini-Hochberg FDR correction
- `medoids.py` — medoid extraction and French vignette retrieval via `face_rlvr`
- `ablation.py` — global vs per-cohort normalization ablation study
- `meta_stability.py` — `compute_meta_stability` across multiple embedding methods to assess cluster robustness
- `safety_analysis.py` — `run_safety_analysis` checking suicide risk distribution across clusters (no cluster should concentrate high-risk patients without clinical rationale)
- `visualization.py` — t-SNE/UMAP projections, cluster × cohort heatmaps, enrichment bar charts

## Development Conventions

- **Python 3.11+**
- **Pydantic v2** for every config / glossary model
- `from __future__ import annotations` and type hints on every public function
- Logging via `logging.getLogger(__name__)`; no `print` in library code
- All vignettes and clinical narratives in **French**
- **No hardcoded CSV column names or clinical thresholds in Python.** Edit YAML instead.
- Extractor / builder functions must be pure and total: unknown or missing CSV fields become `None`, never raise
- Error handling at the patient level: log, keep going, never crash on a single row

### Coding Guidelines

These behavioral guidelines reduce common LLM coding mistakes (derived from [Karpathy's observations](https://x.com/karpathy/status/2015883857489522876)). Follow them for every code change.

**1. Think Before Coding** — Don't assume. Don't hide confusion. Surface tradeoffs.
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

**2. Simplicity First** — Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

**3. Surgical Changes** — Touch only what you must. Clean up only your own mess.
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.
- Every changed line should trace directly to the user's request.

**4. Goal-Driven Execution** — Define success criteria. Loop until verified.
- Transform tasks into verifiable goals (e.g. "add validation" → "write tests for invalid inputs, then make them pass").
- For multi-step tasks, state a brief plan with verification checks per step.
- Strong success criteria let you loop independently. Weak criteria require clarification — ask for it.

### Plotting

All figures **must** be generated with **Plotly** (`plotly.graph_objects` / `plotly.express`) and exported to static **PNG** via **Kaleido** (`fig.write_image()`). Use the project's standard dark theme:

```python
import plotly.graph_objects as go
import plotly.express as px

DARK = dict(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27")
SCALE = 3  # retina-quality export

# For go.Figure:
fig.update_layout(**DARK)

# For px.scatter / px.bar / etc. (template only — layout props via update_layout):
fig = px.scatter(..., template="plotly_dark")
fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27")

# Save:
fig.write_image("path/to/plot.png", width=1100, height=500, scale=SCALE)
```

- Always save PNGs to disk (never inline-only Plotly HTML).
- Prefer `plotly_dark` template with `paper_bgcolor="#0f1117"` and `plot_bgcolor="#1a1d27"`.
- Use `scale=3` for high-DPI export.
- Heatmap colormaps: `Viridis` (composition), `Plasma` (DSM subtypes), `Reds` (missingness), `Blues` (treatment).

## Key Commands

```bash
# Install in dev mode (pandas, numpy, pyyaml, pydantic)
pip install -e ".[dev]"

# Run the test suite
pytest

# Interactive exploration of all 4 cohorts
jupyter notebook notebooks/patient_vignette_explorer.ipynb

# Smoke-test: build one vignette per cohort, print hash + length
python -c "
import pandas as pd, hashlib
from face_rlvr.profiles import (
    extract_bp_patient,  build_bp_profile,
    extract_sz_patient,  build_sz_profile,
    extract_dr_patient,  build_dr_profile,
    extract_asp_patient, build_asp_profile,
)
for name, path, ex, bld in [
    ('BP',  'data/BP.csv',  extract_bp_patient,  build_bp_profile),
    ('SZ',  'data/SZ.csv',  extract_sz_patient,  build_sz_profile),
    ('DR',  'data/DR.csv',  extract_dr_patient,  build_dr_profile),
    ('ASP', 'data/ASP.csv', extract_asp_patient, build_asp_profile),
]:
    df = pd.read_csv(path, nrows=1, low_memory=False)
    profile = bld(ex(df.iloc[0]))
    md5 = hashlib.md5(profile.full_vignette.encode()).hexdigest()[:12]
    print(f'{name}: {md5} ({len(profile.full_vignette)} chars)')
"

# Build harmonized dataset from all cohorts
python -c "
from face_stratification import build_harmonized_dataset, load_feature_schema
schema = load_feature_schema()
ds = build_harmonized_dataset(schema, data_dir='data/')
print(f'Patients: {ds.X.shape[0]}, Features: {ds.X.shape[1]}')
print(f'Cohorts: {ds.cohort_labels.value_counts().to_dict()}')
"

# Run stratification pipeline (harmonize → graph → embed → cluster)
python -c "
from face_stratification import (
    build_harmonized_dataset, load_feature_schema,
    fit_embedding, run_kmeans, compute_cluster_metrics,
)
from face_stratification.models import TransdiagnosticPCA
schema = load_feature_schema()
ds = build_harmonized_dataset(schema, data_dir='data/')
emb = fit_embedding(TransdiagnosticPCA(n_components=20), ds)
ca = run_kmeans(emb, k=5)
metrics = compute_cluster_metrics(ca, ds.cohort_labels)
print(metrics)
"
```

## Current Limitations

- Real FACE data is mostly single-visit (V0). Only BP currently has a partial follow-up (`_n1` suffix columns), so longitudinal analyses are capped at 2 visits per patient.
- Test coverage is minimal — smoke tests and notebook execution are the main regression safety net.
- `patient_vignette_explorer.ipynb` is the canonical usage example and reference for the public API.

---

## Roadmap — Where This Project Is Going

The current codebase has two implemented stages and one planned stage.

### Stage 0 — Deterministic vignette extraction (complete)

A clean, deterministic, YAML-driven pipeline that turns each FACE CSV row into a rich structured `PatientProfile` and a French clinical vignette. This is the `face_rlvr.profiles` package.

### Stage 1 — From nosology-driven psychiatry to data-driven phenotyping (largely complete)

Traditional psychiatric nosology (DSM-5, ICD-11) groups patients into discrete categories (*bipolar*, *schizophrenia*, *autism*, …) that overlap heavily in biology, cognition, symptoms, treatment response, and trajectory. The FACE cohorts are an unusually rich substrate for questioning those boundaries: the same deep-phenotyping battery (cognitive tests, questionnaires, labs, family history, treatment response) is administered across all four pathologies, so a patient can be represented as a single high-dimensional vector regardless of their nosological label.

**Implemented** (in `face_stratification`):
1. Unified transdiagnostic feature representation from `PatientProfile` dataclasses — ~215 features across 21 clinical blocks, spanning instrument totals, clinical sub-scales, and derived composites, with YAML-driven feature schema.
2. Multi-relational patient similarity graph with pairwise-complete masked similarity (no imputation), semantic overlap constraints, cohort-balanced kNN, mutual kNN, and tiered transdiagnostic edge types.
3. Graph-based representation learning: PCA, UMAP, spectral, multiplex spectral baselines + deep GNN embeddings (Graph Autoencoder, GraphCL contrastive SSL) in pure PyTorch.
4. Clustering: k-means, GMM soft clustering, dual-criterion k selection (gap statistic + clinical metrics), bootstrap stability, consensus clustering.
5. Validation: per-cluster feature enrichment (BH-FDR), medoid vignette retrieval, normalization ablation, meta-stability across methods, safety analysis (suicide risk distribution).

**Current work — Stage A feature engineering expansion:**
The feature schema is being expanded from instrument totals to clinically interpretable sub-scales and derived composites. Key additions:
- Sub-scale expansion: BDHI 9 hostility dimensions (BP), PANSS Wallwork 5-factor model (SZ), BRIEF 9 executive T-scores + RBS-R 6 domains + ADI-R 4 domains (ASP), SUMD 9 insight items (SZ), SCIP 10 cognitive subscales (BP)
- Block restructuring: splitting the monolithic `cohort_specific` block into domain-specific blocks (insight, autism_profile, hostility_aggression, treatment_resistance) with proper similarity thresholds
- Derived composites: cognitive domain z-score composites, illness burden index, polypharmacy index, metabolic syndrome flag, waist/height ratio, age-of-onset categories
- Visual inspection pipeline: coverage heatmaps, inter-feature correlation analysis, distribution ridge plots, t-SNE projections colored by new features
- Validation enhancements: DSM subtype recovery test, leave-one-cohort-out stability, SHAP feature importance per cluster

**Remaining work:**
- Treatment response validation analysis
- Longitudinal trajectory validation (pending multi-visit data)
- Final transdiagnostic phenotype taxonomy publication

### Stage 2 — Precision psychiatry dataset for RLVR LLM training

The second research stage is to build a dataset of **precision psychiatry reasoning tasks** for training large language models with **Reinforcement Learning from Verifiable Rewards (RLVR)**. Unlike RLHF, RLVR does not rely on human preference annotations: each task has a deterministic oracle answer computed from the raw patient data *before* the model generates anything, and the reward signal comes from comparing the model's structured answer to that oracle.

**Target task families** (computed from patient profiles and the data-driven phenotypes from Stage 1):

| Family | Example question | Oracle source |
|--------|-----------------|---------------|
| Metabolic & somatic risk | *Does this patient meet IDF metabolic-syndrome criteria? Compute their 10-year Framingham risk.* | `detect_metabolic_syndrome`, `compute_framingham_risk` |
| Treatment analysis | *Which mood stabilizer has the best expected response profile given cluster membership, current labs, and comorbidities?* | Guideline trees + Stage 1 phenotype clusters |
| Diagnostic reasoning | *Which DSM-5 criteria for a current manic episode are met? List the ones that are NOT met.* | `current_episode_criteria` flags |
| Suicide risk assessment | *Classify suicide risk level; list triggering indicators.* | Weighted algorithm over `SuicideRiskPanel`, **safety-gated** (underestimation zeros the reward) |
| Cognitive assessment | *Compute the TMT-B − TMT-A executive control score, z-score it against age norms, interpret.* | `compute_cognitive_z_score` |
| Longitudinal trajectory | *Did MADRS change by more than the RCI between V0 and V1?* | `compute_rci` + paired visits |
| Rating scale interpretation | *Sum PANSS-P items, map to severity band, produce the French clinical sentence.* | `interpret_score` over YAML thresholds |
| Side-effect & safety | *Given current plasma levels and labs, which medication–lab alerts fire?* | `check_medication_lab_alerts` |
| Transdiagnostic reasoning | *Given two patients from different cohorts, which shares a phenotype cluster with the other despite different labels?* | Stage 1 embeddings |
| Data-quality meta-tasks | *Which panels are incomplete? Flag out-of-range labs.* | `compute_data_completeness`, `detect_floor_ceiling_effects` |

**Training loop (planned):**
- Select a patient and a task template → compute the oracle answer deterministically from the `PatientProfile` (no LLM involved).
- Pseudonymize the profile (French names, age jitter) and render a French vignette as the prompt.
- LLM produces a structured reasoning chain that may include Python code blocks.
- Code blocks execute in a restricted sandbox (whitelisted imports, timeout, no network).
- A verifier compares the extracted JSON answer to the oracle, field by field, with per-field comparison modes (`exact`, `set_equal`, `numeric_tolerance`, `subset`).
- Multi-component reward: correctness (dominant weight), reasoning structure, code executability, safety (hard gate for suicide-risk underestimation).
- GRPO update; retry on low composite reward.

The present codebase already delivers the pieces RLVR needs: deterministic `PatientProfile` construction, clinical interpretation functions that double as oracles, a YAML-versioned source of truth for thresholds, and French vignette rendering. The RLVR stage will add the oracle registry, pseudonymization, sandbox, verifier, reward scorer, and GRPO training loop on top of this foundation.

---

## Summary

- **Stage 0 (complete):** deterministic, YAML-driven extraction of rich French patient vignettes from four FACE cohorts (BP, SZ, DR, ASP).
- **Stage 1 (largely complete):** graph-based representation learning discovering data-driven psychiatric phenotypes that cut across DSM boundaries — harmonization, multiplex similarity graphs, GNN embeddings, clustering with clinical validation.
- **Stage 2 (planned):** an RLVR dataset of verifiable precision-psychiatry reasoning tasks, grounded in those phenotypes, for training clinical reasoning LLMs without human preference labels.
