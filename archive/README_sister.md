# FACE-RLVR: Verifiable Clinical Reasoning Dataset from the FACE Psychiatric Cohort

FACE-RLVR generates 1,000 structured clinical reasoning tasks from the [FACE](https://www.fondation-fondamental.org/) (FondaMental Advanced Centers of Expertise) psychiatric cohort for training precision psychiatry LLMs with GRPO-based reinforcement learning.

Every task follows the **oracle-first principle**: ground truth is computed deterministically from real patient data before the LLM generates any output, enabling binary and multi-component reward signals without human annotation.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Interactive Notebook](#interactive-notebook)
- [Project Structure](#project-structure)
- [CLI Reference](#cli-reference)
- [Pipeline Overview](#pipeline-overview)
- [Data Sources](#data-sources)
- [Task Categories](#task-categories)
- [Oracle Functions](#oracle-functions)
- [Reward Scoring](#reward-scoring)
- [Code Sandbox](#code-sandbox)
- [Configuration](#configuration)
- [Modules Reference](#modules-reference)
- [TODOs](#todos)

---

## Quick Start

### Prerequisites

- Python 3.11 or higher
- (Optional) `ANTHROPIC_API_KEY` environment variable for full dataset generation with Claude Opus 4.6 / Sonnet 4.6
- (Optional) `OPENAI_API_KEY` for GPT-5.4 provider

### Installation

```bash
# Clone and install
cd psych-dataset
pip install -e ".[dev]"

# Copy and fill in API keys (needed only for full generation, not dry runs)
cp .env.example .env
# Edit .env with your API keys
```

### First Run

```bash
# Smoke test: load one patient from each cohort and render the vignette
python -c "
import pandas as pd, hashlib
from face_rlvr.profiles import (
    extract_bp_patient, build_bp_profile,
    extract_sz_patient, build_sz_profile,
    extract_dr_patient, build_dr_profile,
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
    print(f'{name}: {len(profile.full_vignette)} chars')
"
```

---

## Interactive Notebook

Explore patient profiles end-to-end for any cohort:

```bash
cd notebooks
jupyter notebook patient_vignette_explorer.ipynb
```

**[`notebooks/patient_vignette_explorer.ipynb`](notebooks/patient_vignette_explorer.ipynb)** demonstrates:

- Loading a patient row from any of the 4 cohorts (BP, SZ, DR, ASP)
- Running the YAML-driven extractor to build a structured ``*PatientData`` dataclass
- Building the full French clinical vignette via the profile builder
- Inspecting individual sections (synthesis, mood, biology, treatment, ...)
- Querying the loaded YAML glossary (instruments, lab ranges, clinical constants)
- Comparing multiple patients side-by-side
- Exporting a vignette to markdown for clinical review

No LLM API keys are needed — all extraction and rendering run locally on the real FACE CSVs.

---

## Project Structure

```
psych-dataset/
├── config/
│   ├── pipeline.yaml                 # Task counts, cohort proportions, LLM settings
│   ├── reward_weights.yaml           # Per-category reward component weights
│   └── reference_tables/             # (placeholder) clinical reference data
├── data/                             # FACE cohort CSV files
│   ├── BP.csv                        # Bipolar disorder (~5,400 patients)
│   ├── SZ.csv                        # Schizophrenia (~2,200 patients)
│   ├── DR.csv                        # Treatment-resistant depression (~350 patients)
│   └── ASP.csv                       # Autism spectrum disorder (~1,300 patients)
├── notebooks/
│   └── patient_vignette_explorer.ipynb  # Load and explore patient profiles per cohort
├── config/
│   ├── pipeline.yaml
│   ├── reward_weights.yaml
│   └── glossary/                     # YAML clinical glossary (instruments, labs, codes,
│       │                             # column maps, BMI/metabolic/Framingham/drug/
│       │                             # med-alert/cognitive constants). Edit YAML, not Python.
│       ├── common/                   # instruments, thresholds, clinical_constants, codes
│       ├── bp/                       # BP-specific instruments + lab ranges + column_map
│       ├── sz/                       # SZ-specific instruments + lab ranges + column_map
│       ├── dr/                       # DR-specific + column_map + categorical_codes
│       └── asp/                      # ASP-specific + column_map + categorical_codes
├── docs/
│   ├── face_rl_task_guide_v2.md      # Detailed task specs with examples per category
│   └── FACE_RLVR_Dataset_Specification.docx
├── src/face_rlvr/
│   ├── config.py                     # Pydantic config models (pipeline, reward weights)
│   ├── oracles/                      # 37 oracle functions across 10 categories
│   ├── pipeline/                     # 8-step generation orchestrator
│   ├── profiles/                     # Per-cohort extractors + profile builders + YAML loader
│   ├── pseudonymization/             # French name pools + demographics
│   ├── reward/                       # Multi-component reward scorer
│   ├── output/                       # TaskRecord schema + JSONL writer
│   └── templates/                    # (placeholder) Jinja2 templates
├── tests/                            # (placeholder) test suite
├── scripts/                          # (placeholder) utility scripts
├── output/                           # Generated datasets (gitignored)
├── pyproject.toml
├── .env.example
└── CLAUDE.md
```

---

## Pipeline Overview

The dataset generation pipeline processes each task through 8 sequential steps:

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: PATIENT SELECTION                                      │
│  Stratified sampling by cohort + difficulty. Checks data        │
│  completeness against oracle's required_variables (≥80%         │
│  for easy/medium, ≥60% for hard). Tracks used patients          │
│  to maximize diversity.                                         │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: ORACLE COMPUTATION                                     │
│  Run the deterministic oracle function on raw patient data.     │
│  Produces ground_truth dict, verifiable_fields, and             │
│  comparison_modes. This happens BEFORE any LLM call.            │
├─────────────────────────────────────────────────────────────────┤
│  Step 3: PSEUDONYMIZATION & VIGNETTE                            │
│  Replace patient identifiers with synthetic French              │
│  demographics (deterministic: same patient+seed = same          │
│  pseudonym). Render the clinical vignette in French.            │
├──────────────────────── DRY RUN STOPS HERE ─────────────────────┤
│  Step 4: SYSTEM PROMPT ASSEMBLY                                 │
│  Select modality-specific prompt (code_required,                │
│  code_preferred, text_with_differential, text_with_safety).     │
│  Inject output schema expectations.                             │
├─────────────────────────────────────────────────────────────────┤
│  Step 5: LLM GENERATION                                        │
│  Call Claude Sonnet 4 (or GPT-4o) with system + user prompt.    │
│  Receive structured reasoning with code blocks.                 │
├─────────────────────────────────────────────────────────────────┤
│  Step 6: CODE EXECUTION                                         │
│  Extract Python code blocks. Execute each in sandboxed          │
│  subprocess (5s timeout, restricted imports, no network).       │
├─────────────────────────────────────────────────────────────────┤
│  Step 7: VERIFICATION                                           │
│  Parse JSON answer block from LLM response. Extract             │
│  reasoning step types. Compare answer to oracle ground truth.   │
│  Compute multi-component reward score.                          │
├─────────────────────────────────────────────────────────────────┤
│  Step 8: QUALITY FILTER & RETRY                                 │
│  If composite reward < 0.8: retry (up to 3×) with failure       │
│  hint appended to prompt. If still < 0.6 after retries:         │
│  flag for manual review.                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

### FACE Cohort CSV Files

The FACE database uses **wide format**: one row per patient, all assessments as columns. Follow-up data (V1) is encoded with a `_n1` column suffix.

| File | Cohort Code | Patients | Columns | Patient ID Column |
|------|-------------|----------|---------|-------------------|
| `BP.csv` | BP | ~5,400 | 2,229 | `usubjid_patients` |
| `SZ.csv` | SZ | ~2,200 | 1,713 | `usubjid_patients` |
| `DR.csv` | DR | ~350 | 2,217 | `usubjid_patients` |
| `ASP.csv` | TSASDI | ~1,300 | 4,326 | `usubjid_identite` |

**Column naming varies across cohorts.** Every CSV column that the extractors read from is declared in ``config/glossary/{cohort}/column_map.yaml``. To adapt to a CSV rename in a future FACE release, edit the YAML file only — no Python changes required.

### Clinical Glossary (`config/glossary/`)

All instrument definitions, lab reference ranges, categorical codes, severity thresholds, and clinical computation constants (BMI, metabolic syndrome, Framingham CV risk, drug-drug interactions, medication-lab monitoring alerts, cognitive norms) live in YAML files under `config/glossary/`, validated by Pydantic v2 models in `src/face_rlvr/profiles/glossary_schema.py`.

Each cohort (BP, SZ, DR, ASP) has its own `instruments.yaml`, `lab_ranges.yaml`, and (where needed) `categorical_codes.yaml`. Shared instruments used by 2+ cohorts live in `config/glossary/common/instruments.yaml` and can be overridden per cohort (e.g., SZ redefines `CGI-S` to use `cgi0101` instead of `cgi01`).

To edit any clinical data — change a column name, update a severity threshold, add a new French label — edit the YAML files only. No Python changes needed. See `config/glossary/README.md` for the contributor guide.

---

## Task Categories

1,000 tasks distributed across 10 categories, each targeting a distinct clinical reasoning skill:

| # | Category | Tasks | Sub-types | Code Modality | Oracle Basis |
|---|----------|-------|-----------|---------------|-------------|
| 1 | Metabolic & Somatic Risk | 120 | 1a–1e | code_required | IDF/ATP-III criteria, lab reference ranges, Framingham risk score, CKD-EPI |
| 2 | Treatment Analysis | 160 | 2a–2f | code_preferred / text | Therapeutic dose ranges, Thase-Rush staging, CANMAT/NICE guidelines, drug interaction database |
| 3 | Diagnostic Reasoning | 130 | 3a–3e | text + differential | FACE diagnosis, DSM-5 criteria, differential evidence weighting |
| 4 | Suicide Risk Assessment | 80 | 4a–4c | text + safety_check | Weighted static/dynamic/protective factor algorithm (safety-gated) |
| 5 | Cognitive Assessment | 100 | 5a–5d | code_required | Age-stratified norm tables, z-score conversion, Reliable Change Index |
| 6 | Longitudinal Trajectory | 120 | 6a–6e | code_required | RCI, treatment response classification, episode counting, GAF trends |
| 7 | Rating Scale Interpretation | 80 | 7a–7c | code_required | Scale scoring arithmetic, severity lookup tables, discrepancy analysis |
| 8 | Side Effect & Safety | 80 | 8a–8d | code_preferred / text | Medication side-effect profiles, AIMS/Barnes scores, monitoring protocols |
| 9 | Transdiagnostic Reasoning | 60 | 9a–9c | code_required / text | Cross-cohort comparison on shared instruments, boundary reasoning |
| 10 | Data Quality Meta-Tasks | 70 | 10a–10c | code_preferred | Completeness rules, plausible value ranges, timeline consistency |

**Difficulty distribution**: 30% easy, 40% medium, 30% hard.

**Cohort distribution**: ~35% BP, ~25% SZ, ~20% DR, ~20% TSASDI.

---

## Oracle Functions

Every task sub-type maps to a deterministic oracle function. Oracles are pure functions: `(PatientRecord, visit_index, difficulty) → OracleResult`. The `OracleRegistry` maps sub-type codes to oracle instances.

### Complete Oracle Registry (37 functions)

| Sub-type | Oracle Class | Category | Code Modality |
|----------|-------------|----------|---------------|
| **1a** | `MetabolicSyndromeOracle` | 1 — Metabolic | code_required |
| **1b** | `LabPanelOracle` | 1 — Metabolic | code_required |
| **1c** | `CardiovascularRiskOracle` | 1 — Metabolic | code_required |
| **1d** | `IatrogenicChangeOracle` | 1 — Metabolic | code_required |
| **1e** | `MonitoringProtocolOracle` | 1 — Metabolic | code_preferred |
| **2a** | `TreatmentAdequacyOracle` | 2 — Treatment | code_preferred |
| **2b** | `TreatmentResistanceOracle` | 2 — Treatment | code_required |
| **2c** | `GuidelineNextStepOracle` | 2 — Treatment | code_preferred |
| **2d** | `DrugInteractionOracle` | 2 — Treatment | code_required |
| **2e** | `DoseOptimizationOracle` | 2 — Treatment | code_required |
| **2f** | `ECTCandidacyOracle` | 2 — Treatment | text_with_structured_differential |
| **3a** | `PrimaryDiagnosisOracle` | 3 — Diagnostic | text_with_structured_differential |
| **3b** | `DifferentialDiagnosisOracle` | 3 — Diagnostic | text_with_structured_differential |
| **3c** | `DSM5SpecifierOracle` | 3 — Diagnostic | code_preferred |
| **3d** | `ComorbidityOracle` | 3 — Diagnostic | code_preferred |
| **3e** | `DiagnosticRevisionOracle` | 3 — Diagnostic | text_with_structured_differential |
| **4a** | `SuicideRiskOracle` | 4 — Suicide Risk | text_with_safety_check |
| **4b** | `RiskChangeOracle` | 4 — Suicide Risk | code_required |
| **4c** | `SafetyPlanOracle` | 4 — Suicide Risk | text_with_safety_check |
| **5a** | `NeuropsychProfileOracle` | 5 — Cognitive | code_required |
| **5b** | `CognitiveTrajectoryOracle` | 5 — Cognitive | code_required |
| **5c** | `CognitiveFunctionalDiscrepancyOracle` | 5 — Cognitive | code_required |
| **5d** | `CognitivePatternOracle` | 5 — Cognitive | code_preferred |
| **6a** | `ClinicallySignificantChangeOracle` | 6 — Longitudinal | code_required |
| **6b** | `TreatmentResponseOracle` | 6 — Longitudinal | code_required |
| **6c** | `EpisodeTrajectoryOracle` | 6 — Longitudinal | code_required |
| **6d** | `RelapseDetectionOracle` | 6 — Longitudinal | code_required |
| **6e** | `FunctioningSymptomTrajectoryOracle` | 6 — Longitudinal | code_required |
| **7a** | `SingleScaleScoringOracle` | 7 — Scales | code_required |
| **7b** | `MultiScaleIntegrationOracle` | 7 — Scales | code_required |
| **7c** | `DiscrepancyAnalysisOracle` | 7 — Scales | code_required |
| **8a** | `SideEffectAttributionOracle` | 8 — Side Effects | code_preferred |
| **8b** | `MovementDisorderOracle` | 8 — Side Effects | code_required |
| **8c** | `SideEffectMonitoringOracle` | 8 — Side Effects | code_required |
| **8d** | `RiskBenefitOracle` | 8 — Side Effects | text_with_structured_differential |
| **9a** | `CrossCohortComparisonOracle` | 9 — Transdiagnostic | code_required |
| **9b** | `DiagnosticBoundaryOracle` | 9 — Transdiagnostic | text_with_structured_differential |
| **9c** | `TransdiagnosticFactorOracle` | 9 — Transdiagnostic | code_preferred |
| **10a** | `MissingDataImpactOracle` | 10 — Data Quality | code_preferred |
| **10b** | `DataPlausibilityOracle` | 10 — Data Quality | code_required |
| **10c** | `TimelineReconstructionOracle` | 10 — Data Quality | code_required |

### OracleResult Structure

Every oracle returns an `OracleResult` with:

- `ground_truth` — deterministic answer dictionary
- `verifiable_fields` — which keys in ground_truth are scored
- `comparison_mode` — per-field: `"exact"`, `"set_equal"`, `"numeric_tolerance"`, `"subset"`, `"ordered_list"`
- `tolerance` — per-field numeric tolerance (for `numeric_tolerance` mode)
- `intermediate_values` — internal computations for code block verification
- `metadata` — extra info passed to vignette/prompt generation

---

## Reward Scoring

Five components combined into a weighted composite:

| Component | Default Weight | What It Measures |
|-----------|---------------|-----------------|
| **Correctness** | 0.40 | Per-field match against oracle ground truth (using comparison modes) |
| **Format** | 0.15 | Presence of structured reasoning steps (`extract`, `compute`, `criterion`, `lookup`, `integrate`) and JSON answer block |
| **Code** | 0.20 | Fraction of extracted Python code blocks that execute without error |
| **Reasoning** | 0.15 | Step type diversity and logical ordering (e.g., `extract` before `compute`) |
| **Safety** | 0.10 | Risk level accuracy; returns 1.0 for all categories except Category 4 |

### Category-Specific Overrides

| Category | Override | Rationale |
|----------|---------|-----------|
| 3 (Diagnostic) | reasoning = 0.25 | Diagnostic integration is the primary skill |
| 4 (Suicide Risk) | safety = 0.30, safety_gate = true | Underestimation penalized 3x; severe underestimation zeros entire reward |
| 7 (Rating Scales) | code = 0.25, format = 0.20 | Format bootstrapping anchor for Stage 1 RL |
| 9 (Transdiagnostic) | reasoning = 0.30 | Highest-difficulty cross-cohort integration |

### Safety Gate (Category 4 Only)

- Risk levels ranked: low (0) → moderate (1) → high (2) → imminent (3)
- Overestimation: mild penalty (0.15 per level)
- Underestimation: 3× penalty (0.45 per level)
- **Severe underestimation** (≥2 levels, e.g., "low" when oracle says "high"): **composite reward = 0.0** (entire reward zeroed, gradient vanishes for this example)

---

## Code Sandbox

LLM-generated Python code blocks are extracted and executed in isolated subprocesses.

**Allowed imports:**
```
math, statistics, numpy, pandas, scipy, scipy.stats,
collections, itertools, functools, json, re, datetime
```

**Security model:**
- Custom `__import__` hook blocks all unlisted modules
- `os.access()` disabled
- Subprocess environment: `PATH=""`, `HOME="/tmp"`
- 5-second timeout per code block
- 256 MB memory limit (configurable)
- Return values captured via `__SANDBOX_RESULT__` stdout marker

---

## Configuration

### `config/pipeline.yaml`

Controls the full generation pipeline:

```yaml
seed: 42                          # Global RNG seed for reproducibility
total_tasks: 1000

cohort_proportions:               # Target distribution across cohorts
  BP: 0.35
  SZ: 0.25
  DR: 0.20
  TSASDI: 0.20

difficulty_distribution:          # Per-category difficulty split
  easy: 0.30
  medium: 0.40
  hard: 0.30

categories:                       # 10 categories with sub-type definitions
  1:
    name: "Metabolic & Somatic Risk Assessment"
    count: 120
    subtypes:
      1a: {count: 30, code_modality: code_required, name: "Metabolic syndrome diagnosis"}
      # ... (see file for full spec)

llm:
  provider: anthropic             # "anthropic" or "openai"
  model: claude-sonnet-4-20250514
  max_tokens: 4096
  temperature: 0.7

sandbox:
  timeout_seconds: 5
  memory_limit_mb: 256
  allowed_imports: [math, statistics, numpy, pandas, ...]

quality:
  min_composite_reward: 0.8       # Retry threshold
  max_retries: 3
  flag_for_review_threshold: 0.6  # Flag for human review

data:
  store_type: csv                 # "csv" for real FACE data, "synthetic" for dev
  csv_path: data/                 # Directory containing BP/SZ/DR/ASP.csv
```

### `config/reward_weights.yaml`

Defines default and per-category reward component weights. See [Reward Scoring](#reward-scoring).

---

## Modules Reference

### `src/face_rlvr/config.py`

Pydantic v2 models for all configuration. Key classes: `PipelineConfig`, `CategoryConfig`, `SubtypeConfig`, `LLMConfig`, `SandboxConfig`, `QualityConfig`, `DataConfig`, `RewardWeights`, `RewardConfig`. Loading functions: `load_pipeline_config(path)`, `load_reward_config(path)`.

### `src/face_rlvr/oracles/`

| Module | Oracle Classes | Category |
|--------|---------------|----------|
| `base.py` | `OracleFunction` (ABC), `OracleResult`, `OracleRegistry` | — |
| `metabolic.py` | MetS IDF, lab panel, Framingham, iatrogenic change, monitoring | 1 |
| `treatment.py` | Adequacy, Thase-Rush staging, guidelines, interactions, dose, ECT | 2 |
| `diagnostic.py` | Primary diagnosis, differential, DSM-5 specifiers, comorbidity, revision | 3 |
| `suicide.py` | Risk stratification (weighted algorithm), risk change, safety plan | 4 |
| `cognitive.py` | Neuropsych profile (z-scores), cognitive trajectory (RCI), discrepancy, pattern | 5 |
| `longitudinal.py` | Clinically significant change, treatment response, episode trajectory, relapse, functioning | 6 |
| `scales.py` | Single-scale scoring, multi-scale integration, self-report vs clinician discrepancy | 7 |
| `side_effects.py` | Attribution, movement disorders (AIMS/Barnes), monitoring, risk-benefit | 8 |
| `transdiagnostic.py` | Cross-cohort comparison, diagnostic boundary, transdiagnostic factors | 9 |
| `data_quality.py` | Missing data impact, plausibility detection, timeline reconstruction | 10 |

### `src/face_rlvr/pipeline/`

| Module | Pipeline Step | Purpose |
|--------|-------------|---------|
| `selector.py` | Step 1 | `plan_tasks()` creates 1,000 task specs with cohort/difficulty assignments; `select_patient()` finds eligible patients |
| `oracle.py` | Step 2 | `build_oracle_registry()` registers all 37 oracle functions |
| `vignette.py` | Step 3 | `generate_vignette()` renders French clinical vignettes with pseudonymized demographics |
| `prompt.py` | Step 4 | `assemble_system_prompt()` selects modality-specific French system prompt; `assemble_user_prompt()` combines vignette + question |
| `llm.py` | Step 5 | `AnthropicProvider` and `OpenAIProvider` implementing `LLMProvider` Protocol; `create_provider()` factory |
| `sandbox.py` | Step 6 | `extract_code_blocks()` and `execute_code_block()` — isolated subprocess execution with import restrictions |
| `verifier.py` | Step 7 | `extract_json_answer()` parses LLM JSON output; `extract_reasoning_steps()` extracts `### [TYPE]` structured steps |
| `quality.py` | Step 8 | `should_retry()`, `should_flag_for_review()`, `retry_hint()` for quality gating |
| `runner.py` | All | `PipelineRunner` — main orchestrator coordinating all 8 steps with retry loop |

### `src/face_rlvr/reward/`

| Module | Purpose |
|--------|---------|
| `scorer.py` | `compute_composite_reward()` — calls 5 scoring functions: `score_correctness()`, `score_format()`, `score_code()`, `score_reasoning()`, `score_safety()`. Applies safety gate for Category 4. |

### `src/face_rlvr/pseudonymization/`

| Module | Purpose |
|--------|---------|
| `names.py` | French first name pools (83 male, 70 female) and last name pool (105 names) |
| `demographics.py` | `pseudonymize()` — deterministic pseudonym generation (same patient+seed → same name); age jitter ±2 years; site anonymization |

### `src/face_rlvr/output/`

| Module | Purpose |
|--------|---------|
| `schema.py` | Pydantic models: `TaskRecord`, `ReasoningStep`, `CodeExecutionResult`, `RewardComponents`, `GenerationMetadata` |
| `writer.py` | `write_dataset()` — JSONL serialization; `compute_stats()` — summary statistics; `print_stats()` — formatted console output |

### `src/face_rlvr/profiles/`

| Module | Purpose |
|--------|---------|
| `common_instruments.py` | Core `InstrumentDefinition`, `SeverityLevel`, `ScoreInterpretation` dataclasses + `interpret_score()` |
| `common_extractors.py` | Shared dataclasses, helpers, shared sub-extractors, and clinical utility functions (BMI, metabolic syndrome, Framingham, drug interactions, cognitive z-scores) — all data-driven from YAML |
| `glossary_schema.py` | Pydantic v2 models validating every YAML file under `config/glossary/` |
| `glossary_loader.py` | Loads and caches the YAML glossary; converts Pydantic models to dataclasses for downstream consumers |
| `{bp,sz,dr,asp}_instruments.py` | Thin shims that load the cohort's instrument registry via the glossary loader |
| `{bp,sz,dr,asp}_extractor.py` | Per-cohort extraction from raw CSV rows into structured dataclasses |
| `{bp,sz,dr,asp}_profile_builder.py` | Per-cohort French vignette rendering |

---

## TODOs

### High Priority

- [ ] **Unit tests for all oracle functions.** Each oracle should be tested with hand-crafted patient data and expected ground truth matching the examples in `docs/face_rl_task_guide_v2.md`. This is the single most important quality assurance step. Currently `tests/` contains only placeholder `__init__.py` files.

- [ ] **Cross-cohort task patient pairing (Category 9a).** `CrossCohortComparisonOracle` requires a `patient_b` parameter for the second patient. The `PipelineRunner._process_task()` does not handle this — it only selects one patient. Needs special logic to select two patients from different cohorts.

- [ ] **Expand oracle column-map coverage.** Many oracle functions reference variables (e.g., `SUICIDE_ATTEMPT_HISTORY`, `CTQ_TOTAL`, `ALCOHOL_INCREASED`) that are not yet declared in any `config/glossary/{cohort}/column_map.yaml`. These unmapped variables always resolve to `None`, causing reduced patient eligibility and oracle failures. Add entries to the column maps to close the gap.

### Medium Priority

- [ ] **Richer vignette templates per category.** Categories 3–10 currently use the generic `_build_generic_vignette()` function in `pipeline/vignette.py`. Category-specific templates should extract and display the clinically relevant variables for each task type (e.g., medication history for Cat 2, neuropsych raw scores for Cat 5, longitudinal score tables for Cat 6).

- [ ] **Longitudinal data support.** Most FACE patients have only V0 (baseline) data. The `_n1` suffix columns provide V1 for some patients in BP, but longitudinal tasks (Cat 6) requiring ≥4 visits fail with synthetic data and may have limited real-data coverage. Investigate which patients have genuine multi-visit data and adjust task counts accordingly.

- [ ] **Jinja2 templates.** The `src/face_rlvr/templates/` and `templates/system_prompts/` directories are empty placeholders. Move the vignette generation logic from procedural Python functions to Jinja2 templates for easier editing by clinicians who don't write Python.

- [ ] **Norm table validation.** Neuropsych norm values in `cognitive.py` (`WAIS_NORMS`, `TMT_NORMS`, `FLUENCY_NORMS`, `CVLT_NORMS`) are simplified age-bracket approximations. These should be validated against published normative manuals and expanded to include education-level stratification.

- [ ] **Diagnostic oracle enrichment.** `PrimaryDiagnosisOracle` (3a) currently returns only the patient's FACE diagnosis as ground truth. It should also generate the supporting clinical evidence from the patient's data (symptom scores, family history, course of illness) so the reward function can verify evidence quality, not just the final diagnosis label.

- [ ] **Comorbidity oracle mapping to real data.** `ComorbidityOracle` (3d) references MINI variables (`MINI_MDE_CURRENT`, `MINI_ANXIETY_GAD`, etc.) that need to be mapped to actual FACE DR CSV column names.

### Lower Priority

- [ ] **CI/CD pipeline.** No GitHub Actions, Makefile, or Docker configuration exists. Add a basic CI with linting (ruff), type checking (mypy), and pytest.

- [ ] **Framingham risk score accuracy.** `CardiovascularRiskOracle` (1c) uses a simplified point-based estimation, not the full Wilson 1998 log-linear model. Replace with the published Framingham equations for clinical accuracy.

- [ ] **Drug interaction database expansion.** `DRUG_INTERACTIONS` in `treatment.py` covers ~10 clinically significant pairs. Expand to cover the full polypharmacy landscape relevant to psychiatric patients (benzodiazepines, antidepressant combinations, CYP interactions).

- [ ] **SCORE2 cardiovascular risk.** The European SCORE2 algorithm is referenced in the project spec but not implemented — only Framingham is currently computed. Add SCORE2 as an alternative for European population-relevant risk estimation.

- [ ] **Dataset versioning and provenance.** Add metadata to the output JSONL recording the pipeline version, config hash, data file checksums, and generation timestamp for reproducibility tracking.

- [ ] **Evaluation harness.** Build a script that loads a generated dataset and runs a target LLM against it, computing reward distributions per category. This is the downstream consumer of the dataset for GRPO training.

---

## Output Format

Each line in the output JSONL file is a `TaskRecord` with this structure:

```json
{
  "task_id": "face_rlvr_0001",
  "category": 1,
  "category_name": "Metabolic & Somatic Risk Assessment",
  "subtype": "1b",
  "subtype_name": "Lab panel interpretation",
  "difficulty": "medium",
  "code_modality": "code_required",
  "cohort": "BP",
  "patient_id_synthetic": "P_04821",
  "vignette": "Patient: Jean-Marc D., 48 ans, suivi au Centre Expert...",
  "question": "Identifiez toutes les valeurs anormales...",
  "system_prompt": "Tu es un psychiatre expert...",
  "oracle_ground_truth": {"abnormal_values": {...}, "metabolic_syndrome_idf": {...}},
  "oracle_verifiable_fields": ["abnormal_values", "metabolic_syndrome_idf"],
  "oracle_comparison_modes": {"abnormal_values": "subset", "metabolic_syndrome_idf": "subset"},
  "llm_response": "### [extract] Extraction des valeurs...",
  "extracted_answer": {"abnormal_values": {...}},
  "code_blocks": ["import numpy as np\n..."],
  "code_execution_results": [{"block_index": 0, "success": true, ...}],
  "reasoning_steps": [{"step_type": "extract", "content": "..."}],
  "reward": {"correctness": 0.85, "format": 0.90, "code": 1.0, "reasoning": 0.75, "safety": 1.0, "composite": 0.87},
  "metadata": {"llm_provider": "anthropic", "llm_model": "claude-sonnet-4-20250514", "retries": 0, ...}
}
```

In dry-run mode, `llm_response`, `extracted_answer`, `code_blocks`, and `code_execution_results` are empty, and all reward scores are 0.0.

---

## License

Internal research use only. The FACE database contains confidential clinical research data from the Fondation FondaMental.
