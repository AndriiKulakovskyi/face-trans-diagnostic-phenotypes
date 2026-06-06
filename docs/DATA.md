# FACE Common-Variables Dictionary + V3 data contract — Reading & Loading Guide

The harmonized data dictionary for the FACE 3-cohort data — Bipolar (BP), Schizophrenia (SZ),
Depression (DR) — and the **V3 data contract** that extends it for patient-level, missingness-aware,
mixed-likelihood modeling. One row per harmonized variable maps it to its per-cohort source column and
records how to harmonize + sanity-check it. **The package loads it for you** — this guide explains the
columns and what V3 adds.

> V3 builds on a harmonization and **no-naive-imputation** foundation and adds modeling metadata (see
> "V3 data contract" below).

## Files
- `data/face-common-vars.xlsx` — the harmonized dictionary (Sheet1; 225 rows, **201 usable**, 16 columns)
- `data/{bipolar,schizophrenia,depression}.csv` — visit-level data (confidential, gitignored)
- `data/site_lookup.csv` — fondacode → site lookup (for `siteid_city`)
- `data/thesaurus/` — per-cohort source dictionaries (reference)

## Columns (current dictionary)
- **Section** — clinical section (AUTO-QUESTIONNAIRES, BILAN BIOLOGIQUE, NEUROPSYCHOLOGIE, …).
- **Label**, **Codage** — human label + summary coding.
- **Why retained / Findings / Rule** — clinical rationale, cross-cohort caveats, harmonization rule.
- **BP / SZ / DR column in CSV** — the per-cohort source column (blank = absent in that cohort).
- **Final dtype** — `float` · `int8 binary` · `int8 categorical` · `int8 ordinal` · `string` · `date`.
- **Final unit / value set** — target unit or value set.
- **Sanity min check / Sanity max check** — per-variable plausibility bounds; out-of-range → NaN
  (never clipped, never imputed).
- **Coverage (cohorts present)** — which cohorts contribute data.
- **Cluster readiness** — READY (3-cohort) / PARTIAL (2-cohort) / NOT USABLE (excluded). The loader
  keeps **READY + PARTIAL** (199 vars).
- **Canonical name (merged single-cohort)** — the harmonized variable name (the modelled feature).

## How the package loads it (don't hand-merge)
```python
from v3.data import build_unified_dataframe, load_variables, to_harmonized_dataset
df = build_unified_dataframe("data", "data/face-common-vars.xlsx",
                             readiness=["READY", "PARTIAL"], format="long")
ds = to_harmonized_dataset(df, load_variables("data/face-common-vars.xlsx"), visit="V0")
```
`build_unified_dataframe` reads each per-cohort source column, applies the harmonization rule
(`rules.py`), then the sanity bounds, and concatenates the cohorts. **NaN = missing, never imputed.**

---

## V3 data contract (what V3 adds)

V3 depends on explicit, machine-readable modeling assumptions — they must live in the dictionary, not
only in code or prose. The contract is two schemas + config files (specified in
[`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md) §2–§3).

**Patient-level baseline schema** (one row per patient): `patient_id · cohort(BP/SZ/DR) · site ·
baseline_date · age · sex · education · diagnosis/DSM-arm · V0 variables · follow-up outcome
availability`. Plus a **long-format observed-cell table** for modeling (one row per observed
`(patient, variable)`), so missing cells are simply absent rather than filled.

**Per-variable modeling metadata** to add to each dictionary row:

| Field | Purpose |
|---|---|
| `likelihood_family` | Gaussian · Student-t · lognormal · ordered-logit/probit · Bernoulli · neg-binomial · ZINB — the observation likelihood that **carries the variable type** |
| `missingness_type` | Structural · Design · Clinical-skip · Sporadic · Informative · Outcome-related (Phase B) |
| `structural_zero_rule` | the deterministic skip-logic decode, if any (e.g. `attempt_count = 0` when `attempt_ever = 0`) |
| `candidate_dimensions` + `primary_expected_dimension` + `plausible_cross_loadings` | the **soft prior loading** of this variable on the 10-candidate ontology (priors, *not* hand-tagged scores) |
| `higher_score_meaning` | reverse-code so **higher = more burden/dysfunction** unless documented (GAF/EGF, EQ-5D VAS, HDL…) |
| `covariate_status` / `outcome_status` | whether the variable is a measurement covariate or a (future) outcome — **outcomes never enter the baseline dimension model** |
| `use_in_core_model` / `use_in_extension_model` | measurement-eligibility tier (Core all-cohort · Partial extension · Diagnosis-specific module · Covariate · Outcome · Excluded) |

Config artifacts: `data_dictionary_v3.csv`, `variable_schema_v3.yaml`, `likelihood_map_v3.yaml`,
`construct_prior_map_v3.yaml`, `soft_loading_prior_matrix.{csv,yaml}`.

### Encoding for V3 (the observation likelihood carries the type)
V3 does **not** force every variable onto one shared pseudo-continuous metric. Deterministic scaling
is kept only where useful (e.g. standardizing approximately-continuous scores); skewed labs are
`log`-transformed; ordinal/binary/count variables keep their nature and get an ordinal/Bernoulli/count
likelihood.

### Diagnosis is a covariate / validation target — never a clustering feature
Keep `arm` (DSM-5 subtype) and `cohort` as **labels** and **measurement-model covariates**. They are
used to *validate* recovered dimensions/strata (η², ARI, per-group composition, confounding, invariance)
and to adjust indicator means — **never** as dimension indicators or clustering inputs.
**Identifiers never modelled on:** `usubjid_patients`, `cohort`, `arm`, `visit`, `visitnum`,
`siteid_city` (kept loadable for site stratification, excluded from features via
`ADMINISTRATIVE_FEATURES`).

---

## Notes
- **Cognition (NEUROPSYCHOLOGIE) is available in all 3 cohorts** (curated in
  `docs/neuropsy_features.yaml`: WAIS standard scores + TMT + verbal memory/fluency).
- `siteid_city` is kept loadable (for site stratification) but excluded from the feature matrix.
- Within-column unit mixing (`mchc` g/L vs g/dL; `hct` % vs L/L) is harmonized by `rules.py`.
- **QA**: the data-layer tests (`tests/v3/`) validate that every variable loads + passes sanity bounds and
  that skip-logic structural-zero decoding is correct — load-bearing for the measurement model.

## Known open data caveats
The sanity bounds + `rules.py` encode the harmonization decisions (unit fixes, sentinel removal, ms→s
ECG conversion, text→code maps). One minor item remains clinician-pending and does **not** affect the
modelled features:
- **Suicide `ltsg07`** — asymmetric "don't know" coding (BP yes/no vs DR yes/no/DK ≈11%); the exact
  BP↔DR alignment of the rarest high-lethality categories (DR n ≤ 7) is approximate.
