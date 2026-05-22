# FACE-RLVR Clinical Glossary

This directory contains the **single source of truth** for all clinical
instrument definitions, lab reference ranges, categorical code lookups, and
clinical computation thresholds used by the FACE-RLVR pipeline.

All files are loaded at runtime by `src/face_rlvr/profiles/glossary_loader.py`
and validated by the Pydantic models in `src/face_rlvr/profiles/glossary_schema.py`.
The Python code in `src/face_rlvr/profiles/` only contains *logic* — extraction,
score interpretation, profile building. All *data* lives here.

## Directory Structure

```
config/glossary/
├── README.md                              (this file)
├── common/
│   ├── instruments.yaml                   # 16 shared instruments (MADRS, YMRS, ...)
│   ├── thresholds.yaml                    # 8 reusable severity band lists
│   ├── clinical_constants.yaml            # BMI, metabolic syndrome, Framingham,
│   │                                      # drug interactions, cognitive norms,
│   │                                      # medication-lab alerts
│   └── categorical_codes.yaml             # MARITAL, EDUCATION, EMPLOYMENT
├── bp/
│   ├── instruments.yaml                   # BP-specific + $registry
│   └── lab_ranges.yaml                    # 25 labs
├── sz/
│   ├── instruments.yaml                   # SZ-specific + overrides
│   └── lab_ranges.yaml                    # 15 labs
├── dr/
│   ├── instruments.yaml                   # DR-specific + overrides
│   ├── lab_ranges.yaml                    # 45 labs
│   └── categorical_codes.yaml             # RESISTANCE_LEVEL
└── asp/
    ├── instruments.yaml                   # ASP-specific + overrides
    ├── lab_ranges.yaml                    # 4 labs
    └── categorical_codes.yaml             # DSM_TYPE, SCHOOL_LEVEL, SCHOOL_TYPE
```

## Conventions

- Every file starts with `$schema_version: 1`
- Instrument IDs with hyphens or special characters must be quoted: `"MADRS"`, `"C-SSRS"`, `"CGI-S"`
- All French text is UTF-8
- Numbers remain unquoted (never `"30"`, always `30`)
- Use block scalars (`|`, `>`) for multi-line clinical sentences
- Lab ordering in `lab_ranges.yaml` is preserved (the vignette renderer depends on it)

## How the loader resolves data

### Instrument precedence

For each cohort, the loader:
1. Loads `common/instruments.yaml` (16 shared instruments)
2. Loads `{cohort}/instruments.yaml`
3. Overlays cohort definitions on top of common (cohort wins if same key)
4. Restricts and reorders the merged dict using `{cohort}/instruments.yaml`'s `$registry.order`

This means cohorts can **inherit** a common instrument unchanged or **override**
it by re-declaring it in their own file (typically to change `total_column`).

### Shared threshold bands

Instruments can either declare their severity bands inline or reference a shared
band list from `common/thresholds.yaml` via `severity_thresholds_ref`:

```yaml
PSQI:
  name: PSQI
  total_column: psqi_
  severity_thresholds_ref: PSQI_THRESHOLDS   # loaded from common/thresholds.yaml
  # ... no severity_thresholds here
```

The two mechanisms are mutually exclusive — the Pydantic schema rejects
instruments that declare both.

### Categorical code aliases

Numeric string keys in categorical code files are auto-expanded with a `.0`
variant at load time. So writing:

```yaml
MARITAL_STATUS_CODES:
  "1": "célibataire"
```

produces a dict with both `"1"` and `"1.0"` → `"célibataire"` entries.
This preserves backward compatibility with the pre-migration Python dicts.

## How to add a new instrument

### Shared instrument (used by 2+ cohorts)

1. Add a new top-level entry to `common/instruments.yaml`:

```yaml
MyInstrument:
  name: MyInstrument
  full_name: "Full English Name"
  full_name_fr: "Nom complet en français"
  domain: depression
  total_column: my_total_col
  score_range: [0, 50]
  higher_is_worse: true
  evaluation_type: auto        # or "hetero"
  clinical_note_fr: "Note clinique en français."
  severity_thresholds:
    - {min_score: 0, max_score: 10, code: normal,
       label_fr: "Pas de symptômes", clinical_meaning_fr: "..."}
    - {min_score: 11, max_score: 30, code: moderate,
       label_fr: "Symptômes modérés", clinical_meaning_fr: "..."}
    - {min_score: 31, max_score: 50, code: severe,
       label_fr: "Symptômes sévères", clinical_meaning_fr: "..."}
```

2. Add the instrument key to the `$registry.order` list of each cohort that
   should use it, and add it to the appropriate `$registry.groups.*` list.

3. Re-run `pytest -q` to validate the YAML loads and vignettes stay consistent.

### Cohort-specific instrument

Add it to `{cohort}/instruments.yaml` directly (not `common/`). Include it in
the cohort's `$registry.order` and appropriate `$registry.groups.*` list.

### Screening instrument (binary positive/negative)

Instead of `severity_thresholds`, use:

```yaml
MyScreening:
  name: MyScreening
  total_column: my_screen_col
  score_range: [0, 20]
  evaluation_type: auto
  screening_threshold: 7
  screening_positive_label_fr: "Dépistage positif"
  screening_negative_label_fr: "Dépistage négatif"
  clinical_note_fr: "..."
```

## How to add a new lab

Edit `{cohort}/lab_ranges.yaml` and add an entry to the `labs` list. The
ordering is significant — it controls the order labs appear in the vignette:

```yaml
labs:
  - csv_col: my_lab_col
    name: "My Lab (English)"
    name_fr: "Mon dosage"
    unit: "mmol/L"
    normal_range: [3.5, 5.5]
    sex_specific: false       # true only if clinical_constants.sex_specific_lab_ranges defines this lab
```

## How to add a new categorical code

Edit `{cohort}/categorical_codes.yaml` (or `common/categorical_codes.yaml` for
shared codes) and add a new top-level dict:

```yaml
MY_NEW_CODE:
  "1": "Catégorie 1"
  "2": "Catégorie 2"
```

Access from Python via `get_cohort_categorical_codes(cohort)["MY_NEW_CODE"]`
or add a loader-backed constant in the relevant extractor file.

## How to modify clinical thresholds

All computation thresholds for BMI, metabolic syndrome, Framingham, drug
interactions, medication-lab alerts, and cognitive norms live in
`common/clinical_constants.yaml`. Edit the values there — the Python functions
in `common_extractors.py` read from this file at load time.

**Important**: After modifying YAML in a running Python session, call
`face_rlvr.profiles.glossary_loader._invalidate_cache()` to force a reload.

## Verification after any change

After editing any YAML file, always run:

```bash
# Smoke test all 4 cohorts
python -c "
import pandas as pd, hashlib
from face_rlvr.profiles import (
    extract_bp_patient, build_bp_profile,
    extract_sz_patient, build_sz_profile,
    extract_dr_patient, build_dr_profile,
    extract_asp_patient, build_asp_profile,
)
for name, path, ex, bld in [
    ('bp', 'data/BP.csv', extract_bp_patient, build_bp_profile),
    ('sz', 'data/SZ.csv', extract_sz_patient, build_sz_profile),
    ('dr', 'data/DR.csv', extract_dr_patient, build_dr_profile),
    ('asp', 'data/ASP.csv', extract_asp_patient, build_asp_profile),
]:
    df = pd.read_csv(path, nrows=2)
    for _, row in df.iterrows():
        profile = bld(ex(row))
        assert profile.full_vignette
    print(f'{name}: OK')
"
```

If the change should preserve byte-identical output (e.g., renaming a column
without changing clinical meaning), hash the vignettes and compare against
baseline:

```bash
python -c "
import pandas as pd, hashlib
from face_rlvr.profiles import extract_bp_patient, build_bp_profile
df = pd.read_csv('data/BP.csv', nrows=1)
vignette = build_bp_profile(extract_bp_patient(df.iloc[0])).full_vignette
print(hashlib.md5(vignette.encode()).hexdigest()[:12])
"
```

## Pydantic validation errors

If the loader raises a `pydantic.ValidationError`, read the error carefully:
- `extra inputs are not permitted` → you added a field the schema doesn't recognize
- `min_score > max_score` → band bounds are inverted
- `cannot set both severity_thresholds and severity_thresholds_ref` → pick one mechanism
- `field required` → you omitted a required field (e.g., `total_column`)

Fix the YAML and re-run. The loader caches aggressively; if you see stale
behavior during interactive development, call `_invalidate_cache()`.
