# FACE Common-Variables Dictionary (v2) — Reading & Loading Guide

The v2 harmonized data dictionary for the FACE 3-cohort data — Bipolar (BP), Schizophrenia (SZ),
Depression (DR). One row per harmonized variable: it maps each variable to its per-cohort source
column and records how to harmonize + sanity-check it. **The package loads it for you** — this
guide explains the columns.

> The v1 dictionary is archived at tag `v1-archive-2026-05-30`; the v2 file replaces it under the
> same name `data/face-common-vars.xlsx`.

## Files
- `data/face-common-vars.xlsx` — the v2 dictionary (Sheet1; 225 rows, **201 usable**, 16 columns)
- `data/{bipolar,schizophrenia,depression}.csv` — visit-level data (confidential, gitignored)
- `data/site_lookup.csv` — fondacode → site lookup (for `siteid_city`)
- `data/thesaurus/` — per-cohort source dictionaries (reference)

## Columns (v2)
- **Section** — clinical section (AUTO-QUESTIONNAIRES, BILAN BIOLOGIQUE, NEUROPSYCHOLOGIE, …).
- **Label**, **Codage** — human label + summary coding.
- **Why retained / Findings / Rule** — clinical rationale, cross-cohort caveats, harmonization rule.
- **BP / SZ / DR column in CSV** — the per-cohort source column (blank = absent in that cohort).
- **Final dtype** — `float` · `int8 binary` · `int8 categorical` · `int8 ordinal` · `string` · `date`.
- **Final unit / value set** — target unit or value set.
- **Sanity min check / Sanity max check** — per-variable plausibility bounds; out-of-range → NaN
  (never clipped, never imputed). Their presence is what flags a dictionary as v2.
- **Coverage (cohorts present)** — which cohorts contribute data.
- **Cluster readiness** — READY (3-cohort) / PARTIAL (2-cohort) / NOT USABLE (excluded). The loader
  keeps **READY + PARTIAL** (199 vars).
- **Canonical name (merged single-cohort)** — the harmonized variable name (the modelled feature).

## How the package loads it (don't hand-merge)
```python
from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset
df = build_unified_dataframe("data", "data/face-common-vars.xlsx",
                             readiness=["READY", "PARTIAL"], format="long")
ds = to_harmonized_dataset(df, load_variables("data/face-common-vars.xlsx"), visit="V0")
```
`build_unified_dataframe` reads each per-cohort source column, applies the harmonization rule
(`rules.py`), then the sanity bounds (auto-enabled in v2), and concatenates the cohorts.
**NaN = missing, never imputed.**

## Encoding by data type (then scaled — see CLAUDE.md §"Data processing")
- **float (continuous)** — labs, ages, scale totals. Skewed positive labs are log1p'd; all
  continuous → winsorize + robust-z clipped ±5 → [−1, 1].
- **int8 binary** — `{0,1}` → min-max → {−1, 1}.
- **int8 categorical / ordinal (Likert)** — small ordered scales → min-max → [−1, 1].
- **string** — mostly free-text labs parsed to numeric by a rule; identifiers are not modelled.
- **date** — e.g. `brthdtc` (NOT USABLE in v2 — redundant with `age`, never modelled).

Items are then aggregated into **construct-level V0 domain scores** (the model inputs).

## Comparing against DSM-5
Keep `arm` (DSM-5 subtype) and `cohort` as **labels** (never modelled). Compare recovered
dimensions / clusters against them via η², ARI, and per-group composition (Phase 4/5).

## v2 notes
- **Cognition (NEUROPSYCHOLOGIE) is available in all 3 cohorts in v2** (curated in
  `docs/neuropsy_features.yaml`: WAIS standard scores + TMT). The old "DR lacks neuropsychology"
  caveat no longer applies.
- `siteid_city` is kept loadable (for site stratification) but excluded from the feature matrix.
- Within-column unit mixing (`mchc` g/L vs g/dL; `hct` % vs L/L) is harmonized by v2 rules.
- **QA**: `scripts/qa_harmonization.py` validates that every variable loads + passes sanity, and
  shows the 3 processing stages per variable (`results/reports/qa_harmonization.html`).

## Known open data caveats
The sanity bounds + `rules.py` encode the harmonization decisions made during curation (unit fixes,
sentinel removal, ms→s ECG conversion, text→code maps; e.g. `hct`/`mchc` within-column scale mixing is
resolved by v2 rules). One minor item remains clinician-pending and does **not** affect the modelled
features:
- **Suicide `ltsg07`** — asymmetric "don't know" coding (BP yes/no vs DR yes/no/DK ≈11%); the exact
  BP↔DR alignment of the rarest high-lethality categories (DR n ≤ 7) is approximate.

(Curation working-notes formerly lived in a root `todo_data_cleaning.md`; every actioned decision is
now encoded in `data/face-common-vars.xlsx` + `rules.py` and logged in `docs/LABBOOK.md`.)
