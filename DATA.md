# FACE Common-Variables Dictionary — Reading & Loading Guide

Harmonized data dictionary for the FACE multi-pathology cohort: Bipolar (BP),
Schizophrenia (SZ), Depression (DR). Use it to pool patients from the three CSV
files into one merged single-cohort dataset for data-driven clustering, then
compare clusters against DSM-5 diagnostics.

## Files
- `face-common-vars.xlsx` — the dictionary (Sheet1, 379 rows × 20 columns)
- `data/bipolar.csv` — BP visit-level (21,343 rows)
- `data/schizophrenia.csv` — SZ visit-level (6,203 rows)
- `data/depression.csv` — DR visit-level (1,953 rows)
- `thesaurus/` — original thesaurus reference (BP / SZ / DR)

## How to read each row of the dictionary (left → right)
1. **What is this variable?** Cols A–G: Section, BP/SZ/DR thesaurus codes,
   Label, summary Codage, clinical "Why retained".
2. **Cross-cohort issue?** Cols H–J: Findings, Rule/action, Observed profile.
3. **Where in each CSV (+ raw CODAGE)?** K+L = BP, M+N = SZ, O+P = DR.
4. **Target post-harmonization?** Q = dtype, R = unit / value-set.
5. **Usable for joint clustering?** S: READY (130) / PARTIAL (221) / NOT USABLE (26).
6. **Merged-dataset column name?** T: Canonical name.

Start with S=READY for a clean joint clustering; add S=PARTIAL for richer features.

## Encoding approaches by data type (col Q / R)
- **`float` (continuous)** — age, labs, scale totals, Z-scores. Verify units
  per cohort using col J observed range; convert mg/dL→mmol/L, mois→années, etc.
- **`int8 binary`** — Y/N: harmonize to `{0=Non, 1=Oui, NA=Unknown}`. BP often
  stores text 'Oui'/'Non'; DR uses 0/1; SZ mixed (verify R425 inversion warning).
- **`int8 ordinal / categorical`** — 3-10 levels. MARISTAT (1-5), STPROF (0-6),
  EMPJOB (0-8 INSEE), EDULEVEL (1-20). Each cohort may use slightly different
  category IDs — see col I per row for the mapping table.
- **`category` / `string`** — free-text in one or more cohorts (e.g., SZ
  `EDULEVEL` has 'BAC' literal); recode using col I rules (e.g., 'BAC' → 12).
- **`date` (YYYY-MM-DD)** — `brthdtc`, `*_mhstdtc`. Verify format consistency
  before pd.to_datetime.
- **`string` identifier** — `usubjid_patients`, `fondacode`. Use only for
  record linkage, not as clustering input.

## Parse-metadata with clinical & scientific relevance
- **`Why retained` (col G)** — psychiatrist-perspective clinical rationale.
- **`Findings` (col H)** — construct mismatches, scale-anchor differences,
  unit traps. Example: R423 `age_first_episode` is mood-onset for BP/DR but
  psychotic-onset for SZ — same biological construct, different clinical
  question.
- **Per-cohort raw CODAGE (cols L, N, P)** — verbatim CODAGE text from each
  thesaurus; use for verification.
- **Cluster readiness (col S)** — READY = 3 cohorts available + comparable
  construct; PARTIAL = 2 of 3 cohorts (Tier B) or with construct caveat; NOT
  USABLE = real data gap (e.g., DR lacks NEUROPSYCHOLOGIE data entirely).

Special cases to apply before pooling:
- **SUICIDE timing items (R245–R277)** — BP/SZ have 4 categorical levels in
  data, DR has 3 — collapse the BP/SZ extra "visit-context" category before pooling.
- **SITEID (R5)** — disjoint across cohorts — recode to city (canonical
  `siteid_city`) using a per-cohort lookup table.
- **R425 PPARTPremier_episode** — DR uses `1=Oui, 2=Non` (not 0/1!) — invert
  before pooling.

## Building the single merged patient list

```python
import pandas as pd
bp = pd.read_csv('data/bipolar.csv', low_memory=False)
sz = pd.read_csv('data/schizophrenia.csv', low_memory=False)
dr = pd.read_csv('data/depression.csv', low_memory=False)
dico = pd.read_excel('face-common-vars.xlsx', sheet_name='Sheet1')
```

### Step 1 — Decide visit-aggregation strategy
CSVs are visit-level. Pick one to get one row per patient:
- Baseline only: `df.sort_values('visitnum').groupby('usubjid_patients').first()`
- Last visit:    `df.sort_values('visitnum').groupby('usubjid_patients').last()`
- Mean numeric:  `df.groupby('usubjid_patients').mean(numeric_only=True)`

### Step 2 — Apply harmonization rules per row
For each dictionary row where S in {READY, PARTIAL}:
- Read K (BP CSV col), M (SZ CSV col), O (DR CSV col)
- Apply Rule in col I per cohort (recoding, unit conversion, free-text parse)
- Cast to dtype in col Q, unit in col R
- Rename the column in each per-cohort dataframe to T (canonical name)

### Step 3 — Concatenate
```python
bp_h['cohort'] = 'BP'    # ARM column also kept for DSM-5 sub-diagnosis
sz_h['cohort'] = 'SZ'
dr_h['cohort'] = 'DR'
merged = pd.concat([bp_h, sz_h, dr_h], axis=0, ignore_index=True)
```

### Step 4 — QC
Verify every canonical column matches across the 3 dataframes; check missingness
is plausible per cohort.

## Comparing clusters against DSM-5 diagnostics

Keep `arm` (DSM-5 text label: 'Bipolaire de type 1/2/...', 'Schizophrénie',
'Trouble dépressif majeur', ...) and `cohort` as LABELS — not as clustering
inputs.

After running clustering on the harmonized feature matrix:
1. **Confusion matrix** cluster_id × `arm` — does the clustering recapitulate
   DSM-5 boundaries or cross them?
2. **Adjusted Rand Index** between cluster assignment and `arm` — quantifies
   concordance with the diagnostic system.
3. **Per-cluster diagnostic composition** — % of each ARM label in each
   cluster — identifies trans-diagnostic clusters (e.g., depressive-onset
   cluster containing both BP-type-2 and unipolar MDD).
4. **Per-ARM cluster spread** — how a single diagnosis splits across clusters —
   identifies biologically distinct subtypes within a clinical category.
5. **Cluster-defining features** — top variables (by Cohen's d vs other
   clusters); interpret biologically using the col G `Why retained` rationales.

This is the value of data-driven clustering: identifying trans-diagnostic
phenotypes that may map onto biology better than the categorical DSM-5 framework.

## Caveats
- DR CSV has no NEUROPSYCHOLOGIE data → 133 cognitive rows are PARTIAL or
  NOT USABLE for DR (need separate file or DR-imputation).
- TRAITEMENTS (medications) section was not extracted — pharmacological
  exposure is not in this dictionary.
- 26 NOT USABLE rows are real data gaps; see col H per row for derivation
  suggestions (e.g., R7 ARMCD: derive from text `arm`).
- Missingness per cohort depends on the visit-selection strategy chosen in
  Step 1 — col J shows visit-level missingness, not patient-level.
