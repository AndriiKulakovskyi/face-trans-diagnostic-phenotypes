# [Title TBD — v2 study] Trans-diagnostic structure across bipolar disorder, schizophrenia and depression (FACE)

**Status — v2 skeleton.** The prior (v1) draft is archived at git tag `v1-archive-2026-05-30`.
**No v2 results are written yet**; the dimensional analysis and stratification have not been
re-run on the re-curated v2 dictionary. Do **not** carry over v1 numbers — every result is to be
re-derived from zero on v2.

**Authors.** [TO CONFIRM — author list, affiliations, corresponding author]

## Abstract
[TODO — write after Phase 4/5.]

## 1. Introduction
[TODO.]

## 2. Methods

### 2.1 Cohorts & data
FACE 3-cohort longitudinal data — Bipolar (n = 6,252), Schizophrenia (n = 2,209), Depression
(n = 552); visits V0 (baseline) → V4. Analyses **anchor on V0**; later visits assess the temporal
coherence of the recovered structure. Data are confidential (Fondation FondaMental) and never committed.

### 2.2 Harmonization & dictionary (v2)
A re-curated common-variables dictionary (`data/face-common-vars.xlsx`, **214 usable variables**)
maps each harmonized variable to its per-cohort source columns, with per-variable sanity bounds
and cross-cohort comparability rules. **No imputation** is performed anywhere; pervasive
missingness is handled by masked methods.

### 2.3 Data processing — three stages
1. **Harmonized variables (native scale):** per-cohort source → harmonization rule (text→code,
   unit fixes) → sanity bounds (out-of-range → NaN, never imputed).
2. **Type-aware scaling to [−1, 1]:** binary/ordinal → min-max; continuous → log (if heavy
   right-skewed) + winsorize(1/99) + robust-z (median/MAD) clipped ±5.
3. **Aggregated V0 domain scores (model inputs):** items → construct-level scores (masked mean of
   signed robust-z; no imputation). These ~69 domain scores — not the raw items — enter the models.
   Aggregation removes item-count weighting bias, raises coverage without imputing, and yields
   interpretable constructs. Cognition follows `docs/neuropsy_features.yaml` (WAIS standard scores
   + TMT). QA: `scripts/qa_harmonization.py` (a 3-part report mirroring the three stages).

### 2.4 Dimensional model
[TODO — masked pairwise-complete correlation → principal-axis factoring + varimax; K chosen by
masked split-half reproducibility; confound control via cohort/site/age/sex η². Re-derive on v2.]

### 2.5 Patient stratification
[TODO — masked similarity → multipartite-spectral embedding → clustering with stability
(bootstrap ARI, consensus PAC) and an independence panel (cluster vs sex/age/cohort/site). Re-derive on v2.]

### 2.6 Outcomes & robustness
[TODO — axes/clusters vs DSM for patient-reported outcomes; ComBat site, CV-refit, leave-one-cohort replication.]

## 3. Results
[TODO — pending Phase 4/5.]

## 4. Discussion
[TODO.]

## References
[TODO.]
