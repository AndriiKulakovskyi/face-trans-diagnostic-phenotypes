# ROADMAP — FACE trans-diagnostic study (v2)

> Single source of truth for *what* we are doing and *why*. The v1 roadmap is archived at git
> tag `v1-archive-2026-05-30`.

## Question
Across bipolar disorder, schizophrenia and depression (FACE), is trans-diagnostic variation
**dimensional** (latent symptom / biology / cognition dimensions) and/or **categorical** (patient
strata)? Re-derived from zero on the re-curated v2 dictionary — no imputation, confound-controlled.

## Design
- **Data:** v2 dictionary (201 usable vars), **V0** anchor; masked / no-imputation throughout. Later
  visits (V1, V2…) test temporal coherence, they don't define the structure.
- **Processing:** harmonize → type-aware scale to [−1, 1] → aggregate to V0 **domain scores**
  (see CLAUDE.md §"Data processing" and the 3-part QA report).
- **Track 1 — dimensional:** masked pairwise-complete correlation → PAF + varimax; K by masked
  split-half reproducibility; confound η² (cohort/site/age/sex); outcomes vs DSM; longitudinal coherence.
- **Track 2 — stratification:** masked similarity → multipartite-spectral embedding → clustering
  + stability (bootstrap ARI, consensus PAC, gap, silhouette) + independence panel; discrete-vs-continuum verdict.

## Phases
- ✅ **Phase 1–2b** — v2 dictionary finalized + locked; pipeline wired to v2; preprocessing
  debugged (explosion fix, type-aware [−1, 1] scaling); 3-part QA report complete.
- ✅ **Phase 3** — clean slate: v1 generated artifacts removed; conclusion docs reset to v2 stubs.
- ✅ **Phase 4** — dimensional analysis on v2 (scripts `01–06`): hierarchical/bifactor measurement
  model → **K=3** trans-diagnostic axes (internalizing · cognition · cardiometabolic),
  **no p-factor** (ECV 0.42); confound-clean, leave-cohort-out reproducible, granularity-invariant.
  (Adding the alcohol/cannabis substance-use-disorder construct collapsed the earlier K=4; the weak
  *illness-course* axis is no longer reproducible — see `docs/FINDINGS.md`.)
- ✅ **Phase 5** — patient stratification on v2 (`07_phase5_stratify.py`): **DIMENSIONAL / continuum**,
  no discrete subtypes beyond the DSM cohorts. Validation arm A–D (`09–15`).
- ✅ **Phase 6** — manuscript + 6 figures from v2 results (`results/manuscript/FACE_trans_diagnostic_v2.docx`,
  `scripts/figures_manuscript.py`); golden tests + `verify.py` **re-baselined to v2** (`pytest` green
  — 99 passed; `verify.py` green). v2 pipeline = scripts `01–15`; the legacy `01–22` pipeline is
  removed (recoverable at tag `v1-archive-2026-05-30`).
