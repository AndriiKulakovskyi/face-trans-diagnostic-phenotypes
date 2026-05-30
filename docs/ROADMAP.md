# ROADMAP — FACE trans-diagnostic study (v2)

> Single source of truth for *what* we are doing and *why*. The v1 roadmap is archived at git
> tag `v1-archive-2026-05-30`.

## Question
Across bipolar disorder, schizophrenia and depression (FACE), is trans-diagnostic variation
**dimensional** (latent symptom / biology / cognition dimensions) and/or **categorical** (patient
strata)? Re-derived from zero on the re-curated v2 dictionary — no imputation, confound-controlled.

## Design
- **Data:** v2 dictionary (214 vars), **V0** anchor; masked / no-imputation throughout. Later
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
- ⬜ **Phase 4** — dimensional analysis on v2. *Checkpoints: structure test (discrete vs dimensional), K-selection.*
- ⬜ **Phase 5** — patient stratification on v2. *Checkpoint: discrete-vs-continuum verdict.*
- ⬜ **Phase 6** — fresh manuscript from v2 results; re-baseline `tests/test_golden_numbers.py`
  and `verify.py` thresholds to v2; confirm `00_run_all.py` end-to-end + `pytest` green.
