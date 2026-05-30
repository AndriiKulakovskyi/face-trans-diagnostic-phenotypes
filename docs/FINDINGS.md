# FINDINGS — FACE trans-diagnostic study (v2) — running log

Paper-oriented log of empirical + methodological findings on the **v2** dictionary. Every number
must be reproducible from the pipeline. **No analysis results yet** — the dimensional analysis and
stratification have not been re-run on v2.

> The v1 findings log is archived at git tag `v1-archive-2026-05-30`. Do **not** carry over v1 numbers.

## Settled — data processing
- **v2 dictionary:** 214 usable variables (a re-curated subset of v1's 361); `qa_harmonization`
  reports 190/190 load + pass sanity (0 fail).
- **Type-aware scaling to [−1, 1]**; robust-z explosion fixed (prolactin |z|≈106→5); **masked /
  no-imputation** design kept (no hard missingness drop).
- See the 3-part QA report (`results/reports/qa_harmonization.html`) and CLAUDE.md §"Data processing".

## Track 1 — dimensional analysis (v2)
[TODO — structure test (discrete vs dimensional), K-selection, axis loadings, confound η², outcomes vs DSM.]

## Track 2 — patient stratification (v2)
[TODO — clusters, stability, independence panel, discrete-vs-continuum verdict.]
