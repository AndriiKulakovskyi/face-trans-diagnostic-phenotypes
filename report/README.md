# FACE-ATLAS — technical report (LaTeX source)

A living technical report for the FACE-ATLAS project:
**A Transdiagnostic Latent-axis Stratification of Severe Mental Illness**
(bipolar disorder · schizophrenia · depression). The document grows as each milestone is completed; this
version documents **Milestone 1 — the measurement map**.

## Build

```bash
make                       # → FACE-ATLAS.pdf  (runs pdflatex twice for the TOC/refs)
# or:
pdflatex FACE-ATLAS.tex && pdflatex FACE-ATLAS.tex
```

Toolchain: **pdflatex** only. The style file `faceatlas.sty` depends solely on packages present in a TeX
Live *basic* install (no `tcolorbox`, `titlesec`, or non-standard fonts) — the coloured callout boxes are
built from `tikz`, and the body font is Latin Modern. It compiles anywhere with a standard TeX Live.

## Structure

```
FACE-ATLAS.tex        main file: title page, abstract, \input of the sections
faceatlas.sty         visual system: palette, callout boxes, section design, milestone strip, math macros
references.tex        bibliography (manual thebibliography)
sections/
  01_introduction.tex   scientific framing, the four-milestone architecture, the invariants
  02_data.tex           the data foundation + the no-imputation pipeline
  03_model.tex          candidate ontology, soft priors, the bifactor/ESEM generative model, mixed likelihoods
  04_estimation.tex     staged continuation, Gaussian marginal = FIML, Woodbury, compute strategy
  05_results.tex        the seven-dimension map (S1–S5) and the candidate adjudication
  06_validation.tex     prior-free refit, PPC/SRMR, WAIC, measurement invariance
  07_engineering.tex    repository, config-first design, tests, the two fixed bugs, reproducibility
  08_roadmap.tex        M2 strata → M3 temporal → M4 prognosis → M5 treatment
  09_limitations.tex    limitations and open problems
  10_appendix.tex       notation, the prior atlas, acceptance gates, glossary
figures/
  prior_atlas.png       the prior (theory) loading atlas (copied from docs/figures/)
```

## Extending the document for later milestones

Each milestone is a self-contained section. To add **M2 (strata)**: write `sections/11_strata.tex`,
`\input` it from `FACE-ATLAS.tex` after the roadmap, change the cover status pill and
`\milestonestrip{2}` → `\milestonestrip{3}`, and move the M2 paragraph in `08_roadmap.tex` from
`\planned` to a back-reference. The callout environments (`keyresult`, `definitionbox`, `invariantbox`,
`methodsbox`, `caveatbox`, `openbox`) and the status pills (`\certified`, `\provisional`, `\rejected`,
`\planned`) are reusable. Note: status pills contain `tikz` and must **not** be placed inside section
titles (a moving argument) — put them at the start of the body text instead.

## Provenance

Every number is drawn from the project's committed reports (`reports/04_stage*`, `reports/05–07_*`) and
configs, and is reproducible from `scripts/01_build_data.py` → `scripts/04_fit.py` → the confirmation and
invariance scripts. No per-patient data appear in the report.
