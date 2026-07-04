# FACE-ATLAS — flagship manuscript (`article_v2`)

A from-scratch, discovery-forward rewrite of the FACE-ATLAS paper, targeting general-interest /
top-tier venues. Biology is the headline; the pipeline is the Methods. Built per the approved
architecture (`article/FLAGSHIP_PAPER_ARCHITECTURE.md`) and plan.

**Title.** *A transdiagnostic map places immunometabolic burden as a distinct, durable, prognostic
axis of severe mental illness.*

**One-sentence thesis.** When routinely-collected biology is placed as a co-equal latent axis
inside a diagnosis-blind transdiagnostic map — measured honestly, never imputed, uncertainty
carried end-to-end — one axis, immunometabolic load, separates from the entire clinical severity
picture, stays put over two years, and marks the worst-prognosis pole — making it a concrete
cohort-enrichment and monitoring target that a diagnosis or a severity score cannot see.

## Structure (compiled `main.pdf`, 18 pages)

| Part | Content |
|------|---------|
| Abstract | Discovery-forward: biology opens, method is one clause, honest bound closes. |
| 1 Introduction (~500 w) | Diagnoses aggregate heterogeneity; biology has entered maps only as a *correlate*; explicit novelty **and** non-novelty. |
| 2 Results — five-beat spine | **Exists** (Fig 1) → **Organizes** (Fig 2) → **Separates** (Fig 3, the hinge) → **Persists** (Fig 4) → **Predicts** (Figs 5–6). |
| 3 Discussion (~600 w) | Three-way convergence; relative (not absolute) independence; stratify-vs-monitor; confident numbered limitations. |
| 4 Methods (at end) | Full pipeline as rigor: observed-cell likelihood, sparse bifactor/ESEM, copula block, freeze-before-validation, errors-in-variables GLM. |
| Extended Data | E1–E6 supporting validation. |

## Main figures

| Fig | File | Beat | Source |
|-----|------|------|--------|
| 1 | `fig1_atlas.png` | Exists | real-data render (`fig2_map`) |
| 2 | `fig2_continuum.png` | Organizes | real-data render (`fig4_continuum`) |
| 3 | `fig3_biology_g.png` | **Separates (hinge)** | real-data render (`fig3_biology_g`) |
| 4 | `fig4_persistence.png` | Persists | real-data render (`fig5_persistence`) |
| **5** | **`fig5_archetype_prognosis.png`** | **Predicts (money figure)** | **NEW — computed this session** |
| 6 | `fig6_prognosis_quant.png` | Predicts (quant) | real-data render (`fig6_prognosis_rebuilt`) |

### The money figure (Fig 5) — newly computed

Computed directly from real patient-level data
(`results/face/prognosis_oop/consolidate/prognosis_patient_risk.parquet`), which carries, per
`patient_id` for all 9,013 patients, both the archetype weights (`arch_w0..w4`) and the two-year
outcomes (`egf__remission_V2`). Panel **a**: all 9,013 patients projected onto the five-archetype
simplex as convex blends. Panel **b**: the same map, the 2,420 with follow-up shaded by two-year
functional remission (bin mean, bins ≥5). The blue→red gradient runs from the well pole (63%
remission) to the immunometabolic pole (22%) — and the immunometabolic corner is worse than the
*equally severe* clean-biology corner, the "a severity score cannot see this" evidence.

Per-archetype two-year functional remission (real data, `egf__remission_V2`):

| Archetype pole | n | remission |
|----------------|---|-----------|
| ↑immunometabolic ↑severity ↑suicidality | 1,426 | **22.3%** (worst) |
| severe, low-biology (↑severity, ↓immuno) | 1,584 | 33.7% |
| ↑developmental ↓immuno ↑suicidality | 2,004 | 37.4% |
| ↑sleep ↑mania ↓severity | 1,448 | 45.9% |
| low-burden / well (↓severity) | 2,551 | **62.7%** (best) |

## Extended Data (E1–E6)

`edfig_consort` (attrition), `edfig_fullatlas` (full 143-indicator atlas), `edfig_invariance`
(cross-cohort invariance), `edfig_loso` (leave-one-cohort-out), `edfig_repbench` (reproducibility),
`edfig_robustness` (confound robustness of the biology axis). All are real-data renders reused from
the original `article/figures/`.

## Building

Uses the same BasicTeX 2026 toolchain as `article/`:
```
export PATH=/usr/local/texlive/2026basic/bin/universal-darwin:$PATH
export TEXMFHOME=/tmp/m5work/texmf      # minimal multirow.sty stub
# build in a dir with sibling ../docs/figures symlink (second \graphicspath)
pdflatex main → bibtex main → pdflatex ×2
```
Latest build: 18 pages, 0 undefined refs/citations, 0 missing figures, 0 `??` markers, 35 citations.

## Notes / open items for the authors

- **Reused main figures carry their original embedded panel titles.** Figs 1–4 and 6 are the
  canonical real-data renders; their in-panel titles (e.g. "A latent atlas of severe mental
  illness") were written for the previous manuscript. For final submission, consider regenerating
  them from `reports/*.csv` via the repo venv so each panel title defers to the new flagship
  caption. This is cosmetic — the data and captions are correct as-is.
- **Fig 5 (money figure) is fully new and self-consistent** with the flagship captions.
- **Nothing in the original `article/` was modified** by this build; `article_v2/` is a parallel tree.
- The abstract's canonical numbers match the source model outputs; the correlated-arm numbers
  (0.10 vs 0.39/0.42) are tagged as the freely-correlated model throughout, resolving the
  ambiguity flagged in the earlier `article/` review.
