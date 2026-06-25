# article/ — the FACE-ATLAS journal manuscript

This folder holds the **journal article** (one comprehensive, discovery-framed paper for a top-tier venue),
distinct from `report/` (the long-form technical report, kept as the canonical backup + Supplementary source).

- **[`ARTICLE_PLAN.md`](ARTICLE_PLAN.md)** — the blueprint: framing, target-journal recommendation,
  title/abstract drafts, section-by-section outline (every claim traced to a committed `reports/NN_*.md`),
  the display-items plan, the **figure & table gap report (§6)**, references strategy, the claims ledger,
  the reviewer pre-mortem, and the writing workflow. **Read this first.**

## Manuscript (first complete draft — compiles)

- **`main.tex`** — preamble, title, abstract, keywords; `\input`s the sections and tables; numeric natbib.
- **`sections/`** — `01_introduction`, `02_results` (R1–R6), `03_discussion` (+ Limitations),
  `05_methods`, `90_boilerplate` (data/code/ethics/contributions — PI to complete `[TODO]`s).
- **`tables/`** — `table1_characteristics.{md,tex}` (aggregate sample table), `table2_dimensions.tex`
  (the nine dimensions + adjudication + invariance).
- **`references.bib`** — 54 PubMed-verified entries (per `docs/LITERATURE_EVIDENCE.md`) + 15 canonical
  methodological references.
- **`main.pdf`** — compiled output (~20 pp). Build (pick one):
  - **Tectonic** (no MacTeX install): `tectonic -X compile main.tex` from `article/`
  - **latexmk** (requires MacTeX/BasicTeX): `latexmk -pdf main.tex`
  - **Wrapper** (tries latexmk → tectonic → Docker): `./compile.sh`

## Figures (publication-spec, regenerated)

- **`make_figures.py`** regenerates all display items at **300 dpi** (PNG + vector PDF) into `figures/`,
  in a unified colourblind-aware house style with panel labels, from committed aggregates only (plus the
  derived per-patient coordinate table for the PCA embedding; no raw clinical value is emitted). Run:
  `python3 article/make_figures.py`.
- Main figures: `fig1_overview` (study schematic), `fig2_map` (loading atlas + $\Phi$), `fig3_biology_g`,
  `fig4_continuum` (PCA embedding + structure gate + archetypes), `fig5_persistence` (trait/state +
  spine-corner), `fig6_prognosis` (atlas + co-informative value). Extended Data: `edfig_m5_treatment`
  (identification + moderation forest).
- Remaining figure polish for submission: journal-specific sizing/CMYK/font embedding and final
  accessibility pass (plan §6.2, item 7).

Provenance: numbers come from `reports/` and `docs/*_FINDINGS.md`; citations are PubMed-verified. No
per-patient data appears here.
