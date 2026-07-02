# Changelog — removal of treatment (M5 / stage v) content

**Date:** 2026-07-02
**Scope decided with author:** edit the LaTeX manuscript only (`article/`); remove the
M5 treatment-**moderation** analysis (stage v); **keep** medication-as-covariate
robustness claims; leave all source code and reproducibility docs untouched.
**Deliverables:** edited sources, a recompiled `main.pdf` that builds cleanly, and this changelog.

**Rationale (author):** the cohort carries no randomized treatment arm, and the
medication/treatment-exposure fields may not have been collected reliably in the cohort
database. Rather than rest any claim on that data, all treatment-moderation content is
removed and the pipeline is reframed from five stages to four.

---

## 1. Result of the change

- The pipeline is now **four stages** (measurement → structure → temporal coherence →
  prognosis) instead of five. The former **stage (v) "Treatment: a bounds-and-defends
  causal analysis"** is gone.
- **Appendix "E — Treatment"** is dropped from the build. Later appendices reletter
  automatically: **F→E** (display items), **G→F** (metrics), **H→G** (variational).
- The manuscript recompiles cleanly: **69 pages** (was 73), **0 undefined references**,
  **0 missing figures**, **0 `??` markers**.
- All **medication-as-covariate robustness** claims are preserved (the immunometabolic
  axis is still "robust to medication, adiposity and site"; the antipsychotic-exposure
  covariate-adjustment ladder is intact).

---

## 2. Removals and reframings, by file

### `main.tex`
- Abstract: **"The pipeline has five stages" → "four stages"**.
- Abstract: removed stage **"(v) A bounds-and-defends causal analysis asks whether the
  map moderates treatment on observational data."**
- Abstract: **"one property ... recurs across all five stages" → "all four stages"**.
- Abstract: removed **"and survives treatment adjustment"** from the immunometabolic-axis
  sentence (this defence relied on the drug-class exposure data).
- Abstract: reframed closing caveat — was *"individual-level clinical utility and treatment
  selection would require external validation, incident-event outcomes and randomized
  treatment data"* → now *"individual-level clinical utility would require external
  validation and incident-event outcomes."*
- Removed **`\input{annex/E_treatment}`** from the appendix list (the file
  `annex/E_treatment.tex` is retained on disk, simply no longer compiled).

### `sections/01_introduction.tex`
- **"five-stage measurement-and-validation pipeline" → "four-stage"**.
- Downstream-test parenthetical **"(structure, temporal coherence, prognosis, treatment
  bounds)" → "(structure, temporal coherence, prognosis)"**.
- **"recurs across all five stages" → "all four stages"** (methodological-discipline paragraph).
- Removed the enumerated stage **"(v) Treatment: a bounds-and-defends causal analysis
  states what observational treatment data can and cannot support."**
- **"Running through all five stages" → "all four stages"**.
- Removed the clause **"— including a treatment-moderation analysis that this observational
  cohort can only bound —"** (kept "We report the modest and null results as deliberately as
  the positive ones").
- Final methods-summary sentence: **"downstream errors-in-variables prognosis and
  treatment-moderation analyses" → "a downstream errors-in-variables prognosis analysis."**

### `sections/02_methods.tex`
- Fixed the opening annex-range cross-reference **`(\ref{ann:measurement}--\ref{ann:treatment})`
  → `(\ref{ann:measurement}--\ref{ann:prognosis})`** (the range now ends at the last
  surviving annex).
- Pre-specification paragraph: removed **"and the treatment-moderation analysis explicitly
  exploratory"** (kept "stratification, temporal and prognostic arms secondary").
- Deleted the entire final subsection **"Treatment moderation: an exploratory negative
  control"** (~25 lines): the propensity/IPTW equation (`eq:m-iptw`), the doubly-robust
  moderation description, the E-value / minimum-detectable-effect text, and the closing
  `Supplement~\ref{ann:treatment}` pointer.

### `sections/03_results.tex`
- Removed the entire **"Treatment moderation: a bounded, defensive result"** paragraph and
  its table **Table `tab:moderation`** ("Treatment moderation on observational
  treatment-as-usual"). The prognosis figure (`fig:prognosis`) now flows directly into the
  variational-re-estimation subsection.
- Fixed a hardcoded appendix letter: **"(Annex~H)" → "(Annex~G)"** (the variational annex,
  relettered after Annex E was removed). This was the only hardcoded appendix letter in the
  document; all other cross-references are `\ref`/`\edfig`-driven and updated automatically.

### `sections/04_discussion.tex`
- Removed the M5 sentences **"On observational, expert-centre treatment-as-usual the map did
  not reliably moderate treatment response ... so the forecast is not a treatment proxy."**
  (both the moderation claim and the drug-class-attenuation defence, which relied on the same
  data). Kept the surrounding "faithful rather than lossy" and "We report these modest and
  null findings as deliberately as the positive ones" framing.
- "In sum" paragraph: **"... a concrete enrichment and monitoring target, and points to a
  specific, testable immunometabolic × treatment hypothesis for prospective study" → "... a
  concrete enrichment and monitoring target for prospective study."**
- Limitations: **"Four classes of limitation" → "Three classes"**.
- Limitations: removed the **"Observational treatment:"** class entirely (confounding by
  indication, E-values 1.1–1.8, drug-class exposures, "bound moderation, not establish
  selection").
- Limitations closing: was *"Establishing individual-level clinical utility or treatment
  guidance would require incident-event outcomes, randomized or trial-arm treatment data, and
  external validation..."* → now *"Establishing individual-level clinical utility would
  require incident-event outcomes and external validation beyond this baseline observational
  cohort."*

### `sections/90_boilerplate.tex`
- Code-availability sentence: module list **"stratification, temporal, prognosis and
  treatment modules" → "stratification, temporal and prognosis modules"** (the
  `src/face/treatment/` code remains in the repository per the agreed scope; it is simply no
  longer advertised as a reported module of the paper).

### `annex/E_treatment.tex`
- **No longer `\input`** into the build (removed from `main.tex`). The file is left in place
  on disk, unmodified, so the analysis is fully recoverable if you later restore it.

### `annex/F_display_items.tex`  (now Appendix E in the build)
- Figure `fig:design` (study-overview, `fig1_overview.png`) caption: **"the five questions
  ... five dependent ... exists, organizes, persists, predicts, guides treatment" → "the four
  questions ... four dependent ... exists, organizes, persists, predicts"**.
- Removed the Extended Data figure **`fig:treatment`** ("Treatment moderation is not
  identified on observational treatment-as-usual", image `edfig_treatment.png`). The figure
  was never cited from the main text, so no dangling reference resulted; the following figure
  (`fig:robustness`) renumbers automatically.
- Removed the claims-ledger row **"A well-identified null for lithium moderation; a testable
  antipsychotic hypothesis. / That the map guides treatment selection, or any causal treatment
  effect on treatment-as-usual."**

### `annex/G_metrics.tex`  (now Appendix F in the build)
- Removed the entire glossary block **"F. What can be said about treatment? (causal bounds,
  M5 — exploratory)"** — five metric rows: E-value, minimum-detectable-effect, propensity-score
  overlap, attenuation-under-treatment-adjustment, and treatment-course ΔAUC. The glossary
  `longtable` now closes cleanly after the representation-sufficiency row.

### `make_figures_copula.py` + `figures/fig1_overview.{png,pdf}`  (figure regenerated)
The study-overview schematic (`fig1_overview.png`, now Extended Data Fig. E1) is a hardcoded
schematic — not data-driven — and its art depicted five stages/questions. With the author's
approval (an explicit, narrow exception to the "code untouched" scope, limited to this one
figure script) the generator `fig1_overview()` in `make_figures_copula.py` was edited and the
figure re-rendered:
- Pipeline box 4 **"Prognosis /\ntreatment" / "2-yr functioning; TAU boundary (M4–M5)" →
  "Prognosis" / "2-yr functioning (M4)"**.
- Question-strip header **"Five questions, increasingly hard:" → "Four questions..."**.
- Question list **`["exists","organizes","persists","predicts","guides Tx?"]` →
  `["exists","organizes","persists","predicts"]`** (the vermilion "guides Tx?" box and the
  trailing arrow removed; the arrow-loop guard changed `i < 4 → i < 3`).
- The **"A = 5 archetypes"** box (box 3) is deliberately **kept** — that is the archetype
  simplex, not a treatment stage.
- Re-rendered with the repository's own Python environment; both `fig1_overview.png` and
  `fig1_overview.pdf` in `article/figures/` are updated. The figure art and the LaTeX caption
  ("four questions") are now consistent.

---

## 3. What was deliberately KEPT (medication as a covariate, not the M5 analysis)

These use medication/adiposity/site as **covariates in robustness checks of the measurement
model**, which is separable from the M5 treatment-moderation analysis and was retained by the
author's instruction:

- Abstract: "correlation with G ≈ 0.10 ... **robust to medication, adiposity and site**".
- `01_introduction.tex`: "metabolic burden **follows medication, adiposity and chronicity**
  more than symptom load"; and the general DSM statement "patients who differ widely in their
  biology, course and **treatment response**".
- `02_methods.tex`: the covariate-adjustment ladder "(age, sex, education, site,
  **antipsychotic exposure**, adiposity)".
- `03_results.tex`: "adjusting each indicator ... then additionally for **antipsychotic
  exposure**, did *not* raise the immunometabolic correlation"; and the `fig:biologyg` caption
  "stays there under **adjustment for medication, adiposity and site**".
- `04_discussion.tex`: "Metabolic burden ... is **driven by medication**, adiposity, age and
  chronicity"; "they survive, and even fall slightly under, **adjustment for medication,
  adiposity and site**"; the general phrases "**treatment intensity**" (cohort description)
  and "not an individual **treatment** rule" (design disclaimer).
- `annex/A_measurement_model.tex`: the criteria and table row that **exclude**
  treatment-/outcome-driven confound variables (e.g. prolactin, oxcarbazepine level,
  clozapine flag) from the latent axes — measurement-model methodology, not the M5 analysis.
- `annex/H_variational.tex` (now Appendix G): the NUTS-vs-VI table row **"Latent treatment"**
  refers to latent-*variable* marginalization (Woodbury), not drug treatment.

---

## 4. Automatic renumbering (verified in `main.aux` after recompile)

| Object | Before | After |
|---|---|---|
| Appendix: measurement | A | A |
| Appendix: stratification | B | B |
| Appendix: temporal | C | C |
| Appendix: prognosis | D | D |
| Appendix: **treatment** | **E** | **(removed)** |
| Appendix: display items | F | **E** |
| Appendix: metrics | G | **F** |
| Appendix: variational | H | **G** |
| Table: treatment moderation (`tab:moderation`) | Table 6 | **(removed)** |
| Extended Data: design/atlas/consort/invariance/repbench/loso/loadinginfo | F1–F7 | **E1–E7** |
| Extended Data: **treatment** (`fig:treatment`) | **F8** | **(removed)** |
| Extended Data: robustness (`fig:robustness`) | F9 | **E8** |
| Extended Data: VI-loadings / VI-Φ | H10 / H11 | **G9 / G10** |

All of these updated automatically because the manuscript uses `\ref`/`\edfig` label
references throughout; the **only two manual cross-reference edits** required were the methods
annex-range endpoint and the one hardcoded "Annex~H" in the results (both listed above).

---

## 5. Build

- Toolchain: system BasicTeX 2026 (`pdflatex` + `bibtex`), `unsrtnat` bibliography style.
- Result: **69 pages, 0 undefined references/citations, 0 missing figures, 0 `??` markers**
  across all passes.
- Files updated in `article/`: `main.pdf` and the regenerated `main.{aux,bbl,blg,log,out}`.

## 6. Recoverability / notes

- `annex/E_treatment.tex` is retained on disk (not deleted, not modified) — restore by
  re-adding `\input{annex/E_treatment}` to `main.tex`.
- Orphaned figure image files remain unused on disk: `figures/edfig_treatment.{png,pdf}`
  (was `fig:treatment`) and `figures/edfig_m5_treatment.{png,pdf}`. They are not referenced by
  the build; left in place per the "article only" scope.
- The study-overview figure (`fig1_overview.png`, Extended Data Fig. E1) **was regenerated**
  (see §2) so its art now matches the four-stage caption. The previous five-question version is
  backed up at `/tmp/m5work/fig1_overview.BEFORE.png` for the current session only.
- `make_figures_copula.py` still contains other functions that reference treatment
  (e.g. `edfig_treatment`, which regenerates the now-orphaned treatment figure image). These
  were **not** touched — only `fig1_overview()` was edited. They are dormant unless explicitly
  invoked and produce images the manuscript no longer includes.
