# FACE-ATLAS — manuscript rewrite change-log

**Scope:** methods-first narrative rewrite + all collected review corrections, applied **in place** to the LaTeX source. Nothing committed — review with `git diff` in `article/` and accept/revert per hunk.

**Totals:** 8 files, **434 insertions / 150 deletions**. Every canonical number preserved (single source of truth: `canonical_numbers.md`). All figure `\label`s preserved, so every `\ref` still resolves. Braces balanced in all 8 files.

**How this was produced:** four editing agents, partitioned by file so no two touched the same source. All edits verified by read-back on disk.

> **One thing to do before submission:** run `pdflatex`/`latexmk` yourself. The edits are syntactically checked (brace balance, ref resolution) but a full compile is the final gate — I cannot run it here.

---

## By file

### `main.tex` (abstract) — +10/-… — Framing track
- **P1 / B-1 (methods-first):** closing sentence "The contribution is the measurement model pipeline; …reveals, not its premise." → three-pillar contribution block (missingness-native/no-imputation, mixed-likelihood, uncertainty propagated end-to-end, map fixed before validation) + finding recast as **existence proof** + "first such map across BP/SZ/MDD" framing. Opener **unchanged** (patch P2 retired — the existing opener is already a strong methods-venue lead).

### `sections/01_introduction.tex` — +18 — Framing track
- **P3 / B-1:** extended "five-stage measurement-and-validation pipeline … the contribution" with the three commitments held jointly (observed-cells only/no imputation; uncertainty end-to-end; map fixed before validation, diagnosis held out) + existence-proof sentence.
- **P3-consistent:** closing-paragraph concession "rather than a new generic estimator" → positive integration-discipline three-pillar claim. Correlated-G / observed-cell-likelihood content preserved.

### `sections/02_methods.tex` — +60 — Methods track
- **P7 / M-1 / K01 (unlikely-prior):** the ~980 "unlikely" cross-loading cells corrected to a **hard zero (δ₀ point mass)** matching the reported map (`soft_unlikely=False`); the N(0,0.05²) form now presented explicitly as a documented **sensitivity arm**, with identifiability rationale.
- **m-5 (targeted ESEM):** clarified this is a **targeted** ESEM (only curated plausible-cross cells horseshoe-freed; ~980 unlikely cells hard-zeroed); cited the all-cells-freed validator (~83% shrink) as evidence the zeros are data-consistent.
- **P6 / B-4 (bifactor robustness):** new `\paragraph{Robustness of the bifactor specification.}` with three guards (correlated-G arm reproduces ordering, immuno corr ≈0.10 vs 0.39/0.42; 83% of freed cells shrink to ≈0; hard-zeros stop G absorbing thin factors). Cites `bonifaycai2017`, `greene2019fit`.
- **P8 / M-3 (convergence gate):** single "R̂=1.03" claim → **two-tier gate** (backbone R̂≤1.01, bulk-ESS≥400, zero divergences; full mixed map R̂≤1.03 within the <1.05 range, `vehtari2017loo`; cross-seed Tucker φ=0.99).
- **P9 / m-8 (full-N rigor):** added the clause stating the reported map is the **full-N=9,013** fit (continuous block via Woodbury marginalization; explicit block cohort-weighted, full N in production); balanced N=2,000 configs framed as diagnostic/sensitivity arms only.

### `sections/03_results.tex` — +366/-… — Results track (heaviest)
- **P4 / M-2 (Fig 6 prognosis):** `fig6_prognosis.png` → `fig6_prognosis_rebuilt.png`; caption + prose rewritten to **lead with the within-cohort remission gradient** (BP A2=27%/A4=73%; SZ 9%/25%; DR 31%/72%; A2 immunometabolic pole lowest-remitting in every diagnosis) and the **η² compactness** (archetype 0.256 vs DSM-5 0.026, 9.7×, explicitly compactness on map axes NOT outcome variance). **+0.010 individual AUC demoted to one sentence.**
- **Worked patient (Fig 2):** `fig_localization.png` (synthetic) → `fig_worked_patient.png` (real patient **BP-62162**: core-tier A2, 86% immunometabolic-archetype weight; immuno +3.57 SD from 6 items; cognition/suicidality prior-dominated). Label `fig:localization` preserved.
- **Adaptive assessment:** `fig_mincount.png` → `fig_adaptive_assessment.png`; caption rewritten for **all-8-axis exact-Fisher** reliability curves (immuno triad BMI/weight/waist → 0.85 in 3 items, plateau ~0.88; mania 2 items max 0.408, substance 4 items max 0.429 = **bank limitation**). Label `fig:mincount` preserved.
- **Value of information:** `fig_voi.png` → `fig_value_of_information.png`; caption for **27-item shared battery → mean reliability 0.70** + cohort collection-gap map (SZ under-measured on immunometabolic 20.7 vs BP 34.9). Label `fig:voi` preserved.
- **B-2 / R01 (LOSO external validity):** NEW `\subsection{The measurement map is stable to holding out recruitment sites}` — 15 folds, VI/GLLVM refit; immuno congruence φ **0.9993–1.0000**; all 8 factors clear their bar in **all 15 folds**; decoupling 0.073–0.082; weakest φ=0.917 (Monaco, n=237); VI-vs-NUTS 0.957–0.999.
- **Applications subsections:** `\subsection{A few well-chosen indicators recover most of each dimension}` (adaptive assessment) and `\subsection{Designing efficient, harmonised batteries}` (value of information).

### `sections/04_discussion.tex` — +59 — Discussion track
- **B-1 opening reframe:** first paragraph rewritten so the three novel-in-combination pillars are named as **the contribution**; immunometabolic dissociation recast as the **existence proof**. BMI/CRP numbers + adiposity caveat preserved.
- **Novelty positioning:** new paragraph vs nearest prior art — HiTOP (`kotov2017hitop`; symptom-only), normative modelling (`marquand2016`, `wolfers2018`; per-measure deviation), Lamers (`lamers2013`, `lamers2020`; within-MDD, biology as correlated **outcome** not a latent axis), B-SNIP biotypes (`clementz2016biotypes`; discrete vs continuum). Novel slot: **routine biology as a co-equal latent axis inside one diagnosis-blind map across BP/SZ/MDD**.
- **m-9 / honest limitation:** new `\emph{Measurement precision is not uniform across axes.}` paragraph — **bank limitation not missingness** (95–96% have ≥1 item); six axes reach 0.85+; mania/substance capped ~0.45; model **degrades gracefully** (wide posteriors); A0/A4 mania-poles survive because a group mean over >1000 patients resolves the contrast to ~0.02 SD; "what to collect next" = graded AUDIT/DAST severity, not more binary screens.

### `annex/B_stratification.tex` — +28 — Methods track
- **P5 / B-3 (Hopkins disclosure):** removed the rule "a continuum verdict requires H→½" (data contradict it); **disclosed** H=0.79–0.81 real vs null 0.756±0.004, **z=8.71 vs uniform** — a large significant departure from **uniformity**, explicitly not evidence of clusters. New paragraph distinguishing cluster **tendency** (non-uniformity) from **separability** (departure from a matched Gaussian); names the silhouette-vs-single-Gaussian-null test as the operative separability criterion (peak silhouette 0.146 over K, <0.15; best-partition real 0.140 vs null 0.137±0.002, z=1.13, n.s.).

### `annex/G_metrics.tex` — +2 — Methods track
- **P5 / B-3 (glossary):** Hopkins row "→0.5 (no tendency)" → true value **0.79–0.81 (real), null 0.756±0.004, z=8.71**, with the note "high tendency vs uniform, NOT vs a matched Gaussian."

### `annex/F_display_items.tex` — +41 — Results track
- **New ED figure `edfig_loso.png`** (`\label{fig:loso}`, referenced from Results via `\edfig{fig:loso}`) — LOSO congruence + decoupling.
- **New supporting figure `fig_loading_vs_info.png`** (`\label{fig:loadinginfo}`, per figure_ordering_plan §5) — loading ≠ information (alcohol/cannabis flags load ~0.95 but Fisher info ~0.01 vs BMI 2.8; cross-family loadings on different link scales, information is the comparable quantity).

---

## Review issues resolved (cross-reference)

| Issue | Where fixed |
|---|---|
| B-1 framing (methods-first) | main.tex P1, 01_intro P3, 04_discussion opening |
| B-2 / R01 external validity | 03_results LOSO subsection + edfig_loso |
| B-3 Hopkins misreport + undisclosed z | G_metrics glossary + B_stratification disclosure |
| B-4 bifactor overfit | 02_methods P6 robustness paragraph |
| M-1 / K01 unlikely-prior equation | 02_methods P7 hard-zero + sensitivity arm |
| M-2 +0.010 AUC over-weighted | 03_results P4 (demoted to a sentence) |
| M-3 convergence gate 1.01 vs 1.03 | 02_methods P8 two-tier gate |
| m-5 targeted-ESEM wording | 02_methods |
| m-8 / P9 full-N rigor | 02_methods |
| m-9 honest mania/substance limitation | 04_discussion |
| Novelty positioning | 04_discussion |
| Four figures = real-data upgrades | 03_results (4 in-place swaps, labels kept) |

## Not done here (need you)
- **Compile:** run `pdflatex`/`latexmk` — final gate I can't run.
- **Boilerplate `90_boilerplate.tex`:** IRB numbers, funding, author contributions, registration IDs still `[TBD]` (m-6) — these are yours to fill.
- If a compile flags a missing citation key, tell me and I'll add a proper `references.bib` entry.
