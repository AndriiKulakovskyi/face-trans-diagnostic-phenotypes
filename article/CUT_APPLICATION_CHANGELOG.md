# FACE-ATLAS — Applied cut-list edits (main text)

_Applied 2026-07-03. Companion to `CUT_CANDIDATES_REVIEW.md`._

Scope: the **full main-text pass** you approved — all 31 candidates flagged in the four narrative
sections (Introduction, Methods, Results, Discussion). **No result value, statistic, figure, table,
equation, or citation was deleted** — every edit removes or tightens *prose*, and each removed
number was verified to survive at its canonical location elsewhere in the document.

**Full accounting of the 40 review candidates:** 31 applied here (4 narrative sections); 1 back-matter
item **deferred** (see below); 8 annex items **not touched** (out of scope for "the body"). 31 + 1 + 8 = 40.

- **Deferred — back matter, `sections/90_boilerplate.tex` L~17 (CONDENSE, low severity):** the Code-availability
  clause "per-patient data are never required to regenerate the aggregate results" restates the Data-availability
  section. This was outside the approved 31-item main-text set (it is back matter, not one of the four narrative
  sections) and is left unchanged pending your go-ahead — say the word and I'll fold it in.

**A note on the four "move-to-supplement" candidates** (Methods L353, Results L597, Discussion L101/L105):
these were executed as **straight deletions from the main text, not physical relocations into an annex file.**
In each case the removed content was already covered by an existing cross-reference to its target annex (Prop.
`fiml`, Annex B `ann:stratification`), so copying the text into the annex would have *duplicated* it rather than
moved it. The net effect matches the "move" intent — the detail now lives only in the annex — but no annex `.tex`
file was edited. Entries below are marked accordingly.

**Net change:** 4 files, +19 / −93 lines, **−456 words** (Intro −103 · Methods −65 · Results −197 · Discussion −91).
**Build:** recompiles clean — 69 pages, 0 undefined references, 0 missing figures, 0 `??` markers.

---

## Introduction (6 candidates)
- **L84 + L85 — CUT.** Dropped the two trailing sentences of the results-preview paragraph
  ("…which we present as a property the method *reveals* rather than the premise it was built on.
  We report the modest and null results as deliberately as the positive ones.") — both restated
  earlier prose / filler.
- **L90 + L91 + L94 — CONDENSE + CLARIFY.** Rewrote the closing "Methodologically, FACE-ATLAS
  proposes…" paragraph: removed the re-listed pipeline components (L91), the second full statement
  of the three commitments (L94, now stated once in the earlier paragraph), and the undefined
  `\Gfac` drop-in (L90). The paragraph now states only the model's distinctive technical framing.
- **L32 — resolved in place.** The "pipeline-as-contribution" framing that appeared twice now
  appears once (kept at its earlier, better-written instance rather than the closing echo).

## Methods (8 candidates)
- **L3 — CUT.** Trimmed the throat-clearing preamble sentence; kept the notation definitions and
  the Supplement pointer that follow.
- **L80 — CUT.** Removed the duplicate "full derivation…in Supplement" pointer (L3 is now the sole one).
- **L206 + L281 — CONDENSE (dedup).** The immunometabolic/cognition/sleep correlation triple
  (0.10 / 0.39 / 0.42) appeared in both the correlated-`\Gfac` estimation paragraph and the
  bifactor-robustness paragraph. Kept the numbers at the estimation paragraph (their definitional
  home); dropped the repeat in the robustness list.
- **L296 — CONDENSE.** Trimmed the restated subsample-rejection rationale; kept the 92% figure.
- **L345 — KEPT (deviation from flag).** The flag called the flat-prior-refit result a duplicate of
  Annex A's Cor. `corA:flat`. Left intact: it is the "First," item of a First/Second/Third in-engine
  enumeration, and the in-engine presentation differs from the annex corollary. Condensing it would
  break the enumeration for little gain. _Flagged here for your call._
- **L353 — MOVE (executed as deletion).** Deleted the tangential "standalone maximum-likelihood arm
  adds no independent evidence" aside from the main identification argument; the argument already lives
  in Annex A / Prop. `fiml`, so it was not re-copied there. No annex file edited.
- **L356 — CUT.** Dropped the duplicated minimum-Tucker `0.96` value from the robustness sentence.

## Results (9 candidates)
- **L8 — CUT.** Removed the "cohorts differ… as expected for these disorders" filler aside.
- **L213 + L280 — CUT + CONDENSE (partial of the flagged move).** The flag proposed compressing the
  four-subsection measurement-design block and moving detail to a supplement. Applied the reversible,
  high-value part: **cut** the entire duplicate subsection "Designing efficient, harmonised batteries"
  (L280 — it restated the preceding paragraph almost verbatim; `fig:voi` is still cited above) and
  **condensed** the overlapping "few well-chosen indicators" prose (L227). The three figures
  (localization / mincount / voi) were **kept in place** — a full physical export to an annex is a
  larger restructure with figure-renumbering consequences. _A full block-move remains available if you want it._
- **L227 — CONDENSE.** Tightened the repeated mania/substance instrument-fix recommendation.
- **L597 — MOVE (executed as deletion).** Deleted the "coarse two-region tessellation" parenthetical
  (it invoked η² values not shown in Results); the detail already lives in Annex B `ann:stratification`,
  so it was not re-copied there. No annex file edited.
- **L701 — CONDENSE.** Trimmed the trait/state "thermometer" paragraph's redundant "the same
  decomposition… sharpens the reading"; kept the `fig:thermometer` reference and the durability spectrum.
- **L740 + L742 — KEPT as canonical.** These carry the per-cohort remission percentages and the
  compactness caveat as the *lead* statement; the duplication was resolved by trimming the later
  restatement instead (L800).
- **L800 — CUT.** Removed the per-cohort remission-% restatement (27→73, 31→72, 9→25) since those
  values are stated at the L740 lead paragraph; kept the odds-ratio and robustness detail here.

## Discussion (8 candidates)
- **L37 — CONDENSE.** Tightened the trailing "This is the precise property…" editorial restatement.
- **L53 — CUT.** Dropped the "strong convergent validity" self-characterization (the convergence
  point is made in the preceding sentence).
- **L101 + L105 — CUT + MOVE (move executed as deletion).** Collapsed the biotypes paragraph's tail:
  removed the repeated equally-severe/opposite-biology contrast (L101), the coarse-tessellation aside
  (L105 — deleted, not re-copied; it is the same object covered by Annex B, also removed from Results
  for consistency), and the symptom-only-general-factor instability tangent; kept the rigorous
  no-biotypes null and the archetype-compactness result.
- **L149 — CONDENSE.** Removed the "more useful than an overstated one" self-praise; kept the
  substantive validity/utility-gap statement.
- **L156 + L159 — CONDENSE + CUT.** Merged the closing "In sum" paragraph's duplicated thesis
  restatement into one sentence; kept the reframing question and the forecast claim.
- **L182 — CLARIFY.** Gave the developmental-risk "state" clause a governing verb and separated it
  from the surrounding limitation list so the recall-noise-vs-true-change logic reads cleanly.

---

## Deviations from the literal flags (for your review)
1. **Methods L345** — kept (would break a First/Second/Third enumeration; genuinely different from the annex corollary).
2. **Results L213 block** — applied the compression (cut the duplicate subsection) but kept the three figures in place rather than physically exporting the block to an annex. The full move is still available on request.
3. **The four "move-to-supplement" items** (Methods L353, Results L597, Discussion L101/L105) were executed as **deletions from the main text, not physical relocations** — in each case the content was already covered by an existing cross-reference to the target annex, so re-copying it would have duplicated rather than moved it. No annex `.tex` file was edited.
4. **Back-matter candidate deferred** (`90_boilerplate.tex` L~17, CONDENSE, low severity) — outside the approved 31-item main-text set; left unchanged pending your go-ahead.
5. Wherever a flagged number appeared twice, the value was **kept at one location** and removed only at the other — no statistic left the paper.

All edits are committed to git and fully reversible (`git revert` / `git checkout`).
