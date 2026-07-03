# FACE-ATLAS — Cut-candidate review

_Editorial review generated 2026-07-03 · moderate threshold · scope: main text + annexes._

> **NOTHING HAS BEEN REMOVED.** This is a review list for you to adjudicate. No manuscript file was modified in producing it. Each entry gives a verbatim anchor quote so the passage can be located exactly; decide keep / cut / condense per row, then I can apply the ones you approve.

## Summary

**40 candidate passages** flagged across 10 sections.

| Dimension | Breakdown |
|---|---|
| Reason | 26 redundant · 9 low value · 5 confusing |
| Recommendation | 13 cut · 19 condense · 4 move to supplement · 4 keep but clarify |
| Severity | 3 high · 16 med · 21 low |
| Cross-ref caution | 6 sit near a referenced `\label` (all condense/reword — none delete a referenced object) |

### Highest-value cuts (start here)

| # | Location | Reason | Rec. |
|---|---|---|---|
| 1 | 01_introduction.tex L94 | Redundant: The same three commitments (no imputation, end-to-end uncertainty propagation, diagnosis-b… | **CUT** |
| 2 | 03_results.tex L213 | Low value: Four consecutive subsections (information accumulation/minimal-indicator rule, 'A few well… | **MOVE→suppl.** |
| 3 | 03_results.tex L280 | Redundant: This entire subsection restates the closing paragraph of the preceding subsection (shared … | **CUT** |

---

## Detailed candidates by section

### Introduction
`sections/01_introduction.tex` — 6 candidate(s)

**1. Line ~94 — Redundant**  
`CUT` · severity high · confidence high

> as an \emph{integration discipline} that binds three commitments rarely held jointly

- **Why flagged:** The same three commitments (no imputation, end-to-end uncertainty propagation, diagnosis-blind fixed map) were already enumerated in full one paragraph earlier ('assembled from three commitments that are rarely held jointly'), making this closing restatement pure repetition.

**2. Line ~32 — Redundant**  
`CONDENSE` · severity med · confidence high

> we treat the pipeline itself---not any one finding---as the contribution

- **Why flagged:** This framing of the pipeline-as-contribution is restated almost identically in the closing paragraph ('The contribution is the measurement model...as an integration discipline'), so the claim appears twice with no added content the second time.

**3. Line ~84 — Redundant**  
`CUT` · severity med · confidence med

> which we present as a property the method \emph{reveals} rather than the premise it was built on

- **Why flagged:** This restates, almost word for word, the earlier claim that 'the immunometabolic dissociation...is the existence proof that this discipline reveals structure a looser measurement would blur,' adding no new content.

**4. Line ~85 — Low value**  
`CUT` · severity med · confidence med

> We report the modest and null results as deliberately as the positive ones

- **Why flagged:** A self-congratulatory methodological virtue statement that adds no new information about the study's design or findings and reads as filler at the end of a dense paragraph.

**5. Line ~90 — Confusing**  
`CLARIFY (keep)` · severity med · confidence med

> a correlated-\Gfac{} arm that turns the biology--severity separation into a measured quantity

- **Why flagged:** The undefined macro/jargon term '\Gfac' is dropped into a dense closing summary sentence without prior definition in this section, leaving the clause opaque.

**6. Line ~91 — Redundant**  
`CONDENSE` · severity low · confidence med

> an archetypal representation of a continuous patient space, and a downstream errors-in-variables prognosis analysis

- **Why flagged:** This closing paragraph re-lists the four pipeline stages (measurement, structure/archetypes, temporal coherence, prognosis) already detailed with the (i)-(iv) breakdown earlier, functioning as a near-duplicate recap rather than new synthesis.


### Methods
`sections/02_methods.tex` — 8 candidate(s)

**7. Line ~206 — Redundant**  
`CONDENSE` · severity med · confidence high

> the freed off-home cells shrink to near zero, recovering the same anchored map

- **Why flagged:** The ≈83%-shrinkage sparse-ESEM validation result is already reported twice earlier in the same subsection with the same number and conclusion.

**8. Line ~281 — Redundant**  
`CONDENSE` · severity med · confidence high

> versus cognition $0.39$ and sleep $0.42$ in the same freely-correlated model

- **Why flagged:** The identical immunometabolic-vs-cognition-vs-sleep correlation figures (0.10/0.39/0.42) were already given in the bifactor-robustness paragraph just above.
- **Cross-ref note:** this paragraph carries label(s) referenced elsewhere — `eq:corrg` (×3). Condense/reword in place; do **not** delete the labelled object.

**9. Line ~345 — Redundant**  
`CONDENSE` · severity med · confidence high

> reproduced the loadings and factor correlations to three decimals

- **Why flagged:** The flat-prior refit result (Tucker congruence 1.00) was already stated as Cor. corA:flat earlier in the measurement-model subsection.
- **Cross-ref note:** this paragraph carries label(s) referenced elsewhere — `eq:m-woodbury` (×1). Condense/reword in place; do **not** delete the labelled object.

**10. Line ~3 — Low value**  
`CUT` · severity low · confidence med

> This section gives a self-contained mathematical account of the model and the analysis

- **Why flagged:** Throat-clearing preamble that just restates the section title and points to the Supplement without adding technical content.

**11. Line ~80 — Redundant**  
`CUT` · severity low · confidence med

> their full derivation, with proofs, is in Supplement

- **Why flagged:** Duplicates the same Supplement pointer already given in the section's opening sentence.
- **Cross-ref note:** this paragraph carries label(s) referenced elsewhere — `eq:factors` (×4). Condense/reword in place; do **not** delete the labelled object.

**12. Line ~296 — Redundant**  
`CONDENSE` · severity low · confidence med

> Balancing by subsampling is rejected because matching the smallest cohort would discard

- **Why flagged:** Restates a claim already made in the Cohorts subsection (balancing by subsampling rejected in favour of cohort weighting) rather than folding the new 92% figure into that earlier statement.

**13. Line ~353 — Low value**  
`MOVE→suppl.` · severity low · confidence med

> a standalone maximum-likelihood arm adds no independent evidence

- **Why flagged:** A tangential negative-result aside about an arm that was not run, better suited to the Supplement than the main identification argument.
- **Cross-ref note:** this paragraph carries label(s) referenced elsewhere — `eq:m-woodbury` (×1). Condense/reword in place; do **not** delete the labelled object.

**14. Line ~356 — Redundant**  
`CUT` · severity low · confidence med

> diagnosis-balanced subsampling, site cluster-bootstrap and inverse-cohort weighting (minimum Tucker

- **Why flagged:** The inverse-cohort-weighting Tucker congruence of 0.96 was already reported earlier in the same subsection as one arm of the robustness battery.
- **Cross-ref note:** this paragraph carries label(s) referenced elsewhere — `eq:m-woodbury` (×1). Condense/reword in place; do **not** delete the labelled object.


### Results
`sections/03_results.tex` — 9 candidate(s)

**15. Line ~213 — Low value**  
`MOVE→suppl.` · severity high · confidence med

> A few well-chosen indicators recover most of each dimension

- **Why flagged:** Four consecutive subsections (information accumulation/minimal-indicator rule, 'A few well-chosen indicators', 'Designing efficient harmonised batteries') elaborate the same value-of-information / battery-design theme with three figures (localization, mincount, voi). This is a methodological digression that outweighs its narrative payoff in the main Results; strong candidate to compress to one subsection + move detail to a supplement.
- _Source: structural read (spans more than one paragraph)._

**16. Line ~280 — Redundant**  
`CUT` · severity high · confidence high

> convert the measurement model into an actionable specification: a common core battery

- **Why flagged:** This entire subsection restates the closing paragraph of the preceding subsection (shared battery, disorder-specific top-ups, pooled-item-parameter caveat) without new evidence.

**17. Line ~227 — Redundant**  
`CONDENSE` · severity med · confidence high

> the fix is a finer or graded severity instrument, not additional weak binary items

- **Why flagged:** This repeats, almost verbatim, the identical instrument-design recommendation already given for mania/substance-use in the preceding minimal-indicator-rule subsection.

**18. Line ~597 — Confusing**  
`MOVE→suppl.` · severity med · confidence med

> The sharper ``driven by activation and suicidality more than severity'' statement

- **Why flagged:** It invokes a 'coarse two-region tessellation' result never defined or shown in this section, so the comparison cannot be evaluated without the supplement.

**19. Line ~701 — Redundant**  
`CONDENSE` · severity med · confidence high

> The same decomposition, shown in full as a

- **Why flagged:** The 'trait/state thermometer' (Fig. thermometer) and its paragraph re-present the SAME trait/state variance decomposition already shown in Fig. traitstate one paragraph earlier; the text itself says 'the same decomposition'. Two figures for one result.
- _Source: structural read (spans more than one paragraph)._

**20. Line ~740 — Redundant**  
`CONDENSE` · severity med · confidence high

> against $0.026$ for DSM-5 category and $0.018$ for cohort

- **Why flagged:** The archetype-versus-DSM-5 eta-squared values and the 9.7-fold figure were already reported in the immediately preceding archetypes subsection.

**21. Line ~800 — Redundant**  
`CUT` · severity med · confidence high

> bipolar $27\%\!\to\!73\%$, depression $31\%\!\to\!72\%$, schizophrenia $9\%\!\to\!25\%$

- **Why flagged:** These exact per-cohort remission percentages were stated one paragraph earlier in the same subsection.

**22. Line ~8 — Low value**  
`CUT` · severity low · confidence high

> differ in age, sex distribution and chronicity, as expected for these disorders

- **Why flagged:** The aside adds no analytic content beyond noting that demographic variation across disorders is unsurprising.

**23. Line ~742 — Redundant**  
`CONDENSE` · severity low · confidence med

> This is compactness on the map's own axes, not a claim about outcome variance explained

- **Why flagged:** This caveat duplicates the same clarification already made in the archetypes subsection and again in the Figure 6 caption.


### Discussion
`sections/04_discussion.tex` — 8 candidate(s)

**24. Line ~101 — Redundant**  
`CUT` · severity med · confidence med

> the immunometabolic and severe, clean-biology corners are equally severe

- **Why flagged:** This restates the identical opposite-biology/comparable-severity contrast between the immunometabolic and clean-biology corners already given in full with the remission percentages in the opening paragraph.

**25. Line ~156 — Redundant**  
`CONDENSE` · severity med · confidence med

> The immunometabolic axis a patient keeps over time forecasts the functional state they move toward

- **Why flagged:** Restates the archetype-prediction and immunometabolic-corner-as-enrichment-target claims already made in detail in the preceding paragraph on incremental prediction.

**26. Line ~182 — Confusing**  
`CLARIFY (keep)` · severity med · confidence high

> developmental risk scores as ``state'' partly because of recall noise in childhood-adversity reports rather than true change

- **Why flagged:** This clause lacks a governing verb and is folded into an unrelated list of limitations, making the logical relationship (recall noise vs. true change) hard to parse.

**27. Line ~37 — Low value**  
`CONDENSE` · severity low · confidence low

> This is the precise property a biology-aware stratification can exploit: it separates patients who look equally ill

- **Why flagged:** Restates the immediately preceding sentence's finding (comparable burden, opposite biology, divergent prognosis) as generic editorial commentary rather than new content.

**28. Line ~53 — Low value**  
`CUT` · severity low · confidence med

> That a completely different method, in different cohorts, converges on a separable biology axis is strong convergent validity

- **Why flagged:** Self-congratulatory editorializing that merely re-labels the IMD-programme convergence point made in the preceding sentence rather than adding evidence.

**29. Line ~105 — Low value**  
`MOVE→suppl.` · severity low · confidence low

> Anchoring \Gfac{} on functioning alone also sidesteps the well-known instability in what a symptom-only general factor means

- **Why flagged:** Tangential aside referencing bifactor-instability literature that is not developed further and could be moved to supplement without loss to the main argument.

**30. Line ~149 — Low value**  
`CONDENSE` · severity low · confidence med

> an atlas explicit about the gap between demonstrated scientific validity and undemonstrated individual-level clinical utility is more useful than an overstated one

- **Why flagged:** Self-congratulatory framing about the paper's own rigor rather than a substantive claim about the data.

**31. Line ~159 — Redundant**  
`CUT` · severity low · confidence med

> Separating biological burden from clinical severity, and showing that the biological part is stable and prognostic, is the step that makes mechanistic stratification

- **Why flagged:** Closing sentence re-summarizes the thesis already stated near-verbatim in the section's opening paragraph.


### Back matter (Data/Code availability)
`sections/90_boilerplate.tex` — 1 candidate(s)

**32. Line ~17 — Redundant**  
`CONDENSE` · severity low · confidence med

> per-patient data are never required to regenerate the aggregate results

- **Why flagged:** This restates the point already made in the Data availability section that no per-patient data are needed/reproduced, just phrased differently.


### Annex A — Measurement model
`annex/A_measurement_model.tex` — 2 candidate(s)

**33. Line ~209 — Redundant**  
`CONDENSE` · severity low · confidence med

> any completed matrix would inject between-cell covariance

- **Why flagged:** The specific causal claim that imputing missing cells would inject spurious covariance that the model would misattribute to a latent factor is stated near-verbatim three times across the annex: the parenthetical in \ref{annA:gen} ('an invented value injects covariance the model would mistake for a factor'), the remark in \ref{annA:mixed} that a missing cell contributes no term, and this restatement in \ref{annA:obs}. Each occurrence carries the identical argument rather than adding a new facet of it.
- **Cross-ref note:** this paragraph carries label(s) referenced elsewhere — `propA:marg` (×3), `eqA:margdist` (×2). Condense/reword in place; do **not** delete the labelled object.

**34. Line ~306 — Redundant**  
`CUT` · severity low · confidence med

> positive evidence the 8-factor map is well-specified

- **Why flagged:** The conclusion that the freed-cross-loading validation exercise supports correct model specification is stated twice within a few sentences of the same subsection: first as 'is positive evidence the structure is well-specified, not an artefact of rigid zeros' and again immediately after the cross-loading table as 'positive evidence the 8-factor map is well-specified.' The second instance adds no new content beyond restating the first.


### Annex B — Stratification
`annex/B_stratification.tex` — 2 candidate(s)

**35. Line ~47 — Redundant**  
`CONDENSE` · severity low · confidence med

> reliably better separated than a structureless Gaussian blob

- **Why flagged:** The silhouette comparison (real 0.140 vs null 0.137±0.002, z=1.13, n.s.) and the conclusion 'not reliably better separated than a structureless Gaussian blob' are stated twice almost verbatim: once in the paragraph introducing the operative separability test, and again immediately after the z_sil equation restating the identical numbers and phrase.

**36. Line ~48 — Redundant**  
`CLARIFY (keep)` · severity low · confidence med

> Cluster tendency and cluster separability are thus dissociated in these data

- **Why flagged:** The claim that Hopkins captures uniformity/tendency rather than separability, and that tendency and separability are dissociated here, is made three times in close succession: in the 'continuum verdict does not require H→½' sentence, in the following 'high Hopkins value therefore does not indicate discrete clusters' paragraph, and again in this closing sentence, each restating the same logical point with only minor rewording.


### Annex C — Temporal coherence
`annex/C_temporal.tex` — 2 candidate(s)

**37. Line ~60 — Confusing**  
`CONDENSE` · severity med · confidence high

> consistent with the ICC ranks

- **Why flagged:** The geometric-route sentence asserts the archetype-weight result is 'consistent with the ICC ranks,' but the very next sentence explicitly disclaims this as 'an independent confirmation of the variance ranks' and reports a near-zero correlation (rho=0.07, p=0.87) between reliable-change rate and ICC. The two statements read as contradictory without reconciling language, leaving the reader unsure whether the geometric evidence supports the ICC ranking or not.

**38. Line ~69 — Redundant**  
`CONDENSE` · severity low · confidence med

> mania is data-limited (two indicators) and its trait/state verdict is not relied upon

- **Why flagged:** The identical caveat 'mania is data-limited (two indicators)' already appears verbatim in the Longitudinal measurement invariance subsection above; restating it unchanged here adds no new information beyond the added 'not relied upon' clause.


### Annex F — Metrics glossary (build letter F)
`annex/G_metrics.tex` — 1 candidate(s)

**39. Line ~70 — Confusing**  
`CONDENSE` · severity low · confidence med

> describes the space ${\approx}3\times$ more tightly at lower BIC

- **Why flagged:** The row reports a BIC gap of 185,600 vs 188,200 (ΔBIC ≈ 2,600) and then glosses this as the free model fitting 'approximately 3x more tightly.' A ΔBIC of that size corresponds to an astronomically large Bayes-factor-equivalent, not a modest 3-fold difference, and no transformation from BIC units to a '3x' ratio is given. A reader trying to reconcile the stated numbers with the '3x' claim will be confused about what quantity is actually being compared.


### Annex G — Variational re-estimation (build letter G)
`annex/H_variational.tex` — 1 candidate(s)

**40. Line ~183 — Redundant**  
`CLARIFY (keep)` · severity low · confidence med

> should be read as directionally correct but not as a magnitude

- **Why flagged:** The conclusion that 'NUTS is retained as the authority for Φ magnitudes' is stated nearly verbatim three times: once at the end of 'NUTS versus variational inference' ('This is why we keep NUTS as the authority for Φ magnitudes...'), again at the end of 'The one gap: inter-factor Φ' ('This is precisely the quantity for which NUTS is retained as the authority...'), and a third time in the closing Summary paragraph. All three convey the identical claim with no new information added.


---

## How this list was produced

- Every section and annex of the current (post-treatment-removal) manuscript was read in full.

- An independent handling-editor pass (reasoning model, per-section and per-annex) proposed candidates at a **moderate** threshold for the main text and a **higher** bar for annexes (annexes are expected to be detailed).

- Every anchor quote was **verified to appear verbatim** in its source file (0 fabricated anchors) and located to a line number.

- A cross-reference safety check mapped each flagged paragraph against every `\ref`/`\eqref`/`\edfig` in the build: passages carrying a referenced label are marked so a condense never orphans a cross-reference.

- Two structural candidates spanning multiple paragraphs (the duplicate trait/state 'thermometer' figure; the four-subsection measurement-design block) were added from the manual read.


_Line numbers are approximate (they shift as edits are applied); the verbatim quote is the reliable anchor._
