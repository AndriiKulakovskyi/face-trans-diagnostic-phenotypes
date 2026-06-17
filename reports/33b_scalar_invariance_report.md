# 33b — scalar (intercept) invariance for latent-mean-change claims (P4-01)

Per continuous home item: ANCOVA `raw_item ~ latent + visit`; the visit coefficient (Δα, standardized) is the intercept shift NOT explained by the latent change. |Δα| HDI (94%) excluding 0 ⇒ that item's intercept drifts (non-scalar). A population latent-mean change is only a clean *patient* change where the intercepts are scalar-invariant.

## Per-axis scalar-invariance verdict
| axis               |   n_items |   n_signif_drift_largeN |   max_abs_delta_alpha |   median_abs_delta_alpha | scalar_verdict   |
|:-------------------|----------:|------------------------:|----------------------:|-------------------------:|:-----------------|
| overall_severity   |        20 |                      14 |                 0.074 |                    0.042 | scalar-invariant |
| sleep              |        18 |                      16 |                 0.305 |                    0.071 | partial          |
| developmental_risk |        24 |                       4 |                 0.688 |                    0.133 | partial          |

## Reading
- The verdict is on the **magnitude** of Δα (standardized intercept drift), not significance: at N in the thousands a trivial ~0.04-SD drift is 'significant' (`n_signif_drift_largeN` is a power artefact), so the practical criterion is |Δα| relative to the reported ~0.3–0.9-SD latent slides.
- **overall_severity** is scalar-invariant (drifts ≤ ~0.07 SD) → its latent-mean slide is a genuine patient change, not a changed ruler — the mean-change claim there is supported.
- Axes flagged **partial / non-scalar** (a few items drift ~0.3–0.7 SD) → soften those mean-change claims toward rank/shape change; developmental's larger drifts are consistent with CTQ **recall noise** (P4-03).
- Binary **suicidality** items need a logistic *threshold* test (not run here); its mean-change claim carries that caveat until tested.
