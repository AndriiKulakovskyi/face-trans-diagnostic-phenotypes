# 35 — G3: trait vs state decomposition (the first headline)

Per axis: between-patient **trait** σ²_b vs within-person **state** σ²_w, with the known M1 measurement variance **plugged** (so reliability can't be mistaken for state). ICC = σ²_b/(σ²_b+σ²_w); trait ≥0.6 · state ≤0.4 (94% HDI clearing 0.5) · else mixed. 4,874 multi-visit patients drive the split; all data used (no completeness selection).

## Trait/state profile — axes ordered by ICC (state → trait)

`license`: G1 status (trait/state claim is measurement-backed for invariant/partial; the not-tested axes are descriptive). `predicted`: the §1.4 geometric expectation.

| axis               |   icc |   icc_lo |   icc_hi | verdict       | predicted   |   pop_slide |   var_between |   var_within |   var_meas | license    |   icc_completers |
|:-------------------|------:|---------:|---------:|:--------------|:------------|------------:|--------------:|-------------:|-----------:|:-----------|-----------------:|
| developmental_risk | 0.39  |    0.34  |    0.438 | state         | trait       |      -0.166 |         0.348 |        0.545 |      0.451 | invariant  |            0.249 |
| suicidality        | 0.46  |    0.427 |    0.492 | mixed         | state       |      -0.886 |         0.233 |        0.273 |      0.315 | not-tested |            0.434 |
| sleep              | 0.49  |    0.473 |    0.508 | mixed         | mixed       |      -0.082 |         0.405 |        0.422 |      0.171 | invariant  |            0.492 |
| overall_severity   | 0.656 |    0.64  |    0.67  | trait         | state       |      -0.344 |         0.389 |        0.204 |      0.104 | invariant  |            0.663 |
| mania_activation   | 0.72  |    0.67  |    0.769 | trait         | state       |      -0.18  |         0.193 |        0.075 |      0.447 | not-tested |            0.813 |
| cognition          | 0.776 |    0.752 |    0.797 | trait         | trait       |      -0.157 |         0.387 |        0.112 |      0.482 | invariant  |            0.767 |
| inflammatory       | 0.854 |    0.833 |    0.874 | trait         | mixed       |       0.049 |         0.38  |        0.065 |      0.368 | partial    |            0.865 |
| metabolic          | 0.932 |    0.928 |    0.937 | trait         | trait       |       0.102 |         0.774 |        0.056 |      0.125 | invariant  |            0.938 |
| substance          | 0.999 |    0.994 |    1     | uninformative | mixed       |      -0.074 |         0.09  |        0     |      0.684 | not-tested |            0.999 |

- `pop_slide` = the cohort's V0→V2 population trend on that axis, **removed by the visit fixed effects before the ICC** — so ICC measures *individual* trait/state on top of any shared slide. A large slide with a high ICC means the cohort moves but individual *ranks* are preserved.

## The §1.4 test — does trait/state align with spine/corner?
- **Spine (overall_severity): ICC 0.66 [0.64, 0.67] → TRAIT** (predicted state). The severity spine is **trait** — not the predicted pure state; see caveats.
- **Biology corners (cognition / metabolic / developmental_risk):** developmental_risk 0.39→state, cognition 0.78→trait, metabolic 0.93→trait (predicted trait). Mixed result — see table.
- **Licensed-axis scorecard:** 3/6 licensed axes match the §1.4 prediction (the measurement-backed verdicts).
- **Measurement-error correction matters:** corrected ICC vs raw ICC (which charges measurement noise to state) diverges most where reliability is low — e.g. developmental_risk 0.56→0.39, suicidality 0.50→0.46, sleep 0.69→0.49.

## Survivorship (completers vs all-available)
- `icc_completers` (col above) refits on the V0+V1+V2 completers only. Large upward shifts vs `icc` would flag that the stable patients are retained (dropout biasing toward trait, per G6). Max |Δ| = 0.14 → mild — the trait/state verdicts are robust to attrition.

## Verdict
The trait/state profile is the **variance route** to the §1.4 prediction; G4 (stage 36) supplies the geometric route, and their agreement is the headline. Trait/state claims are strong on the G1-licensed axes (severity/cognition/metabolic/sleep/developmental), caveated on inflammatory (partial), descriptive on the not-tested explicit axes.

Artifacts: `reports/35_trait_state.csv` · `docs/figures/35_trait_state.png`.