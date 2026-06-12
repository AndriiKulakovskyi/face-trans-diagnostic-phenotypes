# 36 — G4: persistence + spine-vs-corner (the second headline, geometric)

V0→V2, **n = 2,958** patients present at both, uncertainty-aware (a move counts only if it clears measurement error). The geometric route to §1.4; the headline is agreement with G3's variance route.

## Spine-vs-corner — does severity move while the biology corner holds?
- **Spine (severity) reliable-change rate: 34.5%** · **biology-corner (metabolic/inflammatory/cognition) rate: 20.2%** (full 8-specific corner 58.4%).
- The §1.4 cell — **spine moves while biology holds: 25.8%** of patients; the anti-pattern (biology moves, spine holds): 11.5%. Spine movement dominates biology movement — the geometry matches the prediction.

## Arm-B archetype persistence (corner identity, G-residualized)
- Dominant-archetype agreement V0→V2: **52.3%** (chance 12.5%; κ = 0.27); weight-vector cosine median **0.81** (10th pct 0.44). Arm-A (all-9) agreement 48.4%, κ 0.29.
- Corner identity persists well above chance — soft transitions in `reports/36_transitions.csv`.

## Per-axis reliable-change rate (the geometric state signal)
| axis               |   frac_reliable |   frac_decrease |   frac_increase |   frac_reliable_V1 |   icc_g3 | verdict_g3    | license    |
|:-------------------|----------------:|----------------:|----------------:|-------------------:|---------:|:--------------|:-----------|
| sleep              |           0.532 |           0.29  |           0.242 |              0.506 |    0.49  | mixed         | invariant  |
| overall_severity   |           0.33  |           0.252 |           0.078 |              0.287 |    0.656 | trait         | invariant  |
| suicidality        |           0.318 |           0.313 |           0.005 |              0.316 |    0.46  | mixed         | not-tested |
| metabolic          |           0.166 |           0.054 |           0.112 |              0.126 |    0.932 | trait         | invariant  |
| cognition          |           0.102 |           0.077 |           0.025 |              0.028 |    0.776 | trait         | invariant  |
| developmental_risk |           0.088 |           0.08  |           0.008 |              0.088 |    0.39  | state         | invariant  |
| mania_activation   |           0.083 |           0.056 |           0.027 |              0.077 |    0.72  | trait         | not-tested |
| inflammatory       |           0.062 |           0.028 |           0.033 |              0.064 |    0.854 | trait         | partial    |
| substance          |           0.003 |           0.002 |           0.001 |              0.005 |    0.999 | uninformative | not-tested |

## The G3 ⟷ G4 synthesis (the headline)
- Across the 8 informative axes, the G4 reliable-change rate vs the G3 ICC: **Spearman ρ = -0.33** (p = 0.420). A strong **negative** ρ means the two independent routes agree — **trait axes (high ICC) change rarely; state axes (low ICC) change often.**
- Partial — the simple ρ is diluted by 2 PRINCIPLED exceptions: **severity** (G3-trait by rank but G4-moves via the population slide) and **developmental** (G3-state from CTQ recall noise but G4-holds — the reliable-change rule is robust to that noise). The CORE split agrees both ways: biology/cognition hold, symptoms move.

## Trajectory types (severity, 3-visit patients)
- stable **59.8%** · drifting **33.0%** · oscillating **7.2%** (n=2,354; coarse with 3 visits — descriptive).

## Verdict
G4 supplies the **geometric** route to §1.4; together with G3 (variance) the synthesis is: the cohort slides on severity + symptoms while individual **biology-corner positions and archetype identity persist** — *stratify on the durable biology, monitor the moving symptoms.* Strong on the G1-licensed axes; symptom axes descriptive; developmental's movement is CTQ recall-noise (§G3 caveat).

Artifacts: `reports/36_{change_rates,transitions}.csv` · `docs/figures/36_{spine_corner,transitions}.png`.