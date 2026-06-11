# 46 — M4.6 clinical value: AUC · calibration · net benefit

Does adding the transdiagnostic map to the clinician's model change *decisions*? Patient-level 5-fold cross-validated logistic models; the map = Arm-B archetype memberships (⊥G).

## Discrimination (cross-validated AUC)

| endpoint             |    n |   prevalence |   DSM-5 only |   map only |   reference (DSM-5+severity) |   reference + map |
|:---------------------|-----:|-------------:|-------------:|-----------:|-----------------------------:|------------------:|
| cgi_relapse          | 2345 |        0.102 |        0.604 |      0.532 |                        0.871 |             0.872 |
| egf_deterioration    | 2114 |        0.154 |        0.528 |      0.516 |                        0.726 |             0.729 |
| egf_remission        | 2114 |        0.432 |        0.66  |      0.639 |                        0.763 |             0.78  |
| egf_sustained_impair | 1555 |        0.261 |        0.704 |      0.653 |                        0.828 |             0.836 |

Paired AUC gain from adding the map to the reference (bootstrap CI):

| endpoint             |   dAUC_map | ci              |   p_gain>0 |
|:---------------------|-----------:|:----------------|-----------:|
| egf_remission        |      0.017 | [+0.009,+0.026] |      1     |
| egf_deterioration    |      0.003 | [-0.007,+0.013] |      0.735 |
| egf_sustained_impair |      0.008 | [+0.000,+0.015] |      0.979 |
| cgi_relapse          |      0.002 | [-0.004,+0.008] |      0.697 |

## Net benefit (decision-curve analysis)

For the adverse endpoints (flag-for-intervention decisions), the net benefit of `reference + map` vs `reference` and the treat-all / treat-none defaults across decision thresholds — `docs/figures/46_decision_curve.png`. A curve above the others over a clinically plausible threshold band = acting on that model yields more true flags net of false ones.

## Read

- **AUC**: `reference + map` vs the clinician's `reference` — the ΔAUC + CI says whether the map adds discrimination over diagnosis+severity+baseline; `map only` vs `DSM-5 only` is the raw classifier head-to-head.
- **Calibration** (`46_calibration.png`): predicted vs observed risk — usable risks, not just ranking.
- **Net benefit** translates discrimination into a decision: is the map worth acting on.

## Decision for the gate
Confirm the clinical-value verdicts; fold the AUC / net-benefit numbers into `docs/PROGNOSIS_ATLAS.md §5`, then proceed to the robustness sweep (47).

Artifacts: `results/face/m4/clinical_value.csv` · `docs/figures/46_{auc,decision_curve,calibration}.png`.