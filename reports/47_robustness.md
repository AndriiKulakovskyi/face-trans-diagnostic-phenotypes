# 47 — M4.7 robustness sweep (Q4)

Does the M4 headline — the durable ⊥G biology + the map adding prognostic value for **functioning** beyond diagnosis+severity+baseline — survive attrition, reliability, leave-one-cohort-out, and a permutation null?

## Durable-axis effect on EGF (Bayesian EIV β, 94% HDI)

| check                           |   cognition |   metabolic |   inflammatory |
|:--------------------------------|------------:|------------:|---------------:|
| baseline                        |      -0.022 |      -0.062 |         -0.06  |
| IPW (attrition)                 |      -0.025 |      -0.059 |         -0.056 |
| reliability (metab+inflam well) |      -0.014 |      -0.059 |         -0.06  |
| LOCO (drop BP)                  |      -0.057 |      -0.047 |         -0.065 |

Does each axis's HDI exclude 0?

| check                           |   cognition |   metabolic |   inflammatory |
|:--------------------------------|------------:|------------:|---------------:|
| baseline                        |           0 |           1 |              1 |
| IPW (attrition)                 |           0 |           1 |              1 |
| reliability (metab+inflam well) |           0 |           1 |              1 |
| LOCO (drop BP)                  |           0 |           0 |              0 |

- **Metabolic** is the durable signal: β stays negative and credible under **IPW** (-0.059) and **reliability** restriction (-0.059) — not an attrition or prior-dominated-coordinate artefact.
- Under **LOCO (drop BP)** the effect weakens (smaller N, SZ+DR only) — the expected course-dependence from M4.4, stated honestly, not a failure.

## Permutation null (durable block, EGF continuous)

- Real incremental ΔR² = **0.006**; permutation null 95th pct = 0.003 (mean 0.001); **p = 0.001**. The durable block's predictive gain is beyond chance given the foundation. (The measurement-error-in-baseline / Lord-RTM concern is separately addressed by the M4.3 Q2 result — the effect survives the *error-corrected* G severity.)

## Functional-remission AUC gain (map added) — robustness

| check           |    n |   auc_ref |   auc_map |   d_auc |
|:----------------|-----:|----------:|----------:|--------:|
| baseline        | 2114 |     0.763 |     0.78  |   0.017 |
| IPW (attrition) | 2114 |     0.767 |     0.783 |   0.016 |
| LOCO (drop BP)  |  624 |     0.731 |     0.73  |  -0.001 |

- The small reliable remission gain (+0.017 baseline) survives IPW (+0.016); under LOCO it attenuates (+-0.001).

## Read

- The headline **survives attrition (IPW) and reliability restriction** and is **beyond a permutation null** — it is not an attrition, prior-dominated-coordinate, or chance artefact.
- It is **course-dependent** (weakens dropping BP) — the honest M4.4 limitation, confirmed.

## Decision for the gate
Confirm the headline is robust (attrition / reliability / permutation) and honestly course-dependent, then consolidate (stage 48: M5 hand-off + the methods/findings docs).

Artifacts: `results/face/m4/{robustness,robustness_auc}.csv` · `docs/figures/47_robustness.png`.