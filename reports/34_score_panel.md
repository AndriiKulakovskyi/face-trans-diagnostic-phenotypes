# 34 — G2: longitudinal coordinate + membership panel (V0 → V1 → V2)

Per (patient, visit) on the **frozen V0 scale**: 9-dim coordinates + uncertainty, Arm-B (G-residualized) + Arm-A archetype memberships, and the per-axis G1 license. V0 reused from M2.0 (prep validated bit-exact); V1/V2 newly projected under fixed certified parameters (no re-fit).

- **Panel rows:** 16,241 — V0 9,013 · V1 4,270 · V2 2,958.
- **V0 QC:** Arm-A dominant archetype reproduces M2's `patient_strata` at **99.9%** agreement (validates the frozen-scale scoring + the simplex projector).
- **Explicit-projection convergence (V1/V2):** V1 R-hat 1.042/div 0 · V2 R-hat 1.071/div 0.

## Coordinate trajectories (cohort-mean, frozen scale)
Mean coordinate per axis per visit — licensed axes carry patient-change meaning; inflammatory partial, the 3 explicit axes descriptive (per the G1 license).

|                    |     V0 |     V1 |     V2 |
|:-------------------|-------:|-------:|-------:|
| overall_severity   |  0.087 | -0.206 | -0.316 |
| cognition          | -0.005 |  0.069 | -0.13  |
| metabolic          | -0.006 |  0.07  |  0.098 |
| inflammatory       | -0.004 |  0.03  |  0.033 |
| sleep              | -0.011 | -0.088 | -0.085 |
| mania_activation   | -0.009 | -0.135 | -0.2   |
| suicidality        |  0.029 | -0.766 | -0.782 |
| developmental_risk | -0.01  | -0.198 | -0.199 |
| substance          | -0     | -0.067 | -0.079 |

- License attached per axis: overall_severity=invariant, cognition=invariant, metabolic=invariant, inflammatory=partial, sleep=invariant, mania_activation=not-tested, suicidality=not-tested, developmental_risk=invariant, substance=not-tested.

## Artifacts (results/face/m3/, gitignored)
- `panel_coords.parquet` — the tidy (patient_uid, visit) substrate (coords + memberships + license + retention).
- `panel_draws.npz` — [200, 16,241, 9] posterior draws (the uncertainty arm for G3/G4).
- `proj_V{1,2}.npz` — cached explicit projections.

Runtime 2.1 min. Next: stage 35 (G3 trait/state).