# 55 — M5.2a propensity + overlap (the identification gate)

`P(treat | severity[CGI-S + error-corrected G] + DSM-5 arm + demographics + the 9 map coordinates)` per question × contrast. **Overlap decides estimability**; balance (max |SMD|) before vs after stabilized IPTW shows whether weighting can render the arms comparable.

## Overlap + balance by question × mode

| question         | mode              |   n_treated |   n_control |   frac_in_support |   max_smd_before |   max_smd_after | verdict                                  |
|:-----------------|:------------------|------------:|------------:|------------------:|-----------------:|----------------:|:-----------------------------------------|
| lithium_bp       | active_comparator |        1065 |        1077 |          0.995331 |            0.298 |           0.012 | estimable                                |
| lithium_bp       | on_off            |        1065 |        1231 |          0.996516 |            0.372 |           0.028 | estimable                                |
| clozapine_sz     | active_comparator |         175 |         517 |          0.981214 |            0.441 |           0.607 | estimable — residual imbalance (caution) |
| clozapine_sz     | on_off            |         175 |        1918 |          0.988055 |            0.262 |           0.048 | estimable                                |
| antipsychotic_bp | active_comparator |        1700 |         580 |          0.992982 |            0.711 |           0.093 | estimable                                |
| antipsychotic_bp | on_off            |        1700 |         626 |          0.993981 |            0.701 |           0.078 | estimable                                |

## Read
- **Active-comparator** is the primary contrast (both arms treated → indication more similar). `on_off` is the higher-powered sensitivity.
- A `max_smd_after` ≤ 0.1 is good balance, ≤ 0.25 acceptable; `frac_in_support` is the share of patients inside the common propensity range.
- **Channeled** (poor overlap) questions are reported as **non-estimable** — the honest outcome of confounding by indication, not a failure to find an effect.

## Decision for the gate
Carry the **estimable** question×mode cells into the moderation stage (56); report channeled ones as non-estimable. Per-patient PS + IPTW persisted to `results/face/m5/propensity_*.parquet`.

Artifacts: `results/face/m5/propensity_summary.csv` · `docs/figures/55_overlap.png`.