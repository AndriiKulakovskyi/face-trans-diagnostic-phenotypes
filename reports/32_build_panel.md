# 32 — per-visit tables (V1, V2) + the V0 standardization spec

**V0 round-trip QC — ✅ PASS** (max |apply_spec(V0) − prepare().M| = 0.0e+00, NaN masks identical = True). The frozen V0 transform reproduces the fitted V0 matrix exactly, so V1/V2 are scored on the same scale the certified loadings live on (genuine change is preserved, not re-centred).

- Spec: **87 indicators** (18 lognormal with a frozen V0 log-min) → `data/processed/v0_standardization_spec.json` (family / sign / log-min / mean / sd per item).

## Per-visit coverage (modeled continuous block, on the frozen V0 scale)
| visit   |   n_patients |   n_items |   obs_cells |   mean_coverage |   out_of_v0_support_cells |   cov_bp |   cov_sz |   cov_dr |
|:--------|-------------:|----------:|------------:|----------------:|--------------------------:|---------:|---------:|---------:|
| V0      |         9013 |        87 |      526712 |           0.672 |                         0 |    0.763 |    0.428 |    0.62  |
| V1      |         4270 |        86 |      181580 |           0.494 |                         0 |    0.551 |    0.314 |    0.5   |
| V2      |         2958 |        87 |      138201 |           0.537 |                         1 |    0.606 |    0.277 |    0.553 |

- `out_of_v0_support_cells` = follow-up cells whose raw value falls outside V0's lognormal support (→ NaN, treated as missing — never imputed, never clipped).
- Data density thins with the panel (mean coverage V0 0.67 → V1 0.49 → V2 0.54); every patient is still scored from their own observed cells, with uncertainty propagated.

## Artifacts
- `data/processed/baseline_v{1,2}.parquet` — raw harmonized modeled indicators (gitignored).
- `data/processed/v0_standardization_spec.json` — the frozen V0 transform (gitignored).
- `reports/32_coverage_by_visit.csv` · `docs/figures/32_coverage.png`.

Next: stage 33 (G1 longitudinal measurement invariance) → then stage 34 scores this panel.