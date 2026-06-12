# 01 — model-ready V0 baseline (build + persistence)

- **N = 9,013 patients** (full V0, no completeness selection): BP 6,252 · SZ 2,209 · DR 552

- **Modeled indicators: 143** (continuous 88 · explicit 55).
- Mean cell missingness across modeled indicators: **39.8%** — NaN preserved, never imputed.
- Skip-logic structural-zero decoding applied (`apply_skip_logic=True`).
- Best/worst 3-cohort coverage: max 0.94 (suoccur_alcool), min 0.00 (ltsv04).
- **2 indicator(s) with < 30 obs** (ltsv04, ltsv05) — below the engine's min-observation guard, auto-skipped at fit; effective modeled set **141**.
- **Recruitment sites: 21** (administrative — persisted to `data/processed/site_v0.parquet` for the §8 site bootstrap; NOT modeled): BP 15 · SZ 12 · DR 13 distinct sites per cohort.

Artifacts: `data/processed/{baseline_v0,indicator_metadata}.parquet` + `site_v0.parquet` (gitignored) · `reports/01_coverage_by_indicator.csv` · `01_site_coverage.csv`.