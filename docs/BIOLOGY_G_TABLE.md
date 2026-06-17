# Biology–G: the one definitive table (resolves P0-06)

The stale **"92–98% of variance independent of G"** came from a superseded provisional mixed-fit read
(metabolic r ≈ 0.28 → 1−r² ≈ 92%). With the current correlated-G continuous-backbone estimates the
numbers are higher. This is the single source of truth; purge "92–98%" from `RESULTS.md`,
`MEASUREMENT_MAP_EXPLAINED.md`, `STATE.md`, and `report/sections/05_results.tex`.

## Correlated-G factor correlation with G, and variance independent of G

(Correlated-G marginalized model, N ≈ 2,000 balanced, 2 seeds; `reports/10_covariate_sensitivity.csv`.)

| Axis | Φ(G, axis), unadjusted | r² | **1 − r² (variance independent of G)** | Φ(G, axis), covariate-adjusted† |
|---|---:|---:|---:|---:|
| **inflammatory** | 0.071 | 0.005 | **99.5%** | 0.056 |
| **metabolic** | 0.124 | 0.015 | **98.5%** | 0.058 |
| cognition | 0.385 | 0.148 | 85.2% | 0.229 |
| sleep | 0.422 | 0.178 | 82.2% | 0.409 |

Bifactor direct |λ_G| (the stricter identification): metabolic 0.08, inflammatory 0.07.

† Covariate-adjusted = each item residualized on age(spline)+sex+education+site before the factor model
(`prepare(covariate_adjust=True)`; `docs/COVARIATE_SENSITIVITY.md`). Adjustment **lowers** metabolic~G
(0.124→0.058) — the age/sex/site confounding was inflating it — so biology is **more** independent of
functional burden after adjustment, not less.

## Approved wording

> Metabolic and inflammatory burden carry ≈ **98.5–99.5%** of their variance independent of a general
> functional-burden axis (G) — they are **largely (not strictly) independent** of overall functional
> severity, and the least severity-entangled of the nine domains. The independence **survives and
> strengthens** under covariate adjustment.

Do **not** write "orthogonal" (reserve that for the by-construction bifactor null) or "92–98%".
