# 31 — G6: attrition & informative dropout

V0 roster **N = 9,013**; retained at V1 **47.4%** (4,270), at V2 **32.8%** (2,958). Logistic retention ~ the 9 V0 coordinates + cohort (94% CIs; coordinates are z-scored, so OR is per +1 SD on that axis). The question: is the retained sample a *fair* draw from the V0 map, or does position predict who stays?

## Informative-dropout verdict — **informative (MAR-given-V0)**
- **Severity (G):** OR(V1) = **0.97** [0.92, 1.03] → **severity-neutral** — global burden does not predict dropout per +1 SD; the informative signal is on other axes, not severity.
- 4/9 axes are individually informative for V1 retention (94% CI excludes OR=1).
- **Biology corners** (metabolic/inflammatory) show informative dropout: inflammatory OR 0.94 — biology trajectories must carry the IPW caveat.

### Retention odds ratios per V0 axis (per +1 SD)
| axis               |   retained_V1 |   retained_V2 |
|:-------------------|--------------:|--------------:|
| overall_severity   |         0.972 |         0.915 |
| cognition          |         0.825 |         0.829 |
| metabolic          |         1.045 |         1.057 |
| inflammatory       |         0.936 |         0.909 |
| sleep              |         0.958 |         0.999 |
| mania_activation   |         0.93  |         0.898 |
| suicidality        |         1.029 |         1.007 |
| developmental_risk |         0.916 |         0.927 |
| substance          |         1.006 |         1.114 |

- Age/sex-adjusted sanity check (V1, N=9,006): severity OR 0.97 (vs 0.97 unadjusted) — direction unchanged.

## Stayers vs droppers — V0 coordinate profile (Cohen's d, V1)
Positive d = stayers score higher on that axis at V0.
| axis               |   mean_stayer |   mean_dropper |   cohens_d |
|:-------------------|--------------:|---------------:|-----------:|
| overall_severity   |         0.05  |          0.12  |     -0.085 |
| cognition          |        -0.061 |          0.047 |     -0.144 |
| metabolic          |        -0.005 |         -0.007 |      0.002 |
| inflammatory       |        -0.026 |          0.016 |     -0.054 |
| sleep              |        -0.031 |          0.006 |     -0.042 |
| mania_activation   |        -0.05  |          0.028 |     -0.095 |
| suicidality        |         0.036 |          0.024 |      0.012 |
| developmental_risk |        -0.063 |          0.038 |     -0.095 |
| substance          |        -0.013 |          0.012 |     -0.044 |

## Dropout reasons (descriptive; captured for M4, not analysed here)
| cohort   |   n_lost |   refusal |   moved |   diagnosis_change |   deceased |   unknown |   other |   coded |
|:---------|---------:|----------:|--------:|-------------------:|-----------:|----------:|--------:|--------:|
| bp       |      787 |       336 |      78 |                 60 |         19 |        57 |     236 |       0 |
| sz       |      531 |       224 |      22 |                 17 |         12 |        72 |     184 |       0 |
| dr       |      106 |         0 |       0 |                  0 |          0 |         0 |       0 |     106 |

- **Diagnosis-change exits** ('Changement de diagnostic'): BP **60**, SZ **17** (DR reasons are coded, not decoded) — the only internal trace of DSM-5 instability (§A). It is an *exit* signal and the in-data `arm` never updates, so the head-to-head is deferred to M4.
- Deaths (sentinel-corrected dates): BP 20, SZ 12, DR 5.

## Guard & hand-off
- **Dropout ≠ improvement.** Because retention **is** position-dependent, G3/G4 report **completers-only AND all-available**, and the all-available trends carry an **IPW-of-retention** sensitivity refit (weights in `results/face/m3/ipw_weights.parquet`). Divergence between naive and IPW estimates is flagged on every affected figure.
- Conditions: G3 (trait/state) and G4 (persistence). The retained sample is **not** a neutral draw — survivorship is a live threat, handled by IPW + completers-vs-all.

Artifacts: `reports/31_{informative_dropout,stayer_dropper,dropout_reasons}.csv` · `docs/figures/31_attrition.png` · `results/face/m3/ipw_weights.parquet`.