# 23b — archetype location (anchor) uncertainty (P3-04/05)

Anchors re-fit across **80** states (40 M1 posterior draws + 40 patient bootstraps), Hungarian-aligned to the reference. Reports each extreme phenotype's peak axis, its **profile HDI** (anchor-location uncertainty, which the fixed-anchor projection omitted), and a per-archetype stability (min Tucker congruence vs the reference).

## Per-archetype location + stability
| archetype   | name                                              |   dom_share | dominant_axis      |   peak | peak_HDI      |   mean_profile_SD |   min_tucker |
|:------------|:--------------------------------------------------|------------:|:-------------------|-------:|:--------------|------------------:|-------------:|
| A1          | ↑suicidality ↑developmental_risk ↑metabolic       |       0.015 | suicidality        |   8.16 | [2.27, 9.91]  |             1.044 |        0.429 |
| A5          | ↑inflammatory ↑substance ↓suicidality             |       0.016 | inflammatory       |   6.83 | [1.09, 7.01]  |             0.955 |        0.134 |
| A7          | ↑mania_activation ↑sleep                          |       0.055 | mania_activation   |   5.08 | [2.65, 6.82]  |             0.558 |        0.52  |
| A6          | ↑developmental_risk ↓metabolic ↑sleep             |       0.086 | developmental_risk |   5.03 | [4.63, 5.28]  |             0.532 |        0.027 |
| A4          | ↑metabolic ↓suicidality ↓developmental_risk       |       0.132 | metabolic          |   3.77 | [-0.30, 4.51] |             0.691 |        0.036 |
| A3          | ↑sleep ↓cognition ↓developmental_risk             |       0.164 | sleep              |   2.59 | [-1.04, 3.51] |             0.679 |        0.048 |
| A2          | ↑cognition ↑overall_severity ↓suicidality         |       0.168 | cognition          |   2.25 | [0.26, 6.51]  |             0.756 |        0.121 |
| A0          | ↓overall_severity ↓inflammatory ↓mania_activation |       0.364 | overall_severity   |  -1.38 | [-1.87, 1.73] |             0.624 |        0.248 |

- **Rare corners** (↑suicidality ↑developmental_risk ↑metabolic, ↑inflammatory ↑substance ↓suicidality): their peaks carry the widest HDIs (the tails are skewed-biomarker-driven and sparsely populated), now reported as intervals rather than points — so the rare-archetype claims are uncertainty-qualified.
- archetype stability: min Tucker congruence across re-fits **0.027** (worst archetype); mean profile SD **0.730**.

## Artifacts
- `reports/23b_archetype_location.csv` — peak axis, peak HDI, stability per archetype.
- `reports/23b_archetype_profiles_hdi.csv` — full per-(archetype, dim) mean + HDI.
