# 23b — archetype-count comparison (A = 5, 6, 7, 8)

> **⚠️ SUPERSEDED — native 9-d A-sweep. Do not cite these corner profiles/shares as current.**
> This corner-survival table is the **native 9-dimension** archetype sweep (separate `metabolic` +
> `inflammatory` axes, peaks/shares in native z-units — e.g. low-burden pole share 0.549, metabolic corner peak
> 3.1/3.7, inflammatory corner only at A=8). The reported map is the **8-factor Gaussian-copula** map with a
> single `immunometabolic` axis and an **A = 5** simplex. Canonical copula A = 5 corner profiles and population
> shares (A4 0.283 · A3 0.222 · A1 0.176 · A0 0.161 · A2 0.158): **[docs/STRATA_ATLAS.md](../docs/STRATA_ATLAS.md)**,
> **[docs/STRATA_FINDINGS.md](../docs/STRATA_FINDINGS.md)** (Result 3 + 4b), and
> `results/face/strata_oop/consolidate/archetype_profiles.csv`. Kept for provenance only.

Scree has no elbow (continuum) ⇒ A is a parsimony/interpretability choice; archetypes are stable at any A (M2.3 congruence 0.999). Below: which **axis-extreme corners** each A recovers, their peak z and population share. Higher = more burden. Arm A (full 9-d phenotype).

## Corner-survival matrix (peak z of the corner dominant on each axis; '-' = absent)
*The key question: at which A do the biology corners (metabolic, inflammatory) and the rare psychopathology tails (suicidality, mania) appear as their own phenotype?*
|                    | A=5   | A=6   | A=7   | A=8   |
|:-------------------|:------|:------|:------|:------|
| overall_severity   | -     | -     | -     | -     |
| cognition          | -     | 2.2   | 2.2   | 2.3   |
| metabolic          | 3.1   | 3.3   | 3.6   | 3.7   |
| inflammatory       | -     | -     | -     | 6.6   |
| sleep              | 2.8   | 2.8   | 2.6   | 2.6   |
| mania_activation   | -     | -     | 5.1   | 5.0   |
| suicidality        | 5.5   | 7.2   | 8.0   | 8.1   |
| developmental_risk | 5.2   | 5.2   | 5.1   | 5.1   |
| substance          | -     | -     | -     | -     |
| low-burden pole    | -0.2  | -0.5  | -0.6  | -0.3  |

## Explained variance
| A | ev |
|---|---|
| 5 | 0.608 |
| 6 | 0.681 |
| 7 | 0.737 |
| 8 | 0.791 |

## A = 5 — extreme phenotypes (by population share)
|   arch | defining            |   peak_z |   share |
|-------:|:--------------------|---------:|--------:|
|      3 | low-burden pole     |    -0.25 |   0.549 |
|      4 | ↑metabolic          |     3.06 |   0.171 |
|      1 | ↑sleep              |     2.77 |   0.126 |
|      0 | ↑developmental_risk |     5.18 |   0.094 |
|      2 | ↑suicidality        |     5.49 |   0.06  |

## A = 6 — extreme phenotypes (by population share)
|   arch | defining            |   peak_z |   share |
|-------:|:--------------------|---------:|--------:|
|      5 | low-burden pole     |    -0.54 |   0.405 |
|      4 | ↑metabolic          |     3.31 |   0.178 |
|      1 | ↑cognition          |     2.2  |   0.175 |
|      2 | ↑sleep              |     2.81 |   0.121 |
|      0 | ↑developmental_risk |     5.2  |   0.096 |
|      3 | ↑suicidality        |     7.15 |   0.025 |

## A = 7 — extreme phenotypes (by population share)
|   arch | defining            |   peak_z |   share |
|-------:|:--------------------|---------:|--------:|
|      2 | low-burden pole     |    -0.61 |   0.384 |
|      4 | ↑cognition          |     2.17 |   0.176 |
|      6 | ↑sleep              |     2.63 |   0.156 |
|      5 | ↑metabolic          |     3.61 |   0.132 |
|      1 | ↑developmental_risk |     5.14 |   0.087 |
|      3 | ↑mania_activation   |     5.1  |   0.054 |
|      0 | ↑suicidality        |     8.05 |   0.012 |

## A = 8 — extreme phenotypes (by population share)
|   arch | defining            |   peak_z |   share |
|-------:|:--------------------|---------:|--------:|
|      0 | low-burden pole     |    -0.29 |   0.369 |
|      2 | ↑cognition          |     2.3  |   0.165 |
|      3 | ↑sleep              |     2.59 |   0.159 |
|      4 | ↑metabolic          |     3.74 |   0.132 |
|      6 | ↑developmental_risk |     5.1  |   0.085 |
|      7 | ↑mania_activation   |     5.04 |   0.055 |
|      5 | ↑inflammatory       |     6.59 |   0.019 |
|      1 | ↑suicidality        |     8.13 |   0.015 |

## Reading
- A corner 'survives' when an archetype sits at that axis's positive extreme. Smaller A merges the rarer tails into neighbours; larger A resolves them as their own phenotype.
- Pick the smallest A that still resolves the corners you care about (esp. **metabolic + inflammatory** for the biology⊥G story). The choice is interpretability, not fit — every A is a valid soft basis for the same continuum.

Figure: `docs/figures/23b_compare.png` (profile heatmaps, A=5..8).