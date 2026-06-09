# 21 — M2.1 structure-discovery gate (cluster vs continuum vs branched)

The reported SHAPE verdict for the M1 9-D coordinate cloud (§3.1), run on both G-arms and uncertainty-aware over M1 draws — *before* any mixture is fit. Coordinates on native latent z-scale; embeddings are viz-only (never a clustering input).

## Verdict — Arm A: **continuum** · Arm B: **continuum**
**Lead representation (per §3.1): archetypes (continuum-honest soft view).** The other view is reported alongside.

### Arm A (all 9 — severity×profile) — verdict: **continuum** (clustered-signals 1/6)
- Hopkins **0.85** (≈0.5 continuum · →1 clustered)
- GMM-BIC best K **12**, ΔBIC(best vs K=1) 22791, monotone-decreasing **True** (monotone ⇒ over-segmenting, no interior optimum)
- silhouette peak **0.174** (<0.15 ⇒ no separation) · gap-stat K_opt **1**
- dip PC1 p **0.996**, axes multimodal (p<.05) **2**/10
- HDBSCAN clusters **0**, noise frac **1.00**
- uncertainty-aware (over draws): Hopkins 0.81±0.01, GMM K_best mode **4**, distribution {3: 3, 4: 15, 5: 2}

### Arm B (8 specifics — pure profile) — verdict: **continuum** (clustered-signals 2/6)
- Hopkins **0.86** (≈0.5 continuum · →1 clustered)
- GMM-BIC best K **11**, ΔBIC(best vs K=1) 20880, monotone-decreasing **False** (monotone ⇒ over-segmenting, no interior optimum)
- silhouette peak **0.191** (<0.15 ⇒ no separation) · gap-stat K_opt **1**
- dip PC1 p **0.994**, axes multimodal (p<.05) **2**/9
- HDBSCAN clusters **0**, noise frac **1.00**

## Mapper (lens = severity)
- 11 nodes · 10 edges · **1 connected component(s)** (1 chain ⇒ continuum; flares ⇒ branched; multiple islands ⇒ clusters). See figure.

## Reading
- **Continuum / weak-cluster** evidence ⇒ the coordinate space is graded, not a set of natural kinds. The mixture is then reported as a *soft tessellation* and **archetypes lead** (extreme phenotypes + simplex membership) — still a valid, actionable probabilistic decision-region object.
- **Clustered** evidence ⇒ the mixture's discrete regions lead; archetypes complement.
- This is a *precondition* check (§1.7): it does not by itself make strata 'better than DSM-5' — that is the M4/M5 predictive/treatment head-to-head.

## Figures
- `docs/figures/21_selection.png` — BIC / silhouette / gap vs K (both arms).
- `docs/figures/21_embedding.png` — UMAP by cohort / DSM-5 subtype / severity / inflammatory.
- `docs/figures/21_mapper.png` — Mapper graph (lens = severity, node colour = inflammatory).