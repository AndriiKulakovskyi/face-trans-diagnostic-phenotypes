# Soft operational regions on the Gaussian-copula map (M2) — findings

> **Reframe note.** M2 is a *coordinate system + reading guide*, not a typology. The 8-dim copula space is a
> **continuum**; the **continuous coordinates** are the load-bearing object, and the archetype corners /
> K-tessellation are **interpretation lenses**, not discovered decision-regions or natural kinds (M4 confirms:
> *operative K = none*). Canonical companions: [STRATA_OOP_ATLAS.md](STRATA_OOP_ATLAS.md),
> docs/STRATA_READING_GUIDE.md. Read the framing below as the *derivation* of those lenses, not a claim that
> patients fall into discrete strata.

> **Canonical M2 findings record (8-factor map, 2026-06-26; PI sign-off 2026-06-27).** Built on the
> **8-dimension** cohort-weighted full-N Gaussian-copula map — an immunometabolic biology axis (cardiometabolic
> + inflammatory markers on one factor) + the 3 earned cross-loadings (CTQ/PSQI → cognition) + **substance pinned
> orthogonal** (its cross-factor correlations are non-identifiable). Map: `copula/weighted_8d/hs_s5_merged_xc`
> (R̂ 1.03, 0 div). Engine [`src/face/strata/strata_model_oop.py`](../src/face/strata/strata_model_oop.py);
> driver [`notebooks/run_strata_model_oop.py`](../notebooks/run_strata_model_oop.py). Detailed atlas of what
> the K-family and A mean: [STRATA_OOP_ATLAS.md](STRATA_OOP_ATLAS.md).

## What this is

On the fixed 8-factor M1 map, M2 asks whether the transdiagnostic space carves into useful strata. The answer
is a clean transdiagnostic structure with **A = 5 stable archetypes**: a continuum (not biotypes), a nested
K-family with no privileged K, an archetype simplex
that separates biology ⊥ symptoms ⊥ severity, transdiagnostic and not-just-severity. The eight axes are
overall_severity (G), cognition, **immunometabolic**, sleep, mania_activation, suicidality, developmental_risk,
substance.

## Coordinate prep (the copula simplification)

The cohort-weighted full-N copula fit carries the per-patient explicit latents `f_e` for
[G, suicidality, developmental_risk, substance] in its posterior, so no extra full-N projection is needed. We
read `f_e` and condition the marginalized continuous specifics `f_m | f_e` under the shared Φ
(`scoring.coherent_joint_coords`), giving one coherent 8-dim posterior draw per sample. All **9,013 patients**
are scored on all **8 axes** (latent z-scale, 100% finite). Uncertainty propagates honestly: immunometabolic /
cognition / developmental tight, mania / substance / suicidality wide; substance is never "well" (only 2 SUD
binaries) and DR patients are prior-dominated — **no imputation**.

## Result 1 — it is a continuum

The structure gate (Hopkins / dip / silhouette / gap / GMM-BIC / HDBSCAN / Mapper, uncertainty-aware) returns
**continuum**: silhouette peak **0.146** (< 0.15 = no separation), gap k_opt = **1**, HDBSCAN **0 clusters**
(100% noise), Mapper a **single connected component**. Per-axis dip is unimodal for 6/8 axes (p > 0.99); only
**mania_activation** and **suicidality** are multimodal (dip p = 0) — the symptom axes have internal structure,
but the joint space does not separate. The single-Gaussian **falsification null** is decisive: the best
partition separates patients **no better than slicing a structureless Gaussian** — silhouette real **0.140** vs
null **0.137 ± 0.002 (z = 1.13, n.s.)**. So there are **no well-separated, reproducible clusters**; K is a
granularity convention, not a kind-count. Figure: `structure_panel.png`.

## Result 2 — the soft tessellation is a nested K-family (no privileged K)

XD-BIC is essentially flat across K (185.0k–185.6k, < 0.4%; the minimum is at **K = 4** but its stability
drops). We export the family and let downstream validity decide
(`results/face/strata_oop/consolidate/k_family_menu.csv`):

| K | XD-BIC | confident-dominant | seed-ARI | η² spec | η² G | η² mania | η² suicidality | η² immunometabolic |
|---|---|---|---|---|---|---|---|---|
| **2** (contract default) | 185,557 | 1.00 | 0.991 | 0.077 | 0.050 | 0.224 | 0.225 | 0.027 |
| **3** (BIC-near-best, stable) | 185,019 | 0.94 | 0.998 | 0.115 | 0.108 | 0.163 | 0.476 | 0.028 |
| 4 (BIC-min, less stable) | 185,006 | 0.88 | 0.663 | 0.136 | 0.203 | 0.229 | 0.523 | 0.057 |

**The headline nuance:** every K splits *first* on **psychiatric symptom burden** —
mania (η² 0.224) and suicidality (η² 0.225 → 0.476 → 0.523) — and **finer K progressively picks up severity (G:
0.050 → 0.108 → 0.203) and the immunometabolic gradient (0.027 → 0.057)** that K = 2 discards. Biology has no
density *gap*, so it never forms a region boundary at any K (it is an archetype/continuous feature — Result 3),
but finer tiling captures more of its continuous gradient. K = 3 is the sweet spot (BIC-near-best, seed-ARI
0.998, suicidality-dominant); K = 4 is richest on severity/biology but less stable (seed-ARI 0.663). **No K is
"the answer"** — the operative choice is M4/M5's. The `tess_*` columns carry **K = 2** as the M3 contract
default (smallest confidently-assignable, stable convention); the family ships as `tessfam_k{2,3,4}_*`.

## Result 3 — archetypes carry biology ⊥ symptom ⊥ severity (A = 5, stable)

Archetype granularity is chosen by **cross-seed stability** (largest A with min Tucker congruence ≥ 0.8). There
is a clean **stability cliff at A = 6**.

| A | explained variance | stability (Tucker) |
|---|---|---|
| 2 | 0.237 | 0.999 |
| 3 | 0.386 | 0.813 |
| 4 | 0.504 | 0.997 |
| **5 (selected)** | **0.601** | **0.979** |
| 6 | 0.665 | **0.436** ← collapse |
| 7 | 0.720 | 0.197 |
| 8 | 0.763 | 0.545 |

**A = 5 is the last stable archetype count** (EV 0.60). The five
archetypes (z-scale corners; sizes from dominant membership, N = 9,013):

| archetype | profile | size | reading |
|---|---|---|---|
| A0 | ↑sleep (+2.8) ↑mania (+1.7) ↓severity | 1,448 | **activation / sleep-disturbed** (BP-enriched) |
| A1 | ↑severity (+1.9) ↓immunometabolic (−1.9) ↓developmental | 1,584 | **severe, clean-biology** |
| A2 | ↑immunometabolic (+3.5) ↑severity (+2.4) ↑suicidality | 1,426 | **immunometabolic burden** — the biology corner |
| A3 | ↑developmental (+2.8) ↑suicidality (+1.4) ↓immunometabolic | 2,004 | **trauma / suicidality** |
| A4 | ↓sleep (−2.7) ↓severity (−2.0) ↓mania | 2,551 | **low-burden / well** pole |

**What the five archetypes carve:** the symptom space resolves into **two distinct
corners — activation/sleep (A0) and trauma/suicidality (A3)**, and the
biology corner (A2) is *immunometabolic*. This
is the finer, stable structure the 8-factor coordinates support. Biology (A2) is a *direction of maximal
phenotype* with no density gap, so it is an archetype corner the tessellation cannot cleanly split on — **which
is why the archetypes + continuous coordinates, not the tessellation, are load-bearing for biology.** Figure:
`archetype_profiles.png`.

## Result 4 — the regions are useful (internal battery, all PASS)

Run on the K = 2 contract default (gates hold across the confident-stable family):

| criterion | gate | evidence |
|---|---|---|
| assignment (assignable, not 50/50) | **PASS** (conditional) | confident-dominant 1.00; median norm-entropy 0.65 (a soft split) |
| not-just-severity (Q2) | **PASS** | η²(specifics) 0.077 > η²(G) 0.050 — driven by mania 0.224 + suicidality 0.225 |
| transdiagnostic (Q3) | **PASS** | ARI **0.011** vs cohort, **0.006** vs DSM-5 (≈ 0); Cramér's V 0.063 / 0.099 |
| stable / not-a-missingness-artefact (Q4) | **PASS** | seed-ARI **0.991**; coverage classifier acc 0.611 < baseline 0.683 (lift −0.072) |
| tighter than DSM-5 | **PASS** | XD-BIC 185.6k (free, K = 2) vs 188.2k (DSM-5, K = 7); mean η² 0.074 vs 0.026 |

So the soft regions are operationally assignable, driven by specific symptoms/biology beyond severity,
independent of diagnosis, stable, not a coverage artefact, and a tighter description than the DSM-5 subtypes —
**3× the explained variance of DSM-5 at lower BIC**.

## Honest caveats

* **Internal/baseline only.** Whether the regions/archetypes *predict* (2-year course, treatment) is the M3/M4
  question on this object. A2 (immunometabolic biology) and the K = 3/4 biology gradient are the natural carriers
  of durable/prognostic signal — the hypothesis M4/M5 test, and exactly *how* the operative K is chosen
  (incremental validity, not internal parsimony).
* **The K = 2 split is symptom-led and soft.** At K = 2 the tessellation separates on mania + suicidality (η²
  0.22 each); biology (immunometabolic η² 0.027) is a finer-K / archetype feature, not a K = 2 boundary. Median
  norm-entropy 0.65 → the two-region split is genuinely soft (the assignment gate passes *conditionally*).
* **No privileged K.** The tessellation is a convention; the load-bearing objects are the continuous
  coordinates and the **A = 5** archetype simplex. K = 2 is only the M3-contract default.
* **Substance is thin and orthogonal.** 2 SUD binaries, DR = 0; substance is pinned
  ⊥ the correlated block (its cross-factor correlations are non-identifiable), so it does not anchor an
  archetype corner — its coordinate is carried with a prior-dominated reliability tier.

## Hand-off

`results/face/strata_oop/consolidate/patient_strata.parquet` (**9,013 × 50**) carries, in the M3-compatible
contract (keyed `cohort` / `patient_id`; `arm` = validation-only):
* **archetypes (load-bearing):** `arch_w0…w4` (+ `_sd`), `arch_dominant(_name)`, `arch_entropy`,
  `arch_confidence_tier`, `arch_boundary`; the G-residualized arm-B mirror `archB_*`;
* **tessellation contract default (K = 2):** `tess_r0,r1`, `tess_MAP(_name)`, `tess_entropy`,
  `tess_confidence_tier`, `tess_boundary`;
* **nested K-family (conventions; operative K deferred to M4/M5):** `tessfam_k{2,3,4}_*` — the `tessfam_`
  prefix deliberately does **not** match the `tess_` selector, so the family never pollutes the contract.

The **continuous load-bearing coordinates** ship in `results/face/strata_oop/coordinates/`:
`coordinates_full.parquet` (per-axis mean/sd/HDI/n_obs/reliability), `coordinates_draws.npz` (joint draws),
`coordinates_cov.npz` (per-patient covariance S_i). Per-K decision menu: `consolidate/k_family_menu.csv`. Next:
the **M3 temporal-persistence analysis** on this object, then M4 (which selects the operative K by incremental
validity).

Reproduce: `PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_strata_model_oop.py --mode full`.
