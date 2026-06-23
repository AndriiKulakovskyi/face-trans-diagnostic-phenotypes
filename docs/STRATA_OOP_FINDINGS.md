# Soft operational regions on the Gaussian-copula map (M2) — findings

> **Canonical M2 findings record.** Built on the certified cohort-weighted full-N Gaussian-copula 9-dim map.
> **Detailed
> analysis of what the K-family and A mean: [STRATA_OOP_ATLAS.md](STRATA_OOP_ATLAS.md).** Engine:
> [`src/face/strata/strata_model_oop.py`](../src/face/strata/strata_model_oop.py);
> driver: [`notebooks/run_strata_model_oop.py`](../notebooks/run_strata_model_oop.py); figures:
> `docs/figures/strata_oop/`. **Internal/baseline scope** — temporal persistence (M3) and prognosis (M4)
> are deferred to the later reruns on this object. Pending PI sign-off. Updated 2026-06-22.

## What this is

The published M2 strata were built on the older **native** map. The Gaussian-copula vertical now gives the
best map we have (full-N, cohort-weighted, sub-1.05), so M2 is reworked on it — and reframed as the question
actually is: **the 9-dim space is a continuum, not a clustering problem.**

Because it is a continuum, **the load-bearing objects are granularity-free**: (i) the **continuous 9-dim
coordinates** (with per-patient posterior uncertainty + draws), and (ii) the **soft archetype simplex** —
each patient a convex blend of A=4 stable extreme phenotypes. A hard tessellation is, on a continuum, only a
**coarse labelling convention** with no privileged number of regions, so we do **not** feature a single K:
we export a **nested K-family** (K = 2, 3, 4) as conventions and **defer the *operative* granularity — which
K, if any, adds clinical value — to M4/M5 incremental predictive/treatment validity.** The engine *wraps* the
proven kernels (`mixture`/`structure`/`archetypes`/`validation`) — no math is reimplemented.

Three complementary views, all no-hard-edge:
* **Continuous coordinates (load-bearing)** — position on the 9 axes + uncertainty; M3/M4 consume these directly.
* **Archetypes (load-bearing continuum view)** — each patient a convex blend of A=4 extreme phenotypes.
* **Soft tessellation (coarse convention, nested K-family)** — a measurement-error (Extreme-Deconvolution)
  mixture; responsibilities *are* the soft boundaries. Exported at K = 2, 3, 4; no K is privileged.

## Coordinate prep (the copula simplification)

Because the cohort-weighted copula fit was run at full N, its posterior already contains the per-patient
explicit latents `f_e` for [G, suicidality, developmental_risk, substance] — so the old M2.0's expensive
full-N projection is **unnecessary**. We read `f_e` from the posterior and condition the marginalized
specifics `f_m | f_e` under the shared Φ (`scoring.coherent_joint_coords`), giving one coherent 9-dim
posterior draw per sample. All 9,013 patients are scored on all 9 axes (mean ≈ 0, latent z-scale, 100%
finite). Uncertainty propagates honestly: metabolic/inflammatory/developmental are tight (posterior SD
0.24–0.26), mania/substance/suicidality wide (0.48–0.68); substance is never "well" (only 2 SUD binaries) and
552 DR patients are prior-dominated — **no imputation**.

## Result 1 — it is a continuum

The structure-discovery gate (Hopkins / dip / silhouette / gap / GMM-BIC / HDBSCAN / Mapper, uncertainty-aware
over draws) returns **continuum** on the copula map: 3/6 clustered signals, silhouette peak **0.14**
(< 0.15 = no separation), PC1 dip p = **1.0** (unimodal), HDBSCAN **0 clusters**, Mapper a **single connected
component** (one chain). A single-Gaussian falsification null is decisive: the best partition separates
patients **no better than slicing a structureless Gaussian blob** (silhouette real 0.140 vs null 0.141±0.002,
z = -0.36; over draws the GMM-optimal K collapses to 1 in all 20). So there are **no well-separated,
reproducible discrete clusters** — K is a granularity convention, not a discovered kind-count.
Figure: `structure_panel.png`. Full analysis + limits: [STRATA_OOP_ATLAS.md §0b](STRATA_OOP_ATLAS.md).

## Result 2 — the soft tessellation is a nested K-family (no privileged K)

The XD-BIC is essentially **flat** across K = 2–8 (197.9k–199.5k, < 1%; the minimum is at **K = 3**), the
continuum signature of no natural K. Rather than break that flat tie with an internal parsimony rule, we
export the family and let downstream validity decide. The decision menu
(`results/face/strata_oop/consolidate/k_family_menu.csv`):

| K | XD-BIC | confident-dominant | seed-ARI | η² specifics | η² G | η² suicidality | η² metabolic | η² inflammatory |
|---|---|---|---|---|---|---|---|---|
| 2 | 197,963 | 1.00 | 0.998 | 0.122 | 0.008 | 0.543 | 0.003 | 0.004 |
| **3** (BIC-best) | **197,918** | 0.92 | 0.968 | 0.141 | 0.165 | 0.436 | 0.064 | 0.011 |
| 4 | 198,108 | 0.87 | 0.996 | 0.154 | 0.332 | 0.557 | 0.102 | 0.031 |

**The headline nuance:** all Ks split *first* on **psychiatric symptom burden** (suicidality-anchored), but
**finer K progressively captures severity (G) and biology that K = 2 discards** — metabolic η² 0.003 → 0.064 →
0.102, inflammatory 0.004 → 0.011 → 0.031, G 0.008 → 0.165 → 0.332. Biology has no density *gap* (so it never
forms a clean region boundary at any K — see Result 3), but a finer tiling does pick up more of its continuous
gradient. K = 2 is the coarsest, sharpest-assigning convention; K = 3 is BIC-best and still confident + stable;
K = 4 is the richest on severity/biology. **No K is "the answer"** — the operative choice is M4/M5's.

For the M3 hand-off contract a concrete default is still required, so the `tess_*` columns carry **K = 2** (the
smallest confidently-assignable, stable convention) — a **contract default, not a privileged scientific
choice**. The full family ships alongside it as `tessfam_k{2,3,4}_*`.

## Result 3 — archetypes carry the biology⊥symptom structure (A = 4, stable)

Archetype granularity is chosen by **cross-seed stability** (largest A with min Tucker congruence ≥ 0.8), not
parsimony: only **A = 2 (0.999)** and **A = 4 (0.979)** are stable; A = 3/5/6/7/8 collapse (0.05–0.50). The
operational rule selects **A = 4** (EV 0.52). This is copula-specific: the native map's A = 8 does **not**
reproduce here (the copula's honest, wider explicit-axis uncertainty will not support 8 stable corners). The
4 stable archetypes separate biology from symptoms from severity:

| archetype | profile |
|---|---|
| A0 | ↑inflammatory ↑metabolic ↑substance — the **biological corner** |
| A1 | ↓overall_severity ↓developmental ↓sleep — the **low-burden pole** |
| A2 | ↑overall_severity ↓inflammatory ↓substance — **high severity, low biology** |
| A3 | ↑sleep ↑developmental ↑suicidality ↑mania ↓metabolic — the **symptom corner** |

Biology (A0) is a *direction of maximal phenotype* even though it has no density gap, so it shows up as an
archetype corner the tessellation cannot cleanly split on. **This is why the archetypes (and continuous
coords), not the tessellation, are the load-bearing object for biology.** Figure: `archetype_profiles.png`.

## Result 4 — the regions are useful (internal battery, all PASS)

Run on the K = 2 contract default (the gates hold across the confident-stable family):

| criterion | gate | evidence |
|---|---|---|
| assignment (assignable, not 50/50) | **PASS** | confident-dominant 1.00, median entropy 0.51 |
| not-just-severity (Q2) | **PASS** | η²(specifics) 0.122 ≫ η²(G) 0.008 |
| transdiagnostic (Q3) | **PASS** | ARI 0.00 vs cohort, 0.01 vs DSM-5 |
| stable / not-a-missingness-artefact (Q4) | **PASS** | seed-ARI 1.0; coverage perm-p 0.23, lift −0.052 |
| tighter than DSM-5 | **PASS** | XD-BIC 197.9k vs 201.3k; mean η² 0.108 vs 0.025 |

So the soft regions are operationally assignable, driven by specific biology/symptoms beyond severity,
independent of diagnosis, stable, not a coverage artefact, and a tighter description than the DSM-5 subtypes.
Embedding (PCA, **visualization-only**): `embedding.png`.

## Honest caveats

* **Internal/baseline only.** Whether the regions/archetypes *predict* (2-year course, treatment response) is
  the M3/M4 rerun on this object — not claimed here. The biology corner (A0) and the K = 3/4 biology gradient
  are the natural candidates to carry durable/prognostic signal (A0 did on the native map), but that is the
  hypothesis M4/M5 test — and is exactly *how* the operative K should be chosen (by incremental validity, not
  internal parsimony).
* **No privileged K.** The tessellation is a coarse convention; the load-bearing objects are the continuous
  coordinates and the A = 4 archetype simplex. K = 2 is only the M3-contract default.
* **Archetype granularity is copula-sensitive.** Only A = 2 and A = 4 are stable; native A = 8 does not
  reproduce (the copula's wider explicit-axis uncertainty destabilizes high-A corners).
* **Substance is thin** (2 SUD binaries, DR = 0): its coordinate is weakly identified (wide SD), carried with
  a prior-dominated reliability tier.

## Hand-off

`results/face/strata_oop/consolidate/patient_strata.parquet` (**9,013 × 41**) carries, in the M3-compatible
contract (keyed `cohort` / `patient_id`; `arm` = validation-only):
* **archetypes (load-bearing):** `arch_w0…w3` (+ `_sd`), `arch_dominant(_name)`, `arch_entropy`,
  `arch_confidence_tier`, `arch_boundary`;
* **tessellation contract default (K = 2):** `tess_r0,r1`, `tess_MAP(_name)`, `tess_entropy`,
  `tess_confidence_tier`, `tess_boundary`;
* **nested K-family (conventions; operative K deferred to M4/M5):** `tessfam_k{2,3,4}_r*`,
  `tessfam_k{K}_MAP`, `tessfam_k{K}_entropy`, `tessfam_k{K}_tier` — the `tessfam_` prefix deliberately does
  **not** match the `tess_` selector, so the family never pollutes the operational contract.

The **continuous load-bearing coordinates** ship in `results/face/strata_oop/coordinates/`:
`coordinates_full.parquet` (per-axis mean/sd/HDI/n_obs/reliability), `coordinates_draws.npz` (joint posterior
draws), `coordinates_cov.npz` (per-patient covariance S_i). The per-K decision menu is
`consolidate/k_family_menu.csv`. The next step is the **M3 temporal-persistence rerun**, then M4 — which
**selects the operative K from the family by incremental validity.**

Reproduce: `PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_strata_model_oop.py --mode full`
→ `python notebooks/strata_oop_make_figures.py`.
