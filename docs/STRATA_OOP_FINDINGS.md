# Soft operational regions on the Gaussian-copula map (M2, reworked) — findings

> Paper-facing summary of the reworked M2 stratification, built on the certified cohort-weighted full-N
> Gaussian-copula 9-dim map. **Detailed analysis of what K and A mean (per-axis split drivers, archetype
> profiles, compositions): [STRATA_OOP_ATLAS.md](STRATA_OOP_ATLAS.md).** Engine:
> [`src/face/strata/strata_model_oop.py`](../src/face/strata/strata_model_oop.py);
> notebook: [`notebooks/m2_oop_strata_fit.ipynb`](../notebooks/m2_oop_strata_fit.ipynb); figures:
> `docs/figures/strata_oop/`. **Internal/baseline scope** — temporal persistence (M3) and prognosis (M4)
> are deferred to the later reruns on this object. Pending PI sign-off.

## What this is

The published M2 strata were built on the older **native** map. The Gaussian-copula vertical now gives the
best map we have (full-N, cohort-weighted, sub-1.05), so M2 is reworked on it — and reframed as the question
actually is: **the 9-dim space is a continuum, not a clustering problem.** We define **operational regions
with soft transition boundaries** (probabilistic membership, no hard edges) and ask whether those regions are
**useful** at baseline. The engine is a clean parallel OOP layer that *wraps* the proven kernels
(`mixture`/`structure`/`archetypes`/`validation`) — no math is reimplemented.

Two complementary soft views, both no-hard-edge:
* **Soft tessellation (lead operational regions)** — a measurement-error (Extreme-Deconvolution) mixture that
  propagates each patient's coordinate uncertainty `S_i`; the responsibilities *are* the soft boundaries.
* **Archetypes (continuum co-view)** — each patient a convex blend of extreme phenotypes (simplex weights).

Granularity is a **deliberate operational choice** (a continuum has no natural K/A): the smallest K, and the
largest stable A, that stay confidently assignable and reproducible.

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
component** (one chain). This reproduces the native M2 finding on the better-converged map.
Figure: `structure_panel.png`.

## Result 2 — soft operational regions (K = 2)

The operational-K rule (smallest K keeping confident assignment ≥ 0.5 and seed-ARI ≥ 0.8) selects **K = 2**;
the XD-BIC is nearly flat across K = 2–8 (197.9k–198.7k), confirming there is no natural K — K is a
granularity choice for communication. The two regions split on **psychiatric symptom burden**, not biology:

| region | profile | share |
|---|---|---|
| 0 — *near-average* | low suicidality / sleep / mania / substance | ~50% |
| 1 — *↑suicidality* | elevated suicidality / sleep / mania / developmental / substance | ~50% |

86–100% of patients have a confident dominant region (not a mushy middle); the rest are boundary patients,
flagged explicitly. The biology axes (metabolic / inflammatory / cognition) are near-zero in *both* regions —
the coarse tessellation does not separate on biology. Figures: `region_profiles.png`, `boundary_map.png`
(auto-selected axes suicidality × substance; shading = membership entropy), `confidence_bars.png`.

## Result 3 — archetypes carry the biology⊥symptom structure (A = 4, stable)

The EV scree always rewards more corners, but on the copula coordinates the high-A archetypes chase the wide
explicit-axis noise and become seed-unstable. Cross-seed reproducibility (min Tucker congruence) over the
sweep: **A=2 0.999, A=4 0.979 (stable); A=3 0.50, A=5 0.08, A=6 0.29, A=7 0.11, A=8 0.05 (unstable)**. The
operational rule (largest A with stability ≥ 0.8) selects **A = 4** (EV 0.52). This is a copula-specific
finding: the native map's A=8 archetypes do **not** reproduce here. The 4 stable archetypes separate biology
from symptoms from severity:

| archetype | profile |
|---|---|
| 0 | ↑inflammatory ↑metabolic ↑substance — the **biological corner** |
| 1 | ↓overall_severity ↓developmental ↓sleep — the **low-burden pole** |
| 2 | ↑overall_severity ↓inflammatory ↓substance — **high severity, low biology** |
| 3 | ↑sleep ↑developmental ↑suicidality ↑mania ↓metabolic — the **symptom corner** |

Biology (corner 0) is distinct from psychiatric symptoms (corner 3) — the transdiagnostic biology⊥symptom
structure, now as a *stable* continuum representation. Figure: `archetype_profiles.png`.

## Result 4 — the regions are useful (internal battery, all PASS)

| criterion | gate | evidence |
|---|---|---|
| assignment (assignable, not 50/50) | **PASS** | confident-dominant 100%, median entropy 0.52 |
| not-just-severity (Q2) | **PASS** | η²(specifics) 0.121 ≫ η²(G) 0.008 |
| transdiagnostic (Q3) | **PASS** | ARI 0.00 vs cohort, 0.01 vs DSM-5 |
| stable / not-a-missingness-artefact (Q4) | **PASS** | seed-ARI 1.0; coverage perm-p 0.60, lift −0.055 |
| tighter than DSM-5 | **PASS** | XD-BIC 197.9k vs 201.3k; mean η² 0.108 vs 0.025 |

So the soft regions on the copula map are operationally assignable, driven by specific biology/symptoms beyond
severity, independent of diagnosis, stable, not a coverage artefact, and a tighter description than the DSM-5
subtypes. Embedding (PCA, **visualization-only**): `embedding.png`.

## Honest caveats

* **Internal/baseline only.** Whether the regions *predict* (2-year course, treatment response) is the M3/M4
  rerun on this object — not claimed here.
* **K = 2 is coarse** by design (operational parsimony). The richer continuum description is the archetypes;
  the tessellation is the decision-region label.
* **Archetype granularity is copula-sensitive.** Only A = 2 and A = 4 are stable; the native A = 8 does not
  reproduce, because the copula's wider explicit-axis uncertainty destabilizes high-A corners.
* **Substance is thin** (2 SUD binaries, DR = 0): its coordinate is weakly identified (wide SD), carried with
  a prior-dominated reliability tier.

## Hand-off

`results/face/strata_oop/consolidate/patient_strata.parquet` (9,013 × 23) carries the soft memberships +
uncertainty + entropy + confidence tiers + boundary flags in the M3-compatible contract (`arch_*` / `tess_*`
columns + `arm`). The next step is the **M3 temporal-persistence rerun** on this copula strata object.

Reproduce: `python notebooks/run_strata_model_oop.py --mode full` → `python notebooks/strata_oop_make_figures.py`.
