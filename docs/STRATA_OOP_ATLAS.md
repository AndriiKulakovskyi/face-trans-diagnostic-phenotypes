# M2 soft-region atlas — what K and A mean, in detail (copula map)

> Detailed results analysis for the reworked M2 (the continuum of soft operational regions on the
> Gaussian-copula map). Paper-facing summary: [STRATA_OOP_FINDINGS.md](STRATA_OOP_FINDINGS.md). Engine:
> [`src/face/strata/strata_model_oop.py`](../src/face/strata/strata_model_oop.py). Figures:
> `docs/figures/strata_oop/`. All coordinates are on the latent z-scale (0 = population mean, units = SD).

## 0. One continuum, three views — the load-bearing objects and the (un-privileged) K

The structure gate says the 9-dim space is a **continuum** (no discrete clusters). On a continuum there is no
"correct number of groups", so the **load-bearing objects are granularity-free**: the **continuous
coordinates** (+ uncertainty) and the **archetype simplex** (A extreme phenotypes, each patient a blend). A
hard tessellation is only a **coarse labelling convention** — useful for communication, but with **no
privileged K**. We therefore export a **nested K-family** and defer the *operative* K to M4/M5:

| | **K — soft tessellation** | **A — archetypes** |
|---|---|---|
| status | coarse convention (not load-bearing) | **load-bearing** (with the continuous coords) |
| object | `K` soft regions of a measurement-error mixture | `A` extreme phenotypes (corners of the convex hull) |
| membership | responsibilities `r_i` (sum to 1) | simplex weights `w_i` (sum to 1) — a patient *is* a blend |
| what it finds | the dominant **density** split — where the cloud is most separable into regions | the **extremes** — the orthogonal directions of maximal phenotype |
| answer it gives | "which **decision region** is this patient in?" | "**where on the continuum** is this patient — what blend of extremes?" |
| granularity rule | **nested family** (no privileged K); the operative K → M4/M5 incremental validity | largest `A` that stays cross-seed reproducible |
| result here | **K = 2 / 3 / 4 exported** (K = 2 is the M3-contract default only) | **A = 4** |

The headline finding is that the tessellation and the archetypes split on **different axes** — and that
dissociation is exactly the transdiagnostic structure we are after (§3).

---

## 0b. How do we know there are *no clusters*? (the structure gate, and its limits)

"No clusters" is a negative that cannot be *proved*; what we can do is run a convergent battery of
complementary tests and a falsification null, and report honestly. The verdict is **continuum** with a
**clustered-score of 3/6** — i.e. it is a *weak-separation* continuum, not a slam-dunk absence of structure.
The three signals that fire are tendency/shape signals, not separation signals — and a falsification test
shows they are spurious.

**The battery (arm A, all 9 axes, 9,013 patients):**

| test | what it measures | result | reads as |
|---|---|---|---|
| Hopkins | cluster *tendency* (vs uniform) | 0.79 | fired (but see null below) |
| GMM-BIC | does >1 Gaussian fit better than 1? | interior optimum, gain 10,326 | fired (but = non-Gaussian shape) |
| gap statistic | optimal K vs a uniform reference | k=3 | fired |
| **silhouette** | **separation** of the best K | **peak 0.14** (< 0.15) | **continuum** |
| dip (PC1) | modality | p = 1.0, unimodal | continuum |
| HDBSCAN | density clusters (non-convex) | 0 clusters, 100% noise | continuum |
| Mapper | topology | a single connected component | continuum |

**The falsification null — the decisive test.** We compare the real cloud to a **single multivariate Gaussian
with the same mean and covariance** — a cloud that has *no clusters by construction* — and ask whether the
real data's metrics exceed it (`StructureGate.null_comparison`):

| metric | real | single-Gaussian null | z |
|---|---|---|---|
| **best silhouette (separation)** | **0.140** | **0.140 ± 0.003** | **0.1** |
| Hopkins (tendency) | 0.794 | 0.776 ± 0.004 | 4.5 |
| GMM-BIC gain (shape) | 10,326 | 0 ± 0 | — |

The decisive line is the silhouette: the best clustering of the real data separates patients **no better than
clustering a structureless Gaussian blob** (z = 0.1). The Hopkins "tendency" is essentially what the null
already gives (0.79 vs 0.78), so that signal is spurious. The large GMM-BIC gain (real 10,326 vs null 0)
confirms the cloud is **non-Gaussian** — but the silhouette proves those GMM components are **not separated**,
so the non-Gaussianity is *shape* (skew + the archetype corners), not *clusters*.

**Accounting for measurement uncertainty.** The coordinates are posterior means with known per-patient
uncertainty; treating uncertain blobs as points could manufacture or hide structure. Re-running over the
posterior **draws** (`uncertainty_stability`), the GMM-BIC-optimal K is **1 in all 20 draws** (Hopkins
0.77 ± 0.01) — the apparent multi-component structure of the means collapses to a single component once
uncertainty propagates. And the measurement-error mixture itself (Extreme Deconvolution) **deconvolves**
`S_i`: if tight clusters were hiding under measurement noise it would recover them; instead the region
covariances are large and heavily overlapping.

**So the precise, honest claim** is *not* "every patient is the same" and *not* a proven "zero clusters
anywhere." It is: **in the 9-dim baseline copula coordinate space, there are no well-separated, reproducible,
density- or topology-defined discrete clusters** — separation is statistically identical to a structureless
continuum, the cloud is one connected unimodal component, density clustering finds none, and the optimal
component count under uncertainty is 1. That is quantitatively inconsistent with discrete biotypes and (per §4
of the findings) a tighter description than the DSM-5 categories. **What we cannot rule out:** clusters in the
raw-indicator space that the 9-dim factor summary blurs, or structure that emerges only with outcomes/
follow-up (M3/M4). The continuum claim is about *this* baseline measurement map.

---

## 1. K — the soft tessellation splits on psychiatric symptom burden

### 1.1 The K-family (no privileged K)

`K` is a granularity *convention*, not a discovered kind-count (a continuum has no natural K). The XD-BIC is
essentially **flat** across K (a continuum signature — no interior optimum; the minimum is at **K = 3**, by
< 0.03%), so an internal parsimony tiebreak would be false precision. Instead we **export the family and let
M4/M5 pick the operative K by external (predictive/treatment) validity.** The decision menu
(`results/face/strata_oop/consolidate/k_family_menu.csv`) — assignment + stability + what each K splits on:

| K | XD-BIC | confident-dominant | seed-ARI | η² specifics | η² G | η² suicidality | η² metabolic | η² inflammatory |
|---|---|---|---|---|---|---|---|---|
| **2** (contract default) | 197,963 | **1.00** | 0.998 | 0.122 | 0.008 | 0.543 | 0.003 | 0.004 |
| **3** (BIC-best) | **197,918** | 0.92 | 0.968 | 0.141 | 0.165 | 0.436 | 0.064 | 0.011 |
| **4** | 198,108 | 0.87 | 0.996 | 0.154 | **0.332** | 0.557 | **0.102** | 0.031 |

BIC moves < 1% across the whole range (K = 2–8: 197.9k–199.5k) — there is no "right" K. **Two honest reads:**
(i) every K splits *first* on suicidality-anchored symptom burden (the dominant density direction); but
(ii) **finer K progressively captures the severity (G) and biology gradient that K = 2 discards** — metabolic
η² 0.003 → 0.064 → 0.102, inflammatory 0.004 → 0.011 → 0.031, G 0.008 → 0.165 → 0.332. Biology has no density
*gap* (so it never forms a clean region boundary at any K — §2/§3), but a finer tiling picks up more of its
continuous gradient. K = 2 is the sharpest-assigning convention and the **M3-contract default** (a concrete
`tess_*` is needed downstream); K = 3 is BIC-best and still confident + stable; K = 4 is richest on
severity/biology. The full family ships as `tessfam_k{2,3,4}_*` so the operative choice is M4/M5's, not ours.

### 1.2 The two regions are a symptom-burden gradient

The deconvolved (noise-free) region centroids are near **mirror images** on the *psychiatric symptom* axes
and ~**zero** on biology and on overall severity:

| axis | R0 *near-average* (57%) | R1 *↑suicidality* (43%) |
|---|---|---|
| **suicidality** | **−0.54** | **+0.62** |
| mania_activation | −0.24 | +0.28 |
| substance | −0.26 | +0.26 |
| sleep | −0.22 | +0.25 |
| developmental_risk | −0.18 | +0.20 |
| overall_severity (G) | −0.01 | +0.15 |
| cognition | +0.07 | −0.06 |
| inflammatory | −0.06 | +0.06 |
| metabolic | −0.04 | +0.05 |

### 1.3 What actually drives the split (per-axis η²)

η² = the fraction of each axis's variance the K=2 partition explains. The split is **defined by
suicidality** and the symptom cluster; it explains essentially **none** of the biology or severity variance:

| axis | η² of the K=2 split |
|---|---|
| **suicidality** | **0.543** |
| mania_activation | 0.141 |
| substance | 0.134 |
| sleep | 0.081 |
| developmental_risk | 0.060 |
| cognition | 0.012 |
| overall_severity | 0.008 |
| inflammatory | 0.004 |
| metabolic | 0.003 |

**Reading:** the coarsest operationally-useful partition of the copula continuum is a **low- vs
elevated-psychiatric-symptom-burden** split, anchored by suicidality (54% of its own variance), co-varying
with mania / substance / sleep / developmental. It is **not a severity ladder** (G η² 0.008 — this is why the
not-just-severity gate passes) and it is **blind to biology** (metabolic/inflammatory η² ≈ 0.003).

### 1.4 It is transdiagnostic and soft

Both regions mix all three cohorts and all DSM-5 subtypes — the partition is independent of diagnosis
(ARI ≈ 0):

| region | BP | DR | SZ | top DSM-5 subtypes |
|---|---|---|---|---|
| R0 near-average | 64% | 7% | 29% | BP1 30%, BP2 28%, SZ 23% |
| R1 ↑suicidality | 76% | 5% | 18% | BP2 39%, BP1 28%, SZ 13% |

Membership is **soft**: median entropy 0.51; patients near the suicidality midline are boundary cases with
`r ≈ (0.5, 0.5)` — the soft transition boundary is explicit (`tess_entropy`, `tess_confidence_tier` in the
hand-off). (At K=2 every patient's max responsibility is ≥ 0.5 by construction, so the tiers are core/soft;
the boundary tier appears at K ≥ 3.) Figures: `region_profiles.png`, `boundary_map.png`, `confidence_bars.png`.

---

## 2. A — the archetypes reveal biology ⊥ symptoms ⊥ severity

### 2.1 Why A = 4 (and why not 8)

`A` is the number of extreme phenotypes. Explained variance always rewards more corners, but on the copula
coordinates the high-A archetypes chase the **wide explicit-axis uncertainty** (suicidality/substance/
developmental have large posterior SD) and become **seed-unstable** — so we pick the largest A whose
cross-seed reproducibility (min Tucker congruence) stays ≥ 0.8:

| A | explained var | cross-seed stability |
|---|---|---|
| 2 | 0.24 | **0.999** |
| 3 | 0.41 | 0.497 |
| **4** | **0.52** | **0.979** |
| 5 | 0.59 | 0.080 |
| 6 | 0.66 | 0.288 |
| 7 | 0.72 | 0.112 |
| 8 | 0.76 | 0.050 |

Stability is **non-monotonic** with islands at A=2 and A=4. **A = 4** is the operational sweet spot (stable +
half the variance). This is a **copula-specific finding**: the native map's A=8 archetypes (Tucker 0.999
there) do **not** reproduce on the copula coordinates — the copula's honest, wider explicit-axis uncertainty
will not support 8 stable corners.

### 2.2 The four extreme phenotypes

| archetype | overall_severity | cognition | metabolic | inflammatory | sleep | mania | suicidality | developmental | substance | dominant-in |
|---|---|---|---|---|---|---|---|---|---|---|
| **A0 — biological** | +1.61 | +0.26 | **+1.99** | **+2.52** | +0.21 | +0.42 | +1.04 | +0.57 | **+2.55** | 25% |
| **A1 — low-burden pole** | **−1.94** | +0.16 | −0.10 | +0.66 | −1.65 | −1.08 | −1.15 | −1.66 | −0.12 | 36% |
| **A2 — severe, non-biological** | **+1.80** | +0.49 | −0.24 | **−2.93** | −0.49 | −0.37 | −0.69 | −0.70 | **−2.99** | 17% |
| **A3 — psychiatric-symptom** | −0.61 | −0.87 | **−1.75** | −0.77 | **+2.24** | +1.30 | +1.03 | **+2.18** | +0.10 | 22% |

* **A0 — biological corner:** high inflammatory + metabolic + substance (with some severity/suicidality). The
  cardiometabolic-inflammatory extreme.
* **A1 — low-burden pole:** everything low — the "healthy reference" extreme (the modal patient, 36%).
* **A2 — severe but non-biological:** high overall severity with the **lowest** inflammatory/substance — shows
  that severity can be high *with biology dissociated*.
* **A3 — psychiatric-symptom corner:** high sleep + developmental + mania + suicidality, **low** metabolic —
  the symptom extreme.

### 2.3 The key result: A0 and A3 are different corners

The biological extreme (**A0**) and the psychiatric-symptom extreme (**A3**) are **separate archetypes**, and
**A2** shows severity rising while biology falls. So the continuum has (at least) three quasi-independent axes
of extremity — **biology ⊥ symptoms ⊥ severity** — the transdiagnostic dissociation, now as a *stable*
continuum description. Figure: `archetype_profiles.png`.

### 2.4 Everyone is a blend (continuum-honest), and it is transdiagnostic

* **61%** of patients have max archetype weight **< 0.5** — genuinely blended, no "type". Mean archetype
  entropy 0.78; even "dominant" patients average ~0.48 weight on their lead archetype.
* All four archetypes mix cohorts (transdiagnostic): A0 64/8/27, A1 75/2/23, A2 49/14/37, A3 82/4/14
  (BP/DR/SZ %). The severe-non-biological corner (A2) is the most SZ-enriched (37%); the symptom corner (A3)
  the most BP (82%) — but none is a single diagnosis.

---

## 3. Synthesis — why K splits on symptoms but A reveals biology

This is the load-bearing insight, and it is not a contradiction — it is the continuum signature:

* The **mixture (K)** partitions by **density**: it cuts where the point cloud is most separable into regions.
  The highest-variance, most-separable direction is psychiatric symptom burden (suicidality-anchored), so K=2
  cuts there. Biology has **no** density gap — patients vary *continuously* along metabolic/inflammatory — so
  biology does not produce a region split (its K=2 η² ≈ 0.003).
* The **archetypes (A)** find **extremes**, not splits. Biology is a real *direction of maximal phenotype*
  (the A0 corner: inflammatory +2.5, substance +2.5, metabolic +2.0) even though it has no density gap, so it
  shows up as an archetype corner that the tessellation cannot see.

In one sentence: **there are no biology *clusters* (so biology drives no K-split), but there are biology
*extremes* (so biology defines an archetype corner).** The tessellation tells you the dominant operational
decision-axis (symptom burden); the archetypes give you the full set of orthogonal extreme phenotypes (incl.
biology) that span the continuum. They are complementary by construction, and together they say the copula map
is a genuine continuum whose principal operational gradient is psychiatric symptom burden, with biology a real
but continuous, density-gapless, orthogonal dimension.

## 4. Clinical / scientific reading (with the honest limits)

* **Operationally:** the patient *is* their **continuous position** + **blend of the four archetypes** —
  reading their biological (A0) load separately from their symptom (A3) and severity (A2) load. If a coarse
  hard label is wanted, the tessellation family offers it at a granularity of choice: K = 2 (a clean
  symptom-burden split), K = 3 (BIC-best, adds a severity/biology gradient), or K = 4 (richest on
  severity/biology). **No single K is privileged** — which one (if any) earns its keep is decided by outcomes.
* **Transdiagnostic:** no view re-encodes diagnosis (ARI ≈ 0); the structure cuts across BP/SZ/DR and the
  DSM-5 subtypes.
* **Internal/baseline only — and this is *how* the operative K should be chosen.** Whether the regions /
  archetypes *predict* 2-year course or treatment response is the **M3/M4 rerun** on this object, not claimed
  here. The biology corner (A0) and the K = 3/4 biology gradient are the natural candidates to carry
  durable/prognostic signal (A0 did on the native map); M4 selects the operative K from the family by
  **incremental validity over DSM-5 + severity**, which is the non-circular way to honour "choose K for
  actionability" (an internal parsimony tiebreak on a flat basin would not be).
* **Caveats:** the tessellation is a coarse convention (the load-bearing objects are the continuous coords +
  the A = 4 archetypes); biology shows up as an archetype corner / a continuous gradient, not a clean region
  boundary; archetype granularity is copula-sensitive (only A = 2, 4 stable); substance is thin (2 SUD
  binaries, DR = 0) and carried with wide uncertainty.
