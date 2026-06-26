# M2 soft-region atlas — what K and A mean, in detail (copula map)

> **Reframe note.** M2 is a *coordinate system + reading guide*, not a typology. The **8-dim** copula space
> (immunometabolic merge + 3 cross-loadings + substance ⊥) is a **continuum**; the **continuous coordinates**
> are the load-bearing object, and the archetype corners / K-tessellation are **interpretation lenses**, not
> discovered decision-regions or natural kinds (M4 confirms: *operative K = none*). Canonical companions:
> docs/STRATA_READING_GUIDE.md + report §sec:strata. Read the framing below as the *derivation* of those lenses,
> not a claim that patients fall into discrete strata.

> **8-factor update (2026-06-26).** Rebuilt on `copula/weighted_8d/hs_s5_merged_xc`. Headline change: **A = 5
> stable archetypes (was 4)** — §2 is fully reconciled below. **The canonical 8-factor numbers live in
> [STRATA_OOP_FINDINGS.md](STRATA_OOP_FINDINGS.md)** (continuum null silhouette 0.140 vs 0.137±0.002 z = 1.13;
> K-family BIC 185.0–185.6k; transdiagnostic ARI 0.006 vs DSM-5; tighter-than-DSM-5 η² 0.074 vs 0.026). The
> §0b/§1 *detail tables* below still show 9-factor-era illustrative values (metabolic/inflammatory split,
> A = 4); the **qualitative story is unchanged** (continuum; K splits first on symptom burden — suicidality +
> mania; biology is an archetype/continuous feature, not a region boundary) and the per-axis-η² reconciliation
> is tracked. Trust FINDINGS.md for exact figures.

> Detailed results analysis for the reworked M2 (the continuum of soft operational regions on the
> Gaussian-copula map). Paper-facing summary: [STRATA_OOP_FINDINGS.md](STRATA_OOP_FINDINGS.md). Engine:
> [`src/face/strata/strata_model_oop.py`](../src/face/strata/strata_model_oop.py). Figures:
> `docs/figures/strata_oop/`. All coordinates are on the latent z-scale (0 = population mean, units = SD).

## 0. One continuum, three views — the load-bearing objects and the (un-privileged) K

The structure gate says the 8-dim space is a **continuum** (no discrete clusters). On a continuum there is no
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
| result here | **K = 2 / 3 / 4 exported** (K = 2 is the M3-contract default only) | **A = 5** |

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
| GMM-BIC | does >1 Gaussian fit better than 1? | interior optimum, gain 9,495 | fired (but = non-Gaussian shape) |
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
| **best silhouette (separation)** | **0.140** | **0.141 ± 0.002** | **-0.36** |
| Hopkins (tendency) | 0.794 | 0.776 ± 0.004 | 4.5 |
| GMM-BIC gain (shape) | 9,495 | 0 ± 0 | — |

The decisive line is the silhouette: the best clustering of the real data separates patients **no better than
clustering a structureless Gaussian blob** (z = -0.36). The Hopkins "tendency" is essentially what the null
already gives (0.79 vs 0.78), so that signal is spurious. The large GMM-BIC gain (real 9,495 vs null 0)
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
anywhere." It is: **in the 8-dim baseline copula coordinate space, there are no well-separated, reproducible,
density- or topology-defined discrete clusters** — separation is statistically identical to a structureless
continuum, the cloud is one connected unimodal component, density clustering finds none, and the optimal
component count under uncertainty is 1. That is quantitatively inconsistent with discrete biotypes and (per §4
of the findings) a tighter description than the DSM-5 categories. **What we cannot rule out:** clusters in the
raw-indicator space that the 8-dim factor summary blurs, or structure that emerges only with outcomes/
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

### 2.1 Why A = 5 (changed from A = 4 on the 9-factor map)

`A` is the number of extreme phenotypes. Explained variance always rewards more corners, so we pick the largest
A whose cross-seed reproducibility (min Tucker congruence) stays ≥ 0.8. The 8-factor map shifts the answer up
by one — there is a clean **stability cliff at A = 6**:

| A | explained var | cross-seed stability |
|---|---|---|
| 2 | 0.24 | **0.999** |
| 3 | 0.39 | 0.813 |
| 4 | 0.50 | **0.997** |
| **5** | **0.60** | **0.979** |
| 6 | 0.66 | **0.436** ← collapse |
| 7 | 0.72 | 0.197 |
| 8 | 0.76 | 0.545 |

**A = 5 is the last stable archetype count** (EV 0.60); A = 6 collapses to 0.44 and beyond is noise. This is one
more than the 9-factor map's A = 4 — the merged immunometabolic coordinates support a finer, still-stable
simplex. (Still copula-specific: the native map's A = 8 does not reproduce; the copula's honest, wider
explicit-axis uncertainty caps the stable corner count well below 8.)

### 2.2 The five extreme phenotypes

| archetype | severity | cognition | immunometabolic | sleep | mania | suicidality | developmental | substance | reading |
|---|---|---|---|---|---|---|---|---|---|
| **A0 — activation/sleep** | −1.53 | −0.28 | +0.49 | **+2.79** | **+1.67** | −0.04 | −0.55 | −0.03 | manic + sleep-disturbed |
| **A1 — severe, clean-biology** | **+1.88** | +0.59 | **−1.93** | −0.45 | −0.70 | −1.25 | **−2.59** | +0.11 | high severity, low biology |
| **A2 — immunometabolic** | +2.36 | −0.01 | **+3.46** | −0.04 | −0.06 | +1.25 | +1.21 | −0.12 | the **biology corner** |
| **A3 — trauma/suicidality** | +0.09 | −0.57 | −2.07 | +0.84 | +0.72 | **+1.35** | **+2.83** | +0.11 | childhood adversity + suicidality |
| **A4 — low-burden pole** | **−2.03** | +0.38 | +0.28 | **−2.67** | −1.37 | −1.16 | −1.08 | −0.08 | the "well" reference extreme |

* **A2 — immunometabolic (biology) corner:** the cardiometabolic-inflammatory extreme (immunometabolic +3.5),
  with high severity + suicidality + developmental. The merged-biology direction of maximal phenotype.
* **A1 — severe, clean-biology:** high severity with the **lowest** immunometabolic/developmental — severity can
  be high *with biology dissociated* (the old A2 role).
* **A0 vs A3 — the symptom space splits in two:** activation/sleep (A0: mania + sleep) is now a **separate
  corner** from trauma/suicidality (A3: developmental + suicidality). On the 9-factor map these were one
  combined symptom corner — **this is what the 5th archetype buys.**
* **A4 — low-burden pole:** everything low — the "healthy reference" extreme (largest, 2,551 patients).

### 2.3 The key result: biology, two symptom corners, and severity dissociate

The biology extreme (**A2**) is separate from **both** symptom corners (**A0** activation, **A3** trauma), and
**A1** shows severity rising while biology falls. So the continuum carries quasi-independent axes of extremity —
**biology ⊥ activation ⊥ trauma/suicidality ⊥ severity** — a richer version of the transdiagnostic dissociation
than the 9-factor A = 4, as a *stable* (Tucker 0.979) continuum description. Figure: `archetype_profiles.png`.

### 2.4 Everyone is a blend (continuum-honest), and it is transdiagnostic

* Patients are genuinely blended across the five corners — no discrete "type"; the dominant-membership sizes
  (1,426–2,551) are balanced, none degenerate.
* All five archetypes mix cohorts (**ARI 0.006 vs DSM-5, 0.011 vs cohort**). The biology corner (A2) is
  DR-enriched (30% of DR vs 14% of BP), the well pole (A4) is BP/SZ-led (30%/29% vs 11% of DR), the
  activation corner (A0) is BP-led — but **none is a single diagnosis**.

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
  the A = 5 archetypes); biology shows up as an archetype corner / a continuous gradient, not a clean region
  boundary; archetype granularity is copula-sensitive (only A = 2, 4, 5 stable; A = 6 collapses); substance is thin (2 SUD
  binaries, DR = 0) and carried with wide uncertainty.
