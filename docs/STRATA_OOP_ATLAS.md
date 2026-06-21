# M2 soft-region atlas — what K and A mean, in detail (copula map)

> Detailed results analysis for the reworked M2 (the continuum of soft operational regions on the
> Gaussian-copula map). Paper-facing summary: [STRATA_OOP_FINDINGS.md](STRATA_OOP_FINDINGS.md). Engine:
> [`src/face/strata/strata_model_oop.py`](../src/face/strata/strata_model_oop.py). Figures:
> `docs/figures/strata_oop/`. All coordinates are on the latent z-scale (0 = population mean, units = SD).

## 0. Two views of one continuum — why both K and A

The structure gate says the 9-dim space is a **continuum** (no discrete clusters). On a continuum there is no
"correct number of groups" — so we describe it two complementary ways, and the difference between them is the
whole point:

| | **K — soft tessellation** | **A — archetypes** |
|---|---|---|
| object | `K` soft regions of a measurement-error mixture | `A` extreme phenotypes (corners of the convex hull) |
| membership | responsibilities `r_i` (sum to 1) | simplex weights `w_i` (sum to 1) — a patient *is* a blend |
| what it finds | the dominant **density** split — where the cloud is most separable into regions | the **extremes** — the orthogonal directions of maximal phenotype |
| answer it gives | "which **decision region** is this patient in?" | "**where on the continuum** is this patient — what blend of extremes?" |
| granularity rule | smallest `K` that stays confidently assignable + stable | largest `A` that stays cross-seed reproducible |
| result here | **K = 2** | **A = 4** |

Both partition the *same* points; they answer different questions. The headline finding is that they split on
**different axes** — and that dissociation is exactly the transdiagnostic structure we are after.

---

## 1. K — the soft tessellation splits on psychiatric symptom burden

### 1.1 Why K = 2

`K` is a deliberate operational granularity, not a discovered kind-count (a continuum has no natural K). The
XD-BIC is essentially **flat** across K (a continuum signature — no interior optimum), and assignment
confidence falls as K grows, so the rule picks the smallest confidently-assignable, reproducible K:

| K | XD-BIC | confident-dominant | median entropy | seed-ARI |
|---|---|---|---|---|
| **2** | 197,963 | **1.00** | 0.51 | **1.00** |
| 3 | 197,918 | 0.92 | 0.60 | 0.97 |
| 4 | 198,108 | 0.87 | 0.55 | 1.00 |
| 5 | 198,345 | 0.84 | 0.49 | 0.97 |
| 8 | 199,545 | 0.69 | 0.50 | 0.77 |

BIC moves < 1% across the whole range — there is no "right" K; K=2 is the parsimonious decision-region scheme.
(K=3–4 remain confident + stable if finer regions are wanted; they sub-divide the same symptom axis.)

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

* **Operationally:** for a coarse decision-region label, stratify on the **symptom-burden** axis (K=2,
  suicidality-anchored). For the richer phenotype, place a patient as a **blend of the four archetypes** —
  reading their biological (A0) load separately from their symptom (A3) and severity (A2) load.
* **Transdiagnostic:** neither view re-encodes diagnosis (ARI ≈ 0); the structure cuts across BP/SZ/DR and the
  DSM-5 subtypes.
* **Internal/baseline only.** Whether these regions/archetypes *predict* 2-year course or treatment response —
  i.e. whether "splits on symptoms / has a biology corner" is *useful for decisions* — is the **M3/M4 rerun**
  on this object, not claimed here. The biology corner (A0) is the natural candidate to carry durable/prognostic
  signal (it did on the native map), but that is a hypothesis for M3/M4.
* **Caveats:** K=2 is coarse by design; the biology signal lives in the archetypes, not the tessellation;
  archetype granularity is copula-sensitive (only A=2, 4 stable); substance is thin (2 SUD binaries, DR=0) and
  carried with wide uncertainty.
