# Tier-0 / low-effort reruns — results

Four analyses run on the fitted model (`results/face/gllvm_oop/s8_full/model_state.pt`),
reusing the published Fisher-information and reconstruction accounting verbatim. Each maps to
an improvement item in `REVIEW_article_methods.md`.

---

## T0.3 — Information-ranked vs loading-ranked battery *(review item: the cheapest proof of added value)*

**Result: information selection wins at every matched size.** The greedy battery built by
ranking items on Fisher information beats the same battery built by ranking on |loading|:

| Battery size N | Info-ranked | Loading-ranked | **Gap** |
|---:|---:|---:|---:|
| 20 | 0.625 | 0.407 | **+0.218** |
| **27** | **0.705** | **0.599** | **+0.106** |
| 35 | 0.744 | 0.606 | +0.138 |

- My info-rule reproduces the paper's published headline (0.70 at 27 items → 0.705) exactly,
  confirming faithful reuse of the original accounting.
- **This is the demonstrated added value the reviewer demanded for Fig 2's thesis:** at the
  headline 27-item battery, choosing items by information rather than loading buys **+0.11 mean
  reliability** — a decision that materially changes the battery. The loading rule wastes early
  picks on high-loading, low-information items (rare binary flags), stalling near 0.11 until
  item ~16.
- **Where to use it:** new panel in Fig 2 or Fig 4; a sentence in Results §Design. Fixes
  utility-gap #6 and #7, and answers reviewer question #5 for the design half of the paper.

## T2.3 — Woodbury vs naive inversion runtime *(review item T2.3: measured, not asserted)*

**Result: 6.2× faster at the real median, exact to 10⁻¹⁴.**

| Observed items \|C_i\| | Naive (µs) | Woodbury (µs) | Speedup |
|---:|---:|---:|---:|
| 10 | 5.1 | 6.7 | 0.8× |
| 50 | 29.3 | 9.9 | 3.0× |
| 88 (real median) | — | — | **6.2×** (interp.) |
| 100 | 117.7 | 16.0 | 7.3× |
| 125 | 203.9 | 20.5 | 9.9× |

- Correctness: `max|Σ⁻¹_naive − Σ⁻¹_woodbury| = 1.1×10⁻¹⁴`.
- Honest nuance worth stating: below \|C_i\|≈20 the reduction is *slightly slower* (the
  capacitance inverse costs more than a tiny direct inverse) — the win begins once \|C_i\|
  exceeds the factor dimension, which it does for essentially every real patient (median 88,
  p10–p90 = 53–108).
- Replaces the asserted "O(\|C_i\|³) → O((K+1)³)" claim with a measured speedup. Fixes
  utility-gap #9.

## T2.4 — Reconstruction R² sensitivity to archetype count A *(review item T2.4: not fragile)*

**Result: smooth and monotone — the 59% is not an artifact of A=5.**

| A | pooled R² |
|---:|---:|
| 4 | 0.529 |
| 5 | 0.616 (fresh refit) · **0.590 (published fixed corners)** |
| 6 | 0.690 |

The fresh AA refit at A=5 (0.616) reproduces the published fixed-corner value (0.590) within
reinitialization noise. Conclusions do not hinge on the specific choice of five corners.

## T2.5 — R² anchored against the full 0→100% range *(review item T2.5, and it sharpened T1.2)*

**Result — and this one is important.** Placing the 59% on the full ladder:

| Summary | R² |
|---|---:|
| Raw 8-D coordinate (no summary) | 1.000 |
| PCA-5 (best 5-D linear) | 0.796 |
| PCA-4 (fair affine dim) | 0.680 |
| **Random 5-D subspace** (mean, 500 seeds) | **0.624 ± 0.042** |
| **Archetype-5 (this paper)** | **0.590** |
| k-means-5 (hard partition) | 0.325 |
| Single centroid (A=1) | 0.000 |

**A random 5-D subspace retains *more* variance than the archetype summary (0.624 vs 0.590).**
The reason: the map's coordinate variance is near-isotropic — the top principal component holds
only 26.9%, and it takes 5 PCs to reach 80%. The expected variance a Haar-random *k*-D subspace
captures is exactly *k*/*d* = 5/8 = **0.625** regardless of the eigenvalue spectrum (the 500-seed
empirical mean, 0.624, confirms it). The archetype-5 R² of 0.590 therefore sits at only the
**~21st percentile** of random 5-D projections — *below* a typical random subspace (by 0.034 R²
units, z = −0.82), not equal to it.

**This does not weaken the paper — it corrects the framing (review item T1.2), and now with a
number behind it.** Reconstruction R² is the *wrong* yardstick for archetypes: they were never
competing on variance capture, and a random subspace beats them on it. Their value is what no
PCA or random subspace provides — **convex, interpretable, clinically-nameable corners with a
per-patient simplex membership.** Report R² as an honesty check on how much of the continuum the
corners encode, **not** as a competitive metric. Pair this with the non-significant silhouette
(T1.3): together they say the data are a continuum with no natural clusters, so a convex-blend
simplex — not a hard partition, and not a variance-optimal projection — is the right summary.

> **Recommendation:** in Results §archetype, drop any implicit "archetypes retain 59% *vs* PCA's
> 80%" competitive read. Replace with: (a) the silhouette/continuum argument as the lead, (b) the
> anchor ladder showing R² is uninformative here because variance is isotropic, (c) the corners'
> interpretability + per-patient membership as the actual deliverable. The k-means comparison can
> go to supplement (review kill-list); the random-5 anchor should stay — it is the honest proof
> that variance capture is the wrong axis.

---

## New figure

`fig7_added_value.png/.pdf` — two panels: (a) info- vs loading-ranked battery with the +0.11
gap at 27 items; (b) the R² anchor ladder with the archetype-5 / random-5 coincidence and an
A-sensitivity inset. 300 dpi, house palette, render-verified.

## What still needs deciding (not in this batch)

The two **blockers** from the review remain (they need new simulation, not a rerun):
- **T0.1** — downstream decision utility of EAP+S_i vs a sum-score point estimate (higher-impact
  blocker; recommended next).
- **T0.2** — misspecification/perturbed-DGP calibration stress test.

And the reframes **T1.1–T1.4** are pure text edits to the manuscript, now all backed by numbers
(T1.2 and T1.3 especially).
