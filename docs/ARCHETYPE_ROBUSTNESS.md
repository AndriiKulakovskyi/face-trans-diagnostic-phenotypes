# Archetype-robustness battery — is A=4 the right, reproducible reading lens?

> **Why this exists.** The A=4 archetype simplex is load-bearing twice: it is the M2 *reading lens* (the four
> corners + their representative patients) **and** M4's predictive carrier (on the copula objects the
> durable-trio-alone EIV is no longer robust, so the archetype representation carries the prognostic signal).
> Archetypal analysis (AA) *always* returns A corners — so, exactly as M2 falsifies clusters with a single-
> Gaussian null before trusting them, we must show the A=4 corners are (a) the right count, (b) reproducible,
> and (c) not a fitting artefact, before leaning on them. This is the AA counterpart of the structure gate.

Battery: `notebooks/archetype_geometry/robustness_battery.py` (wraps the production kernels in
`src/face/strata/archetypes.py`; **no re-derivation of the map**). Reference corners = the persisted production
fit (`consolidate/archetype_profiles.csv`, arm A). Protocol: reproducibility uses **n_init=4** (single restarts
are unreliable — that is check 3); refit-heavy checks subsample to **N=3,500** (n_init=4 on 3,500 reproduces the
production corners to Tucker ≈0.985, validated). Outputs:
`results/face/strata_oop/archetype_robustness/{archetype_robustness.csv, robustness_detail.json}` +
`docs/figures/archetype_geometry/robustness_battery.png`.

## Verdict — 8 PASS · 1 CONDITIONAL · 0 FAIL

| # | check | result | verdict |
|---|---|---|---|
| 1 | **A-selection** (stability-gated) | chosen **A=4**; cross-seed Tucker {2:1.00, 3:0.94, **4:1.00**, 5:0.51, 6:0.41} — stability holds to A=4 then collapses | **PASS** |
| 2 | **cross-seed reproducibility @A=4** | min Tucker **0.997** (full N, 3 seeds); every corner ≥0.997 | **PASS** |
| 3 | **n_init sensitivity** | single restarts scatter (0.36–0.99); **n_init=4 → 1.00** (why the n_init=4 protocol) | **PASS** |
| 4 | **anchor recovery** (bootstrap / draws) | bootstrap median **0.95**; measurement-draw median **0.69**; 56% refits ≥0.8 | **CONDITIONAL** |
| 5 | **granularity** (A=4 ceiling) | Tucker A4 1.00 → A5 **0.51** (<0.8): A=4 is the reproducibility ceiling; A=3 does not cleanly nest (AA non-hierarchical) | **PASS** |
| 6 | **native A=8 on the copula** | A=8 cross-seed Tucker **0.32** ≪ A=4 1.00 — the native A=8 does **not** transfer | **PASS** |
| 7 | **no degenerate corners** | max pairwise corner cosine **−0.11** (corners point in different directions); min separation 6.2 SD | **PASS** |
| 8 | **membership health** (continuum-honest) | mean blend entropy **0.78**; only 39% near a pole, 25% on a soft boundary — most patients are genuine blends | **PASS** |
| 9 | **split-half out-of-sample** | cross-half Tucker **0.936**; project held-out half onto half-1 anchors → OOS R² 0.51 | **PASS** |

## What it establishes
- **A=4 is the right count, and the corners are reproducible.** Stability is ~1.0 through A=4 and collapses at
  A=5 (0.51); the corners reproduce across seeds (0.997), across independent halves (0.936), across patient
  bootstraps (0.95), and as long as enough restarts are used. They are not duplicates (cosine −0.11) and the
  full **native A=8 does not transfer** to the copula coordinates (0.32) — substantiating the report's claim
  that A=8 is a native-map artefact.
- **A methodological finding, not a bug:** a *single* AA restart is unreliable (congruence to the production
  corners scatters 0.36–0.99) — AA is non-convex. The production protocol (**n_init=4**, best-of-restarts) is
  both **necessary and sufficient** (→ 1.00). Any reuse of these corners must keep n_init≥4.
- **Continuum-honest membership:** mean blend entropy 0.78, only ~39% of patients near any pole — consistent
  with M2's verdict that the cloud is a continuum and the corners describe its *shape*, not discrete kinds.

## The one CONDITIONAL — and why it is the expected, honest answer
Check 4 separates two questions the old single statistic conflated:
- **Sample stability** (bootstrap, posterior-mean coordinates): **0.95** — the corner *structure* is robust to
  which patients are sampled.
- **Location precision under propagated measurement uncertainty** (refit on single M1 posterior draws):
  **0.69** — when the full per-patient measurement error is pushed into the corner *positions*, the rarer
  extreme corners (defined by relatively few extreme patients) wobble.

This is exactly what should happen for convex-hull extremes, and it is **why the load-bearing object of M2 is
the continuous coordinates + their uncertainty, with the corners as a reading lens** — not precise coordinates.
The production pipeline already respects this: anchors are fit on the posterior **mean**, and **membership**
uncertainty (not anchor-location certainty) is what is propagated to each patient (`arch_w*_sd`). So the
CONDITIONAL is not a weakness of the analysis; it is the empirical justification for the reframe.

## Reproduce
```
PYTHONPATH=$PWD/src python notebooks/archetype_geometry/robustness_battery.py   # ~7 min, ~46 AA fits
```
(Companion exploratory geometry — the cloud, density landscapes, candidate extremes —
`notebooks/archetype_geometry/{visualize_cloud,explore_landscape}.py`.)
