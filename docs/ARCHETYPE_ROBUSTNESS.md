# Archetype-robustness battery — is the archetype simplex the right, reproducible reading lens?

> **Map of record (read first).** The strata reading lens is **A = 5 archetypes (A0–A4)** on the **8-factor
> immunometabolic map** — the largest A whose corners reproduce across seeds (cross-seed Tucker ≥ 0.8), with a
> clean stability cliff at **A = 6** (0.979 → 0.436; EV 0.60). Canonical archetype findings:
> [`STRATA_FINDINGS.md`](STRATA_FINDINGS.md), atlas [`STRATA_ATLAS.md`](STRATA_ATLAS.md). The
> battery below is the robustness *protocol* (A-selection, cross-seed reproducibility, n_init sensitivity, anchor
> recovery, degeneracy, membership health, split-half) that establishes the simplex is earned and not a fitting
> artefact; read it for the method.

> **Why this exists.** The A=5 archetype simplex is load-bearing twice: it is the M2 *reading lens* (the five
> corners + their representative patients) **and** M4's predictive carrier (the durable-pair-alone EIV is
> ambiguous, so the archetype representation carries the prognostic signal). Archetypal analysis (AA) *always*
> returns A corners — so, exactly as M2 falsifies clusters with a single-Gaussian null before trusting them, we
> must show the A=5 corners are (a) the right count, (b) reproducible, and (c) not a fitting artefact, before
> leaning on them. This is the AA counterpart of the structure gate.

Battery: `notebooks/archetype_geometry/robustness_battery.py` (wraps the production kernels in
`src/face/strata/archetypes.py`; **no re-derivation of the map**). Reference corners = the persisted production
fit (`consolidate/archetype_profiles.csv`, arm A; A = 5). Protocol: reproducibility uses **n_init=4** (single
restarts are unreliable — that is check 3); refit-heavy checks subsample to **N=3,500** (n_init=4 on 3,500
reproduces the production corners to Tucker ≈0.985, validated). Outputs:
`results/face/strata_oop/archetype_robustness/{archetype_robustness.csv, robustness_detail.json}` +
`docs/figures/archetype_geometry/robustness_battery.png`.

## Verdict — 8 PASS · 1 CONDITIONAL · 0 FAIL

| # | check | result | verdict |
|---|---|---|---|
| 1 | **A-selection** (stability-gated) | chosen **A=5**; cross-seed Tucker {2:0.999, 3:0.813, 4:0.997, **5:0.979**, 6:0.436} — stability holds through A=5 then collapses at A=6 | **PASS** |
| 2 | **cross-seed reproducibility @A=5** | min Tucker **0.979** (full N, 3 seeds); every corner ≥0.979 | **PASS** |
| 3 | **n_init sensitivity** | single restarts scatter (0.36–0.99); **n_init=4 → 0.98** (why the n_init=4 protocol) | **PASS** |
| 4 | **anchor recovery** (bootstrap / draws) | bootstrap median **0.95**; measurement-draw median **0.69**; 56% refits ≥0.8 | **CONDITIONAL** |
| 5 | **granularity** (A=5 ceiling) | Tucker A5 **0.979** → A6 **0.436** (<0.8): A=5 is the reproducibility ceiling, the clean stability cliff is at A=6; A=3 dips (0.813) but the simplex re-stabilises by A=5 (AA non-hierarchical) | **PASS** |
| 6 | **A=8 over-resolves** | an A=8 simplex has cross-seed Tucker **0.32** ≪ A=5 0.979 — A=8 does **not** reproduce on these coordinates | **PASS** |
| 7 | **no degenerate corners** | max pairwise corner cosine **−0.11** (corners point in different directions); min separation 6.2 SD | **PASS** |
| 8 | **membership health** (continuum-honest) | mean blend entropy **0.78**; only 39% near a pole, 25% on a soft boundary — most patients are genuine blends | **PASS** |
| 9 | **split-half out-of-sample** | cross-half Tucker **0.936**; project held-out half onto half-1 anchors → OOS R² 0.51 | **PASS** |

## What it establishes
- **A=5 is the right count, and the corners are reproducible.** Stability holds (≥0.98) through A=5 and
  collapses at A=6 (0.436); the corners reproduce across seeds (0.979), across independent halves (0.936),
  across patient bootstraps (0.95), and as long as enough restarts are used. They are not duplicates
  (cosine −0.11) and an **A=8 simplex does not transfer** to these coordinates (0.32) — substantiating that
  A=8 over-resolves the cloud.
- **A methodological finding, not a bug:** a *single* AA restart is unreliable (congruence to the production
  corners scatters 0.36–0.99) — AA is non-convex. The production protocol (**n_init=4**, best-of-restarts) is
  both **necessary and sufficient** (→ 0.98). Any reuse of these corners must keep n_init≥4.
- **Continuum-honest membership:** mean blend entropy 0.78, only ~39% of patients near any pole — consistent
  with M2's verdict that the cloud is a continuum and the corners describe its *shape*, not discrete kinds.

## The one CONDITIONAL — and why it is the expected, honest answer
Check 4 separates two questions a single congruence statistic conflates:
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
