# M1 cross-loading arm (sensitivity → candidate primary)

> Status: **fit running** (launched 2026-06-26). Results section is filled once the detached fit finishes.
> Canonical M1 is unchanged and untouched while this arm is evaluated.

## Why

The certified M1 is a **hard-zero bifactor** map: each indicator loads only on its home factor + the
general factor G; every specific↔specific cross-loading is fixed at exactly 0. That is a deliberate
identification choice — freeing *all* ~980 `unlikely_cross` cells floods every column and dilutes the
**thin** factors (substance ≈ 4 home items) until they lose identity (the documented reason hard-zero is
the default).

This arm tests the disciplined middle path we agreed on: **allow only the theory-motivated cross-loadings,
warm-started from the hard-zero solution, and grow by evidence — not by permissiveness.**

## What the arm frees (and the selection principle)

The prior matrix (`configs/prior_loading_matrix_v3.csv`) labels every `(item, factor)` cell with a
4-tier theory ontology: `primary` (home), `g_anchor` (G-defining), `plausible_cross` ("theory says this
relation is plausible — let the data decide"), `unlikely_cross` ("theory says no" → hard-zero). The 184
`plausible_cross` cells split three ways:

| sub-kind (rationale) | cells | already free in the hard-zero map? |
|---|---|---|
| "specific item may load on G" | 129 | yes — these are the `bifactor_G` cells |
| "cross-loading window" (MADRS/QIDS/STAI) | 12 | yes — the windows |
| **"theory-motivated cross-loading"** | 46 | **no — this is what the arm frees** |

Among the 109 fitted indicators, the theory-motivated specific cells number **37, and they are entirely
the immunometabolic bridge**: 29 metabolic-home markers allowed onto inflammatory, 8 inflammatory-home
markers onto metabolic. The v3 ontology encodes exactly one cross-*domain* mechanism it considers
biologically real (cardiometabolic ↔ immune share mechanism, so a marker like CRP or a lipid can
genuinely load on both); everything else specific↔specific is `unlikely_cross`.

So the arm is narrow and safe by construction:
- **37 cells between the two best-anchored factors** (metabolic 29 home items, inflammatory 8) — both thick.
- **Zero exposure to thin factors** (substance / mania / suicidality untouched).
- A clean, pre-registered test: *is the metabolic–inflammatory coupling better described as a Φ-correlation
  (currently 0.18) or as direct shared cross-loadings (some markers loading on both)?*

Engineering: `specific_cross=True`, `cross_sd_scale=1.0` → the cells get Normal(0, 0.25) (wide enough to
let the data speak; default would be 0.06, too tight). Warm-started from the certified `s5_9dim_mixed`,
so the backbone starts in the hard-zero basin and the 37 new cells start at 0.

## How to run

```bash
# wiring check (tiny draws; verifies 37 cells freed + warm-start loads):
HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_cross_loading_arm.py --smoke

# full fit — long (~1h); detached + supervised:
python3 scripts/run_job.py xcross -- env HDF5_USE_FILE_LOCKING=FALSE \
    python notebooks/run_cross_loading_arm.py
python3 scripts/status.py            # watch; live log: tail -f logs/xcross.log

# CI-aware loadings once the fit is done (arm-aware; writes reports/xcross_{loadings,phi}.csv):
HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_export_loadings.py \
    --idata results/face/oop_measurement/copula/s5_xcross/idata.nc
```

Outputs land in `results/face/oop_measurement/copula/s5_xcross/` (separate dir; the canonical fit is
never re-run or overwritten).

## Acceptance criteria (does it work, and could it become primary?)

The arm replaces the certified map **only if all hold**:

1. **Convergence is real, not init-stuck.** R̂ ≤ ~1.05, ESS healthy, 0–few divergences — AND congruent
   with an **over-dispersed-init** run (so good R̂ isn't just chains frozen at the warm start).
2. **Thin factors do not dilute.** substance / mania / suicidality home loadings stay where the certified
   map has them (the failure mode hard-zero was protecting against).
3. **Φ stays sensible.** If direct cross-loadings absorb the shared variance, metabolic↔inflammatory Φ
   should *drop* from 0.18 toward 0 — that is the key diagnostic that the coupling was a cross-loading,
   not a factor correlation. The rest of Φ should be stable.
4. **The cross-loadings are earned.** Report which of the 37 have a 95% CI excluding zero; a handful of
   credible, sign-sensible immunometabolic cross-loadings is a positive result, all-straddle-zero is a
   (also informative) null that vindicates the hard-zero default.

If it passes, it is the better-specified map (lets the immunometabolic biology express directly) and
becomes the candidate primary; if it fails any criterion, it stays a documented sensitivity arm.

## Growing by evidence (the roadmap, not this fit)

The 37 immunometabolic cells are the *entire* theory-plausible specific set — there are no more
`plausible_cross` specific cells to add. So extending beyond this is **evidence-driven, not
permissiveness-driven**: keep the confirmed cells, screen the `unlikely_cross` cells for strong residual
signal (modification-index style on held-out cells), free only the handful that earn it, refit
warm-started from this rung, and repeat until new cells stop earning their place — with the same
guardrails (over-dispersed-init check, thin-factor non-collapse, R̂/ESS, Φ stability) at every rung.
Freeing all 980 at once re-hits the dilution wall regardless of the warm start (it is an identification
problem, not a starting-point one).

## Results

_Pending the detached fit. Will record: R̂/ESS/divergences, surviving cross cells (CI≠0) with signs,
the metabolic↔inflammatory Φ shift, thin-factor home-loading stability, and the over-dispersed-init
congruence check._
