# M3 temporal coherence on the Gaussian-copula objects — findings

> **Canonical M3 findings record for the copula rerun.** Reworks the FACE M3 temporal-coherence layer on the
> copula M1/M2 objects (continuum: continuous coordinates + A=4 archetype simplex), parallel to native
> `scripts/30-37`. Engine [`src/face/temporal/temporal_model_oop.py`](../src/face/temporal/temporal_model_oop.py)
> (wraps the proven kernels — `invariance`/`variance`/`persistence`/`membership` + the strata scoring kernels —
> with **no edits** to native M3); driver [`notebooks/run_temporal_model_oop.py`](../notebooks/run_temporal_model_oop.py).
> The native-map [`TEMPORAL_FINDINGS.md`](TEMPORAL_FINDINGS.md) is retained as provenance. Pending PI sign-off.
> Updated 2026-06-22.

## What this is

M2 is a continuum on the copula map; M3 asks whether the copula map + the A=4 archetypes **persist over
V0→V1→V2**: does the measurement hold (G1), is biology trait while symptoms are state (G3), do the archetypes
keep their identity (G4)? It is a **consumer** of the fixed copula M1/M2 — V0 coordinates reused from the M2
object, **V1/V2 scored under the fixed copula M1** (the one new component), then the established temporal
kernels run with `A=4`. No re-discovery, no imputation.

**The new component — scoring V1/V2 under the fixed copula M1.** Native M3 freezes the V0 *parametric*
standardization; the copula M1 uses the frozen rank-INT map `z = Φ⁻¹(F_j(y))` **and** residualizes covariates.
So a follow-up score is: (1) orient + `copula_forward` each gaussianized cell onto the V0 z-scale via the
frozen `CoreData.copula[item]` map; (2) apply the **frozen-V0 covariate residualization** (age-spline+sex+edu+
site FWL) with the visit's covariates; (3) project onto the fixed copula Λ/Φ/σ (continuous:
`conditional_gaussian_draws`; explicit: `project_explicit_full_n`); (4) project onto the A=4 archetypes.
**Validated:** scoring V0 through this path reproduces the M2 copula coordinates at **Pearson r = 0.993–1.000**
across the 6 continuous axes — V0/V1/V2 live on the same scale. Panel: **16,241 rows** (V0 9,013 · V1 4,270 ·
V2 2,958), all 9 axes 100% finite at every visit. Full run 20 min, converged.

## Result 1 — G1: the measurement holds (all backbone axes temporally invariant)

Re-fitting the simple-structure backbone per visit (scale-invariant by design) and Tucker-φ'ing the primary
loadings vs V0: every backbone axis is **invariant** (φ ≥ 0.95):

| axis | min φ (worst follow-up) | license |
|---|---|---|
| sleep | 0.996 | invariant |
| cognition | 0.995 | invariant |
| overall_severity | 0.991 | invariant |
| metabolic | 0.988 | invariant |
| inflammatory | 0.974 | invariant |

Notably **inflammatory is invariant here (0.974)** — it was only *partial* on the native map. The construct is
measured consistently over time, so patient *change* on these axes is interpretable. (The explicit axes —
suicidality/developmental/substance — are not part of the continuous backbone test; mania is data-limited.)

## Result 2 — G3: biology is trait, symptoms are state (the headline, replayed)

Per-axis measurement-error variance decomposition (ICC = trait fraction; the M1 per-patient SD is plugged, so
ICC isolates genuine between-patient *trait* variance from measurement noise + within-person *state*):

| axis | ICC [94% HDI] | verdict |
|---|---|---|
| **metabolic** | 0.91 [0.90, 0.91] | **trait** |
| substance | 0.88 [0.86, 0.90] | trait |
| cognition | 0.70 [0.66, 0.72] | trait |
| inflammatory | 0.62 [0.61, 0.64] | trait |
| overall_severity | 0.62 [0.60, 0.63] | trait (by rank) |
| sleep | 0.47 [0.45, 0.48] | mixed |
| suicidality | 0.42 [0.39, 0.46] | mixed |
| developmental_risk | 0.39 [0.34, 0.44] | state |
| mania_activation | 0.88 [0.81, 0.94] | **uninformative** (data-limited) |

**The durable biology axes (metabolic/cognition/inflammatory) are trait; the moving symptom axes
(developmental, suicidality, sleep) are state** — reproducing the native M3 headline on the better map.
**Severity is trait *by rank*** (ICC 0.62 — patients keep their relative position) **while the population
improves** (pop_slide −0.46 over V0→V2) — the same trait-by-rank / slides-at-population nuance as native.
Clinical logic unchanged: *stratify on the durable biology, monitor the moving symptoms.*

## Result 3 — G4: archetype identity persists (geometry agrees with the variance route)

Over V0→V2 completers (n = 2,958): the **Arm-B (⊥G) archetype weights persist** (weight-vector cosine median
**0.90**, q10 0.70); dominant-archetype agreement is 0.49 (κ 0.30) — the expected *churn of the argmax on a
continuum* (central patients flip label by geometry while their weight vector barely moves). Severity-spine
trajectories are **58% stable / 35% drifting / 7% oscillating**. The G3⟷G4 synthesis (Spearman of per-axis
reliable-change-rate vs ICC) is **ρ = −0.27** — trait axes change less, state axes more, the right direction
(weak at n = 9 axes, p = 0.49). So the variance route (G3) and the geometric route (G4) agree: biology holds,
symptoms move, archetype identity persists.

## Honest caveats

* **G4 reliable-change rate is measurement-precision-confounded.** Metabolic is measured very precisely (tight
  posterior SD), so even tiny real shifts clear the reliable-change threshold — its raw "movement" rate
  overstates instability. The **error-corrected G3 ICC** (which removes measurement variance) is the cleaner
  trait/state signal, and it is unambiguous (metabolic ICC 0.91 = trait). Read G3 as the headline, G4 as the
  geometric corroboration.
* **Mania is uninformative** (ICC high but signal-ratio < 0.5 — only 2 indicators, partial for all patients);
  its trait/state verdict is not load-bearing. Same data limit as native.
* **Developmental scores state** here; as on the native map this partly reflects CTQ recall-noise (the
  construct is trait by design) — interpret with care.
* **V0 reused from the M2 posterior, V1/V2 projected** (the native asymmetry); the V0 reproduction at r ≈ 0.99
  bounds the inconsistency. Internal/temporal-coherence validity only.

## Hand-off

`results/face/temporal_oop/`: `invariance/{license,congruence}.csv`, `trait_state/trait_state.csv`,
`persistence/{reliable_change.csv, persistence.json}`, `attrition/ipw_weights.parquet`, and the M4-contract
hand-off `consolidate/{patient_panel.parquet (16,241 × 95: panel coords + A=4 memberships + G1 license +
retention/IPW), panel_draws.npz}`. Figure: `docs/figures/temporal_oop/trait_state_icc.png`. Reproduce:
`PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_temporal_model_oop.py --mode full`.

**Verdict: the copula map + the A=4 archetypes are temporally coherent** — the measurement holds, biology is
durable (trait) while symptoms slide (state), and archetype identity persists — on the same baseline cohort,
reproducing native M3 with one improvement (inflammatory now invariant) and the honest caveats above.
