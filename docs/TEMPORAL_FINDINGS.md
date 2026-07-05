# M3 temporal coherence on the Gaussian-copula objects — findings

> **Canonical M3 findings record (8-factor map, 2026-06-27; PI sign-off 2026-06-27).** The FACE M3
> temporal-coherence layer on the **8-factor** copula M1/M2 objects (continuum: continuous coordinates +
> **A = 5** archetype simplex; immunometabolic biology axis + substance pinned orthogonal). Engine
> [`src/face/temporal/temporal_model_oop.py`](../src/face/temporal/temporal_model_oop.py) (wraps the
> `invariance`/`variance`/`persistence`/`membership` kernels + the strata scoring kernels); driver
> [`notebooks/run_temporal_model_oop.py`](../notebooks/run_temporal_model_oop.py). Map consumed:
> `copula/weighted_8d/hs_s5_merged_xc`.

## What this is

M2 is a continuum on the 8-factor copula map; M3 asks whether the map + the **A = 5** archetypes **persist over
V0→V1→V2**: does the measurement hold (G1), is biology trait while symptoms are state (G3), do the archetypes
keep their identity (G4)? It is a **consumer** of the fixed copula M1/M2 — V0 coordinates reused from the M2
object, **V1/V2 scored under the fixed copula M1**, then the temporal kernels. No re-discovery on
follow-up.

## Result 1 — G1: the measurement holds (all backbone axes temporally invariant)

Re-fitting the simple-structure backbone per visit (scale-invariant by design) on the **8-factor backbone**
(severity + cognition + **immunometabolic** + sleep) and Tucker-φ'ing the primary loadings vs V0: **every
backbone axis is invariant (φ ≥ 0.95), 4/4:**

| axis | min φ (worst follow-up) | license |
|---|---|---|
| sleep | 0.996 | invariant |
| cognition | 0.995 | invariant |
| overall_severity | 0.991 | invariant |
| **immunometabolic** | **0.987** | **invariant** |

**The immunometabolic biology axis is fully invariant (0.987).** The biology construct is measured consistently over
time, so patient *change* on it is interpretable. (The explicit axes — suicidality/developmental/substance —
are not part of the continuous backbone test; mania is data-limited.)

## Result 2 — G3: biology is trait, symptoms are state (the headline)

Per-axis measurement-error variance decomposition (ICC = trait fraction; the M1 per-patient SD is plugged, so
ICC isolates between-patient *trait* variance from measurement noise + within-person *state*):

| axis | ICC [94% HDI] | pop slide V0→V2 | verdict |
|---|---|---|---|
| **immunometabolic** | **0.91 [0.90, 0.91]** | +0.04 | **trait** (the durable biology anchor) |
| cognition | 0.70 [0.67, 0.72] | −0.14 | **trait** |
| overall_severity | 0.62 [0.60, 0.63] | −0.46 | **trait (by rank)** |
| substance | 0.49 [0.46, 0.53] | −0.17 | mixed |
| sleep | 0.47 [0.45, 0.48] | −0.09 | mixed |
| suicidality | 0.43 [0.39, 0.46] | −0.84 | mixed |
| developmental_risk | 0.39 [0.33, 0.44] | −0.13 | state |
| mania_activation | 0.79 [0.74, 0.85] | −0.19 | **uninformative** (data-limited) |

**The durable biology is trait and is the single most durable axis (immunometabolic ICC 0.91), with cognition
(0.70) trait beside it; the moving symptom axes (developmental, suicidality, sleep) are state/mixed.** Severity is
**trait *by rank*** (ICC 0.62 — patients keep their relative position) **while the population improves**
(pop_slide −0.46; suicidality slides hardest, −0.84). Clinical logic: *stratify on the durable biology, monitor
the moving symptoms.*

## Result 3 — G4: archetype identity persists (continuum-honest)

Over V0→V2 completers (n = 2,958), on the **A = 5** simplex:

* **Archetype weights persist** — Arm-B (⊥G) weight-vector cosine median **0.81** (q10 0.49). Dominant-archetype
  agreement is **0.40** (κ 0.19) — *substantial argmax churn*, which is **expected**: five
  corners create more argmax boundaries, so central patients flip label by geometry while their weight vector
  barely moves. The persistence lives in the **weights** (cosine 0.81), not the hard label.
* **Spine moves while biology holds** — decomposing each patient's Δx into the severity spine vs the biology
  corner (immunometabolic + cognition): **spine-moves-not-biology 0.234 > biology-moves-not-spine 0.163**.
  Severity slides more than biology, consistent with G3. (Stable 58% / drifting 35% / oscillating 7%.)

**Honest caveat:** the G3⟷G4 *cross-route synthesis* (Spearman of per-axis reliable-change-rate
vs ICC) is **ρ = 0.07 (p = 0.87)** — near zero. The two routes do not clearly co-rank (the A = 5 churn + the
data-limited mania/orthogonal substance muddy an 8-point correlation). G3 (the error-corrected ICC) is the clean
headline; read G4 as geometric persistence of identity (cosine 0.81), not as an independent confirmation of the
variance ranks.

## Honest caveats

* **G3 is the headline, G4 the geometric corroboration.** G4 reliable-change is measurement-precision-confounded
  (precisely-measured immunometabolic clears the change threshold on tiny shifts), so read the error-corrected
  G3 ICC for trait/state; it is unambiguous (immunometabolic 0.91 = trait).
* **Substance reads as mixed** (ICC 0.49). Expected: substance is pinned
  **orthogonal** (no Φ borrowing) and is thin (2 SUD binaries) — its coordinate is noisier, so less of its
  variance reads as durable trait.
* **Mania is uninformative** (ICC high but signal-ratio < 0.5 — 2 indicators); its verdict is not load-bearing.
* **Developmental scores state** — this partly reflects CTQ recall-noise (the construct is
  trait by design); interpret with care.
* **V0 reused from the M2 posterior, V1/V2 projected** (the scoring asymmetry); internal/temporal-coherence
  validity only.

## Hand-off

`results/face/temporal_oop/`: `invariance/{license,congruence}.csv`, `trait_state/trait_state.csv`,
`persistence/{reliable_change.csv, persistence.json}`, `attrition/ipw_weights.parquet` (feeds M4 + repbench),
and the M4-contract hand-off `consolidate/{patient_panel.parquet (16,241 × 92: panel coords + **A = 5**
memberships + G1 license + retention/IPW), panel_draws.npz}`. Figure: `docs/figures/m3_temporal/
trait_state_icc.png`. Reproduce:
`PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_temporal_model_oop.py --mode full`.

**Verdict: the 8-factor copula map + the A = 5 archetypes are temporally coherent** — the measurement holds
(4/4 invariant, immunometabolic fully invariant), biology is durable (immunometabolic ICC 0.91) while
symptoms slide, and archetype identity persists in the weights (cosine 0.81), with the honest caveats above
(weak G3⟷G4 cross-route agreement; substance orthogonal).
