# The vertical — consolidated findings (M2 · M3 · M4 · M5)

> **Map of record (read first).** The whole vertical sits on the **8-factor immunometabolic map**: G (overall
> burden) + 7 specific axes — cognition, **immunometabolic** (one biology factor: cardiometabolic +
> inflammatory load together), sleep, mania/activation, suicidality, developmental-risk, and **substance**
> (orthogonal) — with **3 earned cross-loadings** into cognition. The strata reading lens is **A = 5
> archetypes (A0–A4)**, and **A2 is the immunometabolic (biology) corner**: the most distinct (M2), the most
> durable (M3), and the worst-prognosis (M4). Per-milestone canonical records:
> [`STRATA_OOP_FINDINGS.md`](STRATA_OOP_FINDINGS.md), [`STRATA_OOP_ATLAS.md`](STRATA_OOP_ATLAS.md),
> [`TEMPORAL_OOP_FINDINGS.md`](TEMPORAL_OOP_FINDINGS.md), [`PROGNOSIS_OOP_FINDINGS.md`](PROGNOSIS_OOP_FINDINGS.md),
> [`TREATMENT_OOP_FINDINGS.md`](TREATMENT_OOP_FINDINGS.md). diagnosis is validation-only.

> **Paper-facing synthesis.** On the certified cohort-weighted full-N
> **Gaussian-copula** measurement map (M1), the M2 stratification, M3 temporal coherence, M4 prognosis, and M5
> treatment moderation were each rebuilt as **parallel OOP engines that wrap the proven kernels** and leave the
> native pipelines (`scripts/20-57`) untouched. This is the one-page read of what the vertical found and what it
> means.
> Per-milestone canonical records: [`STRATA_OOP_FINDINGS.md`](STRATA_OOP_FINDINGS.md) ·
> [`STRATA_OOP_ATLAS.md`](STRATA_OOP_ATLAS.md) · [`TEMPORAL_OOP_FINDINGS.md`](TEMPORAL_OOP_FINDINGS.md) ·
> [`PROGNOSIS_OOP_FINDINGS.md`](PROGNOSIS_OOP_FINDINGS.md). One-figure summary:
> `docs/figures/copula_vertical/synthesis.png`. Pending PI sign-off. Updated 2026-06-24.

![synthesis](figures/copula_vertical/synthesis.png)

## The integrated headline

One object holds the three milestones together: **a biology-aware, continuous, transdiagnostic map that is
real (M2), durable (M3), and prognostic for functioning (M4)** — with the **biological corner** the most
distinct (M2), the most durable (M3), and the worst-prognosis (M4) of the four extremes. It is not biotypes
and not a clean clinical
category; it is a continuum with a stable set of extremes. Its demonstrated value is **group-level
stratification + continuous functional forecasting**, co-informative with DSM-5 — not an individual yes/no
calculator. The reworks **reproduce the native-map conclusions**, which is itself a robustness result (the
findings do not depend on the Gaussian-vs-copula likelihood).

## The program arc, restated under the continuum

The reframe changes the *object* the program delivers, not its questions. The original four-layer arc read
**diagnostic cohorts → transdiagnostic dimensions → validated strata → prognosis / treatment** — a pipeline whose
third layer was *K discrete patient types you assign and then treat*. The continuum verdict retires that layer as
stated (there are no biotypes), so the arc is now

> diagnostic cohorts → transdiagnostic dimensions → **continuous map + A = 5 archetype simplex (no privileged K)** →
> prognosis / treatment.

The deliverable shape changed with it — from *a typology you assign* to **a continuous coordinate system + a
stable archetype simplex you forecast and group-stratify on**. Every milestone's *question* survives intact; only
its noun changes (from "the strata" to "the coordinates / archetypes"):

| milestone | originally asked | restated under the continuum | status |
|---|---|---|---|
| **M2** | do *validated strata* exist? | is the space types or a continuum? | continuum; load-bearing = coordinates + A = 5 simplex |
| **M3** | do the *strata* persist? | do the coordinates + archetype identity persist? | yes (biology trait, symptoms state); **G5 retired** |
| **M4** | do the *strata* predict, at which *K*? | does the continuous/archetype encoding predict; does any hard partition add value? | yes (functioning); **operative K = none** |
| **M5** | do the *strata* moderate treatment? | what does the map license — describe, predict, or prescribe? | **bounds & defends**: MDE-bounded moderation null (not prescriptive); defends M4 (carrier survives, IPW-robust); describes response heterogeneity. Selection = M5b |

Two sub-goals are *dissolved* rather than answered, and that is itself a finding. **M3's G5** (a
stratum-label-switching test vs DSM-5) is ill-posed without discrete labels — its intent (the map is a more
durable phenotype than diagnosis) is already carried by the trait/state result. **M4's "which K"** is answered
*none* — and the K-family comparison is the falsification that *earns* that answer, not redundant work.

The reframe also disciplines the *ambition*. The demonstrated value is **group-level functional forecasting +
continuous stratification, co-informative with DSM-5** — not an individual biotype-prescribing engine; the
ΔELPD +59 / ΔAUC +0.011 split is the signature of that gap (developed in *Honest tensions* and *Calibrated claim
& what's left* below). What the reframe does **not** change is where genuine new evidence must come from: an
**external cohort** (the map is internally validated only), and **randomized / trial-arm data for a true M5b**
treatment-*selection* test (with the FondaMental prescription data the first place to look). Neither is an
internal re-run — the copula vertical engines already embody the continuum object end to end.

## M2 — the structure: a continuum with stable extremes (no privileged K)

- **Continuum, not biotypes**, established by a single-Gaussian falsification null: the best partition of the
  cloud separates patients no better than a structureless Gaussian (silhouette **0.140 real vs 0.137 ± 0.002
  null, z = 1.13, n.s.**; HDBSCAN 0 clusters; one connected component).
- The tessellation is therefore a **nested K-family (2/3/4), no privileged K**. The load-bearing object is the
  **A = 5 stable archetype simplex** — the largest A whose corners reproduce cross-seed (Tucker ≥ 0.8), with a
  clean stability cliff at A = 6.
- The five corners carry the payload — **biology ⊥ symptoms ⊥ severity**: **A0** activation/sleep, **A1**
  severe clean-biology, **A2** immunometabolic (↑immunometabolic ↑severity ↑suicidality — *the biology
  corner*), **A3** trauma/suicidality, **A4** low-burden / well. All views transdiagnostic (ARI ≈ 0 vs cohort
  and DSM-5). *(Figure panel A.)*

## M3 — the durability: biology is trait, symptoms are state

- **G1 measurement holds**: all **4/4** backbone axes temporally invariant (G, cognition, immunometabolic,
  sleep); the merged **immunometabolic** axis is fully invariant (φ 0.987).
- **G3 trait/state (ICC)**: **immunometabolic 0.91 — the single most durable axis**; cognition 0.70 trait;
  severity **0.62 trait by rank** while the population improves (slide −0.46; suicidality slides hardest −0.84);
  developmental 0.39 state; substance 0.49 (orthogonal + thin). *(Figure panel B.)*
- **G4** archetype identity persists (Arm-B weight-cosine median 0.81); dominant-label churn is higher with 5
  corners (argmax flips while weights barely move). G3⟷G4 cross-route synthesis is weak (ρ ≈ 0.07).
- Clinical logic: **stratify on the durable biology, monitor the moving symptoms.**

## M4 — the predictive value: the durable biology forecasts functioning

- **Operative K = none.** Incremental held-out ΔELPD over DSM-5 + severity + baseline (functioning):
  **archetypes +62.8**, ⊥G archetypes and 7-specifics positive, tessellation K=2/3/4 predictive but adding less,
  durable-pair-alone +2.3 (ambiguous). The continuous/archetype representation **dominates any hard tessellation**.
  *(Figure panel D.)*
- **Functioning, not severity** (severity autoregression-saturated). **Co-informative with DSM-5**
  (+both 62.6 > +DSM-5 29 > +map 17 — complements, not replaces).
- **Prognostic atlas: 2-yr functional remission 17% → 52%** across archetypes, the immunometabolic corner (A2)
  the worst-prognosis pole. *(Figure panel C.)*
- **Robust**: archetype signal survives IPW (+54.4); permutation null vanishes (−2.4); weakens dropping BP —
  **course-dependent** (BP-led).
- **A sufficient representation** (raw-vs-map benchmark, [`M4_REPRESENTATION_BENCHMARK.md`](M4_REPRESENTATION_BENCHMARK.md)):
  against the raw indicators under a matched XGBoost, the 8-factor map is **sufficient for deterioration** (AUC
  tie) and **near-sufficient for recovery** (raw +0.04 AUC) — and that residual is **within-factor compression**
  (92–97% of raw's recovery signal lives inside the 8 factors), not a missing axis; honest uncertainty adds a
  little, the map transports as well/better than raw where it is sufficient. *Structurally faithful, parsimony
  for a sliver of resolution.*

## The chain — the load-bearing achievement

The milestones are one argument about one phenotype:

> **M2: the immunometabolic corner (A2) is a real, distinct extreme → M3: that corner is durable
> (immunometabolic trait, ICC 0.91) → M4: that durable corner predicts 2-year functioning (worst remission, 17%).**

*Persists → predicts*, demonstrated end-to-end for a biology-aware phenotype, on a continuum, transdiagnostically,
with uncertainty propagated and no imputation at any step. A stratification that only recovered severity tiers
would be a re-dressed CGI-S; this one separates patients who look equally ill but are biologically opposite,
and that separation is durable and prognostically meaningful.

## M5 — treatment: bounds and defends the clinical claim (full record: [TREATMENT_OOP_FINDINGS.md](TREATMENT_OOP_FINDINGS.md))

This baseline cohort has **no randomization** (`arm` is a DSM-5 subtype), so treatment *selection* is out of
reach — it is M5b. M5's standalone contribution is to **bound and defend** the vertical's clinical claim. (1)
**The ceiling:** on observational TAU the map does **not** reliably moderate/select treatment — lithium-BP a
**well-identified, MDE-bounded null** (overlap 0.997, E-value 1.06, interaction MDE ≈ 0.19 SD → the design
could have seen an effect and didn't); antipsychotic-BP a confounded *average* effect (E-value 1.77) but **no
reliable moderation** (the two map encodings disagree on the driving axis — false-positive behavior);
clozapine-SZ non-decisive (underpowered, MDE ≈ 0.4–0.7). The map is *prognostic + descriptive, not
prescriptive*. (2) **Defends M4:** the **archetype carrier survives treatment adjustment** (the **A2
immunometabolic corner**, 7.7% attenuation, and **robust to M3 attrition IPW** — 6.4%) — the functional
forecast is not a treatment proxy; *M5 strengthens M4*. (3) **The treatment-course atlas** (the forward-looking
co-headline): the **immunometabolic corner (A2)** carries ~2× the resistance / side-effect risk of the well
pole (A4) — resistance 44%→20%, side-effects 25%→11% — beyond baseline severity + substance comorbidity +
demographics (LR p ≤ 1e-3), within-cohort (composition ≤ 5%, corner×cohort interaction NS). Honest currency:
proven as **stratification for monitoring**, but individual discrimination is modest — response (ΔAUC perm
p=0.010) and side-effects (p=0.015) clear, **resistance is the steepest gradient yet AUC-marginal (p=0.205)**,
the same continuous→group collapse as M4. The map *describes who faces a difficult course* without *selecting* a
drug — selection needs randomized data (**M5b**). Figures:
`docs/figures/treatment_oop/{moderation.png, treatment_course_atlas.png}`.

## Honest tensions (the calibration)

1. **The biology carrier is phenotype-level, not an isolated axis.** The isolated **immunometabolic** durable
   axis alone is an ambiguous EIV predictor (+2.3); the predictive signal lives in the **fuller archetype
   representation** (immunometabolic as the A2 corner), not a standalone single-axis block. "Biology predicts
   functioning" is a *phenotype-level* claim.
2. **Group-level, not individual.** ΔELPD +62.8 collapses to **ΔAUC +0.010** for binary remission — continuous
   forecasting value, not a per-patient risk calculator.
3. **Course-dependent.** Value is BP/DR-driven; weak where the future is baseline-determined (SZ).
4. **"No privileged K" is honest but operationally awkward** — the actionable object is a continuous position /
   archetype blend, not a deployable category.
5. **G4 reliable-change is measurement-precision-confounded** (the precisely-measured immunometabolic axis
   "moves" by that raw metric despite being trait); the error-corrected **G3 ICC is the clean signal**.
6. **Internal validity only**: no external cohort, no causal claim, scale-trajectory surrogates not events,
   2-year horizon; mania is data-limited (uninformative ICC).

## Calibrated claim & what's left

**Scientific validity: yes** — a real, stable, continuum (not biotype) map; biology⊥symptoms⊥severity; durable
biology; a genuine group-level incremental prognostic signal for functioning, co-informative with DSM-5, robust
to attrition/cohort/permutation. **Strong clinical utility: not demonstrated** — small individual-level gain;
treatment moderation does not hold on observational TAU (M5: lithium-BP a well-identified null, antipsychotic-BP
suggestive-unconfirmed E 1.77); internal validity only. Reporting the modest/null pieces plainly is a deliberate
correction to biotype/biomarker overclaiming.

**Remaining:** the full copula vertical (M1→M2→M3→M4→M5) is now reworked. What this baseline cohort cannot
supply: **M5b** (true treatment *selection* — randomized/trial-arm data) and external/causal validation.

## Engineering provenance

Parallel OOP engines, each wrapping the proven kernels with **no edits to the native pipelines**, on branch
`oop-strata-soft-regions`: `src/face/strata/strata_model_oop.py` (M2), `src/face/prognosis/prognosis_model_oop.py`
(M4), `src/face/temporal/temporal_model_oop.py` (M3), `src/face/treatment/treatment_model_oop.py` (M5). Built on
the certified copula M1
(`src/face/models/bayesian/measurement_model_oop.py`, `likelihood_mode="gaussian_copula"`). Validated end-to-end
(M3 V0 reproduces the M2 coords at r ≈ 0.99); uncertainty propagated; no imputation; adversarial structure-testing
(the single-Gaussian null). Outputs under `results/face/{strata_oop,prognosis_oop,temporal_oop}/`; 45 tests across
`tests/{strata,prognosis,temporal}/`.
