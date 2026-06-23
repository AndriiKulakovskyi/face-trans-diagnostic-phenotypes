# M4 prognosis on the Gaussian-copula M2 object — findings

> **Canonical M4 findings record for the copula rerun.** Built on the reworked copula M2 object (continuum:
> continuous coordinates + A=4 archetype simplex + nested K-family). Parallel OOP engine
> [`src/face/prognosis/prognosis_model_oop.py`](../src/face/prognosis/prognosis_model_oop.py) (wraps the proven
> kernels — `glm.fit_glm`, `compare.delta_elpd`, the `reference.py` design builders — with **no edits** to the
> native M4); driver [`notebooks/run_prognosis_model_oop.py`](../notebooks/run_prognosis_model_oop.py); figures
> `docs/figures/prognosis_oop/`. **Internal incremental-association validity only.** Pending PI sign-off. Updated 2026-06-22.

## What this is

M2 is now a **continuum** on the copula map; the clinically "best K" is an *outcome* question, so M4 answers it:
on the fixed copula M2 object, does a baseline coordinate / archetype / tessellation-region **predict a 2-year
outcome incrementally beyond DSM-5 + severity + the baseline outcome**, and *which encoding* — continuous
durable coords, A=4 archetypes, or the tessellation at K=2/3/4 — earns it. Predictors are read at V0 directly
from the copula hand-off (`results/face/strata_oop/`); outcomes are the native-scale follow-up scales
(`data/processed/baseline_v{0,2}.parquet`); attrition IPW is reused from the native M3 (strata-independent).
The errors-in-variables Bayesian GLM (M1 per-patient SD plugged), the LOO-ELPD comparison, and the
diagnosis+severity reference ladder are **reused verbatim**; the only addition is dynamic encoding discovery +
the K-family loop. Primary outcomes: **egf** (functioning, N=2,114 complete-case V0→V2) and **cgi_s**
(severity, N=2,345). Full 4-chain fit: R-hat ≈ 1.0, max Pareto-k ≤ 0.59, 0 divergences (27 min).

## Result 1 — the map predicts 2-year *functioning*, beyond DSM-5 + severity + baseline

Incremental held-out ΔELPD vs the R3y bar (diagnosis + severity + baseline outcome), **egf**:

| encoding | ΔELPD vs R3y | verdict |
|---|---|---|
| **+archetypesA** (A=4, full phenotype) | **+59.4 ± 11.1** | predictive |
| +archetypesB (A=4, ⊥G) | +37.9 ± 9.0 | predictive |
| +specifics8 (8 ⊥G axes, EIV, ceiling) | +37.1 ± 9.2 | predictive |
| +tess_k4 / +tess_k3 / +tess_k2 | +22.3 / +21.8 / +20.2 (± ~7) | predictive |
| +durable (3 trait axes, EIV) | +2.2 ± 3.1 | ambiguous |

For **severity (cgi_s)** the increments are small and mostly ambiguous (+archetypesB +14.0, +archetypesA +13.7
predictive; +specifics8 +11.4, tessellation +4–9, +durable −2.0 — all ambiguous): severity is
**autoregression-saturated** (the baseline CGI-S/G already carries it). This reproduces the native-map M4
headline — *the map forecasts functioning, not severity.*

## Result 2 — the operative K is **none**: the continuum/archetypes win (the answer to the K question)

The whole tessellation K-family is predictive of functioning (every K ≈ +20 ΔELPD), but the **continuous /
archetype** representation dominates it: mean ΔELPD across the two primary outcomes is **+36.6 for
+archetypesA** vs **+15.4 for the best tessellation (K=3)**. So the operative-K selector returns:

> **operative_K = none — the continuous/archetype encoding (+archetypesA) wins; the tessellation adds nothing
> beyond it.**

This is the precise, outcome-grounded resolution of "which K is clinically useful": **no hard K is the right
choice** — the actionable object is the continuous coordinates / A=4 archetype simplex, and any hard
tessellation is a lossy convenience (it throws away predictive signal the archetypes keep). A hard label, if
operationally required, is best at K=3 (BIC-best, predictive, ΔELPD ≈ +22 on functioning) — but it is strictly
dominated by the archetypes.

## Result 3 — co-informative with DSM-5 (complements, does not replace)

Head-to-head on a shared foundation (age+sex+severity+baseline), **egf**: +DSM-5 +29.0, +map +22.2, **+both
+67.4** (all predictive). The map and diagnosis each add, and together add more than either alone — the map
**complements** DSM-5, it does not replace it. For cgi_s the map adds little over DSM-5 (+map +7.6 ambiguous;
+both +47.8) — again the severity-saturation story.

## Result 4 — the archetype prognostic atlas (transdiagnostic functional gradient)

2-year functional remission (GAF ≥ 71) by dominant archetype (pooled across cohorts):

| archetype | 2-yr functional remission | mean EGF@V2 |
|---|---|---|
| A1 — low-burden pole | **60%** | 74.5 |
| A3 — psychiatric-symptom corner | 43% | 68.5 |
| A2 — severe, non-biological | 32% | 63.6 |
| A0 — biological corner (↑inflammatory/metabolic/substance) | **27%** | 61.4 |

A **27%→60%** transdiagnostic gradient; the **biological corner carries the worst functional prognosis** — the
precise value a biology-aware map adds.

## Result 4b — the 27→60 gradient is a *within-diagnosis* effect, not a cohort-composition artefact

The corners have very different cohort mixes (A1 is 74% BP; A0 carries more SZ) and the cohorts have very
different remission floors (BP 33–69%, SZ **8–23%**), so the pooled gradient could be a Simpson's-paradox
artefact. De-confounding it three ways (`notebooks/within_cohort/within_cohort_breakdown.py` →
`results/face/prognosis_oop/within_cohort/`; figure `report/figures/m4_within_cohort.png`):

1. **Within every cohort the rank holds** (A0 worst → A1 best): BP 0.33→0.69, SZ 0.08→0.23, DR 0.38→0.78.
2. **Direct standardization** to a common cohort mix barely moves the gradient (0.27→0.59 vs raw 0.27→0.60):
   **composition explains only ~6%** of the pooled A0→A1 spread — it is a genuine within-diagnosis effect.
3. **Logistic decomposition** `remission ~ corner + cohort (+interaction)`: the cohort-adjusted corner effect
   is large (**A1-vs-A0 OR ≈ 4.2**), the cohort main effect is the **dominant axis** (SZ-vs-BP **OR 0.16** —
   everyone in SZ remits far less), and the **corner×cohort interaction is NS** (p=0.36). So the corner effect
   is *relatively* homogeneous (OR≈4 in every cohort); the *absolute* spread is wider in BP only because SZ
   sits on a low baseline floor (logit non-linearity).

**Within-cohort incremental validity** (does the corner add beyond baseline functioning + severity, fit inside
each cohort; the frequentist complement to the LOCO ΔELPD): **BP yes** (LR χ²=34, p=1.6e-7), **DR yes**
(p=0.02, small n), **SZ no** (p=0.16). This is *why* the LOCO ΔELPD is BP-carried (drop-BP → +5.8 amb.): the
predictive **increment** — an absolute-scale, power-weighted quantity — concentrates in the open-course cohorts
that have room above the floor, even though the *relative* gradient is present everywhere. Reconciles M4's
"course-dependent" verdict precisely.

## Result 5 — group-level forecasting, not a large individual-binary boost (honest)

Deployable-classifier read (5-fold CV AUC, foundation vs +map): egf functional remission **0.745 → 0.756
(ΔAUC +0.011)**; cgi_s remission +0.004. As on the native map, the strong continuous ΔELPD (+59) collapses to
a small binary lift (+0.011) — the map's value is **group-level stratification + continuous functional
forecasting**, not a large individual yes/no gain.

## Result 6 — the archetype signal is robust (attrition / cohort / permutation)

Stressing the **operative winners** (the archetypes, not the now-weak `+durable`) confirms the functioning
signal is real, not an artefact (ΔELPD vs R3y for `+archetypesA` on egf):

| check | ΔELPD | reads as |
|---|---|---|
| base | +59.4 ✓ | the headline |
| IPW (attrition-reweighted) | +59.3 ✓ | not an attrition artefact |
| drop DR / drop SZ | +56.3 / +59.4 ✓ | survives removing either cohort |
| drop BP | +5.8 (amb.) | **BP carries most of it** — course-dependent (open-course BP), as in native M4 |
| permutation null | −1.7 (amb.) | the signal correctly vanishes under shuffled labels |

The clean ⊥G `+archetypesB` behaves identically (base/IPW/drop-DR/drop-SZ all ≈ +38 predictive; permutation
null ≈ 0). So the archetype functioning forecast is **robust to attrition and to dropping DR or SZ, vanishes
under permutation, and is BP-driven** (the episodic, open-course cohort). (`robustness/robustness.csv`.)

## Honest caveats

* **The durable-trio-alone signal does not survive on the copula map.** Unlike the native M4 (where
  metabolic/inflammatory durable EIV cleared its band), here **+durable is ambiguous** (egf +2.2; cgi_s
  negative). The robust, predictive object on the copula map is the **fuller archetype representation**
  (Result 6) — the biology still matters, but as a *corner of the archetype simplex* (A0, the worst-prognosis
  corner), not as the isolated 3-axis durable EIV block. An honest, map-version-specific shift, reported as such.
* **Internal incremental-association only** — not causal, not external; outcomes are re-administered scales,
  not incident events; 2-year horizon; complete-case (IPW is a sensitivity, not the headline).
* **Severity is autoregression-saturated** (small/ambiguous increments) — by design.

## Hand-off

`results/face/prognosis_oop/`:
`incremental/{incremental_comparison.csv, operative_k.json, coef_durable.csv}`,
`reference/elpd_reference.csv`, `transdiagnostic/h2h_dsm5.csv`, `endpoints/archetype_atlas.csv`,
`clinical_value/clinical_value.csv`, `robustness/robustness.csv`, and the M5 hand-off
`consolidate/{prognosis_summary.csv (carries the operative-K verdict), prognosis_patient_risk.parquet (9,013 ×
31)}`. Figure: `docs/figures/prognosis_oop/incremental_added_value.png`. Reproduce:
`PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_prognosis_model_oop.py --mode full`.

**Next (Phase 2, after PI review):** the M3 temporal-coherence OOP engine on the same copula object.
