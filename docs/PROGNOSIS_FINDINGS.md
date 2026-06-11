# M4 — Prognosis: findings

> **Paper-facing findings for Milestone 4 (read first).** Whether a baseline coordinate or stratum on
> the transdiagnostic map **predicts a future clinical outcome, incrementally beyond diagnosis +
> severity** — built on the durable axes M3 certified (cognition, metabolic, inflammatory). Methods of
> record: [`PROGNOSIS_MODEL.md`](PROGNOSIS_MODEL.md); clinician-facing atlas:
> [`PROGNOSIS_ATLAS.md`](PROGNOSIS_ATLAS.md); per-stage detail: [`PROGNOSIS_RESULTS.md`](PROGNOSIS_RESULTS.md)
> + `reports/40–48`. Sibling of [`M1_FINDINGS.md`](M1_FINDINGS.md), [`STRATA_FINDINGS.md`](STRATA_FINDINGS.md),
> [`TEMPORAL_FINDINGS.md`](TEMPORAL_FINDINGS.md).
> *Scope: internal incremental association ("predicts" ≠ "causes"), scale trajectories not events.
> Status: COMPLETE, pending PI sign-off. 2026-06-11.*

---

## Headline

**The transdiagnostic biology map carries a real but *modest*, *group-level* prognostic signal about
2-year functional trajectory — beyond diagnosis and current severity — complementary to DSM-5, not a
replacement, and not an individual risk tool.** A baseline position on the durable ⊥G axes (metabolic,
inflammatory) predicts future functioning over and above the clinician's reference, surviving attrition,
measurement-reliability and permutation checks; the 8 archetypes sort patients into groups whose 2-year
functional-remission rate ranges from **14% to 60%**, transdiagnostically — *though that raw spread partly
reflects baseline-severity differences between archetypes, not the incremental signal alone* (the
severity-adjusted increment is the small ΔELPD/ΔAUC below, §4). The signal is **course-dependent** — large in the
episodic courses (bipolar, depression) where the future is not already fixed by baseline severity, and
absent in the more chronic schizophrenia presentations (foundation saturation). For **severity** (CGI-S)
the map adds little: severity is largely baseline-determined. *"Persists" became "predicts" — for
functioning, in the patients whose course is open.*

It is not a slam-dunk individual-risk tool, and the honest part is the point (§4, §6): the map's
*incremental individual discrimination* is small (ΔAUC +0.017 for functional remission); its value is
as a **prognostic stratification** (group level) and in **continuous functional forecasting**, not a
large boost to binary individual prediction over a strong severity baseline.

---

## 1. The question, and the answer (the four gates)

M4 asks one question — *does the map predict, incrementally beyond diagnosis + severity?* — through four
gates mirroring M2's Q-battery:

| gate | question | answer |
|------|----------|--------|
| **Q1** incremental validity | does the map add held-out predictive signal? | **yes for functioning** (ΔELPD +46 archetypes; ΔAUC +0.017 remission), **no for severity** |
| **Q2** beyond severity | does it survive adjustment for *error-corrected* severity? | **yes** — metabolic β −0.062 survives both CGI-S and the G coordinate |
| **Q3** transdiagnostic / vs DSM-5 | does it beat / cut across the 7 DSM-5 subtypes? | **co-informative** (complements DSM-5) + **course-dependent** (BP/DR, not SZ) |
| **Q4** robust / not artefact | survive attrition, reliability, chance? | **yes** — IPW, reliability, permutation (p=0.001); honestly weakens dropping BP |

---

## 2. The map adds for functioning, not severity (Q1)

On the M3 panel (baseline coordinates scored on the fixed M1/M2 model, uncertainty propagated), added
on top of the **R3y bar** (DSM-5 arm + severity + the baseline value of the outcome), per primary
outcome:

- **Functioning (EGF, N=2,114):** the 8 archetypes add held-out **ΔELPD +46** (Arm-B, G-residualized;
  +58 Arm-A) and the 4-region tessellation **+47** — clearly predictive. The continuous durable trio
  adds little to held-out ELPD (+7, ambiguous) but its coefficients are individually informative (§3).
- **Severity (CGI-S, N=2,345):** the map adds essentially nothing (durable +0; archetypes +13,
  ambiguous) — severity is autoregression-saturated (today's CGI-S + G already determine it).

The reason the strata beat the linear coordinates: the archetypes capture **multivariate corner
structure** (which combination of biology a patient sits in) that three linear terms miss.

---

## 3. Beyond severity — the durable biology is the clean signal (Q2)

The durable ⊥G axes, entered as errors-in-variables (the M1 per-patient SD propagated), on future
functioning:

- **metabolic β = −0.062 [−0.103, −0.022]** and **inflammatory β = −0.060 [−0.112, −0.011]** — both
  exclude 0; higher metabolic/inflammatory burden → worse future functioning.
- **metabolic survives the *error-corrected* G severity** (β −0.055 [−0.095, −0.015]) as well as the
  manifest CGI-S — the load-bearing Q2 result: it is not baseline severity measured noisily.
- cognition does not reach credibility (prior-dominated for the untested, and the weakest signal).

This is the biology⊥G bet from M1 cashed out prognostically: the metabolic/inflammatory corner a
patient *keeps* (M3 trait) forecasts the functional state they *move toward* (M3 state).

---

## 4. Versus DSM-5 — complementary, and course-dependent (Q3)

On a shared foundation (nuisance + baseline outcome + severity), four nested models (D / +DSM-5 /
+map / +both) read the dominance:

- **Co-informative, not "map dominates":** the map adds beyond the 7 DSM-5 subtypes (EGF B−A **+47**)
  **and** DSM-5 adds beyond the map (B−C **+40**). Each carries prognostic information the other lacks —
  diagnosis the categorical illness-type, the ⊥G map the dimensional biology profile. They are
  **complementary lenses**, consistent with the project's four-layer design (diagnosis stays metadata).
- **Course-dependent generalization:** the map's incremental value is large within **BP** (EGF ΔELPD
  +43) and **DR** (OLS ΔR² +0.07), **null within SZ** (−1.8). The probe (`reports/47`) showed this is
  **foundation saturation** — SZ functioning is more baseline-locked (foundation R² 0.26 vs BP's 0.17),
  leaving little residual variance — **not** the map failing in SZ (SZ's outcome variance, archetype
  spread and coordinate quality match BP). The map predicts where the future is *open*.
- Raw (no autoregressive baseline), DSM-5 alone out-classifies the ⊥G map alone — expected, since the
  map deliberately removes the severity axis that drives the categorical functioning gaps.

---

## 5. The prognostic atlas + clinical value (the "so what")

**The archetypes are clinically distinct prognostic groups** (`PROGNOSIS_ATLAS.md`): 2-year functional
remission ranges **14% (suicidality archetype) → 60% (low-burden)**, every archetype transdiagnostic
(all three cohorts present). On the crude separation metric the archetypes separate the **dynamic
transitions** (remission / deterioration / relapse) better than DSM-5, while DSM-5 separates the
**severity-level / sustained** outcomes better — the co-informative split, in the clinic's own units.

**Clinical value, honestly** (cross-validated; `reports/46`): the clinician's reference already predicts
these endpoints well (AUC 0.73–0.87). Adding the map gives a **small but reliable** discrimination gain
for **functional remission (+0.017 [+0.009, +0.026])** and marginal for sustained impairment (+0.008);
nothing for deterioration/relapse (decision curves overlap). The map's value is **stratification +
continuous forecasting**, not a large individual-binary boost — the gap between "+46 ΔELPD" (continuous)
and "+0.017 ΔAUC" (binary) is the cost of collapsing the functional continuum to one threshold.

---

## 6. Discussion — what we predicted, what we observed

| hypothesis (from the M3 hand-off) | observation | verdict |
|---|---|---|
| durable biology predicts future functioning beyond dx+severity | metabolic/inflammatory β exclude 0; archetypes ΔELPD +46 | confirmed (functioning) |
| …and beyond *error-corrected* severity | metabolic survives the G coordinate | confirmed |
| the map beats DSM-5 | **co-informative** — both add; map owns dynamics, DSM-5 owns severity | refined |
| transdiagnostic (holds across cohorts) | **course-dependent** — BP/DR yes, SZ saturated | refined (honest) |
| severity is forecastable from biology | autoregression-saturated; map adds ~0 | refuted (as expected) |
| individual risk discrimination jumps | ΔAUC +0.017 — small but reliable | refined (stratification > individual) |

Three things the analysis taught us: (i) **a stratification's value is group-level** — the same signal
that looked modest as an individual ΔAUC is a 14%→60% spread across groups; (ii) **the conservative
incremental-beyond-baseline frame is honest but low-ceiling** — most of the predictable variance is
autoregressive, so a real effect looks small; (iii) **the SZ null is informative, not a failure** — it
localizes the map's value to the courses where prognosis is genuinely open.

What would have falsified the story (and didn't): the durable β collapsing under the error-corrected
severity (Q2), under IPW, or under reliability restriction; the durable gain not exceeding a permutation
null. It survived each.

---

## 7. Honesty and limits

- **Scale trajectories, not events.** No hospitalization/relapse register exists; the endpoints are
  state transitions defined from repeated GAF/CGI-S (a relapse *surrogate*, not a recorded relapse). A
  substantive de-scope from the §1.7 "relapse/hospitalization/attempt" wording — flagged for the PI.
- **Internal association, not causal or externally validated.** Uncertainty-propagated and
  held-out-validated within FACE, but not a deployable decision rule — "predicts ≠ causes."
- **Course-dependent / BP-concentrated.** Null in SZ (saturation); DR statistically thin (N≈105 —
  ELPD untestable, the OLS ΔR² agrees with BP).
- **Small individual increment** (ΔAUC +0.017) — the map is a stratification/forecasting aid, not an
  individual-risk calculator that supersedes severity.
- **2-year horizon, 3 visits**; the rare archetypes (inflammatory N=174, suicidality N=137) carry wide
  uncertainty.

---

## 8. What this hands to M5

`results/face/m4/prognosis_patient_risk.parquet` — per modelled patient: archetype + cross-validated
functional-remission and -deterioration risk. `results/face/m4/prognosis_summary.csv` — the per-outcome
verdict. The decision-relevant read for M5 (treatment): **stratify and forecast on the durable biology
in the open-course patients; the next question is whether the strata *moderate treatment response*
(stratum × treatment interaction)** — the point at which a phenotype changes management.

---

## Figures

- [`45_atlas_rates.png`](figures/45_atlas_rates.png) — the per-archetype 2-year endpoint atlas (green→red).
- [`45_atlas_trajectories.png`](figures/45_atlas_trajectories.png) — archetypes keep rank as the cohort improves; suicidality is the non-recoverer.
- [`43_added_value.png`](figures/43_added_value.png) — ΔELPD of each representation + the durable-axis forest.
- [`44_dominance.png`](figures/44_dominance.png) — map vs DSM-5 co-informative dominance.
- [`46_auc.png`](figures/46_auc.png) · [`46_decision_curve.png`](figures/46_decision_curve.png) — clinical value.
- [`47_robustness.png`](figures/47_robustness.png) — survives IPW/reliability/permutation; course-dependent.
