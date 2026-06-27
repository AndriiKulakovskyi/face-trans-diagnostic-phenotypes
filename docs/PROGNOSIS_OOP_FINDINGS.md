# M4 prognosis on the Gaussian-copula M2 object — findings

> **Canonical M4 findings record (8-factor map, 2026-06-27; pending PI sign-off).** Built on the **8-factor**
> copula M1/M2/M3 objects (continuum: continuous coordinates + **A = 5** archetype simplex + nested K-family;
> immunometabolic merge + substance orthogonal). Engine
> [`src/face/prognosis/prognosis_model_oop.py`](../src/face/prognosis/prognosis_model_oop.py) (wraps the proven
> `glm.fit_glm` / `compare.delta_elpd` / `reference.py` kernels); driver
> [`notebooks/run_prognosis_model_oop.py`](../notebooks/run_prognosis_model_oop.py). **Internal
> incremental-association validity only.** Inputs: M2 strata (5 archetype weights), M3 panel + draws + IPW.

## What this is

On the fixed M1/M2/M3 objects, an errors-in-variables Bayesian GLM (propagating M1 uncertainty via the panel
draws, attrition-corrected by the M3 IPW) tests whether a baseline coordinate/stratum **forecasts a 2-year
outcome incrementally beyond DSM-5 + severity + the baseline value of that same outcome**. The M4 story
**replays — and the headline forecast is slightly stronger** on the 8-factor map.

## Result 1 — the map predicts 2-year *functioning*, beyond DSM-5 + severity + baseline

Incremental held-out ΔELPD vs the R3y bar (diagnosis + severity + baseline outcome), **egf**:

| encoding | ΔELPD vs R3y | verdict |
|---|---|---|
| **+archetypesA** (A=5, full phenotype) | **+62.8 ± 11.2** | predictive |
| +specifics8 (7 ⊥G axes, EIV) | +38.1 ± 9.2 | predictive |
| +archetypesB (A=5, ⊥G) | +33.5 ± 8.7 | predictive |
| +tess_k3 / +tess_k4 / +tess_k2 | +19.6 / +16.6 / +15.9 (± ~6) | predictive |
| +durable (cognition+immunometabolic, EIV) | +2.3 ± 2.9 | **ambiguous** |

For **severity (cgi_s)** the increments are small/ambiguous (+archetypesB +14.0, +archetypesA +12.0; +durable
−0.7): severity is **autoregression-saturated** (baseline CGI-S/G carries it). *The map forecasts functioning,
not severity* — the native + copula M4 headline, replayed. The archetype forecast is **+62.8 (was +59 on the
9-factor map)** — the 5-corner representation is, if anything, a slightly better functional predictor.

## Result 2 — the operative K is **none**: the continuum/archetypes win (the answer to the K question)

The whole tessellation K-family is predictive (every K ≈ +16–20 ΔELPD), but the **archetype** representation
dominates it: mean ΔELPD across the two outcomes is **+37.4 for +archetypesA** vs **+12.6 for the best
tessellation (K=3)**. The operative-K selector returns:

> **operative_K = none — the continuous/archetype encoding (+archetypesA) wins; the tessellation adds nothing
> beyond it.**

The outcome-grounded resolution of "which K is clinically useful": **no hard K** — the actionable object is the
continuous coordinates / **A = 5** archetype simplex; any hard tessellation is a lossy convenience (it discards
predictive signal the archetypes keep). If a hard label is operationally required, K = 3 is best (BIC-near-best,
ΔELPD ≈ +20 on functioning) — but it is strictly dominated by the archetypes.

## Result 3 — co-informative with DSM-5 (complements, does not replace)

Head-to-head on a shared foundation (age+sex+severity+baseline), **egf**: +DSM-5 **+29.0**, +map **+17.3**,
**+both +62.6** (all predictive). Each adds, and together they add more than either alone — the map
**complements** DSM-5, it does not replace it (on top of DSM-5 the map still adds +33.6; on top of the map
DSM-5 adds +45). For cgi_s the map adds little over DSM-5 (+map +8.3 ambiguous; +both +47.6) — the
severity-saturation story again.

## Result 4 — the archetype prognostic atlas (transdiagnostic functional gradient)

2-year functional remission (GAF ≥ 71) by dominant archetype (pooled over patients with a V2 outcome):

| archetype | 2-yr functional remission |
|---|---|
| A4 — low-burden / well pole | **52%** |
| A0 — activation / sleep | 41% |
| A3 — trauma / suicidality | 28% |
| A1 — severe, clean-biology | 22% |
| **A2 — immunometabolic (biology) corner** | **17%** |

A **17% → 52%** transdiagnostic gradient; **the immunometabolic biology corner (A2) carries the worst
functional prognosis** — the precise value a biology-aware map adds. (On the 9-factor A=4 map this was 27→60%;
the merged-biology corner is now an even sharper low-prognosis pole.)

## Result 4b — the gradient is *within-diagnosis*, not a cohort-composition artefact

The corners have different cohort mixes and the cohorts have very different remission floors (BP/DR open-course,
SZ low-floor), so a pooled gradient must be checked within cohort. De-confounding it three ways
(`notebooks/within_cohort/within_cohort_breakdown.py` → `results/face/prognosis_oop/within_cohort/`):

1. **Within every cohort the rank holds** (immunometabolic corner → well pole): **BP 27% → 73%, DR 31% → 72%,
   SZ 9% → 25%.**
2. **Direct standardization** to a common cohort mix barely moves the gradient (0.23 → 0.62 vs raw 0.17 → 0.52):
   **composition explains only ~4%** of the pooled spread — a genuine within-diagnosis effect (stronger than the
   9-factor map's 6%).
3. **Logistic decomposition** `remission ~ corner + cohort (+interaction)`: the cohort-adjusted best-vs-worst
   corner effect is large (**OR ≈ 6.3**), cohort is the dominant axis (SZ-vs-BP **OR 0.15**), and the
   **corner×cohort interaction is NS** (p = 0.79) — the relative corner effect is cohort-homogeneous; the
   absolute spread is wider in BP only because SZ sits on a low floor.

**Within-cohort incremental validity** (does the corner add beyond baseline functioning + severity, inside each
cohort?): **BP yes** (LR χ² = 59.7, p = 3e-12), **SZ no** (p = 0.25), **DR no** (p = 0.22, underpowered). This
is *why* the LOCO ΔELPD is BP-carried — the predictive increment concentrates in the open-course cohorts that
have room above the floor, even though the relative gradient is present everywhere. Figure
`docs/figures/prognosis_oop/within_cohort_gradient.png`.

## Result 5 — group-level forecasting, not a large individual-binary boost (honest)

Deployable-classifier read (5-fold CV AUC, foundation vs +map): egf functional remission **0.745 → 0.755
(ΔAUC +0.010)**; cgi_s +0.004. As on the native/copula map, the strong continuous ΔELPD (+62.8) collapses to a
small binary lift — the map's value is **group-level stratification + continuous functional forecasting**, not a
large individual yes/no gain.

## Result 6 — the archetype signal is robust (attrition / cohort / permutation)

Stressing the operative winner (`+archetypesA` on egf, ΔELPD vs R3y):

| check | ΔELPD | reads as |
|---|---|---|
| base | +62.8 ✓ | the headline |
| IPW (attrition-reweighted) | **+54.4 ✓** | not an attrition artefact (13% attenuation) |
| drop DR / drop SZ | +56.4 / +61.8 ✓ | survives removing either cohort |
| drop BP | +7.1 (amb.) | **BP carries most of it** — course-dependent (open-course BP) |
| permutation null | −2.4 (amb.) | the signal correctly vanishes under shuffled labels |

The clean ⊥G `+archetypesB` behaves identically (base +33.5 / IPW +27.7 predictive; permutation −2.1 null). So
the archetype functioning forecast is **real (permutation), IPW-robust, and BP-driven** (the episodic,
open-course cohort) — the copula M4 robustness story, replayed.

## Result 7 — the map is a *sufficient representation* (raw-vs-map benchmark) — re-run on the 8-factor map

One fixed regularised XGBoost, identical CV folds, three representations (REF = DSM-5+severity+baseline GAF;
REF+map = 8 coords + uncertainty + A=5 archetypes; REF+raw = raw indicators), predicting recovery
(impaired→GAF≥71) and deterioration (GAF drop ≥10), V1+V2 pooled. The 8-factor map **replays the 9-factor
sufficiency verdict** (`results/face/m4_repbench/`; xgboost run under `OMP_NUM_THREADS=1` to dodge the macOS
libomp segfault):

- **Deterioration: map = raw** (AUC raw−map +0.009 V1 / +0.005 V2 — **tie → sufficient**).
- **Recovery: raw edges the map by ΔAUC ≈ +0.04** (V1 +0.040, V2 +0.039 — "raw-adds", replicated across
  horizons), **but the gap is within-factor compression, not a missing axis**: TreeSHAP puts **92% (V1) / 97%
  (V2)** of raw's recovery-predictive mass *inside* the 8 modelled factors (top drivers CRP/BMI →
  immunometabolic, CVLT/WAIS → cognition, Fagerström → substance, CSM → sleep); the only off-map residual is the
  depression/anxiety window items (STAI/MADRS/QIDS).

**Calibrated claim (unchanged from the 9-factor map):** the 8-factor copula map is a **sufficient** summary for
deterioration and a **near-sufficient, structurally faithful** summary for recovery whose ≈0.04-AUC residual is
item-level compression (≥92% inside its own factors) — parsimony + interpretability for a sliver of resolution.
Methods: [`M4_REPRESENTATION_BENCHMARK.md`](M4_REPRESENTATION_BENCHMARK.md). *(NB the SHAP `home_factor` labels
still carry the pre-merge metabolic/inflammatory names — cosmetic; both fold into immunometabolic and the
within-factor share is unaffected.)*

## Honest caveats

* **The durable-pair-alone signal is ambiguous** (egf +2.3; cgi_s −0.7) — as on the 9-factor copula map. The
  immunometabolic *axis* has a credible adverse direction (EGF coef −0.053 [−0.088, −0.019], p-dir 0.003 —
  higher immunometabolic → worse future functioning), but the **2-axis durable EIV block alone does not beat the
  R3y bar by ELPD**. The robust predictive object is the **fuller A = 5 archetype representation**, where the
  biology lives as the worst-prognosis corner (A2). An honest, map-version-specific carrier.
* **Internal incremental-association only** — not causal, not external; outcomes are re-administered scales, not
  incident events; 2-year horizon; complete-case (IPW is a sensitivity, not the headline).
* **Severity is autoregression-saturated** — by design.

## Hand-off

`results/face/prognosis_oop/`: `incremental/{incremental_comparison.csv, operative_k.json, coef_durable.csv}`,
`reference/`, `transdiagnostic/h2h_dsm5.csv`, `endpoints/archetype_atlas.csv`,
`clinical_value/clinical_value.csv`, `robustness/robustness.csv`, and the M5 hand-off
`consolidate/{prognosis_summary.csv (carries the operative-K verdict), prognosis_patient_risk.parquet}`. Figure:
`docs/figures/prognosis_oop/incremental_added_value.png`. Reproduce:
`PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_prognosis_model_oop.py --mode full`.

**Verdict: the 8-factor map predicts 2-year functioning** — the A = 5 archetypes add ΔELPD +62.8 beyond
DSM-5 + severity + baseline (IPW-robust, permutation-null, co-informative with DSM-5, course-dependent /
BP-led), with a 17→52% transdiagnostic functional-remission gradient (within-diagnosis: composition only ~4%)
whose worst pole is the immunometabolic biology corner. Operative K = none. Not severity. The map is a
sufficient representation for deterioration and near-sufficient (≥92% within-factor) for recovery. Replays the
copula M4 on the better map, in full.
