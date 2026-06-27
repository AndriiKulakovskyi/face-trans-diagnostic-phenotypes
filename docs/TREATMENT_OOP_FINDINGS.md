# M5 treatment — findings (bounds-and-defends)

> **Canonical M5 findings record (8-factor map, 2026-06-27; pending PI sign-off).** The FACE M5
> treatment causal pipeline runs on the **8-factor** copula map + the **A=5** copula archetypes (immunometabolic
> biology factor + substance orthogonal). Engine
> [`src/face/treatment/treatment_model_oop.py`](../src/face/treatment/treatment_model_oop.py) (wraps the
> `treatment.{medications,endpoints,propensity,moderation}` + `prognosis.{glm,reference,compare}` kernels);
> driver [`notebooks/run_treatment_model_oop.py`](../notebooks/run_treatment_model_oop.py). Inputs: the
> 8-factor prognosis_oop frame (coords + A=5 archetypes + covariates + outcomes + IPW) + the
> map-independent harmonized drug-class exposures.

## What this is — the milestone

M5 asks the strongest "actionable" question: does the map **moderate / select** treatment response on
observational treatment-as-usual? It is a **bounds-and-defends** milestone: an MDE-guarded ceiling on
moderation + a confound-survival defense of the M4 forecast + a descriptive treatment-course atlas.

## Result 1 — the bounded boundary: the map does not select/moderate treatment on TAU

`P(treat | severity + DSM-5 + demographics + the 8 copula coords)`, active-comparator primary:

| question | overlap (frac-in-support) | identification |
|---|---|---|
| lithium-BP (vs other maintenance) | 0.998 | **clean** (well-identified contrast) |
| antipsychotic-BP (vs other maintenance) | 0.996 | estimable |
| clozapine-SZ | 0.980 active / 0.988 on-off | estimable (active-comparator channeled → on/off) |

The question is **moderation** — treat × map interaction (per-axis HDI + held-out ΔELPD) + the ATE E-value,
**plus the MDE** (smallest effect the design resolves at 80% power — the guard that makes a null *bounded*):

| question · functioning | ATE (durable / arch) | E-value | interaction MDE | verdict |
|---|---|---|---|---|
| **lithium-BP** | −0.030 / −0.055 | 1.20 / 1.28 | **0.20–0.21** | **well-identified, bounded null** |
| **antipsychotic-BP** | −0.242 / −0.238 | **1.80 / 1.79** | 0.18–0.20 | confounded average effect; **suggestive moderation, unconfirmed** |
| **clozapine-SZ** | +0.033 / +0.020 | 1.21 / 1.16 | **0.38–0.45** | **non-decisive (underpowered)** |

- **Lithium-BP — the clean case.** Near-perfect overlap, ATE indistinguishable from 0 (E 1.20–1.28), both
  representations `moderation_any_axis = False`, and the interactions sit **well inside** an MDE of ≈0.20 SD:
  the design *could* have resolved a meaningful interaction and didn't. A **bounded null** — the map does not
  pick lithium responders. (Lifetime/indication-confounded exposure only makes a null *more* credible.)
- **Antipsychotic-BP — the exception that proves the boundary.** The *average* effect excludes 0 (ATE ≈ −0.24)
  but is **confounding-fragile** (E-value **1.80**), and the **moderation is suggestive but
  unconfirmed**: `any_axis` flags on functioning, but the held-out ΔELPD is weak (durable +3.4 ± 3.4, archetype
  +1.8 ± 4.3 — both bands span 0), so the verdict is *"suggestive (HDI, ΔELPD weak)"* in both representations.
  A hypothesis at most, not a moderation signal.
- **Clozapine-SZ — non-decisive, not a clean null.** Channeled in active-comparator (overlap 0.980, caution) →
  on/off; the ATE is ≈0 but the MDE (≈0.38–0.45 SD on functioning, ≈0.9–1.1 on CGI, n heavily selected) shows
  the arm is **underpowered**, so the null is non-decisive rather than well-identified.

**On observational TAU the map does not reliably moderate or select treatment — an earned, MDE-bounded
boundary, not a failure to look.** This is the ceiling: the map is prognostic + descriptive, not a prescribing
engine.

## Result 2 — M5 strengthens M4: the prognostic carrier is not a treatment proxy

The most damaging alternative to M4 ("the immunometabolic corner forecasts worse functioning") is that those
patients merely got different drugs. We refit the M4 functioning prognosis with vs without the
harmonized drug-class exposures, on the treatment-data subset, **both unweighted and under the M3
strata-independent attrition IPW**:

| carrier | β (no treat) → β (+treat) | attenuation | IPW attenuation | survives |
|---|---|---|---|---|
| **archetype A2** (immunometabolic corner) | −0.210 → −0.194 | **7.7%** | 6.4% | **yes** (IPW-robust) |
| archetype A0 / A3 | −0.141 → −0.133 / −0.154 → −0.144 | 5.3% / 6.6% | 4.3% / 5.7% | **yes** |
| **durable immunometabolic axis** | −0.049 → −0.046 | 6.4% | 4.1% | **yes** |

**The carrier survives treatment adjustment** with only ~4–8% attenuation, **robust to attrition IPW** — and it
survives in **both** representations: the archetype corners (led by **A2, the immunometabolic
biology corner — the M4 worst-prognosis pole**) *and* the durable immunometabolic axis itself. The
immunometabolic axis is durable enough to survive on its own. Cognition and archetype A1 do not
survive — reported honestly. So the map's functional forecast is **not merely unmodelled treatment**. **M5
strengthens M4.** (Honest scope: adjusts for baseline/lifetime drug-class exposure, not time-varying treatment.)

## Result 3 — the treatment-course atlas: the map flags who faces a difficult course (monitoring)

The baseline archetype sorts patients into very different 2-year treatment courses — the **immunometabolic
corner (A2) consistently hardest**, the well pole (A4) easiest (per-corner 2-year rate, Wilson CIs; pooled
BP+SZ — DR endpoints absent):

| corner | treatment-resistant | CGI responds | significant side-effects |
|---|---|---|---|
| **A2 immunometabolic** | **44%** [39,50] | 48% [42,53] | **25%** [20,30] |
| A1 severe·clean-bio | 36% [31,41] | 50% [45,55] | 18% [14,22] |
| A3 trauma | 35% [31,40] | 44% [39,49] | 15% [12,19] |
| A0 activation | 30% [26,35] | 48% [43,53] | 18% [14,22] |
| **A4 well** | **20%** [17,23] | **61%** [57,64] | **11%** [8,13] |

**The atlas is proven, not chance — with the honest layering of the proof:**
- *Stratification (all three).* The corner adds **beyond baseline severity + substance comorbidity +
  demographics** (LR p = 2e-4 / <1e-4 / 1.3e-3 for resistance / response / side-effects), and the gradient is
  **within-cohort, not composition** (composition share ≤ 5%; corner×cohort interaction NS, p 0.31–0.48).
- *Discrimination (the honest limit).* Under a held-out **ΔAUC permutation null**, **response (ΔAUC +0.034,
  p = 0.010) and side-effects (+0.042, p = 0.015) clear cleanly; resistance — the steepest gradient — is
  AUC-marginal (+0.012, p = 0.205)**. Real on *likelihood + stratification*, not on *individual AUC* — the
  **same continuous→group collapse as M4**.

So the defensible claim is **monitoring, not prediction-of-the-individual**: the map flags *which phenotype*
(the immunometabolic corner) tends toward resistance / poor response / side-effect burden (a ~2× risk band) —
useful for surveillance intensity, never a per-patient classifier and never a prescription. The durable axes
alone describe almost no response heterogeneity (ΔAUC ≈ 0); the **archetype configuration** does (ΔAUC
+0.012/+0.034/+0.042). *This is the forward-looking co-headline of M5, alongside the M4-defense (Result 2).*

## Honest caveats

* **Observational TAU only** — confounding by indication is the dominant threat; the per-drug ATEs are
  confounding-fragile (E-values 1.16–1.80). No causal/selection claim.
* **Bounded for lithium, non-decisive for clozapine.** Lithium's null is well-identified (MDE ≈ 0.20);
  clozapine is genuinely underpowered (large MDE, heavily selected) — non-decisive, not a clean null.
* **BP lithium/antipsychotic exposures are lifetime flags** (illness-history-confounded); SZ/DR are current.
  The confounder-survival adjusts for baseline/lifetime drug-class exposure, not time-varying treatment.
* **Held-out ΔELPD on the IPTW-weighted moderation fits is best-effort** (PSIS-LOO degenerates on the
  weight-scaled likelihood) → the moderation verdict rests on the per-axis HDI + E-value + **MDE**. The
  Result-3 heterogeneity ΔAUC are **unweighted**, hence valid.
* **The atlas is stratification, not individual prediction** — proven on gradient + beyond-confounder +
  within-cohort, but discrimination is modest and **resistance is AUC-marginal under permutation** (p = 0.205)
  while response/side-effects clear. BP/SZ-only. Monitoring, never prescribing.
* **No treatment selection.** "Which drug for whom" needs randomized / trial-arm data (a future **M5b**).

## Hand-off

`results/face/treatment_oop/`: `exposures/`, `frame/`, `propensity/{propensity_*, propensity_summary.csv}`,
`moderation/moderation.csv` (+ `ate_se`, `int_ses` for the MDE), `confounder/confounder.csv`,
`tolerability/`, `heterogeneity/heterogeneity.csv` (held-out ΔAUC), `atlas/{treatment_course_atlas.csv,
atlas_gates.csv}`, `consolidate/treatment_summary.csv` (+ `ate_mde`, `int_mde_min/max`, `moderation_verdict`).
Figures: `docs/figures/treatment_oop/{moderation.png, treatment_course_atlas.png}`. Reproduce:
`PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 python notebooks/run_treatment_model_oop.py --mode full`.

**Verdict: M5 bounds and defends the vertical's clinical claim on the 8-factor map** — the map is **prognostic +
descriptive, not prescriptive** on observational TAU (MDE-bounded: lithium a well-identified null, antipsychotic
a confounded average effect with suggestive-unconfirmed moderation, clozapine underpowered), and the M4
functional forecast **survives treatment adjustment** (the immunometabolic archetype corner *and* the
immunometabolic durable axis, IPW-robust) — *not a treatment artifact*. The forward-looking co-headline is the
**treatment-course atlas**: the immunometabolic corner carries ~2× the resistance / side-effect risk of the well
pole (beyond severity + substance + demographics, within-cohort), proven as **stratification for monitoring** —
individual discrimination modest, resistance AUC-marginal. The map **describes who faces a difficult course**, it
does not **select** a drug; selection remains the **M5b** question for randomized data.
