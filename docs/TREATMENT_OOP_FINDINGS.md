# M5 treatment on the Gaussian-copula objects — findings (bounds-and-defends)

> **Canonical M5 findings record for the copula rerun.** Reworks the FACE M5 treatment causal pipeline on the
> copula map + the A=4 copula archetypes (the native `scripts/50-57` pipeline was **retired 2026-06-24**;
> the shared `src/face/treatment/` kernels are the canonical engine). Engine
> [`src/face/treatment/treatment_model_oop.py`](../src/face/treatment/treatment_model_oop.py) (wraps the proven
> kernels — `treatment.{medications,endpoints,propensity,moderation}`, `prognosis.{glm,reference,compare,clinical_value}` —
> with **no edits** to native M5); driver [`notebooks/run_treatment_model_oop.py`](../notebooks/run_treatment_model_oop.py);
> within-cohort de-confounding [`notebooks/within_cohort/treatment_response_breakdown.py`](../notebooks/within_cohort/treatment_response_breakdown.py).
> Pending PI sign-off. Updated 2026-06-24.

## What this is — the milestone, re-scoped

M5 was meant to be the program's payoff: a **treatment decision model** — use the map to choose who gets which
drug. That is **structurally unreachable here**: this baseline cohort has **no randomization** (`arm` is a
DSM-5 subtype, not a randomized assignment) and only coarse, late-found drug-class exposure. Treatment
*selection* is a counterfactual-assignment question observational treatment-as-usual (TAU) cannot answer — it
is genuinely **M5b**, and needs randomized / trial-arm data.

So M5's honest, self-contained contribution is to **bound and defend** the vertical's clinical claim. It does
three things, on the fixed copula M1/M2/M3/M4 objects (never re-scored):

1. **Bounds the ceiling** — maps the limit of what the map licenses (*prognostic + descriptive, not
   prescriptive*) via a rigorous, **MDE-bounded** moderation null.
2. **Defends M4** — shows the prognostic carrier is **not a treatment proxy** (survives adjustment for the
   drugs patients were on, incl. attrition IPW).
3. **Describes** treatment-response heterogeneity — even where it cannot *select*, the map *describes* who
   resists / responds / has side-effects.

The identification-first pipeline — **overlap gate → propensity → doubly-robust EIV moderation → E-value →
MDE** — is the reusable template for "can an *observational* map moderate treatment?". Moderation interacts
treatment with **both** the durable trio (EIV) and the A=4 archetypes (fixed `treat × arch_w`). Treatment
exposures are the map-independent harmonized drug-class flags; the predictor side is the copula prognosis_oop
frame. Full run 12.7 min; native `results/face/m5/` byte-untouched.

## Result 1 — the bounded boundary: the map does not select/moderate treatment on TAU

`P(treat | severity + DSM-5 + demographics + the 9 copula coords)`, active-comparator primary:

| question | overlap | max-SMD after IPTW | identification |
|---|---|---|---|
| lithium-BP (vs other maintenance) | 0.997 | 0.008 | **clean** (well-identified contrast) |
| antipsychotic-BP (vs other maintenance) | 0.996 | 0.079 | estimable |
| clozapine-SZ (on/off) | 0.990 | 0.068 | estimable (active-comparator channeled, SMD 0.335) |

The question is **moderation** — does the map change *who benefits*. Treat × map interaction (per-axis HDI +
held-out ΔELPD) + the ATE E-value, **plus the MDE** (the smallest effect the design resolves at 80% power —
the guard that makes a null *bounded* rather than blind):

| question · functioning | ATE [94% ETI] | E-value | interaction MDE (80%) | verdict |
|---|---|---|---|---|
| **lithium-BP** | −0.003 [−0.13, +0.12] | 1.06 | **0.19–0.22** | **well-identified, bounded null** |
| **antipsychotic-BP** | −0.231 [−0.36, −0.10] | 1.77 | 0.18–0.22 | average effect confounded; **no reliable moderation** |
| **clozapine-SZ** | +0.02 [−0.24, +0.29] | 1.16 | **0.37–0.67** | **non-decisive (underpowered)** |

- **Lithium-BP — the clean case.** Near-perfect overlap, ATE indistinguishable from 0 (E-value 1.06), and the
  per-axis interactions sit **well inside** an MDE of ≈0.19–0.22 SD: the design *could* have resolved a
  meaningful interaction and didn't. This is a **bounded null** — the map does not pick lithium responders.
  (Lithium's exposure is *lifetime* and so indication-confounded, but that only makes a null *more* credible:
  confounding-by-indication would push an estimate *away* from zero, not toward it.)
- **Antipsychotic-BP — the exception that proves the boundary.** The *average* effect excludes 0 (ATE −0.231,
  above its ATE-MDE) but is **confounding-fragile** (E-value 1.77 ≈ native's 1.79) and the **moderation** is
  noise: the held-out ΔELPD is +2.0 ± 3.7 (its band spans 0), and the two representations **disagree on which
  axis carries the interaction** — durable → metabolic [−0.347, −0.104], archetype → arch_w0 [−0.421, −0.125].
  Two encodings of the same patients pointing at different axes for the same drug is textbook false-positive
  behavior. Not a moderation signal — a hypothesis at most (see Appendix).
- **Clozapine-SZ — non-decisive, not a clean null.** Channeled in active-comparator (SMD 0.335); the on/off
  ATE is ≈0 but the MDE (≈0.4–0.7 SD, the largest in the panel — n=516, heavily selected) shows the arm is
  **underpowered**, so the null is non-decisive rather than well-identified.
- **Secondary CGI-response moderation is multiple-comparison noise** — across ~24 axis-interaction HDIs, two
  stray flags appear (lithium·cgi·arch_w1, clozapine·cgi·inflammatory), exactly the expected false-positive
  rate; none survives.

**On observational TAU the map does not reliably moderate or select treatment — an earned, MDE-bounded
boundary, not a failure to look.** This is the ceiling: the map is not a prescribing engine.

## Result 2 — M5 strengthens M4: the prognostic carrier is not a treatment proxy

The most damaging alternative to M4 ("the biological corner forecasts worse functioning") is that it is *really*
"those patients just got different drugs, and the drugs drove the outcome." We refit the copula-M4 functioning
prognosis with vs without the harmonized drug-class exposures, on the treatment-data subset, **both unweighted
and under the M3 strata-independent attrition IPW** (`w_retained_V2`):

| carrier | weighting | n | β (no treat) → β (+treat) | HDI (+treat) | survives | attenuation |
|---|---|---|---|---|---|---|
| **archetype A1** (low-burden) | none | 1324 | 0.164 → 0.156 | [+0.095, +0.217] | **yes** | 4.7% |
| **archetype A1** (low-burden) | ipw_v2 | 1068 | 0.147 → 0.141 | [+0.073, +0.211] | **yes** | 3.9% |
| durable trio | none | 1324 | — | all include 0 | no | — |

The **archetype carrier survives treatment adjustment** with HDI excluding 0 and only ~4–5% attenuation —
**robust to attrition** (it survives the IPW reweighting too). The **durable trio does not survive unweighted**
(native-parity; HDIs include 0), consistent with the copula-M4 finding that the durable-trio-alone is no longer
the robust carrier — the fuller archetype representation is (under IPW the durable *metabolic* axis does sharpen
to exclude 0, β −0.055 [−0.102, −0.007], reported honestly, but the headline carrier is the archetype).

So the map's functional forecast is **not merely unmodelled treatment** — it holds adjusting for the drug
classes the patient was on. **M5 strengthens M4.** (Honest scope: exposures are baseline/lifetime [BP] or
current [SZ/DR] at V0 — this adjusts for *baseline* drug-class exposure, not a marginal structural model over
time-varying treatment.)

## Result 3 — the treatment-course atlas: the map flags who faces a difficult course (monitoring)

The map's forward-looking payoff is **descriptive stratification of treatment course**: the baseline archetype
sorts patients into very different 2-year courses — the **biological corner (A0) consistently hardest**, the
low-burden corner (A1) easiest. This is the M4-atlas analogue, for treatment course (per-corner 2-year rate,
Wilson CIs; pooled BP+SZ — DR endpoints absent):

| corner | treatment-resistant | CGI responds | significant side-effects |
|---|---|---|---|
| **A0 biological** | **43%** [38,47] | 46% [42,51] | **23%** [19,27] |
| A2 severe·low-bio | 38% [33,43] | 50% [44,55] | 18% [14,22] |
| A3 symptom | 33% [29,38] | 45% [40,49] | 17% [14,22] |
| **A1 low-burden** | **20%** [18,23] | **59%** [56,62] | **11%** [9,13] |

**The atlas is proven, not chance — and the honest story is in the layering of the proof:**
- *Stratification (all three).* The archetype corner adds **beyond baseline severity + substance comorbidity +
  demographics** (LR p = 2e-4 / 2e-4 / 3e-3 for resistance / response / side-effects), and the gradient is
  **within-cohort, not composition** (holds inside both BP and SZ; composition share ≤ 6%; corner×cohort
  interaction NS, p 0.19–0.51). Held-out ΔELPD: **+20 / +16.5 / +10.4** (durable trio flat throughout —
  the carrier is the archetype).
- *Discrimination (the honest limit).* Under a held-out **ΔAUC permutation null**, response (p = 0.015) and
  **side-effects (p = 0.005) clear cleanly; resistance — the steepest gradient — is AUC-marginal (p = 0.185)**.
  Its rank-discrimination is small (foundation AUC ≈ 0.68, map +0.015): the resistance signal is real on
  *likelihood and stratification* but not on *individual AUC*. The **same continuous→group collapse as M4**.

So the defensible claim is **monitoring, not prediction-of-the-individual**: the map flags *which phenotype*
tends toward resistance / poor response / side-effect burden (a ~2× risk band) — useful for surveillance
intensity, never a per-patient classifier and never a prescription. And it is a **configural** signal (the
biological *corner*, not any single axis): the durable trio alone is flat and substance alone is only marginal
(p = 0.06 / 0.15 / 0.02), yet the corner adds beyond both — the phenotype, not one lab value. *This is the
forward-looking co-headline of M5, alongside the M4-defense (Result 2).*

## Honest caveats

* **Observational TAU only** — confounding by indication is the dominant threat; the per-drug ATEs are
  confounding-fragile (E-values 1.06–1.78). No causal/selection claim.
* **The boundary is bounded for lithium, non-decisive for clozapine.** Lithium's null is well-identified
  (small MDE); clozapine is genuinely underpowered (large MDE, n=516, heavily selected) — a non-decisive arm,
  not a clean null. Stated, not glossed.
* **BP lithium/antipsychotic exposures are lifetime flags** (illness-history-confounded); SZ/DR are current.
  The confounder-survival adjusts for baseline/lifetime drug-class exposure, not time-varying treatment.
* **Held-out ΔELPD on the IPTW-weighted moderation fits is best-effort** (PSIS-LOO degenerates on the
  weight-scaled likelihood), so the moderation verdict rests on the per-axis HDI + E-value + **MDE**. The
  Result-3 heterogeneity ΔELPD/ΔAUC are **unweighted**, hence valid.
* **The treatment-course atlas is stratification, not individual prediction.** It is proven on the gradient +
  beyond-confounder + within-cohort gates, but held-out discrimination is modest, and **resistance specifically
  is AUC-marginal under permutation** (p = 0.185) while response/side-effects clear (p = 0.015 / 0.005). It is
  BP/SZ-only (DR endpoints absent). Monitoring (who to watch more closely), never prescribing.
* **No treatment selection.** "Which drug for whom" needs randomized / trial-arm data (a future **M5b**),
  which this baseline cohort does not contain.

## Appendix — per-question moderation detail (the demoted point estimates)

The headline is the boundary (Result 1); the per-drug ATE point estimates are confounding-fragile and belong
here, not in the narrative. Full table at `results/face/treatment_oop/consolidate/treatment_summary.csv`
(`ate`, `e_value`, `ate_mde`, `int_mde_min/max`, `moderation_verdict`); per-axis interaction HDIs at
`moderation/moderation.csv` (`int_means`, `int_his`, `int_ses`).

## Hand-off

`results/face/treatment_oop/`: `exposures/`, `frame/`, `propensity/{propensity_*, propensity_summary.csv}`,
`moderation/moderation.csv` (+ `ate_se`, `int_ses` for the MDE), `confounder/confounder.csv` (+ `weighting ∈
{none, ipw_v2}`), `tolerability/tolerability.csv` (ΔELPD), `heterogeneity/heterogeneity.csv` (held-out ΔAUC),
`atlas/{treatment_course_atlas.csv` (per-corner rates + Wilson CIs)`, atlas_gates.csv` (specificity, composition,
corner×cohort interaction, ΔAUC permutation p)`}`, `consolidate/treatment_summary.csv` (+ `ate_mde`,
`int_mde_min/max`). Figures: `docs/figures/treatment_oop/{moderation.png, treatment_course_atlas.png}`
(the atlas also written to `report/figures/m5_treatment_atlas.png`). Reproduce:
`PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_treatment_model_oop.py --mode full`.

**Verdict: M5 bounds and defends the vertical's clinical claim** — the map is **prognostic + descriptive, not
prescriptive** on observational TAU (an MDE-bounded boundary: lithium a well-identified null, antipsychotic a
confounded average effect with no reliable moderation, clozapine underpowered), and the M4 functional forecast
**survives treatment adjustment** (archetype carrier, robust to attrition IPW) — *not a treatment artifact*.
The forward-looking co-headline is the **treatment-course atlas**: the biological corner carries ~2× the
resistance / side-effect risk of the low-burden corner (beyond severity + substance + demographics,
within-cohort), proven as **stratification for monitoring** — honestly, individual discrimination is modest and
resistance is AUC-marginal. The map **describes who faces a difficult course**, it does not **select** a drug;
selection remains the **M5b** question for randomized data.
