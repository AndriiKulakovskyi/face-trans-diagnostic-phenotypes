# M5 treatment moderation on the Gaussian-copula objects — findings

> **Canonical M5 findings record for the copula rerun.** Reworks the FACE M5 treatment causal pipeline on the
> copula map + the A=4 copula archetypes, parallel to native `scripts/50-57`. Engine
> [`src/face/treatment/treatment_model_oop.py`](../src/face/treatment/treatment_model_oop.py) (wraps the proven
> kernels — `treatment.{medications,endpoints,propensity,moderation}`, `prognosis.{glm,reference,compare}` —
> with **no edits** to native M5); driver [`notebooks/run_treatment_model_oop.py`](../notebooks/run_treatment_model_oop.py).
> Pending PI sign-off.
> Updated 2026-06-22.

## What this is

M5 asks the strongest actionable question: does the map **moderate** treatment response — does knowing a
patient's position change *who benefits*? On observational treatment-as-usual (TAU) this is confounded by
indication, so the pipeline is identification-first: **overlap gate** (propensity common support) → **doubly-
robust EIV moderation** (treat × map interaction) + **E-value** → **confounder-survival** (does the copula-M4
carrier survive treatment adjustment?) → **tolerability** (does the map predict response heterogeneity?).
Per the build decision, moderation interacts treatment with **both** the durable trio (native parity, via
errors-in-variables) **and** the A=4 archetypes (the copula-M4 carrier, via a fixed `treat × arch_w`
interaction — archetype memberships are deterministic point values, so EIV is inappropriate for them).
Treatment exposures are the map-independent harmonized drug-class flags; the predictor side is the copula
prognosis_oop frame. Full run 11 min; native `results/face/m5/` byte-untouched.

## Result 1 — the overlap gate: identifiable contrasts (reproduces native)

`P(treat | severity + DSM-5 + demographics + the 9 copula coords)`, active-comparator primary:

| question | overlap | max-SMD after IPTW | verdict |
|---|---|---|---|
| lithium-BP (vs other maintenance) | 0.997 | 0.008 | **estimable** (clean) |
| antipsychotic-BP (vs other maintenance) | 0.996 | 0.079 | estimable |
| clozapine-SZ (on/off) | 0.990 | 0.068 | estimable |

Lithium-BP has near-perfect overlap (a well-identified contrast); clozapine's active-comparator is channeled
(reserved for the resistant), so the on/off sensitivity is used — the same identification picture as native.

## Result 2 — moderation: no reliable drug-specific moderation on TAU (the earned boundary holds)

Treat × map interaction (ΔELPD held-out + per-axis HDI) + the ATE E-value, both representations:

| question · outcome | ATE [94% ETI] | E-value | moderation verdict |
|---|---|---|---|
| **lithium-BP · functioning** | −0.003 [−0.13, +0.12] | 1.06 | **no reliable moderation** (well-identified null) |
| **antipsychotic-BP · functioning** | **−0.231 [−0.36, −0.10]** | **1.77** | suggestive (interaction HDI, ΔELPD weak) |
| antipsychotic-BP · CGI response | −0.29 [−0.66, +0.07] | 1.59 | no reliable moderation |
| clozapine-SZ · functioning | +0.02 [−0.24, +0.29] | 1.16 | no reliable moderation |

(durable and archetype representations agree throughout — e.g. antipsychotic-BP functioning ATE −0.231 durable
vs −0.233 archetype, E-value 1.77/1.78.) **This replays the native verdict on the better map:** lithium-BP a
**well-identified null** (E-value 1.06 — the map does not pick lithium responders); **antipsychotic-BP a
suggestive-but-unconfirmed** signal (ATE excludes 0, **E-value 1.77 ≈ native's 1.79**, but the moderation
ΔELPD does not clear its band); clozapine non-decisive. **On observational TAU the map does not reliably
moderate treatment response** — an earned boundary, not a failure to look.

## Result 3 — confounder-survival: the M4 carrier is not a treatment proxy

Re-fitting the copula-M4 functioning prognosis on the treatment-data subset (N = 1,324), with vs without the
harmonized drug-class exposures (same sample): the **archetype carrier (the low-burden archetype A1) survives
treatment adjustment** — β 0.164 → 0.156, HDI still excludes 0, **4.7% attenuation** (compare native's
metabolic 4.4%). The durable trio does **not** survive (HDI includes 0) — consistent with the copula-M4
finding that the durable-trio-alone is no longer the robust carrier. So the map's functional forecast is **not
merely unmodelled treatment** — it holds adjusting for the drug classes the patient was on. **M5 strengthens
M4.**

## Result 4 — response heterogeneity: the archetypes predict response/resistance/tolerability

Even where the map does not *causally moderate* a specific drug, it **describes** treatment-response
heterogeneity. The A=4 archetypes predict the response endpoints beyond diagnosis + severity (held-out ΔELPD):

| endpoint | archetypes ΔELPD | durable ΔELPD |
|---|---|---|
| treatment resistance | **+20.0 ± 6.8** | −2.6 |
| CGI response | **+16.5 ± 6.2** | −2.9 |
| significant side-effects | **+10.4 ± 5.3** | +2.7 |

The archetype simplex carries a real response-stratification signal (resistance/response > 2·SE); the durable
trio alone does not — again the carrier is the fuller archetype representation.

## Honest caveats

* **ATEs are confounding-fragile** (E-values 1.06–1.78): a modest unmeasured confounder on both treatment and
  outcome would explain them away. The moderation interaction is the cleaner target but is underpowered.
* **Held-out ΔELPD is unreliable on IPTW-weighted fits** (PSIS-LOO degenerates on the weight-scaled
  likelihood), so the moderation verdict rests on the per-axis interaction HDI + the E-value, with ΔELPD
  best-effort. Reported transparently.
* **Observational TAU only.** True treatment *selection* — does assigning by the map improve outcomes —
  requires randomized / trial-arm data (a future **M5b**), which this baseline cohort does not contain.
* Archetypes moderate via a fixed interaction (point memberships), the durable trio via EIV (genuine SD) — the
  appropriate mechanism for each; both agree.

## Hand-off

`results/face/treatment_oop/`: `exposures/treatment_exposures.parquet`, `frame/analysis_frame.parquet`,
`propensity/{propensity_*.parquet, propensity_summary.csv}`, `moderation/moderation.csv`,
`confounder/confounder.csv`, `tolerability/tolerability.csv`, `consolidate/treatment_summary.csv`. Figure:
`docs/figures/treatment_oop/moderation.png`. Reproduce:
`PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_treatment_model_oop.py --mode full`.

**Verdict: the copula M5 reproduces the native earned boundary** — observational TAU does not reliably show the
map moderates treatment (lithium null, antipsychotic suggestive-unconfirmed E 1.77, clozapine non-decisive) —
while **strengthening M4** (the archetype carrier survives treatment adjustment) and showing the **archetypes
predict response heterogeneity**. Treatment *selection* remains an M5b question for randomized data.
