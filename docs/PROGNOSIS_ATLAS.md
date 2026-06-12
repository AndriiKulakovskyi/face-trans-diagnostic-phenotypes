# M4 — The prognostic atlas (clinician-facing: the "so what")

> **Read this first if you are a clinician.** What the transdiagnostic map *adds to clinical
> decisions*: given a patient's baseline biological-dimensional profile, what can we say about their
> **2-year prognosis** that DSM-5 diagnosis + current severity do **not** already tell us — and how
> might that change monitoring and intervention intensity? Methods of record:
> [`PROGNOSIS_MODEL.md`](PROGNOSIS_MODEL.md) (pending); technical findings:
> [`PROGNOSIS_FINDINGS.md`](PROGNOSIS_FINDINGS.md) (pending) + `reports/40–46`. Sibling of
> [`STRATA_ATLAS.md`](STRATA_ATLAS.md). *Internal association, not a validated decision rule —
> hypotheses for prospective clinical validation. Status: M4.5 done; clinical-value metrics (§5) pending stage 46.*

---

## 1. The clinical question — and why it matters

A clinician at baseline already knows the **diagnosis** (BP / SZ / DR, and the DSM-5 subtype) and the
patient's **current severity**. The honest prognostic question is therefore *incremental*:

> *Beyond what diagnosis and today's severity already imply, does a patient's position on the
> transdiagnostic biology map tell us anything **new and useful** about where they will be in 2 years?*

M4 answers yes — **for functional trajectory, and in the patients whose course is not already fixed.**
The value is not a single number; it is that the map sorts patients into groups with **markedly
different 2-year outcomes**, in a way that cuts across diagnosis.

---

## 2. The headline a clinician can act on

Across the 8 transdiagnostic archetypes, the **2-year functional-remission rate ranges from 14% to
60%** — a four-fold difference in the chance of returning to good functioning. Crucially, **every
archetype contains patients from all three diagnostic cohorts** (BP, SZ, DR): these are *not* the
diagnostic groups relabelled. Two patients with the *same diagnosis and similar current severity* can
sit in a 60%-remission archetype or a 15%-remission one — and the map is what distinguishes them.

The map is most informative exactly where it is most needed: in the **episodic courses (bipolar,
depression) where the future is genuinely uncertain**, rather than the more chronic, baseline-locked
schizophrenia presentations where today's state already determines much of the trajectory (§4).

---

## 3. The prognostic atlas — per archetype

2-year outcome rates (functional = EGF/GAF; symptomatic = CGI-S), ordered best → worst functional
prognosis. *Functional remission* = GAF ≥ 71; *deterioration* = GAF drop ≥ 10; *sustained impairment*
= GAF < 61 at both 1y and 2y; *relapse surrogate* = CGI-S rise ≥ 2; *sustained illness* = CGI-S ≥ 4 at
both follow-ups.

| archetype | N | cohort mix (BP/SZ/DR) | func. remission | deterioration | sustained impair | relapse | prognosis |
|---|--:|---|--:|--:|--:|--:|:--|
| **low-burden** | 3,324 | 80/19/2 | **60%** | 16% | 12% | 12% | **best** |
| metabolic | 1,192 | 63/32/5 | 40% | 18% | 28% | 15% | moderate |
| mania/activation | 498 | 86/13/0 | 35% | 20% | 18% | 10% | moderate |
| sleep/circadian | 1,437 | 77/13/11 | 35% | 14% | 33% | 7% | moderate |
| developmental | 765 | 80/14/6 | 30% | 17% | 33% | 6% | moderate |
| high-sev+cognitive | 1,486 | 34/51/15 | 23% | 7%¹ | 55% | 4% | **poor (chronic)** |
| inflammatory | 174 | 59/34/7 | 16% | 16% | 48% | 15% | **poor** |
| **suicidality** | 137 | 72/25/4 | **14%** | 28% | 62% | 22% | **worst** |

¹ The high-severity-cognitive archetype's *low* deterioration rate is a floor effect — these patients
start impaired (mean baseline GAF ≈ 47) and cannot fall much further; their adverse signal is the **55%
sustained impairment**.

**Clinical vignettes (the "so what"):**

- **Low-burden (best):** mostly mild presentations; **60% reach functional remission**, low relapse.
  *Implication:* candidates for monitoring / step-down rather than intensification — the map flags them
  as low-risk regardless of diagnostic label.
- **Metabolic / sleep / developmental (moderate):** ~30–40% remit, but **a third remain functionally
  impaired** at 2 years. *Implication:* the durable biological burden (metabolic especially — the
  cleanest ⊥G signal, §4) marks a slower functional recovery worth proactive functional support.
- **High-severity-cognitive (poor, chronic):** SZ-enriched but transdiagnostic; **55% sustained
  impairment, only 23% remission**, yet *low* relapse — a **chronic, stable-low** course.
  *Implication:* prognosis here is largely baseline-determined (rehabilitation focus over relapse
  watch).
- **Inflammatory & suicidality (worst):** rare but high-risk tails. The **suicidality archetype** has
  the worst trajectory — **only 14% remit, 62% stay impaired, 22% relapse, and it is the one archetype
  that does not recover function over time** (see the trajectory atlas). *Implication:* a prognostic
  flag for **intensive, early, sustained follow-up**.

Figures: [`45_atlas_trajectories.png`](figures/45_atlas_trajectories.png) (the archetypes keep their
rank as the cohort improves — biology is the durable trait; the suicidality corner is the non-recoverer)
· [`45_atlas_rates.png`](figures/45_atlas_rates.png) (the green→red prognostic gradient).

---

## 4. What the map adds *over* DSM-5 — and where

The map and DSM-5 are **complementary, not rivals** (head-to-head, `reports/44`): each adds prognostic
information the other lacks. The division of labour is clinically intuitive:

- **The map owns *who changes*** — it separates functional **remission, deterioration, and relapse**
  better than the 7 DSM-5 subtypes.
- **DSM-5 owns *who stays severe*** — it separates the chronic severity-level outcomes better.

And the map's incremental value is **course-dependent**: large in the **episodic** courses (bipolar,
depression — where baseline doesn't fix the future) and small in schizophrenia (where it does). So the
map is the right tool precisely for the patients whose prognosis is otherwise uncertain. Used together:
**diagnosis for the illness type, the map for the trajectory within it.**

---

## 5. Is it worth using? — clinical value metrics (stage 46)

Patient-level 5-fold cross-validated discrimination, calibration, and net benefit (`reports/46`).
**Two honest messages:**

**(a) The clinician's reference already predicts these endpoints well.** Diagnosis + severity +
baseline outcome reach cross-validated AUC **0.76** (functional remission), **0.73** (deterioration),
**0.83** (sustained impairment), **0.87** (relapse surrogate), with positive net benefit over
treat-all/treat-none on the decision curves. The 2-year course is substantially predictable from
information a clinician already has — which is reassuring, but limits the room left for any add-on.

**(b) The map's *incremental individual* discrimination is small but real for functional outcomes.**
Adding the archetype map to the reference improves AUC by **+0.017 [+0.009, +0.026]** for functional
remission (CI excludes 0; P(gain>0)=1.0) and **+0.008 [0.000, +0.015]** for sustained impairment;
it adds **nothing** for deterioration (+0.003) or the relapse surrogate (+0.002, where the reference
already sits at 0.87). On the decision curves the map and reference overlap for the adverse endpoints —
no extra net benefit at the individual binary-decision level.

| endpoint | reference AUC | + map AUC | ΔAUC (map) | verdict |
|---|--:|--:|--:|:--|
| functional remission | 0.76 | 0.78 | **+0.017** [.009,.026] | reliable small gain |
| sustained impairment | 0.83 | 0.84 | +0.008 [.000,.015] | marginal |
| deterioration | 0.73 | 0.73 | +0.003 | none |
| relapse surrogate | 0.87 | 0.87 | +0.002 | none |

**So what — the honest positioning.** The map's value is **as a transdiagnostic prognostic
*stratification*** (the §3 atlas: a 14%→60% spread in functional-remission rate across groups that cut
across diagnosis) and in **continuous functional-trajectory** forecasting (ΔELPD +46 on the EGF scale,
`reports/43`) — **not** a large boost to *individual binary* risk discrimination over a clinician who
already knows the diagnosis, severity and current functioning. The apparent gap between "+46 ΔELPD" and
"+0.017 ΔAUC" is exactly the cost of collapsing the continuous trajectory to one threshold (GAF≥71) and
scoring it with a rank metric: most of the signal the map adds lives in *where on the functional
continuum* a patient lands, not in one binary cut. Read the map as a tool that **assigns a patient to a
prognostic group and sharpens the functional-trajectory forecast**, complementing — not replacing — the
clinician's severity-based judgement.

---

## 6. Honest limitations (so the "so what" isn't oversold)

- **Trajectories of clinical scales, not hard events.** No hospitalization/relapse register exists in
  the data; the endpoints are state transitions defined from repeated GAF/CGI-S (a relapse *surrogate*,
  not a recorded relapse).
- **Internal association, not a validated rule.** These are within-FACE prognostic associations,
  uncertainty-propagated and held-out-validated, but **not** externally validated or causal — a patient
  cannot yet be managed *by* this atlas; it is a hypothesis-generating decision aid pending prospective
  testing.
- **Concentrated in the episodic courses** (BP, DR); the schizophrenia increment is null (baseline
  saturation), and the rare archetypes (inflammatory N=174, suicidality N=137) have wide uncertainty.
- **2-year horizon, 3 visits** — coarse trajectory typing; longer-term prognosis is out of scope.

---

*Provenance: archetype assignment = M2 Arm-A dominant archetype (`results/face/patient_strata.parquet`);
endpoints = `src/face/prognosis/endpoints.py`; atlas = `scripts/45_endpoints.py` →
`results/face/m4/{archetype_atlas, endpoint_prevalence, archetype_vs_dsm5}.csv`.*
