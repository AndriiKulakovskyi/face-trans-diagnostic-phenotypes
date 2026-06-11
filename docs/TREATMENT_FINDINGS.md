# M5 — Treatment: findings (paper-facing, read first)

> What M5 asked, did, and found. Methods of record: [`TREATMENT_MODEL.md`](TREATMENT_MODEL.md);
> dev record in `reports/50–57_*.md`. *Status: M5 complete (pending PI sign-off). 2026-06-11.*

## The one-paragraph headline

M5 re-opened the treatment question with **real treatment data** — found, late, in the per-cohort
thesaurus `TRAITEMENTS` tabs (it was never in the harmonized common set). Harmonized to common drug-class
exposures and run through a proper causal pipeline (overlap gate → propensity → doubly-robust
errors-in-variables moderation), the honest answer is: **on observational treatment-as-usual, the
transdiagnostic map does not reliably *moderate* treatment response.** Where identification is cleanest —
**lithium in bipolar disorder** — it is a *well-identified null*. A suggestive but unconfirmed
metabolic/inflammatory signal for antipsychotic-treated functioning is a **hypothesis for prospective
testing**, not a claim. The boundary is therefore **earned, not assumed**: the map's clinical value
culminates at M4 (prognosis), and genuine treatment *selection* needs randomized data a future *M5b* would
require. As a bonus, M5 **strengthens M4** — its metabolic functional forecast survives adjustment for the
drugs patients were actually on.

## What M5 did — the causal pipeline

1. **Found + harmonized treatment data** (M5.0–M5.1). Captured by different mechanisms per cohort, all
   reduced to common drug-class exposures: **SZ** per-visit ATC-code lists (current), **DR** drug-class
   strings (current + lifetime), **BP** structured lifetime med-class flags + lithium plasma. Powered
   questions: lithium-in-BP (1,140 on / 1,353 off + plasma), clozapine-in-SZ (180), antipsychotic /
   antidepressant across cohorts.
2. **Overlap gate first** (M5.2a). `P(treat | severity + diagnosis + demographics + the 9 map coords)`,
   then common support + covariate balance before vs after stabilized IPTW. Lithium-BP overlaps almost
   perfectly (SMD 0.30→0.01); antipsychotic-BP balances (0.71→0.09); **clozapine-SZ is *channeled*** —
   reserved for the resistant, so IPTW cannot balance it (active-comparator SMD 0.44→0.61), the honest
   signature of confounding by indication.
3. **Doubly-robust EIV moderation** (M5.2b). On the common-support sample, IPTW + covariate-adjusted EIV
   GLM with the **durable-axis × treatment interaction** as the estimand; the ATE carries an **E-value**.

## The verdict

| question | identification | moderation (durable-axis × treatment) | read |
|---|---|---|---|
| **lithium-BP** | clean (100% overlap) | **null** (ΔELPD −1.4/−2.8; no axis HDI excludes 0) | the map does **not** pick lithium responders — a well-identified negative |
| **antipsychotic-BP** | balanced | **suggestive, unconfirmed** — metabolic −0.15\*, inflammatory −0.26\* on functioning; held-out ΔELPD +4.6±4.2 not confirmed; ATE E-value 1.79 | a **hypothesis** for an RCT, not a claim |
| **clozapine-SZ** | channeled | inflammatory×response −1.3\* but unconfirmed + small | **not trustworthy** (overlap gate) |

(\* HDI excludes 0.) Average treatment effects are **confounding-fragile** throughout (E-values 1.1–1.8) —
exactly the confounding-by-indication the design anticipated.

## M5 strengthens M4

**Treatment-as-confounder** (M5.2c): M4's functioning prognosis re-fit on the treatment-data subset
(N=1,324) with vs without the drug-class exposures — **metabolic→functioning survives** (β −0.051 → −0.048,
4.4% attenuation, HDI still excludes 0). M4's headline forecast is **not merely unmodelled treatment**.
(Inflammatory is directionally stable but the subset underpowers its band; cognition null, as in M4.)

## Why a null here is a genuine result

This is the honest precision-psychiatry answer, not a disappointment. M5 (i) **earned the boundary** —
observational TAU *cannot* demonstrate that the map prescribes (channeling, confounding by indication),
and we showed it rather than assumed it; (ii) delivered a **rigorous null** where identification is clean
(lithium-BP) and a **testable hypothesis** where it is not (metabolic/inflammatory × antipsychotic
functioning); (iii) built a **deployable causal method** (overlap → propensity → doubly-robust EIV
moderation) ready for the data a true M5b needs; and (iv) **back-validated M4**.

## The data ask (a true M5b)

Genuine treatment **selection** ("which drug for this phenotype") needs **randomized / trial-arm or
prospectively-controlled** data — confounding by indication caps what TAU can prove. The feasibility check
([`reports/59_m5b_feasibility.md`](../reports/59_m5b_feasibility.md)) is now done and is decisive on both
halves: **(a)** FACE contains **no randomization** (confirmed across CSVs + thesauri — observational by
design), so true selection requires **external** randomized data linked to FACE patients (the concrete
data-team ask); **(b)** but BP/SZ carry **per-visit medication trajectories with dates**, so a *stronger
observational* M5b — longitudinal / time-varying-treatment (g-methods), an upgrade over M5's baseline
exposure — is feasible **now**, with no new data (DR excluded: no follow-up Rx). The M5 method extends
directly to either.

## Honest limits

Observational (confounding by indication — the dominant threat, E-values 1.1–1.8); **prescription, not
protocol** (no titration/switching); **lifetime (BP) vs current (SZ/DR)** exposure; **within-cohort**
(lithium-BP, clozapine-SZ), not a clean transdiagnostic common-treatment; **channeled treatments
non-estimable** (clozapine); and, per M4, the map's *individual* increments are modest — the value is
group-level stratification and a hypothesis, not an individual prescribing rule.
