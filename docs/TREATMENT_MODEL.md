# M5 — Treatment: methods of record (bounds-and-defends)

> **The methods + math of record for Milestone 5.** Estimand, the treatment-exposure data (now found),
> the causal design for observational treatment moderation, the acceptance gates, and the staged
> pipeline. Read before any M5 modelling work. Sibling of [`PROGNOSIS_MODEL.md`](PROGNOSIS_MODEL.md).
> *Status: methods of record; copula re-scope 2026-06-24 (M5.0 audit 2026-06-11).*
>
> **8-factor rebuild note (2026-06-27; pending PI sign-off).** The M5 method is unchanged, but it now runs on the
> **8-factor** M1–M4 objects (immunometabolic merge + substance orthogonal; **A=5** archetypes). `DURABLE =
> (cognition, immunometabolic)` (via prognosis) + dynamic `arch_cols` carry the change; the only edits were
> dynamic archetype-count fixes in the course atlas. Result **replays the bounds-and-defends in full**:
> **ceiling** — lithium-BP a well-identified **bounded null** (E 1.20–1.28, interaction MDE ≈ 0.20),
> antipsychotic-BP a confounded average effect (E **1.80** ≈ native 1.79) with **suggestive-but-unconfirmed**
> moderation, clozapine-SZ underpowered; **defends M4** — the carrier survives treatment adjustment, now in
> *both* representations (the **A2 immunometabolic archetype corner** 7.7%/6.4% IPW *and* the immunometabolic
> durable axis 6.4%/4.1% — stronger than the 9-factor map); **describes** course heterogeneity (archetype ΔAUC
> +0.012/+0.034/+0.042; the immunometabolic corner carries ~2× resistance/side-effect risk; resistance
> AUC-marginal p 0.205). Findings: [`TREATMENT_OOP_FINDINGS.md`](TREATMENT_OOP_FINDINGS.md).

> **Re-scope (2026-06-24) — what M5 credibly delivers.** This baseline cohort has **no randomization**
> (`arm` is a DSM-5 subtype, not a randomized assignment) and only coarse, late-found drug-class
> exposure, so treatment *selection* (which drug for whom) is structurally out of reach — it is M5b, and
> needs randomized / trial-arm data. M5's standalone contribution is therefore to **bound and defend** the
> vertical's clinical claim: it (1) maps the **ceiling** of what the map licenses — *prognostic +
> descriptive, not prescriptive* — via a rigorous, **MDE-bounded** moderation null; (2) **defends M4** by
> showing the prognostic carrier is not a treatment proxy; and (3) **describes** treatment-response
> heterogeneity. The causal pipeline below — **overlap → propensity → doubly-robust EIV moderation →
> E-value → MDE** — is the reusable template for "can an *observational* map moderate treatment?".

## 1. The pivot, and the estimand

An earlier review concluded M5 was *data-blocked* (no treatment variable in the harmonized set) and
re-scoped it to a tolerability coda. **That premise was wrong:** treatment data exists in the raw
per-cohort files (the thesaurus `TRAITEMENTS` tabs), unharmonized — confirmed and characterized by the
M5.0 audit (`reports/53_treatment_audit.md`). M5 is therefore the **full treatment milestone** it was
always meant to be, answering the program's precision-psychiatry payoff:

> **Does the transdiagnostic map *moderate* treatment response — does the effect of a treatment on the
> 2-year outcome differ by stratum (stratum × treatment) — and does the M4 prognosis survive adjusting
> for treatment?** The valuable claim is *which phenotype benefits from which treatment*; the honest one,
> on observational data, is *moderation under explicit causal assumptions*.

House invariants carry over (fixed M1/M2/M3 objects; observed-cell likelihood, no imputation; diagnosis
as comparator/validation; a signal counts only if it clears its band). The new, dominant invariant is
**causal honesty**: treatment is *prescribed*, not randomized, so every effect is confounded by
indication and is reported as an explicitly-assumption-laden observational estimate, never a trial result.

## 2. Treatment exposures (the new data; M5.0 audit)

Captured by different per-cohort mechanisms, harmonized to **common drug-class exposures**
(`src/face/treatment/medications.py`):
- **SZ** — per-visit ATC-code lists (`med_psy_code_atc`) → classes (current, time-varying; gold standard).
- **DR** — drug-class strings (`psycho_act_cmclas` / `psy_lifetime_cmclas`) → classes (current + lifetime).
- **BP** — structured lifetime med-class flags (`cmoccur_*`) + `lithiumplasma` + a current-med table
  (`med_psy_*`, names; needs a name→class map).

Harmonized exposures + coverage (n exposed): **lithium** BP 4,224 (+plasma) · **antipsychotic** BP 6,824
/ SZ 1,793 / DR 370 · **antidepressant** BP 8,094 / DR 1,118 / SZ 847 · **mood-stabilizer** BP 6,669 /
SZ 480 · **anxiolytic** BP 6,356 / SZ 817 · **clozapine** SZ 511. Analyzable, powered questions:
**lithium-response-in-BP** (the classic), **clozapine-in-SZ** (treatment-resistance), antipsychotic /
antidepressant moderation, and **treatment-as-confounder for M4**. Temporality: SZ/DR current/per-visit;
BP mostly lifetime (illness-history-confounded).

## 3. Outcomes

The treatment-response outcomes from M5.0 (`face.treatment.endpoints`) — CGI-Improvement **response**,
**therapeutic effect**, **resistance**, **side-effects**, **adherence** — plus the M4 functioning/severity
trajectories (EGF, CGI-S, reused). Horizon V2, V1 replication. Response is severity-confounded (M5.0
audit) → adjust, as in M4.

## 4. The causal design — moderation on observational data

The crux. A naive `outcome ~ treatment × stratum` interaction is confounded by indication. The design:
1. **Target-trial emulation** per treatment question (eligibility, the "assignment" = exposed vs not at
   baseline, the outcome window V0→V2), stated explicitly.
2. **Propensity** for treatment: model `P(treatment | confounders)` — baseline severity (CGI-S + the
   error-corrected G), diagnosis/arm, demographics, **and the baseline map coordinates** — then adjust by
   IPW or covariate-adjustment (doubly-robust where feasible).
3. **The moderation estimand**: a conditional average treatment effect that varies by stratum —
   operationally the **treatment × stratum interaction** in a Bayesian outcome GLM (reusing the M4 EIV
   engine), adjusted for the propensity-balanced confounders. A credible non-zero interaction = the map
   moderates response.
4. **Lifetime vs current**: current/time-varying exposures (SZ/DR ATC, BP med table) are the cleaner
   moderation substrate; lifetime exposures (BP `cmoccur_*`) are reported as the confounded-but-powered
   complement, with the limitation stated.
5. **Treatment-as-confounder for M4**: re-fit the M4 prognosis adding treatment exposure — does the
   map's prognostic signal survive? (a rigor check that also strengthens M4). Reported both unweighted
   and under the M3 strata-independent attrition IPW (`w_retained_V2`).
6. **MDE / power closure (the bounded-null guard)**: a null moderation is only informative if the design
   *could* have seen a meaningful effect. From each interaction/ATE posterior SD, report the minimum
   detectable effect at 80% power — `MDE = (z_{0.975} + z_{0.80})·SD ≈ 2.802·SD` — separating a
   **bounded** null (small MDE, e.g. lithium-BP ≈ 0.19 SD) from an **underpowered** one (large MDE, e.g.
   clozapine-SZ ≈ 0.4–0.7 SD). This closes the template: *overlap → propensity → DR-EIV → E-value → MDE*.

## 5. Engine

Reuses the M4 stack — the EIV Bayesian GLM (`face.prognosis.glm`), nested ΔELPD (`compare`), clinical
metrics (`clinical_value`), IPW — adding `face.treatment.medications` (the harmonization layer) and a
propensity model. The interaction model: `g(E[Y]) = confounders + βT·treat + stratum + βTS·(treat×stratum)`,
with the map coordinates entering EIV. Per-cohort fits (the questions are within-cohort). The copula rework
(`src/face/treatment/treatment_model_oop.py`) adds the re-scope machinery: `moderation.mde` /
`moderation.sd_from_eti` (the MDE/power guard, off the already-fitted posteriors — no refit), an
IPW-weighted variant of the confounder-survival stage, and a degeneracy-free **held-out ΔAUC** heterogeneity
stage (`clinical_value.cv_predict` / `paired_auc_delta`) that re-states the response-heterogeneity result in
the clinician's currency without the IPTW-LOO degeneracy.

## 6. Acceptance gates (Q1–Q4)

- **Q1 moderation exists** — the treatment × stratum interaction improves held-out fit (ΔELPD) and/or a
  credible interaction coefficient, for a powered question (lithium-BP, clozapine-SZ).
- **Q2 beyond confounders (make-or-break)** — survives propensity adjustment (severity + diagnosis +
  map) and is not regression-to-the-mean; an E-value / unmeasured-confounding sensitivity is reported.
  **A null counts only if it is bounded** — the MDE must show the design could have resolved a
  clinically meaningful interaction (else the verdict is "underpowered", not "no moderation").
- **Q3 transdiagnostic / specificity** — within-cohort; where a treatment spans cohorts (antipsychotic),
  is the moderation consistent?
- **Q4 robust** — IPW vs covariate-adjustment agree; lifetime-vs-current agree in sign; leave-one-site/
  cohort-out; the interaction beats a permutation null.

## 7. Honest limits

**Confounding by indication is the dominant threat** — observational, not a trial; claims are
moderation *under assumptions* (no unmeasured confounding), with sensitivity analyses, never causal
proof. **Prescription, not protocol** (no dose-titration, switching, or adherence-verified exposure
beyond MARS). **Mostly within-cohort** (lithium-BP, clozapine-SZ); a clean transdiagnostic common-
treatment moderation is limited. **Lifetime exposures (BP)** are illness-history-confounded. And, per M4
/ the M5 coda, the map's *individual* increments are modest — the value is group-level moderation
signal, not an individual prescribing rule.

## 8. Pipeline (`scripts/53–57`)

`53_treatment_audit` *(done)* — exposure feasibility + harmonization audit · `54_exposures` — the
harmonization layer: build the per-(cohort,patient,visit) drug-class exposure table (ATC/class/flag →
common classes, current + lifetime; complete the DR vocab) · `55_propensity` — the treatment propensity
models + balance (per question) · `56_moderation` — the stratum × treatment interaction (lithium-BP,
clozapine-SZ, antipsychotic) · `57_confounder` — treatment-as-confounder for M4 (Q1, Q2) + the
unmeasured-confounding (E-value) sensitivity. Reports `58_dr_mars_fix` (DR-MARS harmonization fix) and
`59_m5b_feasibility` (randomization check / M5b feasibility) close the milestone.
Reuses the M4 engine + the M5.0 response endpoints. The coda scripts (50 inventory, 51 frame, 52
tolerability) stand as the response-outcome groundwork. Tests `tests/m5/`.
