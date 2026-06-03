# Validation plan — does the v2 dimensional model *matter*? (pre-registration)

> 📁 **ARCHIVED — executed pre-registration** (moved to `docs/planning/` in the v2 cleanup). This is
> the plan *as written before the analysis was run*, kept for provenance. Some details describe
> *intended* work and differ from what was built; for what was **actually done** see
> [../PIPELINE.md](../PIPELINE.md) and [../LABBOOK.md](../LABBOOK.md).

> Review & lock before coding. Decides whether the 4 trans-diagnostic axes are a **useful** result
> or a rigorous-but-descriptive one. Builds on the validated dimensional + stratification arms
> (LABBOOK V2-6…V2-13). Axes: `src/trans_diag/axes_v2.py`. Masked / no-imputation throughout.

## The make-or-break question
The dimensional account is internally solid but, as a reviewer noted, **largely confirmatory of
HiTOP/RDoC and of unproven clinical value**. Four studies decide whether it earns its keep, in
priority order. Two are quick and *defend/sharpen* the existing result (A, B); two test *whether it
is useful* (C, D). **D (predictive validity) is the make-or-break.**

## Data reality (inventory, `/tmp` → to promote to `scripts/41_v1v4_inventory_v2.py`)
- **Attrition:** V0 9,013 → V1 4,270 → V2 2,958 → V3 1,955 → **V4 779**. Non-random (informative
  dropout likely). **DR collapses longitudinally (V3 n=3)** → the longitudinal/predictive arm is
  effectively **BP+SZ**; V1–V2 are the workhorses; **V4 is underpowered** (report but don't lead on it).
- **Usable follow-up outcomes** (coverage ≥~50% at V1–V2): GAF (`egf`), FAST, MADRS, YMRS, CGI-S
  (`cgi01`), QIDS, EQ-5D, suicidal ideation (`isf05`), hospitalization count/duration
  (`nboccur/hodur_hospitalisation_lt`). **Unusable:** work disability (~0% after V0), suicide
  *attempts* (`isf08` ~2%), C-SSRS (~16%).
- **Circularity:** MADRS/YMRS/FAST/EGF/QIDS are *both* dimension inputs *and* candidate outcomes.
  Meaningful prediction must be **cross-domain** (V0 cardiometabolic/cognition → V4 functioning/
  hospitalization) and/or **incremental over the V0 baseline of the same outcome** — never the trivial
  V0-depression → V4-depression autocorrelation.

---

## Study A — Confront the cohort confound (quick; defends the result)
**Attack:** the 3 cohorts *are* the 3 diagnoses; the axes might encode between-cohort/batch effects.
**Tests** (`scripts/42_cohort_confound_v2.py`):
1. **Within-cohort re-derivation** — re-run Stage 2→3 (construct scores → Φ₁ → K=4 promax) **within
   BP alone** and **within SZ alone** (DR too small); Tucker congruence vs the pooled loadings. Pass:
   each axis congruent (≥0.85) within each cohort → the axis exists *inside* a diagnosis, not just
   between diagnoses.
2. **Cohort-residualized sensitivity** — residualize construct scores on **cohort + age + sex** (vs
   the primary age+sex), re-derive K=4, congruence vs primary. Pass: axes survive (the structure is
   not the between-cohort mean differences).
**Decision:** axes that replicate within-cohort *and* survive cohort-residualization are robustly
trans-diagnostic. Flag any that don't (candidate: cardiometabolic, the weakest/most cohort-protocol-
sensitive).

## Study B — Symptom–biology orthogonality + the integrated no-p-factor (quick; THE novelty)
**Goal:** make the one non-derivative message rigorous and quantified.
**Tests** (`scripts/43_orthogonality_pfactor_v2.py`):
1. **Cross-block orthogonality** — distribution of construct-construct |r| *between* symptom and
   biology blocks vs *within* block; report mean/CI. Claim: symptom⊥biology (between ≈ 0, within > 0).
2. **p-factor is a symptom-only artifact** — fit the bifactor / compute **ECV on (i) symptom-only
   constructs, (ii) symptom+cognition, (iii) symptom+biology, (iv) the full integrated set (ECV 0.36)**.
   Prediction: ECV_symptom-only is substantially higher → *admitting biology dissolves the general
   factor*. This is the headline, made falsifiable.
**Decision:** if ECV_symptom-only ≫ ECV_integrated and between-block |r| ≈ 0, lead the manuscript with
"a general severity factor is an artifact of symptom-only measurement; an integrated model is
multidimensional." If not, drop this as the headline and fall back to the descriptive framing.

## Study C — Longitudinal coherence (medium; the project's stated design)
**Goal:** are the V0 axes stable/coherent over V1–V4? (V0 defines, later visits validate.)
**Tests** (`scripts/44_longitudinal_coherence_v2.py`; adapt v1 scripts 08/09):
1. **Measurement invariance** — apply the V0 construct definitions to V1, V2 (V3/V4 BP+SZ only),
   re-estimate Φ₁ and K=4 *at each visit*; Tucker congruence of per-visit loadings vs V0. Pass: ≥0.85
   → the same dimensions exist at follow-up.
2. **Score stability** — project the V0 loadings onto each visit (masked scores), compute per-axis
   **rank-order test-retest** (V0↔V1↔V2 Spearman) and **mean-level change** (do scores drift, e.g.,
   internalizing improving with treatment?). Separate *structural* stability (the axes persist) from
   *individual* change (patients move along them).
**Caveats:** completers only (report attrition bias — compare V0 scores of stayers vs dropouts);
treatment effects expected on mean levels (that's fine — rank-order stability is the structural test).
**Decision:** structurally coherent if loadings congruent ≥0.85 at V1–V2; report rank-order stability
as the individual-level result.

## Study D — Predictive validity vs DSM (THE make-or-break)
**Question:** do the V0 axes predict V1–V4 outcomes **better than / incremental to DSM diagnosis**?

### Derived relapse outcome — LOCKED (`/tmp` checks → promote to `scripts/41_*`)
A reviewer wanted mathematical purity, so the definition is *data-verified*, not asserted:
- **REJECTED — hospitalization-count relapse.** `nboccur_hospitalisation_lt` is a *lifetime* count but
  is **non-monotonic in the data — 41% of consecutive pairs DECREASE** (recall/reporting noise). A
  lifetime total cannot legitimately fall, so Δ(count) is not a trustworthy relapse signal. Documented
  data-quality finding; do not use. (Also: its dictionary `Codage` says "mmHg" — a copy-paste bug.)
- **PRIMARY — CGI-S clinical relapse by V2 (binary).** `relapse = 1` if CGI-S (`cgi01`) **rises ≥2
  points OR crosses from <4 to ≥4** ("moderately ill"+) in *either* the V0→V1 or V1→V2 interval.
  Verified: prevalence **20%** (BP 23% / SZ 14% / DR 8%), **n=3,657 evaluable** — clinician-rated,
  well-covered (~74%), an *event* not a self-report scale.
- **SENSITIVITY — mood-syndromal relapse.** MADRS crosses <20→≥20 OR YMRS crosses <12→≥12 (9–12%/
  interval); and per-interval CGI relapse for granularity.
- **Circularity control (essential):** CGI-S loads on the internalizing axis, so the *clean* tests are
  **cross-domain** (V0 **cardiometabolic / cognition / illness-course** → relapse) and **incremental
  over the V0 CGI-S baseline + DSM**. Mood-syndromal is more circular (note it).

**Other outcomes** (V1, V2 primary; V4 secondary/underpowered):
- *functioning / QoL* (cross-domain, non-circular): GAF (`egf`), FAST, EQ-5D.
- *symptom severity* (circularity-controlled): MADRS, YMRS, CGI-S — **only with the V0 baseline of that
  outcome in every model**.
- Hospitalization count/duration: **excluded** as outcomes (same unreliability as above).
**Predictor sets** (all V0):
- M0 = age + sex (+ V0 baseline of the outcome, for symptom outcomes);
- M1 = M0 + **DSM** (cohort + arm);
- M2 = M0 + **4 dims (+ mania + suicidality)**;
- M3 = M0 + DSM + dims.
**Models & metrics:** linear/ridge for continuous (GAF, FAST, MADRS…); Poisson/NB for counts
(hospitalizations); logistic for binary (any hospitalization, GAF<50). **Nested K-fold
cross-validation** (cohort-stratified), report out-of-sample R²/AUC with CIs; nested model comparison
(M2 vs M1 = dims-vs-DSM; M3 vs M1 = dims add over DSM) via cross-validated ΔR²/ΔAUC (+ DeLong / LRT).
**Attrition handling:** primary = completers; sensitivity = inverse-probability-of-attrition weighting
(model dropout from V0 features) + report whether dropout is predicted by the dims (informative
missingness is itself a finding).
**Decision (the verdict):**
- **Earns its keep** if the dims add **cross-validated incremental** value over DSM (M3 > M1, CI
  excludes 0) for ≥1 *hard/cross-domain* outcome (hospitalization or functioning), and ideally M2 ≥ M1.
- **Descriptive only** if dims do not beat DSM out-of-sample on hard outcomes → write it honestly as
  "rigorous dimensional structure, DSM-equivalent for prognosis" (still publishable, still useful).

---

## Cross-cutting rigor (the "no stat mistakes" contract)
- **Out-of-sample everywhere** for D (no in-sample R²/AUC claims); cohort-stratified folds.
- **Circularity control**: V0 baseline of the outcome in every symptom-outcome model; lead on
  cross-domain/hard outcomes.
- **Attrition is a confound, not ignorable** — quantify, weight, and report it.
- **Multiple comparisons**: pre-specify the primary outcomes (hospitalization, GAF); FDR-correct the rest.
- **DR caveat**: longitudinal claims are BP+SZ; state explicitly.
- **Masked / no-imputation**: outcomes available-case; predictors are the existing masked scores.
- **Join-key gotcha (verified):** the raw long frame uses **uppercase** cohort (`BP/SZ/DR`) + `patient_uid`
  (`COHORT::usubjid`); the V0 dimension scores use **lowercase** (`bp/sz/dr`, `patient_id`). Reconcile
  case + key when joining V0 scores to V1–V4 outcomes, or rows silently fail to match (caught in the
  relapse check). Build a single canonical patient key once.

## Sequencing & gates
1. **A + B first** (quick): defend the cohort confound, establish the orthogonality/p-factor headline.
   *Gate:* if B fails (no symptom-only p-factor advantage), drop that headline.
2. **C** (longitudinal coherence): structural validation.
3. **D** (predictive validity): the decision. *Gate:* the verdict (earns-its-keep vs descriptive)
   determines the manuscript's claim and whether we add a treatment-prediction angle.
4. Then **Phase 6** manuscript, framed by the D verdict.

## Deliverables
`scripts/41_v1v4_inventory_v2.py` (promote the inventory) · `42_cohort_confound_v2.py` ·
`43_orthogonality_pfactor_v2.py` · `44_longitudinal_coherence_v2.py` · `45_predictive_validity_v2.py`;
results to `results/hfa/` (gitignored); findings → FINDINGS Tracks 1–3 + LABBOOK.
