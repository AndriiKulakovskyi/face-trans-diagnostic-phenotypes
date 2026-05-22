# Stage C — Minimum Clinical-Feature Panel Held-Out Validation

> **Rename notice.** This document was previously titled
> *Stage C — Biomarker Panel Held-Out Validation*. The rename (and
> the accompanying methodological tightening) reflects our decision
> to reserve the word "biomarker" for validated biological
> measurements. The six panels below are built from routinely
> collected phenotypic variables (family history, suicide history,
> CGI severity, illness duration, PSQI sleep, BIS-10 impulsivity,
> BMI, waist, triglycerides) and are best described as
> **parsimonious clinical discriminators** or **sparse phenotypic
> signatures** — not as biomarkers.

**Sub-project:** `face_stratification`
**Cohort:** FACE V1 baseline, 11,014 patients
**Clusters:** Stage C consensus (k=6, sign-corrected)
**Status:** complete
**Canonical output:** `output/stratification/stage_c/deep_analysis/clinical_panel_validation.json`
**Audit output:** `output/stratification/stage_c/deep_analysis/clinical_panel_validation_leaky.json`
**Legacy path (kept for back-compat):** `biomarker_validation.json` — now
a copy of the sanitised payload.

This document reports the **leakage-safe external validation** of the
Stage C minimum clinical-feature panels. An earlier version of this
document reported training AUCs in the 0.92–1.00 range and claimed
that every panel "generalizes essentially perfectly". That claim was
wrong in the precise sense that the discriminators were allowed to
use the **eight universally-measured features** that seed the Stage A
transdiagnostic similarity graph: `demo_age_years`, `demo_sex_male`,
three substance-use flags (`sub_tobacco_current`,
`sub_alcohol_current`, `sub_cannabis_current`), `sub_use_disorder`,
and the two comorbidity counts (`cm_n_somatic`, `cm_n_psychiatric`).
Because those features are the inputs to the similarity kernel that
defined the clusters in the first place, a logistic regression that
includes them recovers the similarity function circularly and
reports an AUC that is essentially "how well does this logistic
regression approximate the cluster assignment on the very features
that produced it". We now report two variants side-by-side:

- **Sanitised** — whitelist *excluding* the eight embedding-input
  features (41 candidates). This is the variant that should be
  cited.
- **Audit** — whitelist *including* the eight embedding-input
  features (49 candidates). Reported only to quantify the leakage.

---

## 1. Methodology

For each target cluster $c \in \{0, 1, 2, 3, 4, 5\}$:

1. **Build binary labels.** $y = 1$ if the patient is in cluster
   $c$, else $y = 0$.
2. **5-split stratified shuffle CV** (test fraction 20 %) using a
   combined stratum `(y, cohort)`. If any joint stratum has
   fewer than 2 members — which happens for the smallest cluster,
   C0 (n = 117) — we **fall back to y-only stratification** and
   log a warning. The fallback is what lets every cluster
   (including C0) receive a panel under the new validator.
3. **Per-fold feature pipeline**, re-fit on each training slice:
   a. Median imputation + z-score standardization (fit on training
      fold only).
   b. Per-fold univariate AUC ranking; keep features with
      train-fold AUC ≥ 0.55.
   c. Greedy forward selection of up to 6 features on the training
      fold, stopping early if the next candidate adds < 0.001 AUC.
   d. Logistic regression fit on the selected features; threshold
      at the Youden $J$ optimum.
4. **Evaluate on the 20 % held-out slice.**
5. **Minimum positives guard:** `MIN_PANEL_POSITIVES = 10`
   (lowered from the previous 20 so C0 is not silently dropped).

Implementation lives in
`src/face_stratification/stage_c/clinical_panels.py` as
`validate_clinical_feature_panel_cv` /
`validate_all_clinical_feature_panels_cv`. The legacy module
`src/face_stratification/stage_c/biomarkers.py` is now a compatibility
shim that re-exports the new names and emits `DeprecationWarning`
from the legacy function wrappers.

The script that regenerates both JSONs is
`scripts/validate_clinical_panels_cv.py`. The old
`scripts/validate_stage_c_biomarkers.py` is preserved only as a
legacy entry point.

---

## 2. Headline validation results

Values from `clinical_panel_validation.json` (sanitised) and
`clinical_panel_validation_leaky.json` (audit). AUC ± standard
deviation across the 5 shuffle splits.

| C | n⁺    | Sanitised train AUC | **Sanitised test AUC**  | gap    | Audit test AUC  | **inflation** |
|--:|------:|--------------------:|------------------------:|-------:|----------------:|--------------:|
| 0 |   117 |     0.880 ± 0.007  | **0.859 ± 0.021**       | +0.021 | 0.996 ± 0.002  | **+0.137**   |
| 1 | 2,653 |     0.713 ± 0.001  | **0.707 ± 0.003**       | +0.006 | 0.953 ± 0.004  | **+0.247**   |
| 2 | 1,796 |     0.712 ± 0.002  | **0.710 ± 0.010**       | +0.002 | 0.988 ± 0.004  | **+0.278**   |
| 3 |   933 |     0.706 ± 0.003  | **0.696 ± 0.007**       | +0.010 | 0.9995 ± 0.0003 | **+0.303**   |
| 4 | 2,099 |     0.680 ± 0.003  | **0.672 ± 0.012**       | +0.008 | 0.924 ± 0.004  | **+0.252**   |
| 5 | 3,416 |     0.691 ± 0.002  | **0.686 ± 0.005**       | +0.005 | 0.991 ± 0.000  | **+0.305**   |

**Key observations.**

1. **All six clusters now have a panel.** Lowering
   `MIN_PANEL_POSITIVES` from 20 to 10 and adding the y-only
   stratification fallback means that the pediatric cluster C0 is no
   longer silently dropped from the validation — its sanitised panel
   reaches 0.859 held-out AUC, the best of the six.
2. **Sanitised AUCs live in 0.67–0.86.** Train–test gaps are ≤ 0.02
   across all six clusters, so the panels generalise without
   overfitting.
3. **The leakage correction is large.** The audit variant's AUCs
   inflate the sanitised numbers by +0.14 to +0.30 points, with the
   biggest inflation on the transdiagnostic clusters (C3, C5:
   +0.30) and the smallest on the near-homogeneous pediatric C0
   (+0.14). The inflation on C3 pushes the audit AUC to 0.9995 —
   the source of the earlier "perfect biomarker" claim that we now
   retract.

---

## 3. Sanitised feature-selection stability

Features retained in ≥ 80 % of the 5 splits. These define each
cluster's **sparse phenotypic signature**.

| Cluster | Stable features (sanitised, ≥ 80 % split selection)                                                                 |
|--------:|---------------------------------------------------------------------------------------------------------------------|
| C0      | `fh_n_affected_relatives`, `bio_bmi`, `sui_ever_attempt`, `demo_marital_partnered`, `sui_ever_ideation`            |
| C1      | `fh_n_affected_relatives`, `sui_ever_attempt`, `inst_bis10_total`, `demo_education_years_ordinal`, `inst_psqi_total`, `sui_ever_ideation` |
| C2      | `fh_n_affected_relatives`, `demo_marital_partnered`, `psyh_age_first_episode`                                      |
| C3      | `bio_waist_cm`, `psyh_illness_duration_years`, `fh_n_affected_relatives`, `bio_triglycerides`, `inst_bis10_total`, `psyh_age_first_episode` |
| C4      | `psyh_illness_duration_years`, `demo_marital_partnered`, `bio_bmi`, `inst_cgis_total`, `psyh_age_first_episode`    |
| C5      | `fh_n_affected_relatives`, `sui_ever_ideation`, `inst_cgis_total`, `sui_ever_attempt`, `psyh_illness_duration_years`, `fh_bipolar_any` |

`fh_n_affected_relatives` (family psychiatric load) is present in
5 of 6 clusters; suicide history (`sui_ever_attempt` /
`sui_ever_ideation`) is present in 4 of 6; illness duration and
family / clinical-severity flags dominate the rest. **C3's
sanitised signature is the clinically most interesting result of
the rerun**: waist + triglycerides + illness duration + family
load + BIS-10 impulsivity + age at first episode. This is a
genuine metabolic-impulsivity-early-illness phenotype that the
leaky panel had completely masked behind the demographic /
comorbidity shortcut.

By contrast, the **audit** (leaky) panels all look similar to
each other: they select some combination of `cm_n_psychiatric`,
`cm_n_somatic`, `demo_age_years`, `demo_sex_male`,
`sub_use_disorder`. This is the circular "recover the similarity
function from its own inputs" behaviour that motivated the
sanitisation.

---

## 4. What these panels are not

- **They are not deployable tests.** 0.67–0.86 AUC is well below
  what would be needed for any individual triage decision.
- **They are not biomarkers** in the biomedical-test sense. Only a
  handful of features (BMI, waist, triglycerides) are biological,
  and even those are bedside metabolic markers rather than
  validated discriminators.
- **They are not exhaustive.** The whitelist is deliberately the
  subset of features for which cross-cohort availability is
  reasonable, and it omits cohort-specific instruments that might
  carry extra within-cohort information.

## 5. What they are

**Parsimonious clinical discriminators** / **sparse phenotypic
signatures** that compactly describe each Stage C consensus
cluster in ≤ 6 routinely collected variables, with a per-fold,
leakage-safe evaluation protocol that reports exactly how much
of the discriminability is an artefact of using the graph-seeding
features vs. how much is genuine residual signal. Their intended
uses are:

1. Teaching and communication (compact phenotype descriptions).
2. Prospective screening into follow-up sub-studies (enrichment,
   not triage).
3. Candidate oracles for the downstream RLVR precision-psychiatry
   LLM training pipeline.
4. Hypothesis generation (the C3 metabolic-impulsivity signature
   is the clearest example).
5. Auditable, inspectable summaries that are defensible to ethics
   boards and regulators.

---

## 6. Reproducing these numbers

```bash
python scripts/validate_clinical_panels_cv.py
```

The script writes, in place, the three JSON payloads
(`clinical_panel_validation.json`,
`clinical_panel_validation_leaky.json`,
`biomarker_validation.json` — now a copy of the sanitised payload)
and the side-by-side `clinical_panel_validation_summary.csv`.

All three files are regenerated deterministically from the
`consensus_labels.parquet` partition and the harmonized Stage A
dataset; no previously-fit model state is reused.
