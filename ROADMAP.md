# ROADMAP — Trans-diagnostic Phenotypic Subtypes Across BP / SZ / DR

> Author-facing research plan. Lists pre-registered methodological choices,
> phases of work, and the open questions that must be answered as we
> progress. **Update this document whenever a methodological decision is
> made** — git history of this file is the de-facto pre-registration record
> until / unless we move to OSF or a Registered Report.

---

## 1. Research question

The FACE cohort spans three DSM-5 categories (BP, SZ, DR) measured with a
common harmonized dictionary of clinical variables (demographics, social
history, perinatality, neuropsychology, substance use, suicide history, biology,
ECG, hetero- and auto-questionnaires, medical history). DSM-5 categorical
boundaries are widely held to under-fit the biological / functional
heterogeneity of severe mental illness.

**Primary question.** Does unsupervised clustering of FACE patients in the
harmonized clinical-feature space at baseline (V0) reveal **trans-diagnostic
subgroups** — i.e., patient clusters whose composition cuts across DSM-5
labels?

**Secondary question (longitudinal).** Are the V0 cluster boundaries
**stable across annual follow-up** (V1, V2, V3, V4), or do patients drift
between clusters in a way that itself defines clinically meaningful
trajectories?

**Tertiary question (predictive).** Do V0 clusters predict downstream
**functional outcome** (FAST, GAF, EuroQol-5D, WHODAS — exact target TBD per
data availability) at V1+, *over and above DSM-5 label*?

**Quaternary question (treatment, follow-up paper).** Do V0 clusters
differentiate treatment response trajectories?

## 2. Pre-registered hypothesis

**H1 (primary, trans-diagnostic).** Adjusted Rand Index (ARI) between the
discovered V0 cluster labels and the DSM-5 `arm` labels will be **substantially
below 1.0** (concrete threshold to lock in Phase 2: e.g., ARI < 0.4), and at
least one cluster will contain patients from ≥2 DSM-5 categories at
non-trivial proportions (≥ 15% each).

**H2 (stability).** ARI between V0 clusters and independently-derived V1
clusters (computed by re-running the same pipeline on V1 data and matching
patients by `usubjid_patients`) will exceed a stability floor (concrete
threshold to lock in Phase 3, e.g., ARI > 0.5). Stability decays
monotonically with visit distance V0→V1→V2→V3→V4 but does not collapse to
zero.

**H3 (prediction, exploratory).** V0 cluster membership is a significant
predictor of subsequent functional outcome at V1+ in a mixed-effects model
adjusting for `arm`, age, and sex.

## 3. Methodology — locked choices (Phase 1)

The following decisions are committed. Any change requires updating this
file and tagging the commit `methodology-vN`.

| Aspect                       | Decision |
|------------------------------|----------|
| **Variable / patient filter**| Two independent functions in `face_common.filters`: `filter_variables(df, threshold, visit)` and `filter_patients(df, threshold, visit, variables=...)`. Analysis scripts compose them. |
| **Feature anchor**           | V0-anchored: variables and patients passing the filter at V0 form the **canonical feature set**. The same variables are used unmodified at V1–V4 for stability analysis. |
| **Primary clustering**       | **Consensus across hierarchical (Ward + Gower), GMM (BIC-selected k), and HDBSCAN**. A patient-pair is co-clustered IFF ≥ 2 of 3 methods place them together. The individual methods become sensitivity analyses. |
| **Imputation**               | Library exposes three modes: (a) Gower-native partial distances, (b) MICE (m = 20), (c) KNN-within-cohort. **No default** — analysis script must pick and justify. |
| **Stability test**           | Independent re-clustering at V1..V4; match by `usubjid_patients`; report ARI per visit transition. |
| **Validation**               | **Site-based holdout**: training set = a pre-specified subset of FACE sites, validation = remaining sites. The exact site partition is committed in Phase 2 *before any clustering is run* and stored at `results/site_partition.json`. |
| **Cluster-vs-DSM-5 metrics** | ARI (primary); confusion matrix; per-cluster diagnostic composition; per-arm cluster spread. All three computed for each method and the consensus. |
| **Identifier exclusion**     | `usubjid_patients`, `cohort`, `arm`, `visit`, `visitnum`, `fondacode`, `armcd` never enter the feature matrix. |
| **Pre-registration**         | Git-tag only at this stage. Methodology commits tagged `methodology-vN`; the V0 analysis run will be tagged `analysis-v0-primary` immediately before execution. |
| **Submission target**        | Quality-driven, no hard deadline. Cadence in phases below. |

## 4. Phased plan

### Phase 0 — Harmonization (DONE)
- [x] Variable dictionary parsed; 348 feature variables PASS the audit.
- [x] Harmonization registry seeded (31 custom rules + identity-cast fallback).
- [x] Interactive HTML missingness QA report (`reports/qa_missingness.html`).

### Phase 1 — Filter library (DONE)
- [x] `face_common/filters.py` with `filter_variables` + `filter_patients`,
      visit-scoped, returning the filtered frame and a `VariableFilterReport`
      / `PatientFilterReport` carrying a per-element completeness table.
- [x] 24 unit tests in `tests/test_filters.py` (synthetic edge cases:
      threshold edges, identifier preservation, anchor vs. row-by-row,
      candidate restriction, V0Anchor.apply on V1..V4, missing columns).
- [x] `V0Anchor` dataclass + `select_v0_anchor()` helper. `apply(df,
      restrict_visits=...)` projects the V0 selection onto later visits.
- [x] `scripts/v0_anchor.py` — reproducible CLI run that writes
      `results/v0_anchor_features.csv`, `results/v0_anchor_patients.csv`,
      and `results/v0_anchor_meta.json` (timestamp + git rev + thresholds).
- [x] First baseline run at 75% / 75% on READY+PARTIAL: **73 features,
      6,289 V0 patients** (BP 5,101 / SZ 1,734 / DR 444). V3 attrition:
      DR drops to **3 patients** (the known cliff) — flag for any V3
      analysis. See [results/v0_anchor_meta.json](results/v0_anchor_meta.json).

### Phase 2 — Methodology lock + pre-registration (NEXT)
- [ ] Decide the V0 completeness thresholds. Phase-1 baseline at 75% / 75%
      yields 73 features × 6,289 patients. Sweep 60% / 75% / 85% triples
      via `scripts/v0_anchor.py --var-threshold X --pt-threshold Y` and
      pick a primary, justified against (a) feature interpretability and
      (b) per-cohort retention parity.
- [ ] Pick the imputation mode for the primary run; document the choice
      (and the alternatives that will appear as sensitivity analyses).
- [ ] Commit the site partition for site-based holdout to
      `results/site_partition.json`.
- [ ] Tag: `methodology-v1`.
- [ ] Quantify the H1 ARI threshold and H2 stability floor numerically.

### Phase 3 — Primary V0 clustering
- [ ] `face_common/clustering.py`: thin wrappers over scikit-learn /
      scipy / hdbscan, each returning `(labels, params, diagnostics)`.
- [ ] `scripts/cluster_v0.py`: load V0 frame → apply V0 anchor filter →
      impute → run all 3 methods → consensus labels → save to
      `results/v0_clusters.csv` and per-method outputs.
- [ ] `scripts/cluster_v0_report.py`: HTML report mirroring the QA report
      style, with the confusion matrix, ARI, per-cluster DSM-5 composition,
      and the top distinguishing features per cluster (Cohen's d).
- [ ] Decide cluster count for the consensus (likely via plurality of the
      three methods' k-selection; document procedure in advance).

### Phase 4 — Stability at V1..V4
- [ ] `scripts/cluster_stability.py`: re-run the full pipeline at V1, V2,
      V3, V4 with the V0-anchored feature set; compute ARI per visit and a
      per-patient "cluster trajectory" (sequence of cluster labels).
- [ ] Drift visualisation: Sankey diagram of cluster transitions V0→V1,
      V1→V2, …, V3→V4.
- [ ] **DR cohort caveat**: V3 has only 3 patient×visit rows. Either
      exclude DR-V3 or carry it as anecdote — decide and document.

### Phase 5 — Outcome prediction (functional)
- [ ] Identify functional outcome variables in the dictionary (FAST family
      `fast28`, `fast30`, candidate GAF / EuroQol items). Inventory in
      `ROADMAP.md` after explicit grep.
- [ ] Mixed-effects regression: outcome ~ V0_cluster + arm + age + sex +
      (1 | site) + (1 | usubjid_patients).
- [ ] Report effect sizes, p-values, and AUROC if outcomes binarised.

### Phase 6 — FACE-score (predictive model, follow-up)
- [ ] Patient-level summary score derived from V0 cluster posterior +
      key cluster-defining features.
- [ ] Calibration plot, decision-curve analysis.
- [ ] Treatment-response analysis (Phase 7 prerequisite — depends on
      TRAITEMENTS section extraction, currently out of dictionary scope).

### Phase 7 — External validation (manuscript prerequisite)
- [ ] Identify candidate external cohorts: PRISM, B-SNIP, ENIGMA,
      UK Biobank Mental Health module. Verify variable overlap with FACE
      common dictionary.
- [ ] Replication clustering in the external cohort with the V0-anchored
      feature subset.

### Phase 8 — Manuscript
- [ ] Figure inventory locked.
- [ ] Methods section traced cell-by-cell back to git tags.
- [ ] Supplementary code repository frozen at submission tag.

## 5. Open questions (must be resolved before each phase commits)

**Before Phase 2:**
- Q2.1 — What completeness threshold? Default 75% is a placeholder.
  Decision must be informed by the QA report's distribution.
- Q2.2 — Which imputation mode is primary? Gower-native is the simplest
  pre-reg but limits us to hierarchical clustering only at the impute step.
- Q2.3 — Site partition: by city, by recruitment year, or random within
  cohort? Random-within-cohort is least informative but easiest to defend.

**Before Phase 3:**
- Q3.1 — How is k determined for each method? GMM uses BIC. Hierarchical
  needs a cut-height rule (silhouette? gap statistic?). HDBSCAN uses
  `min_cluster_size` (set to a fixed fraction of N).
- Q3.2 — Feature standardisation: z-score within cohort? Across pooled?
- Q3.3 — How to handle the readiness-tier choice? Primary on READY only,
  or READY+PARTIAL? Affects N variables (130 vs 351) and clusters.

**Before Phase 5:**
- Q5.1 — Which functional outcome is primary? FAST total, GAF, or a
  composite? Pre-register the choice.
- Q5.2 — Are outcomes available at all visits or only at scheduled
  follow-ups? Compute completeness in the QA report.

**Before Phase 7:**
- Q7.1 — Which external cohort has the closest variable coverage to
  FACE's V0 anchor feature set? May require a separate harmonization
  exercise per cohort.

## 6. Risks and known limitations

- **DR V3 attrition cliff (n=3)**: any V3-dependent analysis is statistically
  meaningless for DR. Surface in QA, exclude or carry as descriptive only.
- **Cohort imbalance** (BP 6,252 / SZ 2,209 / DR 552): can bias both
  clustering and stability metrics. Sensitivity analysis with cohort-balanced
  bootstrap subsamples required.
- **Harmonization residual error**: 45 audit WARNs are cohort-level value-set
  divergences — clinical alignment task per row. Most will not change cluster
  membership, but track sensitivity by repeating Phase 3 with each WARN row
  excluded.
- **TRAITEMENTS section is unparsed** in the current dictionary. Treatment-
  response analyses are blocked until that section is added.
- **`siteid_city` lookup table** is currently a placeholder — register the
  real per-cohort SITEID→city mapping before Phase 2 site-based holdout.

## 7. Glossary

- **V0** = inclusion / baseline visit. **V1..V10** = year 1..year 10
  follow-up.
- **`arm`** = DSM-5 sub-diagnosis text label ("Bipolaire de type 1",
  "Schizophrénie", "Trouble dépressif majeur", …) — held out as a label,
  never as a feature.
- **`cohort`** = primary DSM-5 category (BP / SZ / DR) — likewise held out.
- **READY / PARTIAL / NOT USABLE** = dictionary `cluster_readiness` tags
  for cross-cohort comparability.
- **Consensus clustering** = the pre-registered primary clustering: a
  patient pair is co-clustered iff ≥ 2 of {hierarchical+Gower, GMM, HDBSCAN}
  put them in the same cluster.
