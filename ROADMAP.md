# ROADMAP — Trans-diagnostic Phenotypic Subtypes Across BP / SZ / DR

> Research plan and methods specification for the FACE clustering study.
> This is the single source of truth for *what* we are doing and *why*.
> **Update it whenever a methodological decision is made** — the git history
> of this file is our de-facto pre-registration trail (see §11). Sections
> 1–6 describe the current design; §11 records how we got here.

---

## 1. Research question & rationale

The FACE cohort spans three DSM-5 diagnostic categories — Bipolar disorder
(BP), Schizophrenia-spectrum (SZ), and Depressive disorder (DR) — all measured
with a single harmonized dictionary of clinical variables (demographics,
social history, perinatality, neuropsychology, substance use, suicide history,
biology, ECG, hetero- and auto-questionnaires, medical history). DSM-5
categorical boundaries are widely held to under-fit the biological and
functional heterogeneity of severe mental illness, and to obscure features
shared *across* diagnoses.

We ask, in order:

1. **Primary (trans-diagnostic).** Does unsupervised clustering of patients in
   the harmonized clinical-feature space at baseline (V0) reveal subgroups
   whose composition **cuts across** DSM-5 labels?
2. **Stability (longitudinal).** Are the V0 clusters **temporally coherent**
   across annual follow-up (V1–V4), or do patients drift between them in a way
   that itself defines meaningful trajectories?
3. **Prediction.** Do V0 clusters predict downstream **functional outcome**
   (GAF/FAST) at later visits, *over and above* the DSM-5 label?
4. **Treatment (follow-up paper).** Do V0 clusters differentiate treatment-
   response trajectories? (Blocked until the TRAITEMENTS section is parsed.)

## 2. Pre-registered hypotheses

- **H1 (trans-diagnostic).** The Adjusted Rand Index (ARI; see §12) between the
  discovered V0 clusters and DSM-5 `arm` labels is **substantially below 1.0**
  (threshold to be fixed numerically at methodology-v1, candidate ARI < 0.4),
  AND at least one cluster mixes ≥ 2 DSM-5 categories at non-trivial proportions
  (≥ 15% each). Paired with a stability check (§6) so that a low ARI is read as
  *real cross-cutting structure*, not noise.
- **H2 (stability).** ARI between V0 clusters and independently re-derived V1
  clusters (matched by `patient_uid`) exceeds a stability floor (candidate
  ARI > 0.5); coherence decays monotonically V0→V4 but does not collapse.
- **H3 (prediction, exploratory).** V0 cluster membership predicts later
  functional outcome in a mixed-effects model adjusting for `arm`, age, sex,
  site.

> ⚠️ **Leakage guard.** `egf` (Échelle Globale de Fonctionnement = GAF) and any
> FAST/WHODAS items are H3 outcomes. They are **held out of the clustering
> feature matrix** — clustering on an outcome and then "predicting" it is
> circular.

## 3. Data & key concepts

| cohort | DSM-5 category | V0 patients |
|---|---|---|
| BP | Bipolar disorder | 6,252 |
| SZ | Schizophrenia-spectrum | 2,209 |
| DR | Depressive disorder | 552 |
| **total** | | **9,013** |

Visits: **V0** (inclusion / baseline) → **V1…V4** (annual follow-up). Attrition
is steep; **DR collapses to 3 patients at V3** — a structural cliff, not a
tunable parameter (§9).

Core concepts:

- **`patient_uid` = `cohort::usubjid_patients`** — the globally-unique patient
  key. `usubjid_patients` is reused across cohorts (970 colliding ids), so all
  patient-level operations key on `patient_uid` (see C5, §11).
- **Readiness tiers** (dictionary `cluster_readiness`): **READY** (130 vars,
  comparable + present in all 3 cohorts), **PARTIAL** (221 vars, 2-of-3 cohorts
  or with a construct caveat), **NOT USABLE** (26 vars).
- **Informative core (67 features)** — the **Stage-1 discovery substrate**:
  the READY tier minus 19 near-constant rare-disease binaries (modal ≥ 95%)
  minus items with column-completeness < 70%. Balanced ~24 biology / ~25
  psychopathology / ~18 demographic-social.
- **Identifiers (never clustered on):** `patient_uid`, `usubjid_patients`,
  `cohort`, `arm`, `visit`, `visitnum`, `fondacode`, `armcd`. `arm` and
  `cohort` are reserved as labels for cluster evaluation (ARI, composition).

## 4. Strategy — staged discovery pipeline

Each stage adds **exactly one** new source of complexity, so when a result
breaks we know which assumption caused it.

| stage | sample | features | new complexity | primary method | question |
|---|---|---|---|---|---|
| **1 · discover** | balanced **198 / cohort** (DR-limited), random-sampled from patients ≥ 90% complete | informative core (67) | imbalance removed; ~no block-missingness | hierarchical (Ward) + Gower | Do clean, balanced data yield trans-diagnostic clusters? |
| **2 · refine** | same discovery set | core + PARTIAL (2-cohort) vars | block-missingness by cohort | **SNF + Leiden** | Do partial variables add unique structure (info gain)? |
| **3 · generalize** | **all 9,013 V0 patients** | informative core | full-sample noise | project Stage-1 model | Do the clusters exist in the patients we did NOT select? **(headline result)** |
| **4 · longitudinal** | all patients, V1–V4 | informative core | time / attrition | re-cluster + match by `patient_uid` | Are clusters temporally coherent, or do they drift? |

**Design principles**

1. **Core-first.** Stage 1 uses only fully-shared variables → one ruler for
   every patient → no data-availability bias, no block-missingness, no
   imputation. Classical clustering is appropriate and sufficient here.
2. **Graph earns its place.** SNF/diffusion is justified specifically by
   **block-structured missingness** (a cohort missing an entire domain — e.g.
   DR has no NEUROPSYCHOLOGIE), which appears only at Stage 2. It must beat the
   Stage-1 classical baseline or it is reported as a negative result. It is not
   used merely because the data has many sections.
3. **Discover-then-project.** Clusters are discovered on a clean, balanced,
   high-completeness subset, then validated by projection onto the full cohort.
   The **headline claim is the Stage-3 full-sample result**; Stage 1 is the
   clean discovery that motivates it. This pre-empts the "you only analysed the
   easy patients" critique.

## 5. Methods

Status legend: **[locked]** committed · **[proposed]** recommended, awaiting
the `methodology-v1` tag · **[open]** see §8.

| aspect | decision |
|---|---|
| **Filter API** | **[locked]** Two composable functions in `face_common.filters`: `filter_variables` and `filter_patients` (both visit-scoped). `select_v0_anchor` composes them. Patients keyed on `patient_uid`. |
| **Feature set** | **[proposed]** Informative core (67) for Stages 1/3/4; add PARTIAL variables at Stage 2. Hold out `egf`/GAF + FAST/WHODAS (outcomes). |
| **Discovery sample** | **[proposed]** Balanced **198 / cohort** at ≥ 90% completeness floor (DR-limited), random-sampled above the floor (not the strict top-N, to avoid extreme-tail bias). Frozen to `results/discovery_set.csv`. |
| **Standardization** | **[open]** z-score continuous features on the pooled discovery set; Gower handles mixed types natively for the hierarchical path. (Within-cohort vs pooled — see Q-std, §8.) |
| **Stage-1 clustering** | **[proposed]** Hierarchical (Ward) on **Gower distance** (mixed-type, NaN-native, interpretable centres). Classical baseline PCA + k-means run alongside; LCA as model-based robustness. |
| **Stage-2 clustering** | **[proposed]** **SNF** (per-domain similarity networks) + **Leiden** community detection — handles block-missingness via per-network diffusion. Hyperparameters (σ, K, T, resolution) chosen by a metasnf-style sweep then committed *before* the primary run. Robustness: ANF, hierarchical+Gower on the union (partial distances). |
| **Cluster count k** | **[proposed]** Pre-registered range k ∈ {2,3,4,5,6}; Stage-1 primary k = silhouette-optimal on Gower; Stage-2 = Leiden modularity. Report all k in supplement. |
| **Imputation** | **[proposed]** **None** for the primary path (Gower / SNF partial distances). MICE (m = 20) as a sensitivity analysis. |
| **Stage-3 projection** | **[proposed]** Assign every V0 patient to the nearest Stage-1 cluster (centroid / trained classifier); recompute ARI + composition on the full 9,013. |
| **Stability (Stage 4)** | **[proposed]** Re-cluster V1–V4 with the V0 feature set; match by `patient_uid`; report (a) ARI per visit, (b) per-patient transition Sankey, (c) Hennig (2007) bootstrap Jaccard ×1000 — clusters with mean Jaccard < 0.5 flagged *dissolved*. |
| **Validation** | **[proposed]** Site-based holdout = sites **{10, 13}** (762 patients, all 3 cohorts); train on the other 19. Partition frozen to `results/site_partition.json` before any primary run. |
| **Cluster–DSM-5 metrics** | **[locked]** ARI (primary), NMI, confusion matrix, per-cluster DSM-5 composition, per-arm cluster spread. Cluster-defining features by Cohen's d. |
| **Pre-registration** | **[locked]** Git-tag only. Methodology commits tagged `methodology-vN`; the primary run tagged `analysis-v0-primary` immediately before execution. |

**Dropped from the original plan** (see C1/C2, §11): the *consensus-of-three
(hierarchical+GMM+HDBSCAN)* rule — GMM's Gaussian likelihood is ill-posed on
the ~40% binary/ordinal features, and consensus clusters lack the per-feature
interpretability the manuscript needs.

## 6. Ablations & negative controls

Run after the Stage-1 primary, before interpreting any cluster as real.

**Ablations** (what drives the clusters):

| axis | levels | metric | tells us |
|---|---|---|---|
| completeness floor (sample) | 95% (129×3) · **90% (198×3)** · 85% (248×3) | label stability of the nested core + ARI on overlap | robust to *who* we include? |
| feature domain (features) | leave-one-domain-out ×8 + domain-only ×8 | ARI vs full | which variable domains *create* the clusters |
| cohort balance (sample) | balanced 1:1:1 vs natural 81:28:7 | ARI + composition | the BP-protocol-bias check |
| method (algorithm) | hier+Gower vs PCA+k-means vs SNF | pairwise ARI | structure method-invariant? |

**Negative controls** (is the structure real?):

- **Missingness-only**: cluster the binary "was this variable observed" matrix.
  If it recovers the phenotype clusters, our clusters are missingness artifacts.
- **Permutation null**: shuffle feature values within cohort, re-cluster; the
  real cluster–DSM-5 ARI must sit outside the permuted distribution.
- **Noise injection**: replace features with marginal-matched noise; clusters
  should degrade.

## 7. Phased plan

### Phase 0 — Harmonization — **DONE**
- [x] Dictionary parsed; 348 feature variables PASS the audit (0 FAIL, 45 WARN).
- [x] Harmonization registry (31 custom rules + identity-cast fallback).
- [x] Interactive missingness QA report (`reports/qa_missingness.html`).

### Phase 1 — Filter library + patient identity — **DONE**
- [x] `face_common/filters.py`: `filter_variables`, `filter_patients`,
      `V0Anchor`, `select_v0_anchor`; reports carry per-element completeness.
- [x] **`patient_uid`** patient key (C5); cross-cohort collision regression
      tests. 26/26 unit tests pass.
- [x] `scripts/v0_anchor.py` → `results/v0_anchor_*` artifacts.

### Phase 2 — Methodology + feasibility — **DONE (sign-off pending)**
- [x] 7×7 threshold sweep + feature-content analysis
      (`reports/phase2_sensitivity.html`).
- [x] Cohort-imbalance / BP-protocol-bias diagnostic; site-partition inventory;
      imputation cost analysis.
- [x] 2024–2026 methods literature scan (`results/phase2_method_scan.md`).
- [x] Discovery-set feasibility (`results/phase2b_feasibility.json`): informative
      core, floor↔N table, selection-bias SMDs.
- [ ] **Tag `methodology-v1`** once §5 [proposed] rows + §8 are signed off.

### Phase 3 — Stage 1: discovery clustering — **NEXT**
- [ ] Freeze the discovery set (198×3 @ 90%) → `results/discovery_set.csv`.
- [ ] `face_common/clustering.py`: thin wrappers (hierarchical+Gower, PCA+kmeans,
      LCA) returning `(labels, params, diagnostics)`.
- [ ] `scripts/cluster_v0.py`: discovery-set → clustering → `results/v0_clusters.csv`.
- [ ] Ablations (§6) + negative controls (§6).
- [ ] `scripts/cluster_v0_report.py`: HTML — confusion matrix, ARI vs DSM-5,
      per-cluster composition, cluster-defining features (Cohen's d).

### Phase 4 — Stage 2: PARTIAL-variable refinement
- [ ] Add 2-cohort PARTIAL variables; SNF (per-domain) + Leiden.
- [ ] Per-variable **trans-diagnostic information gain** vs the Stage-1 core.

### Phase 5 — Stage 3: generalize to the full cohort (**headline**)
- [ ] Project the Stage-1 model onto all 9,013 V0 patients.
- [ ] Full-sample ARI vs DSM-5 + composition; confirm clusters survive outside
      the discovery set.

### Phase 6 — Stage 4: longitudinal stability
- [ ] `scripts/cluster_stability.py`: re-cluster V1–V4, match by `patient_uid`.
- [ ] ARI per visit + transition Sankey + bootstrap Jaccard.
- [ ] DR excluded from V3 metrics (n=3) — descriptive only.

### Phase 7 — Outcome prediction
- [ ] Inventory functional outcomes (GAF/`egf`, FAST `fast28`/`fast30`) +
      per-visit completeness.
- [ ] Mixed-effects: outcome ~ V0_cluster + arm + age + sex + (1|site) +
      (1|patient_uid).
- [ ] Treatment-response analysis — *blocked on TRAITEMENTS parsing*.

### Phase 8 — External validation
- [ ] Candidate cohorts: PRISM, B-SNIP, ENIGMA, UK Biobank MH. Verify variable
      overlap with the V0 feature set; replicate clustering.

### Phase 9 — Manuscript
- [ ] Figure inventory locked; methods traced to git tags; code frozen at the
      submission tag.

## 8. Open decisions (resolve before the phase that needs them)

- **Q-std (Phase 3).** Standardize continuous features within-cohort or pooled?
  Pooled is simpler; within-cohort removes cohort-mean effects but can erase
  real cross-cohort differences. *Lean: pooled, with within-cohort as a
  sensitivity.*
- **Q-site (Phase 2/3).** `siteid_city` is currently raw numeric `siteid`
  (no city lookup). Holdout {10,13} is by raw siteid — acceptable, but register
  the real SITEID→city map if available.
- **Q-balance (Phase 3).** Confirm balanced 198×3 as the discovery primary vs a
  natural-proportion discovery (with the balanced version as sensitivity).
- **Q-protocol (Phase 3).** Inventory FACE protocol-mandated assessments;
  report a sensitivity excluding them so clusters don't echo recruitment design.
- **Q-percohort (optional).** Add a per-cohort completeness floor to
  `filter_variables` (a variable must clear the floor in *each* cohort, not just
  pooled) to catch systematically missing-by-cohort variables.

## 9. Risks & known limitations

- **DR V3 cliff (n=3).** Any V3-dependent DR analysis is meaningless; exclude or
  mark descriptive-only.
- **Cohort imbalance (81/28/7).** Biases unsupervised structure toward BP;
  addressed by balanced discovery + the cohort-balance ablation.
- **Selection bias from completeness floor.** High-completeness patients are
  mildly less severe (SZ YMRS SMD −0.25; lower CGI; lower DR BMI; most
  |SMD| < 0.2). Mitigated by discover-then-project; report the SMD table.
- **Harmonization residual error.** 45 audit WARNs are cohort value-set
  divergences; sensitivity = repeat Stage 1 excluding flagged rows.
- **TRAITEMENTS unparsed.** Blocks treatment-response (Phase 7 / Q4).

## 10. Decision ledger

| id | decision | status |
|---|---|---|
| filter API + `patient_uid` | composable filters keyed on patient_uid | **locked** |
| feature set | informative core (67); PARTIAL added at Stage 2 | proposed |
| discovery sample | balanced 198×3 @ ≥90% completeness | proposed |
| Stage-1 method | hierarchical + Gower (primary); PCA+kmeans, LCA robustness | proposed |
| Stage-2 method | SNF + Leiden (block-missingness) | proposed |
| imputation | none for primary (partial distances); MICE sensitivity | proposed |
| stability | ARI + transition Sankey + Hennig bootstrap Jaccard | proposed |
| validation | discover-then-project + site holdout {10,13} | proposed |
| cluster count | k ∈ {2..6}, silhouette/modularity-optimal | proposed |
| leakage | `egf`/GAF + FAST/WHODAS held out as outcomes | **locked** |
| consensus-of-three | **dropped** (GMM ill-posed on binaries) | **locked** |

## 11. Course-correction log

Chronological pre-registration trail — append, never rewrite.

**C0 — Initial plan.** V0-anchored common-feature matrix at 75/75; consensus of
hierarchical+GMM+HDBSCAN; re-cluster stability; site holdout.

**C1 — Threshold + method revision (Phase 2 sensitivity,
`reports/phase2_sensitivity.html`).** Found: 0.75 default is biology-heavy and
discards 100% of neuropsychology; pooled selection is BP-protocol biased (8/73
features flip under balancing); 12 near-constant binaries carry ~0 signal; GMM
ill-posed on 40/73 binary/ordinal features. *Correction:* drop
consensus-of-three; single interpretable primary; add cohort-balanced
sensitivity.

**C2 — Method re-grounding (graph discussion + `results/phase2_method_scan.md`).**
The correct justification for graphs is **block-structured missingness**, not
multi-modality. *Correction:* SNF + Leiden is the Stage-2 primary; classical
methods are the baseline it must beat; deep-learning methods (GNN/VAE/EHR
foundation models) not worth the cost at N≈6K, p≈60 — held for Phase 8+.

**C3 — Staged discovery design.** Replace the single-shot pipeline with the
4-stage discover → refine → generalize → longitudinal design (§4). Core-first
removes data-availability bias; the graph is deferred until block-missingness
actually appears.

**C4 — Discovery-set feasibility (Phase 2b, `results/phase2b_feasibility.json`).**
"200/cohort at ≥95%" is infeasible (0 patients on the full READY core; 129/cohort
on the informative core). *Corrections:* cluster on the **67-feature informative
core**; discovery floor **90% → 198/cohort** (85%→248, 95%→129 as ablations);
add the leakage guard; quantify and accept a modest severity selection bias.

**C5 — Patient-identity bug fix.** V0 has 9,013 rows but only 7,966 unique
`usubjid_patients` — **970 ids reused across cohorts**. The filters / anchor /
wide-pivot keyed on `usubjid_patients` alone → cross-cohort contamination and
wide-format row merging. *Correction:* loader emits **`patient_uid =
cohort::usubjid_patients`**; all patient-level ops key on it; regression tests
added. Corrected 75/75 anchor counts: 6,896 patients (BP 4,998 / SZ 1,536 /
DR 362). Phase-2b floor counts (per-cohort row counts) were unaffected.

## 12. Glossary

- **V0 / V1…V4** — baseline (inclusion) / annual follow-up visits.
- **`patient_uid`** — globally-unique patient key `cohort::usubjid_patients`.
- **`arm`** — DSM-5 sub-diagnosis text label; held out as a label.
- **`cohort`** — primary DSM-5 category (BP/SZ/DR); held out.
- **Informative core** — the 67-feature Stage-1 discovery substrate (§3).
- **Discovery set** — balanced high-completeness subsample (198×3) for Stage-1
  discovery; projected onto the full cohort at Stage 3.
- **ARI (Adjusted Rand Index)** — agreement between two groupings of the same
  patients, corrected for chance. 1.0 = identical; ~0 = chance-level; can be
  slightly negative. Does not require matching labels or equal cluster counts —
  it compares *co-membership*. We want **low** ARI vs DSM-5 (trans-diagnostic)
  but **high** ARI across visits (stable).
- **NMI** — Normalized Mutual Information; an information-theoretic agreement
  measure, reported alongside ARI.
- **Silhouette** — per-point cluster-cohesion vs separation in [−1, 1]; used to
  pick k for hierarchical.
- **Gower distance** — mixed-type pairwise distance (continuous + binary +
  ordinal) that handles NaN via partial comparisons.
- **SNF (Similarity Network Fusion)** — Wang et al. 2014; fuses per-domain
  patient-similarity networks via cross-diffusion; handles block-missingness.
- **Leiden** — community-detection algorithm (Traag 2019); finds clusters by
  modularity, no k upfront.
- **LCA** — Latent Class Analysis; probabilistic mixture model for categorical
  data.
- **Bootstrap Jaccard (Hennig 2007)** — cluster-wise stability: mean Jaccard
  overlap across resamples; ≥ 0.85 highly stable, < 0.5 dissolved.
- **Cohen's d** — standardized mean difference; used to rank cluster-defining
  features.
