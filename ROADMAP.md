# ROADMAP — Trans-diagnostic Phenotypes Across BP / SZ / DR (FACE)

> Single source of truth for *what* we are doing and *why*. Sections 1–9
> describe the current design; §9–10 the plan and refactor status; §14 the
> course-correction log (append-only — our de-facto pre-registration trail).
> **Update this file whenever a methodological decision is made.**

---

## 1. Research question & rationale

The FACE cohort spans three DSM-5 categories — Bipolar (BP), Schizophrenia-
spectrum (SZ), Depressive disorder (DR) — measured with a common clinical
dictionary. DSM-5 boundaries under-fit the biological/functional heterogeneity
of severe mental illness and obscure features shared *across* diagnoses.

1. **Primary (trans-diagnostic).** Does unsupervised clustering at baseline
   (V0) reveal patient subgroups whose composition **cuts across** DSM-5?
2. **Stability (longitudinal).** Are V0 clusters **temporally coherent** over
   V1–V4, or do patients drift in a way that itself defines trajectories?
3. **Prediction.** Do V0 clusters predict **functional outcome** (later visits)
   over and above the DSM-5 label?
4. **Score (follow-up).** Can a parsimonious **FACE Score** summarise the
   dominant axis and predict outcome / treatment response?

## 2. Pre-registered hypotheses

- **H1 (trans-diagnostic).** ARI between V0 clusters and DSM-5 `arm` is well
  below 1.0 (candidate < 0.4), and ≥1 cluster mixes ≥2 DSM-5 categories at
  ≥15% each. Paired with a stability check so low ARI reads as real structure,
  not noise.
- **H2 (stability).** ARI between V0 clusters and independently re-derived
  V1 clusters (matched by `patient_uid`) exceeds a floor (candidate > 0.5);
  coherence decays V0→V4 but does not collapse.
- **H3 (prediction).** V0 cluster membership predicts later functional outcome
  in a mixed-effects model adjusting for `arm`, age, sex, site.

> ⚠️ **Leakage guard.** `egf`/GAF and any FAST/WHODAS items are outcomes — held
> **out of the clustering feature matrix** (clustering on an outcome then
> "predicting" it is circular).

## 3. Target narrative & paper framing

The clinical collaborator's concept deck (`Idea paper classification…`) frames
the paper as *"Clinical Phenotypes Across Diagnostic Boundaries — the FACE
experience."* Its BLUF is our north star, but **every claim is a target to be
earned, not a result** — and one is likely sign-inverted (see ⚠️). Treat this
table as the contract between the clinical and computational sides:

| deck claim | status | where we earn / verify it |
|---|---|---|
| young-onset transdiagnostic phenotype spanning cohorts | recover | Phase 3 (3-cohort recovery) |
| **"metabolic-impulsive" direction of that phenotype** | **⚠️ verify sign** | Phase 3 — the sister's *corrected* analysis shows the matching cluster (their C3) is **metabolically HEALTHY / lean**, the opposite of the pre-bug "metabolic syndrome" reading. Confirm the actual direction of waist/TG/BIS-10 in our recovered cluster before any headline. |
| 7 clinical axes (age-at-onset, chronicity, metabolic, impulsivity, family load, functioning, comorbidity) define the clusters | hypothesis | Phase 3 — per-cluster Cohen's d, don't assume |
| superiority over DSM/ICD | thesis, not shown | Phase 5 — cluster-vs-DSM head-to-head on outcome |
| FACE Score (waist, TG, age-onset, illness-duration, BIS-10, family load; AUC 0.70) | candidate | Phase 6 — leakage-safe, "enrichment not triage", corrected direction |
| longitudinal / temporal coherence / cluster stability | **our core contribution** | Phases 4–6 |

**Scope.** The deck is 4-cohort (incl. ASP). Our longitudinal data is
3-cohort (no ASP follow-up). The joint structure: **4-cohort cross-sectional
discovery (sister engine) + 3-cohort longitudinal temporal-coherence validation
(our data) + FACE Score.** Our recovery of the 6 non-ASP phenotypes is the
bridge between the two halves. (The ASP cluster sat far from BP/SZ/DR and is
not part of the longitudinal story.)

**Honesty caveats inherited from the sister project:** a rank-biserial sign bug
once inverted every cluster interpretation; a leakage bug once inflated panel
AUCs to 0.99. Both are fixed in the engine — but we re-verify directions and use
the leakage-safe protocol, and we do not call panels "biomarkers."

## 4. Data & key concepts

| cohort | DSM-5 | V0 patients |
|---|---|---|
| BP | Bipolar | 6,252 |
| SZ | Schizophrenia-spectrum | 2,209 |
| DR | Depressive disorder | 552 |
| **total** | | **9,013** |

Visits V0 (baseline) → V1–V4 (annual). **DR collapses to 3 patients at V3** —
a structural cliff; exclude DR from V3 metrics.

- **`patient_uid` = `cohort::usubjid_patients`** — globally-unique key
  (`usubjid_patients` is reused across cohorts; 970 collisions). All
  patient-level ops key on it. The sister engine independently uses the same
  `(cohort, patient_id)` key.
- **Common-variables dictionary** (`face-common-vars.xlsx`): 379 rows; each
  carries `section` (13 clinical blocks), `dtype`, value-set, readiness.
  **This is our feature source.** READY (130) / PARTIAL (221) / NOT USABLE.
- **Informative core (67 features)** — READY minus near-constants minus
  column-completeness < 70%; the candidate clustering feature set.
- **No imputation.** Verified in the engine: matrix keeps NaN; masked
  pairwise-complete similarity; too-sparse patients dropped (min_coverage);
  missingness carried as mask columns. Optional KNN/MICE imputers exist but are
  off by default.
- **Identifiers (never clustered on):** `patient_uid`, `usubjid_patients`,
  `cohort`, `arm`, `visit`, `visitnum`.

## 5. Architecture — our pipeline + reused engine

We **merged the sister `face_stratification` project into this repo** and adopt
its modelling *algorithms*, but **drive them with our 3-cohort common-variables
data**. The sister's 4-cohort clusters are the comparison reference.

```
face-common-vars.xlsx + data/ (BP·SZ·DR, V0–V4)
   │  face_common: harmonize per visit → patient × feature matrix (patient_uid)
   ▼
   adapter → engine HarmonizedDataset(X, schema)   schema from OUR dictionary:
   │                                                block = `section`,
   │                                                metric = by `dtype`
   ▼  reuse engine ALGORITHMS (no imputation):
   │   masked similarity → multipartite spectral embedding → consensus KMeans
   │   → validation suite (ARI/Cramér's V/entropy/LOCO/bootstrap/permutation)
   ▼
   clusters per visit → recover phenotypes (vs sister 4-cohort reference)
                       → temporal coherence V0→V4 → FACE Score
```

**Division of labour:** features + data + longitudinal layer = ours; embedding +
clustering + validation algorithms = reused sister engine.

## 6. Methods

Status: **[locked]** committed · **[proposed]** awaiting `methodology-v1` ·
**[open]** see §11.

| aspect | decision |
|---|---|
| **Feature source** | **[locked]** Our common-variables harmonization (`face_common`), 3 cohorts. Informative core (67) as the candidate set. Hold out `egf`/GAF + FAST/WHODAS. |
| **Schema** | **[proposed]** Built from the dictionary: **block = `section`**, **metric = by `dtype`** (continuous biology/cognition → euclidean; ordinal scale profiles → cosine; mixed/binary → Gower). One YAML, our names, 3 cohorts. |
| **Adapter** | **[to build]** `face_common` matrix → engine `HarmonizedDataset(X, schema)`; the single new glue module. |
| **Embedding** | **[proposed]** Reuse engine `MultipartiteSpectral` — per coverage-partition spectral blocks (`bp+dr+sz`, `bp+dr`, `bp+sz`, `dr+sz` + single-cohort; no ASP), masked similarity, no imputation. |
| **Clustering** | **[proposed]** Reuse engine consensus KMeans with composite k-selection. Report composite-optimal k *and* the k matched to the sister reference. |
| **Imputation** | **[locked]** None (masked similarity). KNN/MICE only as sensitivity. |
| **Validation vs DSM** | **[proposed]** ARI, NMI, Cramér's V, per-cluster cohort entropy, transdiagnostic score, confusion matrix; cluster-defining features by Cohen's d. |
| **Stability** | **[proposed]** ARI(V0↔Vk) + per-patient transition Sankey + bootstrap Jaccard (Hennig, dissolve < 0.5) + **LOCO** (drop each cohort). |
| **Recovery check** | **[proposed]** ARI + semantic signature match of our 3-cohort clusters vs the sister 4-cohort reference (the 6 non-ASP phenotypes). |
| **Reference** | **[locked]** Sister 4-cohort clusters reproduced exactly from the saved embedding (`results/v0_clusters_anchor.csv`, b2). |

## 7. Ablations & negative controls

- **Floor ablation** (sample): 85/90/95% completeness — robustness to which
  patients enter.
- **Feature-domain ablation** (features): leave-one-`section`-out + domain-only,
  ARI vs full — which clinical domains drive the clusters.
- **Cohort-balance ablation**: balanced vs natural proportions — the BP-
  domination check.
- **Negative controls**: missingness-only clustering (must NOT recover the
  phenotype clusters); permutation null (real ARI vs DSM outside the shuffled
  distribution); noise injection.

## 8. Repository structure

Clean split: `src/` = our development base; `archive/` = vendored copied sister
code (reused by import, never developed).

```
face-common-bp-sz-dr/
├── src/face_common/        OUR pipeline — the only code we develop (loader, rules, filters)
├── archive/                VENDORED copied sister code (do not edit)
│   ├── face_stratification/  the reused engine (graph, models, clustering, stage_c, evaluation)
│   ├── face_rlvr/            engine patient extractors + glossary loader
│   ├── data/                 sister 4-cohort V0 CSVs (BP/SZ/DR/ASP)
│   ├── scripts/ notebooks/ tests_face_stratification/ docs/ output/
│   └── README_sister.md
├── config/                 engine config (feature schema + glossary; kept at repo root for parents[3] resolution)
├── data/                   OUR BP/SZ/DR V0–V4 CSVs + data/external (engine reference artifacts)
├── scripts/                OUR scripts only (verify, audit, qa, v0_anchor, phase2*, reproduce_v0_clusters)
├── tests/                  OUR tests (test_filters.py)
├── results/ reports/       our outputs
├── ROADMAP.md CLAUDE.md DATA.md README.md  our docs
└── pyproject.toml          packages: src/face_common + archive engine; pytest pythonpath = [src, archive]
```

## 9. Phased plan

### Phase 0 — Harmonization — **DONE**
- [x] Dictionary parsed; 348 feature variables PASS the audit; QA report.

### Phase 1 — Filter library + patient identity — **DONE**
- [x] `face_common/filters.py` + `patient_uid` fix + 26 tests.

### Phase 2 — Methodology, feasibility, merge — **DONE**
- [x] Threshold sweep + sensitivity report; informative core (67); discovery
      floor 90% → 198/cohort.
- [x] Literature scan; merged sister engine into this repo; verified data joins
      100% (their BP/SZ/DR ⊆ ours); reproduced their 4-cohort clusters exactly
      (b2 → `results/v0_clusters_anchor.csv`); confirmed **no imputation** in the
      engine.

### Phase 3 — V0 trans-diagnostic clustering — **DONE (v1), with a methodology correction**
- [x] `src/face_common/schema_gen.py` — dictionary→`FeatureSchema` generator
      (`section`→block, `canonical_name`→feature id, `dtype`→`FeatureType`,
      source-column presence→`cohorts`).
- [x] `src/face_common/adapter.py` — `to_harmonized_dataset(...)`: V0 long frame →
      engine `HarmonizedDataset` (no imputation), with `normalize_for_embedding`
      (robust per-feature z), `residualize_features` (regress out covariates), a
      `sections` filter (`CLINICAL_SECTIONS`) and confound `exclude`. 22 tests.
- [x] `scripts/cluster_v0.py` (embed + cluster) and `scripts/cluster_v0_profile.py`
      (k-sweep, UMAP, engine enrichment naming → `reports/cluster_v0.html`).
- **Confound trace — the headline lesson.** Clustering on the full numeric
  common-variable set was an artifact ladder: (1) a `brthdtc` date encoded as
  ~1e17 dominated everything; (2) after fixing scale, raw labs/anthropometry
  dominated; (3) after robust z-scoring, the clusters were a **sex×age
  stratification** (cluster↔sex ARI 0.32 > cluster↔cohort 0.19); (4) the sex/age
  signal was carried almost entirely by **physical-comorbidity flags**
  (`*_mhoccur`: lupus→F, MI→older…). The principled configuration:
  **clinical sections only, age/sex-residualized, robust-scaled, `*_mhoccur`
  excluded** (129 features). Earlier "ARI 0.96 / 0.31" numbers were the date
  artifact and are retracted.
- **Result (k=6, 9,013 patients):** the confound is gone (cluster↔sex ARI
  **0.005**, ↔age 0.008) and six **trans-diagnostic symptom phenotypes** emerge
  that cut across BP/SZ/DR (cluster↔cohort ARI **0.024**): childhood
  maltreatment (CTQ↑), depression-severity + poor sleep (MADRS/PSQI↑, **DR-
  enriched** → face-validity), suicidality (C-SSRS), and a denial / response-
  style axis. Bootstrap mean pairwise ARI **0.89**. This **serves the
  cut-across-DSM goal** (§1) and, by construction, does **not** reproduce the
  sister's diagnosis-aligned clusters (ARI vs ref **0.03**).
- **Open fork (needs a call):** (a) *trans-diagnostic discovery* — keep the
  current demographics/comorbidity-free phenotypes; vs (b) *sister recovery* —
  retain diagnosis signal (clusters then partly recapitulate DSM + demographics).
- [ ] Scrutinise the "denial" axis (symptom-minimization response style, not
      psychopathology?); principled k-selection; ablations (symptom-only,
      READY-only); negative controls (§7).

### Phase 4 — Longitudinal coherence (our core)
- [ ] Harmonize V1–V4 with the same schema; assign/re-cluster; ARI(V0↔Vk) +
      transition Sankey + bootstrap/LOCO stability. DR excluded at V3.

### Phase 5 — Outcome prediction & DSM head-to-head
- [ ] Functional outcome (GAF/FAST) ~ V0_cluster + arm + age + sex + (1|site);
      show clusters add information *over* DSM (the "superiority" test).

### Phase 6 — FACE Score
- [ ] Leakage-safe parsimonious score (deck's 6 candidates), corrected
      direction, "enrichment not triage"; calibration; outcome prediction.

### Phase 7 — External validation
- [ ] Candidate cohorts (PRISM/B-SNIP/ENIGMA/UKBB-MH); replicate.

### Phase 8 — Manuscript
- [ ] Figures locked; methods traced to git tags; code frozen at submission.

## 10. Refactor status & remaining steps

The merge + clean reorganization are **done** (Reading B: vendor the whole sister
tree, no engine surgery).

- [x] `src/face_common` is the sole development base; `face_common` imports from
      `src/`.
- [x] **All copied sister code in `archive/`** — `face_stratification` (engine),
      `face_rlvr`, sister scripts/notebooks/tests/docs/output, and the 4-cohort
      `data/*.csv`. Kept importable (pytest `pythonpath = [src, archive]`;
      scripts insert both).
- [x] `config/` kept at repo root so the engine's `parents[3]/config` schema
      resolution still works from `archive/`.
- [x] `pyproject` packages = `src/face_common` + the archived engine;
      `.gitignore` ignores `.env*`/`output/`; `.env` (secrets) never copied.
- [x] Docs rewritten for the clean layout (CLAUDE.md, README.md, ROADMAP §8).
- [x] Verified: 40 unit tests pass; `verify.py` runs end-to-end; both
      `face_common` (src) and `face_stratification` (archive) import.
- [x] **Dictionary→schema generator + `face_common → HarmonizedDataset` adapter**
      built (`src/face_common/schema_gen.py`, `adapter.py`) — the glue between our
      pipeline and the engine. Exposed via `face_common.{to_harmonized_dataset,
      build_feature_schema}`.
- [x] `scripts/cluster_v0.py` — our 3-cohort V0 matrix → engine → clusters
      (`results/cluster_v0_*`). See Phase 3 above for the first recovery result.

Principle: develop only in `src/face_common`; reuse the engine from `archive/`
by import; never edit vendored code. To run the engine's heavy paths:
`pip install -e ".[stratification]"`.

## 11. Open decisions

- **Schema metric mapping** — exact `dtype`→metric rule (esp. ordinal scales:
  cosine vs Gower) and `section`→block granularity (merge tiny sections?).
- **k-selection** — composite score vs silhouette; how to match k to the
  sister reference for the recovery comparison.
- **Standardization** — within-cohort vs pooled robust z-score.
- **Site holdout** — `siteid` partition for cross-site validation.

## 12. Risks & limitations

- **DR V3 cliff (n=3)** — exclude from V3.
- **Cohort imbalance (81/28/7)** — balanced discovery + cohort-balance ablation.
- **Different harmonization than the sister's 184-feature schema** — our
  clusters need not match theirs exactly; recovery is by *semantics* (ARI +
  signature), which is also a robustness result.
- **Metabolic-direction sign** (§3 ⚠️) — verify before any headline.
- **Selection bias from completeness floor** — high-completeness patients
  mildly less severe; mitigate by discover-then-project; report SMDs.

## 13. Decision ledger

| id | decision | status |
|---|---|---|
| feature source | our common-variables (3 cohorts) | **locked** |
| schema | block=`section`, metric=`dtype` | proposed |
| engine | reuse sister algorithms (multipartite spectral + consensus KMeans + validation) | **locked** |
| imputation | none (masked similarity) | **locked** |
| reference | sister 4-cohort clusters (reproduced, b2) | **locked** |
| stability | ARI + transition Sankey + bootstrap Jaccard + LOCO | proposed |
| leakage | hold `egf`/GAF + FAST out of features | **locked** |
| cruft | quarantine sister harmonizer/`face_rlvr`/ASP/RLVR under `reference/` | **locked** (scripts/notebooks done; code pending §10) |
| metabolic direction | verify sign before headline | **locked (must-verify)** |

## 14. Course-correction log

Append-only pre-registration trail.

**C0 — Initial plan.** V0-anchored common-feature matrix at 75/75; consensus of
hierarchical+GMM+HDBSCAN; site holdout.

**C1 — Threshold + method revision (Phase 2 sensitivity).** 0.75 default is
biology-heavy and drops all neuropsych; pooled selection BP-protocol biased;
12 near-constant binaries; GMM ill-posed on binaries. → drop consensus-of-three;
single interpretable primary; cohort-balanced sensitivity.

**C2 — Method re-grounding (graph + literature scan).** Graphs justified by
**block-structured missingness**, not multi-modality. → graph/diffusion primary;
classical baseline must be beaten; deep learning deferred.

**C3 — Staged discovery design.** discover → refine → generalize → longitudinal;
core-first removes data-availability bias.

**C4 — Discovery-set feasibility (Phase 2b).** "200/cohort at ≥95%" infeasible;
use the **67-feature informative core**; floor **90% → 198/cohort**; leakage
guard added; modest severity selection bias accepted.

**C5 — Patient-identity bug fix.** `usubjid_patients` reused across cohorts (970
collisions). Added `patient_uid = cohort::usubjid_patients`; all patient-level
ops key on it; regression tests.

**C6 — Merge with the sister `face_stratification` project.** Same FACE freeze
(their BP/SZ/DR ⊆ ours; 100% join). Adopted their **modelling engine**
(multipartite spectral + consensus clustering + validation); verified **no
imputation**; reproduced their 4-cohort clusters exactly from the saved
embedding (b2). Decided on a **joint publication**: 4-cohort cross-sectional
discovery + our 3-cohort longitudinal validation + FACE Score.

**C7 — Our-features architecture + quarantine.** Decision: the clustering is
driven by **our common-variables pipeline** (3 pathologies; `section`=block,
`dtype`=metric), reusing only the engine *algorithms*; the sister's 4-cohort
clusters are the recovery *reference*. Off-path sister code is **quarantined
under `reference/`** (scripts + notebooks moved; harmonizer/`face_rlvr`/ASP/data
move pending the dataclass extraction, §10). Also folded in the clinical deck as
the §3 paper framing, flagging the **metabolic-direction sign** as a must-verify.

## 15. Glossary

- **`patient_uid`** — `cohort::usubjid_patients`, globally-unique patient key.
- **Common-variables dictionary** — `face-common-vars.xlsx`; our feature source.
- **Informative core** — 67-feature candidate clustering set.
- **HarmonizedDataset** — the engine's data contract: a patient × feature matrix
  + schema, MultiIndexed by `(cohort, patient_id)`.
- **Multipartite spectral** — the reused embedding: per cohort-overlap-partition
  spectral blocks concatenated; masked similarity, no imputation.
- **ARI / NMI** — chance-corrected agreement between two groupings (low vs DSM =
  trans-diagnostic; high across visits = stable).
- **Cramér's V / cohort entropy / transdiagnostic score** — DSM-alignment vs
  cohort-mixing metrics from the engine's validation suite.
- **Bootstrap Jaccard (Hennig)** — cluster-wise stability; < 0.5 dissolved.
- **Reference** — the sister 4-cohort clusters, reproduced exactly (b2),
  against which we check semantic recovery.
