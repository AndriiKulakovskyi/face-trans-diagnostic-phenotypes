# LABBOOK — FACE trans-diagnostic clustering (research notebook)

Chronological trace of the work — what we did, what we observed, what we decided
and **why** — so the research is reproducible and the paper can be written from
first principles. Complements:
- **FINDINGS.md** — distilled, paper-ready results.
- **ROADMAP.md** — the forward plan, hypotheses, paper framing.

Dates: entries before 2026-05-22 are reconstructed by phase (undated); work on
2026-05-22 onward is dated.

---

## E0 · Goal & setup
- **Question:** discover clinical phenotypes that cut **across** DSM-5 (Bipolar /
  Schizophrenia / Depression) and are temporally coherent V0→V4.
- **Design:** a *merged* project — our 3-cohort common-variables longitudinal
  pipeline is the **feature source**; the vendored sister `face_stratification`
  engine (masked-similarity → multipartite-spectral embedding → consensus
  clustering) is **reused, not developed**. The sister's 4-cohort clusters are a
  **comparison reference** (their ASP/autism cluster is out of scope — no
  longitudinal data).
- **Data facts:** 9,013 V0 patients (BP 6,252 / SZ 2,209 / DR 552).
  `patient_uid = cohort::usubjid_patients` (usubjid collides across cohorts,
  970 shared ids — always key on `patient_uid`). **No imputation anywhere**
  (masked pairwise-complete similarity).

## E1 · Harmonization, filters, Phase-2 feasibility (prior phases)
- Dictionary (`face-common-vars.xlsx`, 379 rows) → unified patient×feature
  matrix; **348/348 feature variables pass the audit**; QA missingness report.
- `trans_diag.filters` (variable/patient completeness, V0 anchor) + the
  `patient_uid` collision fix (regression-tested).
- Phase-2 sweeps: informative core ≈ 67 features; discovery floor 90% → ~198
  features/cohort; cohort imbalance + site distribution characterised; **no
  imputation** chosen (masked similarity over KNN/MICE).

## E2 · Merge + reproduce the sister clusters
- Vendored the whole sister tree into `archive/` (import, don't edit). Verified
  our keys join 100% to their labels. **Reproduced their published 7-cluster ×
  cohort contingency exactly** from their saved multipartite embedding
  (`scripts/reproduce_v0_clusters.py` → `results/v0_clusters_anchor.csv`). For
  our BP/SZ/DR the ASP cluster is empty → 6 populated reference clusters.

## E3 · Phase-3 engine bridge
- `src/trans_diag/schema_gen.py` (dictionary → engine `FeatureSchema`) and
  `adapter.py` (`to_harmonized_dataset`: our V0 frame → engine
  `HarmonizedDataset`, no imputation). `scripts/cluster_v0.py` drives the engine.

## E4 · The confound ladder — 2026-05-22 (the key methodological lesson)
Clustering on the full numeric common-variable set repeatedly recovered the
**largest-variance nuisance axis**, peeled back one layer at a time:

| run | config | result | problem |
|---|---|---|---|
| 1 | all 341, raw | bootstrap ARI 0.96, ARI-vs-sister 0.31 | **`brthdtc` date ≈ 1e17** dominated cosine — spurious. **Retracted.** |
| 2 | all 341, robust-scaled | — | clusters = **sex×age strata** (cluster↔sex ARI **0.32** > ↔cohort 0.19) |
| 3 | clinical sections, age/sex-residualized | SZ-pure + DR→BP bridge, stability 0.97 | sex **still** 0.32; drivers = physical comorbidity (`*_mhoccur`) |
| 4 | + `*_mhoccur` excluded (129 feat) | **sex confound gone (0.005)**, ↔age 0.008 | (good) |

- **Observed:** feature std spanned 0.016 → 4.5e17. Cosine is scale-invariant
  *per patient* but **not per feature** → big-magnitude columns dominate.
- **Fix that worked:** clinical sections only, **age/sex-residualized**,
  robust-scaled, `*_mhoccur` (physical comorbidity, which carried the sex/age
  signal: lupus→F, MI→older) excluded, dates/site/IDs dropped.
- **Run-4 result:** six reproducible (bootstrap mean pairwise ARI **0.89**)
  **trans-diagnostic symptom phenotypes** cutting across BP/SZ/DR (cluster↔cohort
  ARI **0.024**): childhood maltreatment (CTQ↑), depression-severity + poor sleep
  (MADRS/PSQI↑, **DR-enriched** → face validity), minimal-suicidality, and a
  **denial/response-style** axis.

## E5 · Item-count weighting discovery — 2026-05-22
- Cosine treats every column as one equal dimension → constructs with many items
  dominate. In the 129-feature clinical set, **SUICIDE = 39 dims (30%)**
  (`isf` 15 + `cssrs` 11 + `ltsg/ltsv` 13); sleep `psqi` 8, trauma `ctq` 8.
- **Conclusion:** the emergent phenotypes are the *most-itemized instruments*,
  not the most clinically important. Must **aggregate items → domain scores**
  before clustering.

## E6 · Scientific fork — 2026-05-22 (decision: A)
Two mutually-exclusive products (diagnosis + demographics are the dominant
variance axes):
- **(A) Trans-diagnostic discovery** — cluster *net of* diagnosis/demographics →
  symptom-dimension phenotypes shared across BP/SZ/DR. *Primary project goal.*
- **(B) Diagnosis-aligned recovery** — keep those axes → clusters recapitulate
  DSM + demographics, resemble the sister's. A concordance check, not discovery.

**Decision: pursue (A).** Therefore matching the sister (ARI-vs-ref) is **not** a
selection criterion; low cohort-ARI is desired. `k=6` so far was only a
placeholder (the sister's non-ASP count) — k must be chosen on internal grounds.

## E7 · Biology re-inclusion + deconfounding method review — 2026-05-22
- **Why biology/constants were excluded in run 4:** a deliberate confound-control
  first pass — labs/vitals/anthropometry are strongly sex/age-dimorphic and were
  the confound source. **But** excluding them throws away the *actionable*
  trans-diagnostic **metabolic axis** (the sister had a DR+SZ metabolic
  partition; the project deck headlines a metabolic axis — direction to verify).
  **Decision: re-include biology, properly deconfounded.**
- **Method review (2025-26), "keep signal / kill confound":**
  - Tier 1 (feature-space): linear residualization → **nonlinear partialling-out
    with cross-fitting** (double-ML) → **ComBat/CovBat/ComBat-GAM** for site.
  - Tier 2 (representation): conditional VAE (works), adversarial (a 2024
    multi-omics benchmark found it *insufficient*), HSIC/independence penalties.
  - Tier 3 (verify): **distance correlation / HSIC** between clusters and
    {age, sex, site} ≈ 0 (nonlinear; stronger than ARI).
  - **Choice:** Tier 1 + Tier 3 — Tier 2 deep deconfounders are ill-suited to our
    **masked no-imputation** data, n≈9k, and interpretability needs (clinical
    paper). Cite the VAE work as a future/sensitivity extension.
- **Decisions locked (2026-05-22):**
  1. **Nonlinear spline age + cross-fit residualization** (rigorous).
  2. **Site left OUT of the main analysis for now**; **ADD ComBat as a
     sensitivity analysis later** ← *remembered (task #43).*
  3. Aggregation = **validated totals → else mean of z-scored items**; biology as
     **clinical composites** (metabolic-syndrome, inflammation, prolactin, …).
  4. k-selection = **consensus + bootstrap stability + gap + interpretability**.
  5. **Start with symptoms + biology** (cognition deferred — availability-confounded).

## E8 · Domain aggregation + nonlinear residualization — 2026-05-22 (done)
- `src/trans_diag/domains.py` — symptom instruments auto-grouped by canonical
  stem (masked mean of robust-z items, min-items threshold); curated biology
  composites with explicit members + directions. 190 items → 72 domains; no
  domain > 1.4% of dims (was 30% for SUICIDE); metabolic_syndrome 90% coverage.
- `residualize_features(spline_df, cross_fit)` — natural-spline age + sex-specific
  curves + K-fold cross-fitting (double-ML partialling-out).

## E9 · Direction-A domain clustering result — 2026-05-22
`scripts/03_cluster_domains.py`: 72 domains → coverage floor 30% (**54 kept**, 18
near-empty dropped incl `cssrs`/`ltsg`/`ltsv`/`mdq`/`cgi`) → spline+cross-fit
residualize on age+sex → robust-z → engine masked-cosine spectral embedding (36-dim,
4 partitions) → stability/PAC/gap/independence k-sweep.

- **Principled k = 5.** Highest bootstrap stability (**ARI 0.972**) and lowest
  consensus **PAC 0.047**; at k≥6 stability falls *and* sex creeps back
  (Cramér's V 0.18→0.24). Gap rises monotonically (not decisive alone).
- **Confound verified removed** (Tier-3): sex Cramér's V **0.041**, age-tertile
  ARI **0.006**, age **dCor 0.117** (small residual), **cohort ARI 0.002** —
  clusters independent of sex, age and diagnosis.
- **Five trans-diagnostic phenotypes** (cohort mix ≈ proportional to sample;
  standardized domain profiles in `reports/cluster_domains.html`):
  0 **metabolic / later-onset** (metabolic_syndrome +0.41σ, later age-of-onset);
  1 **heavy-smoking / hospitalization burden** (smoking +0.76σ, hospitalizations↑,
  low YMRS); 2 **high-functioning / low burden** (EGF/education/QoL↑,
  metabolic −0.89σ, smoking −0.93σ); 3 **manic activation / impulsivity**
  (YMRS +1.33σ, Altman/Mathys/BIS↑, DR ≈ 0); 4 **somatic / medication-burden**
  (somatic +1.53σ, QTc +0.41σ, prolactin +0.37σ). Clusters 3 & 4 are dominated by
  a single strong axis; 0-2 are multivariate.

## E10 · Phenotype profile report — 2026-05-22
- `scripts/cluster_domains_profile.py` → `reports/cluster_domains.html`:
  cluster×domain signature heatmap, UMAP (cluster/cohort), per-cluster enrichment
  bars, medoid vignettes, k-selection figure, independence callout.
- Bug fixed: a CSV→parquet round-trip had coerced `patient_id` int (lost the str
  type), breaking the scores↔embedding join; `03_cluster_domains.py` now writes
  parquet directly and the profile coerces the index defensively.
- **Metabolic axis recovered** as a prominent phenotype (cluster 0 high vs 2 low)
  — the deck's metabolic theme is supported; composite direction is explicit
  (BMI/trig/glucose↑, HDL↓ = higher burden) so no sign-inversion ambiguity.
- Residual to tighten: age dCor 0.117 (consider more spline knots / age²·sex).

---

## E11 · Phase 4 — temporal coherence V0→V4 — 2026-05-22
`scripts/09_longitudinal_coherence.py`. Build the SAME domain scores at every visit
(pooled scaling, per-visit-age spline+cross-fit residualization), assign each
patient-visit to a V0 phenotype, measure persistence.
- **Methodology note (important):** a masked nearest-**centroid** rule could *not*
  reproduce the V0 spectral-embedding clusters (self-ARI **0.024** — centroid vs
  spectral geometry mismatch). Replaced with a **classifier** (HistGradientBoosting,
  NaN-native) trained on V0 domain scores → V0 labels: **shuffled 5-fold accuracy 0.873**
  (k=5, chance 0.20; was 0.842 under un-shuffled CV — see E15b) — phenotypes ARE
  recoverable from domains; rule is valid.
- **Result (now read as a NEGATIVE result):** the discrete clusters do **not**
  persist — ARI(V0↔Vk) **≈0.06–0.07**, persistence **≈37–39%** (V1 n=3782 … V4 n=697),
  barely above chance. Phenotype-dependent: smoking/illness-burden (1) **59%**,
  functioning (2) 48%, metabolic (0) 40%, manic activation (3) 35%, **somatic (4) 14%**.
- **They also cut across DSM-5** (added E11b): ARI(7 DSM-5 subtypes, V0 cluster) = **0.006**
  — each cluster draws from every diagnosis (`longitudinal_dsm_phenotype.csv`; Suppl. Fig S1
  via `17_export_longitudinal_figure.py`).
- **Reframe (decision):** this is the empirical demonstration that *discrete* clustering
  fails — subgroups that neither persist nor align to DSM-5 = **slices of a continuum, not
  natural kinds** — i.e. a negative result that **motivates the dimensional model**, NOT a
  phenotype finding. The discrete flow Sankey is supplement-only, retitled accordingly.
- **Dimensional companion** (`18_export_dimensional_flow.py`, E11c): the *continuous-axis band*
  is retained far better than the discrete label — same-band V0→V1 **0.32–0.60** (depression
  0.60, ADHD/trauma 0.56) vs discrete 0.39 ⇒ the **labels hop, the positions are stable**.
  later_onset ≈ chance (0.32) confirms it is baseline-only/static. DR excluded at V3 (cliff).

## E12 · Step-1 structure test — discrete vs dimensional — 2026-05-23
Triggered by an unconvincing k=5 (flat silhouette ~0.18 at all k, arbitrary-looking
UMAP). `scripts/04_structure_test.py`: eigengap, gap-vs-Gaussian-null, HDBSCAN,
bimodality, DSM-subtype anchor + mood↔psychosis continuum.
- **Verdict: no discrete trans-diagnostic clusters.** Eigenvalues smooth (no gap);
  gap statistic monotone (no natural k); PCA scree gradual (PC1 10%); axes ~unimodal
  (BC 0.56). The only discrete structure is **diagnosis** (HDBSCAN↔cohort ARI **0.70**
  — it recovers SZ/DR/BP blobs). Trans-diagnostic variation is **dimensional**: the 7
  enrolled DSM subtypes order on a mood↔psychosis axis (|Spearman| 0.64–0.79).
- **DSM subtypes (enrolled, 7):** BP-I/II/NOS · schizophrenia/schizoaffective/
  schizophreniform · MDD. (Thesauri list more codes, but only these are enrolled; DR
  is uniformly MDD.) `arm` is the primary-diagnosis column; read it via the
  HarmonizedDataset MultiIndex (string-id concat breaks on float usubjid).
- **Process note:** the script's first auto-verdict said "4/4 discrete" — an
  over-generous heuristic. The HDBSCAN-vs-cohort check (ARI 0.70) + a standardized
  raw-domain PCA overturned it; heuristic fixed to key off HDBSCAN↔cohort ARI +
  gap-monotonicity. *Lesson: don't trust an automated cluster-validity verdict
  without checking what the "clusters" actually are.*
- **Decision (2026-05-23):** pivot to the **dimensional axis model** — classical
  (sklearn: factor analysis / sparse-PCA) + AI (PyTorch autoencoder). Validate axes
  against the 7 subtypes (continuum) + outcomes. The **deep graph embedding** (engine
  `stage_b2` VGAE/contrastive) is **kept in reserve for discrete-structure discovery**
  — worth a future shot, but not the current focus.
- **For the paper Discussion:** we explicitly tested discrete structure and showed the
  only discrete signal is DSM diagnosis itself → motivates the dimensional/HiTOP framing.

## E13 · Dimensional axis model — classical + AI — 2026-05-23
`scripts/05_dimensional_axes.py` (sklearn FA, varimax) + `scripts/06_dimensional_ae.py`
(PyTorch masked autoencoder, no imputation — mask fed to encoder, masked recon loss).
- **Classical:** parallel analysis K=14 (capped 8) → **7 reproducible axes**
  (Tucker congruence ≥0.85; 8th=0.18 noise), **confound-free** (max age/sex |corr|
  0.002): depression-severity (6.3%), later-onset, mania/activation, illness-burden,
  ADHD/impulsivity/trauma, **metabolic/inflammatory**, functioning. Variance diffuse
  ⇒ multi-axial, no dominant factor.
- **AI (masked AE):** CCA with FA = [0.93,0.84,0.80,0.74,0.63,…] → top-5 axes agree;
  recovers **mood↔psychosis continuum |Spearman| 0.89** (best of any method). Small
  age leak (0.15) vs FA 0.002 — note for paper.
- **Convergent validity** across linear/nonlinear + imputed/no-imputation ⇒ the
  dimensional axes are robust, not artifacts. This is the convincing trans-diagnostic
  representation. Reports: `dimensional_axes.html`, `dimensional_ae.html`.
- Note: varimax dispersed the mood↔psychosis axis (onto noise axis8); it is cleanest
  unrotated (PCA) / in the AE. Consider an oblique rotation, or report the AE axis as
  the mood↔psychosis dimension. Axis scores → Phase 4/5.

**Refinement (`07_dimensional_refine.py`):** K chosen by reproducibility-vs-K, not my
arbitrary cap. Split-half Tucker congruence is high only at low K (3/4/6) and
**erratic above** (K=7 0.08, K=8 0.18, K=9 0.88 — varimax factor-splitting + greedy
matching), so select from the stable range ≤8 → **final K=6** (min congruence 0.95,
confound 0.002): depression-severity · later-onset · mania/activation · illness-burden
· metabolic/inflammatory · ADHD/impulsivity-trauma. **No single varimax axis orders the
DSM subtypes** (per-axis centroid |Spearman| ≤0.36); the mood↔psychosis spectrum is a
*cross-axis direction* (AE 0.89), reported honestly, not forced into one factor.
`results/dimensional_final_scores.parquet` (6 axes) is the locked Phase-4/5 input.
Two more "don't trust the auto-pick" catches: the K-rule first grabbed K=9 off the
erratic tail, and a patient-level (vs subtype-centroid) Spearman made the continuum
look like 0.07 — both fixed.

## E14 · Phase 5 — outcome validation (axes vs DSM) — 2026-05-23
`scripts/10_phase5_outcomes.py`: nested 5-fold CV, V1 outcome ~ V0 baseline + age + sex +
{DSM = arm, 7 subtypes} vs {6 axes} vs both. Leakage-safe (predictors V0, outcome V1,
baseline-adjusted → de-circularizes EGF/hospitalization that also feed the axes).
- **QoL (EQ-5D): axes BEAT DSM** (R² 0.333 vs 0.289, +0.044, p<1e-16; combined ≈ axes).
- **Functioning (EGF): axes COMPLEMENT DSM** (combined 0.271 vs 0.239, p<1e-16; axes
  alone ≈ DSM).
- **Hospitalization: DSM dominates** (AUC 0.743; axes add +0.009).
- Effects face-valid: depression-severity → worse functioning/QoL (β −2.48 EGF);
  illness-burden → hospitalization (β +0.35). ⇒ dimensional axes add value for
  symptom-aligned outcomes, not service-use.
- Fixes: cohort+arm collinear → use **arm only** (7 subtypes, implies cohort); binary
  Logit LRT non-convergent (rare schizophréniforme separation) → CV AUC is primary;
  work-disability dropped (no follow-up coverage). Deferred: site/ComBat +
  mixed-effects, V2 replication.

## E15 · Phase 4 on axes — trait↔state stability — 2026-05-23
`scripts/08_longitudinal_axes.py`: project the V0 FA onto V1–V4 (pooled scaling,
per-visit-age residualized; refit axes ≡ locked, Tucker congruence ≥0.94) → V0↔Vk
test-retest r per axis.
- Trait↔state: adhd/trauma **0.62 (trait)** > depression **0.46 (intermediate)** >
  mania 0.35, illness-burden 0.29 (state) > metabolic 0.22 > later-onset 0.06.
- **Honest caveats:** later-onset domains recorded only at V0 → not trackable (0.06 =
  data artifact; mark axis STATIC/baseline-only). Metabolic low r partly attenuation
  (labs less repeated). Symptom axes genuinely fluctuate.
- Ties Phases 4+5: depression axis moderately stable + top outcome predictor;
  trauma/ADHD most trait-like. Lesson: check whether a variable is even *measured* at
  follow-up before interpreting low test-retest as instability.

## E16 · Robustness — V2 replication + site ComBat — 2026-05-23
- **V2 replication** of the Phase-5 head-to-head: QoL axes beat DSM (+0.041 vs V1
  +0.044), functioning complements (combined 0.229 vs 0.172), hospitalization
  DSM-dominated. Replicates cleanly.
- **Site ComBat** (`scripts/13_robustness_site.py`, neuroHarmonize; 20 sites ≥10):
  site batch magnitude 0.044 SD (small); axes survive harmonization (Tucker congruence
  with locked [1,1,1,.98,.98,.99]); head-to-head survives (QoL axes−DSM +0.037,
  functioning combined 0.262). ⇒ the axes are **not a site artifact**.
- Installed: neuroHarmonize, nibabel, neuroCombat. **Closes deferred task #43.**
- Net: the dimensional model is reproducible + confound-free + site-robust +
  V1/V2-replicated, with axes adding value over DSM for QoL/functioning. Manuscript-ready.

## E17 · Cognition BP/SZ sub-analysis — 2026-05-23
`scripts/14_cognition_bpsz.py` (cognition absent in DR by design → BP/SZ only, n=6,099;
4273 BP / 1826 SZ).
- **Polish (the fix):** v1 let WAIS sub-items dominate by count (~25 split domains →
  4 WAIS-flavoured factors, CVLT/TMT under-weighted). Now a **two-level aggregation**
  collapses raw items → instrument **stem-domains** → **7 standard constructs**
  (memory[CVLT], executive[TMT], processing-speed, working-memory, verbal &
  perceptual reasoning, fluency); TMT reverse-signed (confirmed −0.16 vs CVLT).
  Two-level call: `build_domain_scores(X,…,biology={})` → stems, then
  `build_domain_scores(stems,…,biology=COGNITIVE_DOMAINS)` → constructs (the
  composite path matches raw canonicals, so constructs must be built off stems).
- **Two cognitive factors** (parallel analysis K=2): **cog1 general-ability g**
  (percept_reasoning +0.67, working_memory +0.59, verbal_reasoning +0.47,
  memory_cvlt +0.41, fluency +0.26) and **cog2 processing-speed** (proc_speed +0.80,
  executive_tmt −0.22) — the classic **g + speed** structure.
- **Cognition ⊥ symptom axes:** max |r| **0.24** (g ↔ illness-burden −0.24, then
  ↔ metabolic −0.16, ↔ depression −0.13). Semi-independent, not symptom-redundant.
- **Small non-redundant increment to V1 functioning** (EGF, n=2,478): symptom-axes
  R² 0.394 → +cognition 0.398 (Δ **+0.004**; shuffled CV — was 0.169→0.174 under the
  un-shuffled-CV bug, see E15b). Modest but independent.
- Kept OUT of the 3-cohort model (would re-inject the DR-availability confound).

## E18 · Independence refactor — self-sufficient, sharable repo — 2026-05-23
The repo was made fully **self-contained** so collaborators can clone-and-run with no
sister project. (This is why earlier entries — E2/E3/E10, FINDINGS §2.3, ROADMAP — still
name now-deleted exploratory scripts like `cluster_v0.py`, `cluster_v0_profile.py`,
`cluster_domains_profile.py`, `reproduce_v0_clusters.py`: they were superseded here.)
- **Engine internalized.** The vendored `face_stratification` modelling code we actually
  use was copied into `src/trans_diag/engine/` (`MultipartiteSpectralEmbedding`,
  `HarmonizedDataset`, `FeatureSchema`, masked similarity, enrichment, KMeans,
  `bootstrap_stability`). `archive/`, `config/` and `data/external/` were removed.
- **Package renamed** `face_common` → `trans_diag` (kept the `src/` layout).
- **Per-patient data purged from git history** (`git filter-repo`): the 3 raw cohort CSVs,
  `data/external`, and 10 per-patient result artifacts (scores/embedding parquets,
  cluster-assignment CSVs). Aggregate results (loadings, meta JSON, contingencies) still
  ship; per-patient files stay on disk (`.gitignore`d) so the pipeline runs locally.
  `.git` 65M→24M. FACE data is confidential (Fondation FondaMental).
- **Scripts numbered in execution order** `01_…`→`18_` + a `00_run_all.py` orchestrator,
  so a reviewer reads/runs them top-to-bottom. 4 superseded V0-exploration scripts deleted.
  Gotcha: Python module names can't start with a digit, so the shared head-to-head helpers
  (`cv_metric`, `added_axes_test`, `axis_betas`, `OUTCOMES`) were lifted out of
  `phase5_outcomes.py` into importable `src/trans_diag/outcomes.py` before renaming.
- **Reproduction notebook** `notebooks/FACE_reproduction.ipynb` added (runs the numbered
  pipeline, displays **aggregate** outputs only — no per-patient rows — for confidentiality).
- **Verified:** every manuscript number reproduces to ≤**1.8e-12** (BLAS round-off); 54 tests
  pass with no `archive/` on the path; the engine still reproduces the sister contingency.

## E19 · Imputation-free factor model — the 6th-axis ablation — 2026-05-24
Follow-up to the §2.7 mean-fill caveat (the FA input is 65% observed; 35% mean-filled to 0,
the one imputation on the dimensional path). Asked: do the six varimax axes survive WITHOUT the
fill? Re-derived loadings from the **pairwise-complete (masked) correlation matrix** — no cell
ever filled (`scripts/sensitivity_masked_fa.py`; mechanism probe `..._mechanism.py`).
- **Mechanism (the real problem).** For standardized data filled to 0, the fill correlation is
  *exactly* the masked one reweighted by co-observation: `corr_fill ≈ O ∘ corr_masked`,
  `O_AB = n_AB/√(n_A·n_B) ≤ 1` (R²=**0.999**, slope 0.99; naive no-reweight R²=0.91). So mean-fill
  **differentially attenuates** correlations between domains measured in *different* patients —
  partially **re-importing the cohort-by-missingness confound** the masked operators were built to
  exclude (cohort is 98% predictable from the mask).
- **Effect.** 5 of 6 axes reproduce imputation-free (Tucker congruence: depression 0.99, onset
  0.98, mania 0.91, illness-burden 0.97, metabolic 0.96). The **6th — ADHD/impulsivity/trauma
  (WURS/BIS/CTQ) — does NOT** (0.23). Imputation-free, that slot is a **work-disability /
  socio-occupational** factor (arrêt-travail +0.47/+0.45, prof. status +0.32, education +0.20).
  PAF-on-mean-fill reproduces the published sklearn axes at ≥0.97 on all six → it's the
  imputation, not the extraction method.
- **Why the 6th flips.** WURS is BP-only (BP .93 / DR .00 / SZ .10); BIS/PRISM/ESS are BP+DR but
  absent in SZ — co-administered → high mutual overlap (within-O 0.84). Mean-fill preserves their
  cluster while attenuating the lower-overlap cross-cohort work-disability cluster (within-O 0.57).
  The 6th factor under fill is thus partly a *which-battery-was-run* artifact; the masked criterion
  selects the cross-cohort-robust cluster instead. Both are real correlated clusters; the mean-fill
  biases the *selection* of the weakest factor.
- **K is not the issue.** Masked split-half reproducibility supports K=6 (min congruence 0.89),
  collapsing at K=8 like the mean-fill model → the honest model stays ~6-dimensional; only the
  6th axis's identity changes. (Exact K is mildly extraction-dependent: sklearn ML-FA showed a K=5
  dip / K=7 collapse that PAF doesn't — lock the method in the re-analysis.)
- **Decision → final re-analysis.** Adopt the imputation-free (masked-covariance) FA as the
  primary model to remove the residual cohort bias: re-derive loadings + **masked per-patient
  scores** (pairwise-complete, no fill), re-characterize/confound-check the new 6th axis, re-run
  the head-to-head / longitudinal / cognition, report honest deltas (QoL & hospitalization expected
  unchanged — depression & illness-burden axes are clean; functioning's axis-6 contribution should
  transfer to the work-disability axis). Documented as MANUSCRIPT §3.8 (ablation) + Limitation 8.
- Artifacts: `results/sensitivity_masked_fa{,_mechanism}.json`.
- **✅ Re-derivation DONE (steps 2–4).** Masked FA promoted to the package (`src/trans_diag/masked_fa.py`)
  and made the primary model (`07_dimensional_refine.py` rewritten: masked corr → PAF+varimax →
  masked posterior-mean scores; `08_longitudinal_axes.py` now projects the locked loadings by
  masked scoring, not a per-visit refit). Re-ran 07/08/10/11/12/13/14/15/16/18. Results: outcomes
  hold/strengthen (QoL +0.039, functioning combined +0.034, hosp DSM-dominated); 5/6 loadings
  unchanged, 6th = work-disability; **trait-state flipped — metabolic now most trait-like (0.20→0.64),
  the old 0.20 was the mean-fill diluting sparse follow-up labs**; confound clean (age/sex 0.017,
  cohort η²≤0.11, site ≤0.05); CCA(AE,FA) 0.98 (vs the *final* imputation-free model; the deep verification later found 15 had
been comparing the AE to the superseded `05` mean-fill scores → a too-low 0.93, fixed to compare
against `07`). Manuscript fully updated (§2.1/2.2/2.7/2.8, Tables
  2–3, §3.3–3.8, Limitations, Abstract). 54 tests pass.

## E20 · Fold-honest re-fit — removing the head-to-head optimism (Limitation 10) — 2026-05-24
The head-to-head (`10`) scored patients with axis loadings fit once on the **full** sample (`07`),
then used those scores as CV predictors — so each held-out fold helped fit its own axes (a mild
optimism; MANUSCRIPT Limitation 10). New `scripts/20_robustness_cvrefit.py` re-derives the masked
factor model **inside each training fold** (train-only loadings **and** train-only standardization)
and scores the held-out patients from those train-only loadings; 5× shuffled 5-fold. To isolate the
effect it computes, under *identical* folds, axes from full-sample loadings (`ax_all`) vs refit
loadings (`ax_re`); the gap is exactly the optimism honest refitting removes.
- **Result: optimism is negligible.** QoL axes−DSM **+0.040** [+0.039,+0.041] (vs +0.039 full-sample);
  functioning combined 0.400 vs DSM 0.365 (still complementary); hospitalization −0.139 (still
  DSM-dominated). all-data−refit gap **≤0.007 AUC, ≈0 R²**. The all-data axes reproduce the committed
  head-to-head within rounding (EGF 0.365, hosp 0.613, QoL 0.343) → the script is validated.
- **Why so small:** the loadings are a population-level covariance structure estimated over thousands
  of patients, so withholding a few hundred test-fold patients barely moves them. Factor sign/order
  differs across folds — immaterial (ridge/logistic are invariant to a sign flip or column permutation).
- Limitation 10 reframed from "future work should remove" to "measured, negligible" (§3.5, now the
  **fourth** robustness threat). Artifact: `results/robustness_cvrefit.json`. 78 tests pass; ruff clean.

## E21 · Within-FACE held-out replication — transportability (Limitation 9) — 2026-05-24
External replication is unavailable (FACE = one national network); `scripts/21_replication_holdout.py`
tests **transportability** by deriving the model on a held-out partition and applying it to unseen data.
- **Leave-one-cohort-out structure (congruence vs locked axes):** hold out DR → min **0.98**
  (near-identical); hold out SZ → mean 0.93 (work-disability dips to 0.63); hold out **BP** → the
  small SZ+DR partition (n=2,761) underdetermines the 6-factor model — depression/onset/work-disability
  transport (≥0.74) but mania/illness/metabolic don't (0.36/0.67/**0.08**). Honest read: the well-measured
  cross-cohort axes transport; the BP-concentrated-instrument axes (metabolic/mania) need BP in the
  training set — a measurement-coverage effect, not a BP artifact (§3.1 theme).
- **Leave-one-site-out outcomes (LeaveOneGroupOut over 15–18 sites ≥10; axes refit on other sites;
  pooled out-of-site predictions):** QoL axes−DSM **+0.042** (transports to unseen centres), functioning
  combined−DSM +0.033, hosp −0.147. The headline survives site-blocked CV.
- **Leave-one-cohort-out outcomes (predict an UNSEEN diagnosis; axes increment over age+sex+baseline):**
  QoL transports across diagnoses (predict BP +0.029, SZ **+0.058** R²); functioning transports for BP
  (+0.050) but NOT SZ (−0.14 domain-shift) — consistent with functioning being complement-only.
- **Net:** strong within-network transportability for QoL + the well-measured structure; honestly bounded
  for metabolic/mania (without BP) and functioning-in-SZ. Wired into MANUSCRIPT §3.5 + Limitation 9,
  FINDINGS §3g, golden test. Artifact: `results/replication_holdout.json`. **Not** external replication —
  still the #1 outstanding step.

## E22 · FACE clinical scores — explored, then cut — 2026-05-24
Tested whether the 6-axis model reduces to simple clinical scores (the "FACE profile": FACE-D =
QIDS+MADRS+STAI; FACE-M = metabolic-syndrome+cholesterol+inflammation) and to a single
general-severity ("p") score. **Verdict: no usable predictive instrument; cut from the manuscript
(§3.9 removed), keeping only the p-factor negative in §4.6.**
- **FACE-D**: tautological (a depression score from depression scales) and not trans-diagnostic
  (QIDS/MADRS/STAI 0% observed in SZ). No clinical value beyond the MADRS itself.
- **FACE-M**: reproduces the metabolic axis (r=0.88), but near-circularly. Prospectively it does
  NOT forecast metabolic deterioration (confident null on ≥7% weight gain) and only weakly forecasts
  worse QoL/functioning (ΔR²≈0.003). Within-BP it is associated with antipsychotic exposure
  cross-sectionally (β≈0.12 SD, p=.001) but confounded (antidepressant equally large; no clean
  longitudinal effect) — iatrogenicity not isolable with these data (BP-only; prevalent exposure).
- **General ('p') factor**: an oblique (promax) rotation of the masked axes gives mean inter-factor
  *r* ≈ −0.06 → **no general factor**; the dominant single dimension is depression-specific, and a
  one-number score under-performs both DSM and the full model. **Kept** as a §4.6 discussion result
  (confound control + no imputation may dissolve the literature's *p*-factor), backed by
  `scripts/19_pfactor.py` (`results/pfactor.json`) + a golden test.
- **Cleanup**: deleted `19_face_score`/`22_face_m_prospective`/`23_face_m_iatrogenic`, `face_score.py`,
  its tests/exports, artifacts and Fig 7; renamed `24_pfactor`→`19_pfactor`; removed manuscript §3.9
  and updated §4.6 + the rotation limitations; scrubbed the notebook and docs.

## E23 · K=7 re-lock — the externalizing/neurodevelopmental axis — 2026-05-24
Reviewer-prompted deep-dive that started from an AE reconciliation bug and ended in a headline change.
- **AE staleness found.** `06_dimensional_ae.py` defaulted its latent K off the *exploratory* 8-factor
  `05` output and CCA'd against it (committed artifact: K=8, leading CCA 0.935) — so the manuscript's
  "K=6, CCA 0.98, null 0.06, AE age-leak 0.15, AE mood 0.89" was a mix of K=8 values and an
  unreproducible 0.98. Fixed `06` to recompute the masked-FA reference internally at the locked K and
  CCA against it + a 200× row-permutation null (order-independent; no `00_run_all` reorder needed).
- **Non-monotone reproducibility → K=7.** Masked split-half min Tucker congruence: K3 0.98, K4 0.97,
  K5 0.89, **K6 0.886**, **K7 0.911**, K8 0.22 (collapse), K9–11 0.76–0.81, K12 0.43. K=7 is a *local
  maximum* and the last reproducible K. Horn's parallel analysis over-extracts to ~14 at N=9,013; we
  select on cross-sample reproducibility (as the paper already argued), not eigenvalue rules.
- **What the 7th axis is.** At K=7 the K=6 mania/activation+impulsivity factor **splits** (6a3→7a4
  congruence 0.93, 6a3→7a5 0.68) into **pure mania** (Altman/Mathys/YMRS) and a new
  **externalizing/neurodevelopmental** axis (WURS +0.53, BIS +0.40, CTQ +0.38, maternal-suicide +0.23,
  edu −0.23). Anchored by well-observed instruments (CTQ 91%), confound-clean (0.018), near-orthogonal
  (max off-diag r 0.14), the **least diagnosis-bound** axis (DSM η² 0.017). It is the genuine,
  imputation-free counterpart of the ADHD/trauma signal that mean-fill mis-selected as the K=6 6th axis
  (E19/§3.8): the *content* was real; its *selection over work-disability at K=6* was the artifact.
- **Decision gate = parity, not gain.** Predictive head-to-head at K=7 vs K=6: QoL +0.038 vs +0.039,
  functioning combined +0.033 vs +0.034, hosp DSM-dominated — identical within CV noise (a finer
  rotation of nearly the same variance predicts the same). The pivot is justified on **structural
  validity + novelty + reproducibility**, prediction K-robust (a robustness point, not a loss). User
  chose the full pivot with this caveat explicit.
- **Blast radius.** K flipped in `07/08/12/13/19/20/21`; `axes.py` remapped to 7 (new SS order:
  illness-burden now precedes the pure mania axis, externalizing at 5, work-disability at 7); figures
  `15/16/18` generalized to `len(AXIS_NAMES)`; `06` reconciled; golden tests re-derived. Fixed a latent
  leave-one-cohort NaN bug in `21` (`_refit_axes` now NaN-safe when a domain is unobserved in a fold).
  Full `00_run_all` reproduces (22 steps OK, 366 s); 75 tests + `verify.py` pass. Manuscript (title →
  "seven", §2.7/2.8/3.3–3.8/4.1–4.2/4.8, Tables 2–4, abstract), CLAUDE, FINDINGS §3i all updated.

## E24 · De-circularization (`12`) migrated to the masked imputation-free estimator — 2026-05-24
`12_phase5_decircularized.py` had re-fit its de-circularized axes with sklearn `FactorAnalysis` +
`z.fillna(0)` — the very mean-fill §3.8 shows biases the weakest factor — so the robustness check was
probing a mean-fill model, not the published masked one. Swapped `fit_axes` to `masked_loadings` +
`masked_scores` (identical estimator to `07`). Result is unchanged/slightly cleaner and now
apples-to-apples: QoL de-circularized **+0.038** (= the headline), functioning combined **+0.029**,
hosp axes AUC 0.611; `axes_full` (drop nothing) now ≈ the locked model. Updated MANUSCRIPT §3.5 +
Table 3 De-circ column. 75 tests + ruff still pass. (`13` ComBat keeps median-impute *by
construction* — ComBat requires complete data; that imputation is unavoidable, not an oversight.)

## E25 · Parsimonious screening panel — sparse item→axis distillation (§4.5, reviewer 2.1) — 2026-05-25
Clinical-feasibility step: distil the 54-domain battery into a short panel. MultiTaskElasticNet
(l1_ratio 0.8) over the ~225 raw V0 items (questionnaire pool); teacher = the 7 locked axis scores;
row-wise L1 → one shared panel; densest support within a ≤15-item cap; in-fold-honest reconstruction
R² (selection re-run per CV fold, as in `20`). `22_screening_panel.py`.
- **Panel = 11 features**: Altman, YMRS, MADRS, a QIDS item, CTQ, BIS, WURS, EGF, age-at-treatment,
  age-at-first-episode, lifetime-admission count (several are brief instrument *totals*, not single
  questions — stated honestly) + a fixed routine **metabolic-panel** add-on (BMI/waist/trig/HDL/
  glucose/HbA1c/BP) for the metabolic axis.
- **In-fold R²**: mania 0.85, depression 0.83, illness-burden 0.75, externalizing 0.71 (recover);
  later-onset 0.51 (partial); **work-disability 0.09** and **metabolic 0.03** NOT recovered by the
  shared questionnaire (metabolic → 0.29 with the labs add-on). Honest tier: a symptom-optimized
  shared L1 panel drops the axes whose defining items (work-status, labs) it does not share.
- **Decisive**: the panel **preserves the QoL advantage over DSM** — EQ-5D axes−DSM +0.032
  (questionnaire) / +0.035 (+labs) vs +0.038 full — so the cheap panel keeps the part that predicts
  patient-reported outcomes.
- Wired step 22 into `00_run_all`; MANUSCRIPT §2.13 + §4.5 + Table 5 + Figure 7; golden test added.
  Research-grade draft, not a validated instrument (stated). State/trait (reviewer 2.2) deferred to
  a follow-up per "one gap at a time" (designed: MixedLM variance-components on the longitudinal scores).
- **Update (2026-05-25) — three additions** (user-requested): (i) repeated-CV 95% CIs — QoL axes−DSM
  **+0.032 [+0.028,+0.035]** (excludes 0); (ii) the **combined model** — functioning combined−DSM
  **+0.024 [+0.022,+0.025]**, so the panel *complements* DSM on functioning, not just QoL; (iii) a
  **group-aware per-axis panel** (top-2 items/axis, 13 features incl. arret_travail) that recovers
  work-disability (0.09→**0.47**) and illness-burden (0.79) at a small cost to depression
  (0.83→0.75)/externalizing (0.71→0.62) and a slightly smaller QoL edge (+0.025 [+0.022,+0.028]).
  Reframed §2.13/§4.5/Table 5/Fig 7 as a parsimony-vs-coverage trade-off; golden test extended.

## Deferred / open (do not forget)
- **Deep graph embedding** (engine `stage_b2` VGAE/DGI/contrastive) — a future
  attempt at *discrete*-structure discovery, in case a learned representation
  surfaces clusters the masked-cosine spectral view misses.
- ~~ComBat site harmonization~~ — **DONE (E16)**: axes site-robust (congruence ~1).
- ~~Cognition (NEUROPSYCHOLOGIE) domains~~ — **DONE (E17)**: BP/SZ-only sub-analysis
  (DR-missing by design); cognition ⊥ symptoms, small functioning increment.
- **Verify the metabolic-direction sign** once biology composites are in.
- **Outcome/trajectory validation** — the real test that A beats DSM.
- Scrutinise / possibly down-weight the **"denial" response-style axis**.
