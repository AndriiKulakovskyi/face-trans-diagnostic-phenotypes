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
- `face_common.filters` (variable/patient completeness, V0 anchor) + the
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
- `src/face_common/schema_gen.py` (dictionary → engine `FeatureSchema`) and
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
- `src/face_common/domains.py` — symptom instruments auto-grouped by canonical
  stem (masked mean of robust-z items, min-items threshold); curated biology
  composites with explicit members + directions. 190 items → 72 domains; no
  domain > 1.4% of dims (was 30% for SUICIDE); metabolic_syndrome 90% coverage.
- `residualize_features(spline_df, cross_fit)` — natural-spline age + sex-specific
  curves + K-fold cross-fitting (double-ML partialling-out).

## E9 · Direction-A domain clustering result — 2026-05-22
`scripts/cluster_domains.py`: 72 domains → coverage floor 30% (**54 kept**, 18
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
  type), breaking the scores↔embedding join; `cluster_domains.py` now writes
  parquet directly and the profile coerces the index defensively.
- **Metabolic axis recovered** as a prominent phenotype (cluster 0 high vs 2 low)
  — the deck's metabolic theme is supported; composite direction is explicit
  (BMI/trig/glucose↑, HDL↓ = higher burden) so no sign-inversion ambiguity.
- Residual to tighten: age dCor 0.117 (consider more spline knots / age²·sex).

---

## E11 · Phase 4 — temporal coherence V0→V4 — 2026-05-22
`scripts/longitudinal_coherence.py`. Build the SAME domain scores at every visit
(pooled scaling, per-visit-age spline+cross-fit residualization), assign each
patient-visit to a V0 phenotype, measure persistence.
- **Methodology note (important):** a masked nearest-**centroid** rule could *not*
  reproduce the V0 spectral-embedding clusters (self-ARI **0.024** — centroid vs
  spectral geometry mismatch). Replaced with a **classifier** (HistGradientBoosting,
  NaN-native) trained on V0 domain scores → V0 labels: **5-fold accuracy 0.842**
  (k=5, chance 0.20) — phenotypes ARE recoverable from domains; rule is valid.
- **Result:** coherence is **modest and stable** across visits — ARI(V0↔Vk)
  **≈0.06–0.07**, persistence **≈37–39%** (V1 n=3782 … V4 n=697). Phenotype-
  dependent: smoking/illness-burden (1) **59%**, functioning (2) 48%, metabolic
  (0) 40%, manic activation (3) 35%, **somatic (4) 14%**.
- **Attractor:** non-persisters converge toward phenotype 1 (29/36/32% of 0/3/4
  → 1 at V1).
- **Interpretation:** the V0 cross-sectional phenotypes are **part trait, part
  state** — trait-like burden axes (metabolic, smoking/illness-burden, functioning)
  persist; symptom-state axes (mania, somatic) are transient (treatment + episode
  resolution + regression-to-mean). This **nuances the temporal-coherence
  hypothesis**: a single-visit clustering captures state as well as trait. DR
  excluded at V3 (cliff).

## E12 · Step-1 structure test — discrete vs dimensional — 2026-05-23
Triggered by an unconvincing k=5 (flat silhouette ~0.18 at all k, arbitrary-looking
UMAP). `scripts/structure_test.py`: eigengap, gap-vs-Gaussian-null, HDBSCAN,
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
`scripts/dimensional_axes.py` (sklearn FA, varimax) + `scripts/dimensional_ae.py`
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

**Refinement (`dimensional_refine.py`):** K chosen by reproducibility-vs-K, not my
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
`scripts/phase5_outcomes.py`: nested 5-fold CV, V1 outcome ~ V0 baseline + age + sex +
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
`scripts/longitudinal_axes.py`: project the V0 FA onto V1–V4 (pooled scaling,
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

## Deferred / open (do not forget)
- **Deep graph embedding** (engine `stage_b2` VGAE/DGI/contrastive) — a future
  attempt at *discrete*-structure discovery, in case a learned representation
  surfaces clusters the masked-cosine spectral view misses.
- **ComBat site harmonization** as a sensitivity analysis (task #43).
- **Cognition (NEUROPSYCHOLOGIE)** domains — handle non-random battery
  availability before inclusion.
- **Verify the metabolic-direction sign** once biology composites are in.
- **Outcome/trajectory validation** — the real test that A beats DSM.
- Scrutinise / possibly down-weight the **"denial" response-style axis**.
