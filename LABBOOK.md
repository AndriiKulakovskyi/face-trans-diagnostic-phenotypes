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
- **Five trans-diagnostic phenotypes** (cohort mix ≈ proportional to sample):
  0 **metabolic burden / later-onset** (metabolic_syndrome↑); 1 **smoking +
  inflammation / early-onset**; 2 **high-functioning / low burden** (EGF↑,
  metabolic↓ inflammation↓ cholesterol↓); 3 **manic activation + impulsivity +
  ADHD-traits** (YMRS/Altman/Mathys/BIS/WURS↑, DR only 2%); 4 **somatic +
  medication-effects** (somatic↑, prolactin↑, QTc↑, seasonality↑).
- **Metabolic axis recovered** as a prominent phenotype (cluster 0 high vs 2 low)
  — the deck's metabolic theme is supported; composite direction is explicit
  (BMI/trig/glucose↑, HDL↓ = higher burden) so no sign-inversion ambiguity.
- Residual to tighten: age dCor 0.117 (consider more spline knots / age²·sex).

---

## Deferred / open (do not forget)
- **ComBat site harmonization** as a sensitivity analysis (task #43).
- **Cognition (NEUROPSYCHOLOGIE)** domains — handle non-random battery
  availability before inclusion.
- **Verify the metabolic-direction sign** once biology composites are in.
- **Outcome/trajectory validation** — the real test that A beats DSM.
- Scrutinise / possibly down-weight the **"denial" response-style axis**.
