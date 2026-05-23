# FINDINGS — FACE trans-diagnostic clustering (running research log)

Paper-oriented log of empirical + methodological discoveries. Numbers are
reproducible from `scripts/cluster_v0.py` + `scripts/cluster_v0_profile.py`
(artifacts in `results/cluster_v0_*`, report `reports/cluster_v0.html`).
See ROADMAP.md for the plan; this file is the "what we actually learned".

## 1. Data & reconciliation
- **9,013 V0 patients**: BP 6,252 / SZ 2,209 / DR 552.
- `patient_uid = cohort::usubjid_patients` — `usubjid` collides across cohorts
  (970 shared ids); all patient ops must key on `patient_uid`, not `usubjid`.
- The sister 4-cohort reference clusters cover **7,211** of our BP/SZ/DR patients
  (those that passed their coverage filter); the ASP-dominated cluster is empty
  in our cohorts → **6 populated reference clusters**.
- **No imputation anywhere** — masked pairwise-complete cosine similarity.

## 2. Methodological discoveries (paper Methods / Limitations)

### 2.1 The confound ladder (clustering raw EHR features is a trap)
Unsupervised clustering recovers the **largest-variance nuisance axis** unless it
is explicitly removed. We climbed down four rungs:
1. `brthdtc` (birth date) is stored numeric ≈ **1e17** → that one column
   dominated the cosine geometry and produced a *spurious* "great" result
   (bootstrap ARI 0.96, ARI-vs-sister 0.31). **Retracted.**
2. After dropping the date + robust-scaling: **raw labs/anthropometry** (counts
   in the thousands) dominated.
3. After robust z-scoring: clusters became a **sex × age stratification** —
   cluster↔sex ARI **0.32** > cluster↔cohort **0.19**.
4. That sex/age signal was carried **almost entirely by physical-comorbidity
   occurrence flags** (`*_mhoccur`: lupus→female, MI→older …). Excluding them:
   cluster↔sex ARI **0.32 → 0.005**, cluster↔age → **0.008**.

**Principled configuration:** clinical sections only, **age/sex-residualized**,
robust-scaled, `*_mhoccur` excluded, dates/site/IDs dropped (→ 129 features).

### 2.2 Implicit feature weighting = item count (important, fixable)
Cosine treats each column as one equal dimension, so a construct measured by
**many items contributes many dimensions** and dominates. In the 129-feature
clinical set:

| section | dims | | instrument | items |
|---|--:|---|---|--:|
| SUICIDE | 39 (30%) | | isf | 15 |
| AUTO-QUESTIONNAIRES | 35 | | cssrs | 11 |
| HETERO-QUESTIONNAIRES | 15 | | psqi (sleep) | 8 |
| ANTECEDENTS | 12 | | ctq (trauma) | 8 |
| EVALUATION MEDICALE | 10 | | fast (functioning) | 7 |

⇒ The phenotypes that emerged (suicidality, sleep, trauma, depression) are
**precisely the most-itemized instruments**. The clustering currently weights by
item count, not clinical importance. **Fix:** aggregate items → instrument/
domain scores (or per-domain factor scores) so each construct contributes
comparably, *before* embedding.

### 2.3 Engine reused as-is (no surgery)
masked cosine (pairwise-complete) → spectral embedding per cohort-coverage
partition → partition weight `sqrt(n_features × n_patients)` → concat + L2 →
KMeans. We reproduced the sister's published 7-cluster contingency exactly from
their saved embedding (`scripts/reproduce_v0_clusters.py`).

### 2.4 Discrete vs dimensional — the structure is DIMENSIONAL (pivotal)
Step-1 structure test (`scripts/structure_test.py`) on the V0 domain embedding,
prompted by an unconvincing k=5 (flat silhouette, arbitrary-looking UMAP):
- **No discrete trans-diagnostic clusters.** Laplacian eigenvalues rise smoothly
  from ~0 (no eigengap); the gap statistic vs a matched-Gaussian null rises
  **monotonically** (no natural k); standardized PCA scree is gradual (PC1 10%,
  top-5 25%); top axes are ~unimodal (Sarle BC ≈ 0.56).
- **The only discrete structure is DSM diagnosis.** HDBSCAN finds 4 dense regions,
  but they ARE the cohorts (**ARI 0.70 with cohort**: an SZ blob, a DR blob, two BP
  blobs).
- **Trans-diagnostic variation is a continuum.** The 7 enrolled DSM subtypes order
  along a mood↔psychosis axis (|Spearman| 0.79 on the embedding / 0.64 raw-domain
  PCA): MDD → BP-II → BP-I → BP-NOS → schizoaffective → schizophreniform →
  schizophrenia.
- **Implication:** the k=5 KMeans phenotypes were *reproducible slices of a
  continuum* (stable but low-silhouette). The honest, stronger representation is
  **dimensional** (a few interpretable trans-diagnostic axes), not discrete boxes;
  "more k / other clustering" cannot create discrete clusters the data lacks.
- *Process note:* the script's first auto-verdict ("4/4 discrete") was an
  over-generous heuristic, overturned by the HDBSCAN-vs-cohort check + standardized
  PCA; the heuristic was fixed to key off HDBSCAN↔cohort ARI + gap monotonicity.

## 3. Result

### 3a. v1 (item-level clinical, residualized, k=6) — intermediate
Six trans-diagnostic symptom phenotypes (bootstrap ARI 0.89, cohort ARI 0.024),
but the drivers were dominated by the most-itemized instruments (§2.2) and a
"denial" response-style axis — i.e. item-count weighting, not clinical priority.
Superseded by v2.

### 3b. v2 — domain scores + biology, spline-residualized (k=5) — **current**
`scripts/cluster_domains.py`: 72 domain scores → coverage floor 30% (54 kept) →
**nonlinear spline + cross-fit residualization** on age+sex → masked-cosine
spectral embedding.
- **Principled k = 5** (max bootstrap stability **ARI 0.972**, min consensus
  **PAC 0.047**; k≥6 loses stability *and* re-admits sex).
- **Confound verified removed:** sex Cramér's V **0.041**, age-tertile ARI
  **0.006**, age dCor **0.117**, **cohort ARI 0.002** — independent of sex, age,
  and diagnosis (genuinely trans-diagnostic).
- **Five phenotypes** cutting across BP/SZ/DR (standardized domain profiles,
  `reports/cluster_domains.html`): (0) **metabolic / later-onset**
  (metabolic_syndrome↑, later age-of-onset↑, somatic↓); (1) **heavy-smoking /
  hospitalization burden** (smoking↑, hospitalizations↑, YMRS↓); (2)
  **high-functioning / low burden** (EGF↑, education↑, QoL↑; metabolic↓,
  smoking↓); (3) **manic activation / impulsivity** (YMRS **+1.33σ**, Altman/
  Mathys/BIS↑; DR ≈ 0); (4) **somatic / medication-burden** (somatic **+1.53σ**,
  QTc↑, prolactin↑). Clusters 3 & 4 are single-axis-dominated; 0–2 multivariate.
- **Metabolic axis** is now a prominent, explicit phenotype (cluster 0 high vs 2
  low; composite oriented BMI/trig/glucose↑, HDL↓) — supports the deck's
  metabolic theme without sign ambiguity.
- By construction this does **not** reproduce the sister diagnosis-aligned
  clusters — it is trans-diagnostic phenotype discovery (direction A).

### 3c. Discrete clusters are unstable & diagnosis-independent — NEGATIVE result (→ dimensional)
**Reframed (was "temporal coherence"):** this is the empirical demonstration that *discrete*
clustering fails here, motivating the dimensional model — not a phenotype finding. A phenotype
**classifier** (HistGradientBoosting, NaN-native) trained on V0 domain scores → V0 k=5 labels
(**5-fold accuracy 0.842**) is applied to each follow-up visit. (A nearest-centroid rule failed
— self-ARI 0.024 — because it cannot reproduce the V0 spectral-embedding geometry; that
fragility is itself a discreteness red flag.)
- **The discrete clusters do not persist:** ARI(V0↔Vk) **≈0.06–0.07**, persistence **≈37–39%**
  across V1–V4 (n 3782→697) — barely above chance for k=5.
- **They cut across DSM-5:** ARI(7 DSM-5 subtypes, V0 cluster) = **0.006** — each cluster draws
  from every diagnosis (trans-diagnostic, not relabelled DSM). (`longitudinal_dsm_phenotype.csv`;
  Suppl. Fig S1.)
- **Persistence is phenotype-dependent:** trait-like clusters persist (smoking/illness-burden
  **59%**, functioning 48%, metabolic 40%); symptom-state clusters are transient (manic
  activation 35%, somatic **14%**).
- **Verdict (negative result):** forcing discrete clusters yields subgroups that are neither
  temporally stable nor diagnosis-aligned ⇒ **slices of a continuum, not natural kinds**. The
  honest dimensional "flow" (`export_dimensional_flow.py`; §3f) retains a patient's *continuous-
  axis band* far better — same-band V0→V1 persistence **0.32–0.60** (depression 0.60, ADHD/trauma
  0.56) vs the discrete labels' 0.39: the **labels hop, the positions are stable**. Cross-
  sectionally the axes are also **diagnosis-independent** — DSM-5 subtype explains only
  **η² 0.01–0.14** of each axis (depression 0.14 [95% CI 0.13–0.16] highest, ≤0.05 for five of
  six; 2000-sample bootstrap; `dimensional_dsm_eta_squared.csv`, fig6c). DR excluded at V3.

### 3d. Dimensional axis model — the convincing representation (classical + AI)
Following the dimensional verdict (§2.4): model trans-diagnostic variation as
continuous axes — `scripts/dimensional_axes.py` (sklearn varimax factor analysis)
and `scripts/dimensional_ae.py` (PyTorch masked autoencoder, **no imputation**).
- **7 reproducible, confound-free axes** (classical FA; parallel analysis K=14,
  capped 8): (1) **depression/internalizing severity** (QIDS/MADRS/STAI/FAST/PSQI,
  6.3% var), (2) later age-of-onset, (3) **mania/activation** (Altman/YMRS/Mathys/
  BIS), (4) **hospitalization/illness-burden**, (5) ADHD/impulsivity/childhood-trauma
  (WURS/BIS/CTQ), (6) **metabolic/inflammatory** (metabolic_syndrome/cholesterol/
  inflammation), (7) functioning. Split-half **Tucker congruence ≥0.85** for all 7
  (8th = noise, 0.18); **max |corr| with age/sex = 0.002**. Variance is diffuse (no
  dominant factor) ⇒ genuinely multi-axial.
- **Convergent validity (AI):** the masked autoencoder's nonlinear axes agree with
  the classical factors (**canonical correlations 0.93/0.84/0.80/0.74/0.63** for the
  top 5) and recover the **mood↔psychosis continuum** strongly (|Spearman| **0.89**
  on one AE axis; PCA 0.79; varimax dispersed it). Two very different methods
  (linear/imputed vs nonlinear/no-imputation) converge → the axes are robust.
  Caveat: the AE has a small age leak (|corr| 0.15) vs FA's 0.002.
- **Refined final set (`dimensional_refine.py`):** reproducibility-vs-K (split-half
  Tucker congruence) is high only at low K (3/4/6) and erratic above (varimax
  factor-splitting), so the **locked set is K=6** (min congruence 0.95, confound
  0.002): depression-severity · later-onset · mania/activation · illness-burden ·
  metabolic/inflammatory · ADHD/impulsivity-trauma. **No single varimax axis carries
  the mood↔psychosis ordering** (per-axis subtype |Spearman| ≤0.36) — the spectrum is
  a *cross-axis direction* (the AE recovers it at 0.89), stated honestly rather than
  forced into one factor.
- **`results/dimensional_final_scores.parquet` (6 axes) is the trans-diagnostic
  representation** carried into Phase 4 (persistence) and Phase 5 (do axes beat DSM
  on outcomes?).

### 3e. Phase 5 — do the axes beat DSM on outcomes? (the value test)
**Shuffled, repeated** 5-fold CV (R=200; `phase5_outcomes.py` + `phase5_ci.py`) predicting V1
outcomes from V0 axes vs DSM (arm, 7 subtypes), all adjusting for V0 baseline + age + sex
(leakage-safe). **Bugfix (E15b):** the original `cv_metric` used UN-shuffled folds and the
matrix is cohort-ordered (BP…SZ…DR) → cohort-imbalanced folds distorted the R² (EGF baseline
0.19 unshuffled vs 0.33 shuffled). Conclusions unchanged; absolute R² corrected upward,
notably functioning. Numbers below are repeated-CV mean [95% CI]:

| outcome | n | DSM | axes | combined | verdict |
|---|--:|--:|--:|--:|---|
| EQ-5D quality of life (R²) | 2423 | 0.302 | **0.339** | 0.341 | **axes BEAT DSM** (+0.036 [+0.033,+0.039]) |
| EGF functioning (R²) | 3196 | 0.365 | 0.362 | **0.394** | axes **complement** DSM (combined +0.029 [+0.027,+0.030]) |
| any hospitalization (AUC) | 3332 | **0.747** | 0.604 | 0.758 | DSM dominates (axes −0.143 [−0.158,−0.130]) |

- **Conclusion:** the dimensional axes carry information DSM lacks for **patient-
  reported / functional** outcomes (QoL, functioning) — for QoL they *outperform*
  diagnosis — but not for **service-use** (hospitalization), where diagnosis + prior
  hospitalization dominate. Effects are face-valid: depression-severity axis → worse
  functioning/QoL (β −2.48 on EGF); illness-burden axis → more hospitalization (β +0.35).
- This validates direction A for symptom-aligned outcomes. Deferred: site/ComBat +
  mixed-effects, V2 replication; the binary-outcome LRT didn't converge (rare
  schizophréniforme subtype) so the CV AUC is the primary evidence there.
- **De-circularization (`phase5_decircularized.py`; addresses the review's #1 concern):** the
  depression axis contains EQ-5D/EGF/FAST and the illness-burden axis contains the hosp counts,
  so predicting those outcomes from those axes is potentially circular. Refitting the axes
  **without each outcome's own measure(s)** changed essentially nothing: QoL still beats DSM
  (R² **0.340** vs 0.305, +0.036; circular +0.036), functioning still complemented (combined
  **0.392** vs 0.366, +0.026; axes alone ≈ DSM, −0.005), hospitalization still DSM-dominated
  (axes AUC 0.600).
  ⇒ the advantage is **not** an artifact of outcome content — baseline adjustment already
  controls each outcome's V0 value, and the depression axis is carried by QIDS/MADRS/STAI.

### 3f. Phase 4 on axes — temporal stability (trait↔state gradient)
`scripts/longitudinal_axes.py`: project the V0 factor model onto V1–V4 (pooled scaling,
per-visit-age residualized; the refit axes match the locked set, Tucker congruence
≥0.94) → per-axis V0↔Vk test-retest correlation.
- **Trait↔state gradient** (mean V0↔V1/V2 Pearson r): adhd/impulsivity/trauma **0.62
  (trait-like)** > depression-severity 0.46 (intermediate) > mania/activation 0.35,
  illness-burden 0.29 (state-like) > metabolic 0.22 > later-onset 0.06.
- **Honest caveats:** later-onset is **static** — its domains (age of onset / first
  hospitalization) are recorded only at V0, so it can't be tracked (the 0.06 is a data
  artifact, not instability — flag the axis baseline-only). Metabolic's low r is partly
  **attenuation** (labs are repeated less often at follow-up). Symptom axes (depression,
  mania) genuinely fluctuate, as expected clinically.
- **Unifies Phases 4+5:** the depression-severity axis is moderately stable *and* the
  strongest predictor of functioning/QoL; the trauma/ADHD axis is the most trait-like.

### 3g. Robustness — V2 follow-up + site (ComBat) harmonization (all shuffled-CV)
- **V2 follow-up (same cohort, not independent replication):** the head-to-head holds — QoL
  axes beat DSM (R² 0.265 vs 0.230, **+0.034**), functioning axes complement DSM (combined
  0.353 vs 0.306, **+0.047**), hospitalization DSM-dominated (AUC 0.727).
- **Site (ComBat, `scripts/robustness_site.py`, neuroHarmonize; 20 sites ≥10 patients):**
  the site batch effect is small (mean |adjustment| = 0.044 SD); after ComBat-harmonizing
  the domain scores, the 6 axes are essentially unchanged (Tucker congruence with the
  locked axes [1.0, 1.0, 1.0, 0.98, 0.98, 0.99]) — they are **not a site artifact**. The
  head-to-head survives: QoL axes still beat DSM (+0.032), functioning still complements
  (combined 0.386 vs 0.366), hospitalization still DSM-dominated.
- **Conclusion:** the dimensional model and its outcome advantage are **reproducible**
  (split-half 0.95), **confound-free** (age/sex 0.002), **site-robust** (ComBat congruence
  ~1), **de-circularization-robust**, and **consistent across V1+V2** (same cohort) — solid
  for the manuscript, pending independent external replication.
- **Extra robustness checks (`review_checks.py`, addresses review #5–9):** the age/sex
  orthogonality (0.002) is by construction (residualized) — the meaningful test is independence
  from variables NOT removed: **site η² ≤0.05, cohort η² ≤0.10** per axis. AE↔FA agreement is
  real, not a CCA artifact (**leading canonical corr 0.93 vs permutation null 0.06**). Mood↔
  psychosis **ρ=0.79, bootstrap 95% CI [0.75, 0.86]** (but only 7 a-priori-ordered centroids).
  K-selection is **non-monotone** (figS2; K=5 min-congruence 0.31, K=6 0.95 — K chosen on
  reproducibility+interpretability, not a clean elbow). **Honest caveat:** HDBSCAN's cohort
  recovery (ARI 0.70) is **partly a measurement-protocol artifact** — cohort is **97.9%
  predictable from the observation mask alone** (cohorts got different instrument batteries);
  the dimensional verdict is unaffected (still no trans-diagnostic discrete clusters).

### 3h. Cognition (BP/SZ complementary analysis)
Cognition is absent in DR **by design** (0% vs BP 71% / SZ 86%) — including it in the
3-cohort model would re-inject a cohort/availability confound — so it is analysed within
BP/SZ (`scripts/cognition_bpsz.py`; 6,099 patients). To stop WAIS sub-items dominating
by count, raw items are aggregated **items → instrument stem-domains → 7 standard
constructs** (memory[CVLT], executive[TMT], processing speed, working memory, verbal &
perceptual reasoning, fluency; TMT reverse-signed — confirmed −0.16 vs CVLT).
- **Two cognitive factors** (parallel analysis K=2): a broad **general-ability factor**
  (perceptual/verbal reasoning + working memory + memory) and a **processing-speed**
  factor — the classic g + speed structure.
- **Cognition is semi-independent of the symptom axes** (max |r| 0.24): the clearest
  link is **general cognition ↔ illness-burden (−0.24)** (lower ability with more chronic
  illness burden), then ↔ metabolic (−0.16) and depression-severity (−0.13). Not
  redundant with symptoms (matches the cognition-vs-symptom literature).
- **Small, non-redundant increment to functioning** (V1 EGF, BP/SZ n=2,478): symptom-axes
  R² 0.394 → +cognition 0.398 (Δ **+0.004**; shuffled CV) — modest but independent.

## 4. The scientific fork (framing)
Two mutually-exclusive products, because **diagnosis + demographics are the
dominant variance axes** in the data:
- **(A) Trans-diagnostic discovery** — cluster *net of* diagnosis/demographics →
  symptom-dimension phenotypes shared across BP/SZ/DR. Novel; matches the
  project's primary goal (cut across DSM-5). *Current direction.*
- **(B) Diagnosis-aligned recovery** — keep the diagnosis-separating features →
  clusters recapitulate DSM + demographics and resemble the sister's. A
  replication/concordance result, not novel phenotypes.

## 5. Open questions / caveats (Discussion)
- **"Denial" axis** likely a symptom-minimization *response style*, not
  psychopathology; partly inflated by item count. Scrutinise / consider dropping.
- **Modest silhouette (~0.2)** — dimensional phenotypes, not separated islands.
  Validate by **stability + interpretability + outcome prediction**, not silhouette.
- **k not yet principled** — needs a stability-vs-k curve, gap statistic, and/or
  consensus clustering (current k=6 was chosen only to match the sister count).
- **DR V3 attrition cliff** (3 patient×visit rows) — exclude DR at V3 longitudinally.

## 6. To verify before any headline claim
- **Item-count weighting fix** before naming phenotypes definitively (§2.2).
- **Metabolic-direction sign** (deck's metabolic claim) — biology is currently
  excluded; revisit direction if biology is re-introduced as domain scores.
