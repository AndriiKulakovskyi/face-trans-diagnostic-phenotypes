# LABBOOK — FACE trans-diagnostic study (v2) — research notebook

> **LEGACY — V2 benchmark / reference arm.** Historical record of the V2 study (entries V2-1 … V2-23).
> Not the current project. Active source of truth: the V3 precision-psychiatry plan
> ([`../V3_PLAN.md`](../V3_PLAN.md) · [`../ROADMAP.md`](../ROADMAP.md)). See [`README.md`](README.md).

Chronological trace of the v2 study — what we did, observed, decided, and **why**.
> The v1 notebook (entries E1–E26) is archived at git tag `v1-archive-2026-05-30`.

## V2-1 · Restart on the re-curated dictionary — 2026-05-30
Stopped trusting the v1 common-variables set. Snapshotted the full v1 state (tag
`v1-archive-2026-05-30`, branch `archive/v1-research`) and started branch `v2-study`.
Decision: **re-derive every result from zero** on v2; keep the method code + engine.

## V2-2 · v2 dictionary finalized + cognition reconciled — 2026-05-30
- v2 = **214 usable variables** (subset of v1's 361), with structured sanity bounds + coverage.
- Promoted v2 to the canonical `data/face-common-vars.xlsx` (loader auto-detects v2 → sanity
  bounds + v2 rules + fondacode site); archived v1 dictionary.
- Cognition reconciled to `docs/neuropsy_features.yaml`: 6 primary 3-cohort features (verbal
  reasoning, working memory, processing speed ×2, TMT-A/B), education as a covariate; fixed the
  bogus "mmHg"/"free text" labels; re-curated `domains.py` COGNITIVE_COMPOSITES → 5 constructs.

## V2-3 · QA-driven dictionary corrections — 2026-05-30
Dropped (NOT USABLE): `brthdtc` (date→1e18 artifact, redundant with age), `clozapin` (SZ-specific
treatment marker), `hcg_lbstresc` (pregnancy test, ~0 + 333000 sentinel), `mdq` (absent at BP
yearly), `ltsv03` (DR n=0), and 12 near-zero-variance `*_mhoccur` flags. Fixed within-column unit
mixing for `mchc`/`hct` (g/L ÷10 → g/dL; L/L ×100 → %) via new v2 rules. QA: 190/190 pass, 0 fail.

## V2-4 · Preprocessing debug + type-aware scaling — 2026-05-30
- Fixed the robust-z **explosion**: `prolactin` domain |z|≈106 → ≤5 (log1p heavy-skewed labs + clip ±5).
- `normalize_for_embedding` → **type-aware bounded scaling to [−1, 1]** (binary/ordinal min-max;
  continuous robust-z-clip). All 190 post-processed features land in [−1, 1] (0 out of range).
- Confirmed V0 **within-cohort** missingness (97/190 >25%; pooled 128 was inflated by 31 structural
  2-cohort vars). Decision: **keep masked design, no hard missingness drop.**

## V2-5 · QA report (3 parts) — 2026-05-30
`scripts/qa_harmonization.py` → `results/reports/qa_harmonization.html`:
Part 1 harmonized variables (native scale) + sanity + missingness · Part 2 post-processed
variables (type-aware [−1, 1], all 190 in Part-1 order) · Part 3 aggregated V0 domain scores
(the ~69 model inputs). 190/190 pass.

## V2-6 · Aggregation investigation — why aggregate at all? — 2026-05-31
Question raised: do we need construct aggregation before the dimensional analysis, or run on the
190 standardized items directly to avoid dropping signal? **First-principles:** type-aware scaling
fixes *scale* but not (a) **count/redundancy bias** — a construct's geometric weight scales with how
many items its questionnaire happens to have — nor (b) **structured missingness** under no-imputation.
Both distort *any* inner-product / squared-error method (FA, k-means, cosine, **and AE/VAE/VQ-VAE**,
which also force imputation). **Empirical** (`scripts/sensitivity_aggregation.py`): item-level
masked corr cond ≈1.3e9 vs domain 110; within-SZ **67%** of item-pairs have <100 co-obs (only
104/188 items exist in SZ → item structure is BP-carried); suicide block = **19%** of item-axes. Flat
means are lossy (metabolic PA_k=3; CTQ r(mean,PC1)=0.76) **but** the top 4–5 dimensions are
**granularity-invariant** (canonical r ≥ 0.85, perm-null 0.04) — the headline structure is not a
grouping artifact. **Decision:** replace flat masked means with a **hierarchical/bifactor measurement
model in HYBRID mode** (clinical anchors, data-revised), keeping masked / no-imputation. Full rationale
+ evidence: [AGGREGATION_RATIONALE.md](AGGREGATION_RATIONALE.md); plan: [HIERARCHICAL_FA_PLAN.md](planning/HIERARCHICAL_FA_PLAN.md).

## V2-7 · Hierarchical-FA Stages 0–2 — 2026-05-31
- **Stage 0** (`scripts/01_hfa_stage0_itemset.py`) — froze a **188-item** set: added every valid
  measurement (the 34 composite-dropped labs/vitals + 16 previously-NOT-USABLE-but-valid rows incl.
  WAIS matrices/arith/symbols, MDQ, rare `*_mhoccur` flags); excluded identifiers, age/sex
  (residualized), confounds `hcg`/`clozapin`/**`oxcarbaz`** (caught in QA), the D8 branching
  suicide items (<6% obs, 0 complete cases), and by-construction-collinear `tmtba01`. Factorable
  (scree 12.6, 10.1, 6.9…); plain KMO undefined (near-singular cond 1.3e9 → use scree, shrunk-KMO 0.66).
- **Stage 1** (`scripts/02_hfa_stage1_efa.py`) — masked Horn parallel analysis → **42 first-order
  factors, highly nameable**, and they **independently confirm the An2 aggregation problems**:
  metabolic → adiposity / BP / lipids / cholesterol; CTQ **denial (ctq40/41) splits off** from trauma;
  C-SSRS → severity / intensity; ISF → ideation / attempts; dropped labs/vitals **recovered** as
  factors (autonomic-HR, red-cell, inflammation, vit-D). Substantive (top-12) factors reproduce
  **leave-BP-out (mean Tucker congruence 0.91)** → not BP-driven.
- **Stage 2** (`scripts/03_hfa_stage2.py`) — hybrid first-order model: each construct = a
  **within-construct masked 1-factor posterior** (estimated weights, no item-count / total-subscore
  double-count, explicit signs). **84 constructs** (26 multi-item). *Wins:* metabolic split →
  adiposity VAF1 **0.93** / cholesterol 0.90 / BP 0.72 / lipids 0.72 (vs collapsed **0.40**); CTQ
  cleaned (denial dropped, VAF1 0.59); CGI reduced to severity (`cgi01`). **Φ₁: 106/3486 construct
  pairs |r|>0.3 (max 0.74)** with coherent second-order seeds (depression: madrs~qidsr 0.74,
  staya~qidsr 0.73; functioning: cgi~egf −0.69) → second-order layer warranted.
- **Honest limitations (reviewer-facing):** C-SSRS constructs are **sparse** (coverage 6–16%; the
  usable suicidality dimension is ISF `suicidal_ideation`, 0.91); `medical_comorbidity` (24 flags,
  VAF1 0.38) is a heterogeneous bin (kept per decision, flagged); several lab panels are weakly
  unidimensional (electrolytes 0.34, red-cell 0.43, thyroid 0.46). 84 constructs is many — the
  second-order layer (Stage 3) is what reduces them to interpretable dimensions.
- Findings logged in [FINDINGS.md](FINDINGS.md) §Track 1.

## V2-8 · `medical_comorbidity` → data-anchored decomposition — 2026-05-31
The pooled 24-flag `medical_comorbidity` construct was VAF1 0.38 (not one dimension). Split it
**data-anchored, step by step** (`scripts/sensitivity_comorbidity.py`):
- **(1) prevalence:** 13/24 flags are **<2%** (cirrhose n=7, MS 16, HIV 21) → un-clusterable; only
  ~8 flags ≥5%. Cohort-reporting confound noted (migraine 1.6% SZ vs 19% DR).
- **(2) within-BP association (cohort-cleaned):** correlations are tiny (max phi 0.10) but
  co-occurrence **lifts are real (2–3×)**; hierarchical clustering → two interpretable clusters —
  **cardiac** (hta+autcardv+trbrycard, 2.7–2.9×) and **atopic/inflammatory** (acne+eczema+cheveux+
  toxidermi+psoriasis, 1.6–2.5×); head-trauma standalone.
- **(3) validation:** `cardiac_history` VAF1 **0.50** (stable BP/SZ/DR; bootstrap CI tight);
  `atopic_inflammatory` **0.26** (stable, weak). Splitting concentrates the signal (pooled bin
  VAF1 0.06 → cardiac 0.50).
- **Encoded** (`scripts/03_hfa_stage2.py`): `cardiac_history` + `atopic_inflammatory` (flagged
  weak) + standalone {`migraine`, `head_trauma`, `peptic_ulcer`}; the **13 flags <2% dropped from
  the dimensional inputs, retained as Stage-4 validators** (`results/hfa/stage2_comorbidity_validators.csv`)
  — i.e. *does the recovered metabolic/inflammation axis predict real cardiovascular/autoimmune
  history?* **Honest caveat:** even split, somatic-comorbidity constructs are weak (0.26–0.50) — the
  signal is genuinely thin; `atopic_inflammatory` is borderline. Model now has **88 constructs** (169/188 items).

## V2-9 · Stage 3 — second-order trans-diagnostic dimensions — 2026-05-31
`scripts/04_hfa_stage3.py`: factor the construct correlation Φ₁ (75 constructs, coverage ≥30%,
standardized; PSD with **0% neg-eigen mass** — the aggregation conditioning win vs item-level);
oblique (promax); K by masked split-half Tucker congruence; general factor **tested** via
Schmid–Leiman ECV.
- **The stat-correctness audit caught a real bug.** Split-half congruence is **non-monotonic**:
  reproducible at K=2–4 (0.94–0.98), **collapses at K=5 (0.36)**, spurious recovery at K=7–10. A naive
  "max K ≥ 0.85" rule → K=10 with **Heywood loadings** (bio_qt 1.18 > 1 — improper). Fixed to
  **first-collapse-minus-1 → K=4** (0 Heywood, proper solution). Lesson logged.
- **4 reproducible trans-diagnostic dimensions** (ECV **0.36** → multidimensional, **no dominant
  p-factor**; mean |Φ₂| 0.17):
  1. **Internalizing** — qidsr/madrs/staya/fast/eq5d/egf (depression–anxiety–functioning)
  2. **Cognitive impairment** — executive/processing/psychomotor/working-memory/perceptual (edu −)
  3. **Illness course** — age-of-onset + inverse hospitalization burden (later-onset / lower-chronicity)
  4. **Cardiometabolic–inflammatory** — lipids/inflammation/adiposity/BP/hepatic/autonomic
- **Notable vs v1:** 4 dims, not v1's 6 — **mania is NOT a reproducible trans-diagnostic axis**
  (`mania_activation` is a fine construct, VAF1 0.71, but loads <0.30 on all 4 dimensions);
  v1's later_onset + illness_burden merged into one *course* axis. `axes.py` (v1 names) confirmed
  **stale** — names re-derived here, to be **locked after Stage 4**.

## V2-10 · K-selection deep dive — 2026-05-31
The Stage-3 "first-collapse-minus-1 on the MIN congruence" rule (→K=4) was too conservative (the min
collapses if ONE factor is unstable). Per-factor congruence (`scripts/05_hfa_kselect.py`) resolves it:
- #factors reproducing (congruence ≥0.85): K=4→4, K=5→4, K=6→**5**, K=7→6, K=8→8; **Heywood: 0
  through K=6, 1 at K=7**, 2 at K=9. The "K=5 collapse" was a rotation *swap* (cardiac vs trauma
  competing for the 5th slot), not absent structure.
- **4 rock-solid dims** (≥0.97): internalizing, cognition, course, cardiometabolic-inflammatory.
- **5th reproducible @K=6 (0.89, proper): cardiac/somatic-history** (cardiac_history+peptic_ulcer+
  perinatal) — but rests on the weak binary comorbidity constructs.
- **6th (childhood-trauma/ADHD: ctq+wurs) only @K=7 (0.84) where a Heywood appears** → real but not
  cleanly extractable at this N.
- **Robust negatives (true at every K 4–7):** `mania_activation` & `suicidal_ideation` load <0.30
  everywhere → NOT reproducible trans-diagnostic axes; metabolic & inflammation do not separate.
  Parallel-analysis K=19 / Kaiser K=26 over-extract (ignored).
- **Decision (user):** **K=4 primary (+ K=6 as sensitivity).** K≥7 rejected (Heywood).

## V2-11 · Stage 4 — validation (K=4 PASSES) — 2026-05-31
`scripts/06_hfa_stage4.py`. The 4-dimension solution passes every check:
- **Confound-clean:** no dim explained >0.25 by cohort / sex / age / site / missingness
  (internalizing cohort η²=0.09; cognition 0.16 with educ 0.16 = a real correlate, not a confound;
  course & cardiometab ~0.01).
- **Trans-diagnostic *and* clinically valid:** dims vary mostly WITHIN cohorts but show the expected
  between-cohort differences — internalizing highest in **DR** (depression cohort), cognition worst
  in **SZ**. (η² cohort ≤0.16 → not cohort markers.)
- **Leave-cohort-out reproducible:** drop BP → per-dim Tucker congruence min **0.84**; drop SZ → 0.90;
  drop DR → 0.99.
- **Granularity-invariant:** hierarchical-K4 vs flat-domain-K4 canonical r = **[0.99, 0.93, 0.77, 0.39]**
  — top 3 (internalizing/cognition/course) invariant (NOT an aggregation artifact); the 4th differs
  because the hierarchical model adds the recovered labs/vitals the flat domains dropped.
- **Mania resolved (not a bug):** `mania_activation` is well-measured (cov 0.95), clinically valid
  (BP 0.63 > SZ 0.40 > DR −0.04), within-cohort-varying (η² cohort 0.02) — but **orthogonal to all 4
  dims (|r| ≤ 0.09)**. It shares too little variance with other constructs to anchor a second-order
  factor (same for suicidality): a distinct standalone construct, not a missing axis.
- **K=6 sensitivity:** the 5th (cardiac/somatic-history) and 6th (childhood-trauma: ctq+wurs) are
  also confound-clean (η² ≤0.13).
- **Verdict:** the **4-dimension trans-diagnostic structure** (internalizing, cognition, course,
  cardiometabolic-inflammatory; **no p-factor**, ECV 0.36) is reproducible, confound-clean,
  trans-diagnostic, and granularity-invariant. `axes.py` (v1, 6 axes) is **superseded for v2**; the
  v2 axes are defined by `results/hfa/stage3_loadings.csv`. (Polychoric sensitivity, D9, deferred.)

## V2-12 · Dimensional result finalized — axis names, polychoric, mania — 2026-05-31
- **Axis names locked** in `src/trans_diag/axes.py` (NEW v2 source-of-truth; `axes.py` is stale-v1,
  left untouched for the v1 scripts): **dim1 internalizing · dim2 cognition · dim3 illness_course ·
  dim4 cardiometabolic**. Polarity documented (dim3 higher = later-onset / lower-chronicity).
- **Polychoric sensitivity (D9) PASSED** (`scripts/sensitivity_polychoric.py`): of the 3 all-binary
  multi-item constructs (suicidal_ideation, atopic_inflammatory, cardiac_history), tetrachoric scores
  correlate ≥0.96 with Pearson; the 4 dims are **identical** (Tucker congruence 1.00). Crucially,
  `suicidal_ideation` max loading stays ~0.30 under tetrachoric → its absence as a dimension is **not**
  a Pearson-attenuation artifact. The Pearson choice is vindicated; binary-attenuation caveat closed.
- **Mania decision:** `mania_activation` (and `suicidal_ideation`) are valid, well-measured,
  clinically-valid constructs **orthogonal** to the 4 correlated axes (|r| ≤ 0.09, robust to
  polychoric) → reported as **independent standalone dimensions** (`axes.ORTHOGONAL_DIMENSIONS`),
  **included as features in Phase-5 stratification** but NOT part of the correlated factor structure.
  Not dropped, not forced. (Scientific note: mania's independence from internalizing is itself a finding.)

## V2-13 · Phase 5 — stratification: DIMENSIONAL (continuum), not discrete — 2026-05-31
`scripts/07_phase5_stratify.py`. Structure-test battery (eigengap, gap-vs-Gaussian-null, HDBSCAN,
Sarle bimodality, DSM-anchor, bootstrap stability) on **A = 6 axes** (4 dims + mania + suicidal_ideation)
primary and **B = 75 construct scores** via the masked engine embedding (sensitivity).
- **A: DIMENSIONAL / continuum.** HDBSCAN **0 dense clusters (100% noise)**; real−null silhouette gap
  small & non-peaking (0.01–0.05); axes unimodal (Sarle ≤0.51); k-means ARI-vs-DSM ~0.03. (k-means
  bootstrap stability 0.79–0.93 is high, but that's a *continuum artifact* — k-means partitions a blob
  stably; HDBSCAN + unimodality are decisive.)
- **B: only discrete structure = DSM diagnosis.** HDBSCAN's 3 dense clusters are **exactly the 3
  cohorts (ARI = 1.00)**; silhouette gap monotone (no natural k); unimodal. Finer granularity reveals
  **no novel subtypes**.
- **Verdict (both arms agree):** trans-diagnostic structure in FACE is **DIMENSIONAL** — 4 continuous
  axes (+ 2 orthogonal: mania, suicidality), **no discrete patient subtypes** beyond the DSM categories
  themselves. (No p-factor [dimensional arm] + no discrete clusters [this] = a clean dimensional account.)

## V2-14 · Validation Study A — cohort confound: axes are NOT a cohort artifact — 2026-05-31
`scripts/09_cohort_confound.py`. Attack: 3 cohorts = 3 DSM diagnoses → the axes might encode
between-cohort/batch differences. Two defenses (Tucker congruence vs the pooled K=4):
- **Decisive — cohort-residualized:** center each construct within cohort (remove between-cohort
  means), re-derive → **all 4 axes ≥0.96**. The structure is within-cohort covariance, not between-
  cohort means → confound refuted.
- **Within-cohort re-derivation:** **BP (n=6252) reproduces all 4 axes ≥0.95**; **SZ (n=2209)** 3/4
  strong (cognition 0.92, course 0.93, cardiometab 0.87), **internalizing 0.80** (borderline);
  **DR (n=552) underpowered** (cardiometab 0.66 — not interpreted).
- **Verdict (confound):** the 4 axes are within-cohort, cohort-residualization-robust dimensions —
  NOT between-cohort artifacts.

### V2-14a · The internalizing-SZ dig → a cross-cohort measurement-coverage asymmetry (KEY)
Digging into the within-SZ internalizing 0.80 (`/tmp/internalizing_sz.py`) revealed it is **not** a
subtle clinical nuance but a **structural measurement-coverage fact** (data design, not a bug — all
verified PARTIAL with `SZ_col=None`):
- **Internalizing is BP+DR-ANCHORED.** Its defining scales — **MADRS, QIDS, STAI, FAST, Altman,
  PRISM, CSM — are 0% in SZ** (the FACE-SZ cohort used a psychosis battery, not these mood/anxiety/
  functioning self-reports). SZ patients are scored on internalizing only via the surviving 3-cohort
  proxies (GAF `egf`, CGI, PSQI, MARS, EQ-5D) → within-SZ congruence 0.80. **Strong axis in mood
  disorders, proxy axis in schizophrenia.**
- **Cardiometabolic is 3-cohort *core*** (lipids, adiposity, glycemia, CRP/WBC/neutrophils all in SZ)
  with BP+DR-only **peripherals** (autonomic heart-rate, lymphocytes) → minor (within-SZ 0.87).
- **Cognition + illness-course are cleanly 3-cohort** (all top constructs 43–92% in SZ) → fully
  trans-diagnostic.
- **Consequence (must state in the manuscript):** the *fully* trans-diagnostic axes are **cognition,
  illness-course, and (core) cardiometabolic** — i.e. biology/cognition/course; the **internalizing
  (mood) axis is directly measured only in BP+DR**, represented by proxy in SZ. The dimensional / no-
  p-factor / no-subtypes results stand; the "trans-diagnostic" label on *internalizing* is qualified.
- **Implications:** Study B orthogonality → also compute **within BP+DR** (where mood + biology are
  both measured). Study C → internalizing invariance is a BP+DR test. Study D → report internalizing
  prediction split by direct (BP+DR) vs proxy (SZ).

Validation plan: [VALIDATION_PLAN_v2.md](planning/VALIDATION_PLAN_v2.md).

## V2-15 · Study B — symptom⊥biology + the p-factor is a symptom-only artifact (THE headline) — 2026-05-31
`scripts/10_orthogonality_pfactor.py`. Computed **within BP+DR** (clean — mood+biology both
measured, per Study A; pooled near-identical).
- **Orthogonality:** mean |construct r| **within** symptom 0.24 / cognition 0.42 / biology 0.08
  (heterogeneous panels); **between symptom↔biology 0.03, symptom↔cognition 0.07, biology↔cognition
  0.04**. Strongest single symptom↔biology link only **0.15** (FAST↔lipids), **0% of pairs >0.15** —
  no hidden link. → symptoms, biology, cognition are mutually ~orthogonal.
- **p-factor is symptom-bound:** first-factor share (K-free) **symptom-only 0.33 → +cognition 0.27 →
  +biology 0.15 → full 0.09** (monotonic dissolve); ECV(K=4) 0.58→~0.40 consistent. A general factor
  exists *within symptoms* but does NOT span the integrated symptom+biology+cognition space.
- **Not dilution:** biology/cognition are *structured* (they form the coherent Stage-3 cardiometabolic
  & cognition axes) yet orthogonal to symptoms — so no single factor can span them.
- **HEADLINE (the non-derivative message):** *"A general psychopathology (p-)factor is an artifact of
  symptom-only measurement. Biology and cognition are structured but orthogonal to symptoms, so an
  integrated model is genuinely multidimensional with no dominant general factor."* Robust BP+DR↔pooled.

## V2-16 · Study C — longitudinal coherence (V0→V1→V2) — 2026-05-31
`scripts/11_longitudinal_coherence.py` (reuses Stage 0/2/3 logic via importlib — V0 reproduces the
committed Stage-2 scores, |r|=1.00).
- **Structural invariance** (re-derive K=4 per visit; Tucker congruence vs V0): internalizing
  **0.99/0.98**, cardiometabolic **0.98/0.97** (strong); cognition 0.00@V1 / **0.87@V2** (battery is
  baseline-anchored — `wais` ~5% at V1); illness-course 0.87/0.78. → **the dimensional structure
  PERSISTS at follow-up** wherever re-measured.
- **Score stability** (project V0 loadings; Spearman test-retest V0↔V1 / V0↔V2): cardiometabolic
  **0.66/0.62** (trait-stable), internalizing **0.59/0.53** (moderate — episodic mood, as expected),
  cognition 0.49@V2, **illness-course 0.12/0.16 (low)**.
- **Illness-course dig:** its core age-of-onset constructs (`agedebutpremier_episode`, `agetrt`,
  `agedebut_hospitalisation`) are **0% at V1/V2** (baseline-only historical intake items); only the
  noisy lifetime-hospitalization burden is re-collected. So illness-course is a **fixed baseline-
  historical axis, not a longitudinally re-measurable state** — low test-retest is a measurement-design
  artifact, not instability.
- **Verdict:** structure is longitudinally **coherent where measurable**: biology trait-stable, mood
  state-moderate (appropriate), cognition baseline-anchored (V2), illness-course a fixed baseline trait.
  V0-defines/later-validate design supported.
- **Emergent synthesis (across Studies A+C):** the **cardiometabolic axis is the most measurement-
  robust** trans-diagnostic dimension — 3-cohort *and* longitudinally stable; cognition is 3-cohort but
  baseline-anchored; internalizing is BP+DR mood; illness-course is fixed-historical. With Study B
  (symptoms⊥biology, p-factor is symptom-only), the robust trans-diagnostic substrate is **biological/
  cognitive**, while symptom structure is more cohort-/state-specific.

## V2-17 · Study D — predictive validity vs DSM (the make-or-break): MODEST, functioning-specific — 2026-05-31
`scripts/12_predictive_validity.py`. Out-of-sample cohort-stratified 5-fold CV; M0 = age+sex+V0
baseline-of-outcome, M1 = +DSM (`dsm_diagnosis`), M2 = +axes, M3 = +both, M2x = cross-domain axes
(drop internalizing = the non-circular test). **Attrition check passed**: V0 axes → has-V2-followup
AUC 0.531 (near-chance) → completer sample not biased on the axes.
- **Relapse-by-V2 (binary, n=3378):** baseline CGI-S alone AUC **0.765**; **neither DSM nor axes add**
  (Δ axes-vs-DSM +0.004 [CI −0.002,+0.011], ns; cross-domain +0.002 ns). Relapse ≈ baseline severity.
- **GAF@V2 (n=2043):** axes **add over DSM ΔR²=+0.046 [+0.030,+0.062]**; cross-domain (non-circular)
  +0.033 [+0.020,+0.047]; axes≈DSM-alone (0.314 vs 0.310).
- **FAST@V2 (BP+DR, n=1878):** axes **beat DSM** (DSM adds 0.000; Δ axes-vs-DSM +0.038 [+0.022,+0.053]);
  cross-domain +0.026 [+0.013,+0.039].
- **Per-axis (functioning):** led by **illness_course** (ΔR² 0.017/0.010) + internalizing (0.012/0.018);
  cognition & cardiometabolic add little individually (≤0.005) — the functional signal is modest and
  illness-course-led, NOT a strong cognition/biology effect.
- **VERDICT — partially earns its keep:** the dimensions are **DSM-equivalent-to-better and add a
  modest, significant, non-circular increment to FUNCTIONAL prognosis** (ΔR² ~0.04), led by illness-
  course; **no advantage for relapse** (baseline-severity-driven). More than descriptive, but the
  clinical gain over DSM is small — report honestly, do not oversell.

## V2-18 · Study D refined — relapse done right (remission-based + discrete-time survival) — 2026-05-31
`scripts/13_predictive_survival.py`. Fixed the regression-to-mean confound in the original
change-based relapse: at-risk = **V0-remitted** (CGI-S 1–3), relapse = deterioration to **CGI-S ≥4**;
**discrete-time hazard** over person-intervals (attrition handled via censoring); **GroupKFold by
patient** (no leakage); bootstrap CIs by patient; DSM = `dsm_diagnosis` one-hot; **two methods**
(regularized logistic + HistGradientBoosting), identical pipeline per predictor set.
- n=1766 person-intervals (1262 patients, effectively BP+SZ; DR-remitted negligible), 400 events (23%);
  hazard V0→V1 26%, V1→V2 17%.
- **The confound was real:** baseline-only AUC **0.765 (old) → 0.578 (remission-based)** — the old
  "baseline dominates / dims useless" was largely a regression-to-mean artifact.
- **Relapse-from-remission is hard to predict** (best M3 AUC 0.650).
- **Logistic:** dims add over DSM **+0.036 [+0.014,+0.057]** (sig); axes≈DSM alone; cross-domain
  (non-circular) +0.011 (ns). **GBoost:** +0.012 [−0.017,+0.041] (ns) — does not robustly replicate.
- **Read:** with a fair de-confounded outcome, the dims **do add a modest, linear-detectable (not
  boosting-robust) increment over DSM**, carried by **internalizing** (residual symptoms → relapse).
  The modern method tempered, not amplified, the result (small linear signal → regularized linear is
  the stabler estimator). Relapse remains hard to predict. **The original "dims useless for relapse"
  was too harsh — a confound artifact.**

## V2-19 · Relapse prediction — can we reach AUC > 0.7? (yes, legitimately, via early-course) — 2026-05-31
Two leakage-safe attempts (OOF AUC, StratifiedGroup/StratifiedKFold, bootstrap by patient, fair DSM,
logistic + HistGradientBoosting):
- **#1 richer baseline** (`scripts/14_relapse_richbaseline.py`): full **75 construct scores** vs the
  6 axes on the remission-based person-intervals → best AUC **0.636** (gboost rich) / 0.631 (logistic
  axes); Δ(rich vs axes) +0.027 [−0.002,+0.057] **ns**. → the 6-axis compression cost ~nothing;
  **baseline-only relapse tops out ~0.64 — cannot reach 0.7 with more baseline features.**
- **#2 early-course prognosis** (`scripts/15_relapse_trajectory.py`): predict **V1→V2** relapse
  (remitted-at-V1, CGI_V1 controlled) from **V0+V1** (early trajectory: ΔCGI, V1 axes, V1−V0 Δaxes).
  n=989 (BP-heavy), 22% relapse. **AUC ≈ 0.70** (gboost 0.696 full-sample; logistic 0.702 complete-case)
  — reaches the target. Beats **DSM +0.05 [+0.01,+0.10]** and baseline severity +0.08 (both sig). The
  trajectory-over-V0-dims increment is modest/method-dependent (gboost +0.053 sig; logistic +0.001 ns)
  — much of the gain is the V0+V1 dimensional profile.
- **Verdict:** >0.7 is reachable **legitimately only by using early-course (V0+V1) data** — a different,
  clinically reasonable question (early-response prognosis) — NOT by enriching baseline (stays ~0.64).
  No confound/leakage reintroduced (regression-to-mean stays removed, CGI_V1 controlled, OOF CV, fair
  DSM). Honest: right *at* 0.70, BP-dominated, requires the first follow-up year.

## VALIDATION ARM COMPLETE — overall verdict
The v2 dimensional model is **rigorous and partially useful, not transformative**:
- **Solid & validated:** 4 reproducible axes, no p-factor, no subtypes (dimensional); confound-clean &
  cohort-residualization-robust (A); longitudinally coherent (C).
- **Novel insight (B, headline):** symptoms ⊥ biology; the p-factor is a symptom-only artifact.
- **Honest limitations:** internalizing is BP+DR-anchored (SZ proxy); cognition baseline-anchored;
  illness-course fixed-historical; cardiometabolic is the most measurement-robust axis.
- **Clinical utility (D + D-refined):** modest but real — the dims match/beat DSM and add incremental
  prognosis over it: **robustly for functioning** (GAF/FAST, ΔR²~0.04) and **modestly for relapse**
  once the regression-to-mean confound is removed (remission-based discrete-time survival: dims add
  ΔAUC +0.036 over DSM by logistic, borderline/ns by gradient boosting; internalizing-carried). The
  earlier "dims useless for relapse" was a confounded-outcome artifact. Honest bottom line: a
  trans-diagnostic dimensional account at least DSM-equivalent for prognosis, adding a small-to-modest
  incremental forecast of functioning and (de-confounded) relapse.

## V2-20 · Phase 6 — manuscript, figures, golden tests + verify.py — 2026-05-31
- **Manuscript written** (leads with Study B; honest D verdict; measurement-design limits up front; full
  math formalism + pipeline + derivations in Methods; 12-point reviewer-anticipation subsection).
  Source `results/manuscript/manuscript.md` → `FACE_trans_diagnostic_v2.docx` via pandoc (editable OMML
  equations; OOXML-validated) + companion `.pdf`. Plan: `docs/planning/MANUSCRIPT_PLAN.md`.
- **6 figures** (`scripts/figures_manuscript.py` → `results/reports/figures/fig1-6.png`): pipeline ·
  4 axes+Φ₂ · orthogonality heatmap + p-factor dissolution (headline) · continuum · predictive Δ + relapse
  · longitudinal coherence. Reproducible build: `scripts/build_manuscript.py`.
- **Golden tests + verify.py re-baselined to v2.** `tests/test_golden_numbers.py` rewritten to pin the
  manuscript's headline numbers to `results/hfa/` artifacts (cohorts/itemset, construct VAF1, 4-axis
  loadings, no-p-factor ECV 0.36, dimensional-not-categorical, the symptom⊥biology headline, Studies
  A/C/D). `tests/test_axes.py` rewritten to guard `axes` (4 axes + 2 orthogonal). `verify.py` PARTIAL
  threshold 100→75 (v2 adds 86). **`pytest` green: 91 passed; `verify.py` green.** (Golden tests skip on
  a clean clone — `results/hfa/` is gitignored; regenerate via scripts 01–48.)
- **PHASE 6 COMPLETE — the v2 study is fully delivered** (analysis · validation · manuscript · figures · tests).

## V2-21 · Dictionary review — CVLT / fluency / anhedonia + suicide skip-logic — 2026-06-03
A late dictionary review added six variables and decoded the suicide skip-logic; the **K=4 backbone,
the dimensional verdict, and the no-p-factor result all held**.
- **Added** (BP/SZ verbal memory + 3-cohort fluency + QIDS anhedonia): CVLT total / short-delay /
  long-delay free recall, verbal fluency (phonemic + semantic), QIDS item-13 anhedonia → item set
  **188 → 194**; first-order constructs **88 → 94**; Stage-3 input (coverage ≥30%) **75 → 81**.
- **Suicide skip-logic decoded** (`src/trans_diag/skip_logic.py`): the ISF gate items propagate
  structural zeros to their dependents, recovering attempt-count coverage (≈25–38% → 72–92%).
- **What changed:** the **cognition axis is now memory-anchored** (CVLT leads dim2; its sign flipped,
  magnitudes preserved); **anhedonia joined internalizing**; **ECV 0.36 → 0.34** (still no p-factor);
  the weak-axes caveat gained **cardiometabolic** (DR n=552 underpowered). **Suicidality stayed
  orthogonal** even after its coverage was recovered → its standalone status is structural, not a
  missing-data artifact.
- **Synced:** `results/manuscript/manuscript.md`, `tests/test_golden_numbers.py`, `tests/test_skip_logic.py`;
  the v2 docs (`FINDINGS`/`PIPELINE`/`ROADMAP`/`DATA`/`CLAUDE`) re-synced to these numbers in the cleanup.

## V2-22 · Addiction vars added → K=4 collapses to K=3 (illness-course not robust) — 2026-06-04
Two lifetime substance-use-disorder variables (`suoccur_alcool`, `suoccur_cannabis`; BP/SZ 2-cohort
PARTIAL, MINI abuse-or-dependence) were added to the dictionary → a `substance_use_disorder` construct
(VAF₁ 0.86). Re-deriving the whole pipeline from zero **changed the headline dimensionality**.
- **The finding:** including the substance construct in the Stage-3 second-order extraction collapses
  the previously-locked **K=4 split-half congruence 0.96 → 0.31**, while K=3 stays reproducible (0.92),
  so "first-collapse-minus-1" now locks **K=3**. **Counterfactual-confirmed**: drop the one construct
  and K=4 returns at 0.96 exactly. The orthogonal, rare-binary addiction construct (loads ≤0.07 on
  every axis) destabilizes the *weakest* factor's rotation — illness-course was never robust.
- **New structure:** **196 items → 95 constructs → 82 Stage-3 inputs → K=3 axes** — internalizing,
  cognition, **cardiometabolic** (was dim4; illness-course/dim3 dropped). Its inverse-burden term
  (`nboccur_hospitalisation_lt` +0.34) re-surfaces on cardiometabolic; age-of-onset core is now
  sub-threshold (max |loading| 0.29). **ECV 0.34 → 0.42** (still < 0.5, no p-factor). mean|Φ₂| 0.12.
- **Every other verdict held:** dimensional (HDBSCAN noise 0.82, bimodality 0.49, DSM-ARI 0.04);
  symptoms⊥biology (0.031); cohort-residualized congruence 0.98; predictive (GAF +0.044, FAST +0.041,
  de-confounded relapse +0.036, early-course AUC 0.70). Substance-use is orthogonal (≤0.07) — a
  carried standalone, like mania/suicidality, not an axis.
- **Engineering:** `src/trans_diag/axes.py` → 3 axes; all K=4-hardcoded downstream (`06`,`07`,`09`–`15`,
  `sensitivity_polychoric`, `figures_manuscript`) made **K-agnostic** (dims read from the loadings /
  `len(AXIS_NAMES)`); `04` now writes `stage3_meta.json` (K, ECV, mean|Φ₂|). Full `00_run_all` re-run
  clean; **golden tests + `test_axes` re-baselined to K=3** (99 pass); manuscript + 6 figures + docx
  rebuilt; CLAUDE/FINDINGS/ROADMAP/DATA/PIPELINE re-synced. User decision: accept the data-driven K=3.

## V2-23 · Dimensionality robustness (bootstrap) + phenotype-feature atlas — 2026-06-05
Investigating *why* 2 variables flipped K, we ran a **bootstrap robustness analysis** (50 cohort-
stratified resamples; `/tmp/bootstrap_dim.py`). It dissolved the K=3-vs-K=4 question:
- **Eigengaps (95% CI):** gap1 2.88 [2.58,3.16], gap2 2.19 [1.98,2.37] — bounded off 0; **gap4 0.11
  [0.02,0.21]** ≈ 0 → λ4≈λ5 is a **degenerate eigenpair** (illness-course ≈ substance), so "the 4th
  axis" is not individually identified.
- **K is a noisy estimator:** the split-half rule gives K=2 (26%), **K=3 (60%)**, K=5 (4%), K=6 (10%).
- **But the FACTORS are robust:** fixing K=6, every factor recovers in 98–100% of resamples
  (internalizing/cognition/cardiometabolic/illness-course/substance 100%, childhood-ADHD 98%).
→ Resolution: the data is **3 weakly-correlated axes + several reproducible ORTHOGONAL standalones**;
"K" conflates "#reproducible factors" (≥6) with "#correlated axes" (3). Pushing K higher only peels off
narrower clusters (ECG RR/QTc), grab-bags, then Heywood (improper) at K≥12 — not new structure.
- **Deliverable — phenotype atlas (feature view):** `docs/PHENOTYPE_ATLAS.md` + `src/trans_diag/phenotype.py`
  (`PHENOTYPE_FACTORS`, `build_phenotype_factors`: masked mean of sign-oriented standardized construct
  scores, no imputation) + `scripts/export_phenotype_features.py` → `results/hfa/phenotype_features.csv`
  (8 factors × score + `__cov` coverage). Atlas axes track Stage-3 dims (|r| 0.97/0.87/0.81); features
  near-orthogonal (mean |r| 0.09). Coverage is the binding constraint: internalizing SZ-proxy (✓50%=0.41),
  substance BP/SZ-only (DR=0.00), illness-course DR=0.48. `tests/test_phenotype.py` (4 pass).
- **Manuscript integration (2026-06-05):** promoted the bootstrap to `scripts/sensitivity_bootstrap_dimensionality.py`
  (committed; writes `results/hfa/bootstrap_dimensionality.json`; added to `00_run_all`). Rewrote §3.1
  (and abstract, Methods Stage-3, Discussion design-limits) from "illness-course is fragile / collapsed"
  → the accurate "**3 correlated backbone + reproducible orthogonal standalones**; factors stable, count
  noisy" framing — which *corrects* a prior inaccuracy (illness-course is 100% reproducible, just
  orthogonal) and *reinforces* the no-p-factor thesis. New **Fig S1** (`figS1_bootstrap` in
  figures_manuscript) = eigengap CIs · K-distribution · factor-stability. Golden test
  `test_bootstrap_dimensionality` pins mode K=3, factor stability ≥95%, gap1/2 CIs clear of 0, gap4≈0.
  Docx rebuilt; 104 tests pass.
