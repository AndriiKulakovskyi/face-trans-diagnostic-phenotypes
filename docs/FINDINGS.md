# FINDINGS — FACE trans-diagnostic study (v2) — running log

Paper-oriented log of empirical + methodological findings on the **v2** dictionary. Every number
must be reproducible from the pipeline. **The v2 analysis is complete** (dimensional + stratification
+ validation A–D; manuscript delivered).

> **Headline (verified against `results/hfa/`, post-2026-06-03 dictionary review):** 9,013 patients;
> **194** V0 items → **94** constructs → **K=4** axes (internalizing · cognition · illness-course ·
> cardiometabolic); **no p-factor** (ECV **0.34**); dimensional, no subtypes. Some intermediate counts
> in the dated sub-sections below predate that review (the item set was 188 → 194); the K=4 backbone
> and every verdict held — see `docs/LABBOOK.md` V2-21.
>
> The v1 findings log is archived at git tag `v1-archive-2026-05-30`. Do **not** carry over v1 numbers.

## Settled — data processing
- **v2 dictionary:** 199 usable variables (READY + PARTIAL, of 223 entries); `qa_harmonization`
  reports all variables load + pass sanity (0 fail).
- **Type-aware scaling to [−1, 1]**; robust-z explosion fixed (prolactin |z|≈106→5); **masked /
  no-imputation** design kept (no hard missingness drop).
- See the 3-part QA report (`results/reports/qa_harmonization.html`) and CLAUDE.md §"Data processing".

## Track 1 — dimensional analysis (v2)

### Measurement model — why hierarchical/bifactor, not flat domains (settled)
Rationale + full evidence: [AGGREGATION_RATIONALE.md](AGGREGATION_RATIONALE.md). Reproduce:
`scripts/sensitivity_aggregation.py`.
- **Standardization ≠ readiness.** Type-aware scaling fixes *scale*; it leaves **count/redundancy
  bias** and **structured missingness**, which distort every inner-product / squared-error method
  (FA, k-means, cosine, AE/VAE/VQ-VAE). On v2: 5 instruments = 25% of item-axes (suicide block 19%);
  item-level masked corr cond ≈1.3e9 vs domain 110; within-SZ 67% of item-pairs < 100 co-obs.
- **Flat means are lossy but the headline is robust.** metabolic PA_k=3, CTQ r(mean,PC1)=0.76; yet
  the top 4–5 dimensions are **granularity-invariant** (canonical r ≥ 0.85, perm-null 0.04) — the
  primary structure is not a grouping artifact (anti-circularity result for the manuscript).
- **Decision:** hierarchical/bifactor in **hybrid** mode (clinical anchors, data-revised), masked /
  no-imputation. Plan: [HIERARCHICAL_FA_PLAN.md](planning/HIERARCHICAL_FA_PLAN.md).

### First-order structure (Stages 0–2, scripts 30–32)
- **Item set:** 194 V0 items (every valid measurement; identifiers/covariates/confounds/branching-
  suicide/collinear excluded). Factorable (scree 12.6, 10.1, 6.9…); near-singular (plain KMO undefined).
- **Data-driven EFA (Stage 1):** 42 nameable first-order factors; **independently confirms** the
  aggregation problems — metabolic splits into adiposity/BP/lipids/cholesterol, CTQ *denial* splits
  from trauma, C-SSRS → severity/intensity, ISF → ideation/attempts; **dropped labs/vitals recovered**
  (autonomic-HR, red-cell, inflammation, vit-D). Substantive factors reproduce leave-BP-out (0.91).
- **Hybrid first-order model (Stage 2):** 84 constructs (within-construct masked 1-factor scores).
  Metabolic split → adiposity VAF1 0.93 / cholesterol 0.90 / BP 0.72 / lipids 0.72 (collapsed was 0.40);
  CTQ cleaned (denial dropped); CGI → severity only. Φ₁: 106/3486 construct pairs |r|>0.3 (max 0.74),
  coherent second-order seeds (madrs~qidsr 0.74; cgi~egf −0.69).
- **Comorbidity flags decomposed (V2-8):** the pooled 24-flag bin (VAF1 0.38) split data-anchored →
  `cardiac_history` (VAF1 0.50) + `atopic_inflammatory` (0.26, weak) + standalone {migraine,
  head_trauma, peptic_ulcer}; the **13 flags <2% prevalence → Stage-4 validators**, not inputs.
- **Limitations:** C-SSRS sparse (6–16% coverage; ISF `suicidal_ideation` 0.91 is the usable
  suicidality dim); somatic-comorbidity constructs are weak (0.26–0.50; thin signal); some lab panels
  weakly unidimensional (electrolytes 0.34, red-cell 0.43). **K of the final trans-diagnostic axes is
  NOT yet locked** — Stage 3 (second-order, split-half congruence, ECV/ωH) pending. Model: 94 constructs.

### Second-order structure (Stage 3, script 33) — PROVISIONAL (pre-validation)
Φ₁ of 81 constructs (coverage ≥30%, standardized) is well-conditioned (0% neg-eigen mass). K by
masked split-half congruence = **4** (reproducible K2–4 at 0.94–0.98; first collapse K5; the naive
"max-K" rule was a *caught bug* → would have over-extracted to K=10 with Heywood loadings). Solution
proper (0 Heywood). **General factor weak: ECV 0.34 → no dominant p-factor** (multidimensional).
**Four reproducible trans-diagnostic dimensions:**
1. **Internalizing** (depression–anxiety–functioning) · 2. **Cognitive impairment** ·
3. **Illness course** (later-onset / lower-chronicity) · 4. **Cardiometabolic–inflammatory**.
- **vs v1:** 4 not 6 — **mania does not form a reproducible axis** (loads <0.30 everywhere despite a
  good construct); later-onset+burden merged. `axes.py` names are stale v1.
### Validation (Stage 4, script 35) — the 4-dim solution PASSES
- **Confound-clean:** no dim explained >0.25 by cohort/sex/age/site/missingness (max: cognition
  cohort η²=0.16, with educ 0.16 a genuine correlate). **No p-factor** (ECV 0.34).
- **Trans-diagnostic + valid:** internalizing highest in DR, cognition worst in SZ; η² cohort ≤0.16
  (dims cut across diagnoses, not cohort markers).
- **Reproducible:** leave-cohort-out Tucker congruence min 0.84 (drop BP) / 0.90 (drop SZ) / 0.99 (drop DR).
- **Granularity-invariant:** vs flat-domain FA canonical r [0.99, 0.93, 0.77, 0.39] — top 3 invariant
  (anti-circularity confirmed); 4th differs because the hierarchical model adds recovered labs/vitals.
- **Mania/suicidality** are valid standalone constructs **orthogonal to the 4 axes** (|r|≤0.09) — they
  do not anchor a second-order factor (not a coverage bug). v2 ≠ v1's mania axis.
- **THE v2 dimensional result (provisional-final):** 4 reproducible trans-diagnostic dimensions —
  **internalizing · cognition · illness-course · cardiometabolic-inflammatory** — defined by
  `results/hfa/stage3_loadings.csv`. K=6 sensitivity (adds cardiac/somatic-history + childhood-
  trauma) also confound-clean.

[TODO — lock axis names in a v2 source-of-truth; dimensional figures; outcomes vs DSM; Phase 5 stratification.]

## Track 2 — patient stratification (v2)
**Verdict: DIMENSIONAL (continuum), not discrete** (`scripts/40_phase5_stratify.py`). Structure test
on **A = 6 axes** (4 dims + mania + suicidal_ideation) and **B = 81 construct scores** (masked engine
embedding):
- **A:** HDBSCAN 0 dense clusters (100% noise); real−null silhouette gap small/non-peaking (0.01–0.05);
  axes unimodal (Sarle ≤0.51); ARI(k-means, DSM) ~0.03 → continuum.
- **B:** the only dense clusters are the **3 cohorts (ARI 1.00)** → the sole categorical structure is
  DSM diagnosis; no novel trans-diagnostic subtypes.
- Caveat: k-means bootstrap stability is high (0.79–0.93) but that is a continuum artifact (k-means
  partitions a blob stably); HDBSCAN (no density clusters) + unimodal axes are decisive.

**Headline (both arms):** FACE trans-diagnostic structure is **dimensional** — 4 reproducible continuous
axes (internalizing, cognition, illness-course, cardiometabolic-inflammatory) + 2 orthogonal standalone
dims (mania, suicidality); **no general p-factor and no discrete subtypes** beyond DSM categories.

## Track 3 — does it matter? (validation; plan: planning/VALIDATION_PLAN_v2.md)
Relapse outcome **locked** (Study D): hospitalization-count REJECTED (lifetime count non-monotone,
41% spurious decreases); **primary = CGI-S relapse by V2** (rise ≥2 / cross <4→≥4), prevalence 20%
(BP 23 / SZ 14 / DR 8), n=3,657. Longitudinal arm is BP+SZ (DR collapses by V3).
- **Study A — cohort confound: axes are NOT a cohort artifact** (`scripts/42_cohort_confound.py`).
  Cohort-residualized re-derivation reproduces all 4 axes ≥0.96; within-BP ≥0.95 → within-cohort
  covariance, not between-cohort means.
- **Study A dig — cross-cohort measurement-coverage asymmetry (KEY).** The within-SZ internalizing
  0.80 is *structural*, not subtle: **internalizing's defining scales (MADRS, QIDS, STAI, FAST, Altman)
  are 0% in SZ** (PARTIAL/BP+DR by design — FACE-SZ used a psychosis battery). So **internalizing is
  directly measured only in BP+DR; SZ scored by 3-cohort proxies (GAF/CGI/PSQI/EQ-5D)**. Cardiometabolic
  is 3-cohort *core* (lipids/adiposity/CRP) + BP+DR peripherals (HR, lymphocytes). **Cognition &
  illness-course are cleanly 3-cohort.** → The *fully* trans-diagnostic axes are cognition, course,
  core-cardiometabolic; the **mood axis is BP+DR-direct / SZ-proxy** — qualify the "trans-diagnostic"
  label on internalizing. Dimensional/no-p-factor/no-subtypes results unaffected.
- **Study B — symptom⊥biology + p-factor is a symptom-only artifact (HEADLINE)**
  (`scripts/43_orthogonality_pfactor.py`, BP+DR primary). Between-block mean |construct r|:
  symptom↔biology **0.03**, symptom↔cognition 0.07, biology↔cognition 0.04 (within: symptom 0.24,
  cognition 0.42); strongest symptom↔biology pair only 0.15. General-factor first-factor share:
  symptom-only **0.33** → +cognition 0.27 → +biology 0.15 → full **0.09**. → *a p-factor is a
  symptom-only artifact; the integrated symptom+biology+cognition space is multidimensional and
  orthogonal-blocked, no general factor.* Robust BP+DR↔pooled.
- **Study C — longitudinal coherence (V0→V1→V2)** (`scripts/44_longitudinal_coherence.py`).
  *Structural invariance* (re-derive per visit, congruence vs V0): internalizing 0.99/0.98,
  cardiometabolic 0.98/0.97 (strong); cognition 0.87@V2 (0@V1 — baseline-anchored battery); illness-
  course 0.87/0.78 → structure persists. *Score stability* (test-retest): cardiometabolic 0.66/0.62
  (trait), internalizing 0.59/0.53 (state/episodic), cognition 0.49@V2, illness-course **0.12/0.16**
  (its age-of-onset core is baseline-only at V1/V2 → fixed-historical, not re-measurable; not
  instability). **Cardiometabolic is the most measurement-robust axis** (3-cohort + longitudinally
  stable). → with Study B, the robust trans-diagnostic substrate is biological/cognitive.
- **Study D — predictive validity vs DSM (the make-or-break): MODEST, functioning-specific**
  (`scripts/45_predictive_validity.py`; out-of-sample CV; M0=age+sex+V0-baseline). Attrition check
  passed (axes→dropout AUC 0.531). **Relapse-by-V2:** baseline CGI-S AUC 0.765; neither DSM nor axes
  add (ns). **GAF@V2:** axes add over DSM ΔR²=+0.046 [+0.030,+0.062]; cross-domain (non-circular) +0.033.
  **FAST@V2 (BP+DR):** axes beat DSM (DSM adds 0; Δ +0.038 [+0.022,+0.053]). Per-axis: illness_course
  leads (+0.017); cognition/cardiometabolic add little individually. **Verdict: partially earns its
  keep** — DSM-equivalent-to-better + modest non-circular functional-prognosis increment (ΔR²~0.04),
  no relapse advantage *under the change-based outcome* (later shown confounded). Report honestly.
- **Study D refined — relapse done right** (`scripts/46_predictive_survival.py`): the change-based
  relapse was confounded by **regression-to-the-mean** (baseline-only AUC 0.765 → **0.578** under a
  remission-based discrete-time-survival outcome: at-risk = V0-remitted CGI≤3, relapse = deterioration
  to CGI≥4; GroupKFold by patient; logistic + gradient-boosting). De-confounded, the dims **do add a
  modest increment over DSM** (logistic ΔAUC **+0.036 [+0.014,+0.057]**; gboost +0.012 ns —
  borderline/method-dependent), internalizing-carried; relapse stays hard to predict (AUC ≤0.65). So
  the earlier "dims useless for relapse" was a confounded-outcome artifact.
- **Relapse — reaching AUC >0.7** (`scripts/47,48`): **(1) richer baseline** (81 constructs vs 6 axes)
  tops out at 0.636 (Δ+0.027 ns) → baseline-only relapse can't break 0.7. **(2) early-course prognosis**
  (predict V1→V2 from V0+V1: ΔCGI + V1 axes + Δaxes, CGI_V1 controlled, leakage-safe) reaches **AUC ≈
  0.70** (gboost 0.696 / logistic 0.702, n=989), beating DSM +0.05 and baseline +0.08. → >0.7 is
  reachable *only* by adding the early course (a different, clinically-sensible prognostic question),
  not by enriching baseline — and without reintroducing the regression-to-mean confound or leakage.

## Overall verdict (validation arm complete)
The v2 dimensional model is **rigorous and partially useful, not transformative.** Solid: 4 reproducible
dimensional axes, no p-factor, no subtypes; confound-clean (A); longitudinally coherent (C). Novel
(B, headline): **symptoms ⊥ biology; the p-factor is a symptom-only artifact.** Honest limits:
internalizing BP+DR-anchored, cognition baseline-anchored, illness-course fixed-historical;
cardiometabolic is the most measurement-robust axis. Utility (D + refined): modest but real —
matches/beats DSM and adds incremental prognosis over it, robustly for *functioning* (ΔR²~0.04) and
modestly for *relapse* once the regression-to-mean confound is removed (de-confounded ΔAUC +0.036,
borderline). → A trans-diagnostic dimensional account at least DSM-equivalent for prognosis, adding a
small-to-modest incremental forecast of functioning and relapse, biologically grounded.
