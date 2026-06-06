# LABBOOK — FACE V3 precision psychiatry — research notebook

Chronological trace of the **V3** study — what we did, observed, decided, and **why**, step by step.
Results log (numbers): [`FINDINGS.md`](FINDINGS.md). Plan of record: [`V3_PLAN.md`](V3_PLAN.md) ·
roadmap: [`ROADMAP.md`](ROADMAP.md).

> Convention: one entry per step. Each entry = **What we did · Results · Observations · Conclusion /
> decision · Next**. Code lives in `scripts/v3/`, aggregate outputs in `results/v3/` (gitignored).

---

## V3-0 · Project setup: precision psychiatry — 2026-06-05

**What we did.** Adopted the V3 plan as the single source of truth ([`V3_PLAN.md`](V3_PLAN.md)); set up
the current-facing docs (README, CLAUDE.md, ROADMAP, PIPELINE, DATA, FINDINGS) to lead with V3.

**Conclusion.** Direction fixed: diagnostic cohorts → hybrid transdiagnostic dimension discovery →
validated patient strata → prognosis/treatment decision models. Primary engine = patient-level
Bayesian latent model; FIML = confirmatory. The 10 candidate dimensions are a **soft ontology**, not
fixed scores. (Commit `05a33e5` on branch `v3`.)

---

## V3-1 · Eligibility & data-contract audit (Phases A·B·C) — 2026-06-05

**What we did.** Built `configs/candidate_dimensions_v3.yaml` (curated soft-ontology → indicator map)
and `scripts/v3/01_eligibility_audit.py`: for each candidate dimension, resolve its FACE indicators,
attach **per-cohort observed coverage at V0** (no imputation), assign a **likelihood family** per
variable (from dtype), a **missingness taxonomy**, and emit the **soft prior loading matrix**.
Outputs: `results/v3/eligibility/`, `configs/likelihood_map_v3.yaml`, `configs/soft_loading_priors_v3.csv`.

**Results.** V0 N=9,013; 198 usable features. Verdict on the 10 candidates (+4 data-implied):
- **Core, 3-cohort:** overall_severity (=G), cognition (cov ≈ .67/.76/.56), metabolism (≈ .78/.78/.72;
  set up to test the metabolic/inflammatory split), **sleep_circadian** (PSQI total+7 subscores +
  Epworth `ess0109` + morningness `csm`; ≈ .94/.56/.62), mania_activation (YMRS 3-cohort).
- **Core but caveated:** suicidality — present 3-cohort but sparse + **cohort-heterogeneous**
  (BP C-SSRS skip-gated .05 vs SZ ISF .93).
- **Extension (BP/DR; 0% in SZ):** affective_internalizing (MADRS/QIDS/STAI), anhedonia (single item).
- **Module / historical:** substance (BP/SZ), neurodevelopment (proxy), illness_course (historical).
- **UNSUPPORTED in the common dictionary:** negative_symptoms (no PANSS/SANS), sensory_abnormalities
  (0 indicators), impulsivity (no Barratt/UPPS).
- Likelihoods: 95 Gaussian · 40 Bernoulli · 38 ordered-logit · 20 lognormal · 3 neg-binomial.

**Observations.** The audit did its gating job: **3 of the 10 candidates aren't directly measured** and
were stopped before they could be forced into a model. Sleep turned out *stronger* than expected
(confirmed 3-cohort, even a circadian indicator). Mood is BP/DR-only by design.

**Conclusion / decision.** Implied core model: bifactor **G + {cognition, metabolic, inflammatory,
sleep, suicidality}** (3-cohort) + {affective, anhedonia} BP/DR extension + {mania, substance,
neurodevelopment, illness_course} modules. Sensory/negative-symptoms/impulsivity dropped from the core.

**Next.** Missingness atlas to decide MAR vs informative before modeling.

---

## V3-2 · Missingness atlas (Phase B) — 2026-06-05

**What we did.** `scripts/v3/02_missingness_atlas.py`: the observation matrix `R_ij` summarized by
cohort/site/section/dimension, and **per-variable observation-probability models**
(`observed ~ cohort + age + sex + severity`, severity = z(CGI-S, −GAF), fit within each variable's
designed cohorts) to classify missingness as structural / sporadic / informative. Output:
`results/v3/missingness/`.

**Results.** Overall V0 missingness: **BP 36% · SZ 58% · DR 43%**. By block: SZ labs 72% missing, SZ
hetero-Q 73%, DR neuropsych 58%, DR substances 72%, SUICIDE ≈ 62–73% all (skip); demographics 0%.
Drivers: **97 sporadic (MAR-safe) · 67 design/cohort (structural) · 28 informative (severity-related)
· 2 const.**

**Observations.** The headline is **informative (MNAR) missingness in cognition**: TMT-A/B, all three
CVLT recalls, and WAIS digit-span are *less* likely observed as severity rises (p<1e-3) — sicker
patients don't complete neuropsychology. Also `csm`, `wurs`, FAST items, perinatal, some suicide.
SZ biology rests on thin observed support (72% labs missing).

**Conclusion / decision.** MAR observed-likelihood is fine for the ~97 sporadic + structural-by-design
variables; **cognition (and the 28 flagged variables) need a Phase-F missingness-sensitivity arm**
(model `R_ij` jointly), not naive MAR. Expect low per-cohort SZ reliability for the biology factor.

**Next.** Build the Bayesian core engine, with a cognition MNAR arm.

---

## V3-3 · Bayesian core engine (Phase F prototype) — 2026-06-05

**What we did.** `scripts/v3/03_bayesian_core.py` (PyMC 6): patient-level factor model with
**observed-cell likelihood — no imputation** (long (patient,indicator,value) table) on the 3-cohort
**continuous core** (20 indicators: cognition·metabolic·inflammatory·sleep), cohort-stratified
subsample N=1,500. Two parameterizations: a **bifactor** (G + specifics, + a cognition MNAR selection
arm) and a **correlated 4-factor simple structure** (LKJ; yields the factor-correlation matrix Φ).
Output: `results/v3/bayesian/`.

**Results.**
- Bifactor: **weakly identified** (max R-hat 2.04, ESS 2, 0 divergences).
- Correlated 4-factor: **near-converged** (max R-hat 1.06, 39 divergences, ESS 31 — **provisional**).
  Clean loadings (psqi 0.99, wstcir 0.98, wbc 0.89, tmt_b 0.85). **Φ: mean |off-diagonal| ≈ 0.12;**
  cognition×metabolic 0.16, cognition×inflammatory 0.10, cognition×sleep 0.07, **metabolic×inflammatory
  0.28**, sleep ≈ orthogonal to all.

**Observations.** The bifactor's non-identification is *itself* evidence: there is **no dominant
general factor**, so `G` had nothing stable to be. The correlated model says the factors are **weakly
correlated, not one backbone**; **cognition ≈ orthogonal to biology** (under observed-likelihood);
**metabolic and inflammatory are separable** (0.28).

**Conclusion / decision.** Engine + no-imputation pipeline **validated**; qualitative structure
**stable across runs**. Precise Φ values are **provisional** (R-hat 1.06 ≠ the 1.01 bar). This is the
continuous core only (no suicidality/affective yet) on a 1,500 subsample. (Checkpoint commit `89eb0f8`.)

**Next (under review — not yet started).** 2a certify convergence (target_accept 0.99, longer run);
2b extend to the full eligible core (suicidality with ordinal/binary/count likelihoods, affective/
anhedonia BP/DR extension, cognition MNAR arm); then ESEM cross-loadings → scale to full N → Phase H
invariance → Phase J strata.

---

## V3-4 · V0 anchor confirmed + cohort-imbalance correction — 2026-06-05

**What we did.** (1) **Confirmed the V0 anchor** — all three V3 scripts use `visit="V0"` (N=9,013 =
BP 6,252 + SZ 2,209 + DR 552; later visits V1 4,073 → V4 773 are reserved for temporal coherence, never
discovery). (2) **Corrected for BP ≫ SZ ≫ DR** (BP is 11× DR) two ways, one per analysis type:
- **Missingness atlas** — the observation-probability models now use **1/n_cohort weights** (rescaled
  to preserve N, via weighted GLM) so each cohort contributes equally to the pooled fit.
- **Bayesian core** — now subsamples the **500 most-complete patients per cohort** (`--select complete`,
  default): balanced *and* denser — observed-cell density **70% → 95%**. (`--select random`, `--full` kept.)

**Results — the correction changed a conclusion.** Re-running the atlas with 1/n_cohort weighting, the
V3-2 **cognition-MNAR signal turns out to be partly BP-driven**: most cognition tests (TMT-A/B, WAIS,
fluency) drop to **sporadic**; only **CVLT (verbal memory)** stays informative. The robust
informative-missingness under balancing is in **suicidality (ISF battery)** and **self-report
questionnaires** (Altman, STAI, PSQI, CSM, ESS) — sicker patients skip self-reports and suicide-detail
items (sev↓obs, p<0.01).

**Observations.** A BP-dominated analysis over-attributed informative missingness to cognition. The
Phase-F MNAR sensitivity arm should target **suicidality + self-reports (+ CVLT)**, not the whole
cognition block.

**Conclusion / decision.** Cohort balancing is now standard for V3 discovery: **1/n_cohort weighting**
for pooled regressions; **most-complete balanced subsample** for the latent model.

**Balanced Bayesian core re-run (done).** Most-complete 500/cohort (84% dense on full 1,500), 4 chains,
tune 1000, ta 0.97. **The structure is robust to the correction** — loadings + Φ essentially unchanged
vs the random run (psqi 0.97, wstcir 0.96, wbc 0.94; Φ: cognition×inflammatory 0.06, metabolic×
inflammatory 0.19, **mean |Φ| ≈ 0.09**, no general factor). **But convergence got WORSE**: R-hat 1.56,
**648 divergences** (vs 39). Denser data + ta 0.97 → *more* divergences ⇒ the bottleneck is the **LKJ
correlation parameterization geometry** (the model carries unused nuisance `stds` from `LKJCholeskyCov`
and re-choleskys the correlation; sharper likelihood exposes the funnel), **not** data or compute.

**Conclusion.** Cohort balancing is correctly implemented and the **qualitative structure survives it**
(stable across random/balanced, 70%/84% dense). Certifying convergence is now a **model-engineering**
task: re-parameterize the factor correlation (LKJCorr or non-centered, drop the nuisance stds; consider
marginalizing the Gaussian factors). Not a "more samples" problem.

**Next.** Re-parameterize the correlated-factor model for clean geometry → certify (R-hat<1.01, ~0 div)
→ then extend (suicidality + affective + MNAR arm on suicidality/self-reports).

---

## V3-5 · Workbook enrichment + marginalized reparameterization (certifies) — 2026-06-05

**What we did — Part A (fold in the external `FACE_dimension_recommendations.xlsx`).**
- **(a) ESS/CSM fix:** PSQI = 3-cohort sleep core; **ESS/CSM moved to a BP/DR circadian extension** (their
  `obs_sz = 0`). Applied to `configs/candidate_dimensions_v3.yaml` *and* the model SPEC.
- **(b) Unit-mislabel QA:** the workbook flagged `psqi/staya/ess0109/eq5d0206/ymrs/fast…` as "mmHg"/"days".
  Checked every one — **the labels are cosmetic; the sanity bounds are all correct** (psqi 0–21, ymrs 0–60,
  fast 0–72…) and observed data sits inside them. No data loss; no dictionary edit needed.
- **(c) Roles + metabolism trim:** metabolism trimmed from the broad BILAN sweep (60) to the **~26-var
  cardiometabolic+inflammatory core** (peripheral liver/renal/thyroid/red-cell labs → load on G/excluded);
  per-variable roles added (smoking/MARS/psqi16 → covariate; YMRS/Altman/WURS → proxy).

**What we did — Part B (reparameterize for convergence).** The explicit-latent correlated model diverges
under **both** `LKJCholeskyCov` (648 div, V3-4) and `LKJCorr` (250 div, Heywood loadings) — the per-patient
factor scores funnel. Replaced it with a **MARGINALIZED Gaussian factor model**: integrate the factors out,
likelihood = `MVN(ν, ΛΦΛᵀ + Ψ)` on each patient's **observed** cells, grouped by missingness pattern
(still **no imputation**). Engineering: selection-matrix submatrices (a PyTensor advanced-index rewrite bug),
a `--min-group` pattern-frequency filter (bounds the #Cholesky ops → tractable NUTS gradients), `cores=1`
(macOS multiprocessing hung on the pattern graph), numeric coercion of the ArviZ summary.

**Result.** The marginalized model **CONVERGES** — mini check (200/cohort, 7 patterns): **max R-hat 1.02,
0 divergences, min ESS 171** (vs 1.56–2.26 / 250–648 div for the latent models). The **structure is
unchanged**: cognition×metabolic 0.18, cognition×inflammatory −0.01, **metabolic×inflammatory 0.20**
(separable), sleep ≈ orthogonal, **mean |Φ| ≈ 0.08, no general factor**.

**Conclusion / decision.** The clean estimator for V3 discovery is the **marginalized** correlated-factor
model (no latent funnel, certifies, doubles as the FIML observed-likelihood confirmatory model). The
structural finding (no general factor · cognition ≈ ⊥ biology · metabolic/inflammatory separable) is now
robust across **five** runs and certified. Trade-off logged: `--min-group` drops a small rare-pattern tail
(~14% at the mini scale) for tractable gradients.

**Full certified balanced run (done).** 500 most-complete/cohort (N=1,500; 86% dense), 4 chains, tune 800,
ta 0.95, 17 patterns (294/1,500 ≈ **20% rare-pattern tail dropped** by `--min-group 10`): **max R-hat
1.010 · min ESS 1,863 · 0 divergences — CERTIFIED.** Loadings clean (psqi 1.00, wstcir 0.97, wbc 0.94,
tmt_b 0.74). **Φ: cognition×metabolic 0.22 · cognition×inflammatory 0.05 · metabolic×inflammatory 0.17 ·
sleep ≈ orthogonal; mean |Φ| ≈ 0.09 — no general factor.** First properly-converged V3 measurement model;
structure identical to the five provisional runs.

**Caveat.** The certified fit is on the 17 common patterns (≈1,206 patients); the ~20% rare-pattern tail is
a mild extra completeness selection — lower `--min-group` (more Cholesky ops) to keep more, next iteration.

**Next.** Extend to suicidality (ordinal/binary/count) + affective/anhedonia (BP/DR) + cognition MNAR arm;
reduce the rare-pattern drop; then Phase H invariance → Phase J strata.

---

## V3-6 · Extended model (affective + mixed-likelihood suicidality) + visualizations — 2026-06-05

**What we did.** Extended the certified core (03) → `scripts/v3/04_extended_model.py`: the marginalized
Gaussian block grew to **5 factors** by adding **affective** (MADRS/QIDS/STAI/anhedonia, BP/DR) — so the
symptom⊥biology correlation is estimated *inside* the model, not post-hoc — plus an **explicit-latent
suicidality module with mixed likelihoods** (7 ISF Bernoulli + 1 negative-binomial attempt count).
`scripts/v3/05_visualize.py` → aggregate figures in `docs/figures/v3/`; results page
[`V3_RESULTS.md`](V3_RESULTS.md).

**Result — CERTIFIED** (N=1,500 balanced, 4 chains): max R-hat **1.020** · ESS **1,066** · **0 div**.
**Φ (5 factors + suicidality): mean |off-diag| ≈ 0.18 (0.12 excl. sleep-affective) — no general factor.**
- **Affective ⊥ biology:** affective×inflammatory **0.07**, ×metabolic **0.15** → symptoms ≈ orthogonal to
  biology, as a *within-model* correlation (a strong form of the claim).
- **Sleep × affective = 0.68** — the one strong edge; PSQI tightly coupled to depression in BP/DR (flag:
  PSQI may partly index depression-driven sleep complaints; BP/DR-specific).
- **Metabolic × inflammatory 0.20** (separable); **cognition** tracks metabolic 0.26 / affective 0.23.
- **Suicidality** ≈ orthogonal to biology/cognition; modest link to affective (0.14) + sleep (0.17) —
  distress-linked near-standalone.

**Observations / discussion.** (1) Cohort scores **validate** the dimensions clinically: **cognition
worst in SZ, sleep/affective worst in DR, biology flat across cohorts** (truly transdiagnostic). (2) The
sleep-affective coupling is the most interesting new result — questions sleep/affect separability in the
mood cohorts; motivates a residualized-sleep sensitivity. (3) **SZ affective is a proxy** (no SZ affective
indicators) — read with care. (4) The MNAR arm is **not identifiable** on the most-complete subsample
(everyone's complete) → the MNAR result stays in the full-sample atlas (V3-2/V3-4).

**Conclusion.** The certified estimator establishes: no general factor · symptoms ⊥ biology ·
metabolic/inflammatory split.

**Next.** Sleep-affective sensitivity (residualize); shrink the ~23% rare-pattern drop; temporal coherence
+ measurement invariance (Phase H) over V1–V4; then probabilistic strata (Phase J).

---

## V3-7 · Sleep↔affective sensitivity → objective-sleep model (canonical) — 2026-06-05

**What we did.** Investigated the sleep×affective = 0.68 from the extended model. (1) **Item decomposition**
(`scripts/v3/06_sleep_affect_sensitivity.py`): pairwise-complete correlation of each PSQI sub-item with affective
severity (BP/DR) → **objective** parameters (efficiency 0.22, duration 0.19, latency 0.28; composite 0.31)
vs **subjective** items (disturbance 0.34, quality 0.45, **daytime-dysfunction 0.59**; composite 0.61).
(2) **Factor confirmation**: refit the extended model with the objective sleep items only
(`scripts/v3/04_extended_model.py --sleep objective`).

**Result.** Factor-level sleep×affective drops **0.68 → 0.54**, model still **CERTIFIED** (R-hat 1.010,
ESS 991, 0 div). → ~0.14 of the coupling was PSQI method overlap (the depression-overlapping subjective
items, esp. daytime-dysfunction = fatigue/anhedonia); the residual **0.54 is a genuine construct-level
sleep–affect relationship**.

**Conclusion / decision.** Sleep is a **separable** dimension but genuinely moderately correlated with
affect in mood disorders (not an artifact, not a merger). **Adopted the objective sleep factor as
canonical** (PSQI efficiency/duration/latency); the depression-contaminated PSQI items are dropped.
`04 --sleep` default → `objective`. Canonical Φ + figures + discussion: [`V3_RESULTS.md`](V3_RESULTS.md).

**Next.** Phase H temporal coherence (V1–V4) + measurement invariance; shrink the ~23% rare-pattern drop;
Phase J probabilistic strata.

---

## V3-8 — Measurement-layer rebuild: config-first soft-prior ESEM-bifactor (Stages 0–1)

**Why.** A self-audit (holding the *initial soft-prior map* against the *derived model*) found the
certified core was structurally incomplete: the soft-prior matrix never fed the model (hard-coded SPEC,
strict simple structure, no cross-loadings, no general factor), and **"no general factor" had been
concluded from a model that omitted the very severity indicators G should be built from**. Rebuilt the
measurement layer **config-first** so every conclusion is *earned*: `configs/{dimensions,priors,
likelihoods,bayesian_model}.yaml` → `prior_loading_matrix_v3.csv` (96 items × 10 factors, bifactor-
identified G) → `src/v3/latent_models/bayesian` engine that **consumes the prior matrix** (full Λ:
sign-anchored primaries + shrunk cross/bifactor cells), holds **G orthogonal to specifics**, marginalizes
the Gaussian block, and shares Φ with an explicit Z block for non-Gaussian indicators. Staged build,
each certifying (R-hat≤1.01, 0 div) before advancing.

**Stage 0 (reproduce).** Config-first engine in certified mode (simple structure, HalfNormal) reproduces
the certified core **exactly**: loadings identical (psqi11 0.69, qidsr120 1.00, …), Thomson-score
correlations identical (sleep×affective **0.54**, cognition×affective 0.30, metabolic×inflammatory 0.20).
R-hat 1.010, ESS 1857, 0 div. *Exposed + fixed a double-`item_sign` bug* (data and loading both flipped).
**Estimand clarification:** the engine reports the **model Φ parameter** (the principled latent
correlation; sleep×affective **0.40**) *and* the score correlation (0.54). The old 0.54 was the score-
based number; model Φ says sleep is *more* separable from affect than previously reported.

**Stage 1 (the decisive test — does a dedicated-anchor general factor identify?).** **CERTIFIED** (R-hat
1.010, ESS 1533, 0 div, no Heywood). **G identifies** — overturning the premature "no general factor."
Read-out:

- **G is anchored by functional impairment**: FAST items 0.73–1.04, EGF 0.75, EQ-5D 0.60. (CGI severity is
  ordinal — joins at Stage 3 via the explicit-Z block; today G is functioning-anchored. fast29 ≈ 0,
  one weak FAST item.)
- **G is shared with symptoms, not biology.** Specific indicators loading on G: **affective strongly**
  (madrs 0.82, qids 0.69, anhedonia 0.57, STAI 0.54), **cognition moderately** (cvlt 0.32, verbal-fluency
  0.30, processing-speed 0.28, TMT-B 0.25), **metabolic/inflammatory ≈ 0** (all <0.19, most <0.1).
  mean |G loading| over specifics = 0.23; 6/23 load ≥0.3 (the 4 affective + 2 cognition).
- **The specific dimensions survive G.** Affective remains a distinct factor (specific loadings 0.41–0.68
  *on top of* its G loadings); model-Φ metabolic×inflammatory **0.17**, sleep×affective **0.32**,
  cognition×metabolic 0.18 — the biological cluster and sleep stay intact and **off the general axis**.

**Verdict (earned, not assumed).** There **is** a general factor, but it is **not a broad p-factor** — it
is a **functional-impairment / clinical-distress axis** (functioning + mood + some cognition), and it is
**orthogonal to metabolic/inflammatory biology**. So the old headline splits: *"no general factor"* is
**overturned**; *"symptoms/severity ⊥ biology"* is **strengthened** (biology stays off the general axis).
The precision-psychiatry object is **G (impairment/distress) + distinct specific dimensions**, with
biology as its own separable cluster.

**Artifacts.** `results/v3/bayesian/stage{0,1}/{loadings,phi,phi_scores,factor_scores}.csv` +
`stage_report.md` + `diagnostics.json`. Engine: `src/v3/latent_models/bayesian/`; configs as above;
contract tests `tests/v3/`.

**Next.** Stage 2 (free specific↔specific ESEM cross-loadings under soft priors) → Stage 3 (ordinal CGI on
explicit Z — *will sharpen G toward severity*) → Stage 4 (mixed-likelihood suicidality/substance on shared
Z). Then FIML triangulation + the dimension-adjudication table (G adjudicated **confirmed-but-partial**).

---

## V3-8b · Stages 2–3 + structural cleanup — 2026-06-06

**Stage 2 (ESEM cross-loadings) — CERTIFIED** (R-hat 1.01, ESS 800, 0 div). Freeing the theory-motivated
plausible cross-loadings adds little: simple structure mostly holds; specifics stay weakly correlated
(model Φ metabolic×inflammatory 0.17, cognition×metabolic 0.18, sleep×affective 0.32). G unchanged.

**Stage 3 (CGI severity anchors) — NOT yet certified** (R-hat 1.53, ESS 7, 0 div). Adding CGI-S + a
hospitalization count to G de-stabilizes a chain: the zero-inflated `nboccur_hospitalisation_lt`, fit as
lognormal, loads ~0.01 on G (a degenerate direction) and tanks ESS. Mid-fix: dropped it from
`g_anchors_severity` (uncommitted edit). **Stage 4** (mixed-likelihood suicidality/substance) is coded,
not run.

**Structural cleanup (the readability pass).** Created [`STATE.md`](STATE.md) as the single
current-state source. Quarantined the first-generation engine (old `03_bayesian_core` ·
`04_extended_model` · `05_visualize` · `06_sleep_affect_sensitivity`) to `scripts/v3/legacy/`; renumbered
the config-first engine to a contiguous pipeline (`10/11` → `03_build_prior_matrix` /
`04_fit_measurement`). Reconciled the stale "no general factor" headline across `V3_RESULTS` · `FINDINGS`
· `V3_PLAN` · `README` · `CLAUDE`. Full test suite green (115 passed).

**Next.** Certify Stage 3 (re-run after dropping the hospitalization count) → run Stage 4 → regenerate
the results page + figures from the new engine → Phase H invariance.

---

## V3-9 · Delete V2 + first-generation engine — clean V3-only base — 2026-06-06

**What we did.** Removed the V2 benchmark arm (`src/trans_diag/`, the V2 `scripts/*.py` pipeline,
`docs/legacy_v2/`, V2 `results/`+`reports/`) and the first-generation Engine A (`scripts/v3/legacy/`,
the V3-8b quarantine). Re-pointed the four data-layer test files (`test_adapter`, `test_filters`,
`test_skip_logic`, `test_sanity_and_encoding`) from `trans_diag` → `v3.data` and moved them under
`tests/v3/`, so the harmonization / skip-logic / sanity foundation stays tested — now against the V3
fork the pipeline actually uses. Deleted the five V2-only test files. Updated `pyproject.toml`
(`packages = ["src/v3"]`, name `face-v3`) and `conftest.py`.

**Result.** A V3-only tree: `src/v3` · `scripts/v3/01–04` · `configs/` · `tests/v3/` (**84 passing**) ·
V3 docs. **Zero `trans_diag` references remain.** Everything deleted is recoverable from git history.

**Why.** Two parallel engines + a V2 arm + stale docs caused the earlier confusion. A minimal, tested,
single-engine base is the precondition for re-thinking the roadmap (phases/stages) from a place we trust.

**Next.** Re-think the roadmap/plan on this clean base; then certify Stage 3.

---

## Open questions for review

1. ~~**Convergence bar.**~~ **RESOLVED (V3-5):** the marginalized model certifies (R-hat 1.010, 0 div) —
   no need to build on a provisional fit. Remaining sub-question: lower `--min-group` to shrink the ~20%
   rare-pattern drop?
2. ~~**Affective dimension.**~~ **RESOLVED (V3-6):** affective is a strong, well-identified named factor
   (BP/DR) — kept. ~~Sleep↔affective separability?~~ **RESOLVED (V3-7):** sleep is separable; the objective
   sleep factor correlates 0.54 with affect (the 0.68 was PSQI method overlap) — adopted as canonical.
3. **Suicidality DIF.** V3-6 used the common **ISF** core (3-cohort) for suicidality (Bernoulli + NB);
   the BP-specific C-SSRS items were left out. Revisit a DIF-aware C-SSRS extension?
4. ~~**SZ biology reliability.**~~ **ADDRESSED (V3-6):** SZ affective is reported as a *proxy* and SZ
   metabolic rests on thin support — both flagged in [`V3_RESULTS.md`](V3_RESULTS.md). Formal per-cohort
   reliability/invariance is Phase H.
5. **Negative symptoms.** Out of the common-variable core — worth pulling PANSS/SANS from FACE-SZ full
   data as an SZ module later?
