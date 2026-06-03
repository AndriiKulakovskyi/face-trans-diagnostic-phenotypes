# Analysis plan — hierarchical / bifactor measurement model (hybrid), imputation-free

> 📁 **ARCHIVED — executed pre-registration** (moved to `docs/planning/` in the v2 cleanup). This is
> the plan *as written before the analysis was run*, kept for provenance. Some details describe
> *intended* work and differ from what was built; for what was **actually done** see
> [../PIPELINE.md](../PIPELINE.md) and [../LABBOOK.md](../LABBOOK.md).

> **Pre-registration-style plan. Review and lock before coding.** Rationale:
> [../AGGREGATION_RATIONALE.md](../AGGREGATION_RATIONALE.md). Engine: extends `src/trans_diag/masked_fa.py`.
> Goal: replace flat masked-mean domain scores with a **clinically-anchored, data-revised
> (hybrid) hierarchical/bifactor** model that yields **nameable** trans-diagnostic axes while
> staying **masked / no-imputation**. Anchor visit **V0**.

## Objectives & success criteria

1. A **first-order measurement model** (items → construct factors) that is data-supported, not
   hand-asserted, and clinically nameable.
2. A **second-order / bifactor** layer (construct factors → general dimensions) giving nameable
   trans-diagnostic axes + orthogonal specific dimensions.
3. **Success =** (a) axes reproducible (split-half min Tucker congruence ≥ 0.85, the project's
   `select_k` criterion); (b) headline axes granularity-invariant (canonical r ≥ 0.85 vs flat-domain
   and item-level solutions); (c) the *specific* factors **recover** the signals flat means dropped
   (metabolic sub-axes, C-SSRS intensity, labs/vitals factors); (d) every axis survives a confound
   battery (not mostly cohort/site/sex/missingness); (e) each axis has a clinical name.

## Cross-cutting conventions

- **Estimator:** masked pairwise-complete correlation → nearest-PD (`masked_correlation`,
  `min_pair=100`); PAF extraction (`paf_loadings`); rotations as specified per stage; masked
  posterior scores (`masked_scores`) on observed support only. **No cell is ever imputed.**
- **Standardization:** robust-z per item (`domains._robust_z`; log1p for heavy-skew labs).
- **Determinism:** fixed seeds; cohort-stratified shuffles for split-half (the matrix is
  cohort-ordered).
- **K selection:** split-half **Tucker congruence ≥ 0.85** (primary, via 07's `select_k`), Horn
  parallel analysis (secondary/scree). Applied at **both** levels.
- **Reproducibility:** every stage is a numbered, re-runnable script writing aggregates to
  `results/` and a section to a single `results/reports/hierarchical_fa.html`. Promote the
  exploratory `/tmp` analyses to `scripts/sensitivity_aggregation.py` first (so the rationale numbers
  are reproducible).
- **Fit without complete data:** standard CFI/TLI/RMSEA assume complete-data ML; we report instead
  **residual RMSR** (model-implied vs observed masked correlation), **% residuals > 0.10**, common
  variance explained, and **cross-validated congruence**. State this explicitly (reviewer-facing).

---

## Stage 0 — Preliminaries (fix inputs; resolve the 5 decisions below)

**Purpose:** lock the item set, covariate handling, and estimator settings so Stages 1–4 are
deterministic.

- Build the V0 candidate-item matrix from the modelling sections (symptom + biology + cognition +
  vitals/ECG), excluding identifiers, administrative (`siteid_city`), and non-phenotype
  treatment/pregnancy markers.
- Apply the decisions D6–D10 (below).
- Emit: the frozen item list (with section, dtype, cohort coverage, V0 observed fraction), the
  masked correlation matrix, and factorability diagnostics (KMO, Bartlett).

**Outputs:** `results/hfa_items.csv`, `results/hfa_masked_corr_V0.parquet`, factorability summary.

## Stage 1 — Exploratory first-order structure (the data-driven look) — *fork-agnostic*

**Purpose:** *see* the empirical construct structure before committing anchors — where the clinical
grouping is confirmed, where it splits/merges, which items are orphans/cross-loaders.

**Method:**
- EFA on the masked item correlation; **oblique rotation** (oblimin/promax — constructs are
  correlated; we want Φ₁). Number of first-order factors by parallel analysis + a scree/eigen report.
- Map each item to its dominant empirical factor; build an **agreement matrix** vs the clinical
  grouping (`domains.py` constructs + the An2 unidimensionality verdicts).
- Cohort-stratified check: does any factor exist only in BP (i.e. vanishes when BP is dropped)?

**Outputs:** first-order loading table (oblique), item→empirical-factor map, agreement-vs-clinical
matrix, list of: confirmed constructs, constructs to **split** (e.g. metabolic), items to **drop /
reassign** (e.g. CTQ denial, EQ-5D), candidate **new** constructs from currently-dropped labs/vitals.

**Decision fed forward:** the revised construct definitions used to anchor Stage 2.

**Risks/mitigations:** rotational indeterminacy → report rotation + sensitivity; thin SZ support →
report per-factor cohort coverage; over-extraction at this N → cap by parallel analysis + congruence.

## Stage 2 — Hybrid first-order measurement model (anchored + data-revised)

**Purpose:** the data-driven *constructs* — the replacement for `build_domain_scores`' flat means.

**Method (the "fine-tuning" step):**
- Specify a **target loading pattern** from the clinical constructs (each item → its construct),
  *revised by Stage 1*: split multidimensional constructs (An2 PA_k ≥ 2 → metabolic into
  adiposity/lipids/glycemia/BP; PSQI into its components; etc.), drop misfitting items, orient signs
  to "higher = more pathological."
- Estimate loadings anchored to that target via **target/Procrustes rotation** of the PAF solution
  (primary), and as a sensitivity a **penalized estimate** with priors centered on the target
  (anchoring strength = explicit regularizer; the foundation-model fine-tuning analogy made
  mechanical). Anchoring strength is a reported hyperparameter, chosen by split-half stability.
- First-order factor scores via `masked_scores` (observed support only).

**Fit:** residual RMSR, % residuals > 0.10, per-construct common variance; compare anchored vs
free vs flat-mean.

**Outputs:** first-order model spec (item → factor(s), loadings, signs), first-order scores,
fit report. **This replaces flat domain scores as the model's construct layer.**

**Decision fed forward:** locked first-order factor set + Φ₁ (their correlation).

## Stage 3 — Second-order / bifactor general dimensions

**Purpose:** the **nameable trans-diagnostic axes** + orthogonal specific dimensions.

**Method:**
- Factor Φ₁ (first-order factor correlation) → second-order loadings Λ₂; **Schmid–Leiman** transform
  → bifactor (each item: general loading Λ₁Λ₂ + specific loading). Cross-check with a **direct
  bifactor rotation** (e.g. bifactor-T / Jennrich–Bentler).
- **General-factor warrant (D10):** report **ECV** (explained common variance), ωH (hierarchical
  omega), and % uncontaminated correlations — *test* whether a single general factor is justified vs
  a correlated-factors (higher-order, no forced general) solution. Report **both**; do not assume a
  p-factor.
- **K (number of general dimensions):** split-half Tucker congruence ≥ 0.85 (primary) + parallel
  analysis (secondary).
- Scores: `masked_scores` for general + specific factors.
- **Naming:** general = overall severity / p-factor (cross-check vs `19_pfactor`); specifics named by
  top-loading constructs; lock order by SS-loadings and record in `axes.py` (note: current `axes.py`
  names are **stale v1** — to be re-derived here).

**Outputs:** general+specific loading tables and scores, ECV/ωH report, locked K, axis names.

## Stage 4 — Validation (against our worries + the reviewer battery)

1. **Granularity invariance** — canonical correlation of the general axes vs (a) flat-domain axes and
   (b) item-level axes. Target: headline axes r ≥ 0.85 (anti-circularity).
2. **Recovery of dropped signal** — do bifactor **specific** factors capture the metabolic sub-axes,
   **C-SSRS intensity**, and the **red-cell/anthropometric** & **heart-rate/autonomic** factors that
   flat means lost? (Correlate specifics with those item blocks.)
3. **Reproducibility** — split-half min Tucker congruence (full hierarchical solution) + **leave-one-
   cohort-out** (structure must survive dropping BP, and dropping DR).
4. **Confound battery** — regress every axis on cohort, site, sex, age, education, and per-patient
   missingness rate; flag any axis that is mostly confound (esp. the sex-linked red-cell factor).
   Reuses the `15_review_checks` ethos.
5. **Clinical face validity** — axis ↔ known constructs (depression, mania, metabolic, cognition,
   suicidality, trauma); compare against the DSM `arm` labels via η² / ARI (labels never modelled).
6. **Temporal-coherence preview** — project axes onto V1/V2 (existing longitudinal scripts) as a
   sanity check (full longitudinal validation remains its own phase).

**Go/no-go:** adopt hierarchical scores as the model inputs only if 1–5 pass; else document and fall
back to weighted-item or revised flat domains.

---

## Decisions to lock now (Stage 0) — my recommendations

| # | decision | options | **recommendation** | why |
|---|---|---|---|---|
| **D6** | item set | (a) 177 current; (b) **+ the 34 dropped labs/vitals as candidates** | **(b)**, excluding treatment/pregnancy markers (oxcarbaz, β-hCG, clozapine) | An3 showed dropped labs/vitals carry real factors; let the bifactor decide if they form specifics — don't silently drop |
| **D7** | covariates | (a) residualize age/sex/education before factoring; (b) covariates in validation only | **primary = residualize age+sex; education as covariate**; report **un-residualized as sensitivity** | matches the clustering arm; but the red-cell/HR factors are partly sex-linked, so the sensitivity is essential to see what residualization removes |
| **D8** | conditional suicide items (LTSV/LTSG) | (a) keep in global model; (b) **exclude; model attempt-lethality separately in attempters** | **(b)** | 0–1 complete cases; structurally mutually-exclusive method items — invalid in a global factor model |
| **D9** | correlation type | (a) **Pearson now**; (b) polychoric for ordinal/binary | **(a) primary; (b) phase-2 sensitivity** | matches current estimator; polychoric is more correct but needs dense co-observation (thin in SZ) — stage it |
| **D10** | general factor | (a) assume p-factor; (b) **estimate and test (ECV/ωH)** | **(b)** | the p-factor is contested; report bifactor *and* correlated-factors, let warrant decide |

## What this changes in the codebase (later, not now)

- New module `src/trans_diag/hierarchical_fa.py` extending `masked_fa.py` (target rotation,
  Schmid–Leiman, ECV/ωH, second-order extraction).
- New scripts `scripts/30–34_*` (one per stage) + `scripts/sensitivity_aggregation.py` (promotes the
  rationale analyses).
- `domains.build_domain_scores` retained for comparison/sensitivity, not deleted.
- `axes.py` re-derived (current names are stale v1).

## Out of scope (separate phases)

Deep nonlinear cross-checks (masked heterogeneous VAE), VQ-VAE subtyping, LGMM temporal classes —
candidates for later arms; each would consume the construct scores produced here.
