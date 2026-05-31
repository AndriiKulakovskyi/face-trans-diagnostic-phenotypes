# Aggregation rationale — from 190 harmonized variables to nameable trans-diagnostic dimensions

> Methods rationale + empirical evidence. Contributes to `MANUSCRIPT.md` (Methods + Supplement).
> Status: **rationale settled; method (hierarchical/bifactor, hybrid) chosen; analysis plan in
> [HIERARCHICAL_FA_PLAN.md](HIERARCHICAL_FA_PLAN.md).** All numbers below are from internal
> analyses on the v2 V0 data (BP 6252 / SZ 2209 / DR 552; 9013 patients), reproducible via the
> sensitivity scripts noted in the plan.

## TL;DR

After harmonization and type-aware scaling, the V0 matrix holds **190 features**. Feeding them
*directly* to a dimensional, clustering, or deep model is **not** "ready for analysis": two
problems that standardization does **not** touch distort the result —

1. **count / redundancy bias** — a construct's influence scales with *how many items the
   questionnaire happens to have*, not its clinical importance; and
2. **structured missingness** under a strict **no-imputation** rule.

Aggregation into construct scores is one fix. The **current flat-mean implementation is partly
lossy and partly wrong**, but the **headline dimensions are robust to it** (granularity-invariant,
canonical r ≥ 0.85). We therefore move to a **hierarchical / bifactor factor model in *hybrid*
mode** — a clinically-anchored measurement model corrected by the data — which removes the bias
of flat averaging while keeping the axes nameable and the design imputation-free.

---

## 1. The setting

- **190 modelling features** at the baseline visit **V0** (the analysis anchor; V1/V2… are reserved
  to test *temporal coherence*, not to define structure). 193 variables load from the three cohort
  CSVs against the common-variables dictionary; 3 are identifiers, leaving 190 features.
- Features span clinical sections (auto/hetero-questionnaires, suicide, evaluation, antecedents,
  substances, social, hospitalization), **biology** (labs, vitals/ECG), and **cognition** (WAIS/TMT).
- **Ragged granularity.** Some instruments arrive only as a precomputed **total** (MADRS, YMRS,
  Altman, QIDS, EQ-5D…) — there are no constituent items in the CSV; others arrive as **components**
  (C-SSRS 11, ISF 10, PSQI 8, CTQ 8, FAST 7, CGI 5). "Pure item-level" is therefore not even
  available; the matrix is already a mix of granularities.
- **Pervasive, structured missingness**, handled by **masked methods, never imputation** (masked
  pairwise-complete correlation; masked posterior factor scores).
- **Cohort imbalance:** BP 69% / SZ 24% / DR 6% (552 DR patients).

## 2. Why type-aware scaling is necessary but not sufficient

Standardization solves exactly one of three independent problems:

| problem | what it is | fixed by scaling? | fixed by aggregation? |
|---|---|---|---|
| **scale** | labs in mmol/L vs Likert 0–3 vs TMT seconds | **yes** (→ [−1,1]) | n/a |
| **count / redundancy** | a 30-item construct occupies 30 axes; a 1-item construct occupies 1 | **no** | yes |
| **missingness** | no patient is complete; 60–70% of SZ item-pairs never co-observed | **no** | partially |

Scaling equalizes per-**column** variance. It cannot equalize per-**construct** mass, because count
is a *grouping* property, not a scale property. The two remaining problems are what motivate
aggregation.

## 3. The count / redundancy problem (mechanism)

A dimensional/clustering/reconstruction method sees patients as points whose axes are the columns,
and judges everything by sums over columns (distance, covariance, or squared reconstruction error).
The number of columns a construct receives is an **accident of questionnaire design**, yet it sets
the construct's weight. Two faces:

- **Distance (clustering / similarity arm):** the same 1-SD clinical difference contributes in
  proportion to the construct's column count. In a controlled synthetic example (two *equally
  important* constructs, one with 1 item and one with 10 correlated items, all standardized), the
  10-item construct's difference counted **6.4×** more.
- **Covariance (FA / dimensional arm):** correlated ("redundant") items pile their shared variance
  into one large eigenvalue ≈ `1 + (m−1)·ρ`. In the same example (ρ=0.64), **PC1 had eigenvalue
  6.79 and loaded ~0 on the single-item construct and ≈0.32 on each of the ten** — i.e. "the biggest
  dimension of variation" came out as *pure* the 10-item construct, despite equal built-in
  importance. A single-item construct cannot form its own factor.

**Key insight:** m highly-correlated items carry barely more *information* than one, but ~m× the
*geometric weight*. The method mistakes "measured many times" for "large axis of variation." That
mismatch is the bias.

**Why standardization does not help:** it makes each column equal; with unequal columns-per-construct
that is exactly what produces unequal-per-construct (equal × 10 columns = 10×).

**This is method-agnostic.** Any inner-product or squared-error objective inherits it: PCA, FA,
k-means, cosine/spectral similarity, and **autoencoders / VAEs / VQ-VAEs** (a linear AE *is* PCA;
the Gaussian VAE likelihood *is* weighted MSE). Switching to deep models does **not** dissolve the
problem — it inherits it.

**On v2, the bias is real and quantified:** the top 5 instruments occupy **25%** of all item-axes;
the **suicide block alone (C-SSRS+ISF+LTSV+LTSG) = 19.2%** (34 of 177 item-columns).

**Fixes** (both equalize per-construct mass):
- **aggregate** items to one score per construct (what `domains.py` does); or
- **weight** each item by `1/√(construct size)` — fixes count **without** collapsing within-construct
  dimensions (the less-lossy option). In the synthetic example both restored balance (aggregate →
  eigenvalues 1.01/0.99; weight → PC1 loads −1.00 on the single-item construct).

## 4. The missingness problem and the no-imputation rule

Masked pairwise-complete correlation handles missingness for covariance methods without filling any
cell. **AE/VAE/VQ-VAE/GMM cannot** — they require a complete input vector per patient, of which there
are none. They therefore force either **imputation** (which the project forbids — the ablation in
`masked_fa` / LABBOOK E19 shows mean-filling re-imports the cohort×missingness confound,
`corr_fill ≈ O·corr_masked`) or a **masked deep model** that is far more data-hungry than DR (n=552)
can support.

**Aggregation also improves conditioning** (internal analysis, `min_pair=100`):

| | median co-obs / pair | pairs zeroed (<100) | neg-eigen mass (pre-repair) | condition number |
|---|---|---|---|---|
| item (177) | 3,453 | 10.1% | 0.6% | ~1.3×10⁹ |
| domain (69) | 4,102 | 3.4% | 0.0% | 106 |

The pooled item correlation is *carried by BP*: **within SZ only 104/177 items exist and 67% of
item-pairs have <100 co-observed patients** (DR 30%, BP 12%). So an item-level structure is largely
*BP's* structure — a direct threat to a trans-diagnostic claim. Aggregation densifies coverage and
conditions far better.

## 5. Aggregation is a prior; the bias–variance framing

Aggregation = a **hand-coded measurement model** = a strong prior. Strong prior ⇒ **low variance**
(stable, reproducible) but **higher bias** (commits to a model that, if wrong, distorts the truth
regardless of n). The unweighted masked mean makes **four hidden assumptions**; each, when violated,
*is* a source of bias (and is exactly where the current implementation fails):

| the mean assumes… | violated when… | measured on v2 |
|---|---|---|
| (1) **equal item weights** | some items reflect the construct more | mean ignores loadings entirely |
| (2) **unidimensionality** | the construct is several dimensions | metabolic PA_k=**3**; ctq/psqi/isf/hepatic/renal/cgi PA_k=2 |
| (3) **correct membership** | an item doesn't belong | CTQ *denial* items; EQ-5D mis-split into `eq`/`eq5d` |
| (4) **correct sign** | items point opposite ways | renal r(mean,PC1)=**0.57** until signed |

> **Note (corrects a natural misreading):** the bias is **not** caused by aggregating *similar*
> instruments. Aggregating genuinely **redundant/unidimensional** items is *low-bias* (cholesterol
> VAF1 74%, r(mean,PC1)=1.00). Force-aggregating **dissimilar/orthogonal** items is *maximum* bias.
> The rule is: **collapse the redundant; keep the orthogonal separate.**

## 6. What we established about the *current* flat-mean implementation

**It is partly lossy and partly wrong** (per-construct, internal analysis):

| construct | items | PA_k (Horn) | VAF1 | r(mean,PC1) | verdict |
|---|---|---|---|---|---|
| cholesterol, inflammation, processing_speed, fast, cssrs | — | **1** | 37–74% | 0.99–1.00 | averaging safe |
| **metabolic_syndrome** | 8 | **3** | 40% | 0.94 | collapses 3 real sub-axes |
| **ctq** | 8 | 2 | 50% | **0.76** | multidim + denial-item contamination |
| psqi / isf / hepatic / renal / cgi | — | 2 | 28–52% | 0.57–1.00 | mildly–moderately lossy |
| ltsv / ltsg | 7 / 6 | — | — | — | **0–1 complete cases → unusable as composites** |

Also: **34 of 61 biology/vital labs are silently dropped** (thyroid, vit-D, full blood count,
orthostatic BP, heart rate) — they never enter the model.

**But the headline structure is robust to granularity** — the decisive result. Canonical
correlations between item-level and domain-level masked-FA patient scores (permutation null
≈ 0.04):

| K | canonical correlations | reading |
|---|---|---|
| 5 | **0.97 0.95 0.88** 0.67 0.15 | top 3 lock |
| 6 | **0.97 0.96 0.90 0.81** 0.59 0.44 | top 4 lock |
| 7 | **0.98 0.97 0.94 0.85** 0.71 0.46 0.17 | top 4–5 lock |

→ **The primary trans-diagnostic axes are not an artifact of how items were grouped.** This is the
anti-circularity result the study can cite.

The **tail** dimensions diverge, and item-level additionally resolves real but secondary signals the
flat means drop: a **red-cell-mass / anthropometric** factor and a **heart-rate / autonomic** factor
(both from the 34 dropped labs/vitals — note both are partly **sex-linked**), and a **C-SSRS
ideation-intensity** factor (lost to suicide-item averaging; defined on the ~6% with a complete
C-SSRS). Domains were also **more split-half reproducible** at the locking K (0.91 vs 0.79; crude
matcher — to be confirmed with the project's `select_k`).

## 7. Why not "just do AE / VAE / VQ-VAE / (L)GMM" instead of aggregating

| method | item-count bias | needs complete data | identifiable / replicable | role here |
|---|---|---|---|---|
| masked FA (dimensional arm) | yes (controllable) | **no** (pairwise corr) | yes (rotatable, congruence-locked) | primary |
| spectral similarity (stratification arm) | **severe** (geometry) | no | n/a | needs count control |
| linear AE | = PCA | yes → impute | yes | no advantage |
| VAE / VQ-VAE | yes (MSE/codebook) | **yes → impute** unless MIWAE/HI-VAE | **poor** (non-identifiable; collapse) | nonlinear cross-check only |
| (L)GMM / growth mixture | n/a (per-construct) | yes | over-extracts classes | **different (longitudinal) question**; presupposes constructs |

Deep models **inherit** the count bias (via MSE) and **force the imputation** the project rules out;
their latents are non-identifiable, which fights a study that *locks K by reproducibility* and needs
nameable axes. They are candidates for *secondary* arms (VQ-VAE → discrete subtyping; masked
heterogeneous VAE → nonlinear dimensional cross-check; LGMM → the V1/V2 temporal-coherence test fed
by construct scores), not a free replacement for the measurement-model decision.

## 8. The chosen direction — hierarchical / bifactor FA, hybrid mode

**Estimate the measurement model instead of asserting it.** Two levels:

- first level (items → **construct factors**): `z = Λ₁ f₁ + ε₁`, `Cov(f₁)=Φ₁`;
- second level (construct factors → **general dimensions**): `f₁ = Λ₂ f₂ + ε₂` ⇒ `Φ₁ = Λ₂ Φ₂ Λ₂' + Ψ₂`.

A **Schmid–Leiman / bifactor** re-expression gives every item a loading on one **orthogonal general
factor** + one **specific construct factor**. This **relaxes all four flat-mean assumptions**:
(1) loadings are estimated (not flat 1/m); (2) specific factors model multidimensionality
(metabolic's 3 axes become 3 specifics, not 1 mean — **nothing dropped, just organized**);
(3) misfitting items (CTQ denial) get ~0 loading; (4) sign comes from the data.

**Nameable at both levels:** general = *p-factor / overall severity* (cross-check vs `19_pfactor`);
specifics = named construct dimensions (depression-, mania-, metabolic-, cognition-, suicidality-,
trauma-specific…), orthogonal to the general factor.

**Compatible with the masked / no-imputation design:** the whole model is computed from the masked
correlation matrix (`masked_fa.masked_correlation`), extracted with `paf_loadings`, and scored with
`masked_scores` (observed support only). It is an **extension of `masked_fa.py`, not a replacement
of the design.**

### Hybrid mode = "fine-tune the clinical prior on the data"

We anchor first-order factors to the clinical constructs, then let the data revise them (split
multidimensional constructs, drop/reassign misfitting items, orient signs). The user's analogy
holds: the clinical prior is the *foundation model* (inductive bias → generalization on a small,
unbalanced cohort), and the data-driven correction is the *fine-tuning*. **Refinement of the
analogy:** our prior is **theory-based** (clinical knowledge), not pretrained on a large corpus, so
its failure mode is *misspecification*, not *staleness* — which is precisely why we test and revise
it rather than trust it. Mechanically, "fine-tuning with a prior" maps to **target/Procrustes
rotation toward the clinical loading pattern** or **penalized/Bayesian FA with priors centered on
the clinical loadings** — both make the anchoring strength an explicit, tunable regularizer.

Chosen because the cohort is small and unbalanced (DR=552): a pure exploratory solution risks
un-nameable, cohort-artifact factors (we saw thin, sex-linked tail factors); a pure confirmatory
solution re-imposes the flat prior we are trying to escape. Hybrid is the bias–variance sweet spot.

## 9. Decision log

| # | decision | status |
|---|---|---|
| D1 | Aggregate/weight *something* before any geometry/MSE method — count bias is real (25% from 5 instruments) | **settled** |
| D2 | Keep the masked / no-imputation design; do not adopt imputation-requiring deep models as primary | **settled** |
| D3 | Replace flat masked means with a **hierarchical/bifactor** measurement model | **settled** |
| D4 | First level in **hybrid** mode (clinical anchors, data-revised) | **settled (user, this session)** |
| D5 | Axes must be **nameable / clinically meaningful** at both levels | **settled (requirement)** |
| D6 | Item set: include currently-dropped labs/vitals as candidates? | open → plan §Decisions |
| D7 | Residualize on age/sex/education before factoring vs as covariates in validation? | open → plan §Decisions |
| D8 | Conditional/sparse suicide items (LTSV/LTSG) handling | open → plan §Decisions |
| D9 | Pearson now vs polychoric for ordinal/binary | open → plan §Decisions |
| D10 | Bifactor *general* factor: estimate-and-test (ECV) vs assume | open → plan §Decisions |
