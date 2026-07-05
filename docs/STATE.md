# STATE — where the project is right now

> **Read this first.** Updated 2026-06-27.

## TL;DR

**Milestone 1 (M1) — the transdiagnostic dimensional map — is COMPLETE** (PI sign-off 2026-06-27), on the FACE
**V0** baseline (N = 9,013). **Findings + discussion: [`M1_FINDINGS.md`](M1_FINDINGS.md)** (paper-facing
synthesis). Methods of record: [`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md); verdict:
[`ADJUDICATION.md`](ADJUDICATION.md). The map is **8 transdiagnostic dimensions** — a general factor **G
(overall burden)** ⊥ **7 specific axes** {cognition, **immunometabolic**, sleep, mania/activation,
suicidality, developmental-risk, substance} — estimated from observed cells only (no imputation), via one
global, missingness-aware Bayesian sparse-bifactor / ESEM model with mixed likelihoods and a Gaussian-copula
(rank-INT) continuous block, marginalized (Woodbury). It is fit at **full N = 9,013, cohort-weighted**, and
**converged** (R-hat 1.03, 0 divergences; 109 indicators = 88 continuous + 21 explicit). The **immunometabolic** factor is a single biology
axis carrying both cardiometabolic and inflammatory markers (BMI, HbA1c, lipids, CRP, etc.; bmi→0.95,
crp→0.37). The map is otherwise **simple-structure with 3 earned cross-loadings** — CTQ-37 → cognition
(−0.094), PSQI-latency → cognition (+0.057), PSQI-daytime → cognition (−0.070), each with 95% CI excluding 0
— derived under a **regularized ("Finnish") horseshoe** prior on every off-home loading (default-off via
global shrinkage, evidence-on via heavy-tailed local shrinkage, slab-capped) that protects the thin factors
and lets only clinically real cross-talk emerge; a continuous sparse-ESEM validation freeing all off-home
cells shrank ~83% to ≈0 (the simple structure is **earned, not imposed**). G is bifactor-orthogonal,
substance is pinned orthogonal, and the inter-factor Φ is otherwise small (specific–specific mean |Φ| ≈ 0.08;
notable coupling mania–sleep ≈ 0.24). Anhedonia **rejected**; impulsivity/negative-symptoms/sensory
**not_testable**; depression/anxiety are cross-loading **windows**. Engine
`src/face/models/bayesian/measurement_model_oop.py`; data layer `scripts/01_build_data` kept; loadings/Φ in
`reports/copula_8factor_{loadings,phi}.csv`; figures `fig2_map` (dot-atlas), `fig_factors` (lollipops + 8×8 Φ),
`edfig_full_atlas`. **M2 stratification COMPLETE** (PI sign-off 2026-06-27) —
canonical findings [`STRATA_FINDINGS.md`](STRATA_FINDINGS.md), atlas
[`STRATA_ATLAS.md`](STRATA_ATLAS.md), methods [`STRATIFICATION_MODEL.md`](STRATIFICATION_MODEL.md). The transdiagnostic space is a **continuum** (not biotypes): best-partition
silhouette **0.140** is indistinguishable from a structureless-Gaussian null **0.137 ± 0.002 (z = 1.13,
n.s.)**, HDBSCAN 0 clusters, one connected component. The **load-bearing objects are the continuous 8-dim
coordinates + a stable A=5 archetype simplex** (largest A with cross-seed Tucker ≥ 0.8; clean stability cliff
at A=6, 0.979→0.436): **A0** activation/sleep, **A1** severe clean-biology, **A2** immunometabolic (the
biology corner), **A3** trauma/suicidality, **A4** low-burden/well. The split is **not just severity** (mania
η² 0.224 + suicidality η² 0.225 ≫ G η² 0.050) and **transdiagnostic** (ARI 0.006 vs DSM-5; tighter than DSM-5,
η² 0.074 vs 0.026 at lower BIC). The soft tessellation is a coarse convention exported as a **nested K-family
(2/3/4) with no privileged K** — the operative K is deferred to M4/M5 incremental validity (answer: none).
**M3 temporal coherence is COMPLETE** (PI sign-off 2026-06-27) — the map
and strata are **temporally coherent** (V0→V1→V2, n=2,958 completers): scored onto the fixed M1/M2 model
(observed cells, uncertainty propagated, never re-discovered), the measurement holds (G1: all **4/4** backbone
axes invariant — G, cognition, immunometabolic φ 0.987, sleep), and the M2 geometry replays — biology/cognition
are durable (trait) while severity + symptoms slide (state), and archetype identity persists. G3:
**immunometabolic ICC 0.91 — the single most durable axis**; cognition 0.70 trait; severity 0.62 trait-by-rank
(population improves, suicidality slides hardest −0.84); developmental 0.39 state. G4: archetype weights
persist (Arm-B cosine median **0.81**). Findings: [`TEMPORAL_FINDINGS.md`](TEMPORAL_FINDINGS.md).
**M4 prognosis is COMPLETE** (PI sign-off 2026-06-27) — on the fixed M1/M2/M3 objects, a baseline transdiagnostic
profile **predicts 2-year functioning incrementally beyond DSM-5 + severity + baseline functioning**: the
**A=5 archetypes** add **ΔELPD +62.8** (held-out), IPW-robust (+54.4), permutation-null (−2.4),
**course-dependent / BP-led**, and **co-informative with DSM-5** (+both 62.6 > +DSM-5 29 > +map 17 — the map
complements diagnosis). The **archetype prognostic atlas** stratifies 2-year functional remission **17% → 52%**,
the **immunometabolic corner (A2) the worst-prognosis pole** and the well pole (A4) best, **within-diagnosis**
(BP 27→73%, DR 31→72%, SZ 9→25%; composition explains only 4%; cohort-adjusted best-vs-worst OR 6.3). The map
predicts **functioning, not severity** (autoregression-saturated), and the answer to the M2 K-question is
**operative K = none** (the continuous/archetype encoding dominates any hard tessellation). Findings:
[`PROGNOSIS_FINDINGS.md`](PROGNOSIS_FINDINGS.md). Honest: small individual-binary lift (remission AUC
+0.010 — the value is group-level stratification + continuous forecasting); a representation benchmark shows
the map is **sufficient for deterioration** (AUC tie vs raw) and **near-sufficient for recovery** (raw +0.04
AUC; 92–97% within-factor compression — the residual is item-level, not a missing axis). **M5 treatment is
COMPLETE** (PI sign-off 2026-06-27), scoped as **bounds-and-defends** (this baseline cohort has no randomization —
`arm` is a DSM-5 subtype — so treatment *selection* is genuinely **M5b**). Treatment
data, found late in the per-cohort thesaurus `TRAITEMENTS` tabs and harmonized to common drug-class exposures,
runs through a proper causal pipeline — **overlap → propensity → doubly-robust EIV moderation → E-value →
MDE**. **(1) The ceiling:** on observational TAU the map does **not** reliably moderate/select treatment —
lithium-BP a **well-identified, MDE-bounded null** (E 1.20–1.28, interaction MDE ≈ 0.20 → the design
could have seen an effect and didn't), antipsychotic-BP a confounded *average* effect (E 1.80) with
**suggestive-but-unconfirmed** moderation, clozapine-SZ underpowered; the map is *prognostic + descriptive, not
prescriptive*. **(2) Defends M4:** the prognostic carrier **survives treatment adjustment** in both
representations — the **A2 immunometabolic archetype corner** (attenuation 7.7% / 6.4% IPW) **and** the
**immunometabolic durable axis** (6.4% / 4.1%) — not a treatment proxy.
**(3) Describes** course: the immunometabolic corner faces the hardest 2-year course (treatment-resistance 44%,
side-effects 25% vs the well pole's 20% / 11%) — stratification clears (LR p ≤ 1e-3, within-cohort, composition
≤ 5%), discrimination clears for response/side-effects (perm p 0.010 / 0.015), resistance AUC-marginal (p
0.205); archetype ΔAUC +0.012 / +0.034 / +0.042. Canonical engine `src/face/treatment/treatment_model_oop.py`;
findings [`TREATMENT_FINDINGS.md`](TREATMENT_FINDINGS.md).
**The full vertical (M1→M2→M3→M4→M5) is consolidated** in
[`VERTICAL_FINDINGS.md`](VERTICAL_FINDINGS.md). **The program is M1–M5 complete (pending PI
sign-off); a true M5b — treatment *selection* — needs randomized/trial-arm data.**
**Honest positioning (the calibrated claim):** the program demonstrates **scientific validity** — a real,
stable, *continuum* (not biotypes) map carrying a *small but genuine, group-level* incremental prognostic
signal for functioning — but **not strong clinical utility**: the individual-level prognostic gain is small
(remission AUC +0.010) and the map does **not** moderate treatment in observational data. These are
different bars; reporting the modest/null results as such is a deliberate **correction to biotype/biomarker
overclaiming**, not a shortfall. Individual-level utility or treatment guidance would need incident events,
randomized treatment data, and external validation this baseline cohort lacks.
Updated 2026-06-27.

**Exploration / acceleration arm — variational GLLVM (M1, 2026-06-27).** A PyTorch
stochastic-variational re-estimation of the M1 measurement model on the **same** data contract,
ontology, and Gaussian-copula likelihood, but trained by SVI instead of NUTS (engine
`src/face/models/variational/`; methods [`VGLLVM_MODEL.md`](VGLLVM_MODEL.md); findings
[`VGLLVM_FINDINGS.md`](VGLLVM_FINDINGS.md); hand-off `results/face/gllvm_oop/consolidate/`). On the
8-factor map it **reproduces the NUTS M1's map and coordinates** — loadings 8/8
Tucker-congruent (whole-matrix scatter r = 0.993; G 0.974 with the low-rank/ensemble), patient
coordinates 8/8 r ≥ 0.90 — in **~4.5 min on CPU** (vs the NUTS fit's hours), and is **a faithful
generative model** (synthetic-patient marginal KS median 0.04, mean/SD fidelity r ≈ 1.0, correlation
SRMR 0.078 ≈ NUTS). Follow-ups (2026-06-27): the full **posterior-richness ladder** (`q_rank` low-rank →
`full_cov` full K×K covariance) was run against the one gap — the inter-factor correlation Φ. It
shrinks the attenuation monotonically (mean-field 21% → low-rank 20% → full-cov 18%; max cell diff
0.109 → 0.077) **but does not close it**, establishing that the residual is a **structural VI bias on
the correlation hyperparameter** (not a per-patient-posterior limitation) — the boundary for which
NUTS is the authority is now *earned*, not assumed. Also: a
**6-seed ensemble** gives loading credible bands (268/277 cells); a **dot-atlas** loading map
(`docs/figures/gllvm_oop/gllvm_dot_atlas*.png`) and a **generative round-trip**
(`run_gllvm_synthetic_check.py` + `gllvm_synthetic_check.ipynb`). It is **congruent, not certified** —
NUTS stays the authority for Φ, loading uncertainty, and paper claims; the VI arm is for fast reruns /
sensitivity sweeps / synthetic cohorts / an independent-estimator robustness check. NOT committed; pending PI review.

## M2 — stratification (COMPLETE, PI sign-off 2026-06-27)

> **Canonical record: [`STRATA_FINDINGS.md`](STRATA_FINDINGS.md) +
> [`STRATA_ATLAS.md`](STRATA_ATLAS.md).** On the 8-dim coordinates the transdiagnostic space is a
> **continuum** (confirmed by a single-Gaussian falsification null), the archetypes are **A=5 stable**, and the
> tessellation is exported as a **nested K-family (2/3/4) with no privileged K** — the load-bearing objects are
> the continuous coordinates + the A=5 simplex; the operative K is deferred to M4/M5. Engine:
> `src/face/strata/strata_model_oop.py`; hand-off
> `results/face/strata_oop/consolidate/{patient_strata.parquet, k_family_menu.csv}` (9,013 × 50).

**Methods of record: [`STRATIFICATION_MODEL.md`](STRATIFICATION_MODEL.md).** Scope: **internal discovery +
validation** of probabilistic strata on the M1 8-dim V0 coordinates — decision-relevance deferred to
M4 (no outcomes at V0). One engine (`src/face/strata/`), three parts:

- **Structure-discovery gate** (Mapper / dip / Hopkins, run on M1 draws) — *cluster vs continuum vs branched*
  is decided & reported **before** committing to "strata exist."
- **Model A — measurement-error Bayesian mixture (primary):** `x_i ~ Σ_k π_k·Normal(m_k, Σ_k + S_i)`, where
  `S_i` is the M1 per-patient posterior covariance — so coordinate **uncertainty propagates** (prior-dominated
  axes self-down-weight; the no-imputation invariant moves to the coordinate layer). `K` data-driven
  (sparse/DP). Soft responsibilities = the probabilistic decision regions.
- **Model B — archetypal analysis (co-primary):** patients as convex blends of extreme phenotypes (soft
  simplex membership; the continuum-honest view). Report **both** A and B (agree = robust; disagree =
  continuum signal).

**G treated BOTH ways** (decided): Arm A all-8 (severity×profile) ∥ Arm B 7-specifics (pure profile = the
bifactor G-residualized view, since M1's specifics are orthogonal to G). All-8 dims ⇒ **M2.0** must full-N
project the explicit axes + export per-patient covariance/draws + the validation table.
Pipeline prep→structure→mixture→archetypes→validate→atlas→score, each with a discussion
gate. Four validation gates: existence · **not-just-severity (Q2 — the headline, descendant of biology⊥G)** ·
transdiagnostic (Q3) · stable/not-an-artefact (Q4). Visuals first-class (UMAP+PCA embedding, Mapper, profile
heatmaps — viz-only, never a clustering input).

**M2.0 DONE** — all 8 dimensions full-N for 9,013. The explicit axes were projected full-N under fixed
params (no re-fit, no imputation); QC: projection **reproduces the fixed f_e at Pearson r ≈ 1.00**
(0 divergences, R-hat(z_e) 1.04 — per-patient latent mixing, point estimates exact). Cross-cohort means are
clinically coherent (mania↑BP, suicidality↓SZ, developmental↑DR). Artifacts (`results/face/strata_oop/`):
the 8-dim coordinates (mean/SD/HDI/n_obs/reliability) + draws (the uncertainty arm) + the validation table
(cohort + **7 DSM-5 subtypes** + age/sex/edu/site). Engine: `src/face/strata/strata_model_oop.py`.

**M2.1 structure-discovery gate DONE — verdict: CONTINUUM (not discrete clusters).** Battery
(Hopkins · dip · GMM-BIC · silhouette · gap · HDBSCAN · Mapper), both G-arms, uncertainty-aware over draws.
Converging evidence: best-partition silhouette **0.140** indistinguishable from a structureless-Gaussian null
**0.137 ± 0.002 (z = 1.13, n.s.)**, **HDBSCAN 0 clusters**, **one connected component**, **PC1 unimodal**,
GMM-BIC monotone (no elbow), Mapper a single connected chain. UMAP shows **one diffuse cloud with cohorts +
all 7 DSM-5 subtypes fully intermixed** (strongly transdiagnostic) and smooth continuous gradients of severity
and immunometabolic load (biology⊥G). **Implication: archetypes LEAD** (continuum-honest soft view); the
mixture is reported as a *soft tessellation*, **not** natural-kind biotypes — the honest dimensional result,
exactly why the gate ran first. Engine `src/face/strata/strata_model_oop.py` (structure module).

**M2.3 archetypes (LEAD view) DONE.** Archetypal analysis on the coordinates (both G-arms),
uncertainty-aware (M1 draws projected onto fixed archetypes). **A = 5 stable** — the largest A with cross-seed
Tucker ≥ 0.8 and a clean **stability cliff at A = 6** (0.979 → 0.436; EV 0.60). The five corners:
**A0** activation/sleep (↑sleep ↑mania), **A1** severe clean-biology (↑severity, ↓immunometabolic,
↓developmental), **A2** immunometabolic (↑immunometabolic ↑severity ↑suicidality — *the biology corner*),
**A3** trauma/suicidality (↑developmental ↑suicidality), **A4** low-burden/well (everything low). Most patients
are blends (interior of the simplex, continuum-consistent). **Transdiagnostic:** every archetype mixes all
cohorts + all 7 DSM-5 subtypes (Q3 preview). Engine `src/face/strata/strata_model_oop.py` (archetypes module);
figures `fig4b_archetypes`.

**M2.2 mixture-as-tessellation DONE.** Measurement-error mixture (`x_i ~ Σ_k π_k N(m_k, V_k + S_i)`,
S_i = M1 per-patient variance → uncertainty propagates, prior-dominated cells self-down-weight). BIC shows a
**flat basin** (no sharp optimum, continuum-consistent), so no privileged K: a **nested K-family (2/3/4)** is
exported as a coarse convention tiling the continuum, the operative K deferred to M4/M5 incremental validity
(answer: none). Transdiagnostic (each region mixes cohorts + 7 DSM subtypes). Engine
`src/face/strata/strata_model_oop.py` (mixture module).

**M2.4 validation DONE — ALL preconditions pass; descriptive head-to-head vs DSM-5 WON.**
On both views (archetypes lead, tessellation). **Q1** existence: honest CONTINUUM (no biotypes). **Q2
not-just-severity **: the split is driven by **mania (η² 0.224) + suicidality (η² 0.225) ≫ G (η² 0.050)** —
the specific/biological axes, not just severity. **Q3 transdiagnostic **: ARI **0.006 vs DSM-5**, 0.011 vs
cohort — ≈0, cuts across diagnosis. **Q4 stable + not-artefact **: seed-ARI **0.991**; membership is **not**
driven by missingness (coverage→membership classifier below majority). **Head-to-head vs DSM-5**: tighter than
DSM-5 — coordinate η² **0.074 vs 0.026 at lower BIC** (DSM-5 barely structures the coordinates). Descriptive
win only — predictive/treatment is M4/M5. Engine `src/face/strata/strata_model_oop.py` (validation module).

**M2.5 consolidation DONE — M2 COMPLETE (PI sign-off 2026-06-27).** Unified hand-off
`results/face/strata_oop/consolidate/patient_strata.parquet` (9,013 × 50: archetype weights + sd, tessellation
responsibilities, dominant labels, entropy, arm — diagnosis for validation only); paper-facing
[`STRATA_FINDINGS.md`](STRATA_FINDINGS.md) + [`STRATA_ATLAS.md`](STRATA_ATLAS.md). Figures
`fig4_continuum`, `fig4b_archetypes`. **PI sign-off on the findings + atlas locks M2; then M3 temporal
coherence (do the coordinates + phenotype memberships persist V1–V4?).**

## M3 — temporal coherence (COMPLETE, PI sign-off 2026-06-27)

**Findings: [`TEMPORAL_FINDINGS.md`](TEMPORAL_FINDINGS.md); methods: [`TEMPORAL_MODEL.md`](TEMPORAL_MODEL.md).** Does the V0 map + strata cohere and persist over
follow-up (V0→V1→V2, yearly; n=2,958 completers)? Scored onto the **FIXED** M1/M2 model — observed cells only,
uncertainty propagated, **never re-discovered** (V1/V2 scored under the fixed M1 via `copula_forward` +
frozen-V0 covariate-FWL; V0 reproduced at r≈0.99). Engine `src/face/temporal/temporal_model_oop.py`; its
`temporal_oop/attrition/` IPW feeds M4. Window V0–V2 (all 3 cohorts well-represented; DR collapses at V3). G5
(vs DSM-5) **retired/subsumed** (`arm` is time-invariant in-data so the symmetric test is unmeasurable, and on
the continuum there are no strata-labels to switch; its intent is carried by G3/G4, the incremental claim by M4).

- **G1 invariance (precondition) ** — per-visit backbone refit, Tucker φ vs V0: all **4/4** backbone axes
  **invariant** — G, cognition, **immunometabolic (φ 0.987)**, sleep. The merged biology axis is fully
  invariant. The map measures the same constructs at follow-up → licenses reading change as patient-change.
- **G2 substrate** — the patient panel over V0–V2: 8-dim coords + uncertainty + Arm-A/B memberships + per-axis
  license + trait/state. V0 reproduces M2 `patient_strata` at **99.9%**; the V0-standardization spec round-trips
  **bit-exact** (frozen scale, so genuine change is preserved, not re-centred).
- **G3 trait/state (headline).** Measurement-error random-intercept (visit fixed effects remove the population
  trend; known M1 var **plugged**). **Individual ICC:** **immunometabolic ICC 0.91 — the single most durable
  axis**; cognition 0.70 trait; severity 0.62 **trait by rank** (population improves, slide −0.46; suicidality
  slides hardest −0.84); developmental 0.39 state; substance 0.49 (orthogonal + thin); mania uninformative
  (data-limited).
- **G4 persistence + spine-vs-corner (headline) ** — archetype weights persist (Arm-B cosine median **0.81**);
  "spine moves + biology holds" **0.234 > anti-pattern 0.163**. Dominant-label churn is higher with 5 corners
  (expected — argmax flips while the weights barely move).
- **G3 ⟷ G4 synthesis** — both routes agree on the core (biology durable, symptoms move); the simple cross-route
  ρ is weak (≈ 0.07), diluted by **principled** exceptions: severity (trait-by-rank, moves-via-slide) and
  developmental (G3 σ²_w inflated by **CTQ recall noise**; G4's reliable-change rule is the robust route).
- **G6 attrition (honesty)** — dropout **mild** / cognition-leaning (severity neutral, the improved don't
  preferentially leave); verdicts robust completers-vs-all; strata-independent IPW saved (feeds M4).

**Bottom line:** the transdiagnostic map + strata are **temporally coherent** — *stratify on the durable
biology (immunometabolic/cognition), monitor the moving symptoms (severity/suicidality/sleep).*
Caveats carried forward: developmental's "state" = CTQ recall noise (trait by design); substance uninformative
(signal ≪ noise); mania data-limited; 3-visit window. Figures `fig5_persistence`, `edfig_invariance`.
**PI sign-off locks M3; then M4 prognosis — persists ≠ predicts.**

## M4 — prognosis (COMPLETE, PI sign-off 2026-06-27)

Findings [`PROGNOSIS_FINDINGS.md`](PROGNOSIS_FINDINGS.md) · methods [`PROGNOSIS_MODEL.md`](PROGNOSIS_MODEL.md) ·
clinician atlas [`PROGNOSIS_FINDINGS.md`](PROGNOSIS_FINDINGS.md).
Engine `src/face/prognosis/prognosis_model_oop.py` (frame · reference · glm · compare · endpoints ·
clinical_value · transdiagnostic · robustness); the representation benchmark in `src/face/prognosis/repbench/`.
Consumes the **fixed** M1/M2/M3 objects (panel, draws, strata, IPW) — nothing re-discovered or re-scored.

On the M3 panel, an errors-in-variables Bayesian GLM tests whether a baseline coordinate/stratum
predicts a 2-year outcome **incrementally beyond DSM-5 + severity + the baseline outcome (R3y bar)**.
**Verdict (the four gates):** **Q1** the map adds for **functioning** — the **A=5 archetypes** add **ΔELPD
+62.8** (held-out; remission AUC +0.010, CI excl 0) but **not severity** (autoregression-saturated); **Q2** the
durable **immunometabolic** axis has a credible adverse direction and survives the **error-corrected G**
severity (the durable-pair-alone EIV is ambiguous, +2.3 — the predictive carrier is the fuller archetype
representation); **Q3** **co-informative with the 7 DSM-5 subtypes** (+both 62.6 > +DSM-5 29 > +map 17 —
complements, not replaces) and **course-dependent / BP-led** (dropping BP collapses it; SZ null = foundation
saturation, not map failure); **Q4** survives IPW (+54.4) + reliability + permutation (null −2.4). The
**archetype prognostic atlas**: 2-year functional remission **17% → 52%** across the 5 archetypes — the
**immunometabolic corner (A2) the worst-prognosis pole**, the well pole (A4) best — **within-diagnosis** (BP
27→73%, DR 31→72%, SZ 9→25%; composition explains only **4%**; cohort-adjusted best-vs-worst OR 6.3;
interaction NS). The answer to the M2 K-question is **operative K = none** (the continuous/archetype encoding
dominates any hard tessellation). The map's value is **group-level stratification + continuous functional
forecasting**, not a large individual-binary boost (+0.010 AUC). Honest limits: scale trajectories not events;
internal validity; 2-year horizon.
**Representation benchmark** (raw-vs-map; [`M4_REPRESENTATION_BENCHMARK.md`](M4_REPRESENTATION_BENCHMARK.md),
engine `src/face/prognosis/repbench/`): against the raw indicators under a matched XGBoost, the 8-factor map
is **sufficient for deterioration** (AUC tie) and **near-sufficient for recovery** (raw +0.04 AUC) — the
residual is **within-factor compression** (92–97% of raw's recovery signal lives inside the 8 factors), not a
missing axis; honest uncertainty (EIV-GLM) adds a little, the map transports as well/better than raw where it
is sufficient. *Structurally faithful summary, parsimony for a sliver of resolution.* Figures `fig6_prognosis`,
`edfig_repbench`, `edfig_robustness`. **PI sign-off pending.**

## M5 — treatment (COMPLETE, PI sign-off 2026-06-27)

Findings [`TREATMENT_FINDINGS.md`](TREATMENT_FINDINGS.md) · methods [`TREATMENT_MODEL.md`](TREATMENT_MODEL.md).
Engine `src/face/treatment/treatment_model_oop.py` (endpoints · frame · medications · propensity · moderation).
Treatment data was found **late** in the per-cohort thesaurus `TRAITEMENTS` tabs (never in the harmonized set),
harmonized to common drug-class exposures (ATC[SZ] / class-string[DR] / lifetime-flag[BP]).

Scoped as **bounds-and-defends** (this baseline cohort has no randomization — `arm` is a DSM-5 subtype — so
treatment *selection* is genuinely **M5b**). A proper causal pipeline — **overlap gate → propensity (severity +
diagnosis + demographics + the 8 map coords) → doubly-robust EIV moderation (treat × durable-axis + treat ×
A=5 archetypes) + E-value → MDE** — asks whether the map *moderates* treatment response.
**(1) The ceiling:** on observational treatment-as-usual the map **does not reliably moderate / select**
treatment — **lithium-in-BP** (cleanest, 100% overlap) is a **well-identified, MDE-bounded null** (E 1.20–1.28,
interaction MDE ≈ 0.20 → the design could have seen an effect and didn't); **antipsychotic-BP** a **confounded
average effect** (E 1.80) with **suggestive-but-unconfirmed** moderation; **clozapine-SZ** underpowered. The map
is **prognostic + descriptive, not prescriptive**.
**(2) Defends M4:** the prognostic carrier **survives treatment adjustment** in both representations — the
**A2 immunometabolic archetype corner** (attenuation 7.7% / **6.4% IPW**) **and** the **immunometabolic durable
axis** (6.4% / **4.1% IPW**) — not a treatment proxy.
**(3) Describes course:** the immunometabolic corner faces the hardest 2-year course (treatment-resistance 44%,
side-effects 25% vs the well pole's 20% / 11%) — stratification clears (LR p ≤ 1e-3, within-cohort, composition
≤ 5%), discrimination clears for response/side-effects (perm p 0.010 / 0.015), resistance is AUC-marginal (p
0.205); archetype ΔAUC +0.012 / +0.034 / +0.042. The boundary is **earned, not assumed** — genuine treatment
*selection* needs randomized/trial-arm data (a future **M5b**). Figures `edfig_treatment`.

**Follow-ups resolved:** (i) the **DR-MARS** harmonization bug is **fixed** — DR's adherence
score was reverse-coded (mirror of BP/SZ); `face.data.rules.harmonize_mars` reflects DR (10−x) onto the
common scale (DR mean 3.2→6.3, matches SZ), `tests/v3/test_mars_harmonization.py`, [`reports/58_dr_mars_fix.md`](../reports/58_dr_mars_fix.md).
(ii) the **M5b feasibility check** is **done** ([`reports/59_m5b_feasibility.md`](../reports/59_m5b_feasibility.md)):
**no randomization exists in FACE** (confirmed across CSVs + thesauri — it is observational by design), so
a true selection M5b needs **external** randomized/trial-arm data; **but** BP/SZ carry per-visit medication
trajectories with dates, so a **stronger *observational* M5b** (longitudinal/g-methods, time-varying
treatment) is feasible now without new data (DR excluded — no follow-up Rx).

## What's decided

- **Model:** one **global**, missingness-aware Bayesian sparse-bifactor / ESEM — mixed likelihoods, a
  Gaussian-copula (rank-INT) continuous block marginalized (Woodbury), a regularized-horseshoe prior on the
  off-home cross-loadings, observed-cell likelihood (no imputation), **full V0 sample (N = 9,013,
  cohort-weighted)**; **only the global fit is interpreted.**
- **Confirmation:** **in-engine** — prior-free refit + PPC + WAIC (standalone FIML dropped, §5; semopy
  intractable/unreliable on the full backbone, and §3.5 makes the marginal = FIML). **Done** (see below).
- **Dimension set (V0):** `G(overall burden)` ⊥ 7 specifics — `cognition` · `immunometabolic` · `sleep` ·
  `mania/activation` · `suicidality` · `developmental-risk` · `substance`. Rejected: `anhedonia` (thin;
  absorbed by G + depression windows). Not testable: impulsivity, negative symptoms, sensory.
- **Stack:** lean — PyMC + **NumPyro/JAX**. The marginalized (Woodbury) engine **converges on the Mac M4
  (CPU)** — no GPU needed (the marginalization keeps it within budget). YAML configs; Parquet
  model-ready persistence (raw stays CSV); per-stage reports; notebooks.
- **Repo:** package **`src/face/…`**. Canonical engine
  `src/face/models/bayesian/measurement_model_oop.py`; data layer `scripts/01_build_data` (full-N V0 →
  Parquet) kept; loadings/Φ in `reports/copula_8factor_{loadings,phi}.csv`.

## What exists vs. not

- **Exists:** `src/face/data` (harmonization + skip-logic, no imputation); `configs/` ontology + the
  prior loading matrix + the **prior atlas** (`docs/PRIOR_ATLAS.md`); `scripts/01_build_data` (Parquet
  persistence) + the canonical measurement engine `src/face/models/bayesian/measurement_model_oop.py`; tests.
- **The map (FINAL):** **8 dimensions** — G (overall burden) ⊥ the 7 specifics {cognition, **immunometabolic**,
  sleep, mania/activation, suicidality, developmental-risk, substance} — fit jointly at full N = 9,013,
  cohort-weighted (R-hat 1.03, 0 divergences; 109 indicators = 88 continuous + 21 explicit). The immunometabolic factor carries both
  cardiometabolic and inflammatory markers as one biology axis. anhedonia **rejected**;
  impulsivity/negative-symptoms/sensory dropped pre-modeling; depression/anxiety = cross-loading
  windows. Biology is *least severity-entangled* (not strictly ⊥) via the correlated-G test. *(Paper-facing
  detail lives in [`M1_FINDINGS.md`](M1_FINDINGS.md); methods of record in
  [`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md).)*
- **Confirmation result (§5, DONE):** the continuous backbone is **estimator- and prior-robust**. A
  **prior-free** (flat-prior) refit at full N reproduces the soft-prior loadings/Φ **exactly** (Tucker φ =
  1.00 every factor; max |ΔΦ| = 0.00) → not a Bayesian-prior artefact; **PPC** absolute fit SRMR ≈ 0.07
  (misfit only in repeated-measure item clusters); **WAIC** decisively prefers the bifactor over
  unidimensional (Δ≈53k) and correlated-factors (Δ≈2.7k). Artifacts: `reports/05_confirmation_report.md`
  (+ `05_waic.csv`, `05_residual_correlations.csv`); engine `src/face/confirm.py` · `scripts/05_confirm.py`.
- **Invariance result (§8, DONE):** in-engine, per-cohort **simple-structure** fits (the bifactor G is
  multimodal in SZ without FAST), N≈600/cohort × 3 seeds. The map is **largely invariant**
  across BP/SZ/DR: **cognition · immunometabolic · sleep invariant** everywhere; **G
  invariant** except BP–SZ (partial, φ 0.92 — few anchors, no FAST in SZ). Artifacts:
  `reports/06_invariance_report.md` (+ `06_congruence.csv`, `06_dif_items.csv`);
  `src/face/runner.py`. Working pattern: subsample ≈2k + multi-seed + resumable cache + progress.
- **Cross-loading derivation (DONE):** a **regularized ("Finnish") horseshoe** prior on every off-home
  specific↔specific loading — default-off (global shrinkage τ pulls the whole set to ≈0), evidence-on
  (heavy-tailed local shrinkage lets a genuine cross-loading escape), magnitude-capped (slab) — protects the
  thin factors (substance, mania) while letting small, clinically real cross-talk emerge. A continuous
  sparse-ESEM validation freed all off-home cells: ~83% shrank to ≈0 (simple structure **earned**), and the
  credible cells were folded in; the map keeps **3 cross-loadings** — CTQ-37 → cognition (−0.094),
  PSQI-latency → cognition (+0.057), PSQI-daytime → cognition (−0.070), all with 95% CI excluding 0. Given
  total freedom the model reproduces known clinical cross-talk and nothing spurious. Findings record:
  [`HORSESHOE_ESEM.md`](HORSESHOE_ESEM.md), [`CROSS_LOADING_ARM.md`](CROSS_LOADING_ARM.md).
- **Correlated-G sensitivity (§3.1, DONE — biology⊥G refined):** relaxing G⊥specifics (all factors freely
  correlated) → G correlates least with **immunometabolic** (the biology axis) and most with cognition/sleep:
  biology is the **least severity-entangled** domain (not strictly ⊥, but lowest by far).
- **Robustness (§8, DONE):** Tucker congruence φ of the loadings under leave-one-cohort-out +
  diagnosis-balanced subsampling + **site cluster-bootstrap** + **1/n_cohort-weighted fit** — **min φ ≥ 0.85**
  (map not an artefact of cohort imbalance, any single cohort, or site clustering).
- **Scoring (§7, DONE):** per-patient 8-dim coordinates for all 9,013 — continuous-anchored dims via
  conditional-Gaussian from the fixed loadings + the explicit (suicidality/developmental/substance) axes via
  f_e — each with mean/SD/HDI + reliability tier.
- **Atlas + adjudication (§2.3/§6, DONE — M1 LOCK pending PI review):** prior→posterior dot-atlas (`fig2_map`) +
  `docs/ADJUDICATION.md`: G + the 7 specifics confirmed (cognition, **immunometabolic**, sleep,
  mania/activation, suicidality, developmental-risk, substance), **anhedonia rejected**,
  impulsivity/negative/sensory **not_testable**, depression/anxiety = windows. No candidate deferred.
- **Mixed-model PPC (§8, DONE):** absolute-fit check for the non-Gaussian block — the indicators reproduce their
  observed endorsement rates/means (Bayesian p ≈ 0.5); the suicidality factor's binary items all reproduce.
- **M1 complete** — the measurement layer is built, hardened (confirmation/invariance/robustness/PPC),
  scored, and adjudicated on the **8-factor map**. PI sign-off on the adjudication + atlas locks it.
- **Later milestones:** M2 strata · M3 temporal coherence · M4 prognosis · M5 treatment — all COMPLETE on the
  8-factor map (PI sign-off 2026-06-27); see the sections above.

## M1 measurement — per-stage development record (retired)

The staged-fit convergence-checkpoint log has been retired from this
status file to keep one current narrative. The canonical M1 record is
[`M1_FINDINGS.md`](M1_FINDINGS.md) (paper-facing), [`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md)
(methods of record), [`ADJUDICATION.md`](ADJUDICATION.md) (per-candidate verdicts) and
[`RESULTS.md`](RESULTS.md) (findings log).
Biology–G independence is reported in [`RESULTS.md`](RESULTS.md) / [`M1_FINDINGS.md`](M1_FINDINGS.md);
certification tiers in [`CERTIFICATION_TIERS.md`](CERTIFICATION_TIERS.md).

## What to read

[`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md) (methods + math) · [`RESULTS.md`](RESULTS.md) (findings log)
· [`PRIOR_ATLAS.md`](PRIOR_ATLAS.md) (prior
map) · [`../README.md`](../README.md) (overview) · [`../CLAUDE.md`](../CLAUDE.md) (guide) ·
[`DATA.md`](DATA.md) (data contract).
