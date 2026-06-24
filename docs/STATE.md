# STATE — where the project is right now

> **Read this first.** Updated 2026-06-11.

## TL;DR

**Milestone 1 (M1) — the transdiagnostic dimensional map — is COMPLETE** (pending PI sign-off), on the FACE
**V0** baseline (N = 9,013). **Findings + discussion: [`M1_FINDINGS.md`](M1_FINDINGS.md)** (paper-facing
synthesis). Methods of record: [`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md); verdict:
[`ADJUDICATION.md`](ADJUDICATION.md). The map is **9 transdiagnostic dimensions** — a general factor **G
(functional burden)** + **cognition, metabolic, inflammatory, sleep, developmental-risk, suicidality,
mania, substance** — estimated from observed cells only (no imputation), via one global Bayesian sparse
bifactor/ESEM (marginalized continuous core + explicit non-Gaussian block). It is **hardened end-to-end**:
not a prior/estimator artefact (flat-prior φ=1.00, WAIC, PPC §5); largely invariant across BP/SZ/DR (§8);
with its continuous backbone **certified at full N** and the joint 9-dim **documented at the largest N that mixes** (cross-seed Tucker φ 0.993, §4); biology is the least severity-entangled
domain (correlated-G §3.1); resample-robust (min φ ≥ 0.85 under LOCO + site-bootstrap + weighting, §8);
with per-patient coordinates + uncertainty + reliability flags (§7). Anhedonia **rejected**;
impulsivity/negative-symptoms/sensory **not_testable**; depression/anxiety are cross-loading **windows**.
Engine in `src/face/{models/bayesian,confirm,runner,scoring}.py`; data layer `scripts/01_build_data` kept,
native M1 modeling scripts (`04–09,10*,12,13,s5_*`) **retired 2026-06-24** (canonical = copula OOP fit); results
in `reports/01,04–11`. **M2 stratification COMPLETE** (pending PI sign-off), **reworked on the certified Gaussian-copula map** —
canonical findings [`STRATA_OOP_FINDINGS.md`](STRATA_OOP_FINDINGS.md), atlas
[`STRATA_OOP_ATLAS.md`](STRATA_OOP_ATLAS.md), methods [`STRATIFICATION_MODEL.md`](STRATIFICATION_MODEL.md). The transdiagnostic space is a **continuum** (not biotypes; confirmed by a single-Gaussian
falsification null): the **load-bearing objects are the continuous 9-dim coordinates + a stable A=4 archetype
simplex** (biology⊥symptoms⊥severity), and the soft tessellation is a coarse convention exported as a **nested
K-family (2/3/4) with no privileged K** — the operative K is deferred to M4/M5 incremental validity.
Transdiagnostic (ARI≈0 vs DSM-5) and tighter than DSM-5 (descriptive). *M3/M4/M5 as reported below were run on
the prior native-map M2 (8 archetypes / K=4); their rerun on this copula object is pending.* **M3 temporal coherence is COMPLETE** (pending PI sign-off) — the map
and strata are **temporally coherent** (V0→V1→V2): the measurement holds (G1 invariance), and the M2 geometry
replays over time — biology/cognition are durable (trait) while severity + symptoms slide (state), and
archetype identity persists. Findings: [`TEMPORAL_OOP_FINDINGS.md`](TEMPORAL_OOP_FINDINGS.md). **M3 has now been
*reworked on the copula M1/M2 objects*** (parallel OOP engine `src/face/temporal/temporal_model_oop.py`;
canonical [`TEMPORAL_OOP_FINDINGS.md`](TEMPORAL_OOP_FINDINGS.md)): V1/V2 scored under the fixed copula M1
(`copula_forward` + frozen-V0 covariate-FWL; V0 reproduced at r≈0.99) → the result **replays** — G1 all 5
backbone axes invariant (inflammatory now invariant, vs partial native), G3 biology trait (metabolic ICC
**0.91**, cognition 0.70) / symptoms state (developmental 0.39) / severity trait-by-rank with population
improvement, G4 archetype weights persist (cosine 0.90). **M4 prognosis is
COMPLETE** (pending PI sign-off) — on the fixed M1/M2/M3 objects, a baseline transdiagnostic profile
**predicts 2-year functional trajectory incrementally beyond diagnosis + severity** (durable
metabolic/inflammatory ⊥G; the 8 archetypes stratify functional remission **14%→60%**), robust to
attrition/reliability/permutation, **co-informative with DSM-5** (complements, not replaces) and
**course-dependent** (episodic BP/DR, not baseline-saturated SZ); severity itself is
autoregression-determined. Findings: [`PROGNOSIS_OOP_FINDINGS.md`](PROGNOSIS_OOP_FINDINGS.md); clinician atlas:
[`PROGNOSIS_OOP_FINDINGS.md`](PROGNOSIS_OOP_FINDINGS.md). **M4 has now been *reworked on the copula M2 object*** (parallel OOP
engine `src/face/prognosis/prognosis_model_oop.py`; canonical [`PROGNOSIS_OOP_FINDINGS.md`](PROGNOSIS_OOP_FINDINGS.md)):
the result replays — the map predicts 2-yr **functioning** (archetypes ΔELPD **+59** on egf, co-informative with
DSM-5), functional remission **27%→60%** across the A=4 archetypes (biology corner worst), and — the answer to
the M2 K-question — **operative K = none** (the continuous/archetype encoding dominates any hard tessellation;
all K=2/3/4 are predictive of functioning but add less). Honest copula shift: the *durable-trio-alone* EIV is
no longer robust; the predictive object is the fuller archetype representation. **M5 treatment is COMPLETE** (pending PI sign-off) — treatment
data, found late in the per-cohort thesaurus `TRAITEMENTS` tabs and harmonized to common drug-class
exposures, runs through a proper causal pipeline (overlap → propensity → doubly-robust EIV moderation +
E-value); on observational treatment-as-usual the map **does not reliably moderate treatment response**
(lithium-in-BP a *well-identified null*; a *suggestive-but-unconfirmed* metabolic/inflammatory ×
antipsychotic-functioning hypothesis; clozapine *channeled*/non-estimable), the boundary is **earned, not
assumed**, and the metabolic functional forecast **survives treatment adjustment** (strengthens M4).
Findings: [`TREATMENT_OOP_FINDINGS.md`](TREATMENT_OOP_FINDINGS.md). **M5 has now been *reworked on the copula objects***
(parallel OOP engine `src/face/treatment/treatment_model_oop.py`; canonical
[`TREATMENT_OOP_FINDINGS.md`](TREATMENT_OOP_FINDINGS.md)): the earned boundary **replays** — lithium-BP a
well-identified null (E 1.06), antipsychotic-BP suggestive-unconfirmed (ATE −0.23, **E-value 1.77 ≈ native
1.79**), clozapine non-decisive; the **archetype carrier survives treatment adjustment** (4.7% attenuation,
strengthens M4) and the **archetypes predict response heterogeneity** (resistance/response ΔELPD +20/+16).
**The full Gaussian-copula vertical (M1→M2→M3→M4→M5) is now reworked** (consolidated synthesis:
[`COPULA_VERTICAL_FINDINGS.md`](COPULA_VERTICAL_FINDINGS.md)). **The program is M1–M5 complete (pending PI
sign-off); a true M5b — treatment *selection* — needs randomized/trial-arm data.**
**Honest positioning (the calibrated claim):** the program demonstrates **scientific validity** — a real,
stable, *continuum* (not biotypes) map carrying a *small but genuine, group-level* incremental prognostic
signal for functioning — but **not strong clinical utility**: the individual-level prognostic gain is small
(remission ΔAUC +0.017) and the map does **not** moderate treatment in observational data. These are
different bars; reporting the modest/null results as such is a deliberate **correction to biotype/biomarker
overclaiming**, not a shortfall. Individual-level utility or treatment guidance would need incident events,
randomized treatment data, and external validation this baseline cohort lacks.
Updated 2026-06-11.

## M2 — stratification (COMPLETE 2026-06-09, pending PI sign-off)

> **Reworked on the Gaussian-copula map (2026-06-22). Canonical record:
> [`STRATA_OOP_FINDINGS.md`](STRATA_OOP_FINDINGS.md) + [`STRATA_OOP_ATLAS.md`](STRATA_OOP_ATLAS.md).** On the
> copula coordinates the continuum verdict holds (now with a single-Gaussian falsification null), the
> archetypes are **A=4 stable** (native A=8 does not reproduce), and the tessellation is exported as a
> **nested K-family (2/3/4) with no privileged K** — the load-bearing objects are the continuous coordinates +
> the A=4 simplex; the operative K is deferred to M4/M5. Engine: `src/face/strata/strata_model_oop.py`;
> hand-off `results/face/strata_oop/consolidate/{patient_strata.parquet, k_family_menu.csv}`. **The native-map
> detail below is retained as provenance** (it is what the reported M3/M4/M5 consumed).

**Methods of record: [`STRATIFICATION_MODEL.md`](STRATIFICATION_MODEL.md).** Scope: **internal discovery +
validation** of probabilistic strata on the M1 9-dim V0 coordinates — decision-relevance deferred to
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

**G treated BOTH ways** (decided): Arm A all-9 (severity×profile) ∥ Arm B 8-specifics (pure profile = the
bifactor G-residualized view, since M1's specifics are orthogonal to G). All-9 dims ⇒ **M2.0** must full-N
project suicidality/developmental/substance + export per-patient covariance/draws + the validation table.
Pipeline `scripts/20–26` (prep→structure→mixture→archetypes→validate→atlas→score), each with a discussion
gate. Four validation gates: existence · **not-just-severity (Q2 — the headline, descendant of biology⊥G)** ·
transdiagnostic (Q3) · stable/not-an-artefact (Q4). Visuals first-class (UMAP+PCA embedding, Mapper, profile
heatmaps — viz-only, never a clustering input).

**M2.0 DONE (2026-06-09)** — all 9 dimensions now full-N for 9,013 (M1 had left suicidality/developmental/
substance on the ~1,884 fit subsample). The 3 explicit axes were projected full-N under fixed
params (no re-fit, no imputation); QC: projection **reproduces the fixed f_e at Pearson r ≈ 1.00**
(0 divergences, R-hat(z_e) 1.04 — per-patient latent mixing, point estimates exact). Cross-cohort means are
clinically coherent (mania↑BP, suicidality↓SZ, developmental↑DR). Artifacts (`results/face/m2/`,
gitignored): `coordinates_full.parquet` (the M2 input — 9-dim mean/SD/HDI/n_obs/reliability),
`coordinates_draws.npz` ([200,9013,9] — the uncertainty arm), `validation_table.parquet` (cohort + **7
DSM-5 subtypes** + age/sex/edu/site). Engine: `src/face/strata/scoring.py`; `scripts/20_prep_coordinates.py`
→ `reports/20_prep_coordinates.md` + `docs/figures/20_coverage.png`.

**M2.1 structure-discovery gate DONE (2026-06-09) — verdict: CONTINUUM (not discrete clusters).** Battery
(Hopkins · dip · GMM-BIC · silhouette · gap · HDBSCAN · Mapper), both G-arms, uncertainty-aware over draws.
Converging evidence: **gap-stat K=1**, **HDBSCAN 0 clusters (100% noise)**, **PC1 unimodal** (dip p≈0.99),
silhouette peak ≈0.18 (weak), GMM-BIC drops to K≈3 then a flat plateau (no elbow; monotone), Mapper a single
connected chain. UMAP shows **one diffuse cloud with cohorts + all 7 DSM-5 subtypes fully intermixed**
(strongly transdiagnostic) and smooth continuous gradients of severity and inflammatory load (biology⊥G).
(Hopkins 0.85 is the lone high signal — expected upward bias in structured high-dim data; outweighed.)
**Implication (§3.1): archetypes LEAD** (continuum-honest soft view); the mixture is reported as a *soft
tessellation* (~K3–4 captures the anisotropy), **not** natural-kind biotypes — the honest dimensional
result, exactly why the gate ran first. Engine `src/face/strata/structure.py`; `scripts/21_structure.py` →
`reports/21_structure.md` + `docs/figures/21_{selection,embedding,mapper}.png`.

**M2.3 archetypes (LEAD view) DONE (2026-06-09).** Archetypal analysis on the coordinates (both G-arms),
uncertainty-aware (M1 draws projected onto fixed archetypes). **Scree smooth, no elbow** (ev 0.24→0.79 over
A=2→8) ⇒ reconfirms continuum: no natural A, it's a parsimony choice (knee ran to the A=8 cap). Archetypes
**highly stable** (min Tucker congruence 0.999). At A=8 they map cleanly to **one extreme per axis + a
low-burden corner**: A0 low-burden (37%), A2 ↑cognition+severity (16%), A3 ↑sleep (16%), A4 ↑↑metabolic
(13%), A6 ↑↑developmental (8.5%), A7 ↑↑mania (5.5%), and two rare tail-extremes A1 ↑↑suicidality (1.5%) &
A5 ↑↑inflammatory+substance (1.9%). **Distinct metabolic AND inflammatory corners** = biology⊥G as
phenotypes. **75% of patients are blends** (max-weight<0.5; entropy 0.67) — interior of the simplex,
continuum-consistent. **Transdiagnostic:** every archetype mixes all cohorts + all 7 DSM-5 subtypes (Q3
preview), with gradients (DR→cognition/severity+sleep; mania corner BP-heavy). Engine
`src/face/strata/archetypes.py`; `scripts/23_archetypes.py` → `results/face/m2/{archetypes.parquet,
archetype_profiles.csv}` + `reports/23_archetypes.md` + `docs/figures/23_{scree,profiles,membership}.png`.
**A = 8 CONFIRMED (PI, gate 2026-06-09)** — the only A resolving both biology corners (metabolic +
inflammatory). 23b corner-survival: metabolic/developmental/suicidality/sleep appear at A≥5, +cognition A≥6,
+mania A≥7, **+inflammatory only at A=8**; **severity & substance never form a corner** (severity = the
continuum's spine; substance absorbed/noisy). `scripts/23b_archetype_compare.py` →
`reports/23b_archetype_compare.md` + `docs/figures/23b_compare.png`.

**M2.2 mixture-as-tessellation DONE (2026-06-09).** Measurement-error mixture via **Extreme Deconvolution**
(`x_i ~ Σ_k π_k N(m_k, V_k + S_i)`, S_i = M1 per-patient variance → uncertainty propagates, prior-dominated/
DR-absent cells self-down-weight). BIC **flat basin** (K=4 199,325; K=5 199,307; Δ18 — no sharp optimum,
continuum-consistent); reported at **K=4** (M2.1 uncertainty-mode-4). 4 coarse deconvolved regions tiling the
continuum: T0 low-burden (31%), T1 ↑mania+developmental+sleep (12%, BP-heavy), T2 ↑severity+metabolic (32%,
DR/SZ-heavy), T3 ↓metabolic+↓cognition (25%); 92% confident (vs 25% for the finer 8 archetypes — coarse
regions assign sharply, archetype corners blend). Transdiagnostic (each mixes cohorts + 7 DSM subtypes).
Engine `src/face/strata/mixture.py` (XD EM); `scripts/22_mixture.py` → `results/face/m2/{tessellation.parquet,
tessellation_profiles.csv}` + `reports/22_tessellation.md` + `docs/figures/22_*`.

**M2.4 validation DONE (2026-06-09) — ALL preconditions pass; descriptive head-to-head vs DSM-5 WON.**
On both views (archetypes lead, tessellation). **Q1** existence: honest CONTINUUM (no biotypes). **Q2
not-just-severity **: per-axis η² of the tessellation is multi-axis — mania 0.45, developmental 0.35,
severity 0.31, metabolic 0.21, sleep 0.19, cognition 0.17 (η²(G) 0.31 vs mean η²(specifics) 0.20, max
specific 0.45 > G) — driven by the specific/biological axes, not just severity. **Q3 transdiagnostic **:
ARI(partition, cohort)=0.007 / (partition, DSM-5)=0.020 (tessellation), 0.06/0.05 (archetypes) — ≈0, cuts
across diagnosis (Cramér's V 0.18–0.28, weak). **Q4 stable + not-artefact **: tessellation seed ARI 0.987
(archetype congruence 0.999); **coverage→membership classifier acc 0.248 < majority 0.323 (lift −0.08)** —
membership NOT driven by missingness. **Head-to-head vs DSM-5 (§1.7)**: XD BIC free K=4 **199,325** vs DSM-5
7-group **206,016** → free wins with fewer components (tighter description); mean coordinate η² free 0.209
vs DSM-5 **0.048** (DSM-5 barely structures the coordinates). Descriptive win only — predictive/treatment is
M4/M5. Engine `src/face/strata/validation.py` + `mixture.xd_fixed_labels`; `scripts/24_validate.py` →
`reports/24_validation.md` + `docs/figures/24_validation.png`.

**M2.5 consolidation DONE (2026-06-09) — M2 COMPLETE (pending PI sign-off).** Unified hand-off
`results/face/patient_strata.parquet` (9,013 × 29: archetype weights + sd, tessellation responsibilities,
dominant labels, entropy, arm — diagnosis for validation only); paper-facing
[`STRATA_OOP_FINDINGS.md`](STRATA_OOP_FINDINGS.md) + [`STRATA_OOP_ATLAS.md`](STRATA_OOP_ATLAS.md); `scripts/26_score.py`.
Pipeline `scripts/20–26` + `src/face/strata/{scoring,structure,mixture,archetypes,validation}.py`; 90 tests
green. **PI sign-off on the findings + atlas locks M2; then M3 temporal coherence (do the coordinates +
phenotype memberships persist V1–V4?).**

## M3 — temporal coherence (COMPLETE 2026-06-10, pending PI sign-off)

**Findings: [`TEMPORAL_OOP_FINDINGS.md`](TEMPORAL_OOP_FINDINGS.md); methods: [`TEMPORAL_MODEL.md`](TEMPORAL_MODEL.md).** Does the V0 map + strata cohere and persist over
follow-up (V0→V1→V2, yearly)? Scored onto the **FIXED** M1/M2 model — observed cells only, uncertainty
propagated, **never re-discovered**. Engine `src/face/temporal/`; native pipeline `scripts/30–37` **retired
2026-06-24** (copula OOP M3 canonical; its `temporal_oop/attrition/` IPW now feeds M4); **36 tests**.
Window V0–V2 (all 3 cohorts well-represented; DR collapses at V3). G5 (vs DSM-5) **retired/subsumed** (`arm` is
time-invariant in-data so the symmetric test is unmeasurable, and on the continuum there are no strata-labels to
switch; its intent is carried by G3/G4, the incremental claim by M4). Two minimal, default-off `prepare()` adds
(`emit_moments`, `visit=`) proven non-disruptive (90 v3 tests stay green).

- **G1 invariance (precondition) ** — per-visit backbone refit, Tucker φ vs V0: severity/cognition/metabolic/
  sleep/developmental **invariant** (φ 0.96–1.00), **inflammatory partial** (0.90 — acute-phase WBC shift).
  The map measures the same constructs at follow-up → licenses reading change as patient-change. `scripts/33`.
- **G2 substrate** — `results/face/patient_panel.parquet` (16,241 rows over V0–V2): 9-dim coords + uncertainty
  + Arm-A/B memberships + per-axis license + trait/state. V0 reproduces M2 `patient_strata` at **99.9%**; the
  V0-standardization spec round-trips **bit-exact** (frozen scale, so genuine change is preserved, not
  re-centred). `scripts/31–34`.
- **G3 trait/state (headline) — two lenses.** Measurement-error random-intercept (visit fixed effects remove
  the population trend; known M1 var **plugged**). **Population slide:** symptoms slide (suicidality −0.89,
  severity −0.34), biology static (metabolic +0.10, inflammatory +0.05). **Individual ICC:** metabolic 0.93 /
  cognition 0.78 trait; sleep/suicidality/developmental mixed-state; severity 0.66 (trait-by-rank). `scripts/35`.
- **G4 persistence + spine-vs-corner (headline) ** — spine (severity) moves 34.5% > biology corner 20.2%;
  "spine moves + biology holds" **25.8% vs anti-pattern 11.5% (2.2×)**. Arm-B archetype identity persists 52%
  (κ 0.27 vs 12.5% chance, cosine 0.81). `scripts/36`.
- **G3 ⟷ G4 synthesis** — both routes agree on the core (biology durable, symptoms move); the simple ρ is
  diluted by 2 **principled** exceptions: severity (trait-by-rank, moves-via-slide) and developmental (G3 σ²_w
  inflated by **CTQ recall noise**; G4's reliable-change rule is the robust route, says it HOLDS).
- **G6 attrition (honesty)** — dropout **mild** / cognition-leaning (severity neutral, the improved don't
  preferentially leave); verdicts robust completers-vs-all (max |ΔICC| 0.14); IPW saved. `scripts/31`.

**Bottom line:** the transdiagnostic map + strata are **temporally coherent** — *stratify on the durable
biology (cognition/metabolic/inflammatory), monitor the moving symptoms (severity/suicidality/sleep).*
Caveats carried forward: developmental's "state" = CTQ recall noise (trait by design); inflammatory partial;
substance uninformative (signal ≪ noise); mania/suicidality/substance not G1-tested (explicit block); 3-visit
window. **PI sign-off locks M3; then M4 prognosis — persists ≠ predicts.**

## M4 — prognosis (COMPLETE 2026-06-11, pending PI sign-off)

Findings [`PROGNOSIS_OOP_FINDINGS.md`](PROGNOSIS_OOP_FINDINGS.md) · methods [`PROGNOSIS_MODEL.md`](PROGNOSIS_MODEL.md) ·
clinician atlas [`PROGNOSIS_OOP_FINDINGS.md`](PROGNOSIS_OOP_FINDINGS.md).
Engine `src/face/prognosis/` (frame · reference · glm · compare · endpoints · clinical_value ·
transdiagnostic · robustness); native pipeline `scripts/40–48` **retired 2026-06-24** (shared kernels +
`tests/m4/` retained; canonical M4 = copula OOP + the representation benchmark). Consumes the **fixed**
M1/M2/M3 objects (panel, draws, strata, IPW) — nothing re-discovered or re-scored.

On the M3 panel, an errors-in-variables Bayesian GLM tests whether a baseline coordinate/stratum
predicts a 2-year outcome **incrementally beyond DSM-5 + severity + the baseline outcome (R3y bar)**.
**Verdict (the four gates):** **Q1** the map adds for **functioning** (EGF: archetypes ΔELPD +46;
remission AUC +0.017, CI excl 0) but **not severity** (CGI-S autoregression-saturated); **Q2** the
durable metabolic (β −0.062) + inflammatory effects survive the **error-corrected G** severity; **Q3**
**co-informative with the 7 DSM-5 subtypes** (B−A +47, B−C +40 — complements, not replaces) and
**course-dependent** (BP/DR yes, SZ null = foundation saturation, not map failure); **Q4** survives
IPW + reliability + permutation (p=0.001), weakens dropping BP. The **archetype prognostic atlas**:
2-year functional remission **14%→60%** across the 8 archetypes, transdiagnostic. The map's value is
**group-level stratification + continuous functional forecasting**, not a large individual-binary boost
(+0.017 AUC). Honest limits: scale trajectories not events; internal validity; 2-year horizon.
Hand-off `results/face/m4/{prognosis_summary.csv, prognosis_patient_risk.parquet}`.
**Representation benchmark** (raw-vs-map; [`M4_REPRESENTATION_BENCHMARK.md`](M4_REPRESENTATION_BENCHMARK.md),
engine `src/face/prognosis/repbench/`): against the 143 raw indicators under a matched XGBoost, the 9-dim copula
map is **sufficient for deterioration** (AUC tie) and **near-sufficient for recovery** (raw +0.04 AUC) — the
residual is **within-factor compression** (97% of raw's recovery signal lives inside the 9 factors), not a
missing axis; honest uncertainty (EIV-GLM) adds a little, no small-N efficiency edge, the map transports as
well/better than raw where it is sufficient. *Structurally faithful summary, parsimony for a sliver of
resolution.* **PI sign-off pending.**

## M5 — treatment (COMPLETE 2026-06-11, pending PI sign-off)

Findings [`TREATMENT_OOP_FINDINGS.md`](TREATMENT_OOP_FINDINGS.md) · methods [`TREATMENT_MODEL.md`](TREATMENT_MODEL.md) ·
dev record `reports/50–57_*.md`. Engine `src/face/treatment/` (endpoints · frame · medications ·
propensity · moderation); native pipeline `scripts/50–57` **retired 2026-06-24** (shared kernels + `tests/m5/` retained; canonical M5 = the OOP treatment engine). Treatment data was found **late** in
the per-cohort thesaurus `TRAITEMENTS` tabs (never in the harmonized set), harmonized to common drug-class
exposures (ATC[SZ] / class-string[DR] / lifetime-flag[BP]) — the earlier "data-blocked → tolerability
coda" was superseded.

A proper causal pipeline — **overlap gate → propensity (severity + diagnosis + demographics + the 9 map
coords) → doubly-robust EIV moderation (treat × durable-axis) + E-value** — asks whether the map
*moderates* treatment response. **Verdict:** on observational treatment-as-usual the map **does not
reliably moderate** response. **Lithium-in-BP** (cleanest: 100% overlap, SMD 0.30→0.01) is a
**well-identified null**; **antipsychotic-BP** shows a **suggestive but unconfirmed** metabolic (−0.15\*) /
inflammatory (−0.26\*) × functioning interaction (held-out ΔELPD +4.6±4.2 not confirmed; ATE E-value
1.79); **clozapine-SZ** is **channeled** (IPTW SMD 0.44→0.61, non-estimable). ATEs are confounding-fragile
(E 1.1–1.8). **M5 strengthens M4:** the metabolic→functioning forecast **survives** adjustment for the
drug classes patients were on (β −0.051→−0.048, 4.4% attenuation, HDI still excludes 0). The boundary is
**earned, not assumed** — genuine treatment *selection* needs randomized/trial-arm data (a future **M5b**).
Hand-off `results/face/m5/{treatment_exposures, propensity_*, moderation, confounder}.{parquet,csv}`.

**Follow-ups resolved (2026-06-11):** (i) the **DR-MARS** harmonization bug is **fixed** — DR's adherence
score was reverse-coded (mirror of BP/SZ); `face.data.rules.harmonize_mars` reflects DR (10−x) onto the
common scale (DR mean 3.2→6.3, matches SZ), `tests/v3/test_mars_harmonization.py`, [`reports/58_dr_mars_fix.md`](../reports/58_dr_mars_fix.md).
(ii) the **M5b feasibility check** is **done** ([`reports/59_m5b_feasibility.md`](../reports/59_m5b_feasibility.md)):
**no randomization exists in FACE** (confirmed across CSVs + thesauri — it is observational by design), so
a true selection M5b needs **external** randomized/trial-arm data; **but** BP/SZ carry per-visit medication
trajectories with dates, so a **stronger *observational* M5b** (longitudinal/g-methods, time-varying
treatment) is feasible now without new data (DR excluded — no follow-up Rx).

## What's decided

- **Model:** one **global** Bayesian sparse bifactor / ESEM — mixed likelihoods, soft priors,
  observed-cell likelihood (no imputation), **full V0 sample**. Estimated via a **staged continuation**
  (S1→S5); **only the global fit (S5) is interpreted.**
- **Confirmation:** **in-engine** — prior-free refit + PPC + WAIC (standalone FIML dropped, §5; semopy
  intractable/unreliable on the full backbone, and §3.5 makes the marginal = FIML). **Done** (see below).
- **Dimension set (V0):** `G(severity)` · `cognition` · `metabolic` · `inflammatory` · `sleep` ·
  `suicidality` · `developmental-risk` · `mania` · `substance` (mania + substance added once their indicators were ingested). Rejected: `anhedonia` (thin; absorbed by G + depression windows). Not testable: impulsivity,
  negative symptoms, sensory.
- **Stack:** lean — PyMC + **NumPyro/JAX**. The marginalized (Woodbury) engine **certifies on the Mac M4
  (CPU)** — no GPU needed (the marginalization keeps it within budget). YAML configs; Parquet
  model-ready persistence (raw stays CSV); per-stage reports; notebook later.
- **Repo:** package **`src/face/…`** (renamed from `src/v3`, tests green). Pipeline built so far:
  `scripts/01_build_data` (full-N V0 → Parquet) · `scripts/04_fit --stage {1,2}` (one canonical engine,
  `src/face/models/bayesian/continuous_core`: marginalized Woodbury default, explicit-latent + `--gpu`
  optional). S2 stage flags (`correlated`/`windows`/`specific_cross`) live in `scripts/04_fit`.

## What exists vs. not

- **Exists:** `src/face/data` (harmonization + skip-logic, no imputation); `configs/` ontology +
  `prior_loading_matrix_v3.csv` (143 indicators × 10 factors) + the **prior atlas**
  (`docs/PRIOR_ATLAS.md`); `scripts/01_build_data` (Parquet persistence) + `scripts/04_fit` + the
  single marginalized/explicit engine (`continuous_core`; the parallel config-first engine + its
  `bayesian_model.yaml` were retired — one canonical engine now); tests (`tests/v3/`, **90 passing**).
- **The map (FINAL):** **9 dimensions** — G + cognition/metabolic/inflammatory/sleep/developmental-risk/
  suicidality **+ mania + substance** — fit jointly (see the "DONE" bullets below). anhedonia
  **rejected**; impulsivity/negative-symptoms/sensory dropped pre-modeling; depression/anxiety = cross-loading
  windows. Biology is *least severity-entangled* (not strictly ⊥) via the correlated-G test. *(Per-stage
  development detail lives in [`RESULTS.md`](RESULTS.md) and [`M1_FINDINGS.md`](M1_FINDINGS.md); the staged
  S1–S5 log was retired from this status file — see the note below.)*
- **Confirmation result (§5, DONE):** the continuous backbone is **estimator- and prior-robust**. A
  **prior-free** (flat-prior) refit at full N reproduces the soft-prior loadings/Φ **exactly** (Tucker φ =
  1.00 every factor; max |ΔΦ| = 0.00) → not a Bayesian-prior artefact; **PPC** absolute fit SRMR ≈ 0.07
  (misfit only in repeated-measure item clusters); **WAIC** decisively prefers the bifactor over
  unidimensional (Δ≈53k) and correlated-factors (Δ≈2.7k). Artifacts: `reports/05_confirmation_report.md`
  (+ `05_waic.csv`, `05_residual_correlations.csv`); engine `src/face/confirm.py` · `scripts/05_confirm.py`.
- **Invariance result (§8, DONE):** in-engine, per-cohort **simple-structure** fits (the bifactor G is
  multimodal in SZ without FAST), N≈600/cohort × 3 seeds, **9/9 converged**. The map is **largely invariant**
  across BP/SZ/DR (12/15 factor×pair φ ≥ 0.95): **cognition · metabolic · sleep invariant** everywhere; **G
  invariant** except BP–SZ (partial, φ 0.92 — few anchors, no FAST in SZ); **inflammatory non-invariant in
  DR** (φ 0.71/0.75) — **neutrophils load ≈0 in DR** (0.07 vs 0.88), eosinophils high (0.59 vs 0.23): DR's
  inflammatory axis is compositionally different → a **documented non-invariance caveat** for DR
  inflammatory scores. Artifacts: `reports/06_invariance_report.md` (+ `06_congruence.csv`, `06_dif_items.csv`);
  `scripts/06_invariance.py` · `src/face/runner.py`. Working pattern: subsample ≈2k + multi-seed + resumable
  cache + progress (§3.6).
- **S5 documentation (§3.6/§4.5, DONE — largest N that mixes):** the **reported 9-dim joint map**
  (S5; mania + substance integrated, below), multi-seed at
  N≈2,000 cohort-balanced (tune 2000 · draws 1500 · ta 0.9, 2 seeds). **§4.4 rung-3 reparam:** diagnosed
  the slow locus as the **CTQ→G bifactor loadings** (dev is explicit ⇒ 2-explicit-factor ridge, ESS 30);
  the `bifactor_g_sd` knob tightens dev/suic→G toward 0 (they're ≈⊥G) **leaving the biology→G estimand
  free** — cross-loadings ESS 30→85. Result: R-hat **1.03** · struct ESS **114–158** · **0 div** · BFMI
  **0.40** (healthy — no funnel; the limit is ESS-autocorrelation on the **suic~dev Φ + explicit-latent
  coupling**, not geometry). **Cross-seed resample-stability: Tucker φ 0.993**, max |ΔΦ| 0.05 — the reported
  loadings/Φ are stable; suic~dev Φ *precision* is the documented limit (point estimates solid). Artifacts:
  `reports/07_s5_certification_report.md`; `scripts/s5_certify.py` (per-seed resumable cache). Run under
  `caffeinate` + detached (`nohup`/`disown`) — the fix for macOS-sleep/harness interruptions on long fits.
- **Correlated-G sensitivity (§3.1, DONE — biology⊥G refined):** relaxing G⊥specifics (all factors freely
  correlated, simple-structure marginalized model, clean R-hat 1.01 · ESS 421 · **0 div**) → G correlates
  **+0.07 inflammatory · +0.12 metabolic** vs **+0.39 cognition · +0.42 sleep**: biology is the **least
  severity-entangled** domain (not strictly ⊥, but lowest by far). Engine: `g_correlated` Φ is now a
  **unit-row Cholesky** (`pm.LKJCorr(n≥5)` breaks jitter-init; `LKJCholeskyCov` sd funnels → divergences).
  `scripts/s5_corrg.py` → `reports/07_corrG_report.md`.
- **Robustness (§8, DONE):** Tucker congruence φ of the loadings vs the certified S2 reference under
  leave-one-cohort-out + diagnosis-balanced subsampling + **site cluster-bootstrap** + **1/n_cohort-weighted
  fit** (§3.6) — **min φ ≥ 0.85** (map not an artefact of cohort imbalance, any single cohort, or site
  clustering). `scripts/08_robustness.py` → `reports/08_robustness_report.md`.
- **9-dim joint integration (DONE):** mania + substance were **added** (their indicators ingested) as **real dimensions**, so
  the reported map was re-fit at **9 dimensions** — 5 marginalized (cognition/metabolic/inflammatory/
  sleep/**mania**) + 4 explicit (G/suicidality/developmental/**substance**, substance's binary SUD under the
  proper Bernoulli likelihood). Certified: R-hat ≤ 1.04 · ESS ≥ 112 · 0 div · BFMI ≥ 0.41 · cross-seed Tucker
  φ **0.993**. `scripts/s5_certify9.py` → `reports/11_s5_9dim_report.md` (engine: `prepare_mixed` gained
  `explicit_factors`/`min_cohorts`; `S5_FACTORS`).
- **Scoring (§7, DONE):** per-patient coordinates for all 9,013 — 6 continuous-anchored dims (incl. mania)
  via conditional-Gaussian from the fixed 9-dim loadings + 3 explicit (suic/dev/substance) via f_e —
  each with mean/SD/HDI + reliability tier. `scripts/07_score.py` → `results/face/patient_scores.parquet`.
- **Atlas + adjudication (§2.3/§6, DONE — M1 LOCK pending PI review):** prior→posterior heatmap at 9 factors
  (`docs/figures/empirical_atlas.png`) + `docs/ADJUDICATION.md`: **9 confirmed** (G + cognition, metabolic/
  inflammatory split, sleep, developmental-proxy, suicidality, mania, substance), **anhedonia rejected**,
  impulsivity/negative/sensory **not_testable**, depression/anxiety = windows. No candidate deferred.
- **Mixed-model PPC (§8, DONE):** absolute-fit check for the non-Gaussian block (the continuous block was
  §5, SRMR 0.07). True posterior-predictive on the 9-dim cert — **21/22 indicators reproduce** their
  observed endorsement rates/means (Bayesian p ≈ 0.5); lone flag `isf09a` (zero-inflated attempt count,
  item-level — the suicidality factor's 7 binary items all reproduce). `scripts/12_mixed_ppc.py` →
  `reports/12_mixed_ppc_report.md`, `docs/figures/mixed_ppc.png`.
- **Invariance of mania + substance (§8, DONE):** per-cohort joint 9-dim fits — **substance invariant
  BP–SZ** (φ 0.997, loadings converged R-hat ≤1.06; the overall SZ R-hat 1.86 was the under-identified
  mania-in-SZ, not substance); **mania partially invariant** — YMRS holds BP–DR (0.57/0.41), Altman
  doesn't transfer to DR (0.76→0.10, φ 0.764, a real converged partial). `scripts/13_invariance9.py` →
  `reports/13_invariance9_report.md` (engine: `prepare_mixed` gained `cohort_subset`).
- **M1 complete** — the measurement layer is built, hardened (confirmation/invariance/robustness/PPC),
  **documented at the largest N that mixes (9 dims)**, scored, and adjudicated. PI sign-off on the adjudication + atlas locks it; then
  **M2 strata**. *Remaining small follow-ons: bootstrap-robustness + corr-G for mania/substance (they carry
  the 9-dim cross-seed φ 0.993 + low G-loadings); full-N non-Gaussian scoring; hurdle likelihood for isf09a
  if its count precision is needed.*
- **Compute lesson (this session):** full-N S1/S2 ≈ 1 h; the S3+ mixed-likelihood frontier is heavier, so
  S3 checkpoints use a random N=4,000 subsample (§3.6). Engine perf fixes: grouped-GEMM Woodbury (Cholesky
  per observed-pattern, 2.75×), tree-depth cap 8 + ta 0.85 (2.7× at 7 factors). Φ bug fixed (LKJCorr=Cholesky
  → Φ = L Lᵀ). **No GPU was needed** — the reparam ladder (marginalization + rung-3 tightening) fit the
  mixed 9-dim S5 on the Mac via the detached + caffeinate + per-seed-cache pattern.
- **Later milestones:** **M2 strata — plan LOCKED, building** (see the M2 section above + [`STRATIFICATION_MODEL.md`](STRATIFICATION_MODEL.md)); temporal coherence V1–V4 (M3) · prognosis (M4) · treatment (M5) not started.

## M1 measurement — per-stage development record (retired)

The staged-fit development log (the S1–S5 convergence checkpoints and the provisional
7-dimension global fit that the full-N-certified continuous backbone and the
largest-N-documented 9-dimension joint map superseded) has been retired from this
status file to keep one current narrative. The canonical M1 record is
[`M1_FINDINGS.md`](M1_FINDINGS.md) (paper-facing), [`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md)
(methods of record), [`ADJUDICATION.md`](ADJUDICATION.md) (per-candidate verdicts) and
[`RESULTS.md`](RESULTS.md) (findings log), with per-stage tables under `reports/04_stage*`.
Biology–G independence is reported in [`RESULTS.md`](RESULTS.md) / [`M1_FINDINGS.md`](M1_FINDINGS.md);
certification tiers in [`CERTIFICATION_TIERS.md`](CERTIFICATION_TIERS.md).

## What to read

[`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md) (methods + math) · [`RESULTS.md`](RESULTS.md) (findings log)
· [`PRIOR_ATLAS.md`](PRIOR_ATLAS.md) (prior
map) · [`../README.md`](../README.md) (overview) · [`../CLAUDE.md`](../CLAUDE.md) (guide) ·
[`DATA.md`](DATA.md) (data contract).
