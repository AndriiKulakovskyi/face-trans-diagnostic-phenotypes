# M3 — The temporal-coherence model: logic and specification

> Methods-of-record for **Milestone 3 (M3): temporal coherence of the M1 map and the M2 strata across
> follow-up.** This document fixes the *scientific logic*, the *mathematics*, and the *staged estimation*
> of M3; it feeds the manuscript's Methods directly and is the **single methods/plan of record** for the
> temporal layer. Sibling of [`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md) (M1) and
> [`STRATIFICATION_MODEL.md`](STRATIFICATION_MODEL.md) (M2). Scope is **temporal coherence of the V0
> discovery on yearly visits V0 → V1 → V2 only**; prognosis (M4) and treatment (M5) are later milestones.
>
> *Math is plain-text/unicode so it renders anywhere and transcribes to LaTeX.*
> *Status: PLAN — locked at the M3.0 discussion gate, pending PI sign-off. M3.0 (inventory) implemented;
> stages 31–37 held at the gate. Updated 2026-06-10 (the geometric-prediction framing §1.4, Arm-B
> persistence, and the no-completeness-selection rule folded in at the gate).*

---

## 1. Scientific logic

### 1.1 Where M3 sits

The project's four layers stay distinct (M1 §1.1):

```
diagnostic cohorts → transdiagnostic dimensions → validated strata → prognosis / treatment
   (entry metadata)     (M1 — complete, 9-dim)     (M2 — complete)      (M4 / M5)
                                  └──────────── M3: do these cohere & persist over time? ───────────┘
```

M1 delivered a **9-dimension map** with per-patient uncertainty on V0; M2 delivered a
**stratification** of those coordinates — 8 soft **archetypes** + a 4-region measurement-error
**tessellation** — whose headline is that the space is a *continuum* in which **severity (G) is the
spine** and the **specific biology axes are the corners**. Both were discovered on V0.

**M3 asks one question in six parts:** does that signal **cohere and persist over time**? Does the
measurement *mean the same thing* at follow-up (G1)? On the resulting longitudinal substrate (G2),
which of the 9 axes are **trait-like vs state-like** (G3)? Do **archetype/tessellation memberships
persist** — specifically, do patients move along the severity spine while their biology corner stays
(G4)? And is the retained sample a fair one (G6)? M3 is the **precondition + substrate**; it does **not**
claim prognostic, treatment, or external validity. **"Persists" ≠ "predicts."**

### 1.2 Invariants carried forward (non-negotiable)

1. **V0 defines, follow-up validates.** M3 scores V1/V2 onto the **fixed** M1/M2 objects and **never
   re-discovers** the map or the strata on later visits. The certified Λ, Φ, σ, the non-Gaussian item
   parameters, the 8 archetype profiles, and the 4-region tessellation are **frozen** at their V0 values.
2. **No imputation.** Observed-cell likelihood at every visit; attrition is **characterized** (G6), never
   filled.
3. **Diagnosis is metadata.** `arm` (DSM-5 subtype) is validation/benchmark only, never scored. (It is
   also time-invariant in this dataset — §A.)
4. **A change counts only if it exceeds measurement error** (the reliable-change rule, §2).
5. **Internal validity only.** No external cohort.
6. **Only the fixed objects are interpreted.** Per-visit refits exist only for the *confirmatory*
   invariance test (G1) and never replace the reported map.
7. **No completeness selection** (carried from M1 §3.4). Patients are **never** selected on data
   completeness — completeness is *informative* (the acutely ill miss assessments), so selecting "complete"
   patients would shrink state variance and inflate persistence (§5.3). The full retained sample is the
   estimand; sparse data is handled by **uncertainty propagation**, not exclusion.

### 1.3 Decisions locked with the PI (2026-06-09)

- **Balanced full program** — all M3 goals at full depth; **G3 (trait/state)** and **G4 (persistence /
  spine-vs-corner)** are the twin scientific headlines; **G1 (time-invariance)** and **G6 (attrition)**
  are honesty gates.
- **G5 (head-to-head vs DSM-5 over time) is deferred to M4.** DSM-5 diagnosis is time-invariant in the
  data, so a symmetric "strata switch less than DSM labels" test is **not internally measurable** here
  (§A). The signal is *captured* for M4, not analysed in M3.
- **Yearly window V0 → V1 → V2 only.** Dense, all three cohorts well-represented. V3+ (where DR collapses
  to n=3) and the interim `_mois` visits are out of scope for M3.

### 1.4 The central prediction — temporal coherence is a falsifiable test of the M2 geometry

M3 is **not** a generic "do things persist?" check. M2 found a **continuum** in which **severity (G) is the
spine** the cloud slides along and the **specific biology axes are the corners** (the extreme phenotypes;
biology ⊥ G). That geometry makes a **specific, falsifiable temporal prediction**, and testing it is what
"coherent and persistent" *means*:

- **The trait/state profile must align with the spine/corner geometry.** If G is genuinely the spine it must
  be the **most state-like** axis (lowest ICC — it moves with episodes); the corner-defining biology axes
  must be the **most trait-like** (durable phenotypes). **G3 (variance decomposition, §5) and G4 (geometry,
  §6) are two independent routes to this one claim — their *agreement* is the headline of M3.**
- **A spectrum, not a binary.** Expected ordering: G / mania / suicidality most state-like · developmental /
  cognition most trait-like · metabolic / inflammatory / sleep / substance **intermediate** (a stable trait
  *position* plus genuine state fluctuation). "Mixed" is a permitted, expected verdict — the model decides
  (§5.2).
- **The patient-level claim (clinically load-bearing):** phenotype **identity persists while severity
  moves** — a patient slides *down the spine* (severity remits) while staying in the *same corner* (their
  biology phenotype). This is what licenses the project's logic: *stratify on the durable biology, monitor
  the dynamic severity.*

**Falsification — any of these is a reportable negative result:** invariance fails at follow-up (the map is
baseline-specific, §4); the biology corners are **not** trait-like (then they are noise, not phenotypes
worth stratifying on); severity is **not** the most state-like axis (then "spine" was the wrong reading); or
the corner memberships do **not** persist (then the archetypes are not durable types).

---

## 2. The cross-cutting rule: change must exceed measurement error (G0)

Used by G2/G3/G4. For patient *i*, axis *d*, visits *s < t*, on the frozen V0 scale (§3):

```
Δ_{i,d}(s,t) = x_{i,d,t} − x_{i,d,s}
SE_Δ         = sqrt( σ²_{i,d,s} + σ²_{i,d,t} )
RCI_{i,d}    = Δ_{i,d} / SE_Δ          reliable change ⇔ |RCI| ≥ 1.96  (94% HDI of Δ excludes 0)
```

Computed **draw-wise** when posterior draws exist (paired across visits, preserving each patient's
cross-visit covariance and the explicit axes' non-Gaussian skew). Three labels per (patient, axis,
interval): reliable-increase / reliable-decrease / within-noise. A prior-dominated (wide-S) axis yields
few reliable changes **by construction** — the honest behaviour, and the reason G3 must separate
measurement variance from genuine state.

---

## 3. The fixed-model scoring substrate (G2)

### 3.1 The standardization anchor (the load-bearing detail)

The certified loadings live on the **V0 scale**: `prepare()` z-scores each indicator *in-sample*
(`(v − mean)/sd`, lognormal log with an in-sample min-shift, times the burden sign) off
`baseline_v0.parquet`. To score V1/V2 **on the fixed model's scale**, follow-up cells must use the **V0
moments** — re-standardizing per visit would re-centre genuine improvement to ≈0 and erase the change.

We therefore capture a **V0 standardization spec** at the source: a hook in `prepare()` emits, for each
indicator, `(family, sign, mean, sd, logmin)` from the exact loop that builds the fit's matrix (single
source of truth, cannot drift). `apply_spec(spec, B_visit)` reproduces the transform with **frozen V0
stats**; a V1 value outside V0's lognormal support → NaN (correct no-imputation behaviour, **counted and
reported**). The **invariance refit (G1) is the deliberate exception** — it z-scores in-sample per visit,
because Tucker congruence compares loading *shape* and is scale-invariant.

**Guard:** `apply_spec(capture_v0_spec(), baseline_v0)` reproduces `prepare(...).M` to atol 1e-6, and
re-scoring V0 through the spec reproduces M2's V0 coordinates at r ≈ 1.00.

### 3.2 Scoring each visit (pure reuse of the M1/M2 scorers)

Every axis is scored at each visit from its **own observed cells** — *no axis is carried forward* (§1
correction, §5.3):

- **Continuous axes** (severity, cognition, metabolic, inflammatory, sleep, mania): the analytic
  conditional-Gaussian factor-score posterior on the certified loadings —
  `f_i | x_O ~ N( Φ Λ_Oᵀ Σ_OO⁻¹ x_O , Φ − Φ Λ_Oᵀ Σ_OO⁻¹ Λ_O Φ )`, μ = 0 by the frozen standardization
  (`conditional_gaussian_draws`).
- **Explicit axes** (suicidality, developmental_risk, substance): full-N projection — sample each
  patient's `f_e` under the **fixed** certified parameters from their observed non-Gaussian cells, with
  ordinal categories re-mapped to the certified cutpoints (`project_explicit_full_n`,
  `align_ordinals_to_fit`). No re-fit.
- **Memberships per visit:** archetype weights by simplex projection onto the frozen profiles
  (`project_to_Z`, `project_draws` for uncertainty). The **G-residualized Arm B** archetypes (defined in the
  specific-axis subspace, ⊥ G) are the *primary* persistence vehicle — they test corner identity
  *independent of* severity (§6); Arm A (all-9) is the contextual secondary view. Tessellation
  responsibilities under the frozen `(m_k, Σ_k)` with the visit's `S_i` (`mixture._estep_k`) are the coarse
  overlay (needs a 6-line param export from `scripts/22`).

### 3.3 The panel (the substrate every downstream goal reads)

One tidy **long** table, one row per `(patient_uid, visit)` over `{V0, V1, V2}`: the 9 coordinate
means/SDs/HDIs/n_obs/reliability, the archetype weights (+SD)/dominant/entropy, the tessellation
responsibilities/MAP/entropy, the per-axis **G1 invariance license**, and the retention summary. Draws →
a parallel `.npz`. Consolidated hand-off for M4 → `results/face/patient_panel.parquet`.

---

## 4. G1 — Longitudinal measurement invariance (the precondition gate)

A V0→Vk coordinate shift is interpretable as *patient* change only if the measurement holds at Vk. This is
the temporal analogue of M1's cross-cohort invariance (`scripts/06_invariance.py`).

- **Configural** (same pattern): per-visit simple-structure backbone (`prepare(S1_FACTORS,
  correlated=True)` → `corr_no_g_prep`), **visit filter** swapped for the cohort filter, fit per visit.
- **Metric** (equal loadings): Tucker congruence φ of the primary loadings V1/V2 vs V0, same thresholds as
  cross-cohort — **φ ≥ 0.95 invariant · 0.85–0.95 partial · < 0.85 non-invariant**; per-item loading-DIF
  (|Δλ| > 0.20 flagged). Refit z-scored in-sample (φ is scale-free).
- **Scalar** (equal intercepts/thresholds): confounded with true latent-mean change → **anchor-based
  partial scalar**. Fix a few most-invariant anchor items per axis to pin the latent origin; test the rest
  for intercept/threshold drift (94% HDI of Δα excludes 0 → release → documented partial). Where no anchor
  is credible (**mania**, 2 indicators), **decline the level claim** and keep only relative/RCI change.

The per-visit refit may use a **representative, completeness-blind** subsample for speed (a structural test
of the loading *pattern* — as the cross-cohort §6 uses ≈600/cohort). That is *random* subsampling, **not**
completeness selection (§1.2.7); the substantive claims (G3/G4) keep the full retained sample (§5.3).

**Deliverable:** a per-axis verdict {invariant / partial / non-invariant} → `invariance_license.parquet`,
broadcast onto the panel. The license **governs interpretation**: an axis without metric invariance at Vk
is reported descriptively and excluded from G3/G4 *person-change* at that visit (the scores are still
computed — they are needed to test anything — but flagged).

---

## 5. G3 — Trait vs state decomposition (a twin headline)

### 5.1 Estimand and why naïve ICC fails

Per axis: between-patient trait variance σ²_b, within-patient state variance σ²_w, and **known**
measurement variance σ²_e. A standard ICC charges *all* within-patient variation to "state" and so makes
low-reliability axes look spuriously state-like — the exact failure to avoid. The decomposition must
**subtract the known measurement variance**, which M1/M2 already provide per patient per visit. This is the
**variance-decomposition route to the central prediction (§1.4)**: severity must come out most state-like,
the biology corners most trait-like — and it must *agree* with the geometric route (G4, §6).

### 5.2 Method — a Bayesian three-level measurement-error model (the M2 `x ~ N(θ, S)` over time)

```
x_{i,d,t} = μ_d + τ·t + u_{i,d} + ε_{i,d,t} + e_{i,d,t}
u_{i,d}   ~ N(0, σ²_b(d))         patient trait (random intercept)
ε_{i,d,t} ~ N(0, σ²_w(d))         genuine within-person state
e_{i,d,t} ~ N(0, σ²_{i,d,t})      KNOWN M1 error — PLUGGED, not estimated  → separates σ²_e from σ²_w
```

`τ·t` is a fixed visit-time effect so a population trend (everyone improving) is not misread as state.
Estimate only `μ_d, τ, σ²_b, σ²_w` (9 small NumPyro models) on the unbalanced long panel — every observed
(patient, visit) cell contributes, absent visits contribute nothing (no imputation; handles varying #visits
natively). Then:

```
ICC_d = σ²_b / (σ²_b + σ²_w)     trait-like: ICC ≥ 0.6 (CI > 0.5) · state-like: ICC ≤ 0.4 (CI < 0.5) · else mixed
```

An axis whose total signal `σ²_b + σ²_w` is small vs mean σ²_e (no signal above M1 noise) is flagged
**uninformative**, not forced into a bin. Triangulate with a draw-wise raw ICC. Report **completers-only
and all-available** (+ IPW from G6 where dropout is informative). **Expected:** cognition / developmental /
metabolic trait-like; severity / suicidality / mania state-like — but the model decides.

### 5.3 Sampling — full retained sample, uncertainty-propagated (no completeness selection)

The **primary** estimand uses the **full retained sample** at each visit (every observed cell), with the M1
measurement variance plugged (§5.2) so sparse patients self-down-weight through their wide `S_i`.
**Completeness-based subsetting is prohibited as the primary analysis** (§1.2.7): high-completeness patients
are systematically the *stable* ones (the unstable miss visits), so selecting them shrinks σ²_w → axes look
spuriously trait-like and persistence inflates — it would *manufacture* the coherence M3 is testing for. Two
secondary uses are allowed and **labeled as sensitivity**: (i) the **balanced completer panel** (present at
all of V0/V1/V2), reported *alongside* the all-available primary with G6 characterizing how completers
differ; (ii) IPW-of-retention reweighting where G6 finds informative dropout. (Random, completeness-blind
subsampling for the G1 *refit* is fine — §4.)

---

## 6. G4 — Stratum persistence + the spine-vs-corner test (a twin headline)

G4 is the **geometric route** to the central prediction (§1.4): *does the corner persist while the patient
slides along the spine?* Because M2 is a **continuum** with soft, blended memberships (75 % of patients have
max archetype weight < 0.5), persistence is a **coordinate-space, uncertainty-aware** question — *not*
hard-label tracking. A central patient flips argmax-archetype on a tiny move purely by geometry; that is the
continuum, **not instability**. So persistence is read in the coordinates (does the position hold beyond
measurement error, G0?) and is **cleanest and most interpretable at the corners**, expectedly diffuse at the
centre.

- **Spine-vs-corner (the headline test).** Decompose each patient's displacement `Δx_i = Δ_G·e_G + Δ_prof`,
  where `e_G` is the severity axis (spine) and `Δ_prof` is the residual in the 8 specific axes (the corner
  subspace) — the specifics are bifactor-orthogonal to G by construction, so no extra rotation. Test
  **"moves on spine, stable corner"**: `Δ_G` is a reliable change (G0) **and** `‖Δ_prof‖` is *not* reliable
  **and** the dominant corner is unchanged. Population: the paired rate of reliable `Δ_G` vs reliable
  `‖Δ_prof‖` (prediction: the spine moves far more often), as a per-cohort 2×2.
- **Membership persistence — primary on the G-residualized Arm B archetypes.** Arm B lives in the
  specific-axis subspace (⊥ G), so it measures **corner identity independent of severity** — the direct
  formalization of the spine/corner claim; Arm A (all-9) is the contextual secondary view. Statistics: soft
  transition matrices `T[k,l] = Σ_i r_{i,k}(s)·r_{i,l}(t)` over the weight posteriors (boundary patients
  contribute fractionally — honours the continuum); draw-wise weight-vector cosine / TV distance; Cohen's κ
  on MAP labels (secondary). **Prediction: Arm-B (corner) membership persists more than the severity
  coordinate moves.**
- **Trajectory typing** (stable / drifting / oscillating) is **coarse with 3 visits** — descriptive only;
  the coordinate-space persistence + transitions are the robust result.

All statistics over draws; transitions restricted to patients present at both endpoints (no imputed
memberships). Conditioned on G6. **Falsifies the M2 reading** if the corners do not persist or if severity
is not the dominant mover (§1.4).

---

## 7. G6 — Attrition & informative dropout (the honesty gate)

**Estimand:** is retention at V1/V2 related to V0 coordinates/strata/diagnosis — *do the sicker leave, or
the improved?* **Method:** logistic `retained_{i,Vk} ~ V0 9-dim coords (+reliability) + V0 archetype +
cohort + age + sex + site` (effect sizes + 94% CIs; sign of the severity coefficient; whether biology
corners drop differentially); a stayers-vs-droppers V0-profile test. **"Dropout ≠ improvement" guard:**
always report completers-only **and** all-available; primary analyses are MAR-given-observed (the G3/G4
models use all cells), with an **inverse-probability-of-retention-weighted** sensitivity refit flagged
where dropout is informative. Runs **early** (needs only V0 coordinates + the retention table); its verdict
tags every survivorship-sensitive claim downstream. The raw dropout-reason text (diagnosis-change exits in
BP and SZ; deaths, sentinel-corrected) is captured here for M4 — descriptive only in M3 (§A).

---

## 8. M3.0 — the longitudinal-coverage inventory (implemented; the gate evidence)

`scripts/30_inventory.py` (+ `face.temporal.dropout.retention_table`) establishes feasibility before any
scoring. **Findings (2026-06-09):**

- **Retention:** V0 9,013 → V1 4,270 (47%) → V2 2,958 (33%); BP/SZ/DR all well-represented at V1/V2.
- **Per-axis coverage at V1/V2 is healthy** — 8 of 9 axes keep multiple re-administered indicators with
  ≥30 obs (metabolic 32, suicidality 21, developmental 18–19, inflammatory 14, severity 10–12, cognition
  11, sleep 9, substance 4). **`mania_activation` has only 2 indicators → thin** (scored, caveated).
  No axis is coverage-limited.
- **Re-administration ≠ state (the load-bearing nuance).** The within-patient V0→V1 change rate measures
  whether an item was *re-collected with varying answers*, **not** whether the construct moved. The CTQ
  childhood-trauma items — a definitionally *fixed* history — reach change rates ≈ 0.9 from recall noise
  alone. So the inventory decides **coverage only**; **trait vs state is G3** (§5), which deconvolves
  measurement error.
- **Correction surfaced by the gate:** the plan had assumed `developmental_risk` is "static-by-construction
  / carry V0 forward." The data refute this — its CTQ/family items are **re-administered** (only one
  indicator, `epilepsie_mhoccur`, is near-static). Therefore **every axis is scored per visit from its own
  observed cells** (§3.2); scoring `developmental_risk` per visit is in fact *required* to let G3 show it is
  trait-like (most of its V0→V1 variation is CTQ recall noise → high σ²_e, low σ²_w).

Artifacts: `reports/30_{inventory.md, retention.csv, axis_coverage.csv, indicator_temporal.csv}` ·
`docs/figures/30_{retention, axis_coverage}.png`.

---

## 9. Engine & pipeline

**Engine** `src/face/temporal/` (mirrors `src/face/strata/`, reuse-first): `dropout` (G6 retention +
extractor + informative-dropout), `standardize` (V0 spec — the core new piece), `panel` (per-visit tables
+ panel assembly), `score` / `membership` (thin reuse of the M1/M2 scorers), `invariance` (G1, adapts
`scripts/06`), `variance` (G3), `persistence` (G4). **Reused, no new math:** `conditional_gaussian_draws`,
`project_explicit_full_n`, `align_ordinals_to_fit`, `project_to_Z`, `project_draws`, `xd_em`/`_estep_k`,
`tucker_phi` + the `06` harness, `eta_squared`/`ari`/`cramers_v`, `prepare`/`prepare_mixed`.

**Pipeline** `scripts/30–37` (run order = file order; each writes `reports/3N_*.md` + a figure, then a
discussion gate): 30 inventory **(done)** · 31 attrition (G6) · 32 build-panel + V0 spec · 33 invariance
(G1) · 34 score-panel (G2) · 35 variance (G3) · 36 persistence (G4) · 37 consolidate. The one M2 change is
a 6-line `tessellation_fit.npz` export in `scripts/22` (secondary view only).

---

## 10. Sequencing, gates, and risks

**Order:** M3.0 → G6 (early honesty; pure V0 + retention) → build substrate + V0 spec → G1 (license) → G2
(score with license) → G3 → G4 → consolidate. **Gates:** G1 licenses interpreting G2 as patient-change per
axis; G6 conditions G3/G4 for survivorship.

**Risks** (mitigations in §3/§5/§7): (1) re-standardization erasing level change → the V0 spec + the 1e-6
round-trip; (2) survivorship faking improvement → G6 early, completers-vs-all, IPW; (3) low-reliability
axes looking state-like → G3 plugs known σ²_e, G0 needs per-patient change > own error; (4) scalar/latent-
mean confound → anchor-based partial scalar, decline level claims for mania; (5) thin axes (mania,
substance) → reported as scope facts; (6) cognition practice/retest effects → flagged caveat, partly
detectable via the G1 intercept-drift test; (7) 3-visit window limits trajectory typing → lead with
persistence.

---

## 11. What M3 will and will not claim

**Will claim (internal only):** per-axis longitudinal-invariance verdicts; a frozen-scale coordinate +
membership panel over V0–V2; a measurement-corrected trait/state decomposition of the 9 axes; membership
persistence + the spine-vs-corner geometry; **whether the trait/state profile aligns with the spine/corner
geometry — the central prediction (§1.4), tested two independent ways (G3 ⟷ G4)**; an honest attrition
characterization. **Will not claim:**
external / prognostic / treatment value (M4/M5); DSM-5 head-to-head over time (deferred to M4, §A); causal
interpretation. "Persists" = within-cohort temporal coherence under the stated invariance + survivorship
caveats.

---

## Appendix A — Why the DSM-5 head-to-head (G5) is deferred to M4

Verified in the raw data: **DSM-5 diagnosis (`arm`) is time-invariant** — 0 / 9,013 patients change arm
across visits (it is the enrollment diagnosis, carried forward by the harmonizer). Diagnostic conversion
surfaces only as a **sparse exit reason** in the lost-to-follow-up text ("Changement de diagnostic"):
**BP 60 patients, SZ 17** (G6 / stage 31); DR records numeric reason codes (not decoded). It is an *exit*
signal — the patient leaves the cohort and the in-data `arm` never updates — so a symmetric "our strata
switch less than DSM labels" test is **not internally measurable** in M3: DSM labels cannot switch here by
construction. The arm + diagnosis-change signals are **captured by G6** so M4 can use them against genuine
outcomes; the temporal "better than DSM-5" claim lives in M4's predictive limb, not M3.
