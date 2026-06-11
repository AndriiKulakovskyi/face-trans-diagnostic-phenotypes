# M5 — Treatment-response heterogeneity: methods of record

> **The methods + math of record for Milestone 5.** Estimand, the feasibility re-scope, the
> response/resistance/tolerability/adherence endpoints, the severity-confound problem, the engine
> (reused from M4), the acceptance gates, and the staged pipeline. Read before any M5 modelling work.
> Sibling of [`PROGNOSIS_MODEL.md`](PROGNOSIS_MODEL.md). *Status: PLAN (M5.0 in progress). 2026-06-11.*

## 1. The feasibility re-scope (read first)

The classic M5 — *does a stratum **moderate** the effect of treatment X vs Y?* — is **not answerable in
FACE**: the harmonized record contains **no treatment-assignment / medication-class / prescription
variable** (`arm` is the DSM-5 subtype; `agetrt` is age-at-first-treatment, historical). You cannot
estimate a treatment×stratum interaction when the treatment is unobserved.

What the data *does* observe — everyone is on naturalistic **treatment-as-usual (TAU)**, and we have its
**response**: CGI-Improvement (`cgi02`), therapeutic effect (`cgi03a`) and side-effects (`cgi03b`) from
the CGI efficacy index, and adherence (`mars`), all re-administered at V1/V2 (raw harmonized layer; not
in the processed tables). So M5 is re-scoped from treatment **moderation/selection** to **treatment-
response heterogeneity**:

> **Does a baseline transdiagnostic coordinate or stratum predict who **responds to**, is **resistant
> to**, **tolerates**, and **adheres to** treatment-as-usual — incrementally beyond DSM-5 diagnosis and
> baseline severity — thereby flagging treatment-resistant and side-effect-prone phenotypes?**

**The de-scope, stated up front (the M5 analogue of M4's "no events"):** TAU is *unobserved*, so M5 can
say *which phenotypes respond / resist / tolerate*, **not which drug to prescribe** — response
*stratification*, not treatment *selection*.

**Scope decision (2026-06-11) — a focused coda, not a full milestone.** A helicopter-view feasibility
review concluded that, given the data, M5's endpoints split three ways: (i) **moderation/selection**
(the valuable claim) is **data-blocked** — no treatment variable; (ii) **response / resistance** are
**largely M4-redundant** — they are severity outcomes, and M4 already showed the map predicts functioning
but adds ~nothing to severity (autoregression-saturated), a redundancy the M5.0 severity-confound audit
confirms; (iii) **tolerability (side-effects)** is the **one genuinely novel, severity-clean** test the
map never had. So M5 is scoped as a **coda**: (a) the novel **tolerability** test — does the map (esp.
the metabolic / inflammatory archetypes) predict side-effect burden beyond diagnosis + severity (the
pre-registerable *metabolic-phenotype × side-effects* bet)?; (b) a descriptive **treatment-resistance**
reframe of the M4 prognostic atlas (communication value, with the by-design severity caveat); and (c) an
explicit **boundary statement** — genuine treatment-guidance requires prescription/treatment-identity
data FACE lacks, making a data-acquisition check the path to a real M5 (a future *M5b*). The program's
*demonstrated* clinical value currently culminates at M4 (prognosis).

**Distinct from M4.** M4 predicted future severity/functioning *levels* (EGF, CGI-S). M5 uses the
treatment-response-specific signals M4 never touched (the clinician's judgement of how the treatment is
working + tolerability + adherence). House invariants carry over: fixed M1/M2/M3 objects (never
re-discovered/re-scored), observed-cell likelihood (no imputation), diagnosis as comparator/validation
only, internal association ("predicts" ≠ "causes" ≠ "moderates"), a signal counts only if it clears its
uncertainty band.

## 2. Endpoints (configs/m5_outcomes.yaml)

Built from the response signals on their native CGI codings (0 = not-assessed → NaN, never imputed):

| endpoint | definition | role |
|---|---|---|
| **response** | CGI-Improvement responder: `cgi02 ∈ {1,2}` (much / very-much improved) | primary |
| **therapeutic_effect** | CGI therapeutic effect marked/moderate: `cgi03a ∈ {1,2}` | primary (supporting) |
| **resistance** | treatment-resistant: `cgi01 ≥ 4` (moderately ill+) **and** not a responder (`cgi02 ≥ 3`) at the horizon | primary |
| **side_effects** | significant side-effects: `cgi03b ≥ 3` (interfere with functioning) | primary (tolerability) |
| **low_adherence** | `mars` ≤ threshold (default ≤ 5 on 0–10) | secondary (disentangles pseudo- vs true resistance) |

Horizon V2 primary, V1 replication. CGI codings: CGI-I 1 = very-much-improved … 7 = very-much-worse,
0 = NA; therapeutic-effect 1 = marked … 4 = unchanged/worse; side-effects 1 = none … 4 = outweigh.

**Coda roles (post-feasibility, §1):** `side_effects` is the **primary, novel** test; `response` /
`therapeutic_effect` are **confirmatory** (expected M4-redundant — severity outcomes); `resistance` is a
**descriptive reframe** of the M4 atlas (by-design severity-confounded — never a credited map effect);
`low_adherence` is **BP/SZ only** (DR excluded — MARS mis-scaled, M5.0 data-QC). All response endpoints
are **BP/SZ** (DR has no CGI efficacy index).

## 3. The nested ladder and the severity-confound (Q2 is make-or-break)

The hazard that decides M5: **response is severity-confounded.** CGI-Improvement is entangled with
baseline severity (regression-to-the-mean: sicker patients have more room to "improve"; the strata
differ in severity), so a naive "this archetype responds more" can be pure RTM. The ladder, per
endpoint, on one complete-case sample:

```
R0  age + sex + site(random intercept)
R1  + DSM-5 arm
R2  + baseline severity     [CGI-S ; and the error-corrected G coordinate]
R3  + baseline state        (baseline CGI-S / functioning — the autoregressive anchor)
T   + map (durable coords EIV ; Arm-B archetypes ; tessellation)
```

Guards: (i) adjust for baseline severity (manifest CGI-S **and** error-corrected G) — the load-bearing
Q2 control; (ii) prefer the RTM-robust ANCOVA for any level-based endpoint; (iii) note the tolerability
and adherence endpoints are *not* severity-RTM-prone — cleaner tests of the map's treatment-relevant
signal. M5.0 quantifies the confound (endpoint × baseline-severity association) so the bar's burden is
explicit before any map term is added.

## 4. Engine (reuse M4) + one new build step

The estimand, the EIV Bayesian GLM (`face.prognosis.glm` — uncertainty-propagated coordinates, site
random intercept, `az.from_dict` idata), the nested-comparison ΔELPD/LOO (`compare`), the
cross-validated clinical metrics (`clinical_value` — responder AUC, calibration, net benefit), IPW, and
the robustness sweep all carry over unchanged. **New:** the response signals live in the raw harmonized
layer, not `baseline_v{0,1,2}.parquet`, so `face.treatment.frame` extends the extraction to pull
`cgi02/03a/03b`, `mars`, `cgi01` (native, skip-logic, NaN-honest) and join them to the fixed panel
coordinates + strata + covariates + IPW. Engine `src/face/treatment/` (endpoints · frame; reuses
`face.prognosis.*` for modelling). Pipeline `scripts/50–57`.

## 5. Acceptance gates (Q1–Q4)

- **Q1 incremental validity** — the stratum/coordinate beats the diagnosis+severity bar on held-out
  ΔELPD (and a credible coefficient) for a response/resistance/tolerability endpoint.
- **Q2 beyond severity (make-or-break)** — survives adjustment for the error-corrected G severity and
  baseline state; the RTM null does not explain it. Without this, a "response" signal is just severity.
- **Q3 transdiagnostic / vs DSM-5** — within-cohort consistency + the head-to-head dominance vs the 7
  DSM-5 subtypes (outcome ELPD, never agreement).
- **Q4 robust** — IPW (attrition), reliability-stratified, leave-one-cohort-out, permutation null.

## 6. The treatment-response atlas + the falsifiable hypotheses

The clinician-facing deliverable (M5 sibling of the prognostic atlas): per archetype, the 2-year
**response rate, treatment-resistance rate, side-effect burden, and adherence**. Two sharp,
pre-registered, falsifiable hypotheses connecting M4 → M5:
- the **inflammatory** archetype (worst M4 functional prognosis) → **treatment-resistant**;
- the **metabolic** archetype → **higher side-effect burden** (the metabolic-phenotype × metabolic-
  side-effects bet).

## 7. Pipeline — the coda (`scripts/50–53`)

`50_inventory` *(done)* — coverage + endpoint prevalence + the circularity & severity-confound audits ·
`51_frame` — the M5 frame: extract the response signals from the raw layer, join to the fixed panel +
strata + covariates + IPW (DR excluded from adherence) · `52_tolerability` — the novel test (map →
side-effect burden, incremental beyond diagnosis+severity, Q1/Q2 + the metabolic-phenotype hypothesis)
+ the per-archetype side-effect / treatment-resistance atlas + clinical AUC · `53_consolidate` —
findings + the **boundary/data-ask** + STATE/CLAUDE. Reuses the M4 engine
(`face.prognosis.{glm,reference,compare,clinical_value}`); the response/resistance endpoints are run as
confirmatory checks (expected M4-redundant), reported honestly. Tests `tests/m5/`.

## 8. Honest limits

TAU **unobserved** → response stratification, not treatment selection (no drug attribution);
**severity-confounded** response (Q2-gated, RTM-checked); **pseudo- vs true-resistance** (a "resistant"
patient may be non-adherent — `mars` disentangles, partially); clinician-rated CGI; observational,
internal validity only; 2-year horizon; the rare archetypes carry wide uncertainty. Next, only if
treatment-identity data can be linked: true treatment **moderation** (M5b).
