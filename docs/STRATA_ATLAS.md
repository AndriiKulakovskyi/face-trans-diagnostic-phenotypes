# STRATA ATLAS — the M2 stratification map (§8)

> The detailed per-view record of Milestone 2: the structure verdict, the archetype atlas (lead), the soft
> tessellation, and the validation summary. Methods of record: [`STRATIFICATION_MODEL.md`](STRATIFICATION_MODEL.md);
> **findings + discussion: [`STRATA_FINDINGS.md`](STRATA_FINDINGS.md)**; detailed development record:
[`STRATA_RESULTS.md`](STRATA_RESULTS.md); the measurement layer:
> [`M1_FINDINGS.md`](M1_FINDINGS.md). Per-patient hand-off: `results/face/patient_strata.parquet`
> (gitignored). Every number is backed by a `reports/2x_*.md`. Updated 2026-06-09.

## The map in one line

On the certified M1 9-dimension coordinates (N = 9,013, with per-patient uncertainty), the transdiagnostic
space is a **graded continuum, not discrete biotypes**, represented two complementary, uncertainty-propagating
ways: **8 extreme phenotypes (archetypes — lead)** + a **4-region soft tessellation** — both **transdiagnostic**
(ARI ≈ 0 vs diagnosis), **specific-axis-driven** (not severity), **stable**, **not a missingness artefact**,
and a **tighter description than DSM-5**. Internal/descriptive validity only; actionability → M4/M5.

## Structure verdict — CONTINUUM (§3.1; `reports/21_structure.md`)

Run on both G-arms, uncertainty-aware over draws; conservative synthesis defaults to continuum unless
evidence converges on clusters.

| diagnostic | result | reads as |
|---|---|---|
| Gap statistic | **K = 1** | no clustering over a single blob |
| HDBSCAN | **0 clusters, 100% "noise"** | no density peaks/valleys |
| Hartigan dip (PC1) | **p ≈ 0.99** | unimodal main axis |
| KMeans silhouette | peak **≈ 0.18**, declining | no separated groups at any K |
| GMM-BIC | drop to K≈3 then **flat plateau** | tiles anisotropy, no natural K |
| XD mixture BIC | **flat basin** (K4 199.3k / K5 199.3k) | no separating K |
| Archetype scree | **smooth, no elbow** (ev 0.24→0.79) | no natural number of extremes |
| Mapper (lens=severity) | **1 connected chain** | graded backbone, not islands |
| UMAP | one diffuse blob; cohorts + 7 DSM-5 subtypes **fully intermixed**; smooth severity & inflammatory gradients | continuum + transdiagnostic + biology⊥G |

*(Hopkins 0.85 is the lone high signal — known upward bias in structured high-dimensional data — and is
outweighed.)* Figures: `docs/figures/21_{selection,embedding,mapper}.png`.

## Archetype atlas — the lead view (A = 8; §3.3; `reports/23_archetypes.md`)

Eight stable extreme phenotypes (cross-seed Tucker congruence **0.999**; explained variance 0.79); each
patient is a convex blend (75% are blends, not dominated by one). Profiles in z-units (higher = more burden);
"defining axes" = the extremes; share = % of patients whose dominant (argmax) weight is this archetype.

| # | data-driven label | defining profile (peak z) | share | diagnostic composition (validation-only) | coverage caveat |
|---|---|---|---|---|---|
| A0 | **low-burden pole** | all axes low (sev/sleep/dev/metab/inflam ≈ −1) | **37%** | BP-heavy; few DR | — |
| A2 | **high severity + cognitive burden** | cognition 2.3, severity 2.1, ↓suicidality | 16% | draws the most **SZ (760)** and **DR (222 = 40% of DR)** | — |
| A3 | **sleep/circadian** | sleep 2.6, ↓cognition, ↓dev | 16% | mixed, BP-leaning | — |
| A4 | **metabolic** | metabolic 3.7, ↓suicidality, ↓dev | 13% | mixed (BP 755 · SZ 380 · DR 57) | — |
| A6 | **developmental / early-adversity** | developmental 5.1, ↓metabolic, ↑sleep | 8.5% | BP/SZ-leaning | — |
| A7 | **mania / activation** | mania 5.0, ↑sleep, ↑inflam | 5.5% | **almost entirely BP** (DR ≈ 0) | mania *partial* for all patients (2 indicators) |
| A5 | **inflammatory (+ substance)** | inflammatory 6.6, substance 2.4, ↓suicidality | **1.9%** | mixed | rare tail; inflammatory prior-dominated for 1,684; substance 2-cohort |
| A1 | **suicidality** | suicidality 8.1, developmental 2.6, metabolic 2.4, substance 2.1 | **1.5%** | mixed | rare tail of a skewed latent |

**Non-corners (a result, §F8):** **overall severity** forms no corner — it is the continuum's *spine* (every
archetype sits at some severity); **substance** forms no corner — it self-down-weights (2-cohort, DR-absent,
noisy) and appears only as a side-loading on A5. Corner-survival across A (`reports/23b_archetype_compare.md`):
metabolic/developmental/suicidality/sleep at A≥5, +cognition A≥6, +mania A≥7, **+inflammatory only at A=8** —
A=8 is the parsimony that resolves **both** biology phenotypes. Figures: `docs/figures/23_{scree,profiles,membership}.png`,
`23b_compare.png`.

## Soft tessellation — the coarse overlay (K = 4; §3.2; `reports/22_tessellation.md`)

Measurement-error mixture via Extreme Deconvolution (`x_i ~ Σ_k π_k N(m_k, V_k + S_i)`) — deconvolves the
known per-patient noise; 92% of patients have a confident MAP component. Four regions of the continuum (not
kinds):

| # | label | profile (m_k, z) | share | composition |
|---|---|---|---|---|
| T0 | **low-burden** | ↓developmental, ↓sleep, ↓severity | 31% | mixed |
| T2 | **high-severity + metabolic** | severity 0.6, metabolic 0.4 | 32% | DR/SZ-heavy (the acute/impaired region) |
| T3 | **low-metabolic / better-cognition** | ↓metabolic −0.6, ↓cognition −0.5 | 25% | BP-leaning |
| T1 | **mania + developmental + sleep** | mania 1.3, developmental 1.2, sleep 0.5 | 12% | **BP-heavy** |

Figures: `docs/figures/22_{bic,profiles,membership}.png`.

## Validation summary (§7; `reports/24_validation.md`)

| gate | metric | result | verdict |
|---|---|---|---|
| **Q1 existence** | structure battery | continuum (no discrete clusters) | honest — no biotype claim |
| **Q2 not-just-severity** | per-axis η² of partition | mania 0.45 · dev 0.35 · severity 0.31 · metabolic 0.21 · sleep 0.19 · cognition 0.17; max specific > G | **✔** specifics drive it |
| **Q3 transdiagnostic** | ARI vs cohort / DSM-5 | **0.007 / 0.020** (tess); 0.06 / 0.05 (arch); Cramér's V 0.18–0.28 | **✔** cuts across diagnosis |
| **Q4 stable** | seed ARI / congruence | tess **0.987**; arch **0.999** | **✔** reproducible |
| **Q4 not-artefact** | coverage→membership classifier | acc **0.248** vs majority 0.323 (**lift −0.08**) | **✔** not missingness-driven |
| **vs DSM-5 (descriptive)** | XD BIC, free vs DSM-5-constrained | **199,325** (K=4) vs **206,016** (7 groups) | **✔** tighter, fewer components |
| **vs DSM-5 (descriptive)** | mean coordinate η² | free **0.209** vs DSM-5 **0.048** | **✔** DSM-5 barely structures the coordinates |

Figure: `docs/figures/24_validation.png`.

## The per-patient hand-off

`results/face/patient_strata.parquet` (9,013 × 29; gitignored): `cohort · patient_id · arch_w0…w7
(+ _sd) · arch_dominant(_name) · arch_entropy · tess_r0…r3 · tess_MAP(_name) · tess_entropy · arm`
(diagnosis carried for validation only). Uncertainty preserved so M3/M4 can propagate it. Underlying
coordinates + draws: `results/face/m2/{coordinates_full.parquet, coordinates_draws.npz}`.

## M2 status & honest boundaries

**M2 is complete (pending PI sign-off).** The stratification layer is a soft, uncertainty-propagating
representation of a transdiagnostic continuum — archetypes (lead) + tessellation — that is internally
validated (real · stable · transdiagnostic · specific-driven · not-an-artefact) and descriptively tighter
than DSM-5. **What M2 does not claim:** that the strata are natural kinds; any predictive, prognostic,
temporal, or treatment value; external validity. **Deferred (by design):** decision-relevance — the
predictive (M4) and treatment-moderation (M5) head-to-heads vs DSM-5 (the validators that matter, §1.7),
and temporal persistence (M3). *PI sign-off on this atlas + the findings locks M2; then M3.*
