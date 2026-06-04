# Manuscript plan — FACE trans-diagnostic v2 (Phase 6)

> 📁 **ARCHIVED — executed pre-registration** (moved to `docs/planning/` in the v2 cleanup). This is
> the plan *as written before the manuscript was written*, kept for provenance. For the delivered
> paper see [../../results/manuscript/manuscript.md](../../results/manuscript/manuscript.md).

> Companion to `results/manuscript/manuscript.md` (source) → `results/manuscript/FACE_trans_diagnostic_v2.docx`.
> Figures: `scripts/figures_manuscript.py` → `results/reports/figures/fig1..6.png`.

## Working title
*Symptoms are orthogonal to biology: an integrated, imputation-free dimensional model dissolves the
general psychopathology factor across bipolar disorder, schizophrenia and major depression (FACE).*

## Lead message (non-derivative contribution)
Most dimensional/HiTOP evidence is **symptom-only** and recovers a general *p*-factor. We build an
**integrated** symptom + biology + cognition measurement model under a strict **no-imputation** design
and show the *p*-factor is a **symptom-only artifact**: admit *structured* biology/cognition and the
general factor monotonically dissolves (first-factor share 0.33 → 0.09). Lead with **Study B**; report
**Study D honestly**; put **measurement-design limits up front**.

## Section order & budget (~8,000 words main text)
1. **Introduction** — categorical→dimensional; the *p*-factor controversy; the symptom-only gap; three questions (dimensional vs categorical; does a general factor survive integration; incremental over DSM).
2. **Methods** — full formalism: cohorts → 3-stage processing (scaling eqns) → why aggregate (count-bias derivation; conditioning) → masked estimator (pairwise-complete → nearest-PD; PAF; masked posterior scores) → hierarchical model Stages 0–3 (within-construct 1-factor; second-order promax; Schmid–Leiman ECV; split-half Tucker K) → stratification engine → validation A–D (confound η²; LCO congruence; granularity CCA; discrete-time survival; circularity & attrition).
3. **Results** — R1 four axes + no *p*-factor; R2 dimensional not categorical; R3 **symptoms ⊥ biology (headline)**; R4 cohort-confound refuted + measurement-coverage map; R5 longitudinal coherence; R6 predictive validity vs DSM.
4. **Discussion** — interpretation; measurement-design limits up front; clinical reading; **Anticipated objections** subsection (12 reviewer points); strengths/limits; conclusion.
5. **Supplement** — derivations, per-construct fit, sensitivity (polychoric, K=6, aggregation invariance).

## Figures (6 main; all regenerate from `results/hfa/`)
- **F1** design & analytic pipeline.
- **F2** four axes (top loadings 2×2 + Φ₂ + ECV 0.36).
- **F3** *headline*: block-ordered construct-correlation heatmap + *p*-factor dissolution.
- **F4** dimensional, not categorical (silhouette real-vs-null; unimodal axes; overlap scatter).
- **F5** predictive validity vs DSM (incremental forest + relapse-AUC narrative incl. early-course).
- **F6** longitudinal coherence (structural invariance + score test–retest).

## Tables
- **T1** cohorts & attrition (V0→V4).
- **T2** the four dimensions: defining constructs + validation (cohort η², leave-cohort-out, invariance, test–retest).
- **T3** predictive validity vs DSM (outcome, n, M0–M3, Δ, 95% CI).

## Reviewer anticipation (pre-empted in text)
confirmatory-of-HiTOP → headline B is the novelty · cohort = diagnosis → Study A residualization ·
aggregation circularity → granularity-invariance · "trans-diagnostic" overclaim → internalizing
BP+DR-anchored stated up front · dissolution = dilution → biology/cognition are *structured* yet
orthogonal · tiny prognostic gains → reported honestly, not oversold · relapse gameable →
hospitalization-count rejected + regression-to-mean removed + no leakage · DR tiny → leave-DR-out 0.99 ·
no ML-CFA → no-imputation precludes complete-data ML; fit by cross-validated congruence.

## Remaining Phase-6 tasks (post-manuscript)
Re-baseline `tests/test_golden_numbers.py` + `verify.py` thresholds to v2 (separate from this deliverable).
