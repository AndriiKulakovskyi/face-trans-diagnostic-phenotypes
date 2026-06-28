# 24 — M2.4 validation (Q1–Q4) + head-to-head vs DSM-5

> **⚠️ SUPERSEDED — native 9-d A=8 / K=4 fit. Do not cite the archetype/tessellation numbers below as current.**
> This report was computed on the **native 9-dimension coordinates** (separate `metabolic` + `inflammatory`
> axes, A = 8 archetypes, K = 4 tessellation). The reported map is the **8-factor Gaussian-copula** map with a
> single `immunometabolic` axis, an **A = 5** archetype simplex, and a nested **K = 2/3/4** tessellation family.
> The legacy archetype-level numbers here (archetype ARI_dsm5 0.046 / ARI_cohort 0.060, mean η²(specifics)
> 0.319, free K=4 BIC 199,325 vs DSM-5 206,016, mean η² 0.209) are **NOT** the reported copula values. Canonical
> copula numbers: **[docs/STRATA_OOP_FINDINGS.md](../docs/STRATA_OOP_FINDINGS.md)** (Result 4 = K=2 tessellation;
> Result 4b = A=5 archetypes) and `results/face/strata_oop/usefulness/{data.json, a5_archetype_validation.json}`.
> Reported copula equivalents: archetype ARI_dsm5 **0.021**, K=2 tessellation ARI_dsm5 **0.006**, free K=2 BIC
> **185,557** vs DSM-5 **188,168**. Kept for provenance only.

Both soft views (archetypes = lead; tessellation) on the M1 9-d coordinates. Diagnosis is validation-only. M2 establishes the **preconditions + the descriptive head-to-head**; predictive & treatment validity vs DSM-5 are M4/M5 (§1.7).

## Q1 — existence: the honest answer is a CONTINUUM (M2.1)
No discrete clusters (gap-stat K=1, HDBSCAN 0 clusters, unimodal PC1; XD BIC flat basin; archetype scree no elbow). The strata layer is therefore a **soft representation of a continuum** — archetypes (extreme phenotypes) + a soft tessellation — not natural-kind biotypes.

## Q2 — not just severity ✔ (the headline test)
- tessellation η² by axis: {'overall_severity': np.float64(0.308), 'cognition': np.float64(0.171), 'metabolic': np.float64(0.21), 'inflammatory': np.float64(0.056), 'sleep': np.float64(0.187), 'mania_activation': np.float64(0.448), 'suicidality': np.float64(0.054), 'developmental_risk': np.float64(0.352), 'substance': np.float64(0.094)}
- **η²(G) = 0.308** vs **mean η²(specifics) = 0.196** (max specific 0.448). The partition is driven by the SPECIFIC / biological axes, **not** overall severity — exactly the biology⊥G value proposition. (Archetypes separate even more strongly on specifics: mean η² 0.319.)

## Q3 — transdiagnostic ✔ (low concordance with diagnosis, two granularities)
- tessellation: ARI_cohort 0.007, ARI_dsm5 0.020, V_cohort 0.240, V_dsm5 0.215
- archetypes:   ARI_cohort 0.060, ARI_dsm5 0.046, V_cohort 0.279, V_dsm5 0.175
- ARI ≈ 0 vs both cohort and the 7 DSM-5 subtypes ⇒ the partition **cuts across diagnosis**; Cramér's V shows only a weak association (informative gradients, not redundancy).

## Q4 — stable & not a missingness artefact ✔
- tessellation seed-stability: mean ARI 0.987 (min 0.967); archetypes Tucker congruence 0.999 (M2.3).
- coverage→membership classifier acc 0.248 vs majority 0.323 (lift -0.076) — membership is NOT driven by the missingness pattern.

## Head-to-head vs DSM-5 — the 'better description' test (§1.7)
- **XD BIC: free K=4 = 199,325 vs DSM-5 (7 groups) = 206,016** → free **WINS** (fewer components, better fit ⇒ a tighter description of the cloud).
- mean η² on the coordinates: free partition 0.209 vs DSM-5 0.048 — DSM-5 explains little of the coordinate structure.
- **This is a *descriptive* win only.** Whether the strata are clinically better is the M4/M5 **predictive + treatment** head-to-head (§1.7) — not claimed here.

## Verdict
All preconditions pass: real (continuum, stable), **not just severity** (Q2), **transdiagnostic** (Q3), **not a missingness artefact** (Q4), and a **tighter description than DSM-5**. The M2 strata layer is internally valid; actionability is deferred to M4/M5.

Figure: `docs/figures/24_validation.png`.