# Response to the pre-submission issue register

Point-by-point response to the external pre-submission audit. Each item was verified against the actual
code/docs (with `file:line` evidence) before acting. Status legend: **FIXED** (code/analysis change),
**DOC** (wording/number correction), **DONE-ALREADY** (the project already does this), **REFUTED** (the
claim is incorrect), **DOCUMENTED** (real limitation, now explicitly stated). All work is on branch
`remediation/issue-register`.

**Headline outcome:** the audit was largely fair and technically sharp. We implemented the substantive
fixes and re-ran the affected analyses — and **not one scientific conclusion was overturned**. The fixes
make the methods defensible *and* show the results are robust to them: biology⊥G **strengthens** under
covariate adjustment; the coherent-scoring/full-`S_i` fixes leave the map unchanged; the rare archetypes
are now uncertainty-qualified (reinforcing the no-biotype framing); the durable-biology prognosis survives
leave-one-anchor-out. A few claims were incorrect (P0-02 conflated three fits; P0-03 quotes a phrase absent
from the repo) and several concerns were already handled (P0-07, P5-02, P6-01/06, P8-01/03).

---

## P0 — submission blockers

| # | Status | Action / evidence |
|---|---|---|
| P0-01 certification language | **DOC** | The 9-dim mixed block meets a weaker bar than the strict gate; the code's `certified` flag is already correctly False for it. Adopt explicit tiers (full-N certified continuous backbone / largest-N-documented mixed block / internally-validated downstream) consistently — see `docs/CERTIFICATION_TIERS.md`. |
| **P0-04 covariates** | **FIXED** | The published equation advertised `β_jᵀc_i` (age/sex/edu/site) that the engine never implemented. Added `prepare(covariate_adjust=True)` (FWL residualization; `02_build_covariates.py`) + `10_covariate_sensitivity.py`. **Result: biology⊥G survives and *strengthens*** — metabolic~G 0.124→0.058, inflammatory 0.071→0.056 (the confounding was inflating it). `docs/COVARIATE_SENSITIVITY.md`. |
| P0-05 ESEM hard-zero | **DOC + sensitivity** | The ~980 `unlikely` cells are hard-zeroed, not the documented `Normal(0,0.05)`. `prepare(soft_unlikely=True)` instantiates the soft prior; `10b_softzero_sensitivity.py` shows soft vs hard agree (the cells carry no signal). Reword: "Bayesian sparse bifactor with selected ESEM windows; unlikely cells shrunk/fixed to ~0". |
| P0-06 biology-G arithmetic | **DOC** | The stale "92–98%" came from the superseded r=0.28; with the clean correlated-G Φ 0.071/0.124 it is **98.5–99.5%** (metabolic 98.5%, inflammatory 99.5%). One definitive table is the single source of truth (`docs/BIOLOGY_G_TABLE.md`); the manuscript states "largely independent", not a percentage. |
| P0-07 factor-identity attribution | **REFUTED** | Two clearly-labelled code paths (bifactor / correlated-G) and a two-column attribution table already exist (`RESULTS.md:584`, `05_methods.tex:206`). |
| P0-08 external validation | **DOCUMENTED** | Absent and honestly disclosed; the title/abstract make only internal claims and list external validation among the not-met requirements. No external cohort available. |
| P0-02 sample-size mismatch | **REFUTED + DOC** | The *reported* fit is N≈2000/1884-explicit, exactly as stated; the N=4000/5000 figures are disclaimed S3 checkpoints / a historical first fit the reviewer conflated. Residual stale "reported 7-dim/5000 map" labels corrected. |
| P0-03 "additional sampling, not a change of method" | **REFUTED** | The quoted phrase does not exist in the repo; the docs honestly disclose the full-N claim applies to the Gaussian block only and attribute convergence to reparameterization. |

## P1 — measurement / inference

| # | Status | Action / evidence |
|---|---|---|
| P1-02 skip-logic double-counting | **DOCUMENTED** | Real but low-impact (the suicidality factor is carried by 7 binary ISF items that pass PPC); the over-counting is documented. |
| P1-03 isf09a hurdle | **FIXED (opt-in) + DOCUMENTED** | Implemented a hand-written differentiable hurdle (`_hurdle_nb_logp`; `pm.HurdleNegativeBinomial`'s betainc gradient fails under JAX). It fixes the item-level PPC but destabilizes the suicidality↔developmental Φ cell (re-fit seed-1 R-hat 1.55 vs 1.01), so it is an **opt-in sensitivity**; the plain-NB original fit remains the reported map with `isf09a` as a documented item-level misfit (the reviewer's accepted alternative). |
| P1-04/05/06/07 PPC / method effects / reliability / standardized loadings | **PARTIAL** | Reliability tiers + per-axis coverage already reported; expanded PPC / method-factor / standardized-loading layers are noted as further work. |
| P1-08 prior-free "exactly" | **DOC** | Softened to "to 3 d.p." + noted the refit is genuinely independent (flat priors, fresh seed, no warm-start; verified distinct idata MD5). |
| P1-09 WAIC-only / PSIS-LOO | **DOC** | Removed the stray "PSIS-LOO" claim (`az.loo` is M4-only); M1 confirmation is legitimately WAIC + PPC + prior-free refit. |
| P1-10 count prior | **DOC** | Code uses `HalfNormal(2.0)` (the reported fit's prior); corrected the docs/manuscript (they said `Exponential(1)`) to match, on the NB concentration. |

## P2 — scoring / uncertainty

| # | Status | Action / evidence |
|---|---|---|
| P2-01 posterior-mean scoring | **FIXED + DOC** | Corrected the false "propagates parameter uncertainty" docstring; the coherent scorer propagates explicit-latent + conditional + cross-block uncertainty (measurement params at posterior mean = the documented full-N frontier). |
| P2-02 incoherent joint 9D draws | **FIXED** | `coherent_joint_coords`: every 9D draw is one model state with the explicit-block's **own G** + `f_m\|f_e` under the shared Φ (was: two unrelated objects, wrong G). QC: map robust (per-dim r≥0.996). |
| P2-03 fit index reconstructed | **FIXED** | `face.io.manifest` persists the exact `index.parquet` + a manifest (N, cohort, seed, index-hash, commit, versions) per fit. |
| P2-04 diagonal vs full `S_i` | **FIXED + sensitivity** | `coherent_joint_coords` exports the full per-patient `S_i`; XD now accepts it (`22b`). Result: diagonal-`S` justified (partition ARI 0.914, means move ≤0.045). |

## P3 — stratification

| # | Status | Action / evidence |
|---|---|---|
| P3-01 structure-gate BIC mislabel | **DOC** | The report's gate table cited the XD equation/numbers for a row the gate computes with sklearn-GMM BIC; relabelled to the gate's own monotone GMM-BIC (article was already correct). |
| P3-02 "unanimous no-biotypes" | **PARTIAL** | Soften to "no evidence for well-separated biotypes" (planned); the gate is already conservative. |
| P3-03 A=8 wording | **DONE-ALREADY** | Manuscript already uses "archetypal corners". |
| P3-04/05 archetype location uncertainty | **FIXED** | `archetype_location_uncertainty` re-fits anchors over 40 draws + 40 bootstraps (Hungarian-aligned); rare corners now interval-qualified (A1 suicidality [2.27, 9.91], A5 inflammatory [1.09, 7.01]) — reinforces the soft-corners-not-biotypes framing. |
| P3-06 missingness-artefact metric | **FIXED** | Added balanced-accuracy / macro-F1 / log-loss / permutation; coverage predicts membership **below chance** (perm p=1.00) — conclusion upheld and strengthened. |
| P3-07 soft-weight transitions | **PARTIAL** | Noted for temporal-strata reporting. |

## P4 — temporal

| # | Status | Action / evidence |
|---|---|---|
| P4-01 scalar/threshold invariance | **FIXED** | `intercept_drift` (anchor-based scalar-invariance ANCOVA; `33b`). overall_severity is scalar-invariant (drifts ≤0.07 SD → mean-slide is genuine patient change); sleep/developmental partial → softened; binary suicidality flagged for a logistic threshold test (claim softened meanwhile). |
| P4-02 inflammatory partial invariance | **REFUTED** | Already computed, licensed, and consistently caveated (acute-phase WBC; DR non-invariance disclosed). |
| P4-03 developmental recall noise | **FIXED (evidence)** | The scalar test provides the recall-noise evidence — developmental's intercept drifts (≤0.69 SD) are the CTQ reporting instability, now distinguished from true change. |

## P5 — prognosis

| # | Status | Action / evidence |
|---|---|---|
| P5-01 leave-one-anchor-out | **FIXED** | `47b`: re-scored G with the EGF anchor masked (corr 0.987 with full G); the durable-biology ΔR² for 2y EGF is **identical** (0.0039, p<0.001) — survives, not circular. |
| P5-02 group-level framing | **DONE-ALREADY** | The findings already frame it as modest, group-level, ΔAUC +0.017, "not an individual-risk calculator". |
| P5-03/04/05 hard endpoints / tipping-point / multiplicity | **PARTIAL** | Scale-trajectory limit documented; IPW (attrition) already a robustness mode; tipping-point + predeclared multiplicity noted (the effect is small, so its attrition tipping-point is modest — consistent with the "modest signal" framing). |

## P6 — treatment

| # | Status | Action / evidence |
|---|---|---|
| P6-01 target-trial framing | **DONE-ALREADY** | Eligibility / assignment / comparator / estimand are named and operationalized (overlap gate, active-comparator). |
| P6-02/03/04/05 timing / mediator / time-varying / shrinkage | **PARTIAL** | M5 is baseline-observational by design and heavily hedged; add the "baseline biological state, not pre-treatment biology" caveat (P6-03). |
| P6-06 clozapine non-estimable | **DONE-ALREADY** | Declared channeled / non-estimable in code, findings, and manuscript. |

## P7 — reproducibility

| # | Status | Action / evidence |
|---|---|---|
| P7-01 lockfile | **FIXED** | Regenerated `requirements.lock` (75 exact pins incl. the Bayesian backend; dropped stale V2 deps + the foreign path); `scripts/gen_lockfile.py`; CI now installs `.[bayesian]`. |
| P7-02 golden tests | **FIXED** | `tests/golden/`: Woodbury ≡ dense MVN, hurdle ≡ scipy, conditional scoring ≡ analytic, XD-EM recovery + deconvolution, coherent `f_m\|f_e`, synthetic-data engine recovery. |
| P7-03 synthetic generator | **FIXED** | `synthetic/generate_face_like.py` + `FACE_DATA_DIR` override; the engine runs on synthetic data and recovers the planted structure. |
| P7-04 source data per figure | **PARTIAL** | Figures regenerate from shareable aggregates; a per-figure `source_data/` bundle is noted for submission. |
| P7-05 availability statements | **PARTIAL** | Add journal-format Data/Code Availability sections (confidential-data + synthetic + aggregate). |

## P8 — manuscript positioning

| # | Status | Action / evidence |
|---|---|---|
| P8-01 too broad | **REFUTED** | One dominant headline (biology least severity-entangled) + a descending chain; later layers downgraded in-text. |
| P8-02 negative findings as novelty | **DONE-ALREADY** | "a calibrated map, not an overstated one". |
| P8-03 clinical-utility language | **REFUTED** | No overclaim terms; explicit scientific-vs-clinical boundary. |
| P8-04 claims matrix | **FIXED** | `docs/CLAIM_MATRIX.md`. |
| P8-05 references | **PARTIAL** | Expand methods citations (Bayesian SEM, ESEM, invariance, MNAR, XD, target-trial). |
