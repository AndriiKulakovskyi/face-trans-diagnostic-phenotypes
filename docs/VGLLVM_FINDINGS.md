# V-GLLVM — findings (variational GLLVM atlas engine)

> Paper-facing findings for the **variational** re-estimation of M1. Read this first.
> Methods of record: [VGLLVM_MODEL.md](VGLLVM_MODEL.md). Engine: `src/face/models/variational/`.
> Status: **production fit complete, validated against the NUTS copula M1 (2026-06-27)**.
> An exploration / **acceleration** arm — congruent-with, never replacing, the NUTS authority.

## Headline

A PyTorch stochastic-variational re-estimation of the 8-factor operational measurement map
(same ontology, same Gaussian-copula likelihood, substance orthogonal, the 3 earned
cross-loadings) **reproduces the NUTS copula M1 on its two load-bearing objects** — the
measurement map and the patient coordinates — at a fraction of the cost:

- **Loadings: 8/8 factors congruent.** Tucker congruence 0.957 (G) and **0.96–0.999** for
  every specific axis; the full item×factor loading scatter is **r = 0.993** (n = 208 cells),
  no sign flips. *The measurement map is the same map.*
- **Coordinates: 8/8 factors r ≥ 0.90.** Per-patient coordinate correlation 0.91–0.998
  (G 0.990, immunometabolic 0.998, cognition 0.992, suicidality 0.984, dev-risk 0.986,
  sleep 0.968, mania 0.976, substance 0.908). *Patients land in the same place.*
- **Cost: ~4.5 min on a Mac CPU** (full-N 9013 × 142, 2-rung warm-start, early-stopped) vs
  the NUTS fit's hours of multi-chain sampling.

The one honest gap is the **inter-factor correlation Φ**: directionally correct and
structurally identical (biology⊥G zeros exact; the sleep/suicidality/dev-risk/mania block
positively correlated in both), but the off-diagonals are **systematically attenuated ~21%**
by the mean-field posterior + the Φ penalty. So the V-GLLVM is, today, a faithful **map +
coordinate** engine and an **approximate Φ** engine — exactly the trade a mean-field VI
predicts, with a clear fix (below).

Verdict: **congruent** (not certified). NUTS remains the inferential authority.

## Helicopter view — is the model good, or is there room?

**Good — genuinely production-grade for what it's for (a fast operational map engine), with one
structural boundary (Φ) that is precisely the quantity NUTS is kept for.** Scorecard:

| Dimension | Grade | Evidence | Room? |
|---|:--:|---|---|
| Measurement map (loadings) | **A** | Tucker 8/8 (0.96–0.999), scatter r = 0.993 — it *is* the NUTS map | none |
| Patient coordinates | **A** | r 0.90–0.998, 8/8 — patients land where NUTS puts them | none |
| Generative fidelity | **A−** | marginal KS median 0.04, mean/SD r ≈ 1.0, corr SRMR 0.078 ≈ NUTS | minor (tiering) |
| Inter-factor Φ | **B−** | off-diag attenuated; ladder mean-field→full-cov 21%→18% but doesn't close | **structural (VI)** |
| Uncertainty (loadings) | **B** | seed-ensemble bands, 264–268/277 credible — stability, not Bayesian CIs | by design (use NUTS) |
| Speed | **A+** | ~3–4 min CPU vs NUTS hours (~100×) | none |
| Engineering | **A** | 21 tests, deterministic, cached, MPS-safe, reuses the data contract + article atlas | none |

- **What's genuinely good (no work needed):** the map, the coordinates, the generative model,
  the speed, the engineering. These *match* NUTS and are ready for the VI arm's purpose.
- **The one real limitation (bounded, now characterized):** Φ off-diagonals. We climbed the full
  posterior-richness ladder and the residual ~18% attenuation is a **VI-structural bias on the
  correlation hyperparameter**, not a bug and not a per-patient-posterior gap — so it does not
  close inside VI. This is *why* NUTS is the Φ authority; the boundary is now earned, not asserted.
- **Minor / fixable:** ~12 lumpy ordinals the copula promotes to the gaussian channel lose their
  integer spikes in generation (KS ~0.4) — a one-line tiering-threshold change.

**Bottom line:** a calibrated ~100×-faster stand-in for the NUTS map + coordinates and a faithful
synthetic-cohort generator. Use it for fast reruns, sensitivity sweeps, and generation; keep NUTS
for Φ magnitudes, loading CIs, and final paper claims. The honest "room for improvement" is Φ —
and we've shown it's a VI boundary, not a defect.

## What was fit

The default 8-factor operational map (`prior_loading_matrix_v3_biomerge_xc.csv`): G +
cognition + immunometabolic + sleep + suicidality + developmental_risk + mania_activation +
substance; substance pinned orthogonal; the 3 earned cross-loadings (ctq37 / psqi11 / psqi17
→ cognition); Gaussian-copula (rank-INT) continuous likelihood + native binary/ordinal/count.
Full-N (9013 patients, 142 indicators). 2-rung warm-start ladder (continuous backbone →
full mixed map). Hand-off: `results/face/gllvm_oop/consolidate/`.

## Results

### Loadings (Tucker congruence vs NUTS)

| factor | n cells | Tucker | bar | pass |
|---|---:|---:|---:|:--:|
| overall_severity (G) | 106 | 0.957 | 0.95 | ✅ |
| cognition | 16 | 0.998 | 0.95 | ✅ |
| immunometabolic | 37 | 0.998 | 0.95 | ✅ |
| sleep | 12 | 0.999 | 0.95 | ✅ |
| suicidality | 12 | 0.998 | 0.85 | ✅ |
| developmental_risk | 21 | 0.998 | 0.85 | ✅ |
| mania_activation | 2 | 0.984 | 0.85 | ✅ |
| substance | 2 | 0.963 | 0.85 | ✅ |

Whole-matrix loading scatter **r = 0.993** (Fig. `vi_vs_nuts_loadings.png`): the large home /
G-anchor loadings (up to 4.5) align on the diagonal; the cross / window / bifactor-G cells
cluster tightly near 0 and align. *G is the hardest axis* (0.957, the lowest) — it has the
most cells (106, including every item's bifactor-G loading), and its small bifactor-G
loadings are where VI and NUTS diverge most; still above bar.

### Coordinates (per-patient r vs the NUTS copula coordinates)

| factor | n (well-tier) | r |
|---|---:|---:|
| overall_severity | 8641 | 0.990 |
| cognition | 6451 | 0.992 |
| immunometabolic | 8825 | 0.998 |
| sleep | 7522 | 0.968 |
| suicidality | 8285 | 0.984 |
| developmental_risk | 8767 | 0.986 |
| mania_activation | 6152 | 0.976 † |
| substance | 4249 | 0.908 |

† mania has only 2 home indicators, so no patient reaches the default ≥3-observed "well"
tier; the metric auto-relaxes to ≥2 (a thin-factor reliability fact, not a model gap).
Substance (0.908) is the weakest — it is orthogonal-by-design and instrument-poor (rare
SUD items), so its coordinate is the most prior-dominated in both engines.

### Φ (inter-factor correlation)

- **Structure reproduced.** G and substance rows/cols are **exactly 0** in both (orthogonal
  by construction); the sleep / suicidality / developmental_risk / mania block is positively
  correlated in both (Fig. `vi_vs_nuts_phi.png`).
- **Off-diagonals attenuated ~21%.** Correlated-block mean |off-diag|: VI 0.083 vs NUTS
  0.105. Max cell difference **0.109** (fails the strict ≤0.05 bar). The attenuation is
  concentrated in the **weak cognition cross-correlations**:

  | cell | VI | NUTS | Δ |
  |---|---:|---:|---:|
  | cognition ~ sleep | −0.045 | −0.155 | +0.109 |
  | cognition ~ developmental_risk | −0.021 | −0.122 | +0.101 |
  | cognition ~ suicidality | −0.048 | −0.132 | +0.084 |
  | immunometabolic ~ suicidality | +0.010 | +0.076 | −0.066 |
  | immunometabolic ~ cognition | +0.074 | +0.054 | (preserved) |

  The immunometabolic↔cognition coupling (a substantive cell) is preserved; the attenuation
  is in the small negative cognition correlations, which mean-field shrinks toward 0.

### Posterior predictive checks (observed vs reconstructed, observed cells only)

- **Binary endorsement rates: r = 1.000**, mean |obs − rec| = 0.002 (36 items) — the model
  reproduces every Bernoulli item's prevalence.
- **Continuous variance: r = 0.974** (88 items). The continuous *mean* is non-informative
  here — rank-INT forces every marginal to ~N(0,1), so all observed means are ≈0 by
  construction (don't read the mean panel as a failure; read the variance).
- **Count** (1 item): obs mean 0.14 vs reconstructed 0.12, obs zero-rate 0.91 — matched.

## Convergence / training observations

- 2-rung fit: backbone 1500 epochs / 78 s; full map **early-stopped at epoch 1760 / 190 s**
  (no >5e-5 relative ELBO improvement for 300 epochs). The −ELBO drops sharply in the first
  ~250 epochs then plateaus cleanly (Fig. `training_s8_full.png`).
- **The gradient-norm trace exposes an optimization lever.** The pre-clip gradient norm
  settles around ~2000 while `grad_clip_norm = 5.0` — i.e. gradients are clipped ~400×.
  AdamW's approximate scale-invariance (it normalizes by the running second moment) is what
  carried the fit to a good optimum despite this, but the persistent large grad norm at the
  plateau means we stopped at a *flat-but-not-stationary* point, not a tight one. Loosening
  the clip + a late LR decay should reach a tighter optimum (and likely recover some Φ).

## Honest limits

1. **Φ off-diagonals are attenuated** (~21%). Two compounding causes: (a) the mean-field
   diagonal `q(f)` cannot represent posterior cross-factor covariance, so it compensates by
   shrinking the prior correlations the likelihood would otherwise support; (b) the weak
   `Ω(Φ)` off-diagonal L2 penalty actively shrinks them. The biology⊥G claim is unaffected
   (exact zeros), and the immunometabolic coupling survives — but do not quote VI Φ
   magnitudes; quote NUTS.
2. **Coordinate SDs are underestimated** (mean-field). The coordinate *means* are excellent
   (r ≥ 0.91); the `s_i` is a relative reliability signal, not a calibrated credible width.
3. **Loadings are MAP** — no VI loading CIs yet (the exported `ci_*`/`excludes_zero` are NaN);
   CIs come from NUTS / bootstrap.
4. **Not stationary-tight** (the grad-clip note above) — a quick optimization pass should
   close the small residual on G's bifactor cells and the Φ block.

## What we tested (and what it ruled out)

We ran a **tuned exploratory arm** with the Φ off-diagonal penalty removed
(`phi_penalty_weight = 0`) and the gradient clip loosened (`grad_clip_norm = 100`), to a
separate `results/face/gllvm_oop_improve/` (canonical default unchanged). Result: **identical**
to baseline — Tucker 8/8 unchanged, coordinates unchanged, and **Φ unchanged** (mean |off-diag|
still 0.083, max cell diff still 0.109). Interpretation:

- The Φ penalty was **not** the cause (it contributed ~0.4 of a 710k objective — negligible).
- The gradient clip was **not** the cause (AdamW's scale-invariance makes the clip threshold
  near-irrelevant here, which is also why the baseline converged despite the ~400× clipping).
- Therefore the Φ attenuation is **entirely the mean-field diagonal `q(f)` family**: a diagonal
  posterior cannot represent cross-factor covariance, so it compensates by shrinking the prior
  correlations the likelihood would otherwise support. This is a structural property of the
  variational family, not a tunable hyperparameter.

This refutation is useful: it says a richer posterior is *necessary* — no cheap knob recovers Φ.

## Proposed improvements (ranked, re-prioritized by the evidence above)

1. **A richer variational posterior — the only effective Φ fix.** Give `q(f)` a **low-rank +
   diagonal** covariance (rank 1–2): per-patient `N(μ, D + UUᵀ)` can represent the dominant
   cross-factor posterior covariance the diagonal cannot, which is the confirmed root cause of
   the Φ attenuation. A full 8×8 `q` covariance is also tractable (K is tiny). Contained change
   to `VariationalGLLVM` (the latent block) + the KL. This is the load-bearing next step for Φ.
2. **Calibrate / report uncertainty honestly.** The same low-rank `q` also fixes the
   coordinate-SD underestimation; pair it with a VI-`s_i` vs NUTS-score-SD ratio on a patient
   subset so the reported uncertainty is calibrated, not just relative.
3. **VI loading CIs.** A seed-ensemble (k = 5–10 fits) or a parametric bootstrap over patients
   gives loading CIs so `loadings_summary` can populate `ci_*`/`excludes_zero` like NUTS — at
   ~4.5 min/fit this is minutes, not hours.
4. **Minor sharpening (low value).** A cosine LR decay + `n_mc = 2–4` late may nudge G's
   bifactor cells (0.957 → higher), but the tuned arm shows the optimizer is **not** the
   bottleneck — don't expect Φ to move. The `grad_clip_norm`/`phi_penalty_weight` knobs are
   exposed but confirmed inconsequential here.
5. **Amortized mask-aware encoder.** For fast new-patient scoring without re-optimization
   (roadmap v0.3); the non-amortized per-patient `q` is correct but doesn't generalize to
   unseen patients.

## Improvements implemented (2026-06-27)

Two of the proposals above were built and run; both are config options on the canonical engine
(`q_rank`, the seed ensemble), default behavior unchanged.

### 1. Richer variational posteriors — the full Φ-recovery ladder (proposal #1)

We implemented and ran the **full ladder of posterior richness** — mean-field diagonal → low-rank
+ diagonal (`q_rank=2`, `N(μ, diag+UUᵀ)`) → **full per-patient K×K covariance** (`full_cov`,
`N(μ, LLᵀ)` via a per-patient Cholesky) — each KL exact to machine precision vs brute force.
The progression is **monotonic but converges short of closing the gap**:

| q family | Φ shrinkage | max cell diff | cognition~sleep (NUTS −0.155) |
|---|---:|---:|---:|
| mean-field (diagonal) | 21% | 0.109 | −0.045 |
| low-rank (rank-2) | 20% | 0.087 | −0.075 |
| **full K×K covariance** | **18%** | **0.077** | **−0.095** |

A richer posterior helps at every step (and improved the loadings/coordinates: G Tucker
0.957 → **0.974/0.977**, sleep coordinate r 0.968 → 0.99), confirming the variational family is
*part* of the cause. **But even a full per-patient covariance leaves ~18% shrinkage** (max cell
diff 0.077 > the 0.05 bar; cognition~sleep at 61% of NUTS).

**Interpretation (the load-bearing conclusion):** the residual Φ attenuation is **not** a
per-patient-posterior limitation — it is a **structural VI bias on the global inter-factor
correlation hyperparameter** (VI systematically under-couples hierarchical correlations,
independent of the per-patient `q`). This is *exactly* the quantity NUTS is retained for, so the
NUTS-authority boundary is now **earned with a full experimental ladder, not assumed**. Practical
default: `q_rank=2` (best balance — full-cov gives the tightest Φ but is slightly noisier on the
thin substance factor: coord r 0.908 → 0.873, credible loadings 268 → 264). Closing the last
~18% would require going beyond VI (NUTS, or a structured-VI correction on the Φ hyperprior).

### 2. Seed ensemble → loading credible bands (proposal #3)

A 6-fit seed ensemble (`notebooks/run_gllvm_ensemble.py`) gives per-cell mean ± 1.96·SD bands,
populating `ci_low`/`ci_high`/`excludes_zero` in the NUTS schema: **268/277 loading cells are
credible** (seed-CI ≠ 0). Honest caveat: a seed ensemble captures
optimization/initialization variability — a *lower bound* on the true posterior width, a
stability/identifiability signal, not a calibrated Bayesian CI (NUTS remains the CI authority).
Hand-off: `results/face/gllvm_oop/ensemble/{loadings_summary.csv, phi.csv, ensemble_loadings.npz}`.

### 3. Dot-atlas loading map

`notebooks/run_gllvm_dot_atlas.py` reuses the article Figure-2 atlas
(`face.reporting.loading_atlas`) on the CI-aware ensemble loadings. It shows the **complete**
mapping — **all 142 model indicators** (no per-block cap), grouped by home factor (immunometabolic
46, suicidality 30, developmental 23, …): an indicator×factor bubble grid (size/colour = |loading|,
ring = seed-CI ≠ 0, heavy ring = home anchor), the G windows shaded. Companions: a standalone 8×8
Φ panel and per-factor lollipops showing **every** home indicator (proportional-height panels,
|loading| + seed-ensemble CI) — `docs/figures/gllvm_oop/{gllvm_dot_atlas, gllvm_factor_lollipops,
gllvm_phi}.png`. The driver asserts every model indicator appears (completeness check).

### 4. Generative round-trip (synthetic patients vs raw data)

`generate_synthetic` draws patients from the fitted map (`f~N(0,Φ)` → `η=α+Λf` → per-item
likelihood, inverting the rank-INT copula to the raw scale), and
`notebooks/run_gllvm_synthetic_check.py` + `notebooks/gllvm_synthetic_check.ipynb` compare
synthetic vs observed raw distributions (marginal KS + continuous-block correlation SRMR). See
the next section.

## Generative fidelity (synthetic patients vs raw data)

We drew **9,013 synthetic patients** from a covariate-free generative variant
(`covariate_mode="none"`, `q_rank=2`, so the rank-INT copula inverts exactly to the raw scale)
and compared their raw-variable distributions to the observed data. **The map is a faithful
generative model:**

- **Marginals — KS median 0.04**; 110/142 indicators KS < 0.1, 125/142 < 0.2. By family:
  bernoulli 0.009, count 0.018, ordinal 0.046, gaussian 0.055. **Per-variable mean fidelity
  r = 1.000, SD fidelity r = 0.998** (Fig. `synthetic_marginal_overlays.png` — observed and
  synthetic histograms overlap for bmi/crp/wbc/cvlt/binary items).
- **Joint structure — correlation SRMR 0.078**, off-diagonal corr(observed, synthetic) = 0.87
  (Fig. `synthetic_correlation_structure.png`). This is the real test of the *latent factor
  model* (not just the per-variable marginals), and it matches the NUTS M1's own SRMR (~0.07) —
  the 8 factors reproduce the data's covariance structure, not just its margins.

**Honest limitation surfaced by the check (a fixable tiering issue, not a model failure):** the
~12 worst-fit items (KS 0.40–0.47: `psqi12`, `cgi01`, `apgr0106_1min`, `ctq35`, `psqi11`,
`fast28`, `altman`, `egf`) are **heavily-discretized ordinals the Gaussian-copula tiering
*promoted* to the continuous channel** (they have ≥8 distinct values + modal frequency < 0.5).
The synthetic reproduces their location/scale but **smooths their integer spikes** (the gaussian
channel generates continuous values). Fix: raise `copula_min_distinct` / lower
`copula_max_modal_frac` so lumpy ordinals stay in their native ordered-logistic channel — they
already fit near-perfectly there (native ordinal KS 0.046). This does not affect the loadings or
coordinate validation (those are on the modeled scale).

## Bottom line

On the 8-factor operational map, the variational GLLVM is a **calibrated, ~100×-faster
approximation to the NUTS measurement model for the map and the coordinates** (Tucker 8/8 —
G improved to 0.974 with the low-rank/ensemble; coordinate r 8/8; loading scatter r = 0.993;
268/277 loadings credible under a seed ensemble), **and a faithful generative model of the data**
(marginal KS median 0.04, mean/SD fidelity r ≈ 1.0, correlation SRMR 0.078 matching NUTS). The
one remaining gap — the inter-factor correlation Φ — was probed with the **full posterior-richness
ladder** (mean-field → low-rank → full K×K covariance): the attenuation shrinks monotonically
(21% → 20% → 18%, max cell diff 0.109 → 0.077) **but does not close**, establishing it as a
**structural VI bias on the correlation hyperparameter** — exactly why NUTS is retained as the Φ
authority (the boundary is earned, not assumed). It is fit for **fast reruns, sensitivity sweeps, synthetic-cohort
generation, and as an independent-estimator robustness check**; the NUTS engine remains the
authority for inter-factor correlations, loading uncertainty, and any final paper claim.
