# V-GLLVM — Variational mixed-likelihood GLLVM atlas engine (methods)

> Methods of record for the **variational** re-estimation arm of M1. Status: implemented,
> smoke-verified, **held at a discussion gate before the production fit**. This is an
> exploration / **acceleration** arm — congruent-with and authority-defers-to the NUTS
> copula M1. It does **not** replace the certified measurement model.
> Engine: `src/face/models/variational/` · driver: `notebooks/run_gllvm_model_oop.py`.

## 1. What it is and why

The certified M1 (`measurement_model_oop`, NUTS/PyMC) is the audit-grade engine: it yields
posterior samples of loadings, factor correlations, residual variances, and patient
coordinates. It is statistically strong but slow.

The V-GLLVM keeps the **same measurement model** — one general burden axis G + specific
axes, an ontology-constrained **linear** decoder, positive home loadings, hard-zero
forbidden cells, mixed per-item likelihoods, observed-cell likelihood (no imputation), and
per-patient coordinates with uncertainty — but replaces NUTS sampling with **stochastic
variational optimization** (PyTorch). The goal is a fast operational atlas engine calibrated
against the NUTS authority, not a replacement for it.

## 2. Defaults: the 8-factor operational map

The engine defaults to the current downstream-canonical map (the estimand M2–M5 consume):

- **Ontology**: `configs/prior_loading_matrix_v3_biomerge_xc.csv` — metabolic + inflammatory
  merged into one `immunometabolic` home factor; the 3 earned cross-loadings
  (`ctq37`/`psqi11`/`psqi17` → cognition) are the only freed `plausible_cross` cells
  (`specific_cross=True`); everything else hard-zero. No horseshoe.
- **Factors** (Λ/Φ column order): `overall_severity` (G) · cognition · immunometabolic ·
  sleep · suicidality · developmental_risk · mania_activation · substance.
- **Φ structure**: G (index 0) **and** substance (index 7) pinned orthogonal; the remaining
  specifics correlate.
- **Likelihood**: Gaussian-copula (rank-INT) for the continuous + high-cardinality-promoted
  block; native Bernoulli / ordered-logistic / negative-binomial for binary + low-ordinal +
  count items — the same tiering as `MeasurementConfig.with_gaussian_copula()`.

## 3. Generative model

For patient `i`, factor vector `f_i ∈ R^K`, item `j`:

- **Prior**: `f_i ~ N(0, Φ)` with Φ bifactor-faithful (G and any pinned axis orthogonal;
  the specific block carries a normalized-Cholesky correlation — mirrors
  `BayesianBifactorESEM._build_phi`).
- **Decoder** (linear, the measurement map): `η_ij = α_j + λ_jᵀ f_i`. The loading matrix Λ
  is sparse by the ontology — home/anchor cells positive (`softplus`), plausible-cross /
  window / bifactor-G cells signed, forbidden cells **exact structural zero**.
- **Per-item likelihood** `x_ij ~ F_j(η_ij, θ_j)`: Gaussian (rank-INT z, learned residual σ
  floored at `psi_floor=0.05`), Bernoulli-logit, ordered-logistic (monotone cutpoints,
  intercept absorbed by cutpoints), negative-binomial (`μ = exp η`, learned dispersion).

**Key difference from NUTS**: the NUTS engine marginalizes the continuous block (Woodbury)
and keeps explicit latents only for discrete items. The V-GLLVM keeps a per-patient
variational latent for **every** factor and attaches every likelihood directly to `f_i` — so
there is no continuous/explicit split and no conditional-Gaussian decomposition. The item
set is the **full** ~143 indicators (both modeling blocks).

## 4. Variational posterior and objective

- **Posterior**: per-patient `q_i(f_i)` (non-amortized `nn.Embedding`); coordinates are `μ_i`
  with uncertainty `s_i`. Reparameterized draws make the latent differentiable. Default is
  **mean-field** `N(μ_i, diag(s_i²))`; `q_rank > 0` adds a **low-rank + diagonal** covariance
  `N(μ_i, diag + U_i U_iᵀ)` (matrix-determinant-lemma KL) which represents the cross-factor
  posterior covariance the diagonal cannot — it partially recovers the Φ off-diagonals (see
  findings). `s_i` (mean-field) underestimates posterior SD; the low-rank `q` improves it.
- **Negative ELBO** (minimized): `J = E_q[L_obs] + Σ_i KL[q_i ‖ N(0,Φ)] + Ω(Λ) + Ω(Φ)`.
  - `L_obs` is over **observed cells only** (the mask removes missing — no imputation).
  - The KL uses the **same full Φ** the decoder implies; the K×K Cholesky / inverse / logdet
    run on CPU/float64 (MPS lacks robust `linalg.cholesky`), per-patient terms stay on-device.
  - `Ω(Λ)` is the per-cell ontology penalty `½ Σ_free ((λ−m)/v)²` (m, v read verbatim from
    the prior matrix — home cells at 0.6, bifactor-G of thin explicit factors tightened to
    sd 0.05). `Ω(Φ)` weakly stabilizes the off-diagonal.
  - **Minibatch**: `J_batch = (N/B)(L_obs + KL) + Ω(Λ) + Ω(Φ)` — the per-patient terms scale
    to the population; the global penalties are added once (never scaled). The per-patient
    `q` embeddings sit in a **no-weight-decay** optimizer group (weight decay there would be
    a spurious second prior on the coordinates).

Loadings/Φ/σ/cutpoints are **MAP (point) estimates** in this version. Loading uncertainty is
deferred (bootstrap / seed ensemble); NUTS remains the uncertainty authority.

## 5. Staging and outputs

A 2-rung warm-start ladder — continuous backbone (`s1_backbone`: G, cognition,
immunometabolic, sleep) → full 8-factor map (`s8_full`, name-matched warm-start of shared
loadings + coordinate columns) — protects the thin factors (substance, mania) from
ELBO-flat collapse. Each stage writes, to `results/face/gllvm_oop/<stage>/`:
`coordinates.parquet` (the `{factor}__mean/sd/hdi_low/hdi_high/n_obs/reliability` schema),
`loadings_summary.csv` (the 11-column NUTS schema; `ci_*`/`excludes_zero` NaN — MAP),
`phi.csv`, `training_history.csv`, `model_state.pt`, `manifest.json` (cache key =
`MODEL_VERSION + stage_spec + config_sig`). The `consolidate/` hand-off carries the canonical
`s8_full` exports.

## 6. Validation against NUTS (`validate.py`)

The acceptance gate is **congruence**, not certification:

- **Tucker congruence** of loading columns per factor (≥ 0.95 major axes; ≥ 0.85 thin),
  sign/column aligned.
- **Coordinate Pearson r** per factor over well-observed patients (≥ 0.90 well-anchored).
- **Φ agreement** — immunometabolic-block correlation cells + the trivially-orthogonal
  G/substance rows; per-cell |ΔΦ| ≤ 0.05.
- **Posterior predictive checks** over observed cells (continuous mean/var, binary rate,
  ordinal frequency, count zero-rate).

Targets are exported on the fly from the cached 8-factor copula idata
(`results/face/oop_measurement/copula/weighted_8d/hs_s5_merged_xc/idata.nc`) via the
certified `export_loadings_summary` / `export_phi`.

## 6b. Generative use (synthetic patients)

The GLLVM is a proper generative model: `face.models.variational.generative.generate_synthetic`
draws `f ~ N(0, Φ)` → `η = α + Λ f` → per-item likelihood samples, inverting the rank-INT copula
back to the **raw clinical scale** (stored in `GLLVMData.copula`; exact under
`covariate_mode="none"`).  `run_gllvm_synthetic_check.py` + `gllvm_synthetic_check.ipynb` compare
synthetic vs observed raw distributions (marginal KS + continuous-block correlation SRMR) — a
direct generative-fidelity test.  A seed ensemble (`run_gllvm_ensemble.py`) gives loading credible
bands; the dot-atlas (`run_gllvm_dot_atlas.py`) renders the CI-aware loading map.

## 7. Honest limits

- Mean-field diagonal `q` against a correlated Φ **underestimates** posterior SD — `s_i` is a
  *relative* reliability signal (more observed home indicators → smaller `s_i`), calibrated
  to NUTS, not a Bayesian credible width. Coordinate **means** are far more robust than SDs.
- Global parameters are MAP — loading CIs come from NUTS / bootstrap, not VI here.
- Linear decoder by design (the loading matrix is the measurement map; a neural decoder would
  weaken clinical interpretability).
- New-patient scoring currently requires optimization (no amortized encoder yet — a v0.3
  roadmap item).

## 8. How to run

```bash
pip install -e ".[variational]"        # adds torch>=2.2 (kept out of core deps)
pytest tests/golden/test_gllvm_*.py    # CPU, deterministic
# fast wiring check (tiny epochs, balanced subsample, single rung):
HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_gllvm_model_oop.py --mode smoke
# production fit (detached; after the gate):
python3 scripts/run_job.py gllvm -- python notebooks/run_gllvm_model_oop.py \
    --mode production --consolidate
# congruence vs NUTS (after the production fit):
python notebooks/run_gllvm_validation.py
```

(The JAX `XLA_FLAGS` gotcha does not apply — this is torch, not numpyro.)
