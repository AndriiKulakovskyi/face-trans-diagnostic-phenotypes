"""Export a fitted Gaussian-copula measurement model to portable parameters, and generate synthetic
patients from it (transformed-Gaussian generative model), for faithful-reproduction checks.

The copula measurement model is invertible, so it doubles as a generator:

    eta_i ~ N(0, Phi)                                  latent factor scores
    z_ij  = Lambda_j . eta_i + eps_ij,  eps ~ N(0, sigma_j^2)   latent Gaussian indicators
    continuous item:  y_ij = F_j^{-1}(Phi_std(z_ij))   invert the empirical-CDF (copula) map
    explicit item:    draw from the fitted GLM on the explicit factor coordinates f_e = eta[:, e_cols]
                      (Bernoulli / NegBinomial / ordered-logistic)

This preserves (a) each continuous indicator's marginal exactly (copula), (b) the dependence structure
(Lambda Phi Lambda' + Psi), and (c) cross-block dependence (continuous and explicit items share eta).

    from face.models.bayesian.synthetic import export_fitted_model, save_fitted_model, generate_synthetic
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .measurement_model_oop import (
    DEFAULT_EXPLICIT_FACTORS,
    G_KEY,
    MeasurementDataset,
    copula_invert,
)

N_COPULA_KNOTS = 2000  # quantile knots stored per continuous item for the inverse map (dense tails -> faithful heavy-tailed marginals)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _native_invert(z_std: np.ndarray, family: str, sign: int, log_min, mu: float, sd: float) -> np.ndarray:
    """Invert the NATIVE (pre-copula) encoding: standardized z -> original clinical scale.

    Native encoding was  z = (sign * transform(y) - mu) / sd  with transform = log for lognormal labs,
    identity otherwise.  So  transform(y) = sign * (z*sd + mu);  y = inv_transform(...).  This is the
    native model's IMPLIED marginal -- a (log-)normal -- which is exactly the assumption the copula
    replaced; heavy-tailed / zero-inflated items therefore come out mis-specified here by construction.
    """
    oriented = z_std * sd + mu
    values_log = oriented * int(sign)               # sign in {-1, +1}, so /sign == *sign
    if family == "lognormal":
        values_log = np.clip(values_log, -50.0, 50.0)
        if log_min is not None and log_min <= 0:
            return np.expm1(values_log) + float(log_min) - 1e-6   # inverse of log1p(y - log_min + 1e-6)
        return np.exp(values_log)
    return values_log


@dataclass
class FittedMeasurementModel:
    """Portable posterior-mean parameters of a fitted copula measurement model."""
    factor_cols: list[str]
    items: list[str]                       # continuous (copula) block items, in Lam row order
    home: list[str]
    signs: dict[str, int]
    Lam: np.ndarray                        # (J, F) continuous loadings
    Phi: np.ndarray                        # (F, F) factor correlations
    sigma: np.ndarray                      # (J,) continuous residual SDs (incl. psi_floor)
    copula: dict[str, tuple[np.ndarray, np.ndarray]]  # item -> (sorted oriented values, sorted z) knots
    e_cols: list[int]                      # explicit-factor columns in factor_cols
    explicit: dict[str, dict]              # item -> {family, home_e, a, lh, lg, nb_alpha?, cutpoints?}
    meta: dict                             # provenance (likelihood_mode, cohort_weighted, n_fit, ...)
    mode: str = "gaussian_copula"          # continuous-block inversion: "gaussian_copula" | "native"
    native: dict[str, tuple] = field(default_factory=dict)  # native mode: item -> (family, sign, log_min, mu, sd)


def _pm(idata, name):
    return np.asarray(idata.posterior[name].mean(("chain", "draw")).values)


def _draws(idata, name) -> np.ndarray:
    """Posterior draws of ``name`` flattened over (chain, draw): shape ``(n_draws, *param_dims)``.

    Unlike ``_pm`` (posterior mean) this keeps the full draw axis so equal-tailed
    credible intervals can be read off with ``np.quantile``.  Used by the
    CI-aware loading export; ``_pm`` stays the mean-only path for Phi and the
    portable ``FittedMeasurementModel``.
    """
    da = idata.posterior[name]
    return da.stack(__s=("chain", "draw")).transpose("__s", ...).values


def export_fitted_model(idata, mixed, config, *, meta: dict | None = None) -> FittedMeasurementModel:
    """Extract posterior-mean parameters from a fitted (copula) mixed model into a portable object."""
    base = mixed.base
    mode = "native" if config.likelihood_mode == "native" else "gaussian_copula"
    Lam = _pm(idata, "Lam")
    Phi = _pm(idata, "Phi")
    sigma = float(config.psi_floor) + _pm(idata, "sigma")
    # continuous-block inversion map: empirical-CDF knots (copula) OR parametric moments (native)
    copula: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    native: dict[str, tuple] = {}
    if mode == "gaussian_copula" and base.copula:
        q = np.linspace(0.0, 1.0, N_COPULA_KNOTS)
        for it, (vals, zs) in base.copula.items():
            idx = np.clip((q * (len(vals) - 1)).round().astype(int), 0, len(vals) - 1)
            copula[it] = (np.asarray(vals)[idx], np.asarray(zs)[idx])
    elif mode == "native":
        for it in base.items:
            family, sign, log_min, mu, sd = base.moments[it]   # (family, sign, log_min, mu, sd)
            native[it] = (str(family), int(sign), (None if log_min is None else float(log_min)),
                          float(mu), float(sd))
    # explicit-item GLM params
    post = idata.posterior
    explicit: dict[str, dict] = {}
    for it in mixed.bin_items:
        explicit[it] = {"family": "bernoulli", "home_e": int(mixed.ng_home[it]),
                        "a": float(_pm(idata, f"a_{it}")), "lh": float(_pm(idata, f"lh_{it}")),
                        "lg": float(_pm(idata, f"lg_{it}"))}
    for it in mixed.cnt_items:
        explicit[it] = {"family": "neg_binomial", "home_e": int(mixed.ng_home[it]),
                        "a": float(_pm(idata, f"a_{it}")), "lh": float(_pm(idata, f"lh_{it}")),
                        "lg": float(_pm(idata, f"lg_{it}")),
                        "nb_alpha": float(_pm(idata, f"alpha_{it}"))}
    for k, it in enumerate(mixed.ord_items):
        explicit[it] = {"family": "ordered_logistic", "home_e": int(mixed.ng_home[it]),
                        "lh": float(_pm(idata, f"lh_{it}")), "lg": float(_pm(idata, f"lg_{it}")),
                        "cutpoints": _pm(idata, f"c_{it}").tolist(), "ord_K": int(mixed.ord_K[k])}
    return FittedMeasurementModel(
        factor_cols=list(base.factor_cols), items=list(base.items), home=list(base.home),
        signs=dict(base.signs), Lam=Lam, Phi=Phi, sigma=sigma, copula=copula,
        e_cols=list(mixed.e_cols), explicit=explicit, mode=mode, native=native,
        meta=(meta or {}) | {"likelihood_mode": config.likelihood_mode,
                             "cohort_weighted": bool(config.cohort_weighted), "J": int(Lam.shape[0]),
                             "F": int(Lam.shape[1]), "n_explicit": len(explicit)},
    )


def export_loadings_summary(idata, mixed, config, *, hdi_prob: float = 0.95) -> pd.DataFrame:
    """Tidy long table of posterior factor loadings WITH equal-tailed credible intervals.

    One row per *meaningful* (item, factor) cell, combining the two places the copula mixed
    model stores loadings:

    * the **continuous block** ``Lam`` (J x F): home loadings (``primary`` / ``g_anchor``),
      specific-item bifactor-G loadings (``bifactor_G``), and depression/anxiety ``window``
      loadings on G.  Hard-zero ``unlikely`` cells are never sampled (Lam draws are exactly 0)
      and are dropped — only interpretable cells are emitted.
    * the **explicit block** — for each Bernoulli / negative-binomial / ordered-logistic item,
      two scalar loadings: ``lh_<item>`` on its home explicit factor and ``lg_<item>`` on G.

    The home factor of an explicit item is ``factor_cols[e_cols[ng_home[item]]]`` — ``ng_home``
    indexes INTO ``e_cols``, not into ``factor_cols`` directly.

    Columns: ``item, factor, home, block, likelihood_family, kind, loading`` (signed posterior
    median), ``abs_loading, ci_low, ci_high, excludes_zero``.  ``hdi_prob`` parameterizes the
    equal-tailed interval (default 95% -> 2.5 / 97.5 quantiles); column names stay ``ci_*`` to
    signal equal-tailed (not HDI).
    """
    base = mixed.base
    factor_cols = list(base.factor_cols)
    items = list(base.items)
    home = list(base.home)
    lo_q, hi_q = (1.0 - hdi_prob) / 2.0, 1.0 - (1.0 - hdi_prob) / 2.0

    ds = MeasurementDataset(config)
    spec = ds.loading_spec(
        base,
        windows=True,
        bifactor_g_sd={f: 0.05 for f in DEFAULT_EXPLICIT_FACTORS if f != G_KEY},
    )
    kind = spec.kind  # {(j, c): kind_str}
    meta = ds.meta

    def _family(item: str, fallback: str = "") -> str:
        return str(meta.loc[item, "likelihood_family"]) if item in meta.index else fallback

    rows: list[dict] = []

    # --- continuous block: Lam (n_draws, J, F) ---
    Lam = _draws(idata, "Lam")
    q = np.quantile(Lam, [lo_q, 0.5, hi_q], axis=0)  # (3, J, F)
    for j, item in enumerate(items):
        h = home[j]
        fam = _family(item)
        for c, factor in enumerate(factor_cols):
            k = kind.get((j, c))
            if k is None and factor != h and factor != G_KEY:
                continue  # hard-zero unlikely cell -> not interpretable
            lo, med, hi = float(q[0, j, c]), float(q[1, j, c]), float(q[2, j, c])
            if k is None and abs(med) < 1e-9 and abs(lo) < 1e-9 and abs(hi) < 1e-9:
                continue  # G/home safety-net cell that is in fact a structural zero
            kk = k or ("primary" if factor == h else "bifactor_G" if factor == G_KEY else "cross")
            rows.append(
                dict(item=item, factor=factor, home=h or "", block="continuous",
                     likelihood_family=fam, kind=kk, loading=med, abs_loading=abs(med),
                     ci_low=lo, ci_high=hi, excludes_zero=bool(lo > 0.0 or hi < 0.0))
            )

    # --- explicit block: lh_<item> (home) + lg_<item> (G) ---
    fam_by = {**{it: "bernoulli" for it in mixed.bin_items},
              **{it: "neg_binomial" for it in mixed.cnt_items},
              **{it: "ordered_logistic" for it in mixed.ord_items}}
    for item in list(mixed.bin_items) + list(mixed.cnt_items) + list(mixed.ord_items):
        home_factor = factor_cols[mixed.e_cols[mixed.ng_home[item]]]
        for var, factor, kk in ((f"lh_{item}", home_factor, "primary"),
                                (f"lg_{item}", G_KEY, "bifactor_G")):
            d = _draws(idata, var)
            lo, med, hi = (float(v) for v in np.quantile(d, [lo_q, 0.5, hi_q]))
            rows.append(
                dict(item=item, factor=factor, home=home_factor, block="explicit",
                     likelihood_family=fam_by[item], kind=kk, loading=med, abs_loading=abs(med),
                     ci_low=lo, ci_high=hi, excludes_zero=bool(lo > 0.0 or hi < 0.0))
            )

    df = pd.DataFrame(rows)
    fidx = {f: i for i, f in enumerate(factor_cols)}
    df["__f"] = df["factor"].map(fidx)
    df = df.sort_values(["item", "__f"]).drop(columns="__f").reset_index(drop=True)
    return df


def export_phi(idata, factor_cols: list[str]) -> pd.DataFrame:
    """Posterior-mean 9x9 inter-factor correlation matrix Phi, factor-labeled index/columns."""
    Phi = _pm(idata, "Phi")
    return pd.DataFrame(Phi, index=list(factor_cols), columns=list(factor_cols))


def save_fitted_model(model: FittedMeasurementModel, path: str | Path) -> Path:
    """Persist as a directory: arrays.npz (Lam/Phi/sigma + copula knots + cutpoints) + model.json."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {"Lam": model.Lam, "Phi": model.Phi, "sigma": model.sigma}
    for it, (v, z) in model.copula.items():
        arrays[f"cop_v__{it}"] = v
        arrays[f"cop_z__{it}"] = z
    np.savez_compressed(path / "arrays.npz", **arrays)
    (path / "model.json").write_text(json.dumps({
        "factor_cols": model.factor_cols, "items": model.items, "home": model.home,
        "signs": model.signs, "e_cols": model.e_cols, "explicit": model.explicit,
        "copula_items": list(model.copula), "mode": model.mode,
        "native": {it: list(v) for it, v in model.native.items()}, "meta": model.meta}, indent=2))
    return path


def load_fitted_model(path: str | Path) -> FittedMeasurementModel:
    path = Path(path)
    arr = np.load(path / "arrays.npz")
    d = json.loads((path / "model.json").read_text())
    copula = {it: (arr[f"cop_v__{it}"], arr[f"cop_z__{it}"]) for it in d["copula_items"]}
    native = {it: tuple(v) for it, v in d.get("native", {}).items()}
    return FittedMeasurementModel(
        factor_cols=d["factor_cols"], items=d["items"], home=d["home"],
        signs={k: int(v) for k, v in d["signs"].items()}, Lam=arr["Lam"], Phi=arr["Phi"],
        sigma=arr["sigma"], copula=copula, e_cols=d["e_cols"], explicit=d["explicit"],
        mode=d.get("mode", "gaussian_copula"), native=native, meta=d["meta"])


def generate_synthetic(model: FittedMeasurementModel, n: int, *, seed: int = 0) -> pd.DataFrame:
    """Generate ``n`` synthetic patients on the ORIGINAL indicator scales.  Continuous items are inverted
    through the copula map (``mode='gaussian_copula'``) or the parametric (log-)normal native encoding
    (``mode='native'``); explicit items are drawn from their fitted GLMs.  Columns = continuous + explicit
    items; values are NOT z-scored (raw clinical scale)."""
    rng = np.random.default_rng(seed)
    F = len(model.factor_cols)
    L = np.linalg.cholesky(model.Phi + 1e-10 * np.eye(F))
    eta = rng.standard_normal((n, F)) @ L.T                     # eta ~ N(0, Phi)
    out: dict[str, np.ndarray] = {}
    # --- continuous (copula) block ---
    J = model.Lam.shape[0]
    z = eta @ model.Lam.T + rng.standard_normal((n, J)) * model.sigma[None, :]   # z ~ N(0, LamPhiLam'+Psi)
    # The stored copula map uses the DATA's standard-normal rank-INT scale; the model-implied per-item
    # variance diag(Lam Phi Lam' + Psi) is ~1 but not exactly, so standardize z per item before inverting
    # (u = Phi(z / implied_sd)) -- otherwise heavy-tailed marginals come out mis-scaled.
    implied_sd = np.sqrt(np.clip(np.einsum("jf,fg,jg->j", model.Lam, model.Phi, model.Lam)
                                 + model.sigma ** 2, 1e-9, None))
    for j, it in enumerate(model.items):
        z_std = z[:, j] / implied_sd[j]                           # standardize to the encoded N(0,1) scale
        if model.mode == "gaussian_copula" and it in model.copula:
            out[it] = copula_invert(z_std, *model.copula[it]) / model.signs.get(it, 1)  # F^-1(Phi(z_std))
        elif it in model.native:
            out[it] = _native_invert(z_std, *model.native[it])    # parametric (log-)normal inverse
    # --- explicit block (drawn from fitted GLMs on the explicit factor coords) ---
    f_e = eta[:, model.e_cols]                                   # (n, Ke); column 0 is G
    g = f_e[:, 0]
    for it, p in model.explicit.items():
        fh = f_e[:, p["home_e"]]
        if p["family"] == "bernoulli":
            prob = _sigmoid(p["a"] + p["lh"] * fh + p["lg"] * g)
            out[it] = rng.binomial(1, prob).astype(float)
        elif p["family"] == "neg_binomial":
            mu = np.exp(np.clip(p["a"] + p["lh"] * fh + p["lg"] * g, -20, 20))
            a = max(p["nb_alpha"], 1e-3)
            out[it] = rng.negative_binomial(a, a / (a + mu)).astype(float)
        elif p["family"] == "ordered_logistic":
            eta_lin = p["lh"] * fh + p["lg"] * g
            cut = np.asarray(p["cutpoints"])
            # PyMC OrderedLogistic: P(y > k) = sigmoid(eta - c_k)
            surv = _sigmoid(eta_lin[:, None] - cut[None, :])     # (n, K-1) = P(y>k)
            p_ge = np.concatenate([np.ones((n, 1)), surv], axis=1)        # P(y>=k), k=0..K-1
            p_le_prev = np.concatenate([surv, np.zeros((n, 1))], axis=1)  # P(y>k), k=0..K-1
            probs = np.clip(p_ge - p_le_prev, 1e-9, None)
            probs /= probs.sum(1, keepdims=True)
            u = rng.random(n)
            out[it] = (u[:, None] > np.cumsum(probs, axis=1)).sum(1).astype(float)
    return pd.DataFrame(out)
