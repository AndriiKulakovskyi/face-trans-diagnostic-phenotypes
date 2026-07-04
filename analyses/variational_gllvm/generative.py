"""Generate synthetic patients from a fitted variational GLLVM, and compare their raw-variable
distributions to the observed data — a direct test of the generative model's fidelity.

Generative recipe (the GLLVM is a proper generative model):

    f ~ N(0, Phi)                      # latent patient coordinate
    eta = alpha + Lambda f             # ontology-constrained linear predictor
    x_j ~ F_j(eta_j, theta_j)          # per-item likelihood (gaussian / bernoulli / ordinal / count)

For continuous (Gaussian-copula) items the draw is on the rank-INT z scale and is inverted back
to the **raw clinical scale** through the stored empirical map ``GLLVMData.copula`` (the inverse
rank-INT, ``y = F_j^{-1}(Phi(z))``).  Discrete draws are un-oriented (and ordinal codes mapped
back to their original category values).  The inversion is exact only when the rank-INT was not
covariate-residualized, so generate from a ``covariate_mode="none"`` fit for a clean raw-scale
comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_synthetic(model, data, n: int, *, seed: int = 0) -> pd.DataFrame:
    """Draw ``n`` synthetic patients on the **raw clinical scale**.

    ``model`` is a fitted :class:`VariationalGLLVM`; ``data`` the :class:`GLLVMData` it was fit
    on (carries the ontology, families, signs, ordinal category maps, and copula inversion map).
    Returns a DataFrame ``n x len(items)`` with the indicator columns on their native scale.
    """
    rng = np.random.default_rng(seed)
    K = model.K
    phi = model.phi_matrix().astype("float64")
    chol = np.linalg.cholesky(phi + 1e-8 * np.eye(K))
    f = rng.standard_normal((n, K)) @ chol.T  # f ~ N(0, Phi)

    lam = model.loadings().astype("float64")  # (J, K)
    alpha = model.alpha.detach().cpu().numpy().astype("float64")  # (J,)
    sigma = model.sigma().detach().cpu().numpy().astype("float64")
    count_alpha = model.count_alpha().detach().cpu().numpy().astype("float64")
    eta = alpha[None, :] + f @ lam.T  # (n, J)

    out: dict[str, np.ndarray] = {}
    for j, item in enumerate(data.items):
        fam = data.families[j]
        sign = int(data.item_signs.get(item, 1))
        eta_j = eta[:, j]
        if fam == "gaussian":
            z = eta_j + sigma[j] * rng.standard_normal(n)
            cop = data.copula.get(item)
            if cop is not None:
                sorted_oriented, sorted_z = cop
                y_oriented = np.interp(z, sorted_z, sorted_oriented)  # F^-1(Phi(z))
                out[item] = sign * y_oriented  # un-orient to the raw scale
            else:
                out[item] = z
        elif fam == "bernoulli":
            x = (rng.random(n) < _sigmoid(eta_j)).astype("float64")  # oriented (1 = more burden)
            out[item] = 1.0 - x if sign < 0 else x
        elif fam == "ordinal":
            out[item] = _sample_ordinal(model, j, eta_j, data, sign, rng)
        elif fam == "count":
            mu = np.exp(np.clip(eta_j, -10, 10))
            a = count_alpha[j]
            p = a / (a + mu)  # NB mean = a(1-p)/p = mu
            out[item] = rng.negative_binomial(a, np.clip(p, 1e-6, 1 - 1e-6)).astype("float64")
    return pd.DataFrame(out, columns=list(data.items))


def _sample_ordinal(model, j: int, eta_j: np.ndarray, data, sign: int, rng) -> np.ndarray:
    cuts = model.ordered_cutpoints(j).detach().cpu().numpy().astype("float64")
    C = cuts.size + 1
    cdf = _sigmoid(cuts[None, :] - eta_j[:, None])  # P(y<=c), (n, C-1)
    probs = np.concatenate(
        [cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1.0 - cdf[:, -1:]], axis=1
    )  # (n, C)
    probs = np.clip(probs, 1e-9, None)
    probs /= probs.sum(1, keepdims=True)
    u = rng.random((eta_j.size, 1))
    cat = (np.cumsum(probs, axis=1) < u).sum(axis=1)  # oriented code 0..C-1
    cat = np.clip(cat, 0, C - 1)
    if sign < 0:
        cat = (C - 1) - cat
    cat_values = data.ord_category_maps.get(j)
    if cat_values is not None and len(cat_values) == C:
        lut = np.asarray(cat_values, dtype="float64")
        return lut[cat]
    return cat.astype("float64")


# --------------------------------------------------------------------------- comparison
def marginal_summary(observed: pd.DataFrame, synthetic: pd.DataFrame,
                     families: dict[str, str]) -> pd.DataFrame:
    """Per-variable observed-vs-synthetic marginal summary + KS distance (observed cells only)."""
    from scipy.stats import ks_2samp  # type: ignore[reportMissingImports]

    rows = []
    for item in synthetic.columns:
        o = pd.to_numeric(observed[item], errors="coerce").to_numpy("float64")
        o = o[np.isfinite(o)]
        s = synthetic[item].to_numpy("float64")
        s = s[np.isfinite(s)]
        if o.size == 0 or s.size == 0:
            continue
        ks = float(ks_2samp(o, s).statistic)
        rows.append({
            "item": item, "family": families.get(item, ""), "n_obs": int(o.size),
            "obs_mean": float(o.mean()), "syn_mean": float(s.mean()),
            "obs_sd": float(o.std()), "syn_sd": float(s.std()),
            "obs_median": float(np.median(o)), "syn_median": float(np.median(s)),
            "obs_zero_rate": float((o == 0).mean()), "syn_zero_rate": float((s == 0).mean()),
            "ks": ks,
        })
    return pd.DataFrame(rows).sort_values("ks", ascending=False).reset_index(drop=True)


def correlation_block(observed: pd.DataFrame, synthetic: pd.DataFrame,
                      items: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Observed vs synthetic pairwise (Spearman) correlation matrices over ``items`` + the SRMR
    (root-mean-square off-diagonal difference) — the test of the joint/factor structure."""
    cols = [c for c in items if c in observed.columns and c in synthetic.columns]
    co = observed[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    cs = synthetic[cols].corr(method="spearman")
    co = co.reindex(index=cols, columns=cols)
    cs = cs.reindex(index=cols, columns=cols)
    diff = (co.to_numpy() - cs.to_numpy())
    off = ~np.eye(len(cols), dtype=bool)
    valid = off & np.isfinite(diff)
    srmr = float(np.sqrt(np.nanmean(diff[valid] ** 2))) if valid.any() else float("nan")
    return co, cs, srmr
