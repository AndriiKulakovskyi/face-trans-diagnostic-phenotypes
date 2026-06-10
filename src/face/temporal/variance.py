"""G3 — trait vs state decomposition (docs/TEMPORAL_MODEL.md §5).

Per axis, a Bayesian measurement-error random-intercept model, the trait integrated out analytically:

    x_{i,t} ~ Normal(mu_t + u_i,  sqrt(sigma_w^2 + s^2_{i,t})),   u_i ~ Normal(0, sigma_b^2)

`mu_t` are **visit fixed effects** (any population-trajectory shape, so a trend is NOT misread as state);
the **known** M1 measurement variance `s^2_{i,t}` is PLUGGED (not estimated), so `sigma_w^2` captures only
genuine within-person *state* (the excess over measurement error) — a low-reliability axis can't look
spuriously state-like. Marginalizing the Gaussian trait `u_i` gives, per patient, a rank-1 + diagonal MVN
over that patient's visits, leaving just 5 parameters per axis (fast). The trait ratio

    ICC = sigma_b^2 / (sigma_b^2 + sigma_w^2)

is high for trait-like axes (stable between-patient differences), low for state-like axes. Single-visit
patients anchor `mu_t` and the total variance; the multi-visit patients drive the trait/state split — all
data used, no completeness selection (§5.3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_VMAP = {"V0": 0, "V1": 1, "V2": 2}
TRAIT_HI, STATE_LO = 0.6, 0.4          # ICC verdict bands (CI must clear 0.5)


def patient_patterns(panel: pd.DataFrame) -> dict:
    """Group panel ROW indices by patient visit-pattern (axis-independent → computed once).

    Returns ``{visit_tuple: row_idx [N_pat, n]}`` where each row block is one patient's visits, sorted.
    """
    d = pd.DataFrame({"uid": panel["patient_uid"].to_numpy(),
                      "t": panel["visit"].map(_VMAP).to_numpy(),
                      "row": np.arange(len(panel))}).sort_values(["uid", "t"])
    key = d.groupby("uid", sort=False)["t"].agg(tuple)
    rows = d.groupby("uid", sort=False)["row"].agg(list)
    pat = pd.DataFrame({"key": key.to_numpy(), "rows": rows.to_numpy()})
    return {k: np.array(g["rows"].tolist(), dtype="int64") for k, g in pat.groupby("key", sort=True)}


def fit_trait_state(x: np.ndarray, s: np.ndarray, patterns: dict, *, keys=None,
                    draws: int = 600, tune: int = 600, chains: int = 2, seed: int = 20260609) -> dict:
    """Fit the marginalized trait/state model for ONE axis. ``x``/``s`` are the per-row coordinate mean/SD;
    ``patterns`` from `patient_patterns`; ``keys`` restricts the visit-patterns used (e.g. only the 3-visit
    pattern → completers). Returns posterior arrays for sigma_b^2 / sigma_w^2 / ICC / mu and max R-hat."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    keys = list(keys) if keys is not None else list(patterns)
    blocks = [(np.asarray(k, "int32"), x[patterns[k]].astype("float64"),
               (s[patterns[k]].astype("float64")) ** 2) for k in keys]

    def model():
        mu = numpyro.sample("mu", dist.Normal(0.0, 1.0).expand([3]))
        sb = numpyro.sample("sigma_b", dist.HalfNormal(1.0))
        sw = numpyro.sample("sigma_w", dist.HalfNormal(1.0))
        for p, (t, X, S2) in enumerate(blocks):
            n = t.shape[0]
            loc = mu[jnp.asarray(t)]                                   # [n] visit means
            cov = sb ** 2 * jnp.ones((n, n)) + (sw ** 2 + jnp.asarray(S2))[..., None] * jnp.eye(n)
            numpyro.sample(f"y{p}", dist.MultivariateNormal(loc, covariance_matrix=cov),
                           obs=jnp.asarray(X))

    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=tune, num_samples=draws,
                num_chains=chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    post = mcmc.get_samples()
    sb2 = np.asarray(post["sigma_b"]) ** 2
    sw2 = np.asarray(post["sigma_w"]) ** 2
    rhat = max(float(np.nanmax(summary(mcmc.get_samples(group_by_chain=True), prob=0.9)[v]["r_hat"]))
               for v in ("sigma_b", "sigma_w", "mu"))
    return {"sigma_b2": sb2, "sigma_w2": sw2, "icc": sb2 / (sb2 + sw2),
            "mu": np.asarray(post["mu"]), "rhat": rhat}


def _verdict(icc_lo: float, icc_hi: float) -> str:
    if icc_lo > 0.5 and (icc_lo + icc_hi) / 2 >= TRAIT_HI:
        return "trait"
    if icc_hi < 0.5 and (icc_lo + icc_hi) / 2 <= STATE_LO:
        return "state"
    return "mixed"


def decompose(panel: pd.DataFrame, axes, patterns: dict, *, keys=None, hdi: float = 0.94,
              **fit_kw) -> pd.DataFrame:
    """Per-axis trait/state table: variance components (posterior means), ICC + HDI, verdict, and a
    signal-vs-measurement-noise ratio (flags axes with little real signal as `uninformative`)."""
    lo_q, hi_q = (1 - hdi) / 2, 1 - (1 - hdi) / 2
    rows = []
    for ax in axes:
        x = panel[f"{ax}__mean"].to_numpy().astype("float64")
        s = panel[f"{ax}__sd"].to_numpy().astype("float64")
        r = fit_trait_state(x, s, patterns, keys=keys, **fit_kw)
        icc_lo, icc_hi = float(np.quantile(r["icc"], lo_q)), float(np.quantile(r["icc"], hi_q))
        vb, vw = float(r["sigma_b2"].mean()), float(r["sigma_w2"].mean())
        v_meas = float(np.nanmean(s ** 2))
        verdict = _verdict(icc_lo, icc_hi)
        if (vb + vw) < 0.5 * v_meas:
            verdict = "uninformative"
        pop_slide = float((r["mu"][:, 2] - r["mu"][:, 0]).mean())   # population V0→V2 trend (removed before ICC)
        rows.append(dict(axis=ax, var_between=round(vb, 3), var_within=round(vw, 3),
                         var_meas=round(v_meas, 3), icc=round(float(r["icc"].mean()), 3),
                         icc_lo=round(icc_lo, 3), icc_hi=round(icc_hi, 3), pop_slide=round(pop_slide, 3),
                         signal_ratio=round((vb + vw) / v_meas, 2), rhat=round(r["rhat"], 3),
                         verdict=verdict))
    return pd.DataFrame(rows)


def raw_icc(panel: pd.DataFrame, axes, *, min_visits: int = 2) -> dict:
    """Triangulation: naive ICC from the raw coordinate means (measurement error NOT removed), on patients
    with ≥`min_visits`. Conflates measurement noise into the within term → lower than the corrected ICC;
    the gap shows how much the measurement-error correction matters."""
    out = {}
    sub = panel[panel["n_visits"] >= min_visits]
    for ax in axes:
        g = sub.groupby("patient_uid")[f"{ax}__mean"]
        pm = g.mean()                                          # patient means
        grand = pm.mean()
        vb = float(((pm - grand) ** 2).mean())                 # between-patient
        vw = float(sub.groupby("patient_uid")[f"{ax}__mean"].var(ddof=0).mean())   # within-patient
        out[ax] = round(vb / (vb + vw), 3) if (vb + vw) > 0 else float("nan")
    return out
