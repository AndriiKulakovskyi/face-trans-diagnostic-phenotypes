"""The M4 outcome GLM — a small Bayesian regression with optional errors-in-variables predictors.

One engine for the whole nested ladder (reference rungs R0..R3y and the test models). The design is
deliberately the M3 `variance.py` idiom: a bespoke NumPyro model, NUTS, the **known per-patient
measurement SD plugged** for any error-prone predictor (the EIV route), and per-observation
log-likelihood emitted so `compare.py` can rank models by held-out ELPD (LOO).

Linear predictor:  eta_i = alpha + X_i·beta (+ u_site[i]) (+ xi_i·beta_eiv)
  - X        : error-free design (z-scored continuous + dummies), no intercept column
  - u_site   : optional site random intercept (partial-pooled nuisance, present from R0)
  - xi       : optional LATENT true values of error-prone predictors, with z_obs ~ Normal(xi, sd) and
               the known sd PLUGGED — so a wide-posterior coordinate self-down-weights and beta_eiv is
               attenuation-corrected (the Bayesian-ESEM uncertainty paying off). Used from stage 43.

Likelihoods: gaussian (identity), bernoulli (logit), ordinal (cumulative logit / OrderedLogistic).
`weights` scales the per-obs likelihood (IPW; the LOO of a weighted fit is non-standard, so the
headline ladder is fit unweighted and IPW is a stage-46 sensitivity). Predictors are assumed already
standardized by `reference.py`, matching the project's z-scored-predictor / Normal(0,1)-prior idiom.
"""
from __future__ import annotations

import numpy as np

_FAMILIES = {"gaussian", "bernoulli", "ordinal"}


def fit_glm(y, X, *, family: str = "gaussian", group=None, n_groups: int | None = None,
            eiv_obs=None, eiv_sd=None, eiv_interact=None, weights=None, n_cat: int | None = None,
            draws: int = 1000, tune: int = 1000, chains: int = 4, seed: int = 20260610,
            target_accept: float = 0.9) -> dict:
    """Fit the outcome GLM. `X` is [N, P] (no intercept column); `y` is [N] (z-scored for gaussian,
    0/1 for bernoulli, 0..K-1 ints for ordinal). `group`/`n_groups` add a site random intercept;
    `eiv_obs`/`eiv_sd` ([N, K]) add K errors-in-variables predictors. Returns the arviz InferenceData
    (with per-obs log-likelihood of the outcome site `y`), a coefficient summary, and convergence."""
    import arviz as az
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    if family not in _FAMILIES:
        raise ValueError(f"family {family!r} not in {sorted(_FAMILIES)}")
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y)
    N, P = X.shape
    has_eiv = eiv_obs is not None
    if has_eiv:
        eiv_obs = np.asarray(eiv_obs, dtype="float64").reshape(N, -1)
        eiv_sd = np.asarray(eiv_sd, dtype="float64").reshape(N, -1)
        K = eiv_obs.shape[1]
    w = None if weights is None else np.asarray(weights, dtype="float64")
    ix = None if eiv_interact is None else np.asarray(eiv_interact, dtype="float64").reshape(N)  # treat×axis moderator

    def model():
        alpha = numpyro.sample("alpha", dist.Normal(0.0, 2.0))
        eta = alpha + jnp.zeros(N)
        if P:
            beta = numpyro.sample("beta", dist.Normal(0.0, 1.0).expand([P]))
            eta = eta + jnp.asarray(X) @ beta
        if group is not None:
            tau_g = numpyro.sample("tau_site", dist.HalfNormal(1.0))
            u = numpyro.sample("u_site", dist.Normal(0.0, 1.0).expand([n_groups]))
            eta = eta + tau_g * u[jnp.asarray(group)]                  # non-centered site intercept
        if has_eiv:
            mu_x = numpyro.sample("mu_x", dist.Normal(0.0, 1.0).expand([K]))
            tau_x = numpyro.sample("tau_x", dist.HalfNormal(1.0).expand([K]))
            xi_raw = numpyro.sample("xi_raw", dist.Normal(0.0, 1.0).expand([N, K]).to_event(1))
            xi = mu_x + tau_x * xi_raw                     # non-centered — avoids the EIV funnel (high-K)
            numpyro.sample("z", dist.Normal(xi, jnp.asarray(eiv_sd)).to_event(1), obs=jnp.asarray(eiv_obs))
            b_eiv = numpyro.sample("beta_eiv", dist.Normal(0.0, 1.0).expand([K]))
            eta = eta + xi @ b_eiv
            if ix is not None:                                  # moderation: treat × latent-axis
                b_int = numpyro.sample("beta_eiv_int", dist.Normal(0.0, 1.0).expand([K]))
                eta = eta + (jnp.asarray(ix)[:, None] * xi) @ b_int

        if family == "gaussian":
            sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
            obs = dist.Normal(eta, sigma)
        elif family == "bernoulli":
            obs = dist.Bernoulli(logits=eta)
        else:  # ordinal — cumulative logit
            c = numpyro.sample("cutpoints",
                               dist.TransformedDistribution(dist.Normal(0.0, 2.0).expand([n_cat - 1]),
                                                            dist.transforms.OrderedTransform()))
            obs = dist.OrderedLogistic(eta, c)
        with numpyro.plate("obs", N):
            if w is not None:
                with numpyro.handlers.scale(scale=jnp.asarray(w)):
                    numpyro.sample("y", obs, obs=jnp.asarray(y))
            else:
                numpyro.sample("y", obs, obs=jnp.asarray(y))

    mcmc = MCMC(NUTS(model, target_accept_prob=target_accept), num_warmup=tune, num_samples=draws,
                num_chains=chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    idata = _build_idata(model, mcmc, site="y")           # az.from_dict — avoids from_numpyro re-trace leak
    coef = _coef_table(idata, P)
    diag = _diag(idata)
    return {"idata": idata, "coef": coef, "family": family, "n": int(N), "n_pred": int(P),
            "rhat": diag["rhat"], "divergences": diag["div"], "ess": diag["ess"]}


def _build_idata(model, mcmc, *, site: str):
    """Assemble InferenceData from raw samples via `az.from_dict`. We do NOT use `az.from_numpyro`:
    arviz>=1.1 re-traces the model to infer predictive dims, which leaks a JAX tracer when the
    likelihood is wrapped in a `scale` (IPW) handler. Computing the posterior, per-obs log-likelihood
    and divergences from the samples sidesteps the re-trace entirely."""
    import arviz as az
    import numpyro

    post = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}  # [C, D, ...]
    c, d = next(iter(post.values())).shape[:2]
    ll = np.asarray(numpyro.infer.log_likelihood(model, mcmc.get_samples())[site])        # [C*D, N]
    ll = ll.reshape(c, d, ll.shape[-1])
    data = {"posterior": post, "log_likelihood": {site: ll}}                              # arviz>=1.1 nested-dict
    extra = mcmc.get_extra_fields(group_by_chain=True)
    if "diverging" in extra:
        data["sample_stats"] = {"diverging": np.asarray(extra["diverging"])}
    return az.from_dict(data)


def _summarize(arr) -> dict:
    a = np.asarray(arr).ravel()
    return {"mean": float(a.mean()), "sd": float(a.std()),
            "eti_lo": float(np.quantile(a, 0.03)), "eti_hi": float(np.quantile(a, 0.97)),
            "p_direction": float((a > 0).mean())}


def _coef_table(idata, P):
    """Tidy fixed-effect posterior summary (mean, sd, 94% ETI, P(>0)) for alpha + beta + beta_eiv.
    Computed from the posterior directly (version-proof vs az.summary column renames)."""
    import pandas as pd

    post = idata.posterior
    rows = []
    if "alpha" in post:
        rows.append({"term": "alpha", **_summarize(post["alpha"].values)})
    for v in ("beta", "beta_eiv", "beta_eiv_int"):
        if v in post:
            arr = post[v].values
            arr = arr.reshape(-1, arr.shape[-1])
            for i in range(arr.shape[1]):
                rows.append({"term": f"{v}[{i}]", **_summarize(arr[:, i])})
    return pd.DataFrame(rows)


def _diag(idata) -> dict:
    """Max R-hat / min ESS over the structural params, and divergence count."""
    import arviz as az
    import pandas as pd

    have = [v for v in ["alpha", "beta", "beta_eiv", "sigma", "tau_site", "cutpoints"]
            if v in idata.posterior]
    summ = az.summary(idata, var_names=have)
    rc = "r_hat" if "r_hat" in summ.columns else "rhat"
    ec = next((c for c in summ.columns if c.startswith("ess")), None)
    rhat = float(pd.to_numeric(summ[rc], errors="coerce").max())
    ess = float(pd.to_numeric(summ[ec], errors="coerce").min()) if ec else float("nan")
    try:
        div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    except (KeyError, AttributeError):
        div = 0
    return {"rhat": rhat, "ess": ess, "div": div}
