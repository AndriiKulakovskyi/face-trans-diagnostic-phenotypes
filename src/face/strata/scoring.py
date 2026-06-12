"""M2.0 — full-N projection of the explicit (non-Gaussian) factors + uncertainty export.

The certified 9-dim mixed fit (``results/face/s5_cert9_s1``) instantiated explicit latents
``f_e = (overall_severity, suicidality, developmental_risk, substance)`` only for its ~1,884-patient
fit subsample. To stratify on all nine dimensions at full N (M2), we **project** ``f_e`` onto every
patient: hold the certified measurement parameters FIXED (continuous loadings Λ, factor correlation
Φ, residual σ, and each non-Gaussian item's intercept / home-loading / G-loading / cutpoints /
dispersion) and sample only each patient's ``f_e`` from their OBSERVED cells (continuous + binary /
ordinal / count) — a per-patient conditional posterior, **not** a re-fit (§6 M2.0).

This reuses the exact S3b/S5 mixed likelihood (continuous block marginalized via the Woodbury kernel;
non-Gaussian items on ``f_e``), so the projection is consistent with how M1 estimated ``f_e``. The
no-imputation invariant carries over: an unobserved cell contributes no term, so a patient is placed
only by what they actually have (e.g. DR patients have no substance items → their substance score is
prior-dominated, not imputed).

The continuous-anchored axes (overall_severity / cognition / metabolic / inflammatory / sleep /
mania) are scored full-N analytically by ``face.scoring.conditional_gaussian_scores`` already; this
module adds the explicit block and a draw-wise uncertainty export.
"""
from __future__ import annotations

import numpy as np

from face.models.bayesian.continuous_core import _patterns, _woodbury_potential


def fixed_params(idata, mp) -> dict:
    """Posterior-mean measurement parameters from the certified mixed fit, by name.

    Returns a dict with: ``Lam`` [Jc, F], ``Phi`` [F, F], ``sigma`` [Jc] (residual SD, *including*
    the psi_floor as used in the fit), and per non-Gaussian item the scalars/vectors used by its
    likelihood — binary/count: ``a_/lh_/lg_`` (count also ``alpha_``); ordinal: ``lh_/lg_`` + cutpoints
    ``c_`` (ordinals have no intercept — cutpoints carry it, matching ``build_mixed``)."""
    post = idata.posterior
    mean = lambda v: np.asarray(post[v].mean(("chain", "draw")).values)  # noqa: E731
    P = {"Lam": mean("Lam"), "Phi": mean("Phi"), "sigma": 0.05 + mean("sigma")}  # psi_floor=0.05
    for it in mp.bin_items:
        P[f"a_{it}"] = float(mean(f"a_{it}"))
        P[f"lh_{it}"] = float(mean(f"lh_{it}"))
        P[f"lg_{it}"] = float(mean(f"lg_{it}"))
    for it in mp.cnt_items:
        P[f"a_{it}"] = float(mean(f"a_{it}"))
        P[f"lh_{it}"] = float(mean(f"lh_{it}"))
        P[f"lg_{it}"] = float(mean(f"lg_{it}"))
        P[f"alpha_{it}"] = float(mean(f"alpha_{it}"))
    for it in mp.ord_items:
        P[f"lh_{it}"] = float(mean(f"lh_{it}"))
        P[f"lg_{it}"] = float(mean(f"lg_{it}"))
        P[f"c_{it}"] = mean(f"c_{it}")
    return P


def build_projection(mp, P, psi_floor: float = 0.05):
    """The fixed-parameter projection model: only the per-patient explicit latents ``z_e`` are free.

    Mirrors ``continuous_core.build_mixed`` exactly, but every measurement parameter is a numpy
    constant from ``P`` (the certified posterior means) — so the posterior over ``z_e`` (hence
    ``f_e``) is the conditional projection of each patient onto the fixed map. The continuous block is
    the same marginalized Woodbury (residual ``r = x − f_e Bᵀ``); the non-Gaussian items keep their
    Bernoulli / NegBinomial / OrderedLogistic likelihoods on ``f_e``."""
    import pymc as pm
    import pytensor.tensor as pt

    base = mp.base
    M = base.M
    N, Jc = M.shape
    F = len(base.factor_cols)
    Ke, Km = len(mp.e_cols), len(mp.m_cols)
    mask = (~np.isnan(M)).astype("float64")
    x = np.nan_to_num(M, nan=0.0)
    kobs = mask.sum(1)
    log2pi = float(np.log(2.0 * np.pi))
    pat_mask, pat_inv = _patterns(mask)

    # fixed conditional-decomposition constants (numpy), from the certified Λ, Φ
    Phi, Lam = P["Phi"], P["Lam"]
    e, m = mp.e_cols, mp.m_cols
    Phi_ee = Phi[np.ix_(e, e)]
    Phi_mm = Phi[np.ix_(m, m)]
    Phi_me = Phi[np.ix_(m, e)]
    Mmat = Phi_me @ np.linalg.inv(Phi_ee)               # [Km, Ke]
    S = Phi_mm - Mmat @ Phi_me.T                         # [Km, Km] residual cov
    C_S = np.linalg.cholesky(S + 1e-8 * np.eye(Km))
    L_ee = np.linalg.cholesky(Phi_ee + 1e-8 * np.eye(Ke))
    Bmat = (Lam[:, e] + Lam[:, m] @ Mmat)               # [Jc, Ke] mean loadings on f_e
    Lt = (Lam[:, m] @ C_S)                              # [Jc, Km] residual loadings
    sig2 = P["sigma"] ** 2                               # [Jc]

    with pm.Model() as model:
        z = pm.Normal("z_e", 0.0, 1.0, shape=(N, Ke))
        f_e = pm.Deterministic("f_e", z @ L_ee.T)       # Cov(rows) = Phi_ee
        r = pt.as_tensor(x) - f_e @ Bmat.T              # [N, Jc] continuous residual
        ll = _woodbury_potential(pt, r, mask, pt.as_tensor(Lt), pt.as_tensor(sig2),
                                 pat_mask, pat_inv, kobs, Km, log2pi)
        pm.Potential("cont_ll", ll.sum())

        for k, it in enumerate(mp.bin_items):
            y = mp.Bin[:, k]
            obs = np.flatnonzero(~np.isnan(y))
            eta = P[f"a_{it}"] + P[f"lh_{it}"] * f_e[:, mp.ng_home[it]][obs] + P[f"lg_{it}"] * f_e[:, 0][obs]
            pm.Bernoulli(f"y_{it}", logit_p=eta, observed=y[obs].astype("int8"))
        for k, it in enumerate(mp.cnt_items):
            y = mp.Cnt[:, k]
            obs = np.flatnonzero(~np.isnan(y))
            eta = P[f"a_{it}"] + P[f"lh_{it}"] * f_e[:, mp.ng_home[it]][obs] + P[f"lg_{it}"] * f_e[:, 0][obs]
            pm.NegativeBinomial(f"y_{it}", mu=pt.exp(eta), alpha=P[f"alpha_{it}"],
                                observed=np.rint(y[obs]).astype("int64"))
        for k, it in enumerate(mp.ord_items):
            y = mp.Ord[:, k]
            obs = np.flatnonzero(~np.isnan(y))
            eta = P[f"lh_{it}"] * f_e[:, mp.ng_home[it]][obs] + P[f"lg_{it}"] * f_e[:, 0][obs]
            pm.OrderedLogistic(f"y_{it}", eta=eta, cutpoints=pt.as_tensor(P[f"c_{it}"]),
                               observed=y[obs].astype("int32"), compute_p=False)
    return model


def project_explicit_full_n(mp, idata, *, draws: int = 500, tune: int = 600, chains: int = 2,
                            target_accept: float = 0.9, seed: int = 20260609):
    """Sample the per-patient explicit latents ``f_e`` under fixed certified parameters (full N).

    Returns ``dict`` with ``mean``/``sd`` [N, Ke], the thinned posterior ``draws`` [S, N, Ke], the
    explicit factor names ``fcols``, and sampler ``diag`` (max R-hat over z_e, #divergences)."""
    import arviz as az
    import pymc as pm

    P = fixed_params(idata, mp)
    model = build_projection(mp, P)
    with model:
        ip = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept,
                       nuts_sampler="numpyro", random_seed=seed, progressbar=True,
                       idata_kwargs={"log_likelihood": False})
    fe = np.asarray(ip.posterior["f_e"].values)          # [chains, draws, N, Ke]
    S = fe.shape[0] * fe.shape[1]
    fe = fe.reshape((S,) + fe.shape[2:])                  # [S, N, Ke]
    try:
        rh = az.rhat(ip, var_names=["z_e"])["z_e"].values
        max_rhat = float(np.nanmax(rh))
    except Exception:
        max_rhat = float("nan")
    ndiv = int(ip.sample_stats["diverging"].values.sum()) if "diverging" in ip.sample_stats else -1
    # explicit factor order = mp.e_cols order in base.factor_cols
    fcols = [mp.base.factor_cols[c] for c in mp.e_cols]
    return dict(mean=fe.mean(0), sd=fe.std(0), draws=fe, fcols=fcols,
                diag={"max_rhat": max_rhat, "divergences": ndiv, "n_draws": S})


def align_ordinals_to_fit(mp_full, cert_index, B_full) -> dict:
    """Re-encode full-N ordinals with the CERTIFIED fit's category coding, so the fixed cutpoints
    remain valid. The certified subsample defines, per ordinal item, the category→code map (rank of
    its observed categories). Applied to all patients: categories above the top certified category
    collapse into the top code (standard ordered-logistic top-category absorption), below → 0, unseen
    interior → nearest lower. ``ord_K`` is set to the certified K. Returns #re-mapped per item.

    (Necessary because a fresh full-N recode can see rare categories absent from the fit subsample —
    e.g. isf08a's raw {6,7,10} beyond the certified {0..4} — which would mis-align the cutpoints.)"""
    import pandas as pd
    clipped = {}
    for k, it in enumerate(mp_full.ord_items):
        uniq = np.sort(pd.to_numeric(B_full.loc[cert_index][it], errors="coerce").dropna().unique())
        remap = {float(v): i for i, v in enumerate(uniq)}
        K = len(uniq)
        raw = pd.to_numeric(B_full.loc[mp_full.base.index][it], errors="coerce").to_numpy()
        code = np.full(len(raw), np.nan)
        nremap = 0
        for i, v in enumerate(raw):
            if np.isnan(v):
                continue
            if v in remap:
                code[i] = remap[v]
            elif v > uniq[-1]:
                code[i] = K - 1; nremap += 1
            elif v < uniq[0]:
                code[i] = 0; nremap += 1
            else:
                code[i] = remap[float(uniq[uniq <= v].max())]; nremap += 1
        mp_full.Ord[:, k] = code
        mp_full.ord_K[k] = K
        clipped[it] = nremap
    return clipped


def explicit_nobs(mp) -> dict:
    """Observed non-Gaussian home-indicator count per patient, per explicit factor (reliability)."""
    base = mp.base
    N = base.M.shape[0]
    fcols = [base.factor_cols[c] for c in mp.e_cols]
    col = {f: i for i, f in enumerate(fcols)}
    n_obs = np.zeros((N, len(fcols)), dtype=int)
    blocks = [(mp.bin_items, mp.Bin), (mp.ord_items, mp.Ord), (mp.cnt_items, mp.Cnt)]
    for items, arr in blocks:
        for k, it in enumerate(items):
            home_e = mp.ng_home[it]                       # index into mp.e_cols order
            n_obs[:, home_e] += (~np.isnan(arr[:, k])).astype(int)
    return {"n_obs": n_obs, "fcols": fcols}


def conditional_gaussian_draws(M: np.ndarray, post, factor_cols: list[str], *, n_draws: int = 200,
                               psi_floor: float = 0.05, seed: int = 20260609):
    """Draw-wise continuous factor scores: per patient, sample from the conditional-Gaussian posterior
    ``f_i | x_O ~ N(mean_i, cov_pattern)`` (observed cells only), evaluated at posterior-mean loadings.

    Returns ``mean``/``sd`` [N, F] (matching ``conditional_gaussian_scores``) plus ``draws``
    [n_draws, N, F] for the uncertainty-export / archetype-over-draws arm. The conditional cov depends
    on the row only through its observed pattern, so each pattern's mean-map + cov-chol is computed once."""
    rng = np.random.default_rng(seed)
    Lam = np.asarray(post["Lam"].mean(("chain", "draw")).values)
    Phi = np.asarray(post["Phi"].mean(("chain", "draw")).values)
    sig = psi_floor + np.asarray(post["sigma"].mean(("chain", "draw")).values)
    N, _ = M.shape
    F = len(factor_cols)
    mask = ~np.isnan(M)
    X = np.nan_to_num(M, nan=0.0)
    pats, inv = np.unique(mask, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    mean = np.full((N, F), np.nan)
    sd = np.full((N, F), np.nan)
    draws = np.full((n_draws, N, F), np.nan, dtype="float32")
    for p in range(pats.shape[0]):
        cols = np.flatnonzero(pats[p])
        rows = np.flatnonzero(inv == p)
        if cols.size == 0:
            # no observed cells: prior N(0, Phi)
            mean[rows] = 0.0
            sd[rows] = np.sqrt(np.clip(np.diag(Phi), 0.0, None))
            L = np.linalg.cholesky(Phi + 1e-8 * np.eye(F))
            draws[:, rows, :] = (rng.standard_normal((n_draws, len(rows), F)) @ L.T).astype("float32")
            continue
        LamO = Lam[cols]
        So = LamO @ Phi @ LamO.T + np.diag(sig[cols] ** 2)
        try:
            Soi = np.linalg.inv(So)
        except np.linalg.LinAlgError:
            Soi = np.linalg.pinv(So)
        B = Phi @ LamO.T @ Soi                            # [F, k]
        mu = X[np.ix_(rows, cols)] @ B.T                  # [len(rows), F]
        cov = Phi - B @ LamO @ Phi                        # [F, F] (pattern-level)
        cov = 0.5 * (cov + cov.T)
        sd_p = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        L = np.linalg.cholesky(cov + 1e-8 * np.eye(F))
        mean[rows] = mu
        sd[rows] = sd_p
        eps = rng.standard_normal((n_draws, len(rows), F))
        draws[:, rows, :] = (mu[None, :, :] + eps @ L.T).astype("float32")
    return dict(mean=mean, sd=sd, draws=draws)
