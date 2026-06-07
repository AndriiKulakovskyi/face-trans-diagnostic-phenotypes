"""Continuous-core measurement engine (Stages S1–S2) — marginalized bifactor/ESEM, full-N V0.

Driven by ``configs/prior_loading_matrix_v3.csv``; reads ``data/processed/baseline_v0.parquet``
(built by ``scripts/01_build_data.py``). Factor set: a general factor **G**
(``overall_severity``) + the continuous specific factors **cognition / metabolic / inflammatory /
sleep**. Continuous (gaussian / lognormal) indicators only — the binary/ordinal/count blocks are
added at later stages (S3+).

Two stages share one builder, gated by flags resolved from the prior matrix:

    S1  cross=False, windows=False, Phi = I        the certified "first stable fit"
    S2  cross=True,  windows=True,  Phi = Phi_full  ESEM cross-loadings + MADRS/QIDS/STAI windows
                                                    + inter-dimension correlations Phi

Model (the doc's §3.1):
    G_i  ~ Normal(0, 1)                          general factor (per patient)
    D_i  ~ Normal(0, Phi_spec)                   specific factors (Phi=I at S1; LKJ at S2)
    eta_ij = lambda_jG * G_i + sum_k lambda_jk * D_ik
    X_ij ~ Normal(eta_ij, sigma_j)               OBSERVED cells only (NaN -> no term; no imputation)

Loading cells come from the prior matrix x the stage flags:
  * ``pos``  (primary / G-anchor home cells) — ``TruncatedNormal(mu, sd, >0)``, sign-anchored on the
    burden-oriented data (this anchors the rotation; the specific factors keep their orientation).
  * ``signed`` (bifactor-G cells; at S2 also specific<->specific ``plausible_cross`` cells and the
    window cells) — ``Normal(mu, sd)``, shrunk toward 0. The ~980 ``unlikely`` cells stay hard-zero
    (the default ``plausible_only`` ESEM arm); the bifactor G cells keep sd 0.25, specific<->specific
    cross cells are tightened by ``cross_sd_scale`` so Phi carries the inter-factor association.

Bifactor identification: **G is held orthogonal to the specifics** (Phi_full has an identity G
row/col; only the specific block correlates via LKJ). The correlated-G sensitivity variant is an S5
addition, not done here.

The marginalized (Woodbury, low-rank) likelihood integrates the latents out and is fully vectorized
over patients with a 0/1 mask — no per-patient funnel, no observed-pattern grouping, no patient
dropped — which is what lets the full N = 9,013 certify on the Mac CPU. The Phi reparameterization
``Lam_tilde = Lam @ chol(Phi_full)`` makes ``Sigma = Lam Phi Lam' + Psi = Lam_tilde Lam_tilde' + Psi``,
so the exact S1 Woodbury kernel runs unchanged on ``Lam_tilde``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
PROC = REPO / "data" / "processed"
MATRIX = REPO / "configs" / "prior_loading_matrix_v3.csv"

S1_FACTORS = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep"]
WINDOWS = ["madrs", "qidsr120", "staya"]   # cross-loading windows (no home factor; §2)
G_KEY = "overall_severity"


@dataclass
class CorePrep:
    M: np.ndarray                 # [N, J] z-scored, oriented continuous block; NaN = missing
    items: list[str]
    home: list[str]               # home factor per item ("" for the cross-loading windows)
    factor_cols: list[str]        # [G, *specifics], the loading-matrix column order
    spec_factors: list[str]       # specific factors (G excluded), in column order
    g_col: int                    # index of G in factor_cols (0)
    pos_cells: list               # [(j, c, mu, sd)] TruncatedNormal>0 (primary / G-anchor)
    sgn_cells: list               # [(j, c, mu, sd)] Normal (bifactor-G / cross / window)
    correlated: bool              # Phi = LKJ over specifics (else I); the S2 inter-dimension Phi
    cohort: np.ndarray
    index: pd.Index
    kind: dict = field(default_factory=dict)   # {(j, c): "g_anchor|primary|bifactor_G|cross|window"}


def prepare(factors: list[str] = S1_FACTORS, *, correlated: bool = False,
            windows: bool = False, specific_cross: bool = False,
            cross_sd_scale: float = 0.25, window_sd_scale: float = 1.0,
            n_subsample: int | None = None, seed: int = 20260605) -> CorePrep:
    """Load the persisted V0 baseline, encode the continuous block for `factors`, and resolve
    the per-cell loading priors from the matrix. Three orthogonal switches deform S1 -> S2:

      * `correlated`  — estimate Phi (LKJ) over the specifics instead of Phi = I. The S2
        inter-dimension correlations; G stays orthogonal (bifactor).
      * `windows`     — add the MADRS/QIDS/STAI window items (no home factor) as signed
        cross-loadings onto G / cognition / sleep. These are the well-identified ESEM
        cross-loadings (a window is not a dimension, so window->factor is a regression onto an
        anchored factor, not a rotation), kept at the full plausible sd (`window_sd_scale`=1).
      * `specific_cross` — free the specific<->specific (all metabolic<->inflammatory) cross-
        loadings. **Off by default at S2:** freeing them *both ways* alongside a free
        Phi_{metab,inflam} is rotationally aliased (it made full-N intractable — NUTS crawls the
        ridge), and they are *not separately identifiable* from Phi. With them off, **Phi carries
        the metabolic/inflammatory association** (the identified estimand). `cross_sd_scale` is the
        ridge guard used only if they are re-enabled (sensitivity arm).

    S1 = all switches off (Phi = I, simple structure). S2 = `correlated=windows=True,
    specific_cross=False`. `n_subsample` (random) is for smoke tests only.
    """
    m = pd.read_csv(MATRIX)
    meta = m.drop_duplicates("item").set_index("item")[
        ["likelihood_family", "modeling_block", "item_sign"]]
    home = (m[m.prior_type.isin(["primary", "g_anchor"])].drop_duplicates("item")
            .set_index("item")["factor"].to_dict())
    cell = {(r.item, r.factor): (r.prior_type, float(r.prior_mean), float(r.prior_sd))
            for r in m.itertuples()}
    spec_factors = [f for f in factors if f != G_KEY]
    factor_cols = [G_KEY] + spec_factors
    col = {f: i for i, f in enumerate(factor_cols)}

    items = sorted(it for it in home
                   if home[it] in factors and meta.loc[it, "modeling_block"] == "continuous")
    use_windows = bool(windows)
    if use_windows:                        # window items: no home, plausible_cross onto our factors
        win = [w for w in WINDOWS if w in meta.index
               and meta.loc[w, "modeling_block"] == "continuous"
               and any(cell.get((w, f), ("", 0, 0))[0] == "plausible_cross" for f in factors)]
        items = sorted(set(items) | set(win))

    B = pd.read_parquet(PROC / "baseline_v0.parquet")
    items = [it for it in items if it in B.columns]
    cohort = np.asarray(B.index.get_level_values("cohort"))

    cols = {}
    for it in items:
        v = pd.to_numeric(B[it], errors="coerce").astype(float)
        if meta.loc[it, "likelihood_family"] == "lognormal":
            mn = np.nanmin(v.values)
            v = np.log1p(v - mn + 1e-6) if (np.isfinite(mn) and mn <= 0) else np.log(v)
        v = int(meta.loc[it, "item_sign"]) * v
        sd = v.std()
        cols[it] = (v - v.mean()) / sd if sd and sd > 0 else v * 0.0
    Mdf = pd.DataFrame(cols, index=B.index)

    if n_subsample and n_subsample < len(Mdf):
        rng = np.random.default_rng(seed)
        ix = np.sort(rng.choice(len(Mdf), size=n_subsample, replace=False))
        Mdf, cohort = Mdf.iloc[ix], cohort[ix]

    homes = [home.get(it, "") for it in items]
    pos_cells, sgn_cells, kind = [], [], {}
    for j, it in enumerate(items):
        h = homes[j]
        for f in factor_cols:
            c = col[f]
            ptype, mu, sd = cell.get((it, f), ("unlikely_cross", 0.0, 0.05))
            if f == h:                                    # home cell -> positive anchor
                pos_cells.append((j, c, mu, sd))
                kind[(j, c)] = "g_anchor" if f == G_KEY else "primary"
            elif f == G_KEY:                              # bifactor-G cell (signed, sd unscaled)
                if h:                                     # homed specific item -> always (bifactor)
                    gx = cell.get((it, G_KEY), ("plausible_cross", 0.0, 0.25))
                    sgn_cells.append((j, c, gx[1], gx[2]))
                    kind[(j, c)] = "bifactor_G"
                elif use_windows and ptype == "plausible_cross":   # window on G
                    sgn_cells.append((j, c, mu, sd))
                    kind[(j, c)] = "window"
            elif ptype == "plausible_cross":             # specific-column cross-loading
                if h == "":                              # window -> specific (kept free)
                    if use_windows:
                        sgn_cells.append((j, c, mu, sd * window_sd_scale))
                        kind[(j, c)] = "window"
                elif specific_cross:                     # specific <-> specific (ridge-guarded)
                    sgn_cells.append((j, c, mu, sd * cross_sd_scale))
                    kind[(j, c)] = "cross"

    return CorePrep(M=Mdf.to_numpy(), items=items, home=homes, factor_cols=factor_cols,
                    spec_factors=spec_factors, g_col=col[G_KEY],
                    pos_cells=pos_cells, sgn_cells=sgn_cells,
                    correlated=bool(correlated), cohort=cohort, index=Mdf.index, kind=kind)


def _cell_arrays(cells):
    if not cells:
        z = np.zeros(0)
        return z.astype("int64"), z.astype("int64"), z, z
    r = np.array([c[0] for c in cells], "int64")
    cc = np.array([c[1] for c in cells], "int64")
    mu = np.array([c[2] for c in cells], "float64")
    sd = np.array([c[3] for c in cells], "float64")
    return r, cc, mu, sd


def _build_phi(pm, pt, prep, lkj_eta: float):
    """Phi_full [F, F]: G orthogonal (identity row/col); specific block ~ LKJ correlation.
    Returns (Phi, R) where R = chol(Phi) for the Woodbury reparameterization. Phi = I at S1."""
    F = len(prep.factor_cols)
    if not prep.correlated or F <= 2:
        return pt.eye(F), pt.eye(F)
    spec_idx = [i for i in range(F) if i != prep.g_col]
    ns = len(spec_idx)
    craw = pm.LKJCorr("Phi_spec", n=ns, eta=lkj_eta)          # PyMC 6: full [ns, ns] matrix
    Cs = pt.tril(craw, -1) + pt.tril(craw, -1).T + pt.eye(ns)
    E = np.zeros((F, ns))
    for k, i in enumerate(spec_idx):
        E[i, k] = 1.0
    eg = np.zeros((F, 1)); eg[prep.g_col, 0] = 1.0
    Phi = pt.as_tensor(E) @ Cs @ pt.as_tensor(E).T + pt.as_tensor(eg @ eg.T)
    return Phi, pt.linalg.cholesky(Phi)


def _build_loadings(pm, pt, prep, J, F):
    """Assemble Lam [J, F] from the pos (TruncatedNormal>0) and signed (Normal) cells."""
    pr, pc, pmu, psd = _cell_arrays(prep.pos_cells)
    sr, sc, smu, ssd = _cell_arrays(prep.sgn_cells)
    Lam = pt.zeros((J, F))
    if len(pr):
        vpos = pm.TruncatedNormal("lam_pos", mu=pmu, sigma=psd, lower=0.0, shape=len(pr))
        Lam = pt.set_subtensor(Lam[pr, pc], vpos)
    if len(sr):
        vsgn = pm.Normal("lam_cross", mu=smu, sigma=ssd, shape=len(sr))
        Lam = pt.set_subtensor(Lam[sr, sc], vsgn)
    return Lam


def build_marginalized(prep: CorePrep, psi_floor: float = 0.05, lkj_eta: float = 2.0):
    """Marginalized (Woodbury, low-rank) bifactor/ESEM — funnel-free, no per-patient latents.

    Integrates G, D out: each patient's observed cells ~ MVN(0, Lam Phi Lam' + diag(psi)). With
    Lam_tilde = Lam chol(Phi), Sigma = Lam_tilde Lam_tilde' + diag(psi) so the per-patient work is
    the S1 O(F^2) matrix-determinant-lemma + Woodbury kernel (F = 1 + #specifics), fully vectorized
    over patients via a 0/1 mask (no pattern grouping, no patient dropped). Run via
    `pm.sample(nuts_sampler="numpyro")` so JAX vmap-vectorizes the batched F×F linalg.
    """
    import pymc as pm
    import pytensor.tensor as pt

    M = prep.M
    N, J = M.shape
    F = len(prep.factor_cols)
    mask = (~np.isnan(M)).astype("float64")
    x = np.nan_to_num(M, nan=0.0)
    kobs = mask.sum(1)
    log2pi = float(np.log(2.0 * np.pi))

    with pm.Model() as model:
        Lam = _build_loadings(pm, pt, prep, J, F)
        pm.Deterministic("Lam", Lam)
        Phi, R = _build_phi(pm, pt, prep, lkj_eta)
        pm.Deterministic("Phi", Phi)
        Lt = Lam @ R                                                   # [J, F] reparam loadings

        sigma = psi_floor + pm.HalfNormal("sigma", 1.0, shape=J)
        psi = sigma ** 2
        W = mask / psi[None, :]                                        # [N, J] precision (0 if missing)

        P = W[:, :, None] * Lt[None, :, :]                             # [N, J, F]
        A = pt.eye(F)[None, :, :] + (P.transpose(0, 2, 1) @ Lt)        # [N, F, F] = I + Lt'WLt
        b = (W * x) @ Lt                                               # [N, F] = Lt'Wx
        Lc = pt.linalg.cholesky(A)                                     # batched (JAX-vectorized)
        logdetA = 2.0 * pt.log(pt.diagonal(Lc, axis1=-2, axis2=-1)).sum(-1)
        sol = pt.linalg.solve_triangular(Lc, b[:, :, None], lower=True)[:, :, 0]
        quadA = (sol ** 2).sum(-1)
        term1 = (W * x ** 2).sum(1)
        logdetPsi = (mask * pt.log(psi)[None, :]).sum(1)
        ll = -0.5 * (kobs * log2pi + logdetPsi + logdetA + term1 - quadA)  # [N]
        pm.Potential("obs_ll", ll.sum())
    return model


def build_model(prep: CorePrep, lkj_eta: float = 2.0):
    """Explicit-latent bifactor/ESEM (triangulation arm). Returns a PyMC model with the latents
    instantiated: G ~ N(0,1); specifics non-centered with Cov = Phi_spec. Mathematically the same
    posterior as `build_marginalized`; slower at full N (kept for subsample triangulation)."""
    import pymc as pm
    import pytensor.tensor as pt

    M = prep.M
    N, J = M.shape
    F = len(prep.factor_cols)
    Ks = F - 1
    obs_r, obs_c = np.where(~np.isnan(M))
    y = M[obs_r, obs_c].astype("float64")
    obs_r = obs_r.astype("int64"); obs_c = obs_c.astype("int64")

    with pm.Model() as model:
        Lam = _build_loadings(pm, pt, prep, J, F)
        pm.Deterministic("Lam", Lam)
        Phi, R = _build_phi(pm, pt, prep, lkj_eta)
        pm.Deterministic("Phi", Phi)

        G = pm.Normal("G", 0.0, 1.0, shape=N)                         # G orthogonal to specifics
        z = pm.Normal("z", 0.0, 1.0, shape=(N, Ks))
        if prep.correlated and Ks >= 2:
            # specifics correlated: G is column 0, so R is block-diag [[1,0],[0,Rspec]] and
            # D = z @ Rspec' gives Cov(D rows) = Phi_spec. Phi=I at S1 -> Rspec=I -> D=z.
            D = pm.Deterministic("D", z @ R[1:, 1:].T)
        else:
            D = z

        lamG = Lam[:, prep.g_col]                                      # g_col == 0
        lamS = Lam[:, 1:]                                              # [J, Ks] specifics
        alpha = pm.Normal("alpha", 0.0, 1.5, shape=J)
        sigma = 0.05 + pm.HalfNormal("sigma", 1.0, shape=J)
        eta = alpha[None, :] + G[:, None] * lamG[None, :] + D @ lamS.T
        pm.Normal("y", mu=eta[obs_r, obs_c], sigma=sigma[obs_c], observed=y)
    return model


def warmstart_initvals(prep: CorePrep, reports_dir: Path | None = None) -> dict | None:
    """Continuation warm-start (§4.2): seed S2's loadings/residuals from the certified S1 posterior.

    Maps S1's per-(item, factor) loadings (`reports/04_stage1_loadings.csv`) and per-item residual
    scales (`results/face/stage1/idata.nc`) onto S2's cell order by NAME (S2 reorders items + adds
    the windows). Cross-loadings + window cells start at their prior mean (0); Phi at I (left to the
    sampler). This puts every chain in the S1 basin so the cross-loadings/Phi deform from the
    scientifically-correct, S1-continuous solution rather than a cold random mode."""
    import arviz as az
    rep = reports_dir or REPO / "reports"
    f = rep / "04_stage1_loadings.csv"
    if not f.exists():
        return None
    s1 = pd.read_csv(f)
    s1_load = {(r.item, r.factor): float(r.loading) for r in s1.itertuples()}

    lam_pos = np.array([max(0.02, s1_load.get((prep.items[j], prep.factor_cols[c]), mu))
                        for (j, c, mu, sd) in prep.pos_cells], dtype="float64")
    lam_cross = np.array([s1_load.get((prep.items[j], prep.factor_cols[c]), 0.0)
                          if prep.kind[(j, c)] == "bifactor_G" else 0.0
                          for (j, c, mu, sd) in prep.sgn_cells], dtype="float64")
    init = {"lam_pos": lam_pos, "lam_cross": lam_cross}

    nc = REPO / "results" / "face" / "stage1" / "idata.nc"
    if nc.exists():
        try:
            sig1 = az.from_netcdf(str(nc)).posterior["sigma"].mean(("chain", "draw")).values
            p1_items = prepare().items                      # deterministic S1 item order
            sig_map = {it: float(sig1[k]) for k, it in enumerate(p1_items) if k < len(sig1)}
            init["sigma"] = np.array([sig_map.get(it, 0.8) for it in prep.items], dtype="float64")
        except Exception:
            pass
    return init


def thomson_scores(prep: CorePrep, Lam: np.ndarray, Phi: np.ndarray,
                   psi: np.ndarray) -> np.ndarray:
    """Post-hoc regression (Thomson) factor scores from the marginalized fit, observed cells only.

    Sigma = Lam Phi Lam' + diag(psi); per observed-pattern B = Phi Lam_obs' Sigma_obs^{-1}; score row
    = (x_obs) B'. Provisional (a checkpoint read-out for stratification later, not a reported S2 claim)."""
    M = prep.M
    N, F = M.shape[0], Lam.shape[1]
    Sig = Lam @ Phi @ Lam.T + np.diag(psi)
    out = np.full((N, F), np.nan)
    pat: dict = {}
    for i in range(N):
        o = tuple(np.flatnonzero(~np.isnan(M[i])))
        if o:
            pat.setdefault(o, []).append(i)
    for o, rows in pat.items():
        oi = list(o)
        B = Phi @ Lam[oi].T @ np.linalg.pinv(Sig[np.ix_(oi, oi)])
        out[rows] = np.nan_to_num(M[np.ix_(rows, oi)]) @ B.T
    return out
