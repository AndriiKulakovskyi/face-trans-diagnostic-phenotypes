"""Continuous-core measurement engine (Stage 1) — explicit-latent bifactor, full-N V0.

Driven by ``configs/prior_loading_matrix_v3.csv``; reads ``data/processed/baseline_v0.parquet``
(built by ``scripts/01_build_data.py``). Stage-1 factor set: a general factor **G**
(``overall_severity``) + the continuous specific factors **cognition / metabolic / inflammatory /
sleep**. Continuous (gaussian / lognormal) indicators only — the binary/ordinal/count blocks and
ESEM cross-loadings are added at later stages.

Model (the doc's "first stable fit", §3.1 / §4.2):
    G_i  ~ Normal(0, 1)                          general factor (per patient)
    D_ik ~ Normal(0, 1)                          specific factors, INDEPENDENT (bifactor)
    eta_ij = alpha_j + lambda_jG * G_i + sum_k lambda_jk * D_ik
    X_ij ~ Normal(eta_ij, sigma_j)               OBSERVED cells only (NaN -> no term; no imputation)

Loadings come from the prior matrix: each item loads positively on its home factor (primary /
G-anchor; sign-anchored on the burden-oriented data), each specific item carries a signed bifactor
cell on G (plausible_cross). Simple structure otherwise (specific x specific cross-loadings are a
Stage-2 ESEM addition). Data are oriented (higher = burden), log-transformed (lognormal), z-scored.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
PROC = REPO / "data" / "processed"
MATRIX = REPO / "configs" / "prior_loading_matrix_v3.csv"

S1_FACTORS = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep"]


@dataclass
class CorePrep:
    M: np.ndarray                 # [N, J] z-scored, oriented continuous block; NaN = missing
    items: list[str]
    spec_factors: list[str]       # specific factors (G excluded), in column order
    g_anchor_items: list[int]     # item indices whose home is G (load + on G only)
    spec_items: list[int]         # item indices with a specific home factor
    spec_home: dict               # {item_idx: specific-factor column}
    cellG: dict                   # {item_idx: (mu, sd)} G-anchor prior
    cellGx: dict                  # {item_idx: (mu, sd)} specific item's bifactor-G prior
    cellHome: dict                # {item_idx: (mu, sd)} specific home prior
    cohort: np.ndarray
    index: pd.Index


def prepare(factors: list[str] = S1_FACTORS, n_subsample: int | None = None,
            seed: int = 20260605) -> CorePrep:
    """Load the persisted V0 baseline, encode the continuous block for `factors`, and resolve
    the per-cell loading priors from the matrix. `n_subsample` (random) is for smoke tests only;
    the reported fit uses the full sample."""
    m = pd.read_csv(MATRIX)
    meta = m.drop_duplicates("item").set_index("item")[
        ["likelihood_family", "modeling_block", "item_sign"]]
    home = (m[m.prior_type.isin(["primary", "g_anchor"])].drop_duplicates("item")
            .set_index("item")["factor"].to_dict())
    cell = {(r.item, r.factor): (r.prior_type, float(r.prior_mean), float(r.prior_sd))
            for r in m.itertuples()}
    spec_factors = [f for f in factors if f != "overall_severity"]

    items = sorted(it for it in home
                   if home[it] in factors and meta.loc[it, "modeling_block"] == "continuous")

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

    scol = {f: i for i, f in enumerate(spec_factors)}
    g_anchor_items, spec_items, spec_home = [], [], {}
    cellG, cellGx, cellHome = {}, {}, {}
    for j, it in enumerate(items):
        h = home[it]
        if h == "overall_severity":
            g_anchor_items.append(j)
            _, mu, sd = cell[(it, "overall_severity")]
            cellG[j] = (mu, sd)
        else:
            spec_items.append(j)
            spec_home[j] = scol[h]
            _, mu, sd = cell[(it, h)]
            cellHome[j] = (mu, sd)
            gx = cell.get((it, "overall_severity"), ("plausible_cross", 0.0, 0.25))
            cellGx[j] = (gx[1], gx[2])

    return CorePrep(M=Mdf.to_numpy(), items=items, spec_factors=spec_factors,
                    g_anchor_items=g_anchor_items, spec_items=spec_items, spec_home=spec_home,
                    cellG=cellG, cellGx=cellGx, cellHome=cellHome,
                    cohort=cohort, index=Mdf.index)


def build_model(prep: CorePrep):
    """Explicit-latent bifactor model (independent specifics). Returns a PyMC model with
    deterministics `lamG_full` [J] and `lamS_full` [J, Ks] for reading loadings back."""
    import pymc as pm
    import pytensor.tensor as pt

    M = prep.M
    N, J = M.shape
    Ks = len(prep.spec_factors)
    obs_r, obs_c = np.where(~np.isnan(M))
    y = M[obs_r, obs_c].astype("float64")
    obs_r = obs_r.astype("int64")
    obs_c = obs_c.astype("int64")

    def _arr(items, d, k):  # gather mu (k=0) / sd (k=1) for an item list
        return np.array([d[j][k] for j in items], dtype="float64")

    gA = np.array(prep.g_anchor_items, dtype="int64")
    gS = np.array(prep.spec_items, dtype="int64")
    sp_k = np.array([prep.spec_home[j] for j in prep.spec_items], dtype="int64")

    with pm.Model() as model:
        G = pm.Normal("G", 0.0, 1.0, shape=N)
        D = pm.Normal("D", 0.0, 1.0, shape=(N, Ks))            # independent specifics (bifactor)

        lamG = pt.zeros(J)
        if len(gA):
            vg = pm.TruncatedNormal("lamG_anchor", mu=_arr(prep.g_anchor_items, prep.cellG, 0),
                                    sigma=_arr(prep.g_anchor_items, prep.cellG, 1),
                                    lower=0.0, shape=len(gA))
            lamG = pt.set_subtensor(lamG[gA], vg)
        if len(gS):
            vgx = pm.Normal("lamG_spec", mu=_arr(prep.spec_items, prep.cellGx, 0),
                            sigma=_arr(prep.spec_items, prep.cellGx, 1), shape=len(gS))
            lamG = pt.set_subtensor(lamG[gS], vgx)

        lamS = pt.zeros((J, Ks))
        if len(gS):
            vsp = pm.TruncatedNormal("lamS_home", mu=_arr(prep.spec_items, prep.cellHome, 0),
                                     sigma=_arr(prep.spec_items, prep.cellHome, 1),
                                     lower=0.0, shape=len(gS))
            lamS = pt.set_subtensor(lamS[gS, sp_k], vsp)

        alpha = pm.Normal("alpha", 0.0, 1.5, shape=J)
        sigma = 0.05 + pm.HalfNormal("sigma", 1.0, shape=J)

        eta = alpha[None, :] + G[:, None] * lamG[None, :] + D @ lamS.T     # [N, J]
        pm.Normal("y", mu=eta[obs_r, obs_c], sigma=sigma[obs_c], observed=y)
        pm.Deterministic("lamG_full", lamG)
        pm.Deterministic("lamS_full", lamS)
    return model


def build_marginalized(prep: CorePrep, psi_floor: float = 0.05):
    """Marginalized (Woodbury, low-rank) bifactor — funnel-free, no per-patient latents.

    Integrates G, D out (Phi = I): each patient's observed cells ~ MVN(0, Lam Lam' + diag(psi)).
    Computed via the matrix-determinant lemma + Woodbury so the per-patient work is O(F^2)
    (F = 1 + #specifics = 5), fully vectorized over patients with a 0/1 mask (no pattern
    grouping, no patient dropped). Tiny parameter space -> mixes fast -> certifies on CPU.
    Run via `pm.sample(nuts_sampler="numpyro")` so JAX vmap-vectorizes the batched k×k linalg.
    """
    import pymc as pm
    import pytensor.tensor as pt

    M = prep.M
    N, J = M.shape
    F = 1 + len(prep.spec_factors)
    mask = (~np.isnan(M)).astype("float64")
    x = np.nan_to_num(M, nan=0.0)
    kobs = mask.sum(1)
    log2pi = float(np.log(2.0 * np.pi))

    gA = np.array(prep.g_anchor_items, "int64")
    gS = np.array(prep.spec_items, "int64")
    sp_k = np.array([prep.spec_home[j] for j in prep.spec_items], "int64")

    def _a(items, d, i):
        return np.array([d[j][i] for j in items], "float64")

    with pm.Model() as model:
        lamG = pt.zeros(J)
        if len(gA):
            vg = pm.TruncatedNormal("lamG_anchor", mu=_a(prep.g_anchor_items, prep.cellG, 0),
                                    sigma=_a(prep.g_anchor_items, prep.cellG, 1), lower=0.0, shape=len(gA))
            lamG = pt.set_subtensor(lamG[gA], vg)
        if len(gS):
            vgx = pm.Normal("lamG_spec", mu=_a(prep.spec_items, prep.cellGx, 0),
                            sigma=_a(prep.spec_items, prep.cellGx, 1), shape=len(gS))
            lamG = pt.set_subtensor(lamG[gS], vgx)
        lamS = pt.zeros((J, F - 1))
        if len(gS):
            vsp = pm.TruncatedNormal("lamS_home", mu=_a(prep.spec_items, prep.cellHome, 0),
                                     sigma=_a(prep.spec_items, prep.cellHome, 1), lower=0.0, shape=len(gS))
            lamS = pt.set_subtensor(lamS[gS, sp_k], vsp)
        Lam = pt.concatenate([lamG[:, None], lamS], axis=1)               # [J, F]

        sigma = psi_floor + pm.HalfNormal("sigma", 1.0, shape=J)
        psi = sigma ** 2
        W = mask / psi[None, :]                                            # [N, J] precision (0 if missing)

        P = W[:, :, None] * Lam[None, :, :]                                # [N, J, F]
        A = pt.eye(F)[None, :, :] + (P.transpose(0, 2, 1) @ Lam)           # [N, F, F] = I + Lam'WLam
        b = (W * x) @ Lam                                                  # [N, F] = Lam'Wx
        Lc = pt.linalg.cholesky(A)                                         # batched (JAX-vectorized)
        logdetA = 2.0 * pt.log(pt.diagonal(Lc, axis1=-2, axis2=-1)).sum(-1)
        sol = pt.linalg.solve_triangular(Lc, b[:, :, None], lower=True)[:, :, 0]
        quadA = (sol ** 2).sum(-1)
        term1 = (W * x ** 2).sum(1)
        logdetPsi = (mask * pt.log(psi)[None, :]).sum(1)
        ll = -0.5 * (kobs * log2pi + logdetPsi + logdetA + term1 - quadA)  # [N]
        pm.Potential("obs_ll", ll.sum())
        pm.Deterministic("lamG_full", lamG)
        pm.Deterministic("lamS_full", lamS)
    return model
