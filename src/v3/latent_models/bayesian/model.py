"""V3 soft-prior ESEM-bifactor engine — model builder (config-first).

Constructs the PyMC measurement model ENTIRELY from the prior loading matrix + a stage
spec. No hard-coded SPEC, no simple-structure onehot, no fixed loading prior.

Architecture (resolves the marginalize-vs-share tension):
  * CONTINUOUS block — Gaussian/lognormal indicators. Factors are INTEGRATED OUT:
      Sigma = Lam Phi Lam' + diag(psi), summed over observed-cell patterns via per-pattern
      Cholesky (the certified, divergence-free geometry). Lam is the FULL loading matrix
      from the prior tiers: primary cells sign-anchored (positive), cross/bifactor cells
      signed and shrunk toward 0. G is a column held ORTHOGONAL to the specifics (bifactor).
  * EXPLICIT block — ordinal / binary / count indicators. A non-centered Z[N, k] ~ MVN(0,
      Phi_sub) is instantiated for the factors that carry non-Gaussian indicators (each such
      factor ALSO has >=1 continuous indicator, so it is tied into Sigma and Phi is shared
      and jointly estimated). Z feeds ONLY the non-Gaussian likelihoods.

The loading prior comes from configs/prior_loading_matrix_v3.csv; the stage toggles
(include_g, cross_loadings, prior_mode, explicit) come from configs/bayesian_model.yaml.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
MATRIX = REPO / "configs" / "prior_loading_matrix_v3.csv"


def load_cell_priors(matrix_path: Path = MATRIX) -> dict:
    """{(item, factor): (prior_type, mean, sd)} from the prior loading matrix."""
    m = pd.read_csv(matrix_path)
    return {(r.item, r.factor): (r.prior_type, float(r.prior_mean), float(r.prior_sd))
            for r in m.itertuples()}


def continuous_factor_order(data, spec: dict, specific_order: list[str]) -> list[str]:
    """Column order for the continuous block: [G?] + specifics present, in config order."""
    present = set(data.cont_home)
    specifics = [f for f in specific_order if f in present and f != spec.get("general_factor")]
    g = spec.get("general_factor")
    cols = ([g] if (spec.get("include_g") and g in present) else []) + specifics
    return cols


def _cell_active(ptype: str, factor: str, g_key: str, spec: dict) -> str | None:
    """Return 'pos' | 'signed' | None for whether/how a (item,factor) cell is estimated."""
    if ptype in ("primary", "g_anchor"):
        return "pos"
    if ptype == "g_anchor_on_specific":
        return None                                   # severity anchor ~0 on specifics (hard)
    if ptype == "plausible_cross":
        if factor == g_key:
            return "signed" if spec.get("include_g") else None      # bifactor G loading
        return "signed" if spec.get("cross_loadings") else None
    if ptype == "unlikely_cross":
        return "signed" if spec.get("cross_loadings") else None
    return None


def build_model(data, spec: dict, cell_priors: dict, specific_order: list[str],
                psi_floor: float = 0.05, lkj_eta: float = 2.0):
    """Build the staged PyMC model. Returns (model, meta) where meta carries the factor
    order and loading-cell index needed to read posterior loadings back."""
    import pymc as pm
    import pytensor.tensor as pt

    g_key = spec.get("general_factor")
    prior_mode = spec.get("prior_mode", "soft")
    factor_cols = continuous_factor_order(data, spec, specific_order)
    F = len(factor_cols)
    col = {f: i for i, f in enumerate(factor_cols)}
    g_col = col.get(g_key) if spec.get("include_g") else None

    Mv = data.M
    N, J = Mv.shape
    sign = np.array([data.item_sign.get(it, 1) for it in data.cont_items], dtype=float)

    # ---- enumerate active loading cells from the prior matrix x stage toggles ----
    pos_r, pos_c, pos_mu, pos_sd = [], [], [], []
    sgn_r, sgn_c, sgn_mu, sgn_sd = [], [], [], []
    for j, it in enumerate(data.cont_items):
        home = data.cont_home[j]
        for f in factor_cols:
            ptype, mean, sd = cell_priors.get((it, f), ("unlikely_cross", 0.0, 0.05))
            # an item's home cell is 'primary' regardless of which row the matrix stored
            if f == home:
                ptype = "g_anchor" if f == g_key else "primary"
            kind = _cell_active(ptype, f, g_key, spec)
            if kind == "pos":
                pos_r.append(j); pos_c.append(col[f]); pos_mu.append(mean); pos_sd.append(sd)
            elif kind == "signed":
                sgn_r.append(j); sgn_c.append(col[f]); sgn_mu.append(mean); sgn_sd.append(sd)

    pos_r = np.array(pos_r); pos_c = np.array(pos_c)
    sgn_r = np.array(sgn_r); sgn_c = np.array(sgn_c)

    # ---- explicit-block factors: those carrying non-Gaussian indicators this stage ----
    expl_items = _stage_explicit_items(data, spec)
    z_factors = sorted({data.expl_home[it] for it in expl_items if data.expl_home[it] in col},
                       key=lambda f: col[f])
    zc = {f: i for i, f in enumerate(z_factors)}

    meta = {"factor_cols": factor_cols, "g_col": g_col,
            "pos_cells": list(zip(pos_r.tolist(), pos_c.tolist())) if len(pos_r) else [],
            "sgn_cells": list(zip(sgn_r.tolist(), sgn_c.tolist())) if len(sgn_r) else [],
            "z_factors": z_factors, "expl_items": expl_items}

    with pm.Model() as model:
        # ---------- factor correlation Phi (G orthogonal to specifics) ----------
        # Built with selection matrices (no pytensor advanced indexing): scatter the
        # specific-block LKJ correlation into the specific positions; G row/col stays
        # identity (orthogonal). E_spec/e_g are numpy constants in factor_cols order.
        if spec.get("include_g") and g_col is not None and F > 1:
            spec_idx = [i for i, f in enumerate(factor_cols) if f != g_key]
            n_spec = len(spec_idx)
            if n_spec >= 2:
                craw = pm.LKJCorr("Phi_spec", n=n_spec, eta=lkj_eta)
                Cs = pt.tril(craw, -1) + pt.tril(craw, -1).T + pt.eye(n_spec)
            else:
                Cs = pt.eye(max(n_spec, 1))
            E = np.zeros((F, n_spec))
            for k, i in enumerate(spec_idx):
                E[i, k] = 1.0
            eg = np.zeros((F, 1)); eg[g_col, 0] = 1.0
            Phi = pt.as_tensor(E) @ Cs @ pt.as_tensor(E).T + pt.as_tensor(eg @ eg.T)
        elif F >= 2:
            craw = pm.LKJCorr("Phi_spec", n=F, eta=lkj_eta)
            Phi = pt.tril(craw, -1) + pt.tril(craw, -1).T + pt.eye(F)
        else:
            Phi = pt.eye(F)
        pm.Deterministic("Phi", Phi)

        # ---------- continuous (marginalized) block ----------
        nu = pm.Normal("nu", 0.0, 1.0, shape=J)
        psi = psi_floor + pm.HalfNormal("psi_raw", 1.0, shape=J)

        Lam = pt.zeros((J, F))
        if len(pos_r):
            if prior_mode == "certified":
                lam_pos = pm.HalfNormal("lam_pos", 0.6, shape=len(pos_r))
            else:
                lam_pos = pm.TruncatedNormal("lam_pos", mu=np.array(pos_mu),
                                             sigma=np.array(pos_sd), lower=0.0,
                                             shape=len(pos_r))
            lam_pos = lam_pos * pt.as_tensor(sign[pos_r])      # orient by item burden sign
            Lam = pt.set_subtensor(Lam[pos_r, pos_c], lam_pos)
        if len(sgn_r):
            lam_sgn = pm.Normal("lam_cross", mu=np.array(sgn_mu), sigma=np.array(sgn_sd),
                                shape=len(sgn_r))
            Lam = pt.set_subtensor(Lam[sgn_r, sgn_c], lam_sgn)
        pm.Deterministic("Lam", Lam)

        Sigma = Lam @ Phi @ Lam.T + pt.diag(psi)
        ll = 0.0
        for o, rows in data.patterns.items():
            m, oi = len(o), list(o)
            Sel = np.zeros((m, J)); Sel[np.arange(m), oi] = 1.0
            St = pt.as_tensor(Sel)
            Lc = pt.linalg.cholesky(St @ Sigma @ St.T + 1e-6 * pt.eye(m))
            sol = pt.linalg.solve_triangular(
                Lc, (pt.as_tensor(Mv[np.ix_(rows, oi)]) - St @ nu).T, lower=True)
            ll = ll + (-0.5 * (m * np.log(2 * np.pi) + 2 * pt.log(pt.diag(Lc)).sum()
                               + (sol ** 2).sum(axis=0))).sum()
        pm.Potential("cont_ll", ll)

        # ---------- explicit block (shared Phi via Z) ----------
        if z_factors:
            kz = len(z_factors)
            zidx = [col[f] for f in z_factors]
            Ez = np.zeros((F, kz))                      # selection (no advanced indexing)
            for k, i in enumerate(zidx):
                Ez[i, k] = 1.0
            Phi_z = pt.as_tensor(Ez).T @ Phi @ pt.as_tensor(Ez)
            Lz = pt.linalg.cholesky(Phi_z + 1e-6 * pt.eye(kz))
            Zraw = pm.Normal("Z_raw", 0.0, 1.0, shape=(N, kz))
            Z = pm.Deterministic("Z", Zraw @ Lz.T)             # [N, kz] ~ MVN(0, Phi_z)
            _add_explicit_likelihoods(pm, pt, data, spec, Z, zc)

    return model, meta


def _stage_explicit_items(data, spec: dict) -> list[str]:
    """Explicit items active at this stage = union of the named explicit groups."""
    import yaml
    cfg = yaml.safe_load((REPO / "configs" / "bayesian_model.yaml").read_text())
    groups = cfg.get("explicit", {})
    want: list[str] = []
    for g in spec.get("explicit", []) or []:
        want += groups.get(g, [])
    present = set(data.bin_items) | set(data.ord_items) | set(data.cnt_items)
    return [it for it in want if it in present]


def _add_explicit_likelihoods(pm, pt, data, spec, Z, zc) -> None:
    """Attach ordinal / binary / count likelihoods on the shared latent Z (observed cells)."""
    expl = set(_stage_explicit_items(data, spec))
    # binary (Bernoulli)
    for k, it in enumerate(data.bin_items):
        if it not in expl or data.expl_home[it] not in zc:
            continue
        y = data.Bin[:, k]; obs = np.flatnonzero(~np.isnan(y))
        if len(obs) < 30:
            continue
        a = pm.Normal(f"a_{it}", 0.0, 1.5)
        lj = pm.HalfNormal(f"lam_{it}", 0.8)
        pm.Bernoulli(f"y_{it}", logit_p=a + lj * Z[obs, zc[data.expl_home[it]]],
                     observed=y[obs].astype("int8"))
    # ordinal (ordered logistic)
    for k, it in enumerate(data.ord_items):
        if it not in expl or data.expl_home[it] not in zc:
            continue
        y = data.Ord[:, k]; obs = np.flatnonzero(~np.isnan(y))
        if len(obs) < 30:
            continue
        K = int(data.ord_K[k])
        cut = pm.Normal(f"c_{it}", mu=np.linspace(-1.5, 1.5, K - 1), sigma=2.0,
                        shape=K - 1, transform=pm.distributions.transforms.ordered)
        lj = pm.HalfNormal(f"lam_{it}", 0.8)
        pm.OrderedLogistic(f"y_{it}", eta=lj * Z[obs, zc[data.expl_home[it]]], cutpoints=cut,
                           observed=y[obs].astype("int32"), compute_p=False)
    # count (negative binomial)
    for k, it in enumerate(data.cnt_items):
        if it not in expl or data.expl_home[it] not in zc:
            continue
        y = data.Cnt[:, k]; obs = np.flatnonzero(~np.isnan(y))
        if len(obs) < 30:
            continue
        a = pm.Normal(f"a_{it}", 0.0, 1.5)
        lj = pm.HalfNormal(f"lam_{it}", 0.8)
        alpha = pm.HalfNormal(f"alpha_{it}", 2.0)
        pm.NegativeBinomial(f"y_{it}", mu=pt.exp(a + lj * Z[obs, zc[data.expl_home[it]]]),
                            alpha=alpha, observed=np.rint(y[obs]).astype("int64"))
