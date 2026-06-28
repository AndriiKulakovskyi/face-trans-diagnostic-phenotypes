"""Congruence metrics: variational GLLVM vs the NUTS copula M1 fit.

The VI engine is an acceleration arm; this module quantifies how faithfully it reproduces
the NUTS authority on the same 8-factor operational map:

* **Tucker congruence** of the loading columns, per factor (sign/column aligned);
* **Pearson correlation** of the per-patient coordinates, per factor, over well-observed
  patients;
* **Phi agreement** — the inter-factor correlation cells (immunometabolic block + the
  trivially-orthogonal G / substance rows).

The pure metric functions take DataFrames and are torch-free.  The NUTS targets are
materialized on the fly from the cached idata via the certified
``export_loadings_summary`` / ``export_phi`` (no separate 8-dim CSV exists yet).  The
verdict is worded "congruent", not "certified": NUTS remains the inferential authority.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Acceptance bars (from the M1 adjudication congruence floor).
TUCKER_MAJOR = 0.95
TUCKER_THIN = 0.85
COORD_R = 0.90
PHI_CELL = 0.05
MAJOR_AXES = ("overall_severity", "cognition", "immunometabolic", "sleep")


# --------------------------------------------------------------------------- loaders
def load_vi(vi_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load a VI ``consolidate/`` (or stage) directory's exports."""
    vi_dir = Path(vi_dir)
    out: dict[str, pd.DataFrame] = {
        "loadings": pd.read_csv(vi_dir / "loadings_summary.csv"),
        "phi": pd.read_csv(vi_dir / "phi.csv", index_col=0),
    }
    coord_path = vi_dir / "coordinates.parquet"
    if coord_path.exists():
        out["coordinates"] = pd.read_parquet(coord_path)
    return out


def nuts_targets_from_idata(
    idata_path: str | Path,
    config,
    *,
    factors: list[str],
    explicit_factors: list[str] | None = None,
    specific_cross: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Export the NUTS loadings_summary + Phi from a cached copula idata.

    ``config`` is the ``MeasurementConfig`` the fit ran under (8-factor biomerge_xc, copula,
    substance-orthogonal); ``factors`` is the fit's factor list (e.g. ``F8_FIT``).  Rebuilds
    the ``MixedData`` deterministically (same data contract) so ``export_loadings_summary``
    can resolve item/factor names, then reads the posterior with the certified exporters.
    """
    import arviz as az  # type: ignore[reportMissingImports]

    from face.models.bayesian.measurement_model_oop import (
        DEFAULT_EXPLICIT_FACTORS,
        MeasurementDataset,
    )
    from face.models.bayesian.synthetic import export_loadings_summary, export_phi

    idata = az.from_netcdf(str(idata_path))
    ds = MeasurementDataset(config)
    explicit = list(explicit_factors or DEFAULT_EXPLICIT_FACTORS)
    mixed = ds.mixed(list(factors), explicit_factors=explicit)
    loadings = export_loadings_summary(idata, mixed, config, specific_cross=specific_cross)
    phi = export_phi(idata, mixed.base.factor_cols)
    return loadings, phi


# --------------------------------------------------------------------------- metrics
def _loading_pivot(loadings: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Wide item x factor loading matrix from a long loadings_summary frame."""
    piv = loadings.pivot_table(index="item", columns="factor", values="loading", aggfunc="first")
    return piv.reindex(columns=factors)


def tucker_congruence_per_factor(
    load_vi: pd.DataFrame, load_nuts: pd.DataFrame, factors: list[str] | None = None
) -> pd.DataFrame:
    """Tucker's congruence coefficient per factor between the VI and NUTS loading columns.

    Columns correspond by name (both fits share the factor order); a per-column sign flip
    is applied if the raw congruence is negative (factors are sign-identified only up to the
    home-loading convention).  Computed over the items present in BOTH columns.
    """
    factors = factors or sorted(set(load_vi["factor"]) | set(load_nuts["factor"]))
    a = _loading_pivot(load_vi, factors)
    b = _loading_pivot(load_nuts, factors)
    rows = []
    for f in factors:
        if f not in a.columns or f not in b.columns:
            continue
        va, vb = a[f], b[f]
        common = va.notna() & vb.notna()
        x, y = va[common].to_numpy(float), vb[common].to_numpy(float)
        if x.size == 0 or np.allclose(x, 0) or np.allclose(y, 0):
            rows.append({"factor": f, "n_cells": int(x.size), "tucker": np.nan, "sign_flip": False})
            continue
        num = float(x @ y)
        den = float(np.linalg.norm(x) * np.linalg.norm(y))
        t = num / den if den > 0 else np.nan
        flip = bool(t < 0)
        rows.append({"factor": f, "n_cells": int(x.size), "tucker": abs(t), "sign_flip": flip})
    df = pd.DataFrame(rows)
    if not df.empty:
        bar = [TUCKER_MAJOR if f in MAJOR_AXES else TUCKER_THIN for f in df["factor"]]
        df["bar"] = bar
        df["pass"] = df["tucker"] >= df["bar"]
    return df


def coordinate_correlation_per_factor(
    coords_vi: pd.DataFrame,
    coords_nuts: pd.DataFrame,
    factors: list[str],
    *,
    min_n_obs: int = 3,
) -> pd.DataFrame:
    """Per-factor Pearson r between VI and NUTS patient coordinate means, over patients with
    >= ``min_n_obs`` observed home indicators for that factor (the "well" reliability tier).
    Both frames use the ``{factor}__mean`` / ``{factor}__n_obs`` schema.

    Aligns on the common (cohort, patient_id) index; falls back to positional alignment when
    both frames are full-N in the same patient order (verified for the baseline/strata pair).
    """
    common = coords_vi.index.intersection(coords_nuts.index)
    if len(common) > 0 and len(common) >= 0.5 * min(len(coords_vi), len(coords_nuts)):
        vi, nuts = coords_vi.loc[common], coords_nuts.loc[common]
    elif len(coords_vi) == len(coords_nuts):
        vi, nuts = coords_vi.reset_index(drop=True), coords_nuts.reset_index(drop=True)
    else:
        return pd.DataFrame()  # cannot align
    rows = []
    for f in factors:
        col = f"{f}__mean"
        if col not in vi or col not in nuts:
            continue
        x, y = vi[col].to_numpy(float), nuts[col].to_numpy(float)
        nobs = (vi[f"{f}__n_obs"].to_numpy() if f"{f}__n_obs" in vi
                else np.full(x.size, min_n_obs))
        # Auto-relax the reliability threshold for thin factors (e.g. mania has only 2 home
        # indicators, so no patient can reach the default "well" tier of >=3 observed).
        thr, relaxed = min_n_obs, False
        keep = np.isfinite(x) & np.isfinite(y) & (np.asarray(nobs, float) >= thr)
        while keep.sum() < 5 and thr > 1:
            thr -= 1
            relaxed = True
            keep = np.isfinite(x) & np.isfinite(y) & (np.asarray(nobs, float) >= thr)
        if keep.sum() < 5:
            rows.append({"factor": f, "n": int(keep.sum()), "r": np.nan,
                         "signed_r": np.nan, "min_n_obs": thr, "relaxed": relaxed})
            continue
        r = float(np.corrcoef(x[keep], y[keep])[0, 1])
        rows.append({"factor": f, "n": int(keep.sum()), "r": abs(r), "signed_r": r,
                     "min_n_obs": thr, "relaxed": relaxed})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["pass"] = df["r"] >= COORD_R
    return df


def phi_agreement(phi_vi: pd.DataFrame, phi_nuts: pd.DataFrame) -> dict:
    """Off-diagonal agreement of the two Phi matrices (aligned by factor name)."""
    factors = [f for f in phi_vi.index if f in phi_nuts.index]
    a = phi_vi.loc[factors, factors].to_numpy(float)
    b = phi_nuts.loc[factors, factors].to_numpy(float)
    off = ~np.eye(len(factors), dtype=bool)
    diff = np.abs(a - b)[off]
    cells = {}
    for fi in ("overall_severity", "cognition", "immunometabolic"):
        for fj in factors:
            if fi in factors and fi != fj:
                i, j = factors.index(fi), factors.index(fj)
                cells[f"{fi}~{fj}"] = {"vi": float(a[i, j]), "nuts": float(b[i, j])}
    return {
        "factors": factors,
        "frobenius_offdiag": float(np.sqrt((np.abs(a - b)[off] ** 2).sum())),
        "max_abs_offdiag_diff": float(diff.max()) if diff.size else np.nan,
        "pass": bool(diff.max() <= PHI_CELL) if diff.size else False,
        "key_cells": cells,
    }


# ----------------------------------------------------------------------------- PPC
def ppc_from_fit(fit: dict) -> pd.DataFrame:
    """Per-item observed-vs-reconstructed posterior predictive summary over observed cells.

    Continuous: mean & variance; bernoulli: endorsement rate; ordinal: modal-category rate;
    count: zero-rate & mean.  Uses the variational posterior mean ``mu`` pushed through the
    decoder (a fast deterministic PPC; full simulation is a v0.2 upgrade).
    """
    import torch

    model = fit["model"]
    data = fit["data"]
    device = model.q_mu.weight.device
    with torch.no_grad():
        f, _mu, _lv, _U = model.sample_latent(
            torch.arange(model.N, device=device), deterministic=True
        )
        eta = model.linear_predictor(f)[0].cpu().numpy()  # (N, J)
    x = data.M_raw
    mask = np.isfinite(x)
    rows = []
    for j, item in enumerate(data.items):
        obs = mask[:, j]
        if obs.sum() == 0:
            continue
        fam = data.families[j]
        xo = x[obs, j]
        eo = eta[obs, j]
        rec: dict = {"item": item, "family": fam, "n_obs": int(obs.sum())}
        if fam == "gaussian":
            rec |= {"obs_mean": float(xo.mean()), "rec_mean": float(eo.mean()),
                    "obs_var": float(xo.var()), "rec_var": float(eo.var() + model.sigma()[j].item() ** 2)}
        elif fam == "bernoulli":
            p = 1.0 / (1.0 + np.exp(-eo))
            rec |= {"obs_rate": float(xo.mean()), "rec_rate": float(p.mean())}
        elif fam == "count":
            mu = np.exp(np.clip(eo, -10, 10))
            rec |= {"obs_zero_rate": float((xo == 0).mean()), "obs_mean": float(xo.mean()),
                    "rec_mean": float(mu.mean())}
        elif fam == "ordinal":
            rec |= {"obs_modal_rate": float(np.bincount(xo.astype(int)).max() / xo.size)}
        rows.append(rec)
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- report
def run_congruence(
    vi_dir: str | Path,
    nuts_loadings: pd.DataFrame,
    nuts_phi: pd.DataFrame,
    *,
    nuts_coords: pd.DataFrame | None = None,
    out_csv: str | Path | None = None,
) -> dict:
    """Compute the full congruence report and (optionally) write ``validation_report.csv``."""
    vi = load_vi(vi_dir)
    factors = list(nuts_phi.index)
    tucker = tucker_congruence_per_factor(vi["loadings"], nuts_loadings, factors)
    phi = phi_agreement(vi["phi"], nuts_phi)
    coord = None
    if nuts_coords is not None and "coordinates" in vi:
        coord = coordinate_correlation_per_factor(vi["coordinates"], nuts_coords, factors)
    report = {
        "tucker": tucker,
        "phi": phi,
        "coordinates": coord,
        "verdict": _verdict(tucker, phi, coord),
    }
    if out_csv is not None:
        _write_report(report, Path(out_csv))
    return report


def _verdict(tucker: pd.DataFrame, phi: dict, coord: pd.DataFrame | None) -> str:
    parts = []
    if not tucker.empty:
        parts.append(f"Tucker: {int(tucker['pass'].sum())}/{len(tucker)} factors congruent")
    parts.append("Phi: " + ("congruent" if phi.get("pass") else
                            f"max offdiag diff {phi.get('max_abs_offdiag_diff'):.3f}"))
    if coord is not None and not coord.empty:
        parts.append(f"Coords: {int(coord['pass'].sum())}/{len(coord)} factors r>={COORD_R}")
    return "; ".join(parts)


def _write_report(report: dict, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    t = report["tucker"].copy()
    if not t.empty:
        t.insert(0, "metric", "tucker")
        frames.append(t.rename(columns={"tucker": "value"}))
    if report["coordinates"] is not None and not report["coordinates"].empty:
        c = report["coordinates"].copy()
        c.insert(0, "metric", "coordinate_r")
        frames.append(c.rename(columns={"r": "value"}))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)
