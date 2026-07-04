#!/usr/bin/env python3
"""Generate a synthetic, FACE-like dataset so the measurement engine runs WITHOUT confidential data (P7-03).

The raw FACE cohort data are confidential, which blocks external reproduction. This builds a synthetic
baseline with the SAME shape as the real one — the real modelled-indicator set + likelihood families +
burden signs from ``configs/loading_matrix.csv``, FACE-like cohort imbalance (BP ≫ SZ > DR) and
missingness — but drawn from a KNOWN bifactor model (G ⟂ specifics; biology near-⟂G by construction).
Point the engine at it with ``FACE_DATA_DIR`` and you can run / certify on synthetic data and recover the
planted structure.

    python3 synthetic/generate_face_like.py                       # -> synthetic/data/*.parquet + truth.json
    FACE_DATA_DIR=synthetic/data python3 scripts/04_fit.py --stage 1

Writes ``baseline_v0.parquet`` (+ ``covariates_v0`` + ``site_v0``) and ``truth.json`` (the planted Λ/σ).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.measurement.kernel import G_KEY, MATRIX, S1_FACTORS  # noqa: E402

COHORTS = {"bp": 0.62, "sz": 0.24, "dr": 0.14}            # FACE-like imbalance
# Planted G-loading per factor: biology near-⟂G (the headline), cognition/sleep partly track G.
G_LOAD = {"cognition": 0.35, "sleep": 0.30, "metabolic": 0.08, "inflammatory": 0.07}


def generate(n: int = 900, seed: int = 0, out: str | Path | None = None, miss: float = 0.25):
    rng = np.random.default_rng(seed)
    m = pd.read_csv(MATRIX)
    meta = m.drop_duplicates("item").set_index("item")[["likelihood_family", "modeling_block", "item_sign"]]
    home = (m[m.prior_type.isin(["primary", "g_anchor"])].drop_duplicates("item")
            .set_index("item")["factor"].to_dict())
    items = sorted(it for it in home
                   if home[it] in S1_FACTORS and meta.loc[it, "modeling_block"] == "continuous")
    factor_cols = [G_KEY] + [f for f in S1_FACTORS if f != G_KEY]
    fcol = {f: i for i, f in enumerate(factor_cols)}
    F = len(factor_cols)

    # Planted loadings: positive home loading + the factor's planted G-loading; everything else 0.
    Lam = np.zeros((len(items), F))
    sigma = rng.uniform(0.5, 0.9, size=len(items))
    for j, it in enumerate(items):
        Lam[j, fcol[home[it]]] = rng.uniform(0.45, 0.8)
        Lam[j, fcol[G_KEY]] = G_LOAD.get(home[it], 0.2)

    # Latent factors: G ⟂ specifics, specifics independent (Φ = I) — the S1 identification.
    fmat = rng.normal(size=(n, F))
    eta = fmat @ Lam.T
    X = eta + rng.normal(size=eta.shape) * sigma[None, :]

    df = pd.DataFrame(X, columns=items)
    for it in items:                                     # native direction + family shape
        sgn = int(meta.loc[it, "item_sign"])
        df[it] = sgn * df[it]
        if meta.loc[it, "likelihood_family"] == "lognormal":
            df[it] = np.exp(0.5 * df[it] - 0.5 * df[it].min() + 1.0)   # strictly positive; log ∝ latent

    coh = rng.choice(list(COHORTS), size=n, p=list(COHORTS.values()))
    idx = pd.MultiIndex.from_arrays([coh, [f"S{i:05d}" for i in range(n)]],
                                    names=["cohort", "patient_id"])
    df.index = idx
    df = df.mask(rng.random(df.shape) < miss)            # MCAR missingness; NaN preserved (no imputation)

    cov = pd.DataFrame({"age": rng.normal(40, 12, n).clip(18, 80),
                        "sex": rng.integers(0, 2, n).astype(float),
                        "education_years": rng.normal(12, 3, n).clip(0, 22),
                        "edulevel": rng.integers(0, 5, n).astype(float)}, index=idx)
    site = pd.Series(rng.integers(0, 8, n).astype(float), index=idx, name="siteid_city")

    outdir = Path(out) if out is not None else REPO / "synthetic" / "data"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / "baseline_v0.parquet")
    cov.to_parquet(outdir / "covariates_v0.parquet")
    site.to_frame().to_parquet(outdir / "site_v0.parquet")
    truth = {"items": items, "factor_cols": factor_cols, "home": {it: home[it] for it in items},
             "Lam_true": Lam.tolist(), "sigma_true": sigma.tolist(), "G_load": G_LOAD, "n": n, "seed": seed}
    (outdir / "truth.json").write_text(json.dumps(truth, indent=2))
    return outdir, truth


def main() -> None:
    outdir, truth = generate()
    print(f"wrote synthetic FACE-like data -> {outdir}")
    print(f"  {len(truth['items'])} continuous indicators · {len(truth['factor_cols'])} factors · n={truth['n']}")
    print(f"  planted G-loadings: {truth['G_load']} (biology near-⟂G by construction)")


if __name__ == "__main__":
    main()
