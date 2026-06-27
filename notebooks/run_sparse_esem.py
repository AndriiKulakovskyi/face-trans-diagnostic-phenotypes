#!/usr/bin/env python
"""Sparse-ESEM VALIDATION + SELECTOR for the 8-factor operational map (decoupled architecture).

Fits the continuous backbone (G, cognition, immunometabolic, sleep, developmental_risk, mania_activation)
with EVERY off-home specific cross-loading freed under a regularized horseshoe (stable variant: fixed global
tau + lighter Student-t local tails), warm-started from the hard-zero backbone — so the data pull cross-
loadings *out of zero* only where they genuinely support them. It then:

  1. VALIDATES the operational hard-zero map: what fraction of cross-loadings stay ~0 (the zeros are earned,
     not just imposed) and whether the home loadings move when cross-loadings are allowed (backbone stability).
  2. SELECTS the few credible cross-loadings (95% CI excludes 0) — the candidates to fold into the
     operational map — and flags the largest cell(s) for genuine-vs-artifact review.

    HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_sparse_esem.py --smoke
    python3 scripts/run_job.py sparse_esem -- env HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_sparse_esem.py
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import xarray as xr


def _find_repo_root(start: Path) -> Path:
    for c in (start, *start.parents):
        if (c / "src" / "face" / "models").exists() and (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repo root not found")


REPO = _find_repo_root(Path(__file__).resolve())
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
for m in [n for n in sys.modules if n == "face" or n.startswith("face.")]:
    f = getattr(sys.modules[m], "__file__", None)
    if f and SRC not in Path(f).resolve().parents:
        del sys.modules[m]

import arviz as az  # noqa: E402

from face.models.bayesian.measurement_model_oop import (  # noqa: E402
    MeasurementConfig,
    MeasurementDataset,
    StageDefinition,
    StageRunner,
)
from face.reporting.loading_atlas import DISP  # noqa: E402

MERGED = REPO / "configs" / "prior_loading_matrix_v3_biomerge.csv"
F6 = ["overall_severity", "cognition", "immunometabolic", "sleep", "developmental_risk", "mania_activation"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--tau0", type=float, default=0.05)
    p.add_argument("--slab-c", type=float, default=0.30)
    p.add_argument("--local-df", type=float, default=3.0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.smoke:
        a.n, a.draws, a.tune = 600, 60, 80
    cont = dict(correlated=True, windows=True, mixed=False, balanced=True, n_subsample=a.n, seed=20260605)
    ch = 4 if not a.smoke else 2
    backbone = StageDefinition("hs_s3_merged", F6, draws=1000, tune=1000, chains=ch, target_accept=0.95, **cont)
    sparse = StageDefinition("sparse_esem_6f", F6, draws=a.draws, tune=a.tune, chains=ch, target_accept=0.95, **cont)

    base = MeasurementConfig().with_gaussian_copula()
    base = replace(base, prior_matrix=MERGED, output_dir=base.output_dir / "copula" / "horseshoe_8d")
    hs = base.with_horseshoe(tau0=a.tau0, slab_c=a.slab_c, fixed_tau=True, local_df=a.local_df)

    print(f"[esem] stable horseshoe: fixed tau={a.tau0}, slab_c={a.slab_c}, local Student-t df={a.local_df}; "
          f"N={a.n} draws={a.draws}", flush=True)

    ds = MeasurementDataset(hs)
    spec = ds.loading_spec(ds.core(F6, correlated=True, windows=True, balanced=True, n_subsample=a.n,
                                   seed=20260605), windows=True)
    print(f"[esem] freed cross-loading cells: {len(spec.hs_cells)} (every off-home specific cell)", flush=True)

    # fit (warm-started from the cached hard-zero backbone, so cross-loadings start at 0)
    idata, man = StageRunner(hs).run_stage(sparse, overwrite=a.overwrite, prev_stage=backbone)
    print(f"[esem] diagnostics={man.get('diagnostics', {})}", flush=True)

    # --- analysis: convergence on the loadings, sparsity, the credible selections ---
    out = hs.output_dir / sparse.name
    post = xr.open_dataset(out / "idata.nc", group="posterior")
    for v in ["Lam", "lam_pos", "lam_hs", "Phi", "sigma"]:
        if v in post.data_vars:
            r = az.rhat(post[v]).values
            print(f"[esem]   {v:8s} R-hat max={np.nanmax(r):.2f} mean={np.nanmean(r):.2f}", flush=True)

    core = ds.core(F6, correlated=True, windows=True, balanced=True, n_subsample=a.n, seed=20260605)
    items, facs = core.items, core.factor_cols
    lam = post["lam_hs"].stack(s=("chain", "draw")).transpose("s", ...).values  # (S, n_hs)
    med = np.median(lam, axis=0)
    lo, hi = np.quantile(lam, 0.025, axis=0), np.quantile(lam, 0.975, axis=0)
    cred = (lo > 0) | (hi < 0)
    absmed = np.abs(med)
    print(f"\n[esem] SPARSITY: {(absmed < 0.05).mean():.0%} of {len(med)} cross-loadings shrunk to |median|<0.05; "
          f"{int(cred.sum())} are credible (95% CI excludes 0)", flush=True)
    print("[esem] SELECTED cross-loadings (credible), sorted by |median|:", flush=True)
    order = np.argsort(-absmed)
    rows = []
    for k in order:
        if not cred[k]:
            continue
        it, fac = items[spec.hs_cells[k][0]], facs[spec.hs_cells[k][1]]
        home = core.home[spec.hs_cells[k][0]]
        print(f"    {DISP.get(it, it):28s} (home {home:18s}) -> {fac:18s}  {med[k]:+.3f}  "
              f"[{lo[k]:+.3f}, {hi[k]:+.3f}]", flush=True)
        rows.append(dict(item=it, home=home, factor=fac, loading=float(med[k]),
                         ci_low=float(lo[k]), ci_high=float(hi[k])))
    import csv
    rep = REPO / "reports" / "sparse_esem_credible_cross.csv"
    with open(rep, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item", "home", "factor", "loading", "ci_low", "ci_high"])
        w.writeheader(); w.writerows(rows)
    print(f"\n[esem] wrote {rep} ({len(rows)} selected cross-loadings)", flush=True)


if __name__ == "__main__":
    main()
