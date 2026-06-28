#!/usr/bin/env python
"""Synthetic-patient generative check for the variational GLLVM.

Fits a clean generative variant of the 8-factor map (``covariate_mode="none"`` so the rank-INT
copula inverts exactly to the raw scale; low-rank q), draws N synthetic patients from the fitted
generative model, and compares their **raw-variable distributions** to the observed data:

  * per-variable marginal overlays (a representative panel across families),
  * a marginal summary table with KS distances (observed vs synthetic), and
  * the observed-vs-synthetic correlation matrices on the continuous block + the SRMR (the test
    of whether the latent factor structure reproduces the joint distribution).

    HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_gllvm_synthetic_check.py
    # quick wiring check:
    HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_gllvm_synthetic_check.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for c in (start, *start.parents):
        if (c / "src" / "face" / "models").exists() and (c / "pyproject.toml").exists():
            return c
    raise RuntimeError(f"Could not locate FACE repository root from {start}")


REPO = _find_repo_root(Path(__file__).resolve())
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
for m in [n for n in sys.modules if n == "face" or n.startswith("face.")]:
    f = getattr(sys.modules[m], "__file__", None)
    if f and SRC not in Path(f).resolve().parents:
        del sys.modules[m]

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from face.models.variational.generative import (  # noqa: E402
    correlation_block,
    generate_synthetic,
    marginal_summary,
)
from face.models.variational.gllvm_model_oop import (  # noqa: E402
    GLLVMConfig,
    GLLVMRunner,
)

# A representative panel spanning factors + families for the marginal-overlay figure.
PANEL = ["egf", "cgi01", "bmi", "hba1c", "crp", "wbc", "psqi", "cvlt_total_recall",
         "tmt_a_time_sec", "altman", "isf01", "suoccur_alcool"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=None, help="number of synthetic patients (default: = N observed)")
    p.add_argument("--epochs", type=int, default=2500)
    p.add_argument("--q-rank", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260605)
    p.add_argument("--smoke", action="store_true", help="tiny fit + small synthetic for wiring")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def marginal_overlays(observed: pd.DataFrame, synth: pd.DataFrame, families: dict,
                      figdir: Path) -> str:
    items = [it for it in PANEL if it in synth.columns and it in observed.columns]
    ncol = 4
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.1 * nrow))
    for ax, it in zip(np.atleast_1d(axes).ravel(), items, strict=False):
        o = pd.to_numeric(observed[it], errors="coerce").dropna().to_numpy()
        s = synth[it].dropna().to_numpy()
        fam = families.get(it, "")
        if fam in ("bernoulli", "ordinal", "count"):
            vals = np.union1d(np.unique(o), np.unique(s))
            if vals.size <= 12:
                w = 0.4
                ax.bar(vals - w / 2, [np.mean(o == v) for v in vals], width=w, label="observed",
                       color="#4878a8", alpha=0.85)
                ax.bar(vals + w / 2, [np.mean(s == v) for v in vals], width=w, label="synthetic",
                       color="#e76f51", alpha=0.85)
            else:
                _hist(ax, o, s)
        else:
            _hist(ax, o, s)
        ax.set_title(it, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in np.atleast_1d(axes).ravel()[len(items):]:
        ax.axis("off")
    handles, labels = np.atleast_1d(axes).ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle("V-GLLVM generative check: observed (blue) vs synthetic (orange) raw distributions", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = figdir / "synthetic_marginal_overlays.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def _hist(ax, o, s):
    lo = float(np.nanpercentile(np.concatenate([o, s]), 0.5))
    hi = float(np.nanpercentile(np.concatenate([o, s]), 99.5))
    bins = np.linspace(lo, hi, 40)
    ax.hist(o, bins=bins, density=True, color="#4878a8", alpha=0.55, label="observed")
    ax.hist(s, bins=bins, density=True, color="#e76f51", alpha=0.55, label="synthetic")


def correlation_figure(co: pd.DataFrame, cs: pd.DataFrame, srmr: float, figdir: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (mat, title) in zip(axes, [(co, "observed"), (cs, "synthetic"),
                                       (co - cs, "observed - synthetic")], strict=False):
        vmax = 0.4 if "-" in title else 1.0
        im = ax.imshow(mat.to_numpy(), vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    # observed vs synthetic off-diagonal correlation scatter
    off = ~np.eye(len(co), dtype=bool)
    x = co.to_numpy()[off]; y = cs.to_numpy()[off]
    keep = np.isfinite(x) & np.isfinite(y)
    axes[2].set_title(f"obs - syn  (SRMR={srmr:.3f})", fontsize=10)
    fig.suptitle(f"Continuous-block correlation structure · SRMR = {srmr:.3f} · "
                 f"corr(obs,syn off-diag) = {np.corrcoef(x[keep], y[keep])[0, 1]:.3f}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = figdir / "synthetic_correlation_structure.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def main() -> None:
    args = parse_args()
    cfg = GLLVMConfig()
    cfg = replace(cfg, covariate_mode="none", include_covariates=False, q_rank=args.q_rank,
                  epochs=args.epochs, seed=args.seed,
                  output_dir=REPO / "results" / "face" / "gllvm_oop_gen")
    if args.smoke:
        cfg = replace(cfg.with_smoke_defaults(), covariate_mode="none", include_covariates=False,
                      q_rank=args.q_rank, output_dir=cfg.output_dir)

    runner = GLLVMRunner(cfg)
    print(f"[gen] fitting generative model (covariate_mode=none, q_rank={cfg.q_rank}) ...", flush=True)
    fit = runner.run_plan(overwrite=args.overwrite)
    data = fit["data"]
    model = fit["model"]

    n = args.n or len(data.index)
    print(f"[gen] drawing {n} synthetic patients ...", flush=True)
    synth = generate_synthetic(model, data, n=n, seed=args.seed + 1)

    baseline = pd.read_parquet(cfg.processed_dir / "baseline_v0.parquet")
    observed = baseline[[c for c in data.items if c in baseline.columns]]
    families = {it: data.families[i] for i, it in enumerate(data.items)}

    figdir = REPO / "docs" / "figures" / "gllvm_oop"
    figdir.mkdir(parents=True, exist_ok=True)
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    ms = marginal_summary(observed, synth, families)
    ms.to_csv(out / "marginal_summary.csv", index=False)
    cont_items = [it for i, it in enumerate(data.items) if data.families[i] == "gaussian"]
    co, cs, srmr = correlation_block(observed, synth, cont_items)
    synth.to_parquet(out / "synthetic_patients.parquet")

    f1 = marginal_overlays(observed, synth, families, figdir)
    f2 = correlation_figure(co, cs, srmr, figdir)

    by_fam = ms.groupby("family")["ks"].median().round(3).to_dict()
    summary = {
        "n_synthetic": int(n), "q_rank": cfg.q_rank,
        "ks_median": round(float(ms["ks"].median()), 3),
        "ks_p90": round(float(ms["ks"].quantile(0.9)), 3),
        "ks_median_by_family": by_fam,
        "correlation_srmr": round(srmr, 3),
        "worst_items": ms.head(6)[["item", "family", "ks"]].to_dict("records"),
    }
    (out / "generative_check_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n=== GENERATIVE CHECK ===", flush=True)
    print(json.dumps(summary, indent=2, default=float), flush=True)
    print(f"\n[gen] figures: {Path(f1).name}, {Path(f2).name}", flush=True)
    print(f"[gen] report -> {out}", flush=True)


if __name__ == "__main__":
    main()
