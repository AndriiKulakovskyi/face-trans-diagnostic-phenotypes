#!/usr/bin/env python
"""Figures for the variational GLLVM arm: training curves + VI-vs-NUTS congruence.

Reads the fitted outputs under ``results/face/gllvm_oop/`` (stage training histories, the
``consolidate/`` exports, the cached ``nuts_targets/``, and ``metrics.json``) and writes PNGs
to ``docs/figures/gllvm_oop/``.  Run AFTER the production fit + ``run_gllvm_validation.py``.

    python notebooks/gllvm_oop_make_figures.py
"""
from __future__ import annotations

import sys
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

import json  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from face.models.variational.gllvm_model_oop import GLLVMConfig, GLLVMVisualizer  # noqa: E402

GOOD, BAD = "#2a9d8f", "#e76f51"
KIND_COLOR = {"primary": "#264653", "g_anchor": "#264653", "bifactor_G": "#e9c46a",
              "window": "#8ab17d", "cross": "#e76f51"}


def training_curve_figs(cfg: GLLVMConfig, viz: GLLVMVisualizer) -> list[str]:
    out = []
    for stage_dir in sorted(cfg.output_dir.glob("*/")):
        hist = stage_dir / "training_history.csv"
        if not hist.exists():
            continue
        df = pd.read_csv(hist)
        name = stage_dir.name
        out.append(viz.training_curves(df, title=f"GLLVM {name} training", name=f"training_{name}"))
    return out


def loading_scatter(cfg: GLLVMConfig, viz: GLLVMVisualizer) -> str | None:
    vi_p = cfg.output_dir / "consolidate" / "loadings_summary.csv"
    nuts_p = cfg.output_dir / "nuts_targets" / "loadings_summary.csv"
    if not (vi_p.exists() and nuts_p.exists()):
        return None
    vi = pd.read_csv(vi_p)
    nuts = pd.read_csv(nuts_p)
    merged = vi.merge(nuts, on=["item", "factor"], suffixes=("_vi", "_nuts"))
    if merged.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 6))
    for kind, grp in merged.groupby("kind_vi"):
        ax.scatter(grp["loading_nuts"], grp["loading_vi"], s=18, alpha=0.7,
                   label=kind, color=KIND_COLOR.get(kind, "#888"))
    lim = max(abs(merged["loading_vi"]).max(), abs(merged["loading_nuts"]).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.5)
    r = float(np.corrcoef(merged["loading_vi"], merged["loading_nuts"])[0, 1])
    ax.set_xlabel("NUTS loading"); ax.set_ylabel("VI loading")
    ax.set_title(f"VI vs NUTS loadings (n={len(merged)}, r={r:.3f})")
    ax.legend(fontsize=8)
    path = viz.config.figure_dir / "vi_vs_nuts_loadings.png"
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return str(path)


def phi_side_by_side(cfg: GLLVMConfig, viz: GLLVMVisualizer) -> str | None:
    vi_p = cfg.output_dir / "consolidate" / "phi.csv"
    nuts_p = cfg.output_dir / "nuts_targets" / "phi.csv"
    if not (vi_p.exists() and nuts_p.exists()):
        return None
    vi = pd.read_csv(vi_p, index_col=0)
    nuts = pd.read_csv(nuts_p, index_col=0)
    factors = [f for f in vi.index if f in nuts.index]
    vi, nuts = vi.loc[factors, factors], nuts.loc[factors, factors]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (mat, title) in zip(axes, [(vi, "VI Phi"), (nuts, "NUTS Phi"),
                                       (vi - nuts, "VI - NUTS")], strict=False):
        vmax = 1.0 if "Phi" in title else 0.3
        im = ax.imshow(mat.to_numpy(), vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        ax.set_xticks(range(len(factors))); ax.set_yticks(range(len(factors)))
        ax.set_xticklabels(factors, rotation=90, fontsize=7)
        ax.set_yticklabels(factors, fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.7)
    path = viz.config.figure_dir / "vi_vs_nuts_phi.png"
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return str(path)


def metrics_bars(cfg: GLLVMConfig, viz: GLLVMVisualizer) -> str | None:
    mp = cfg.output_dir / "metrics.json"
    if not mp.exists():
        return None
    m = json.loads(mp.read_text())
    tucker = pd.DataFrame(m.get("tucker", []))
    coord = pd.DataFrame(m.get("coordinates", []))
    n = 1 + (1 if not coord.empty else 0)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4))
    axes = np.atleast_1d(axes)
    if not tucker.empty:
        ax = axes[0]
        colors = [GOOD if p else BAD for p in tucker.get("pass", [False] * len(tucker))]
        ax.bar(tucker["factor"], tucker["tucker"], color=colors)
        ax.axhline(0.95, ls="--", c="gray", lw=0.8, label="0.95 (major)")
        ax.axhline(0.85, ls=":", c="gray", lw=0.8, label="0.85 (thin)")
        ax.set_ylim(0, 1.02); ax.set_ylabel("Tucker congruence"); ax.set_title("Loadings: VI vs NUTS")
        ax.set_xticks(range(len(tucker)))
        ax.set_xticklabels(tucker["factor"], rotation=45, ha="right", fontsize=8); ax.legend(fontsize=7)
    if not coord.empty:
        ax = axes[1]
        colors = [GOOD if p else BAD for p in coord.get("pass", [False] * len(coord))]
        ax.bar(coord["factor"], coord["r"], color=colors)
        ax.axhline(0.90, ls="--", c="gray", lw=0.8, label="0.90")
        ax.set_ylim(0, 1.02); ax.set_ylabel("coordinate Pearson r"); ax.set_title("Coordinates: VI vs NUTS")
        ax.set_xticks(range(len(coord)))
        ax.set_xticklabels(coord["factor"], rotation=45, ha="right", fontsize=8); ax.legend(fontsize=7)
    path = viz.config.figure_dir / "vi_vs_nuts_metrics.png"
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return str(path)


def ppc_fig(cfg: GLLVMConfig, viz: GLLVMVisualizer) -> str | None:
    pp = cfg.output_dir / "consolidate" / "ppc.csv"
    if not pp.exists():
        return None
    df = pd.read_csv(pp)
    panels = [("gaussian", "obs_mean", "rec_mean", "continuous mean"),
              ("bernoulli", "obs_rate", "rec_rate", "binary rate"),
              ("count", "obs_mean", "rec_mean", "count mean")]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
    for ax, (fam, ox, rx, title) in zip(axes, panels, strict=False):
        sub = df[df["family"] == fam]
        if sub.empty or ox not in sub or rx not in sub:
            ax.set_title(f"{title} (n/a)"); continue
        ax.scatter(sub[ox], sub[rx], s=20, alpha=0.7, color="#264653")
        lo = float(min(sub[ox].min(), sub[rx].min())); hi = float(max(sub[ox].max(), sub[rx].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        r = float(np.corrcoef(sub[ox], sub[rx])[0, 1]) if len(sub) > 2 else float("nan")
        ax.set_xlabel(f"observed {title}"); ax.set_ylabel(f"reconstructed {title}")
        ax.set_title(f"PPC {fam} (n={len(sub)}, r={r:.2f})")
    fig.suptitle("Posterior predictive checks (observed vs reconstructed)")
    path = viz.config.figure_dir / "ppc_observed_vs_reconstructed.png"
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return str(path)


def main() -> None:
    cfg = GLLVMConfig()
    viz = GLLVMVisualizer(cfg)
    made: list[str] = []
    made += training_curve_figs(cfg, viz)
    for fn in (loading_scatter, phi_side_by_side, metrics_bars, ppc_fig):
        p = fn(cfg, viz)
        if p:
            made.append(p)
    print(f"[figures] wrote {len(made)} figures to {cfg.figure_dir}:", flush=True)
    for p in made:
        print(f"  {Path(p).name}", flush=True)


if __name__ == "__main__":
    main()
