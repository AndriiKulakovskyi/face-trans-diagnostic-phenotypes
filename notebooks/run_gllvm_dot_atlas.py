#!/usr/bin/env python
"""Dot-atlas loading map for the variational GLLVM (8-factor operational map).

Shows the **complete** indicator→factor mapping for every variable in the model (no per-block
cap): an indicator×factor bubble grid (dot size/colour = |loading|, ring = seed-ensemble CI ≠ 0,
heavy ring = home anchor), grouped by home factor, with the G window items shaded.  Companion
outputs: a standalone 8×8 Φ panel and per-factor interpretability lollipops that show ALL home
indicators (proportional-height panels).  Reuses the shared article atlas
(`face.reporting.loading_atlas`).

Input: the CI-aware loadings written by the seed ensemble
(`results/analyses/variational_gllvm/ensemble/loadings_summary.csv`); falls back to the single-fit
`consolidate/loadings_summary.csv` (NaN CIs) if the ensemble has not been run.

    python notebooks/run_gllvm_dot_atlas.py
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

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from face.reporting import loading_atlas as LA  # noqa: E402

# 8-factor operational map house style.
AXES = ["overall_severity", "cognition", "immunometabolic", "sleep",
        "suicidality", "developmental_risk", "mania_activation", "substance"]
AXLAB = {"overall_severity": "General burden", "cognition": "Cognition",
         "immunometabolic": "Immunometabolic", "sleep": "Sleep", "suicidality": "Suicidality",
         "developmental_risk": "Developmental risk", "mania_activation": "Mania / activation",
         "substance": "Substance"}
AXTAG = {"overall_severity": "G", "cognition": "D1", "immunometabolic": "D2", "sleep": "D3",
         "suicidality": "D4", "developmental_risk": "D5", "mania_activation": "D6",
         "substance": "D7"}
BLOCK_COLORS = {"overall_severity": "#444444", "cognition": "#0072B2", "immunometabolic": "#D55E00",
                "sleep": "#009E73", "suicidality": "#CC79A7", "developmental_risk": "#E69F00",
                "mania_activation": "#56B4E9", "substance": "#7E2F8E"}
CMAP = plt.get_cmap("viridis")
SUBTITLE = ("ALL model indicators · columns: G (burden backbone) + specific axes D1–D7   ·   "
            "ring = seed-ensemble CI ≠ 0   ·   shaded = depression/anxiety windows on G")


def load(path: Path) -> pd.DataFrame:
    L = LA.load_loadings(path)
    # Windows have an empty home in the VI export; place them in the G block (as the atlas expects).
    L.loc[L["home"].isin(["", "nan"]) | L["home"].isna(), "home"] = "overall_severity"
    L.loc[L["kind"] == "window", "home"] = "overall_severity"
    return L


def dot_atlas(L: pd.DataFrame, figdir: Path, name: str = "gllvm_dot_atlas") -> tuple[str, int]:
    """Complete atlas: every indicator that has a home (or window) loading, one row each."""
    rows = LA.atlas_rows(L, AXES, None)  # None = NO cap; show all variables
    n = len(rows)
    fig, ax = plt.subplots(figsize=(11.5, 0.165 * n + 2.5))
    sc = LA.draw_dot_atlas(ax, L, AXES, rows, cmap=CMAP, block_colors=BLOCK_COLORS,
                           axlab=AXLAB, axtag=AXTAG, g_key="overall_severity", subtitle=SUBTITLE,
                           label_fs=5.4)
    LA.atlas_legends(fig, ax, sc, face="#3b6")
    ax.set_title(f"V-GLLVM loading atlas — all {n} indicators", fontsize=12, pad=22)
    path = figdir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path), n


def phi_panel(phi_path: Path, figdir: Path, name: str = "gllvm_phi") -> str | None:
    if not phi_path.exists():
        return None
    phi = pd.read_csv(phi_path, index_col=0).loc[AXES, AXES]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(phi.to_numpy(), vmin=-0.6, vmax=0.6, cmap="RdBu_r")
    ax.set_xticks(range(len(AXES))); ax.set_yticks(range(len(AXES)))
    ax.set_xticklabels([f"{AXTAG[a]}·{AXLAB[a]}" for a in AXES], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([f"{AXTAG[a]}·{AXLAB[a]}" for a in AXES], fontsize=7)
    for i in range(len(AXES)):
        for j in range(len(AXES)):
            v = phi.to_numpy()[i, j]
            if i != j and abs(v) >= 0.02:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6,
                        color="#222" if abs(v) < 0.4 else "white")
    ax.set_title("Φ — inter-factor correlation (G + substance pinned ⊥)", fontsize=9, pad=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    path = figdir / f"{name}.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return str(path)


def lollipops(L: pd.DataFrame, figdir: Path, name: str = "gllvm_factor_lollipops") -> tuple[str, int]:
    """Per-factor lollipops showing ALL home indicators; panel heights ∝ indicator counts."""
    counts = {}
    for f in AXES:
        if f == "overall_severity":
            sub = L[(L["factor"] == f) & (L["kind"].isin(["g_anchor", "window"]))]
        else:
            sub = L[(L["home"] == f) & (L["factor"] == f) & (L["kind"] == "primary")]
        counts[f] = max(2, sub["item"].nunique())
    total = sum(counts.values())
    fig = plt.figure(figsize=(11, 0.165 * total + 4))
    gs = fig.add_gridspec(len(AXES), 1, height_ratios=[counts[f] for f in AXES], hspace=0.45)
    for i, f in enumerate(AXES):
        ax = fig.add_subplot(gs[i])
        LA.draw_lollipop(ax, L, f, color=BLOCK_COLORS[f], axlab=AXLAB, axtag=AXTAG,
                         g_key="overall_severity", top=999)  # 999 -> all home indicators
    fig.suptitle("V-GLLVM factor interpretability — ALL home indicators (|loading| + seed-ensemble CI)",
                 fontsize=12, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    path = figdir / f"{name}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path), total


def main() -> None:
    out = REPO / "results" / "face" / "gllvm_oop"
    figdir = REPO / "docs" / "figures" / "gllvm_oop"
    figdir.mkdir(parents=True, exist_ok=True)
    ens = out / "ensemble" / "loadings_summary.csv"
    src = ens if ens.exists() else out / "consolidate" / "loadings_summary.csv"
    phi = (out / "ensemble" / "phi.csv") if ens.exists() else (out / "consolidate" / "phi.csv")
    print(f"[atlas] loadings <- {src}", flush=True)
    L = load(Path(src))
    n_model = L["item"].nunique()

    atlas_path, n_atlas = dot_atlas(L, figdir)
    loll_path, n_loll = lollipops(L, figdir)
    phi_path = phi_panel(phi, figdir)

    made = [atlas_path, loll_path] + ([phi_path] if phi_path else [])
    print(f"[atlas] model indicators: {n_model} | atlas rows: {n_atlas} | lollipop indicators: {n_loll}", flush=True)
    # Completeness check: every model indicator must appear in the atlas.
    rows_items = {it for it, _ in LA.atlas_rows(L, AXES, None)}
    missing = set(L["item"]) - rows_items
    if missing:
        print(f"[atlas] WARNING: {len(missing)} indicators not shown: {sorted(missing)}", flush=True)
    else:
        print(f"[atlas] OK — all {n_model} model indicators are mapped in the atlas", flush=True)
    n_cred = int(L["excludes_zero"].fillna(False).sum())
    print(f"[atlas] {n_cred}/{len(L)} cells with CI ≠ 0", flush=True)
    print("[atlas] wrote:", flush=True)
    for p in made:
        print(f"  {Path(p).name}", flush=True)


if __name__ == "__main__":
    main()
