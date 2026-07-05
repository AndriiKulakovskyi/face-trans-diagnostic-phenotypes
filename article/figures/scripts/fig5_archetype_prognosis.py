#!/usr/bin/env python
"""
Figure 5 (money figure): archetype-prognosis atlas.

Panel (a): all N=9,013 patients projected onto the five-archetype simplex as
barycentric blends of their archetype weights, coloured by dominant archetype.
Panel (b): the same simplex, the subset with two-year follow-up shaded by a
smoothed functional-remission field (hexbin bin-means) — the immunometabolic
pole (A2, upper-right) is the worst-prognosis corner (22%) and the low-burden
pole (A4) the best (63%).

Data source (per-patient archetype weights AND two-year outcomes together, so
no fragile join is needed):
    results/m4_prognosis/consolidate/prognosis_patient_risk.parquet

Output: article/figures/fig5_archetype_prognosis.{png,pdf}
Maps to manuscript Figure 5 (article_v2, fig:money).

Computed in-session (2026-07-03) from the frozen prognosis consolidation; this
script reproduces that figure exactly. Run with the repo venv:
    ../.venv/bin/python figures/scripts/fig5_archetype_prognosis.py
"""
import os

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# repo root = two levels up from this file (article/figures/scripts/ -> repo)
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
OUT = os.path.join(ROOT, "article", "figures")
PARQUET = os.path.join(ROOT, "results", "m4_prognosis",
                       "consolidate", "prognosis_patient_risk.parquet")


# Global state
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.linewidth": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.grid": False,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.labelweight": "normal",
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


pr = pd.read_parquet(PARQUET)

W = pr[[f"arch_w{i}" for i in range(5)]].to_numpy(float)
W = W / W.sum(1, keepdims=True)

order = [3, 2, 1, 4, 0]
names = {0: "A0 activation /\nsleep", 1: "A1 severe,\nlow-biology", 2: "A2 ↑immunometabolic\n(biology pole)",
         3: "A3 trauma /\nsuicidality", 4: "A4 low-burden /\nwell"}
base = np.array([-18, 54, 126, 198, 270])
slot_of = {idx: k for k, idx in enumerate([2, 1, 0, 4, 3])}
corner = {}
for idx in range(5):
    a = np.deg2rad(base[slot_of[idx]])
    corner[idx] = np.array([np.cos(a), np.sin(a)])
C = np.array([corner[i] for i in range(5)])
XY = W @ C

rem = pd.to_numeric(pr["egf__remission_V2"], errors="coerce").to_numpy()
has = np.isfinite(rem)
dom = W.argmax(1)
domcol = {0: "#B42318", 1: "#6D28D9", 2: "#0F766E", 3: "#B45309", 4: "#47556A"}
overall_rem = np.nanmean(rem) * 100

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.0, 6.5))


def draw_pentagon(ax, a2_rad=1.16):
    poly = np.vstack([C, C[0]])
    ax.plot(poly[:, 0], poly[:, 1], color="#C7CED9", lw=1.0, ls="--", zorder=1)
    for idx in range(5):
        rad = a2_rad if idx == 2 else 1.16
        x, y = C[idx] * rad
        ax.text(x, y, names[idx], ha="center", va="center", fontsize=7.3, fontweight="bold",
                color=domcol[idx], zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=domcol[idx], lw=0.9))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.6, 1.42)
    ax.set_ylim(-1.62, 1.6)


draw_pentagon(axA)
for idx in range(5):
    m = dom == idx
    axA.scatter(XY[m, 0], XY[m, 1], s=4, c=domcol[idx], alpha=0.35, lw=0, zorder=3)
axA.set_title("a   All 9,013 patients are blends of five archetypes", loc="left", fontsize=10.5, fontweight="bold")
axA.text(0.0, -1.52, "one point = one patient · position = archetype-weight blend", ha="center", fontsize=7.2, color="#5B6573")

draw_pentagon(axB, a2_rad=1.03)
norm = mcolors.TwoSlopeNorm(vcenter=overall_rem, vmin=15, vmax=70)
hb = axB.hexbin(XY[has, 0], XY[has, 1], C=rem[has] * 100, reduce_C_function=np.mean,
                gridsize=17, mincnt=5, cmap="RdBu", norm=norm, zorder=2, edgecolors="white", linewidths=0.2)
cb = fig.colorbar(hb, ax=axB, fraction=0.045, pad=0.15)
cb.set_label("2-year functional remission (%)  ·  bin mean", fontsize=8)
cb.ax.tick_params(labelsize=7)
axB.text(*(C[2] * 1.20 + np.array([0.0, -0.26])), "22%", ha="center", va="center", fontsize=10, fontweight="bold", color="#B2182B", zorder=8)
axB.text(*(C[4] * 1.16 + np.array([0.0, -0.26])), "63%", ha="center", va="center", fontsize=10, fontweight="bold", color="#2166AC", zorder=8)
axB.set_title("b   The immunometabolic pole is the worst-prognosis corner", loc="left", fontsize=10.5, fontweight="bold")
axB.text(0.0, -1.52, f"n = {int(has.sum()):,} with 2-year follow-up · bins ≥5 patients · what a diagnosis or severity score cannot see",
         ha="center", fontsize=7.0, color="#5B6573")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_archetype_prognosis.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig5_archetype_prognosis.pdf"), bbox_inches="tight")
print("wrote", os.path.join(OUT, "fig5_archetype_prognosis.png"))
