"""V3 — visualize the extended measurement model (aggregate figures only).

Reads the extended-model outputs (results/v3/bayesian_ext/) and writes publication-style figures to
docs/figures/v3/ (tracked — all AGGREGATE: a correlation heatmap, a correlation network, a loadings
chart, and per-cohort score distributions; never per-patient rows).

Run (after scripts/v3/04_extended_model.py):  python3 scripts/v3/05_visualize.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "results" / "v3" / "bayesian_ext"
FIG = ROOT / "docs" / "figures" / "v3"
COLORS = {"cognition": "#4E79A7", "metabolic": "#F28E2B", "inflammatory": "#E15759",
          "sleep": "#76B7B2", "affective": "#59A14F", "suicidality": "#B07AA1"}


def heatmap(C: pd.DataFrame):
    labels = list(C.columns)
    M = C.to_numpy().copy()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if abs(M[i, j]) > 0.45 else "black", fontsize=9)
    ax.set_title("Factor correlation matrix Φ\n(5 latent dimensions + suicidality)", fontsize=11)
    fig.colorbar(im, fraction=0.046, pad=0.04, label="correlation")
    fig.tight_layout(); fig.savefig(FIG / "phi_heatmap.png", dpi=140); plt.close(fig)


def network(C: pd.DataFrame, thresh=0.10):
    labels = list(C.columns); n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    xy = np.column_stack([np.cos(ang), np.sin(ang)])
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    for i in range(n):
        for j in range(i + 1, n):
            r = C.iloc[i, j]
            if abs(r) >= thresh:
                ax.plot(*zip(xy[i], xy[j], strict=False), lw=0.6 + 6 * abs(r),
                        color="#C44E52" if r > 0 else "#4C72B0", alpha=0.55, zorder=1)
                mx, my = (xy[i] + xy[j]) / 2
                ax.text(mx, my, f"{r:.2f}", fontsize=8, ha="center", va="center",
                        color="#333", zorder=3,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    for k, lab in enumerate(labels):
        ax.scatter(*xy[k], s=1500, color=COLORS.get(lab, "#888"), zorder=2, edgecolors="white", linewidths=2)
        ax.text(*(xy[k] * 1.34), lab, ha="center", va="center", fontsize=10.5,
                color=COLORS.get(lab, "#333"), weight="bold", zorder=4)
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.axis("off")
    ax.set_title(f"Dimension correlation network (|r| ≥ {thresh})\n"
                 "thin/weak edges ⇒ no general factor", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "correlation_network.png", dpi=140); plt.close(fig)


def loadings(L: pd.DataFrame):
    L = L.copy()
    order = ["cognition", "metabolic", "inflammatory", "sleep", "affective", "suicidality"]
    L["__o"] = L["factor"].map({f: i for i, f in enumerate(order)})
    L = L.sort_values(["__o", "loading"])
    fig, ax = plt.subplots(figsize=(6.4, 8.2))
    ax.barh(range(len(L)), L["loading"], color=[COLORS.get(f, "#888") for f in L["factor"]])
    ax.set_yticks(range(len(L))); ax.set_yticklabels(L["indicator"], fontsize=7.5)
    ax.set_xlabel("standardized loading (higher value = more burden)")
    ax.set_title("Measurement model — indicator loadings by dimension", fontsize=11)
    ax.axvline(0, color="#999", lw=0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[f]) for f in order if f in set(L["factor"])]
    ax.legend(handles, [f for f in order if f in set(L["factor"])], fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "loadings.png", dpi=140); plt.close(fig)


def scores_by_cohort(sc: pd.DataFrame):
    facs = [c for c in sc.columns if c.startswith("F_")]
    cohorts = ["bp", "sz", "dr"]
    fig, axes = plt.subplots(1, len(facs), figsize=(2.05 * len(facs), 3.6), sharey=True)
    for ax, f in zip(axes, facs, strict=False):
        data = [sc.loc[sc["cohort"] == c, f].dropna().to_numpy() for c in cohorts]
        bp = ax.boxplot(data, labels=[c.upper() for c in cohorts], patch_artist=True, showfliers=False, widths=0.6)
        for patch, c in zip(bp["boxes"], cohorts, strict=False):
            patch.set_facecolor({"bp": "#4E79A7", "sz": "#E15759", "dr": "#59A14F"}[c]); patch.set_alpha(0.7)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_title(f.replace("F_", ""), fontsize=9.5); ax.axhline(0, color="#bbb", lw=0.6)
    axes[0].set_ylabel("posterior factor score (higher = more burden)")
    fig.suptitle("Dimension scores by diagnostic cohort (validation, not a clustering feature)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "scores_by_cohort.png", dpi=140); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    C = pd.read_csv(SRC / "phi.csv", index_col=0)
    L = pd.read_csv(SRC / "loadings.csv")
    heatmap(C); network(C); loadings(L)
    f = SRC / "factor_scores.csv"
    if f.exists():
        scores_by_cohort(pd.read_csv(f))
    print("wrote figures ->", FIG.relative_to(ROOT))
    for p in sorted(FIG.glob("*.png")):
        print("  ", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
