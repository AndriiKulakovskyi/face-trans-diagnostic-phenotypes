#!/usr/bin/env python3
"""Exploratory geometry of the Gaussian-copula map — DATA DISCOVERY, not archetype confirmation.

Project all 9,013 patients (9-dim copula coordinates) to 2D and render: (1) smooth density / "energy"
landscapes (PCA + UMAP), (2) a per-axis gradient atlas (which directions organize the cloud), and
(3) a candidate-extremes panel (where each axis's extreme patients sit — do the extremes pile into a few
directions = candidate corners, or spread continuously?). Deliberately does NOT presuppose the A=4
archetypes; they are over-plotted only as a faint reference.

HONESTY (read every figure with this): a 2D projection of a 9-D cloud DISTORTS — proximity in 2D ≠
proximity in 9-D, and UMAP in particular can manufacture apparent islands. The M2 single-Gaussian
falsification null already says there are NO well-separated clusters, so the density map *should* be one
smooth, unimodal mass; treat any apparent "peak" as a hypothesis to TEST (structure gate), never as a kind.

    PYTHONPATH=$PWD/src python notebooks/archetype_geometry/explore_landscape.py
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "notebooks" / "archetype_geometry"
COORDS = REPO / "results" / "face" / "strata_oop" / "coordinates" / "coordinates_full.parquet"
PROFILES = REPO / "results" / "face" / "strata_oop" / "consolidate" / "archetype_profiles.csv"
AXES = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep",
        "mania_activation", "suicidality", "developmental_risk", "substance"]
PRETTY = {"overall_severity": "severity (G)", "cognition": "cognition", "metabolic": "metabolic",
          "inflammatory": "inflammatory", "sleep": "sleep", "mania_activation": "mania",
          "suicidality": "suicidality", "developmental_risk": "developmental", "substance": "substance"}
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list(
    "face", ["#f7f7f7", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
ACCENT, INK, MUTE = "#2B4C8C", "#14181F", "#5B6573"


def load():
    d = pd.read_parquet(COORDS)
    X = d[[f"{a}__mean" for a in AXES]].to_numpy("float64")
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return d, X, (X - mu) / sd, mu, sd


def embeds(Xs):
    Xc = Xs - Xs.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca = Xc @ Vt[:2].T
    ev = (S ** 2) / (S ** 2).sum()
    import umap
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.30, random_state=0, metric="euclidean")
    um = reducer.fit_transform(Xs)
    return pca, ev, um, Vt, reducer


def density_grid(emb, bins=170, smooth=3.2):
    h, xe, ye = np.histogram2d(emb[:, 0], emb[:, 1], bins=bins)
    h = gaussian_filter(h, smooth)
    return h.T, [xe[0], xe[-1], ye[0], ye[-1]]


def _land(ax, emb, title, *, cmap="magma"):
    g, ext = density_grid(emb)
    ax.imshow(g, origin="lower", extent=ext, aspect="auto", cmap=cmap)
    ax.contour(g, levels=7, extent=ext, colors="white", linewidths=0.4, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10, color=INK)


def _grad(ax, emb, val, title):
    o = np.argsort(np.abs(val - np.median(val)))            # plot extreme points last (on top)
    sc = ax.scatter(emb[o, 0], emb[o, 1], c=val[o], cmap=SEQ, s=3, alpha=0.55,
                    linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9.5, color=INK)
    return sc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d, X, Xs, mu, sd = load()
    pca, ev, um, Vt, reducer = embeds(Xs)
    g = d["overall_severity__mean"].to_numpy("float64")     # severity = the spine
    print(f"loaded {len(X):,} patients · PCA var {ev[0]:.0%}/{ev[1]:.0%}", flush=True)

    # ---- Figure 1: the energy landscape (PCA + UMAP density) + severity & biology gradients ----
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    _land(ax[0, 0], pca, f"PCA density landscape (variance-faithful · {ev[0]:.0%}+{ev[1]:.0%} var)")
    _land(ax[0, 1], um, "UMAP density landscape (proximity · can fake islands)")
    s1 = _grad(ax[1, 0], pca, g, "PCA tinted by severity (G) — the spine")
    fig.colorbar(s1, ax=ax[1, 0], fraction=0.046).ax.tick_params(labelsize=7)
    s2 = _grad(ax[1, 1], pca, d["inflammatory__mean"].to_numpy("float64"),
               "PCA tinted by inflammatory — a crossing gradient")
    fig.colorbar(s2, ax=ax[1, 1], fraction=0.046).ax.tick_params(labelsize=7)
    fig.suptitle("The copula map as a landscape — is it one smear, or are there sub-populations?",
                 y=0.99, fontsize=12, fontweight="bold", color=ACCENT)
    fig.text(0.5, 0.005, "Smooth density (brighter = more patients). A single broad peak = continuum "
             "(consistent with the single-Gaussian null). Read as SHAPE, not kinds.",
             ha="center", fontsize=8, color=MUTE)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(OUT / "landscape_density.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote landscape_density.png", flush=True)

    # ---- Figure 2: gradient atlas — every axis over the (UMAP) map ----
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for axn, a in zip(axes.ravel(), AXES):
        _grad(axn, um, d[f"{a}__mean"].to_numpy("float64"), PRETTY[a])
    fig.suptitle("Gradient atlas: each of the 9 axes over the map (UMAP). Smooth gradients (not patches) "
                 "= a continuum organized by crossing directions", y=0.995, fontsize=12,
                 fontweight="bold", color=ACCENT)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "gradient_atlas.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote gradient_atlas.png", flush=True)

    # ---- Figure 3: candidate extremes — where does each axis's extreme phenotype sit? ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    pal = plt.cm.tab10(np.linspace(0, 1, len(AXES)))
    for emb, ax, name in [(pca, axes[0], "PCA"), (um, axes[1], "UMAP")]:
        gd, ext = density_grid(emb)
        ax.imshow(gd, origin="lower", extent=ext, aspect="auto", cmap="Greys", alpha=0.55)
        for j, a in enumerate(AXES):                          # top-1.5% on each axis (the + extreme)
            v = d[f"{a}__mean"].to_numpy("float64")
            thr = np.quantile(v, 0.985)
            m = v >= thr
            ax.scatter(emb[m, 0], emb[m, 1], s=10, color=pal[j], alpha=0.8, linewidths=0,
                       label=PRETTY[a] if name == "PCA" else None)
        # A=4 archetype corners (faint reference, NOT the lead here)
        prof = pd.read_csv(PROFILES)
        prof = prof[prof["arm"] == "A_all9"][AXES].to_numpy("float64")
        cz = (prof - mu) / sd
        cz2 = (cz - Xs.mean(0)) @ Vt[:2].T if name == "PCA" else reducer.transform(cz)
        ax.scatter(cz2[:, 0], cz2[:, 1], s=320, marker="*", facecolor="none",
                   edgecolor="black", linewidths=1.6, zorder=6)
        for k, (xx, yy) in enumerate(cz2):
            ax.annotate(f"A{k}", (xx, yy), fontsize=9, fontweight="bold", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{name}: per-axis extremes (top 1.5%)",
                                                           fontsize=10, color=INK)
    axes[0].legend(loc="upper left", fontsize=7, frameon=False, markerscale=1.4, ncol=1)
    fig.suptitle("Candidate corners: where do the extremes of each axis live? Overlapping colours = one "
                 "shared corner; separated colours = distinct candidate phenotypes (★ = the A=4 fit)",
                 y=0.99, fontsize=11.5, fontweight="bold", color=ACCENT)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "candidate_extremes.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote candidate_extremes.png", flush=True)
    print(f"\ndone -> {OUT.relative_to(REPO)}/{{landscape_density,gradient_atlas,candidate_extremes}}.png")


if __name__ == "__main__":
    main()
