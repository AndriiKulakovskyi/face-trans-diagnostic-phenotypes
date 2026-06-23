#!/usr/bin/env python3
"""Visualize the N×9 patient-coordinate cloud (the copula M2 map) as a smooth 2D density, with the A=4
archetype corners overlaid — to SEE whether the cloud is one continuous blob with corners at the rim
(archetypal-analysis story) rather than separated clumps (clusters).

Two projections, side by side:
  - PCA shadow of the cloud  (honest linear projection: preserves convex geometry as a 2D shadow);
  - the archetype plane      (the 2D plane best separating the 4 corners, so all four are visible).

For each, a Gaussian-smoothed 2D density (the "energy"/proximity surface) + the 4 archetype corners
(projected the same way) + the cloud centroid. A second row colours patients by their dominant archetype
(soft assignment), so you can see the corners "pull" the population.

    PYTHONPATH=$PWD/src python notebooks/archetype_geometry/visualize_cloud.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from face.strata.archetypes import project_to_Z  # noqa: E402

S = REPO / "results" / "face" / "strata_oop"
OUT = REPO / "docs" / "figures" / "archetype_geometry"
OUT.mkdir(parents=True, exist_ok=True)
AXES = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep",
        "mania_activation", "suicidality", "developmental_risk", "substance"]
NAMES = ["A0 · biological", "A1 · low-burden", "A2 · severe (non-bio)", "A3 · symptom"]
PAL = {0: "#0F766E", 1: "#2B4C8C", 2: "#B42318", 3: "#6D28D9"}     # biology / low-burden / severe / symptom


def _density(p2, bins=170, sigma=3.0, pad=1.6):
    xlo, xhi = p2[:, 0].min() - pad, p2[:, 0].max() + pad
    ylo, yhi = p2[:, 1].min() - pad, p2[:, 1].max() + pad
    H, xe, ye = np.histogram2d(p2[:, 0], p2[:, 1], bins=bins, range=[[xlo, xhi], [ylo, yhi]])
    return gaussian_filter(H.T, sigma=sigma), 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])


def _panel_density(ax, p2, z2, ev, title):
    Hs, xc, yc = _density(p2)
    ax.contourf(xc, yc, Hs, levels=16, cmap="magma")
    ax.scatter(p2[:, 0].mean(), p2[:, 1].mean(), s=60, marker="o", facecolor="white",
               edgecolor="black", lw=0.8, zorder=5, label="centroid")
    for a in range(z2.shape[0]):
        ax.scatter(*z2[a], s=240, marker="*", facecolor="white", edgecolor="black", lw=1.0, zorder=6)
        ha = "right" if z2[a, 0] > p2[:, 0].mean() else "left"
        dx = -8 if ha == "right" else 8
        ax.annotate(NAMES[a], z2[a], color="white", fontsize=9, fontweight="bold",
                    xytext=(dx, 7), textcoords="offset points", ha=ha, zorder=7,
                    annotation_clip=False)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if ev is not None:
        ax.set_xlabel(f"axis 1 ({ev[0]:.0%} var)"); ax.set_ylabel(f"axis 2 ({ev[1]:.0%} var)")
    ax.set_xticks([]); ax.set_yticks([])


def _panel_dom(ax, p2, z2, dom, title):
    for k in range(4):
        m = dom == k
        ax.scatter(p2[m, 0], p2[m, 1], s=3, c=PAL[k], alpha=0.30, linewidths=0,
                   rasterized=True, label=f"{NAMES[k]} ({m.mean():.0%})")
    for a in range(z2.shape[0]):
        ax.scatter(*z2[a], s=260, marker="*", facecolor=PAL[a], edgecolor="black", lw=1.1, zorder=6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.6, markerscale=2.5, loc="best")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    df = pd.read_parquet(S / "coordinates" / "coordinates_full.parquet")
    X = df[[f"{a}__mean" for a in AXES]].to_numpy("float64")
    ap = pd.read_csv(S / "consolidate" / "archetype_profiles.csv")
    Z = ap[ap.arm == "A_all9"].sort_values("archetype")[AXES].to_numpy("float64")   # [4,9]
    print(f"X {X.shape} · Z {Z.shape}")

    # dominant archetype via simplex projection (native scale, as the canonical fit)
    W = project_to_Z(X, Z)
    dom = W.argmax(1)
    blends = float((W.max(1) < 0.5).mean())
    print("dominant shares:", {NAMES[k]: round(float((dom == k).mean()), 3) for k in range(4)},
          "| blends (max w<0.5):", round(blends, 3))

    mu = X.mean(0); Xc = X - mu

    # (1) PCA shadow of the cloud
    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcaP = Xc @ Vt[:2].T;  pcaZ = (Z - mu) @ Vt[:2].T
    ev = (sv ** 2) / (sv ** 2).sum()

    # (2) archetype plane: 2 principal directions of the 4 (centred) archetype locations
    Zc = Z - Z.mean(0)
    _, _, Va = np.linalg.svd(Zc, full_matrices=False)
    planeP = Xc @ Va[:2].T;  planeZ = (Z - mu) @ Va[:2].T

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.5))
    _panel_density(axes[0, 0], pcaP, pcaZ, ev, "Patient density — PCA shadow of the 9-dim cloud")
    _panel_density(axes[0, 1], planeP, planeZ, None, "Patient density — the archetype plane (corners separated)")
    _panel_dom(axes[1, 0], pcaP, pcaZ, dom, "Soft archetype assignment — PCA shadow")
    _panel_dom(axes[1, 1], planeP, planeZ, dom, "Soft archetype assignment — archetype plane")
    fig.suptitle(f"The transdiagnostic continuum and its A=4 corners  (N={len(X):,}; "
                 f"{blends:.0%} of patients are blends)", y=0.995, fontsize=13,
                 fontweight="bold", color="#1E366B")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    p = OUT / "cloud_archetypes.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", p.relative_to(REPO))


if __name__ == "__main__":
    main()
