#!/usr/bin/env python3
"""23b — compare archetype solutions A=5,6,7,8 (which axis-corners survive at each A).

The scree has no elbow (continuum) ⇒ A is a parsimony/interpretability choice. This tabulates, for each
A, the extreme phenotypes (defining axis + peak z + population share + explained variance) and a
corner-survival matrix (does a metabolic / inflammatory / suicidality / mania / ... corner exist at each
A?), so the reported A can be chosen on interpretability. Arm A (full 9-d phenotype). Native z-scale.

    python3 scripts/23b_archetype_compare.py
Writes reports/23b_archetype_compare.md + docs/figures/23b_compare.png.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")
REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
M2 = REPO / "results" / "face" / "m2"
SEED = 20260609
CANON = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep", "mania_activation",
         "suicidality", "developmental_risk", "substance"]
AS = [5, 6, 7, 8]
POLE_THR = 0.8     # if no axis exceeds this (positive), the archetype is the low-burden pole / near-avg


def classify(z):
    mx = int(np.argmax(z)); mxv = float(z[mx])
    if mxv < POLE_THR:
        return ("low-burden pole" if z.mean() < -0.3 else "near-average"), mxv, mx
    return (f"↑{CANON[mx]}", mxv, mx)


def main():
    from face.strata.archetypes import explained_variance, fit_aa

    df = pd.read_parquet(M2 / "coordinates_full.parquet")
    X = df[[f"{f}__mean" for f in CANON]].to_numpy()

    sols = {}
    for A in AS:
        _, Z, W, rss = fit_aa(X, A, seed=SEED, n_init=4)
        dom = W.argmax(1)
        share = np.bincount(dom, minlength=A) / len(dom)
        rows = []
        for a in range(A):
            lab, peak, ax = classify(Z[a])
            rows.append({"arch": a, "defining": lab, "peak_z": round(peak, 2),
                         "share": round(float(share[a]), 3)})
        sols[A] = {"Z": Z, "ev": explained_variance(X, rss),
                   "table": pd.DataFrame(rows).sort_values("share", ascending=False)}
        print(f"A={A}: ev={sols[A]['ev']:.3f}", flush=True)

    # corner-survival matrix: rows = axes (+ low-burden pole), cols = A; cell = peak z of the corner
    # dominant on that axis (max over archetypes), '-' if none.
    surv = pd.DataFrame(index=CANON + ["low-burden pole"], columns=[f"A={A}" for A in AS], data="-")
    for A in AS:
        Z = sols[A]["Z"]
        for a in range(A):
            lab, peak, ax = classify(Z[a])
            key = "low-burden pole" if "pole" in lab or lab == "near-average" else CANON[ax]
            cur = surv.loc[key, f"A={A}"]
            if cur == "-" or peak > float(cur):
                surv.loc[key, f"A={A}"] = round(peak, 1)

    _fig(sols)

    md = ["# 23b — archetype-count comparison (A = 5, 6, 7, 8)", "",
          "Scree has no elbow (continuum) ⇒ A is a parsimony/interpretability choice; archetypes are stable "
          "at any A (M2.3 congruence 0.999). Below: which **axis-extreme corners** each A recovers, their "
          "peak z and population share. Higher = more burden. Arm A (full 9-d phenotype).", "",
          "## Corner-survival matrix (peak z of the corner dominant on each axis; '-' = absent)",
          "*The key question: at which A do the biology corners (metabolic, inflammatory) and the rare "
          "psychopathology tails (suicidality, mania) appear as their own phenotype?*",
          surv.to_markdown(), "",
          "## Explained variance", "| A | ev |", "|---|---|",
          *[f"| {A} | {sols[A]['ev']:.3f} |" for A in AS], ""]
    for A in AS:
        md += [f"## A = {A} — extreme phenotypes (by population share)",
               sols[A]["table"].to_markdown(index=False), ""]
    md += ["## Reading",
           "- A corner 'survives' when an archetype sits at that axis's positive extreme. Smaller A merges "
           "the rarer tails into neighbours; larger A resolves them as their own phenotype.",
           "- Pick the smallest A that still resolves the corners you care about (esp. **metabolic + "
           "inflammatory** for the biology⊥G story). The choice is interpretability, not fit — every A is a "
           "valid soft basis for the same continuum.",
           "", "Figure: `docs/figures/23b_compare.png` (profile heatmaps, A=5..8)."]
    (REPORTS / "23b_archetype_compare.md").write_text("\n".join(md))
    print("\n".join(md))


def _fig(sols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    for ax, A in zip(axes.flat, AS, strict=False):
        Z = sols[A]["Z"]
        labs = [f"{classify(Z[a])[0]}" for a in range(A)]
        im = ax.imshow(Z, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
        ax.set_xticks(range(len(CANON))); ax.set_xticklabels(CANON, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(A)); ax.set_yticklabels(labs, fontsize=8)
        for i in range(A):
            for j in range(len(CANON)):
                ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(Z[i, j]) > 1.4 else "black")
        ax.set_title(f"A = {A}  (ev {sols[A]['ev']:.2f})")
    fig.suptitle("M2.3b — archetype profiles across A (which axis-corners survive)", y=1.0, fontsize=14)
    fig.colorbar(im, ax=axes, shrink=0.5, label="archetype coordinate (z; higher = more burden)")
    fig.savefig(FIGS / "23b_compare.png", dpi=120, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
