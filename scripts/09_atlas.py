#!/usr/bin/env python3
"""09 — the prior → posterior empirical atlas (§2.3), the manuscript centerpiece.

Side-by-side loading heatmaps: PRIOR (theory — the soft-prior means from configs/) vs POSTERIOR
(data — the reported map's loadings), for the continuous indicators (comparable z-scored scale),
rows grouped by home factor. Shows the 10 theoretical candidates being confirmed, reshaped, or dropped
by the FACE data. Verdicts come from docs/ADJUDICATION.md.

    python3 scripts/09_atlas.py

Writes docs/figures/empirical_atlas.png (+ empirical_atlas.csv).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
MATRIX = REPO / "configs" / "prior_loading_matrix_v3.csv"
POST = REPO / "reports" / "04_stage5_loadings.csv"
FIGS = REPO / "docs" / "figures"

FACTORS = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep",
           "developmental_risk", "suicidality"]
SHORT = {"overall_severity": "G", "cognition": "cog", "metabolic": "metab", "inflammatory": "inflam",
         "sleep": "sleep", "developmental_risk": "dev", "suicidality": "suic"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = pd.read_csv(MATRIX)
    post = pd.read_csv(POST)
    post = post[post.block == "continuous"]                       # comparable z-scored scale
    home = (m[m.prior_type.isin(["primary", "g_anchor"])].drop_duplicates("item")
            .set_index("item")["factor"].to_dict())

    # items: continuous, home in our 7 factors, present in the posterior map; grouped by home factor
    items = [it for it in post.item.unique() if home.get(it) in FACTORS]
    items = sorted(items, key=lambda it: (FACTORS.index(home[it]), it))
    fi = {f: i for i, f in enumerate(FACTORS)}

    prior = np.zeros((len(items), len(FACTORS)))
    poster = np.zeros((len(items), len(FACTORS)))
    pri_cell = {(r.item, r.factor): (r.prior_type, float(r.prior_mean)) for r in m.itertuples()}
    pos_cell = {(r.item, r.factor): float(r.loading) for r in post.itertuples()}
    for i, it in enumerate(items):
        for f, c in fi.items():
            pt_, mu = pri_cell.get((it, f), ("", 0.0))
            prior[i, c] = mu if pt_ in ("primary", "g_anchor", "plausible_cross") else 0.0
            poster[i, c] = pos_cell.get((it, f), 0.0)

    pd.DataFrame(poster, index=items, columns=FACTORS).to_csv(FIGS / "empirical_atlas.csv")

    # group boundaries (rows) for separators + factor labels on the y axis
    bounds, labels = [], []
    last = None
    for i, it in enumerate(items):
        if home[it] != last:
            bounds.append(i); labels.append((i, SHORT[home[it]])); last = home[it]

    fig, axes = plt.subplots(1, 2, figsize=(11, max(8, len(items) * 0.13)), sharey=True)
    for ax, data, title in [(axes[0], prior, "PRIOR (theory)"), (axes[1], poster, "POSTERIOR (data)")]:
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(FACTORS)))
        ax.set_xticklabels([SHORT[f] for f in FACTORS], rotation=45, ha="right", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        for b in bounds[1:]:
            ax.axhline(b - 0.5, color="k", lw=0.6)
        ax.set_yticks([])
    for b, lab in labels:                                         # factor group labels down the left
        axes[0].text(-1.4, b + 0.5, lab, fontsize=9, fontweight="bold", va="top", ha="right",
                     rotation=0, color="#333")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="loading")
    fig.suptitle("Prior → Posterior empirical atlas — 10 candidates: 7 confirmed (G + cognition, "
                 "metabolic/inflammatory split, sleep, developmental, suicidality),\n"
                 "anhedonia rejected, mania/substance deferred, impulsivity/negative/sensory "
                 "not-testable (§6)", fontsize=10, y=0.995)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "empirical_atlas.png", dpi=130, bbox_inches="tight")
    print(f"wrote docs/figures/empirical_atlas.png ({len(items)} continuous items × {len(FACTORS)} "
          f"factors) + empirical_atlas.csv")


if __name__ == "__main__":
    main()
