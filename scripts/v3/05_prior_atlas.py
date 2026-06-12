#!/usr/bin/env python3
"""Prior atlas — visualize the soft-prior loading matrix (the THEORY half of the map).

    python3 scripts/v3/05_prior_atlas.py

Reads  configs/prior_loading_matrix_v3.csv
Writes docs/figures/prior_atlas.png            (indicator x factor heatmap)
       docs/figures/prior_atlas_by_factor.csv  (per-factor summary)

This is the *prior* (theoretical) loading atlas: where each instrument is EXPECTED to load
(primary / G-anchor), MAY load (plausible cross), or is suppressed (soft-zero), per the
candidate ontology in configs/dimensions.yaml. It is the "before" of the prior -> posterior
comparison (docs/MEASUREMENT_MODEL.md §2.3); the empirical atlas (posterior loadings, with
uncertainty) is produced after the global model is fit. No patient data is involved — this is
pure config, safe to share/commit.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MATRIX = REPO / "configs" / "prior_loading_matrix_v3.csv"
FIGDIR = REPO / "docs" / "figures"

FACTOR_ORDER = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep",
                "suicidality", "developmental_risk", "anhedonia", "mania_activation", "substance"]
FACTOR_LABEL = dict(zip(FACTOR_ORDER,
    ["G: severity / burden", "cognition", "metabolic", "inflammatory", "sleep",
     "suicidality", "developmental-risk", "anhedonia", "mania-activation", "substance"]))


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(MATRIX)

    # prior-permitted loading magnitude = |mean| + sd  (config-derived: primary ~0.9,
    # plausible-cross ~0.25, unlikely/soft-zero ~0.05, g_anchor_on_specific ~0).
    m["strength"] = m["prior_mean"].abs() + m["prior_sd"]

    # home factor (for row ordering) = where the item is primary / g_anchor
    home = (m[m.prior_type.isin(["primary", "g_anchor"])]
            .drop_duplicates("item").set_index("item")["factor"].to_dict())
    items_all = list(dict.fromkeys(m["item"]))
    cross = [it for it in items_all if it not in home]            # cross-loading windows (no home)

    def rank(it: str) -> tuple:
        h = home.get(it)
        return (FACTOR_ORDER.index(h) if h in FACTOR_ORDER else 99, it)
    row_order = sorted(home, key=rank) + sorted(cross)

    P = (m.pivot(index="item", columns="factor", values="strength")
           .reindex(index=row_order, columns=FACTOR_ORDER))

    n = len(row_order)
    fig, ax = plt.subplots(figsize=(7.6, max(8.0, n * 0.135)))
    cmap = LinearSegmentedColormap.from_list(
        "prior", ["#f7f7f7", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
    im = ax.imshow(P.values, aspect="auto", cmap=cmap, vmin=0.0, vmax=0.95)

    ax.set_xticks(range(len(FACTOR_ORDER)))
    ax.set_xticklabels([FACTOR_LABEL[f] for f in FACTOR_ORDER], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(row_order, fontsize=4.4)
    ax.tick_params(length=0)

    # white separators between factor row-blocks
    prev, bounds = object(), []
    for i, it in enumerate(row_order):
        h = home.get(it, "__cross__")
        if h != prev:
            bounds.append(i); prev = h
    for b in bounds[1:]:
        ax.axhline(b - 0.5, color="white", lw=1.4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("prior-permitted loading  (|mean| + sd)", fontsize=8)
    ax.set_title("FACE V0 — prior loading atlas (theory: soft-prior map)\n"
                 "primary / G-anchor (dark) · plausible cross-loading (mid) · soft-zero (light)",
                 fontsize=9)
    fig.tight_layout()
    out = FIGDIR / "prior_atlas.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out.relative_to(REPO))

    # ---- per-factor summary (primary indicators + likelihood mix) ----
    prim = m[m.prior_type.isin(["primary", "g_anchor"])].drop_duplicates("item")
    pc = (m[m.prior_type == "plausible_cross"].groupby("factor")["item"].nunique())
    rows = []
    for f in FACTOR_ORDER:
        pf = prim[prim.factor == f]
        liks = dict(pf["likelihood_family"].value_counts())
        rows.append({"factor": f, "n_primary": len(pf),
                     "n_plausible_cross_in": int(pc.get(f, 0)),
                     "likelihoods": "; ".join(f"{k}:{v}" for k, v in liks.items())})
    summ = pd.DataFrame(rows)
    summ.to_csv(FIGDIR / "prior_atlas_by_factor.csv", index=False)
    xl = sorted(cross)
    print("\nper-factor primary indicators:")
    print(summ.to_string(index=False))
    print(f"\ncross-loading windows ({len(xl)}):", xl)
    print("total modeled indicators:", len(items_all))


if __name__ == "__main__":
    main()
