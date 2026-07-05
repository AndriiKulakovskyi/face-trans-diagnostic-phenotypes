"""Figures for the M4 representation benchmark — reads results/analyses/representation_benchmark/*.csv, writes PNGs to
docs/figures/repbench/ and report/figures/. Run after the P1/P2 drivers.

    PYTHONPATH=$PWD/src python notebooks/repbench_make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "face" / "m4_repbench"
DOCS = ROOT / "docs" / "figures" / "repbench"
REPORT = ROOT / "report" / "figures"
for d in (DOCS, REPORT):
    d.mkdir(parents=True, exist_ok=True)

C = {"REF": "#8A8A86", "REF+LAT-A": "#2B4C8C", "REF+RAW": "#B45309"}      # arm colours
LABEL = {"REF": "REF (clinician)", "REF+LAT-A": "REF + 9-dim map", "REF+RAW": "REF + raw 143"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.dpi": 150, "savefig.facecolor": "white"})


def _save(fig, name):
    fig.tight_layout()
    for d in (DOCS, REPORT):
        fig.savefig(d / name, bbox_inches="tight")
    plt.close(fig)


def fig_sufficiency():
    s = pd.read_csv(RES / "scalar.csv")
    s = s[s.scope == "pooled"]
    arms = ["REF", "REF+LAT-A", "REF+RAW"]
    cells = [("egf_recovery", "V1"), ("egf_recovery", "V2"),
             ("egf_deterioration", "V1"), ("egf_deterioration", "V2")]
    labels = ["recovery\nV1", "recovery\nV2", "deterior.\nV1", "deterior.\nV2"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(cells))
    w = 0.26
    for i, arm in enumerate(arms):
        vals = [s[(s.target == t) & (s.horizon == h) & (s.arm == arm)]["auc"].iloc[0] for t, h in cells]
        ax.bar(x + (i - 1) * w, vals, w, color=C[arm], label=LABEL[arm])
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("out-of-fold AUC")
    ax.set_ylim(0.45, 0.80)
    ax.set_title("Sufficiency: raw vs the 9-dim map (pooled)\nrecovery — raw edges +0.04;  deterioration — tie")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    _save(fig, "repbench_sufficiency.png")


def fig_recovery_gap():
    df = pd.read_csv(RES / "recovery_gap_shap_V2.csv")
    raw = df[df.block == "raw"].copy()
    on = raw[raw.home_factor.notna()]["mean_abs_shap"].sum()
    off = raw[raw.home_factor.isna()]["mean_abs_shap"].sum()
    frac_on = on / (on + off)
    top = raw.sort_values("mean_abs_shap", ascending=False).head(12).iloc[::-1]

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1, 1.7]})
    a0.bar([0, 1], [frac_on, 1 - frac_on], color=["#2B4C8C", "#B45309"], width=0.6)
    a0.set_xticks([0, 1])
    a0.set_xticklabels(["within the\n9 factors", "off-map\n(windows)"])
    a0.set_ylabel("share of raw recovery SHAP mass")
    a0.set_ylim(0, 1)
    for xi, v in zip([0, 1], [frac_on, 1 - frac_on], strict=True):
        a0.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=11)
    a0.set_title("Raw's recovery edge is\nwithin-factor compression")

    cols = top["home_factor"].fillna("off-map")
    palette = {f: plt.cm.tab10(i) for i, f in enumerate(sorted(cols.unique()))}
    a1.barh(range(len(top)), top["mean_abs_shap"], color=[palette[f] for f in cols])
    a1.set_yticks(range(len(top)))
    a1.set_yticklabels(top["feature"], fontsize=8)
    a1.set_xlabel("mean |SHAP|")
    a1.set_title("Top recovery predictors (raw) — coloured by home factor")
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[f]) for f in palette]
    a1.legend(handles, list(palette), fontsize=7, loc="lower right", frameon=False)
    _save(fig, "repbench_recovery_gap.png")


def fig_learning_curve():
    lc = pd.read_csv(RES / "learning_curve_recovery_V2.csv")
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for arm in ["REF", "REF+LAT-A", "REF+RAW"]:
        g = lc[lc.arm == arm].sort_values("N")
        ax.plot(g["N"], g["mean"], "-o", color=C[arm], label=LABEL[arm], ms=4)
        ax.fill_between(g["N"], g["mean"] - g["std"], g["mean"] + g["std"], color=C[arm], alpha=0.12)
    ax.set_xlabel("training-set size N")
    ax.set_ylabel("AUC (recovery, pooled V2)")
    ax.set_title("Efficiency: no small-N advantage for the map\n(raw dominates at every N)")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    _save(fig, "repbench_learning_curve.png")


def fig_loco():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    arms = ["REF", "REF+LAT-A", "REF+RAW"]
    for ax, tgt, ttl in zip(axes, ("egf_recovery", "egf_deterioration"),
                            ("recovery", "deterioration"), strict=True):
        lo = pd.read_csv(RES / f"loco_{tgt}_V2.csv")
        lo = lo[lo.held_out.isin(["bp", "sz"])]                 # well-powered held-out cohorts
        held = ["bp", "sz"]
        x = np.arange(len(held))
        w = 0.26
        for i, arm in enumerate(arms):
            vals = [lo[(lo.held_out == h) & (lo.arm == arm)]["auc_oos"].iloc[0] for h in held]
            ax.bar(x + (i - 1) * w, vals, w, color=C[arm], label=LABEL[arm] if ax is axes[0] else None)
        ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([f"held-out\n{h.upper()}" for h in held])
        ax.set_title(ttl)
        ax.set_ylim(0.45, 0.80)
    axes[0].set_ylabel("out-of-sample AUC")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Transport (LOCO): the map transfers as well/better than raw for deterioration", y=1.02)
    _save(fig, "repbench_loco.png")


def main():
    fig_sufficiency()
    fig_recovery_gap()
    fig_learning_curve()
    fig_loco()
    print("wrote 4 figures to", DOCS, "and", REPORT)


if __name__ == "__main__":
    main()
