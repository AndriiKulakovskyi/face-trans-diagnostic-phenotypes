"""
FACE-ATLAS methods paper — Fig 5: posterior-coordinate calibration.

Two panels, both instrument-forward (the map is a MEASUREMENT INSTRUMENT):
  LEFT  — empirical vs nominal credible level (mean over 8 axes + joint 8-D
          ellipsoid + the 8 individual axes threaded by the house palette),
          against the y=x identity ("perfect calibration").
  RIGHT — 95% interval coverage stratified by #observed home-items on the axis
          (pooled across axes), with ±2 SE tolerance band and per-bin binomial
          error bars; the 0-item bin is the prior-returned correctness check.

Reads the frozen analysis products in article_methods/analysis/. Run from
article_methods/figures/ so the relative savefig lands there.
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = os.path.expanduser("~/Desktop/face-common-bp-sz-dr")
AN = os.path.join(ROOT, "article_methods", "analysis")

# ---- house palette (thread these exact colours; §4.1) --------------------------
c8 = {"overall_severity": "#333333", "cognition": "#55a868", "immunometabolic": "#c44e52",
      "sleep": "#8172b3", "suicidality": "#dd8452", "developmental_risk": "#4c72b0",
      "mania_activation": "#937860", "substance": "#da8bc3"}
FACT = ["overall_severity", "cognition", "immunometabolic", "sleep",
        "suicidality", "developmental_risk", "mania_activation", "substance"]
AGG = "#111111"       # aggregate empirical-coverage series (mean/pooled over 8 axes)
JOINT = "#1b9e77"     # joint 8-D ellipsoid (comparator, teal — not in c8)
REFGREY = "#8a8a8a"   # identity line / tolerance band

# ---- figure-style ladder --------------------------------------------------------
try:
    apply_figure_style(sizes=(8, 7, 6))          # noqa: F821 (kernel-injected helper)
except NameError:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "axes.titlelocation": "left", "axes.titleweight": "normal",
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

# ================================================================================
# DATA
# ================================================================================
cov = pd.read_csv(os.path.join(AN, "coverage_calibration.csv"), comment="#")
nob = pd.read_csv(os.path.join(AN, "coverage_by_nobs.csv"), comment="#")

noms = np.array(sorted(cov.nominal_level.unique()))                       # 0.5 0.8 0.9 0.95
allc = cov[cov.axis == "ALL"].sort_values("nominal_level")["empirical_coverage"].values
has_joint = (cov.axis == "joint_ellipsoid").any()
if has_joint:
    joint = cov[cov.axis == "joint_ellipsoid"].sort_values("nominal_level")["empirical_coverage"].values
per_axis = {f: cov[cov.axis == f].sort_values("nominal_level")["empirical_coverage"].values
            for f in FACT}

order = ["0", "1-2", "3-5", "6-10", "11+"]
pool = nob[nob.axis == "ALL_pooled"].copy()
pool["nobs_bin"] = pd.Categorical(pool.nobs_bin, order, ordered=True)
pool = pool.sort_values("nobs_bin")
cov95 = pool["nominal_0p95_empirical_coverage"].values.astype(float)
nbin = pool["n"].values.astype(int)
se_bin = np.sqrt(0.95 * 0.05 / nbin)                # per-bin binomial SE
SE_REF = 0.0022                                     # 0.95-level MC SE (n_sim=10000), summary.md

# ================================================================================
# FIGURE
# ================================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.5),
                               gridspec_kw={"wspace": 0.30, "width_ratios": [1, 1]})

# ---- LEFT: empirical vs nominal --------------------------------------------------
lo, hi = 0.45, 0.985
axL.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=REFGREY, zorder=1)
axL.text(0.585, 0.545, "perfect calibration\n($y=x$)", fontsize=6, color="#6a6a6a",
         ha="left", va="center", rotation=39, rotation_mode="anchor")

# 8 individual axes as threaded (c8) points, drawn as a coloured HALO behind the
# focal mean marker so their clustering on the identity line is visible (they all
# fall in 0.945-0.953 at the 0.95 level — no cancellation behind the mean).
for f in FACT:
    axL.scatter(noms, per_axis[f], s=46, color=c8[f], alpha=0.60,
                linewidths=0, zorder=2)
# joint 8-D ellipsoid (comparator)
if has_joint:
    axL.plot(noms, joint, "-", color=JOINT, lw=1.1, zorder=3)
    axL.scatter(noms, joint, s=30, marker="s", color=JOINT, edgecolor="white",
                linewidth=0.4, zorder=4)
# mean over 8 axes (focal) — smaller solid dot centred in the coloured halo
axL.plot(noms, allc, "-", color=AGG, lw=1.3, zorder=5)
axL.scatter(noms, allc, s=24, marker="o", color=AGG, edgecolor="white",
            linewidth=0.5, zorder=6)

# headline value-on-mark: the 95% level lands at 0.949
axL.annotate(f"{allc[-1]:.3f}", xy=(noms[-1], allc[-1]), xytext=(noms[-1] - 0.028, allc[-1] + 0.028),
             fontsize=6.4, color=AGG, ha="right",
             arrowprops=dict(arrowstyle="-", lw=0.5, color=AGG))

axL.set_xlim(lo, hi); axL.set_ylim(lo, hi)
axL.set_aspect("equal")
axL.set_xticks(noms); axL.set_yticks(noms)
axL.set_xticklabels([f"{v:g}" for v in noms]); axL.set_yticklabels([f"{v:g}" for v in noms])
axL.set_xlabel("Nominal credible level")
axL.set_ylabel("Empirical coverage")
axL.set_title("Credible intervals are calibrated\nacross all levels", fontsize=8)

from matplotlib.legend_handler import HandlerTuple

# multi-colour swatch (4 representative c8 hues) honestly keys the threaded per-axis points
halo_swatch = tuple(Line2D([0], [0], marker="o", color=c8[f], lw=0, markersize=5, alpha=0.7)
                    for f in ["overall_severity", "immunometabolic", "sleep", "cognition"])
handlesL = [Line2D([0], [0], marker="o", color=AGG, lw=1.3, markeredgecolor="white",
                   markersize=5.5, label="mean over 8 axes")]
labelsL = ["mean over 8 axes"]
if has_joint:
    handlesL.append(Line2D([0], [0], marker="s", color=JOINT, lw=1.1, markeredgecolor="white",
                           markersize=5.5, label="joint 8-D ellipsoid"))
    labelsL.append("joint 8-D ellipsoid")
handlesL.append(halo_swatch)
labelsL.append("8 individual axes")
axL.legend(handlesL, labelsL, loc="lower right", fontsize=6.2, frameon=False,
           handletextpad=0.5, labelspacing=0.4, borderaxespad=0.3,
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)})

# ---- RIGHT: coverage vs #observed home-items -------------------------------------
x = np.arange(len(order))
axR.axhspan(0.95 - 2 * SE_REF, 0.95 + 2 * SE_REF, color=REFGREY, alpha=0.18, zorder=1)
axR.axhline(0.95, ls="--", lw=1.0, color=REFGREY, zorder=2)
axR.errorbar(x, cov95, yerr=2 * se_bin, fmt="o", color=AGG, ecolor=AGG,
             elinewidth=1.0, capsize=3, capthick=1.0, markersize=6,
             markeredgecolor="white", markeredgewidth=0.5, zorder=4)

# n annotations per bin (above each point, clear of the error bar)
for xi, ci, si, ni in zip(x, cov95, se_bin, nbin):
    axR.annotate(f"n={ni:,}", xy=(xi, ci + 2 * si), xytext=(0, 5),
                 textcoords="offset points", ha="center", va="bottom",
                 fontsize=5.6, color="#555")

axR.text(0.98, 0.95 + 2 * SE_REF, "nominal 0.95 $\\pm$ 2 SE", fontsize=6.2, color="#6a6a6a",
         ha="left", va="center", transform=axR.get_yaxis_transform())
axR.annotate("0 items \u2192 prior returned\n(nominal by construction)",
             xy=(0, cov95[0] - 2 * se_bin[0]), xytext=(0.15, 0.938),
             textcoords="data", fontsize=5.9, color="#555", ha="left", va="top",
             arrowprops=dict(arrowstyle="-", lw=0.5, color="#999",
                             connectionstyle="arc3,rad=-0.2"))

axR.set_xlim(-0.5, len(order) - 0.5)
axR.set_ylim(0.934, 0.962)
axR.set_xticks(x); axR.set_xticklabels(order)
axR.set_xlabel("Observed home-items on the axis")
axR.set_ylabel("95% interval coverage")
axR.set_title("Coverage holds from prior-dominated\nto fully observed", fontsize=8)

fig.savefig("fig5_calibration.png", dpi=300, bbox_inches="tight")
fig.savefig("fig5_calibration.pdf", bbox_inches="tight")
print("wrote fig5_calibration.png / .pdf")
print("LEFT mean/8 :", np.round(allc, 4))
print("RIGHT pooled:", np.round(cov95, 4), "| n:", list(nbin))
