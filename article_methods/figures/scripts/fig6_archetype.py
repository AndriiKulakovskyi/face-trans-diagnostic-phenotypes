"""
FACE-ATLAS methods paper — Fig 6: archetype simplex 'fog' + reconstruction fidelity.

Presents the 5 archetypes as a MODELLING SUMMARY of the continuum (instrument-forward),
not as discrete clinical types.

  A — simplex 'fog': 9,013 patients projected into the A=5 pentagon, coloured by dominant
      corner. Most patients sit in the interior => blends, not types.
  B — archetype-weight entropy histogram (normalized H/ln5 in [0,1]); median + the
      no-majority fraction quantify the continuum.
  C — reconstruction fidelity: per-axis R^2 of the 5-corner convex blend vs the PCA5 linear
      upper bound and the k-means5 hard-partition lower bound; OVERALL summary marked.
  D — reconstruction error vs archetype-weight entropy: high-entropy 'fog' patients do NOT
      reconstruct worse (weak negative trend).

Reads the frozen products in article_methods/analysis/ + the strata parquet. Run from
article_methods/figures/ so the relative savefig lands there.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy.random import default_rng

ROOT = os.path.expanduser("~/Desktop/face-common-bp-sz-dr")
AN = os.path.join(ROOT, "article_methods", "analysis")

# ---- house palette (thread c8 on axis identity; §4.1) ---------------------------
c8 = {"overall_severity": "#333333", "cognition": "#55a868", "immunometabolic": "#c44e52",
      "sleep": "#8172b3", "suicidality": "#dd8452", "developmental_risk": "#4c72b0",
      "mania_activation": "#937860", "substance": "#da8bc3"}
disp8 = {"overall_severity": "General burden (G)", "cognition": "Cognition",
         "immunometabolic": "Immunometabolic", "sleep": "Sleep", "suicidality": "Suicidality",
         "developmental_risk": "Developmental risk", "mania_activation": "Mania/activation",
         "substance": "Substance"}

# archetype corner palette — Okabe-Ito (CVD-safe, §4.5); distinct from c8 and from Panel-C series
ARCH_C = {0: "#CC79A7", 1: "#E69F00", 2: "#D55E00", 3: "#0072B2", 4: "#009E73"}
ARCH_NM = {0: "Sleep /\nactivation", 1: "High-burden,\nclean biology", 2: "Immuno-\nmetabolic",
           3: "Developmental /\nsuicidality", 4: "Low-burden"}

# Panel-C 3-series palette (§4.2): archetype focal, two bounding comparators muted
COL_ARCH = "#0b7a6f"   # focal — the interpretable convex blend
COL_PCA = "#555555"    # PCA5 upper bound (best 5-D linear)
COL_KM = "#b0b0b0"     # k-means5 lower bound (hard partition)

try:
    apply_figure_style(sizes=(8, 7, 6))          # noqa: F821 (kernel-injected)
except NameError:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "axes.titlelocation": "left", "axes.titleweight": "normal",
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})

# ================================================================================
# DATA
# ================================================================================
# --- simplex weights + entropy (Panels A, B) ---
ps = pd.read_parquet(os.path.join(ROOT, "results/face/strata_oop/consolidate/patient_strata.parquet"))
W = ps[[f"arch_w{a}" for a in range(5)]].values.astype(float)
W = W / W.sum(1, keepdims=True)
dom = W.argmax(1)
maxw = W.max(1)
Hn = -(W * np.log(np.clip(W, 1e-12, None))).sum(1) / np.log(5)   # normalized entropy in [0,1]
N = len(W)
frac_interior = float((maxw < 0.5).mean())     # no-majority blend
frac_corner = float((maxw > 0.8).mean())        # near a single corner
medH = float(np.median(Hn))

# pentagon vertices (top-first, clockwise) and patient positions
ang = np.pi / 2 + np.arange(5) * 2 * np.pi / 5
V = np.c_[np.cos(ang), np.sin(ang)]
P = W @ V

# --- reconstruction fidelity (Panels C, D) ---
fd = np.load(os.path.join(AN, "archetype_reconstruction_figdata.npz"), allow_pickle=True)
axes_r = [str(a) for a in fd["axes"]]
r2_arch = fd["r2_archetype"]; r2_pca = fd["r2_pca5"]; r2_km = fd["r2_kmeans5"]
ov_arch = float(fd["overall_r2_archetype"]); ov_pca = float(fd["overall_r2_pca5"]); ov_km = float(fd["overall_r2_kmeans5"])
eb_lo = fd["entropy_bin_lo"]; eb_hi = fd["entropy_bin_hi"]
eb_err = fd["entropy_bin_mean_err_sd"]; eb_n = fd["entropy_bin_n"]
r_pear = float(fd["corr_entropy_error_pearson"])

# ================================================================================
# FIGURE  — 2x2
# ================================================================================
fig = plt.figure(figsize=(10.0, 8.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 0.92],
                      wspace=0.26, hspace=0.34)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])

# ---- A: simplex fog -------------------------------------------------------------
_ord = default_rng(0).permutation(N)   # shuffle so no colour is drawn last on top
axA.scatter(P[_ord, 0], P[_ord, 1], c=[ARCH_C[d] for d in dom[_ord]],
            s=4, alpha=0.38, linewidths=0, zorder=2)
poly = np.vstack([V, V[0]])
axA.plot(poly[:, 0], poly[:, 1], color="0.55", lw=1.0, zorder=4)
# per-corner label placement: side labels grow OUTWARD (ha) so they clear the marker
# and the panel edge; top/bottom labels centred.  (x, y, ha, va)
lab_pos = {
    0: (0.00, 1.42, "center", "bottom"),   # top          — Sleep/activation
    1: (-1.16, 0.70, "right", "center"),    # upper-left   — High-burden
    2: (-0.66, -1.32, "center", "top"),     # lower-left   — Immunometabolic
    3: (0.66, -1.32, "center", "top"),      # lower-right  — Developmental/suicidality
    4: (1.16, 0.70, "left", "center"),      # upper-right  — Low-burden
}
for a in range(5):
    axA.scatter(*V[a], color=ARCH_C[a], s=70, zorder=6, edgecolor="black", linewidth=0.8)
    lx, ly, ha, va = lab_pos[a]
    axA.text(lx, ly, ARCH_NM[a], ha=ha, va=va, fontsize=6.6, fontweight="bold",
             color=ARCH_C[a], zorder=7)
axA.set_aspect("equal"); axA.axis("off")
axA.set_xlim(-1.95, 1.95); axA.set_ylim(-1.70, 1.78)
axA.set_title("Patients are blends of the five archetypes,\nnot discrete types",
              fontsize=8, loc="left", x=0.05)
try:
    panel_letter(axA, "a", dx=-0.05, dy=1.14)     # noqa: F821
except NameError:
    axA.text(-0.05, 1.14, "a", transform=axA.transAxes, fontweight="bold", fontsize=11, va="bottom")

# ---- B: entropy histogram -------------------------------------------------------
axB.hist(Hn, bins=50, color="#8172b3", alpha=0.85, edgecolor="white", linewidth=0.2)
axB.axvline(medH, color="#c44e52", lw=1.6, zorder=5)
axB.annotate(f"median {medH:.2f}", xy=(medH, axB.get_ylim()[1] * 0.9),
             xytext=(medH - 0.03, axB.get_ylim()[1] * 0.9), fontsize=6.4, color="#c44e52",
             ha="right", va="center")
axB.set_xlim(0, 1)
axB.set_xlabel("Archetype-weight entropy\n(0 = single type, 1 = uniform 5-way blend)")
axB.set_ylabel("Patients")
axB.set_title("Most patients are interior blends", fontsize=8, loc="left")
axB.text(0.03, 0.80, f"{100*frac_interior:.0f}% no-majority blend\nonly {100*frac_corner:.1f}% near one corner",
         transform=axB.transAxes, fontsize=6.4, color="#444", va="top")
try:
    panel_letter(axB, "b", dx=-0.16, dy=1.0)     # noqa: F821
except NameError:
    axB.text(-0.16, 1.0, "b", transform=axB.transAxes, fontweight="bold", fontsize=11, va="bottom")

# ---- C: reconstruction fidelity per axis ---------------------------------------
order_r = list(np.argsort(r2_arch))          # ascending -> best at top after y flip
y = np.arange(len(order_r))
for yi, idx in zip(y, order_r):
    trio = [r2_km[idx], r2_pca[idx], r2_arch[idx]]
    axC.plot([min(trio), max(trio)], [yi, yi], color="0.82", lw=1.4, zorder=1)  # groups the 3 methods
    axC.scatter(r2_km[idx], yi, marker="v", s=30, color=COL_KM, edgecolor="0.4",
                linewidth=0.4, zorder=3)
    axC.scatter(r2_pca[idx], yi, marker="^", s=30, color=COL_PCA, edgecolor="white",
                linewidth=0.3, zorder=3)
    axC.scatter(r2_arch[idx], yi, marker="o", s=52, color=COL_ARCH, edgecolor="white",
                linewidth=0.5, zorder=4)
# OVERALL summary row above a divider (not a peer axis)
y_ov = len(order_r) + 0.7
axC.axhline(len(order_r) - 0.15, color="0.7", lw=0.6, ls=":")
trio_ov = [ov_km, ov_pca, ov_arch]
axC.plot([min(trio_ov), max(trio_ov)], [y_ov, y_ov], color="0.82", lw=1.4, zorder=1)
axC.scatter(ov_km, y_ov, marker="v", s=34, color=COL_KM, edgecolor="0.4", linewidth=0.4, zorder=3)
axC.scatter(ov_pca, y_ov, marker="^", s=34, color=COL_PCA, edgecolor="white", linewidth=0.3, zorder=3)
axC.scatter(ov_arch, y_ov, marker="o", s=60, color=COL_ARCH, edgecolor="white", linewidth=0.5, zorder=4)
axC.annotate(f"{ov_arch:.2f}", xy=(ov_arch, y_ov), xytext=(0, 7), textcoords="offset points",
             ha="center", va="bottom", fontsize=6.6, color=COL_ARCH, fontweight="bold")

axC.axvline(0, color="0.6", lw=0.6, zorder=0)   # R^2 = 0 reference (substance dips below)
yticks = list(y) + [y_ov]
ylabs = [disp8[axes_r[i]] for i in order_r] + ["OVERALL\n(pooled)"]
axC.set_yticks(yticks); axC.set_yticklabels(ylabs)
# thread c8 onto the per-axis tick labels (axis identity), OVERALL stays black bold
for tl, idx in zip(axC.get_yticklabels()[:-1], order_r):
    tl.set_color(c8[axes_r[idx]])
axC.get_yticklabels()[-1].set_fontweight("bold")
axC.set_ylim(-0.6, y_ov + 0.9)
axC.set_xlim(-0.08, 1.03)
axC.set_xlabel("Reconstruction $R^2$  (coordinate variance retained)")
axC.set_title("The 5-corner convex blend retains 59% of coordinate variance", fontsize=8, loc="left")
handlesC = [
    Line2D([0], [0], marker="o", color=COL_ARCH, lw=0, markeredgecolor="white", markersize=6.5,
           label="5-corner convex blend"),
    Line2D([0], [0], marker="^", color=COL_PCA, lw=0, markersize=6,
           label="PCA-5 (best 5-D linear)"),
    Line2D([0], [0], marker="v", color=COL_KM, lw=0, markeredgecolor="0.4", markersize=6,
           label="k-means-5 (hard partition)")]
axC.legend(handles=handlesC, loc="lower right", fontsize=6.0, frameon=False,
           handletextpad=0.4, labelspacing=0.35, borderaxespad=0.4)
try:
    panel_letter(axC, "c", dx=-0.30, dy=1.0)     # noqa: F821
except NameError:
    axC.text(-0.30, 1.0, "c", transform=axC.transAxes, fontweight="bold", fontsize=11, va="bottom")

# ---- D: reconstruction error vs entropy ----------------------------------------
xmid = 0.5 * (eb_lo + eb_hi)
sizes = 18 + 90 * (eb_n / eb_n.max())          # area ~ n (small-n bins de-emphasized)
axD.scatter(xmid, eb_err, s=sizes, color="#4c72b0", alpha=0.85, edgecolor="white",
            linewidth=0.5, zorder=3)
# annotate the two smallest-n (noisy) bins so they're not over-read
for xi, yi, ni in zip(xmid, eb_err, eb_n):
    if ni <= 10:
        axD.annotate(f"n={ni}", xy=(xi, yi), xytext=(0, -9), textcoords="offset points",
                     ha="center", va="top", fontsize=5.4, color="#888")
axD.set_xlim(0, 1)
axD.set_xlabel("Archetype-weight entropy (0\u21921)")
axD.set_ylabel("Reconstruction error\n(RMSE across 8 axes, SD units)")
axD.set_title("'Fog' patients do not reconstruct worse", fontsize=8, loc="left")
axD.text(0.97, 0.93, f"Pearson $r$ = {r_pear:.2f}\n(weakly negative)",
         transform=axD.transAxes, fontsize=6.2, color="#444", ha="right", va="top")
axD.text(0.5, -0.30, "marker area \u221d patients per bin", transform=axD.transAxes,
         fontsize=5.6, color="#888", ha="center", va="top")
try:
    panel_letter(axD, "d", dx=-0.18, dy=1.0)     # noqa: F821
except NameError:
    axD.text(-0.18, 1.0, "d", transform=axD.transAxes, fontweight="bold", fontsize=11, va="bottom")

fig.savefig("fig6_archetype.png", dpi=300, bbox_inches="tight")
fig.savefig("fig6_archetype.pdf", bbox_inches="tight")
print("wrote fig6_archetype.png / .pdf")
print(f"N={N}  median_entropy={medH:.3f}  interior={frac_interior:.3f}  corner={frac_corner:.4f}")
print(f"OVERALL R2: archetype={ov_arch:.3f}  PCA5={ov_pca:.3f}  kmeans5={ov_km:.3f}")
print("per-axis order (worst->best):", [axes_r[i] for i in order_r])
