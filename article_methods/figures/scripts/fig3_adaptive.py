"""
FACE-ATLAS methods paper — Fig 3 (adaptive assessment + patient-adaptive CAT overlay).
NEW composition. Reads article_methods/analysis/cat_vs_upperbound.csv (method in {upper_bound, cat}).
LEFT : population-mean fixed-order reliability curves, all 8 axes (reproduces the published
       fig_adaptive_assessment left panel), house palette, direct end-of-line labels, 0.90 guide,
       bank-capped axes (mania 0.408 / substance 0.429) marked.
RIGHT: upper_bound (solid) vs CAT (dashed) for suicidality (genuine Bernoulli-topped signature)
       and immunometabolic (Gaussian-topped, curves coincide). Demonstrates the fixed-order
       population-mean curve is a TIGHT upper bound on adaptive efficiency.
Run FROM article_methods/figures/ .  Requires figure-style apply_figure_style()/panel_letter().
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"; os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = "/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr"
df = pd.read_csv(f"{ROOT}/article_methods/analysis/cat_vs_upperbound.csv", comment="#")

c8 = {"overall_severity":"#333333","cognition":"#55a868","immunometabolic":"#c44e52","sleep":"#8172b3",
      "suicidality":"#dd8452","developmental_risk":"#4c72b0","mania_activation":"#937860","substance":"#da8bc3"}
disp8 = {"overall_severity":"General burden (G)","cognition":"Cognition","immunometabolic":"Immunometabolic",
         "sleep":"Sleep","suicidality":"Suicidality","developmental_risk":"Developmental risk",
         "mania_activation":"Mania/activation","substance":"Substance"}

# --- figure-style apply_figure_style + panel_letter (inlined for standalone reproducibility) ---
def apply_figure_style(*, sizes=(8,7,6)):
    base, secondary, tick = sizes
    mpl.rcParams.update({
        "font.family":"sans-serif","font.size":base,"axes.labelsize":base,"axes.titlesize":base,
        "legend.fontsize":secondary,"xtick.labelsize":tick,"ytick.labelsize":tick,"axes.linewidth":0.6,
        "xtick.direction":"out","ytick.direction":"out","xtick.major.size":3,"ytick.major.size":3,
        "xtick.major.width":0.6,"ytick.major.width":0.6,"axes.spines.top":False,"axes.spines.right":False,
        "legend.frameon":False,"figure.dpi":200,"savefig.dpi":300,"savefig.bbox":"tight",
        "axes.titlelocation":"left","lines.linewidth":1.2,"patch.linewidth":0.6,
        "pdf.fonttype":42,"ps.fonttype":42})
def panel_letter(ax, letter, dx=-0.18, dy=1.02):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=mpl.rcParams["font.size"]+2,
            fontweight="bold", va="bottom", ha="left")
apply_figure_style()

def curve(method, ax):
    s = df[(df.method==method)&(df.axis==ax)].sort_values("n_items")
    return s.n_items.values.astype(float), s.reliability.values.astype(float)

full_axes = ["suicidality","immunometabolic","developmental_risk","overall_severity","cognition","sleep"]
capped    = ["mania_activation","substance"]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 5.0),
                               gridspec_kw={'wspace':0.34, 'width_ratios':[1.32, 1]})

# ---------------- LEFT ----------------
axL.axhspan(0.20, 0.50, color="#d0d0d0", alpha=0.16, zorder=0)
for ax in full_axes + capped:
    k, rr = curve("upper_bound", ax)
    axL.plot(k, rr, marker='o', ms=2.6, lw=1.5, color=c8[ax], zorder=3)
axL.plot([0.5, 20.2], [0.90, 0.90], color='#888', ls='--', lw=0.9, zorder=1)
axL.text(0.7, 0.912, "reliability 0.90", fontsize=6, color='#666')

XG = 20.9
endpts = [(ax,)+(lambda kr:(kr[0][-1],kr[1][-1]))(curve("upper_bound",ax)) for ax in full_axes]
endpts.sort(key=lambda t: t[2], reverse=True)
for (ax, xe, ye), sy in zip(endpts, np.linspace(0.923, 0.828, len(endpts))):
    axL.plot([xe, XG], [ye, sy], color=c8[ax], lw=0.6, alpha=0.8, zorder=2)
    axL.text(XG+0.4, sy, disp8[ax], color=c8[ax], fontsize=6.4, va='center', ha='left')

for ax, yoff in [("substance", 0.028), ("mania_activation", -0.028)]:
    k, rr = curve("upper_bound", ax); xe, ye = k[-1], rr[-1]
    axL.scatter([xe],[ye], s=16, color=c8[ax], zorder=4)
    axL.text(xe+0.5, ye+yoff, f"{disp8[ax]} ({ye:.3f})", color=c8[ax], fontsize=6.2,
             va='center', ha='left', fontweight='bold')
axL.text(4.6, 0.335,
         "Bank-limited: only 2 (mania) / 4 (substance)\nindicators exist in the FACE battery —\na content limit, not adaptivity",
         fontsize=5.7, color="#666", ha='left', va='center')
axL.set_xlabel("Items administered (most-informative first)")
axL.set_ylabel("Reliability of the axis score")
axL.set_title("Fixed most-informative order: a few items capture most information\n(population-mean upper bound, all eight axes)",
              fontsize=7.6, loc='left')
axL.set_ylim(0.20, 0.955); axL.set_xlim(0.5, 26.5); axL.set_xticks([1,5,10,15,20])
panel_letter(axL, 'A')

# ---------------- RIGHT ----------------
for ax in ["suicidality","immunometabolic"]:
    ku, ru = curve("upper_bound", ax); kc, rc = curve("cat", ax)
    axR.plot(ku, ru, ls='-', lw=1.7, color=c8[ax], zorder=3, solid_capstyle='round')
    axR.plot(kc, rc, ls=(0,(3,2)), lw=1.5, color='white', zorder=3.4)
    axR.plot(kc, rc, ls=(0,(3,2)), lw=1.3, color=c8[ax], zorder=3.6)
for ax, yl in [("suicidality", 0.9202), ("immunometabolic", 0.8802)]:
    axR.text(20.4, yl, disp8[ax], color=c8[ax], fontsize=6.6, va='center', ha='left', fontweight='bold')
axR.annotate("Suicidality (Bernoulli-topped): CAT \u22120.012 at\n2 items (\u03b8\u0302 still noisy), then ~+0.003 beyond 4",
             xy=(2, 0.8252), xytext=(3.2, 0.798), fontsize=5.9, color=c8["suicidality"], va='center',
             arrowprops=dict(arrowstyle="->", color=c8["suicidality"], lw=0.7, connectionstyle="arc3,rad=0.18"))
axR.annotate("Immunometabolic (Gaussian-topped):\nCAT rides the fixed order exactly \u2014\ninformation is independent of \u03b8",
             xy=(9, 0.8797), xytext=(2.2, 0.923), fontsize=5.9, color=c8["immunometabolic"], va='center',
             arrowprops=dict(arrowstyle="->", color=c8["immunometabolic"], lw=0.7))
handles = [Line2D([0],[0], color="#555", lw=1.6, ls='-', label="Population-mean fixed order (upper bound)"),
           Line2D([0],[0], color="#555", lw=1.4, ls=(0,(3,2)), label="Patient-adaptive (CAT)")]
axR.legend(handles=handles, fontsize=5.9, loc='lower right', frameon=False,
           handlelength=2.4, labelspacing=0.6, borderpad=0.4)
axR.set_xlabel("Items administered (most-informative first)")
axR.set_ylabel("Reliability of the axis score")
axR.set_title("Patient-adaptive selection nearly coincides with the fixed order:\nthe upper bound is tight",
              fontsize=7.6, loc='left')
axR.set_ylim(0.78, 0.935); axR.set_xlim(0.5, 26.0); axR.set_xticks([1,5,10,15,20])
panel_letter(axR, 'B')

fig.subplots_adjust(top=0.90, bottom=0.105, left=0.075, right=0.985)
fig.savefig("fig3_adaptive.png", dpi=300, bbox_inches='tight')
fig.savefig("fig3_adaptive.pdf", bbox_inches='tight')
