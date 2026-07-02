"""
Figure-generation code for FACE-ATLAS: fig6_prognosis_rebuilt.png

Provenance: extracted verbatim from artifact lineage (version_id be3d9846-abf2-4716-b73a-3a8ecbac6ad8).
Environment: face-dev
NOTE: these figures were produced in a shared `face-dev` kernel session in which the
fitted GLLVM model state (results/face/gllvm_oop/s8_full/model_state.pt) and derived
arrays (loadings, sigmas, families, coordinates) were loaded once and reused across
cells. This file is the exact producing cell; if run standalone it may require that
shared setup (model load + per-family Fisher-information arrays) to be present.
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")
    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)

apply_figure_style()

atlas = pd.read_csv("/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr/results/face/prognosis_oop/endpoints/archetype_atlas.csv")
egf = atlas[atlas.outcome == "egf"].copy()

short = {0: "A0 ↑sleep/mania", 1: "A1 ↑burden", 2: "A2 ↑immuno", 3: "A3 ↑dev/suic", 4: "A4 low-sev"}
coh_c = {"bp": "#4c72b0", "sz": "#c44e52", "dr": "#55a868"}
coh_nm = {"bp": "Bipolar", "sz": "Schizophrenia", "dr": "Depression"}
order = [4, 0, 1, 3, 2]

fig = plt.figure(figsize=(11, 3.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.95], wspace=0.42)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])
axC = fig.add_subplot(gs[2])

ypos = {a: i for i, a in enumerate(order)}
axA.axhspan(ypos[2] - 0.42, ypos[2] + 0.42, color="#c44e52", alpha=0.08, zorder=0)
for coh in ["bp", "dr", "sz"]:
    sub = egf[(egf.cohort == coh) & (egf.n >= 50)]
    xs = [sub[sub.archetype == a].remission_rate.iloc[0] * 100 if len(sub[sub.archetype == a]) else np.nan for a in order]
    ys = [ypos[a] for a in order]
    axA.plot(xs, ys, marker='o', ms=6, lw=1.3, color=coh_c[coh], label=coh_nm[coh], zorder=3, alpha=0.9)
axA.set_yticks(range(len(order)))
axA.set_yticklabels([short[a] for a in order], fontsize=7)
axA.get_yticklabels()[ypos[2]].set_color("#c44e52")
axA.get_yticklabels()[ypos[2]].set_fontweight('bold')
axA.set_xlabel("Two-year functional remission (EGF, %)")
axA.set_xlim(0, 85)
axA.set_ylim(-0.6, 4.6)
axA.set_title("(a) Remission gradient holds within every diagnosis", fontsize=7.4, loc='left')
axA.legend(fontsize=6, frameon=False, loc='lower right', title="cohort", title_fontsize=6)

labels = ["Archetype\n(A=5)", "DSM-5\ndiagnosis", "Cohort"]
vals = [0.2558, 0.0264, 0.0181]
barc = ["#c44e52", "#8c8c8c", "#c0c0c0"]
b = axB.bar(labels, vals, color=barc, width=0.62, zorder=3)
axB.set_ylabel("Mean η² across the 8 map axes")
axB.set_ylim(0, 0.31)
axB.set_title("(b) Archetypes summarize the map\n9.7× more tightly than DSM-5", fontsize=7.4, loc='left')
for rect, v in zip(b, vals):
    axB.text(rect.get_x() + rect.get_width() / 2, v + 0.007, f"{v:.3f}", ha='center', fontsize=7, fontweight='bold')
axB.annotate("", xy=(0, 0.285), xytext=(1, 0.285), arrowprops=dict(arrowstyle="<->", color="#c44e52", lw=1))
axB.text(0.5, 0.292, "9.7×", ha='center', fontsize=8, color="#c44e52", fontweight='bold')

enc = ["A=5 archetypes", "Continuous\n(8 axes)", "A=5 (G-free)", "Tessellation\n(K=3)", "Durable\nbiology axis"]
elpd = [62.8, 38.1, 33.5, 19.6, 2.3]
ecol = ["#c44e52", "#4c72b0", "#a0a0a0", "#a0a0a0", "#d0d0d0"]
yb = np.arange(len(enc))[::-1]
axC.barh(yb, elpd, color=ecol, zorder=3, height=0.62)
axC.set_yticks(yb)
axC.set_yticklabels(enc, fontsize=6.3)
for y, v in zip(yb, elpd):
    axC.text(v + 1, y, f"+{v:.0f}", va='center', fontsize=6.5, fontweight='bold' if v == 62.8 else 'normal')
axC.set_xlabel("Δ ELPD, held-out functioning")
axC.set_xlim(0, 72)
axC.set_title("(c) Archetype encoding predicts\nfunctioning best", fontsize=7.4, loc='left')

fig.suptitle("Prognostic reach: the archetype coordinate carries outcome structure diagnosis misses", fontsize=9.2, fontweight='bold', y=1.02)
fig.text(0.5, -0.10, "(a) Two-year functional-remission rate by baseline dominant archetype, shown within each cohort (cells n≥50): the immunometabolic pole (A2, shaded) is the "
         "lowest-remitting archetype in every diagnosis — a within-diagnosis gradient, not a cohort-composition artefact. (b) Variance in the eight map axes explained by "
         "archetype vs DSM-5 vs cohort (mean η²=0.256 / 0.026 / 0.018); the archetype coordinate is a 9.7× more compact summary of where a patient sits than the diagnostic label "
         "(compactness on the map axes, not outcome variance). (c) Held-out incremental predictive value (ΔELPD) for two-year functioning: the A=5 archetype encoding is the "
         "strongest single predictor, ahead of the continuous coordinates and every hard tessellation. The individual-level remission AUC gain over diagnosis is small (+0.010); "
         "the map's prognostic value is in stratification and durable measurement, not individual point prediction.",
         ha='center', fontsize=5.2, color='#555', wrap=True)
fig.subplots_adjust(top=0.82, bottom=0.30, left=0.10, right=0.985)
fig.savefig("fig6_prognosis_rebuilt.png", dpi=200, bbox_inches='tight')