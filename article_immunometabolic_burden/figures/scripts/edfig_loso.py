"""
Figure-generation code for FACE-ATLAS: edfig_loso.png

Provenance: extracted verbatim from artifact lineage (version_id cdb599be-7fa7-4ff5-a265-9fe8d7546b61).
Environment: face-dev
NOTE: these figures were produced in a shared `face-dev` kernel session in which the
fitted GLLVM model state (results/face/gllvm_oop/s8_full/model_state.pt) and derived
arrays (loadings, sigmas, families, coordinates) were loaded once and reused across
cells. This file is the exact producing cell; if run standalone it may require that
shared setup (model load + per-family Fisher-information arrays) to be present.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.linewidth": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.grid": False,
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
})

import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_OUT = _os.path.dirname(_HERE)                       # article/figures/
# LOSO summary lives only in the artifact store upstream; a committed copy is kept
# alongside the figure scripts so this figure reproduces without store access.
loso = pd.read_csv(_os.path.join(_OUT, "data", "loso_summary_production.csv"))

lut = {1:"Bordeaux",2:"Créteil",3:"Montpellier",4:"Grenoble",5:"Nancy*",6:"Marseille",
       7:"Paris-Larib.",8:"Versailles",9:"Monaco",10:"Clermont-F.",
       12:"Colombes",13:"Lyon",14:"Strasbourg*",15:"Grenoble-2",16:"Besançon"}
loso["city"] = loso["site"].astype(int).map(lut)

d = loso.sort_values("n_held").reset_index(drop=True)
y = np.arange(len(d))
labels = [f"{c}  (n={int(n)})" for c, n in zip(d.city, d.n_held)]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={'wspace': 0.55, 'width_ratios': [1, 1]})

# ---- (a) immunometabolic loading congruence per held-out site ----
axA.axvspan(0.95, 1.001, color="#dfe9df", alpha=0.5, zorder=0)
axA.axvline(0.95, color="#55a868", ls='--', lw=1, zorder=1)
axA.text(0.9505, 0.3, "congruence bar 0.95", fontsize=5.6, color="#3d7a4d", rotation=90, va='bottom')
axA.scatter(d.immuno_tucker, y, s=34, color="#c44e52", zorder=3, edgecolor='white', linewidth=0.5)
for yi, v in zip(y, d.immuno_tucker):
    axA.text(v-0.0006, yi, f"{v:.4f}", fontsize=5.2, color="#8f353a", va='center', ha='right')
axA.set_yticks(y); axA.set_yticklabels(labels, fontsize=6)
axA.set_xlim(0.949, 1.0009); axA.set_ylim(-0.7, len(d)-0.3)
axA.set_xlabel("Immunometabolic loading congruence (Tucker φ)")
axA.set_title("(a) The immunometabolic axis is invariant to holding out any site", fontsize=7.2, loc='left')

# ---- (b) decoupling per held-out site ----
full_dec = 0.080
axB.axvspan(0.0, 0.15, color="#e7edf5", alpha=0.6, zorder=0)
axB.axvline(full_dec, color="#4c72b0", ls='--', lw=1, zorder=1)
axB.text(full_dec+0.0015, 0.3, "full-sample ≈0.08", fontsize=5.6, color="#3a5a86", rotation=90, va='bottom')
axB.scatter(d.decouple_immuno, y, s=34, color="#4c72b0", zorder=3, edgecolor='white', linewidth=0.5)
axB.set_yticks(y); axB.set_yticklabels([]); axB.set_ylim(-0.7, len(d)-0.3)
axB.set_xlim(0, 0.15)
axB.set_xlabel("|corr(immunometabolic, general burden)|")
axB.set_title("(b) The burden-dissociation reproduces in every fold", fontsize=7.2, loc='left')

fig.suptitle("Leave-one-site-out external validity: the map and its headline finding survive out-of-site refitting",
             fontsize=9, fontweight='bold', y=1.005)
fig.text(0.5, -0.075,
  "Each row is one of the 15 sites (N≥100) held out entirely; the GLLVM is refit on the remaining sites and the held-out site's loadings are projected. "
  "(a) The immunometabolic loading pattern is essentially identical whichever site is removed (Tucker φ 0.999–1.000, all far above the 0.95 bar). "
  "All eight factors cleared their pre-set congruence bar in all 15 folds; the single weakest factor was φ=0.917 (Monaco, n=237), a thin 2–4-item axis still above the 0.85 bar for thin factors. "
  "(b) The score-level dissociation between the immunometabolic axis and general burden holds in every fold (|r|=0.073–0.082, vs ≈0.08 on the full sample) — the finding is not driven by any single site.",
  ha='center', fontsize=5.1, color='#555', wrap=True)
fig.subplots_adjust(top=0.86, bottom=0.24, left=0.155, right=0.985)
fig.savefig(_os.path.join(_OUT, 'edfig_loso.png'), dpi=200, bbox_inches='tight')
fig.savefig(_os.path.join(_OUT, 'edfig_loso.pdf'), bbox_inches='tight')
print("wrote", _os.path.join(_OUT, 'edfig_loso.png'))
