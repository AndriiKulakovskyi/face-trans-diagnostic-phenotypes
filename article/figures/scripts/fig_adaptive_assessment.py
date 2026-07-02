"""
Figure-generation code for FACE-ATLAS: fig_adaptive_assessment.png

Provenance: extracted verbatim from artifact lineage (version_id e5e23816-6ab3-47a0-96b3-e415afc39256).
Environment: face-dev
NOTE: these figures were produced in a shared `face-dev` kernel session in which the
fitted GLLVM model state (results/face/gllvm_oop/s8_full/model_state.pt) and derived
arrays (loadings, sigmas, families, coordinates) were loaded once and reused across
cells. This file is the exact producing cell; if run standalone it may require that
shared setup (model load + per-family Fisher-information arrays) to be present.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt

# apply_figure_style (auto-injected skill helper)
META_GREY = "#888888"

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

# Load model state
sd = torch.load('/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr/results/face/gllvm_oop/s8_full/model_state.pt', map_location="cpu", weights_only=False)
st = sd["state_dict"]
items = sd["items"]
families = sd["families"]
factor_cols = sd["factor_cols"]
J = len(items)
F_ = len(factor_cols)

# Reconstruct effective params
raw_loading = st["raw_loading"]
loading_free = st["loading_free"].bool()
loading_positive = st["loading_positive"].bool()
lam = torch.zeros_like(raw_loading)
pf = loading_free & loading_positive
sf = loading_free & (~loading_positive)
lam = torch.where(pf, F.softplus(raw_loading) + 1e-5, lam)
lam = torch.where(sf, raw_loading, lam)
alpha = st["alpha"]
psi_floor = 0.30
sigma = psi_floor + F.softplus(st["raw_sigma"])
count_alpha = F.softplus(st["raw_count_alpha"]) + 1e-3

def item_info_on_factor(j, fidx):
    ljf = float(lam[j, fidx])
    if abs(ljf) < 1e-8:
        return 0.0
    fam = families[j]
    a = float(alpha[j])
    if fam == "gaussian":
        return ljf**2 / float(sigma[j])**2
    elif fam == "bernoulli":
        p = 1 / (1 + np.exp(-a))
        return ljf**2 * p * (1 - p)
    elif fam == "count":
        mu = np.exp(np.clip(a, -10, 10))
        r = float(count_alpha[j])
        return ljf**2 * (mu * r / (r + mu))
    elif fam == "ordinal":
        cuts_key = f"ordinal_cutpoints.{j}"
        if cuts_key not in st:
            return 0.0
        raw = st[cuts_key]
        if raw.numel() == 1:
            cuts = raw
        else:
            first = raw[:1]
            inc = F.softplus(raw[1:]) + 1e-3
            cuts = torch.cat([first, first + torch.cumsum(inc, 0)])
        cuts = cuts.numpy()
        s = 1 / (1 + np.exp(-(cuts - 0.0)))
        cdf = np.concatenate([[0.0], s, [1.0]])
        probs = np.diff(cdf)
        probs = np.clip(probs, 1e-8, 1)
        dsk = -s * (1 - s)
        dprob = np.zeros(len(probs))
        dsk_full = np.concatenate([[0.0], dsk, [0.0]])
        for k in range(len(probs)):
            dprob[k] = dsk_full[k + 1] - dsk_full[k]
        info_eta = np.sum(dprob**2 / probs)
        return ljf**2 * info_eta
    return 0.0

# Build per-factor optimal-battery curves
factor_map = {f: i for i, f in enumerate(factor_cols)}
curves8 = {}
for f, fidx in factor_map.items():
    rows = []
    for j in range(J):
        info = item_info_on_factor(j, fidx)
        if info > 1e-6:
            rows.append((items[j], families[j], info))
    df = pd.DataFrame(rows, columns=["item", "family", "info"]).sort_values("info", ascending=False).reset_index(drop=True)
    df["cum_prec"] = 1.0 + df["info"].cumsum()
    df["reliability"] = 1 - 1 / df["cum_prec"]
    df["k"] = np.arange(1, len(df) + 1)
    curves8[f] = df

# Setup for plotting
disp8 = {
    "overall_severity": "General burden (G)", "cognition": "Cognition",
    "immunometabolic": "Immunometabolic", "sleep": "Sleep",
    "suicidality": "Suicidality", "developmental_risk": "Developmental risk",
    "mania_activation": "Mania/activation", "substance": "Substance"
}
c8 = {
    "overall_severity": "#333333", "cognition": "#55a868", "immunometabolic": "#c44e52",
    "sleep": "#8172b3", "suicidality": "#dd8452", "developmental_risk": "#4c72b0",
    "mania_activation": "#937860", "substance": "#da8bc3"
}
order8 = ["overall_severity", "immunometabolic", "developmental_risk", "suicidality",
          "cognition", "sleep", "mania_activation", "substance"]
short_axes = {"mania_activation", "substance"}

imc = curves8["immunometabolic"].head(20)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.9), gridspec_kw={'wspace': 0.30, 'width_ratios': [1.15, 1]})
kmax = 15
CALLOUT_X = 5.4

# shaded under-instrumented ceiling band
axL.axhspan(0.2, 0.5, color="#d0d0d0", alpha=0.18, zorder=0)
axL.text(CALLOUT_X, 0.475, "reliability ceiling < 0.5", fontsize=5.6, color="#8a8a8a",
         ha='left', va='center', style='italic', zorder=1)

for f in order8:
    c = curves8[f].head(kmax)
    shortf = f in short_axes
    axL.plot(c.k, c.reliability, marker='o', ms=3, lw=1.4, color=c8[f], label=disp8[f], zorder=3)
    if shortf:
        xe, ye = c.k.iloc[-1], c.reliability.iloc[-1]
        axL.text(xe + 0.35, ye, f"{int(c.k.iloc[-1])} items", fontsize=5.6, color=c8[f],
                 va='center', ha='left', fontweight='bold')
    else:
        axL.text(min(len(c), kmax) + 0.15, c.reliability.iloc[-1],
                 f"{c.reliability.iloc[-1]:.2f}", fontsize=5, color=c8[f], va='center')

axL.axhline(0.9, color='#888', ls='--', lw=0.9)
axL.text(0.6, 0.912, "reliability 0.90", fontsize=6, color='#666')
axL.text(CALLOUT_X, 0.31,
         "Mania & substance: only 2–4 indicators exist\nin the FACE battery — a bank limitation, not\nmissing data (95–96% of patients have ≥1 item)",
         fontsize=5.6, color="#555", ha='left', va='center')
axL.set_xlabel("Items administered (most-informative first)")
axL.set_ylabel("Reliability of the axis score")
axL.set_title("All eight axes: a few items capture most — where the bank allows", fontsize=7.4, loc='left')
axL.set_ylim(0.2, 0.98)
axL.set_xlim(0.5, kmax + 1.6)
axL.legend(fontsize=5.5, frameon=False, loc='lower right', ncol=1)

# RIGHT: immuno worked example
axR.plot(imc.k, imc.reliability, marker='o', ms=4, lw=1.6, color="#c44e52", zorder=3)
axR.axhline(0.85, color='#888', ls='--', lw=0.9)
axR.text(11, 0.837, "reliability 0.85", fontsize=6, color='#666')
lbls = {1: ("BMI", (5, -9)), 2: ("weight", (5, -10)), 3: ("waist", (4, 7))}
for i, r in imc.head(3).iterrows():
    nm, off = lbls[int(r.k)]
    axR.annotate(nm, (r.k, r.reliability), textcoords="offset points", xytext=off, fontsize=6, color="#c44e52")
axR.axvline(3, color="#c44e52", ls=':', lw=1, alpha=0.6)
axR.text(3.4, 0.62, "3 items →\nreliability 0.85", fontsize=6.5, color="#c44e52", va='center')
axR.set_xlabel("Items administered (most-informative first)")
axR.set_ylabel("Reliability of immunometabolic score")
axR.set_title("Worked example: immunometabolic, 3 of 42 items", fontsize=7.6, loc='left')
axR.set_ylim(0.55, 0.92)
axR.set_xlim(0.5, 20)

fig.suptitle("Adaptive assessment: how many instruments are enough?", fontsize=9, fontweight='bold', x=0.5, y=1.0)
fig.text(0.5, -0.035,
         "Exact Fisher information from the fitted GLLVM, per likelihood family (Gaussian, logistic, graded-response, negative-binomial), so all eight axes are on the "
         "same footing. Three to six items recover most of each well-instrumented axis (immunometabolic triad BMI/weight/waist → 0.85). Mania/activation (2 items) and "
         "substance (4 items) cannot exceed reliability ~0.45: the FACE bank is too small for those constructs — a specification for the next data-collection wave, not a modelling failure.",
         ha='center', fontsize=5.2, color='#666')
fig.subplots_adjust(top=0.89, bottom=0.19, left=0.075, right=0.985)
fig.savefig("fig_adaptive_assessment.png", dpi=200, bbox_inches='tight')