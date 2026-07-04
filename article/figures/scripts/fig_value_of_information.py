"""
Figure-generation code for FACE-ATLAS: fig_value_of_information.png

Provenance: extracted verbatim from artifact lineage (version_id b20b8658-bc27-44ab-af04-783d6808aeac).
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
import torch
import torch.nn.functional as F

META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")
    try:
        import glob
        import os
        import sys

        import matplotlib.font_manager as fm
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

sd = torch.load('/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr/results/face/gllvm_oop/s8_full/model_state.pt', map_location="cpu", weights_only=False)
st = sd["state_dict"]
items = sd["items"]
families = sd["families"]
factor_cols = sd["factor_cols"]
J = len(items)

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
        return ljf ** 2 / float(sigma[j]) ** 2
    elif fam == "bernoulli":
        p = 1 / (1 + np.exp(-a))
        return ljf ** 2 * p * (1 - p)
    elif fam == "count":
        mu = np.exp(np.clip(a, -10, 10))
        r = float(count_alpha[j])
        return ljf ** 2 * (mu * r / (r + mu))
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
        info_eta = np.sum(dprob ** 2 / probs)
        return ljf ** 2 * info_eta
    return 0.0


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

allpool = []
for f in factor_cols:
    for _, r in curves8[f].iterrows():
        allpool.append((r["item"], f, r["info"]))
pool8 = pd.DataFrame(allpool, columns=["item", "axis", "info"]).sort_values("info", ascending=False).reset_index(drop=True)

prec8 = {a: 1.0 for a in factor_cols}
budget8 = []
for k, (_, it) in enumerate(pool8.iterrows(), start=1):
    prec8[it.axis] += it["info"]
    mean_rel = np.mean([1 - 1 / prec8[a] for a in factor_cols])
    budget8.append(dict(k=k, item=it["item"], axis=it.axis, mean_rel=mean_rel))
budget8 = pd.DataFrame(budget8)

cov = pd.read_parquet('/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr/results/face/gllvm_oop/consolidate/coordinates.parquet')

c8 = {"overall_severity": "#333333", "cognition": "#55a868", "immunometabolic": "#c44e52", "sleep": "#8172b3",
      "suicidality": "#dd8452", "developmental_risk": "#4c72b0", "mania_activation": "#937860", "substance": "#da8bc3"}
disp8 = {"overall_severity": "General burden", "cognition": "Cognition", "immunometabolic": "Immunometabolic", "sleep": "Sleep",
         "suicidality": "Suicidality", "developmental_risk": "Developmental risk", "mania_activation": "Mania/activation", "substance": "Substance"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.2, 5.0), gridspec_kw={'wspace': 0.42, 'width_ratios': [1.1, 1]})

axL.plot(budget8.k, budget8.mean_rel, lw=1.2, color="#444", zorder=2)
for f in factor_cols:
    sub = budget8[budget8.axis == f]
    axL.scatter(sub.k, sub.mean_rel, s=16, color=c8[f], label=disp8[f], zorder=3, edgecolor='white', linewidth=0.3)
axL.axhline(0.70, color='#888', ls='--', lw=0.9)
axL.text(0.7, 0.712, "mean reliability 0.70", fontsize=6, color='#666')
axL.axvline(27, color='#888', ls=':', lw=0.9)
axL.text(27.5, 0.12, "27 items\n(all 8 axes)", fontsize=5.6, color='#666')
axL.set_xlabel("Items in the shared battery (most-informative first, all 8 axes)")
axL.set_ylabel("Mean reliability across the 8 axes")
axL.set_title("A shared 27-item battery covers all eight axes at mean reliability 0.70", fontsize=6.9, loc='left')
axL.set_xlim(0, 40)
axL.set_ylim(0, 0.78)
axL.legend(fontsize=5.4, frameon=False, loc='lower right', ncol=2)

gaprows = []
for f in factor_cols:
    n = pd.to_numeric(cov[f + "__n_obs"], errors="coerce")
    for coh in ["bp", "sz", "dr"]:
        mask = cov.index.get_level_values(0) == coh
        gaprows.append(dict(axis=f, cohort=coh, mean_nobs=float(n[mask].mean())))
gap8 = pd.DataFrame(gaprows).pivot(index="axis", columns="cohort", values="mean_nobs").loc[factor_cols, ["bp", "sz", "dr"]]
gap8n = gap8.div(gap8.max(axis=1), axis=0)
im = axR.imshow(gap8n.values, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect='auto')
axR.set_xticks(range(3))
axR.set_xticklabels(["Bipolar", "Schizophrenia", "Depression"], fontsize=6.5)
axR.set_yticks(range(len(factor_cols)))
axR.set_yticklabels([disp8[f] for f in factor_cols], fontsize=6.3)
for i, f in enumerate(factor_cols):
    for j, coh in enumerate(["bp", "sz", "dr"]):
        axR.text(j, i, f"{gap8.loc[f, coh]:.0f}", ha='center', va='center', fontsize=6.3, fontweight='bold',
                 color='black' if gap8n.values[i, j] > 0.55 else 'white')
axR.set_title("Items collected per axis (green = best-measured cohort)", fontsize=6.9, loc='left')
cb = fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04)
cb.set_label("relative to best-measured cohort", fontsize=5.5)
cb.ax.tick_params(labelsize=5.5)

# Title and bottom caption paragraph intentionally removed — that text now lives in the
# LaTeX figure caption (\caption{} for fig:voi in sections/03_results.tex).
fig.subplots_adjust(top=0.955, bottom=0.11, left=0.075, right=0.985)
fig.savefig("fig_value_of_information.png", dpi=200, bbox_inches='tight')
