"""
Figure-generation code for FACE-ATLAS: fig_loading_vs_info.png

Provenance: extracted verbatim from artifact lineage (version_id ecfd1aed-0f2e-4407-9b89-dfc10ab9899e).
Environment: face-dev
NOTE: these figures were produced in a shared `face-dev` kernel session in which the
fitted GLLVM model state (results/analyses/variational_gllvm/s8_full/model_state.pt) and derived
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

# skill:figure-style kernel.py (auto-injected on skill load)
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

# Load model state
sd = torch.load('/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr/results/analyses/variational_gllvm/s8_full/model_state.pt', map_location="cpu", weights_only=False)
st = sd["state_dict"]
items = sd["items"]
families = sd["families"]
factor_cols = sd["factor_cols"]
J = len(items)

# Reconstruct effective params
raw_loading = st["raw_loading"]
loading_free = st["loading_free"].bool()
loading_positive = st["loading_positive"].bool()
lam = torch.zeros_like(raw_loading)
pf = loading_free & loading_positive
sf = loading_free & (~loading_positive)
lam = torch.where(pf, F.softplus(raw_loading) + 1e-5, lam)
lam = torch.where(sf, raw_loading, lam)  # J x 8
alpha = st["alpha"]
psi_floor = 0.30
sigma = psi_floor + F.softplus(st["raw_sigma"])
count_alpha = F.softplus(st["raw_count_alpha"]) + 1e-3

factor_map = {f: i for i, f in enumerate(factor_cols)}


def item_info_on_factor(j, fidx):
    """Fisher information item j contributes to latent factor fidx, at theta=0 (population mean)."""
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


# Build per-factor optimal-battery curves
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

# Build D: per-item loading vs fisher information
rows = []
for f in factor_cols:
    fidx = factor_map[f]
    cvals = curves8[f].set_index("item")["info"]
    for j in range(J):
        ljf = float(lam[j, fidx])
        if abs(ljf) < 1e-8:
            continue
        iv = float(cvals.loc[items[j]]) if items[j] in cvals.index else np.nan
        rows.append(dict(item=items[j], axis=f, family=families[j], loading=abs(ljf), fisher=iv))
D = pd.DataFrame(rows)

Dp = D[D.fisher.notna() & (D.loading > 0.05)].copy()

# High loading but near-zero information (the paradox items)
para = D[(D.loading > 0.8) & (D.fisher < 0.05)].sort_values("loading", ascending=False)

famc = {"gaussian": "#4c72b0", "bernoulli": "#c44e52", "ordinal": "#dd8452", "count": "#55a868"}
famnm = {"gaussian": "Gaussian (continuous)", "bernoulli": "Bernoulli (binary)", "ordinal": "Ordinal (graded)", "count": "Count (NB)"}

plt.close('all')
apply_figure_style()

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={'wspace': 0.27, 'width_ratios': [1.25, 1]})

# ---- (a) loading vs information ----
for fam in ["gaussian", "bernoulli", "ordinal", "count"]:
    s = Dp[Dp.family == fam]
    axA.scatter(s.loading, s.fisher + 1e-3, s=22, color=famc[fam], alpha=0.72, edgecolor='none', label=famnm[fam])
axA.set_yscale("log")
axA.set_xlabel("Loading on home axis  |λ|  (each family on its own link scale)")
axA.set_ylabel("Fisher information contributed (common latent metric, log)")
axA.set_title("(a) A large loading does not mean a large contribution", fontsize=7.4, loc='left')
axA.legend(fontsize=5.8, frameon=False, loc='lower right')
# single combined callout for the two flags, placed low-left to avoid the dense cluster
axA.annotate("alcohol & cannabis flags\n(binary, |λ|≈0.92–0.96)\ninfo ≈ 0.01",
             xy=(0.94, 0.011), xytext=(1.55, 0.0022), fontsize=5.8, color="#8f353a", ha='left', va='center',
             arrowprops=dict(arrowstyle="->", color="#8f353a", lw=0.7, connectionstyle="arc3,rad=-0.2"))
# BMI contrast label placed above-left of the blue anchor cluster
bmi = Dp[Dp.item == "bmi"].iloc[0]
axA.annotate(f"BMI (Gaussian)\n|λ|={bmi.loading:.2f}, info={bmi.fisher:.1f}", xy=(bmi.loading, bmi.fisher),
             xytext=(0.30, 3.4), fontsize=5.8, color="#3a5a86", ha='left', va='center',
             arrowprops=dict(arrowstyle="->", color="#3a5a86", lw=0.7))
axA.set_xlim(-0.05, 4.9)

# ---- (b) mechanism: binary information vs prevalence ----
p = np.linspace(0.005, 0.5, 200)
for lam_v, col in [(0.95, "#c44e52"), (0.7, "#dd8452"), (0.5, "#8c8c8c")]:
    axB.plot(p * 100, lam_v ** 2 * p * (1 - p), color=col, lw=1.6, label=f"|λ|={lam_v}")
for _, r in para.iterrows():
    prev = {"suoccur_alcool": 1.4, "suoccur_cannabis": 1.2}[r["item"]]
    axB.scatter([prev], [r.fisher], s=42, color="#8f353a", zorder=5, edgecolor='white', lw=0.6)
axB.annotate("alcohol & cannabis flags\n(~1% endorsed): info ≈ 0.01\ndespite |λ|≈0.95", xy=(1.3, 0.011),
             xytext=(9, 0.045), fontsize=5.8, color="#8f353a", va='center',
             arrowprops=dict(arrowstyle="->", color="#8f353a", lw=0.7))
axB.set_xlabel("Item endorsement rate (%)")
axB.set_ylabel("Fisher information at θ=0")
axB.set_title("(b) A rare binary flag is nearly uninformative,\nwhatever its loading", fontsize=7.4, loc='left')
axB.legend(fontsize=6, frameon=False, loc='upper left', title="binary item", title_fontsize=6)
axB.set_xlim(0, 50)

fig.suptitle("Loading ≠ information: the model weights items by what they reveal, not by how strongly they load",
             fontsize=9, fontweight='bold', y=1.0)
fig.text(0.5, -0.065,
         "(a) Each point is one indicator: its loading on its home axis (on that likelihood family's natural link scale, so cross-family magnitudes are not directly comparable) versus the "
         "Fisher information it contributes to that axis score in the common latent metric — the quantity that IS comparable, computed exactly per family at the population mean. Continuous "
         "markers with moderate loadings dominate; two substance flags load near 0.95 yet contribute ~0.01, a hundred-fold less than a mid-loading continuous item. (b) For a binary item, "
         "information is λ²·p(1−p): at ~1% endorsement it collapses regardless of loading. This is why the substance axis, nominally carried by two high-loading flags, is in fact near-empty — "
         "and why an information-based reading, not a loading-based one, is the correct lens on a mixed-type measurement model.",
         ha='center', fontsize=5.1, color='#555', wrap=True)
fig.subplots_adjust(top=0.86, bottom=0.245, left=0.085, right=0.985)
fig.savefig("fig2_information.png", dpi=300, bbox_inches='tight')
fig.savefig("fig2_information.pdf", bbox_inches='tight')
