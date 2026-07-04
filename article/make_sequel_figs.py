"""Three Tier-1 sequel figures for the article.

  fig_voi          value-of-information / adaptive battery (greedy vs random)
  fig_simplexfog   membership-entropy "fog" over all 9,013 patients in the A=5 simplex
  fig_traitstate   trait/state thermometer (between/within/measurement variance per axis)

All from the real reported objects. Run from article/:  python make_sequel_figs.py
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

ROOT = os.path.expanduser("~/Desktop/face-common-bp-sz-dr")
OUT  = os.path.join(ROOT, "article", "figures")
FACT = ["overall_severity","cognition","immunometabolic","sleep",
        "mania_activation","suicidality","developmental_risk","substance"]
NICE = {"overall_severity":"G burden","cognition":"cognition","immunometabolic":"immuno",
        "sleep":"sleep","mania_activation":"mania","suicidality":"suicid.",
        "developmental_risk":"develop.","substance":"substance"}
summary = {}

# =====================================================================================
# FIGURE 1 -- value of information / adaptive battery
# =====================================================================================
ld   = pd.read_csv(f"{ROOT}/reports/copula_8factor_loadings.csv")
cont = ld[ld["block"]=="continuous"]
Lw   = cont.pivot_table(index="item", columns="factor", values="loading", fill_value=0.0)\
           .reindex(columns=FACT, fill_value=0.0)
items = Lw.index.tolist(); Lam = Lw.values
Phi  = pd.read_csv(f"{ROOT}/reports/copula_8factor_phi.csv", index_col=0)\
         .reindex(index=FACT, columns=FACT).values
sig2 = np.clip(1.0 - np.einsum("ij,jk,ik->i", Lam, Phi, Lam), 0.05**2, None)
home = cont.drop_duplicates("item").set_index("item")["home"].reindex(items).values
J    = len(items)

def genSD(S):                                   # volume^(1/2) per dim = exp(logdet/(2*8))
    return np.exp(np.linalg.slogdet(S)[1] / (2*8))

def run_order(order):
    S = Phi.copy(); g = [genSD(S)]
    for j in order:
        Sl = S @ Lam[j]; denom = sig2[j] + Lam[j] @ Sl
        S = S - np.outer(Sl, Sl) / denom; g.append(genSD(S))
    return np.array(g)

# greedy: pick the indicator with the largest log-volume reduction at each step
S = Phi.copy(); remaining = list(range(J)); greedy_order = []; greedy_gain = []
for _ in range(J):
    best, bestgain = None, -1
    for j in remaining:
        Sl = S @ Lam[j]; gain = np.log1p((Lam[j] @ Sl) / sig2[j])
        if gain > bestgain: bestgain, best = gain, j
    Sl = S @ Lam[best]; S = S - np.outer(Sl, Sl) / (sig2[best] + Lam[best] @ Sl)
    # expected information gain (Gaussian entropy reduction) = 1/2 log(1 + SNR)
    greedy_order.append(best); greedy_gain.append(0.5*bestgain); remaining.remove(best)
g_greedy = run_order(greedy_order)

rng = np.random.default_rng(0)
rand = np.array([run_order(rng.permutation(J)) for _ in range(200)])
r_med, r_lo, r_hi = np.median(rand,0), np.percentile(rand,10,0), np.percentile(rand,90,0)
g_worst = run_order(greedy_order[::-1])          # reverse-greedy ~ worst-first proxy

TARGET = 0.5
m_greedy = int(np.argmax(g_greedy <= TARGET))
m_rand   = int(np.argmax(r_med  <= TARGET))
summary["voi"] = dict(m_greedy=m_greedy, m_rand=m_rand, floor=round(g_greedy[-1],3))

fig, (axA, axB) = plt.subplots(1,2, figsize=(13.4,5.4),
                               gridspec_kw=dict(width_ratios=[1.25,1], wspace=0.26))
ms = np.arange(J+1)
axA.fill_between(ms, r_lo, r_hi, color="0.7", alpha=.35, label="random (10–90%)")
axA.plot(ms, r_med, color="0.45", lw=2, label="random (median)")
axA.plot(ms, g_greedy, color="crimson", lw=2.6, label="adaptive (max-information)")
axA.plot(ms, g_worst, color="steelblue", lw=1.6, ls=":", label="worst-first")
axA.axhline(TARGET, color="seagreen", ls="--", lw=1.2)
axA.annotate(f"adaptive: {m_greedy} tests", (m_greedy,TARGET), textcoords="offset points",
             xytext=(6,8), color="crimson", fontsize=9, fontweight="bold")
axA.annotate(f"random: {m_rand} tests", (m_rand,TARGET), textcoords="offset points",
             xytext=(6,-14), color="0.4", fontsize=9)
axA.plot([m_greedy],[TARGET],"o",color="crimson"); axA.plot([m_rand],[TARGET],"o",color="0.45")
axA.text(J*0.62, TARGET+0.012, "target SD = 0.50", color="seagreen", fontsize=9)
axA.set_xlim(0,J); axA.set_ylim(0.3,1.02)
axA.set_xlabel("number of instruments administered"); axA.set_ylabel("overall position uncertainty (geom.\\ SD)")
axA.set_title("A   Choosing the next test by information reaches\nthe same precision in far fewer instruments",
              loc="left", fontsize=10.5, fontweight="bold")
axA.legend(fontsize=8.6, loc="upper right", framealpha=.92); axA.grid(alpha=.25)

# Panel B: the first greedy picks as a battery, coloured by home factor
NB = 14
def hof(j):
    h = home[j]
    return h if isinstance(h, str) else "overall_severity"
fac_order = [hof(j) for j in greedy_order[:NB]]
palette = {f: cm.tab10(k) for k,f in enumerate(FACT)}
bars = axB.barh(range(NB), greedy_gain[:NB][::-1],
                color=[palette.get(f, (.6,.6,.6,1)) for f in fac_order[::-1]], edgecolor="k", lw=.4)
labels = [f"{items[greedy_order[i]][:16]}  ({NICE.get(hof(greedy_order[i]),'?')})" for i in range(NB)][::-1]
axB.set_yticks(range(NB)); axB.set_yticklabels(labels, fontsize=7.6)
axB.set_xlabel("information gain  $\\frac{1}{2}\\log(1+\\lambda^{\\top}\\!S\\lambda/\\sigma^2)$ (nats)")
axB.set_title("B   The adaptive battery diversifies across axes\n(pick #1…#%d, coloured by home factor)" % NB,
              loc="left", fontsize=10.5, fontweight="bold")
axB.grid(alpha=.25, axis="x")
for ext in ("pdf","png"): fig.savefig(f"{OUT}/fig_voi.{ext}", dpi=200, bbox_inches="tight")
print("wrote fig_voi", summary["voi"])

# =====================================================================================
# FIGURE 2 -- the simplex "fog": membership entropy over all patients
# =====================================================================================
ps = pd.read_parquet(f"{ROOT}/results/face/strata_oop/consolidate/patient_strata.parquet")
W  = ps[[f"arch_w{a}" for a in range(5)]].values
W  = W / W.sum(1, keepdims=True)
H  = -(W*np.log(np.clip(W,1e-12,None))).sum(1)         # entropy, max = ln5
maxw = W.max(1)
summary["fog"] = dict(medH=round(float(np.median(H)),3), lnK=round(float(np.log(5)),3),
                      frac_interior=round(float((maxw<0.5).mean()),3),
                      frac_corner=round(float((maxw>0.8).mean()),3))

ang = np.pi/2 + np.arange(5)*2*np.pi/5
V   = np.c_[np.cos(ang), np.sin(ang)]                  # pentagon vertices
P   = W @ V                                            # patient 2-D positions
Anm = ["A0 activation/\nsleep","A1 severe\nclean-biology","A2 immuno-\nmetabolic",
       "A3 trauma/\nsuicidality","A4 low-burden/\nwell"]

fig2 = plt.figure(figsize=(12.6,5.6))
gs = fig2.add_gridspec(1,2, width_ratios=[1.25,1], wspace=0.22)
ax = fig2.add_subplot(gs[0,0])
# Colour each patient by its DOMINANT (argmax) archetype so the directional lean of the
# cloud toward each corner is visible; the cloud still fills the interior (the "fog"),
# but now one can read WHICH corner each patient tilts toward. Mixing entropy itself is
# shown quantitatively in panel B. Palette is colourblind-safe (Okabe-Ito) and matches
# the corner markers/labels 1:1.
dom  = W.argmax(1)                                     # dominant archetype per patient
ARCH_C = {0:"#CC79A7", 1:"#E69F00", 2:"#D55E00", 3:"#0072B2", 4:"#009E73"}
from numpy.random import default_rng

_ord = default_rng(0).permutation(len(P))              # shuffle so no colour is drawn last on top
ax.scatter(P[_ord,0], P[_ord,1], c=[ARCH_C[d] for d in dom[_ord]],
           s=4, alpha=.45, linewidths=0, zorder=2)
poly = np.vstack([V, V[0]]); ax.plot(poly[:,0], poly[:,1], color="0.55", lw=1.1, zorder=4)
for a in range(5):
    ax.scatter(*V[a], color=ARCH_C[a], s=55, zorder=6, edgecolor="black", linewidth=.7)
    dx,dy = V[a]*1.15
    ax.text(dx, dy, Anm[a], ha="center", va="center", fontsize=8.3, fontweight="bold",
            color=ARCH_C[a])
ax.set_aspect("equal"); ax.axis("off"); ax.set_ylim(-1.5, 1.62)
ax.set_title("A   Patients as blends of the five archetypes",
             loc="left", fontsize=10.5, fontweight="bold", pad=14)
_leg = [Line2D([0],[0], marker="o", ls="", markerfacecolor=ARCH_C[a], markeredgecolor="none",
               markersize=6, label=Anm[a].replace("\n"," ").replace("- ","-")) for a in range(5)]
ax.legend(handles=_leg, title="dominant archetype", fontsize=6.6, title_fontsize=7,
          loc="lower left", bbox_to_anchor=(-0.02,-0.02), frameon=False, handletextpad=.3,
          labelspacing=.3)
axh = fig2.add_subplot(gs[0,1])
axh.hist(H, bins=60, color="slateblue", alpha=.85)
axh.axvline(np.median(H), color="crimson", lw=2, label=f"median {np.median(H):.2f}")
axh.axvline(np.log(5), color="0.4", ls="--", lw=1.2, label=f"max = ln 5 = {np.log(5):.2f}")
axh.set_xlabel("entropy of a patient's archetype weights (nats)"); axh.set_ylabel("patients")
axh.set_title("B   Most patients are interior blends, not corners\n"
              f"{100*summary['fog']['frac_interior']:.0f}% no-majority blend; "
              f"only {100*summary['fog']['frac_corner']:.1f}% near one corner",
              loc="left", fontsize=10.5, fontweight="bold")
axh.legend(fontsize=8.6); axh.grid(alpha=.25)
for ext in ("pdf","png"): fig2.savefig(f"{OUT}/fig_simplexfog.{ext}", dpi=200, bbox_inches="tight")
print("wrote fig_simplexfog", summary["fog"])

# =====================================================================================
# FIGURE 3 -- trait/state thermometer
# =====================================================================================
ts = pd.read_csv(f"{ROOT}/results/face/temporal_oop/trait_state/trait_state.csv")
ts = ts.sort_values("icc", ascending=True).reset_index(drop=True)   # ascending so trait at top
tot = ts[["var_between","var_within","var_meas"]].sum(1)
fb, fw, fm = (ts["var_between"]/tot, ts["var_within"]/tot, ts["var_meas"]/tot)
names = [{"overall_severity":"general burden (G)","cognition":"cognition","immunometabolic":"immunometabolic",
          "sleep":"sleep","mania_activation":"mania/activation","suicidality":"suicidality",
          "developmental_risk":"developmental risk","substance":"substance"}[a] for a in ts["axis"]]

fig3, ax = plt.subplots(figsize=(9.6,5.4)); y = np.arange(len(ts))
ax.barh(y, fb, color="#c0392b", label="between-patient (trait)")
ax.barh(y, fw, left=fb, color="#2980b9", label="within-patient (state)")
ax.barh(y, fm, left=fb+fw, color="0.75", label="measurement error")
for i in range(len(ts)):
    tag = "  trait" if ts["icc"][i]>=0.6 else ("  state" if ts["icc"][i]<0.45 else "  mixed")
    extra = "  (data-limited)" if ts["verdict"][i]=="uninformative" else ""
    ax.text(1.01, y[i], f"ICC {ts['icc'][i]:.2f}{tag}{extra}", va="center", fontsize=8.6)
ax.axvline(0.5, color="k", ls=":", lw=1, alpha=.5)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
ax.set_xlim(0,1); ax.set_xlabel("share of an axis's total variance")
ax.set_title("Trait/state thermometer: durable biology vs moving symptoms\n"
             "(error-corrected trait fraction = ICC = between / (between + within))",
             fontsize=11, fontweight="bold", loc="left")
ax.legend(ncol=3, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5,-0.22), framealpha=.9)
ax.margins(x=0); plt.subplots_adjust(right=0.80)
for ext in ("pdf","png"): fig3.savefig(f"{OUT}/fig_traitstate.{ext}", dpi=200, bbox_inches="tight")
print("wrote fig_traitstate")
print("ICC by axis:", {a:round(i,2) for a,i in zip(ts['axis'],ts['icc'])})
print("SUMMARY", summary)
