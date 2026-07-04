"""
Figure-generation code for FACE-ATLAS: fig_worked_patient.png

Provenance: extracted verbatim from artifact lineage (version_id 0bbd7815-3d9f-4ee1-823b-29a03c883f40).
Environment: face-dev
NOTE: these figures were produced in a shared `face-dev` kernel session in which the
fitted GLLVM model state (results/face/gllvm_oop/s8_full/model_state.pt) and derived
arrays (loadings, sigmas, families, coordinates) were loaded once and reused across
cells. This file is the exact producing cell; if run standalone it may require that
shared setup (model load + per-family Fisher-information arrays) to be present.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

# apply_figure_style (from figure-style skill)
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

# Load data
coords = pd.read_parquet("/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr/results/face/gllvm_oop/consolidate/coordinates.parquet")
aa = pd.read_csv("/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr/results/face/prognosis_oop/endpoints/archetype_atlas.csv")
L = pd.read_csv("/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr/results/face/gllvm_oop/consolidate/loadings_summary.csv")
strata = pd.read_parquet("/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr/results/face/strata_oop/consolidate/patient_strata.parquet")

factors = ["overall_severity","cognition","immunometabolic","sleep","suicidality","developmental_risk","mania_activation","substance"]

# Build gload and prim from loadings
gload = L[L.kind=="bifactor_G"].set_index("item")["loading"].to_dict()
prim = L[L.kind=="primary"].copy()

# Compute info curves helper
def item_curve(factor, cap=6):
    it = prim[prim.factor==factor].copy()
    it["lG"] = it["item"].map(gload).fillna(0)
    it["ls"] = it.loading.astype(float)
    it["psi"] = (1 - it["lG"]**2 - it["ls"]**2).clip(lower=0.05)
    it["iinfo"] = it["ls"]**2 / it["psi"]
    it = it.sort_values("iinfo", ascending=False).head(cap)
    prec = 1 + it["iinfo"].cumsum()
    return list(range(1, len(it)+1)), list(1-1/prec), list(it["item"])

# Patient profile
pid = ("bp", "62162")
row = coords.loc[pid]
prof = []
for f in factors:
    prof.append(dict(factor=f,
        mean=float(pd.to_numeric(row[f+"__mean"])),
        sd=float(pd.to_numeric(row[f+"__sd"])),
        lo=float(pd.to_numeric(row[f+"__hdi_low"])),
        hi=float(pd.to_numeric(row[f+"__hdi_high"])),
        nobs=int(pd.to_numeric(row[f+"__n_obs"]))))
prof = pd.DataFrame(prof)

s0 = strata.set_index(["cohort","patient_id"]).loc[pid]

# EGF remission data
egf = aa[aa.outcome=="egf"]
cell = egf[(egf.cohort=="bp") & (egf.archetype==2)].iloc[0]
bp_all = egf[egf.cohort=="bp"]
bp_avg = bp_all.n_rem.sum() / bp_all.n.sum()
bp = aa[(aa.outcome=="egf") & (aa.cohort=="bp")].set_index("archetype")
arch_order_bp = bp.remission_rate.sort_values().index.tolist()

disp = {"overall_severity":"General burden","cognition":"Cognition","immunometabolic":"Immunometabolic",
        "sleep":"Sleep","suicidality":"Suicidality","developmental_risk":"Developmental risk",
        "mania_activation":"Mania/activation","substance":"Substance"}

prof["disp"] = prof.factor.map(disp)
prof2 = prof.sort_values("mean").reset_index(drop=True)

fig = plt.figure(figsize=(10,6.4))
gs = GridSpec(2,2,figure=fig,hspace=0.52,wspace=0.34,height_ratios=[1,0.9])

axA = fig.add_subplot(gs[0,0])
for i,r in prof2.iterrows():
    obs = r.nobs>0; col = "#c44e52" if r.factor=="immunometabolic" else ("#4c72b0" if obs else "#bbbbbb")
    axA.plot([r.lo,r.hi],[i,i],color=col,lw=2 if obs else 1.4,alpha=0.85,zorder=2,ls='-' if obs else ':')
    axA.scatter(r["mean"],i,s=34,color=col,zorder=3,edgecolor='white',linewidth=0.5)
    axA.text(r.hi+0.15,i,f"{r.nobs} items" if obs else "no data (prior)",fontsize=5.5,va='center',color=col if obs else "#999")
axA.axvline(0,color='#888',ls='--',lw=0.8)
axA.set_yticks(range(len(prof2))); axA.set_yticklabels(prof2.disp,fontsize=6.5)
axA.set_xlabel("Posterior latent score (SD units, 95% HDI)"); axA.set_xlim(-2.6,5.3)
axA.set_title("A. Where this patient sits — with honest uncertainty",fontsize=7.6,loc='left')

axB = fig.add_subplot(gs[0,1])
w = [s0[f"arch_w{i}"] for i in range(5)]
anames = ["A0 ↑sleep/mania","A1 ↑burden","A2 ↑immuno","A3 ↑dev/suic","A4 low-sev"]
bc = ["#bbbbbb","#bbbbbb","#c44e52","#bbbbbb","#bbbbbb"]
axB.barh(range(5),w,color=bc,zorder=3)
axB.set_yticks(range(5)); axB.set_yticklabels(anames,fontsize=6.5); axB.invert_yaxis()
axB.set_xlabel("Archetype weight"); axB.set_xlim(0,1)
axB.set_title("B. Archetype mixture: 86% immunometabolic pole",fontsize=7.6,loc='left')
axB.text(0.85,2,"86%",fontsize=8,color="#c44e52",fontweight='bold',va='center',ha='right')

axC = fig.add_subplot(gs[1,0])
albl = {0:"A0",1:"A1",2:"A2",3:"A3",4:"A4"}
for j,a in enumerate(arch_order_bp):
    rr_ = bp.loc[a]; hl = (a==2)
    axC.barh([j],[rr_.remission_rate*100],color="#c44e52" if hl else "#cccccc",zorder=3,height=0.62,
             xerr=[[rr_.remission_rate*100-rr_.rem_lo*100],[rr_.rem_hi*100-rr_.remission_rate*100]] if hl else None,
             error_kw=dict(lw=1,capsize=2))
    lab = f"{rr_.remission_rate:.0%}  ◄ this patient" if hl else f"{rr_.remission_rate:.0%}"
    axC.text(rr_.rem_hi*100+2 if hl else rr_.remission_rate*100+1.5,j,lab,fontsize=6,va='center',
             color="#c44e52" if hl else "#888",fontweight='bold' if hl else 'normal')
axC.set_yticks(range(5)); axC.set_yticklabels([albl[a] for a in arch_order_bp],fontsize=6.5)
axC.set_xlabel("Predicted EGF remission within bipolar (%)"); axC.set_xlim(0,85)
axC.set_title("C. Prognosis: lowest-remitting archetype within BP (27% vs 73%)",fontsize=7.2,loc='left')

axD = fig.add_subplot(gs[1,1])
kk, rr, items = item_curve("cognition")
axD.plot([0]+kk,[0]+rr,marker='o',ms=4,color="#4c72b0",zorder=3)
axD.set_xlabel("Cognition items to administer next"); axD.set_ylabel("Cognition reliability")
axD.set_title("D. Next best data to collect (currently 0 items)",fontsize=7.6,loc='left')
axD.set_ylim(0,0.9); axD.set_xlim(0,6)
axD.annotate(f"first item ({items[0]})\n→ reliability {rr[0]:.2f}",xy=(1,rr[0]),xytext=(1.7,0.22),
             fontsize=5.8,color="#4c72b0",arrowprops=dict(arrowstyle="->",color="#4c72b0",lw=0.7))

# Title and bottom caption paragraph intentionally removed — that text now lives in the
# LaTeX figure caption (\caption{} for fig:localization in sections/03_results.tex).
fig.subplots_adjust(top=0.955,bottom=0.075,left=0.13,right=0.97)
fig.savefig("fig_worked_patient.png",dpi=200,bbox_inches='tight')
