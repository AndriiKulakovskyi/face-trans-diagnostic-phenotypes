"""Localization / uncertainty figures for the Results section (continuous Gaussian-copula block).

Fig A (fig_localization): posterior position uncertainty collapses as instruments accumulate.
Fig B (fig_mincount):     the closed-form cut-off for a minimal number of indicators per factor.

Both use the real reported objects: loadings Lambda (continuous block, 88 standardized
indicators), factor correlations Phi, residual variances sigma_j^2 = 1 - communality_j.
The continuous block is used because there the EAP update is exactly Gaussian/closed-form
(eq:m-scores); the 21 binary/ordinal indicators add further (weaker) information.

    S(m)^-1 = Phi^-1 + sum_{j<=m} lambda_j lambda_j^T / sigma_j^2      (information adds)

Run from the article/ directory:  python make_localization_figs.py
"""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.optimize import nnls
from scipy.stats import chi2
from scipy.spatial import ConvexHull

ROOT = os.path.expanduser("~/Desktop/face-common-bp-sz-dr")
OUT  = os.path.join(ROOT, "article", "figures")
FACT = ["overall_severity","cognition","immunometabolic","sleep",
        "mania_activation","suicidality","developmental_risk","substance"]
NICE = {"overall_severity":"G burden","cognition":"cognition","immunometabolic":"immuno",
        "sleep":"sleep","mania_activation":"mania","suicidality":"suicid.",
        "developmental_risk":"develop.","substance":"substance"}

# ---- real continuous-block objects -------------------------------------------
ld   = pd.read_csv(f"{ROOT}/reports/copula_8factor_loadings.csv")
cont = ld[ld["block"] == "continuous"]
Lam  = (cont.pivot_table(index="item", columns="factor", values="loading", fill_value=0.0)
            .reindex(columns=FACT, fill_value=0.0))
items = Lam.index.tolist(); Lam = Lam.values                       # J x 8
Phi  = pd.read_csv(f"{ROOT}/reports/copula_8factor_phi.csv", index_col=0)\
         .reindex(index=FACT, columns=FACT).values
comm = np.einsum("ij,jk,ik->i", Lam, Phi, Lam)
sig2 = np.clip(1.0 - comm, 0.05**2, None)
J    = len(items)
Phi_inv = np.linalg.inv(Phi)
iG, iIM = FACT.index("overall_severity"), FACT.index("immunometabolic")

ap = pd.read_csv(f"{ROOT}/results/face/strata_oop/consolidate/archetype_profiles.csv")
ap = ap[ap["arm"] == "A_all9"].sort_values("archetype")
Z  = ap[FACT].values                                               # 5 corners x 8
TRUE = 2                                                           # A2 immunometabolic
# an A2-dominant interior blend: clearly inside the simplex (off the A2 vertex),
# biology-led so the continuous block localizes it well
w_true = np.array([0.05, 0.10, 0.70, 0.05, 0.10])                  # A0,A1,A2,A3,A4
f_star = Z.T @ w_true                                             # inside the simplex

# ---- simulate one patient; reveal continuous instruments in random order ------
rng   = np.random.default_rng(7)
x_obs = Lam @ f_star + rng.normal(0, np.sqrt(sig2))
order = rng.permutation(J)

def posterior(m):
    P, b = Phi_inv.copy(), np.zeros(8)
    if m > 0:
        C = order[:m]; Lc = Lam[C]; w = 1.0/sig2[C]
        P = P + (Lc.T * w) @ Lc; b = (Lc.T * w) @ x_obs[C]
    S = np.linalg.inv(P); return S @ b, S

ms = np.arange(0, J+1)
mean = np.zeros((J+1,8)); Sd = np.zeros((J+1,8)); gsd = np.zeros(J+1); Sall=[]
for m in ms:
    mu,S = posterior(m); mean[m]=mu; Sd[m]=np.sqrt(np.diag(S))
    gsd[m]=np.exp(0.5*np.mean(np.log(np.diag(S)))); Sall.append(S)

def simplex_w(x):
    A = np.vstack([Z.T, 1e3*np.ones(len(Z))]); bb = np.concatenate([x,[1e3]])
    w,_ = nnls(A, bb); s = w.sum()
    return w/s if s>0 else np.ones(len(Z))/len(Z)

true_mem = simplex_w(f_star)                                       # exact membership of the patient
tw_m,tw_lo,tw_hi,ent = [],[],[],[]
for m in ms:
    wm = simplex_w(mean[m]); ent.append(-(wm*np.log(wm+1e-12)).sum())
    smp = rng.multivariate_normal(mean[m], Sall[m], size=150)
    tw  = np.array([simplex_w(s)[TRUE] for s in smp])
    tw_m.append(tw.mean()); tw_lo.append(np.percentile(tw,10)); tw_hi.append(np.percentile(tw,90))
tw_m,tw_lo,tw_hi,ent = map(np.array,(tw_m,tw_lo,tw_hi,ent))

# ============================ FIGURE 1: localization ===========================
plt.rcParams.update({"font.size":10.5,"axes.titlesize":11,"axes.labelsize":10.5,
                     "axes.titleweight":"bold"})
fig = plt.figure(figsize=(15.5, 7.8))
gs  = fig.add_gridspec(2, 3, width_ratios=[1.7,1,1], hspace=0.42, wspace=0.30)

# --- A: localization funnel ---
axA = fig.add_subplot(gs[:,0], projection="3d")
levels = [l for l in [0,1,2,3,5,8,13,21,34,55,J] if l <= J]
th = np.linspace(0,2*np.pi,80); k = np.sqrt(chi2.ppf(0.95,2))
cmap=cm.viridis; norm=Normalize(0,J); Xs,Ys,Zs=[],[],[]
for m in levels:
    S2=Sall[m][np.ix_([iG,iIM],[iG,iIM])]; L=np.linalg.cholesky(S2)
    pts=mean[m,[iG,iIM]][:,None]+k*(L@np.vstack([np.cos(th),np.sin(th)]))
    axA.plot(pts[0],pts[1],zs=m,color=cmap(norm(m)),lw=2.0)
    Xs.append(pts[0]);Ys.append(pts[1]);Zs.append(np.full_like(th,m))
Xs,Ys,Zs=map(np.array,(Xs,Ys,Zs))
axA.plot_surface(Xs,Ys,Zs,facecolors=cm.viridis(norm(Zs)),alpha=0.16,linewidth=0,shade=False)
axA.plot(mean[:,iG],mean[:,iIM],zs=ms,color="crimson",lw=1.5,alpha=0.9)
# simplex (convex hull of 5 corners) on the (G, immuno) plane at the top
cg = Z[:,[iG,iIM]]; hull = ConvexHull(cg); vp = np.append(hull.vertices, hull.vertices[0])
axA.plot(cg[vp,0], cg[vp,1], zs=J, color="0.55", lw=1.2, ls="-")
for a,zc in enumerate(cg):
    axA.scatter([zc[0]],[zc[1]],[J],color="0.4",s=26)
    axA.text(zc[0],zc[1],J,f" A{a}",color="0.3",fontsize=8.5)
axA.scatter([f_star[iG]],[f_star[iIM]],[J],color="crimson",s=150,marker="*",
            edgecolor="k",zorder=12)
axA.plot([f_star[iG]]*2,[f_star[iIM]]*2,[0,J],color="crimson",ls=":",lw=1)
axA.set_xlabel("G  (overall burden)",labelpad=8); axA.set_ylabel("immunometabolic",labelpad=8)
axA.set_zlabel("instruments observed  m",labelpad=5); axA.set_zlim(0,J)
axA.set_box_aspect((1,1,1.4)); axA.view_init(elev=22,azim=-54)
axA.set_title("A   Localization funnel: the 95% position ellipse collapses\n"
              "to the true patient (★, inside the A0–A4 simplex)",loc="left",pad=0)
cb=fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=axA,location="left",pad=0.04,shrink=0.5)
cb.set_label("instruments m")

# --- B: anisotropy ---
axB=fig.add_subplot(gs[0,1:])
for d in range(8):
    n_d=int((Lam[:,d]**2>1e-4).sum())
    axB.plot(ms,Sd[:,d],lw=1.6,alpha=0.9,label=f"{NICE[FACT[d]]} ({n_d})")
axB.plot(ms,gsd,color="k",lw=2.6,ls=(0,(4,1)),label="all axes (geom.)")
axB.axhline(1.0,color="0.6",ls="--",lw=1); axB.text(J*0.6,1.015,"prior SD = 1",color="0.5",fontsize=8.5)
axB.set_xlim(0,J); axB.set_ylim(0,1.06)
axB.set_xlabel("instruments observed  m"); axB.set_ylabel("posterior SD per axis")
axB.set_title("B   Uncertainty is anisotropic: well-instrumented axes\n"
              "collapse first; thin factors stay uncertain",loc="left")
axB.legend(ncol=3,fontsize=7.6,loc="upper right",framealpha=.9); axB.grid(alpha=.25)

# --- C: archetyping ---
axC=fig.add_subplot(gs[1,1:])
axC.fill_between(ms,tw_lo,tw_hi,color="crimson",alpha=.16,label="10–90% posterior")
axC.plot(ms,tw_m,color="crimson",lw=2.3,label="P(corner A2)")
axC.axhline(true_mem[TRUE],color="0.35",ls=":",lw=1.4,label=f"true = {true_mem[TRUE]:.2f}")
axC.axhline(0.2,color="0.6",ls="--",lw=1); axC.text(J*0.02,0.215,"chance (1/5)",color="0.5",fontsize=8)
axC.set_ylim(0,1.0); axC.set_xlim(0,J)
axC.set_xlabel("instruments observed  m"); axC.set_ylabel("membership in true corner A2")
axC.set_title("C   Archetyping sharpens: membership estimate converges\n"
              "on the true blend; entropy of the 5 weights falls (dashed)",loc="left")
ax2=axC.twinx(); ax2.plot(ms,ent,color="navy",lw=1.5,ls="--",alpha=.8)
ax2.set_ylabel("weight entropy (nats)",color="navy"); ax2.tick_params(axis="y",colors="navy")
ax2.set_ylim(0,np.log(5)*1.05)
axC.legend(loc="lower right",fontsize=8.3,framealpha=.9); axC.grid(alpha=.25)

for ext in ("pdf","png"):
    fig.savefig(f"{OUT}/fig_localization.{ext}", dpi=200, bbox_inches="tight")
print("wrote fig_localization")

# ============================ FIGURE 2: min-count ==============================
fig2,(ax,axi) = plt.subplots(1,2,figsize=(12.6,5.7),
                             gridspec_kw=dict(width_ratios=[2.35,1],wspace=0.27))
# --- A: SD vs number of indicators, by loading; real factors overlaid ---
mm = np.arange(1,41)
for lam in [0.7,0.6,0.5,0.4,0.3]:
    sd = 1/np.sqrt(1+ mm*lam**2/(1-lam**2))
    ax.plot(mm, sd, lw=2, label=f"λ = {lam:.1f}")
ax.axhline(0.5,color="0.3",ls="--",lw=1.3)
ax.text(39.5,0.515,"SD = 0.50   (reliability 0.75,  I ≥ 3)",ha="right",fontsize=8.8,color="0.3")
ax.axhspan(0,0.5,color="seagreen",alpha=0.06)
ax.text(38,0.06,"acceptable",color="seagreen",ha="right",fontsize=10,style="italic")
# real continuous-block specific factors: (n_home, realized SD, dominant anchor, label offset)
real = {"immuno":(37,0.243,"BMI .95",(-2,-15)), "developmental":(12,0.281,"CTQ .93",(5,11)),
        "cognition":(11,0.267,"CVLT .89",(-2,-17)), "sleep":(9,0.306,"PSQI .88",(4,10)),
        "substance":(2,0.660,"max .72",(9,-3)), "mania":(2,0.783,"no anchor",(9,3))}
for nm,(n,sd,anc,off) in real.items():
    col = "seagreen" if sd<=0.5 else "firebrick"
    ax.scatter([n],[sd],color=col,marker="D",s=60,zorder=6,edgecolor="k",lw=0.6)
    ax.annotate(f"{nm}\n({anc})",(n,sd),textcoords="offset points",xytext=off,
                fontsize=8,ha="left",color=col,fontweight="bold")
ax.set_xlim(0,40); ax.set_ylim(0,1.0)
ax.set_xlabel("number of (standardized) indicators on the factor,  $m$")
ax.set_ylabel("posterior SD of the factor coordinate")
ax.set_title("A   Acceptable uncertainty needs information $I\\geq3$, not a fixed count\n"
             r"$\mathrm{SD}=1/\sqrt{1+I},\;\; I=\sum_j \lambda_j^2/(1-\lambda_j^2)$",
             loc="left",fontsize=10.5,fontweight="bold")
ax.legend(title="per-indicator loading",fontsize=8.6,loc="center right",framealpha=.92)
ax.grid(alpha=.25)
# --- B: per-indicator information is steeply nonlinear in the loading ---
ll = np.linspace(0.1,0.96,200); axi.plot(ll, ll**2/(1-ll**2), color="k", lw=2)
for lam in [0.3,0.5,0.7,0.95]:
    iv = lam**2/(1-lam**2); axi.scatter([lam],[iv],color="firebrick",s=34,zorder=5)
    axi.annotate(f"  i={iv:.1f}",(lam,iv),textcoords="offset points",
                 xytext=(4,-2 if lam<0.9 else -10),fontsize=8.3)
axi.axhline(3,color="seagreen",ls="--",lw=1.2); axi.text(0.12,3.25,"I = 3 bar",color="seagreen",fontsize=8.3)
axi.set_xlabel("indicator loading  $\\lambda$")
axi.set_ylabel("information per indicator  $i(\\lambda)$")
axi.set_title("B   one strong marker\noutweighs many weak ones",loc="left",fontsize=10.5,fontweight="bold")
axi.set_xlim(0.1,0.97); axi.set_ylim(0,9.8); axi.grid(alpha=.25)
for ext in ("pdf","png"):
    fig2.savefig(f"{OUT}/fig_mincount.{ext}", dpi=200, bbox_inches="tight")
print("wrote fig_mincount")
print("summary:", {m:(round(gsd[m],3)) for m in [0,3,8,21,J]}, "P(A2)_final", round(tw_m[-1],2))
