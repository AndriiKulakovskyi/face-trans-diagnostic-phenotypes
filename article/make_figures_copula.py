#!/usr/bin/env python3
"""
FACE-ATLAS — publication figure system (Gaussian-copula vertical).

Regenerates all main + Extended Data figures directly from the canonical
copula results under results/face/{strata_oop,temporal_oop,prognosis_oop,
treatment_oop,m4_repbench} and reports/. Uniform style, colorblind-safe
(Okabe-Ito), vector PDF + 300-dpi PNG into article/figures/.

Run:  cd <repo root>;  python article/make_figures_copula.py
"""
import json, os, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "results", "face", *p)
REP = lambda *p: os.path.join(ROOT, "reports", *p)
OUT = os.path.join(ROOT, "article", "figures")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------- style
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", verm="#D55E00", purple="#CC79A7",
          grey="#7F7F7F", lgrey="#D9D9D9")
COH = dict(bp=OI["blue"], sz=OI["orange"], dr=OI["green"])
COH_LBL = dict(bp="Bipolar", sz="Schizophrenia", dr="Depression")
# archetype palette: infographic-like + colorblind-safe.
# A0 biological = orange (matches infographic), A1 low-burden = green (matches infographic),
# A2 severe-nonbio = blue, A3 symptom = reddish-purple (accessible, mutually distinct).
ARCH_C = {0: OI["orange"], 1: OI["green"], 2: OI["blue"], 3: OI["purple"]}
ARCH_N = {0: "Biological", 1: "Low-burden", 2: "Severe /\nnon-biological", 3: "Symptom"}
ARCH_N1 = {0: "Biological", 1: "Low-burden", 2: "Severe/non-bio", 3: "Symptom"}
BIO = OI["verm"]   # biology highlight

AXES9 = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep",
         "mania_activation", "suicidality", "developmental_risk", "substance"]
AXLAB = {"overall_severity": "General\nburden (G)", "cognition": "Cognition",
         "metabolic": "Metabolic", "inflammatory": "Inflammatory", "sleep": "Sleep",
         "mania_activation": "Mania", "suicidality": "Suicidality",
         "developmental_risk": "Developmental", "substance": "Substance"}
AXLAB1 = {k: v.replace("\n", " ") for k, v in AXLAB.items()}

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.edgecolor": "#444444",
    "xtick.color": "#444444", "ytick.color": "#444444",
    "axes.titleweight": "bold", "legend.frameon": False, "legend.fontsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

def panel(ax, lab, x=-0.16, y=1.06, size=13):
    ax.text(x, y, lab, transform=ax.transAxes, fontsize=size, fontweight="bold",
            va="top", ha="left")

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("  wrote", name)

# ============================================================================ FIG 1
def fig1_overview():
    fig = plt.figure(figsize=(11.0, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.32, 1.0], hspace=0.36)

    # --- top: the four-layer arc -------------------------------------------------
    axT = fig.add_subplot(gs[0]); axT.axis("off"); axT.set_xlim(0, 100); axT.set_ylim(0, 34)
    layers = [
        ("Diagnostic\ncohorts", "BP 6,252 · SZ 2,209\nDR 552  ·  N = 9,013", OI["lgrey"], "#333333"),
        ("Transdiagnostic\ndimensions", "1 general (G) + 8\nspecific axes (M1)", OI["sky"], "#08306b"),
        ("Continuous map +\nA = 4 archetypes", "continuum, no\nprivileged K (M2)", OI["green"], "#00441b"),
        ("Prognosis /\ntreatment", "2-yr functioning;\nTAU boundary (M4–M5)", OI["orange"], "#7f2704"),
    ]
    x0, w, gap, yc, h = 4, 19.0, 5.7, 17, 15
    for i, (t, sub, c, tc) in enumerate(layers):
        x = x0 + i * (w + gap)
        box = FancyBboxPatch((x, yc - h/2), w, h, boxstyle="round,pad=0.5,rounding_size=1.4",
                             fc=c, ec="none", alpha=0.92, mutation_aspect=0.7)
        axT.add_patch(box)
        axT.text(x + w/2, yc + 3.0, t, ha="center", va="center", fontsize=10.5,
                 fontweight="bold", color=tc)
        axT.text(x + w/2, yc - 3.6, sub, ha="center", va="center", fontsize=8.2, color=tc)
        if i < 3:
            axT.add_patch(FancyArrowPatch((x + w + 0.6, yc), (x + w + gap - 0.6, yc),
                          arrowstyle="-|>", mutation_scale=16, lw=2.2, color="#555555"))
    axT.text(50, 31.5, "A latent atlas of severe mental illness", ha="center",
             fontsize=12.5, fontweight="bold")
    axT.text(50, 1.5, "diagnosis enters only as a held-out validation label — never as a model input",
             ha="center", fontsize=8.5, style="italic", color="#666666")

    # --- bottom-left: three invariants ------------------------------------------
    axL = fig.add_subplot(gs[1, 0]); axL.axis("off")
    axL.set_xlim(0, 100); axL.set_ylim(0, 24)
    axL.text(2, 22.5, "Three load-bearing invariants", fontsize=10, fontweight="bold")
    inv = ["No imputation — the model is fit to each patient's observed cells only "
           "(mean cell missingness 39.8%, preserved).",
           "Diagnosis is metadata — cohort and DSM-5 subtype are covariates / validation labels, never indicators.",
           "Baseline (V0) defines the map — the 12- and 24-month visits validate it; no discovery on follow-up."]
    for i, s in enumerate(inv):
        y = 18.4 - i * 4.3
        axL.add_patch(plt.Circle((3.2, y + 0.4), 0.7, color=OI["green"]))
        axL.text(3.2, y + 0.4, "✓", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        axL.text(5.6, y + 0.4, s, va="center", fontsize=8.6)
    # five questions strip
    axL.text(2, 4.2, "Five questions, increasingly hard:", fontsize=9, fontweight="bold")
    qs = ["exists", "organizes", "persists", "predicts", "guides Tx?"]
    qx = 30
    for i, q in enumerate(qs):
        x = qx + i * 14.0
        c = OI["green"] if i < 4 else OI["verm"]
        axL.add_patch(FancyBboxPatch((x, 1.6), 12.0, 4.0, boxstyle="round,pad=0.2,rounding_size=0.8",
                      fc="white", ec=c, lw=1.6))
        axL.text(x + 6.0, 3.6, q, ha="center", va="center", fontsize=8.4, fontweight="bold", color=c)
        if i < 4:
            axL.annotate("", (x + 13.4, 3.6), (x + 12.1, 3.6),
                         arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#888888"))
    save(fig, "fig1_overview")

# ============================================================================ FIG 2
def fig2_map():
    L = pd.read_csv(REP("11_s5_9dim_loadings.csv"))
    # build |loading| matrix: rows = indicators grouped by home factor; cols = 9 axes
    prim = L[L["factor"] == L["home"]].copy()  # primary rows for ordering
    order_axes = AXES9
    # group indicators by home factor in axis order, keep those with a decent primary loading
    rows = []
    for ax_ in order_axes:
        sub = prim[prim["home"] == ax_].copy()
        sub["abs"] = sub["loading"].abs()
        sub = sub.sort_values("abs", ascending=False).head(7)  # cap per block for legibility
        for _, r in sub.iterrows():
            rows.append((r["item"], ax_))
    items = [r[0] for r in rows]
    M = np.zeros((len(items), len(order_axes)))
    Lkey = {(r["item"], r["factor"]): r["loading"] for _, r in L.iterrows()}
    for i, (it, _) in enumerate(rows):
        for j, ax_ in enumerate(order_axes):
            M[i, j] = abs(Lkey.get((it, ax_), 0.0))

    fig = plt.figure(figsize=(12.0, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.62)

    axA = fig.add_subplot(gs[0])
    im = axA.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1.0)
    axA.set_xticks(range(len(order_axes)))
    axA.set_xticklabels([AXLAB1[a] for a in order_axes], fontsize=7.2, rotation=35, ha="right")
    # block separators + home-factor row labels
    boundaries, c0 = [], 0
    for ax_ in order_axes:
        n = sum(1 for _, h in rows if h == ax_)
        if n:
            axA.text(-0.9, c0 + n/2 - 0.5, AXLAB1[ax_], ha="right", va="center",
                     fontsize=7.4, fontweight="bold", color="#333333")
            c0 += n; boundaries.append(c0 - 0.5)
    for b in boundaries[:-1]:
        axA.axhline(b, color="white", lw=1.4)
    axA.set_yticks([]); axA.set_ylabel("indicators (grouped by home dimension)", fontsize=8.5)
    axA.set_title("Posterior loading atlas", loc="center")
    cb = fig.colorbar(im, ax=axA, fraction=0.045, pad=0.02); cb.set_label("|posterior loading|", fontsize=8)
    panel(axA, "a", x=-0.28)

    # --- Phi heatmap (continuous backbone + explicit, 7-dim certified) -----------
    P = pd.read_csv(REP("04_stage5_phi.csv"), index_col=0)
    axB = fig.add_subplot(gs[1])
    Pv = P.values.copy()
    lab = [AXLAB1.get(c, c) for c in P.columns]
    im2 = axB.imshow(Pv, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    axB.set_xticks(range(len(lab))); axB.set_xticklabels(lab, rotation=45, ha="right", fontsize=8)
    axB.set_yticks(range(len(lab))); axB.set_yticklabels(lab, fontsize=8)
    for i in range(len(lab)):
        for j in range(len(lab)):
            axB.text(j, i, f"{Pv[i,j]:.2f}", ha="center", va="center", fontsize=7.4,
                     color="white" if abs(Pv[i, j]) > 0.32 else "#222222")
    axB.set_title("Inter-factor correlations  Φ", loc="center")
    fig.colorbar(im2, ax=axB, fraction=0.045, pad=0.02)
    axB.text(0.5, -0.34, "G is orthogonal by construction (top row/col = 0);\nspecific–specific |Φ| ≈ 0.10, "
             "metabolic–inflammatory = 0.19", transform=axB.transAxes, ha="center", va="top", fontsize=7.3,
             color="#555555")
    panel(axB, "b", x=-0.30)
    save(fig, "fig2_map")

# ============================================================================ FIG 3
def fig3_biology_g():
    cg = pd.read_csv(REP("07_corrG_phi.csv")).set_index("domain")
    cf = pd.read_csv(REP("12_biology_g_confound.csv")).set_index("domain")
    order = ["sleep", "cognition", "metabolic", "inflammatory"]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    x = np.arange(len(order)); w = 0.36
    unадj = [cg.loc[d, "corrG_phi_with_G"] for d in order]
    adj = [cf.loc[d, "A2_antipsychotic"] for d in order]
    cols = [OI["grey"] if d in ("sleep", "cognition") else BIO for d in order]
    b1 = ax.bar(x - w/2, unадj, w, color=cols, label="unadjusted")
    b2 = ax.bar(x + w/2, adj, w, color=cols, alpha=0.45, hatch="//",
                edgecolor="white", label="+ medication, adiposity, site")
    # bifactor |lambda_G| as points
    lam = [cg.loc[d, "bifactor_loading_on_G"] for d in order]
    ax.scatter(x - w/2, lam, marker="D", s=34, color="black", zorder=5, label="bifactor |λ$_G$|")
    for xi, v in zip(x - w/2, unадj):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhspan(0, 0.15, color=OI["green"], alpha=0.06)
    ax.text(3.45, 0.075, "biology\nzone", ha="center", va="center", fontsize=7.5,
            color=OI["green"], fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([AXLAB1[d] for d in order])
    ax.set_ylabel("correlation with general burden  Φ(G, ·)")
    ax.set_ylim(0, 0.5)
    ax.set_title("Metabolic and inflammatory load are the least burden-entangled domains")
    ax.legend(loc="upper right", fontsize=7.6)
    ax.text(0.0, -0.20, "Adjustment for medication, adiposity and site does not raise the biology bars — "
            "if anything it lowers them.", transform=ax.transAxes, fontsize=7.6, color="#555555")
    save(fig, "fig3_biology_g")

# ============================================================================ FIG 4
def fig4_continuum():
    struct = json.load(open(R("strata_oop", "structure", "data.json")))
    prof = pd.read_csv(R("strata_oop", "consolidate", "archetype_profiles.csv"))
    dr = np.load(R("strata_oop", "coordinates", "coordinates_draws.npz"))
    X = dr["draws"].mean(axis=0)  # (9013, 9) posterior-mean coords, dim order = dims
    dims = list(dr["dims"])
    coords = pd.read_parquet(R("strata_oop", "coordinates", "coordinates_full.parquet"))
    cohort = coords["cohort"].str.lower().values

    fig = plt.figure(figsize=(12.2, 8.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.18], hspace=0.42, wspace=0.34)

    # (a) single-Gaussian falsification null --------------------------------------
    axA = fig.add_subplot(gs[0, 0])
    sil = struct["diagnostics_A"]["silhouette"]["silhouette"]
    ks = sorted(int(k) for k in sil)
    sv = [sil[str(k)] for k in ks]
    fn = struct["falsification_null"]
    axA.plot(ks, sv, "-o", color=OI["blue"], ms=4, lw=1.6, label="real data")
    nm, nsd = fn["null_mean"]["best_silhouette"], fn["null_sd"]["best_silhouette"]
    axA.axhline(nm, color=OI["verm"], lw=1.4, ls="--", label="Gaussian null")
    axA.axhspan(nm - 2*nsd, nm + 2*nsd, color=OI["verm"], alpha=0.12)
    axA.axhline(0.15, color="#999999", lw=0.8, ls=":", label="separation floor (0.15)")
    axA.set_xlabel("number of clusters K"); axA.set_ylabel("best silhouette")
    axA.set_ylim(0.07, 0.165)
    axA.set_title("No clusters beyond a\nGaussian blob", fontsize=9.2)
    axA.legend(fontsize=6.6, loc="upper right", handlelength=1.4)
    axA.text(0.03, 0.04, f"peak {fn['real']['best_silhouette']:.3f} vs null {nm:.3f},  z = {fn['z']['best_silhouette']:.2f}\n"
             f"GMM-best K = 1 in 20/20 draws",
             transform=axA.transAxes, fontsize=6.4, color="#555555", va="bottom")
    panel(axA, "a")

    # (b,c,d) PCA embeddings ------------------------------------------------------
    Xs = (X - X.mean(0)) / X.std(0)
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    PC = Xs @ Vt[:2].T
    rng = np.random.default_rng(0); idx = rng.permutation(len(PC))
    gi = {d: dims.index(d) for d in dims}
    def emb(ax, color, title, cmap=None, disc=None, cbar_lbl=None):
        if disc is not None:
            for k, c in disc.items():
                m = cohort[idx] == k
                ax.scatter(PC[idx][m, 0], PC[idx][m, 1], s=2.2, c=c, alpha=0.45, lw=0,
                           label=COH_LBL[k], rasterized=True)
            ax.legend(markerscale=4, fontsize=6.8, loc="upper right", handletextpad=0.1)
        else:
            sc = ax.scatter(PC[idx][:, 0], PC[idx][:, 1], s=2.2, c=color[idx], cmap=cmap,
                            alpha=0.55, lw=0, rasterized=True,
                            vmin=np.percentile(color, 2), vmax=np.percentile(color, 98))
            cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02); cb.set_label(cbar_lbl, fontsize=7.4)
            cb.ax.tick_params(labelsize=6)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(title, fontsize=9.2)
        ax.set_xticks([]); ax.set_yticks([])
    axB = fig.add_subplot(gs[0, 1]); emb(axB, None, "by diagnosis\n(fully intermixed)", disc=COH)
    panel(axB, "b")
    axC = fig.add_subplot(gs[0, 2]); emb(axC, X[:, gi["overall_severity"]], "by general burden\n(smooth gradient →)",
                                         cmap="viridis", cbar_lbl="G")
    panel(axC, "c")
    axD = fig.add_subplot(gs[1, 0]); emb(axD, X[:, gi["inflammatory"]], "by inflammatory load\n(a different direction)",
                                         cmap="rocket" if "rocket" in plt.colormaps() else "magma", cbar_lbl="inflammatory")
    panel(axD, "d")

    # (e) A=4 archetype profiles --------------------------------------------------
    axE = fig.add_subplot(gs[1, 1:])
    A = prof[prof["arm"] == "A_all9"].set_index("archetype")
    order = AXES9; xx = np.arange(len(order)); w = 0.2
    for a in range(4):
        vals = [A.loc[a, c] for c in order]
        axE.bar(xx + (a-1.5)*w, vals, w, color=ARCH_C[a], label=f"A{a} {ARCH_N1[a]}")
    axE.axhline(0, color="#444444", lw=0.8)
    axE.set_xticks(xx); axE.set_xticklabels([AXLAB1[a] for a in order], rotation=35, ha="right", fontsize=7.4)
    axE.set_ylabel("archetype coordinate (z)")
    axE.set_title("Four archetypal extremes read the continuum (biology ⊥ symptoms ⊥ severity)", fontsize=9.4)
    axE.legend(ncol=4, fontsize=7.2, loc="upper center", bbox_to_anchor=(0.5, 1.22), columnspacing=1.0)
    axE.set_ylim(-3.4, 3.2)
    axE.text(0.5, -0.40, "A0 peaks on metabolic / inflammatory / substance with high severity; "
             "A2 is equally severe but biology-low — the dissociation made visible.",
             transform=axE.transAxes, ha="center", va="top", fontsize=7.0, color="#555555")
    panel(axE, "e", x=-0.075)
    save(fig, "fig4_continuum")

# ============================================================================ FIG 5
def fig5_persistence():
    ts = pd.read_csv(R("temporal_oop", "trait_state", "trait_state.csv")).set_index("axis")
    order = ts.sort_values("icc", ascending=True).index.tolist()
    fig = plt.figure(figsize=(11.4, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.34)

    axA = fig.add_subplot(gs[0])
    y = np.arange(len(order))
    vals = [ts.loc[a, "icc"] for a in order]
    lo = [ts.loc[a, "icc"] - ts.loc[a, "icc_lo"] for a in order]
    hi = [ts.loc[a, "icc_hi"] - ts.loc[a, "icc"] for a in order]
    def cof(a, v):
        if a in ("metabolic", "inflammatory", "cognition"): return BIO
        if a == "overall_severity": return OI["blue"]
        return OI["grey"]
    cols = [cof(a, ts.loc[a, "icc"]) for a in order]
    # mania uninformative -> hatch
    bars = axA.barh(y, vals, xerr=[lo, hi], color=cols, error_kw=dict(lw=0.8, ecolor="#666"))
    for i, a in enumerate(order):
        if a == "mania_activation":
            bars[i].set_alpha(0.35); bars[i].set_hatch("//")
    axA.axvline(0.5, color="#999", ls="--", lw=0.9)
    axA.text(0.5, len(order)-0.3, "trait →", fontsize=7.4, color="#777", ha="left")
    axA.text(0.5, len(order)-0.3, "← state ", fontsize=7.4, color="#777", ha="right")
    axA.set_yticks(y); axA.set_yticklabels([AXLAB1[a] for a in order], fontsize=8)
    axA.set_xlabel("trait fraction (test–retest ICC, V0→V2)")
    axA.set_xlim(0, 1)
    axA.set_title("Biology is trait; symptoms are state")
    for i, a in enumerate(order):
        axA.text(vals[i] + (hi[i] if a!='mania_activation' else 0) + 0.02, y[i],
                 f"{vals[i]:.2f}", va="center", fontsize=7.2)
    axA.legend(handles=[Line2D([],[],color=BIO,lw=6,label="biology / cognition"),
                        Line2D([],[],color=OI["blue"],lw=6,label="general burden"),
                        Line2D([],[],color=OI["grey"],lw=6,label="symptom axes"),
                        Line2D([],[],color=OI["grey"],lw=6,alpha=0.35,label="data-limited (mania)")],
               loc="lower right", fontsize=6.8)
    panel(axA, "a", x=-0.30)

    # (b) population slide vs individual rank stability
    axB = fig.add_subplot(gs[1])
    ax_sub = ["metabolic", "inflammatory", "cognition", "overall_severity", "suicidality"]
    icc = [ts.loc[a, "icc"] for a in ax_sub]
    slide = [ts.loc[a, "pop_slide"] for a in ax_sub]
    cc = [BIO if a in ("metabolic","inflammatory","cognition") else (OI["blue"] if a=="overall_severity" else OI["grey"]) for a in ax_sub]
    axB.scatter(icc, slide, s=90, c=cc, zorder=5, edgecolor="white", lw=1.0)
    for a, xi, yi in zip(ax_sub, icc, slide):
        axB.annotate(AXLAB1[a], (xi, yi), (xi, yi+0.05), fontsize=7.2, ha="center")
    axB.axhline(0, color="#444", lw=0.8)
    axB.set_xlabel("individual rank stability (ICC)")
    axB.set_ylabel("population change V0→V2  (z)")
    axB.set_title("Symptoms slide; biology holds")
    axB.text(0.02, 0.03, "patients keep their\nbiological rank while\nthe cohort improves\non symptoms/severity",
             transform=axB.transAxes, fontsize=7.0, color="#555", va="bottom")
    panel(axB, "b", x=-0.26)
    save(fig, "fig5_persistence")

# ============================================================================ FIG 6
def fig6_prognosis():
    atlas = pd.read_csv(R("prognosis_oop", "endpoints", "archetype_atlas.csv"))
    inc = pd.read_csv(R("prognosis_oop", "incremental", "incremental_comparison.csv"))
    h2h = pd.read_csv(R("prognosis_oop", "transdiagnostic", "h2h_dsm5.csv"))
    cv = pd.read_csv(R("prognosis_oop", "clinical_value", "clinical_value.csv"))

    fig = plt.figure(figsize=(12.4, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.15, 0.95], wspace=0.40)

    # (a) archetype prognostic atlas: pooled functional remission by archetype + cohort
    e = atlas[atlas["outcome"] == "egf"].copy()
    pooled = (e.groupby("archetype").apply(lambda d: np.average(d["remission_rate"], weights=d["n_rem"]))
              .reindex([0,1,2,3]))
    axA = fig.add_subplot(gs[0])
    order_a = [0, 2, 3, 1]  # worst->best
    xx = np.arange(4)
    axA.bar(xx, [pooled[a] for a in order_a], color=[ARCH_C[a] for a in order_a], width=0.7)
    for i, a in enumerate(order_a):
        axA.text(i, pooled[a]+0.015, f"{pooled[a]*100:.0f}%", ha="center", fontsize=9, fontweight="bold")
    axA.set_xticks(xx); axA.set_xticklabels([f"A{a}\n{ARCH_N1[a]}" for a in order_a], fontsize=7.6)
    axA.set_ylabel("2-yr functional remission (GAF ≥ 71)")
    axA.set_ylim(0, 0.7); axA.set_title("Archetype prognostic atlas")
    axA.annotate("", (0.0, 0.64), (3.0, 0.64), arrowprops=dict(arrowstyle="<|-|>", color="#666", lw=1.2))
    axA.text(1.5, 0.66, "27% → 60%  (biological corner worst)", ha="center", fontsize=7.6, color="#444")
    panel(axA, "a", x=-0.22)

    # (b) incremental held-out value, functioning (egf): encodings vs reference
    axB = fig.add_subplot(gs[1])
    e2 = inc[inc["outcome"] == "egf"].set_index("model")
    rows = [("+archetypesA", "A=4 archetypes"), ("+specifics8", "8 specific axes"),
            ("+tess_k3", "hard tiling (K=3)"), ("+durable", "biology axes alone")]
    yy = np.arange(len(rows))[::-1]
    vals = [e2.loc[m, "d_elpd_vs_ref"] for m, _ in rows]
    ses = [e2.loc[m, "se_d_elpd"] for m, _ in rows]
    cols = [OI["green"] if e2.loc[m, "verdict"] == "predictive" else OI["lgrey"] for m, _ in rows]
    axB.barh(yy, vals, xerr=ses, color=cols, error_kw=dict(lw=0.8, ecolor="#666"))
    axB.set_yticks(yy); axB.set_yticklabels([l for _, l in rows], fontsize=8)
    axB.axvline(0, color="#444", lw=0.8)
    axB.set_xlabel("held-out ΔELPD vs diagnosis + severity + baseline")
    axB.set_title("What carries the signal")
    for y_, v, s in zip(yy, vals, ses):
        axB.text(v + s + 1.5, y_, f"+{v:.0f}", va="center", fontsize=7.6)
    axB.text(0.02, -0.30, "the continuous archetype encoding dominates any hard partition\n"
             "(operative K = none); the biology axes alone no longer suffice",
             transform=axB.transAxes, fontsize=7.0, color="#555")
    panel(axB, "b", x=-0.18)

    # (c) co-informative with DSM-5 + group-vs-individual
    axC = fig.add_subplot(gs[2])
    hh = h2h[h2h["outcome"] == "egf"].set_index("model")
    bars = ["+map", "+dsm5", "+both"]
    lbl = ["map", "DSM-5", "both"]
    vals = [hh.loc[b, "d_elpd_vs_ref"] for b in bars]
    cc = [OI["green"], OI["blue"], OI["purple"]]
    axC.bar(range(3), vals, color=cc, width=0.66)
    for i, v in enumerate(vals):
        axC.text(i, v+1.2, f"+{v:.0f}", ha="center", fontsize=8, fontweight="bold")
    axC.set_xticks(range(3)); axC.set_xticklabels(lbl, fontsize=8)
    axC.set_ylabel("ΔELPD over shared foundation")
    axC.set_title("Map complements DSM-5")
    dauc = cv[cv.outcome == "egf"]["delta_auc"].iloc[0]
    axC.text(0.5, -0.30, f"co-informative, not redundant.\nIndividual remission AUC gain only +{dauc:.3f}\n"
             "— value is group-level forecasting", transform=axC.transAxes, ha="center",
             fontsize=7.0, color="#555")
    panel(axC, "c", x=-0.30)
    save(fig, "fig6_prognosis")

# ============================================================================ ED FIGS
def edfig_treatment():
    ts = pd.read_csv(R("treatment_oop", "consolidate", "treatment_summary.csv"))
    atl = pd.read_csv(R("treatment_oop", "atlas", "treatment_course_atlas.csv"))
    g = pd.read_csv(R("treatment_oop", "atlas", "atlas_gates.csv")).set_index("endpoint")
    fig = plt.figure(figsize=(12.0, 4.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.32)

    # (a) moderation: ATE with E-value, functioning, archetype representation
    axA = fig.add_subplot(gs[0])
    f = ts[(ts.outcome == "functioning") & (ts.representation == "durable")].copy()
    qs = ["lithium_bp", "antipsychotic_bp", "clozapine_sz"]
    qlbl = ["Lithium\n(bipolar)", "Antipsychotic\n(bipolar)", "Clozapine\n(schizophrenia)"]
    f = f.set_index("question").reindex(qs)
    yy = np.arange(len(qs))[::-1]
    ate = f["ate"].values
    # 94% ETI approx from MDE not available; use stored ate +/- (reconstruct from e-value not possible) -> show ATE points + verdict
    cols = [OI["blue"], OI["orange"], OI["grey"]]
    axA.scatter(ate, yy, s=70, c=cols, zorder=5, edgecolor="white", lw=1)
    # represent moderation MDE band around 0 to show "bounded"
    for y_, q in zip(yy, qs):
        mlo, mhi = f.loc[q, "int_mde_min"], f.loc[q, "int_mde_max"]
        axA.plot([-mhi, mhi], [y_-0.16, y_-0.16], color="#bbb", lw=2, solid_capstyle="butt")
    axA.axvline(0, color="#444", lw=0.8)
    axA.set_yticks(yy); axA.set_yticklabels(qlbl, fontsize=8)
    axA.set_xlabel("average treatment effect on functioning (SD)")
    axA.set_title("Treatment moderation not identified on TAU")
    for y_, q in zip(yy, qs):
        ev = f.loc[q, "e_value"]
        axA.text(f.loc[q, "ate"], y_+0.18, f"E={ev:.2f}", ha="center", fontsize=7, color="#555")
    axA.text(0.02, -0.28, "grey bars = interaction MDE (smallest detectable moderation).\n"
             "Lithium: well-identified bounded null. Antipsychotic: confounded average\n"
             "effect, no reliable moderation. Clozapine: underpowered.",
             transform=axA.transAxes, fontsize=6.8, color="#555")
    panel(axA, "a", x=-0.22)

    # (b) treatment-course atlas
    axB = fig.add_subplot(gs[1])
    ep_order = ["ep_resistance", "ep_response", "ep_side_effects"]
    ep_lbl = ["treatment-\nresistant", "CGI\nresponds", "significant\nside-effects"]
    pooled = atl[atl.cohort == "pooled"]
    w = 0.2; xx = np.arange(len(ep_order))
    for a in range(4):
        rates = [pooled[(pooled.endpoint==ep) & (pooled.archetype==a)]["rate"].iloc[0] for ep in ep_order]
        axB.bar(xx + (a-1.5)*w, rates, w, color=ARCH_C[a], label=f"A{a} {ARCH_N1[a]}")
    axB.set_xticks(xx); axB.set_xticklabels(ep_lbl, fontsize=8)
    axB.set_ylabel("2-yr rate")
    axB.set_title("Treatment-course atlas: biological corner hardest")
    axB.legend(ncol=4, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, 1.16), columnspacing=0.8)
    # annotate permutation AUC p
    for i, ep in enumerate(ep_order):
        p = g.loc[ep, "delta_auc_perm_p"]
        axB.text(i, 0.66, f"AUC perm p={p:.3f}", ha="center", fontsize=6.4, color="#777")
    axB.set_ylim(0, 0.72)
    axB.text(0.02, -0.26, "Stratification clears (beyond severity+substance+demographics, p≤0.003); "
             "individual discrimination modest\nand resistance is AUC-marginal (p=0.185). Monitoring, not prescribing.",
             transform=axB.transAxes, fontsize=6.8, color="#555")
    panel(axB, "b", x=-0.13)
    save(fig, "edfig_treatment")

def edfig_repbench():
    sc = pd.read_csv(R("m4_repbench", "scalar.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, (tgt, ttl) in zip(axes, [("egf_deterioration", "Deterioration"), ("egf_recovery", "Recovery")]):
        d = sc[(sc.target == tgt) & (sc.scope == "pooled") & (sc.horizon == "V2")].set_index("arm")
        arms = [("REF", "DSM-5 + severity\n+ baseline"), ("REF+LAT-A", "+ 9-dim map"), ("REF+RAW", "+ 143 raw\nindicators")]
        vals = [d.loc[a, "auc"] for a, _ in arms]
        cc = [OI["grey"], OI["green"], OI["blue"]]
        ax.bar(range(3), vals, color=cc, width=0.62)
        for i, v in enumerate(vals):
            ax.text(i, v+0.004, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in arms], fontsize=7.4)
        ax.set_ylim(0.6, 0.80); ax.set_ylabel("AUC (held-out)")
        ax.set_title(ttl)
    axes[0].text(0.5, 0.9, "map = raw\n(sufficient)", transform=axes[0].transAxes, ha="center",
                 fontsize=8, color=OI["green"], fontweight="bold")
    axes[1].text(0.5, 0.9, "raw +0.04\n(near-sufficient;\n97% within factors)", transform=axes[1].transAxes,
                 ha="center", fontsize=7.6, color="#555")
    fig.suptitle("The 9-dim map is a sufficient representation of 143 raw indicators", fontsize=10, fontweight="bold")
    panel(axes[0], "a"); panel(axes[1], "b")
    save(fig, "edfig_repbench")

def edfig_invariance():
    cg = pd.read_csv(R("temporal_oop", "invariance", "congruence.csv"))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    piv = cg.pivot_table(index="factor", columns="visit", values="phi_min")
    order = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep"]
    piv = piv.reindex(order)
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w/2, piv["V1"], w, color=OI["sky"], label="12 months")
    ax.bar(x + w/2, piv["V2"], w, color=OI["blue"], label="24 months")
    ax.axhline(0.95, color=OI["verm"], ls="--", lw=1, label="invariance threshold (0.95)")
    ax.set_ylim(0.9, 1.005); ax.set_xticks(x); ax.set_xticklabels([AXLAB1[a] for a in order], fontsize=8)
    ax.set_ylabel("Tucker congruence φ vs baseline")
    ax.set_title("The measurement holds over follow-up (longitudinal invariance)")
    ax.legend(fontsize=7.4, loc="lower right")
    ax.text(0.0, -0.16, "All five backbone axes remain invariant at 12 and 24 months; "
            "inflammatory (φ=0.974) is now invariant on the copula map.",
            transform=ax.transAxes, fontsize=7.2, color="#555")
    save(fig, "edfig_invariance")

def edfig_robustness():
    rb = pd.read_csv(R("prognosis_oop", "robustness", "robustness.csv"))
    d = rb[(rb.outcome == "egf") & (rb.encoding == "+archetypesA")].set_index("check")
    order = ["base", "ipw", "drop_dr", "drop_sz", "drop_bp", "permutation"]
    lbl = ["base", "attrition IPW", "drop DR", "drop SZ", "drop BP", "permutation null"]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    yy = np.arange(len(order))[::-1]
    vals = [d.loc[c, "d_elpd_vs_ref"] for c in order]
    ses = [d.loc[c, "se_d_elpd"] for c in order]
    cols = [OI["green"] if d.loc[c, "verdict"] == "predictive" else OI["lgrey"] for c in order]
    ax.barh(yy, vals, xerr=ses, color=cols, error_kw=dict(lw=0.8, ecolor="#666"))
    ax.set_yticks(yy); ax.set_yticklabels(lbl, fontsize=8.2)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("held-out ΔELPD (functioning, archetypes)")
    ax.set_title("The functional forecast is robust — and course-dependent")
    for y_, v, s in zip(yy, vals, ses):
        ax.text(v + s + 1.5, y_, f"{v:+.0f}", va="center", fontsize=7.6)
    ax.text(0.02, -0.17, "Survives attrition reweighting and dropping DR or SZ; vanishes under permutation; "
            "weak when bipolar is removed\n(the signal is carried by the episodic, open-course cohorts).",
            transform=ax.transAxes, fontsize=7.0, color="#555")
    save(fig, "edfig_robustness")

def edfig_consort():
    fig, ax = plt.subplots(figsize=(8.0, 4.6)); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    steps = [("Baseline V0\nN = 9,013", "BP 6,252 · SZ 2,209 · DR 552", OI["blue"], 82),
             ("12-month V1\nn = 4,270 (47.4%)", "BP 49.2% · SZ 43.6% · DR 42.2%", OI["sky"], 50),
             ("24-month V2\nn = 2,958 (32.8%)", "BP 35.6% · SZ 26.9% · DR 24.5%", OI["green"], 18)]
    for t, sub, c, y in steps:
        ax.add_patch(FancyBboxPatch((30, y-9), 40, 16, boxstyle="round,pad=0.4,rounding_size=2",
                     fc=c, ec="none", alpha=0.9))
        ax.text(50, y+1.5, t, ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")
        ax.text(50, y-5, sub, ha="center", va="center", fontsize=7.4, color="white")
    for y0, y1, n in [(73, 59, "dropout 4,743"), (41, 27, "dropout 1,312")]:
        ax.annotate("", (50, y1), (50, y0), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#666"))
        ax.text(72, (y0+y1)/2, n, fontsize=7.2, color="#777", va="center")
    ax.text(50, 96, "Longitudinal retention (CONSORT-style)", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(50, 3, "Attrition is informative (retention ~ baseline coordinates); addressed by inverse-probability weighting.",
            ha="center", fontsize=7.2, style="italic", color="#666")
    save(fig, "edfig_consort")

if __name__ == "__main__":
    print("Building FACE-ATLAS copula figures ->", OUT)
    for fn in [fig1_overview, fig2_map, fig3_biology_g, fig4_continuum, fig5_persistence,
               fig6_prognosis, edfig_treatment, edfig_repbench, edfig_invariance,
               edfig_robustness, edfig_consort]:
        try:
            fn()
        except Exception as e:
            import traceback; print("  FAILED", fn.__name__, "->", e); traceback.print_exc()
    print("done.")
