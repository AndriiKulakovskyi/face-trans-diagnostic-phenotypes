#!/usr/bin/env python3
"""Publication-spec figures for the FACE-ATLAS journal article.

Regenerates every main + Extended Data figure at 300 dpi with a unified,
colourblind-aware house style and panel labels, into article/figures/.

Reads only shareable aggregates (reports/*.csv, docs/figures/*.csv, configs/) plus
the derived per-patient coordinate table for the PCA embedding (the 2-D projection
written out is itself aggregate; no raw clinical value is emitted). Never reads raw
per-cohort CSVs. Run:  python3 article/make_figures.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"
DFIG = REPO / "docs" / "figures"
CONFIGS = REPO / "configs"
M2 = REPO / "results" / "face" / "m2"
OUT = REPO / "article" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette (colourblind-aware: blue/teal/amber/red, Okabe-Ito-adjacent) ----
INK, MUTE, ACCENT, ACCENTDK = "#14181F", "#5B6573", "#2B4C8C", "#1E366B"
TEAL, AMBER, RED, VIOLET = "#0F766E", "#B45309", "#B42318", "#6D28D9"
COH = {"bp": "#2B4C8C", "sz": "#0F766E", "dr": "#B45309"}      # cohort colours
SEQ = LinearSegmentedColormap.from_list("face", ["#f7f7f7", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
DIV = LinearSegmentedColormap.from_list("facediv", ["#B42318", "#f6e9e6", "#f7f7f7", "#dbe5f3", "#2B4C8C"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9, "axes.edgecolor": MUTE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTE, "ytick.color": MUTE, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": ACCENTDK, "figure.dpi": 150, "savefig.dpi": 300, "axes.titlepad": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "legend.frameon": False,
})

FACTORS9 = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep",
            "developmental_risk", "suicidality", "mania_activation", "substance"]
PRETTY = {"overall_severity": "G (severity)", "cognition": "cognition", "metabolic": "metabolic",
          "inflammatory": "inflammatory", "sleep": "sleep", "developmental_risk": "developmental",
          "suicidality": "suicidality", "mania_activation": "mania", "substance": "substance"}


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


def _panel(ax, letter, dx=-0.06, dy=1.10):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=12, fontweight="bold",
            color=INK, va="bottom", ha="right")


# ============================================================ FIG 1 — study overview
def fig1_overview():
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, tc="white", fs=8.6, bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.4",
                                    fc=fc, ec=ec, lw=1.2, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal", zorder=3)

    def arrow(x0, y0, x1, y1, c=MUTE):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                     color=c, lw=1.6, zorder=1))

    ax.text(50, 60, "A transdiagnostic clinical–biological map of severe mental illness",
            ha="center", va="center", fontsize=12, fontweight="bold", color=ACCENTDK)

    # input cohorts
    box(2, 44, 22, 9, "FACE baseline\nBP · SZ · DR\nN = 9,013 (21 sites)", ACCENT, ACCENTDK, fs=8.3)
    # engine
    box(27, 44, 29, 9, "One global Bayesian\nbifactor / ESEM\n(observed cells, mixed)", TEAL, "#0b5a54", fs=8.3)
    # output map
    box(59, 44, 24, 9, "Certified 9-dimension\ntransdiagnostic map", ACCENTDK, INK, fs=8.6)
    arrow(24, 48.5, 27, 48.5); arrow(56, 48.5, 59, 48.5)

    # invariants strip
    ax.text(50, 39.5, "Invariants:  no imputation (observed-data likelihood)   •   "
            "diagnosis is metadata   •   baseline defines, follow-up validates",
            ha="center", va="center", fontsize=7.8, color=MUTE, style="italic")

    # five questions funnel
    q = [("1  EXISTS", "9-dim map;\nbiology ⊥ severity", ACCENT),
         ("2  ORGANIZES", "continuum,\nnot biotypes", TEAL),
         ("3  PERSISTS", "durable biology,\nmoving symptoms", "#3A6EA5"),
         ("4  PREDICTS", "functioning,\nmodest · group-level", AMBER),
         ("5  TREATMENT", "no reliable\nmoderation (TAU)", RED)]
    x0, w, gap = 3, 17.4, 1.2
    for i, (head, sub, c) in enumerate(q):
        x = x0 + i * (w + gap)
        box(x, 16, w, 13, f"{head}\n\n{sub}", c, c, fs=8.2)
        if i < 4:
            arrow(x + w, 22.5, x + w + gap, 22.5)
    ax.text(50, 9.5, "exists  →  organizes  →  persists  →  predicts  →  treatment boundary",
            ha="center", va="center", fontsize=8.4, color=INK, fontweight="bold")
    ax.text(50, 5.6, "A calibrated account: a real, stable continuum with a small group-level prognostic "
            "signal—positive and null results reported alike.", ha="center", va="center",
            fontsize=7.6, color=MUTE, style="italic")
    _save(fig, "fig1_overview")


# ============================================================ atlas helpers (from report house code)
def _atlas_matrix():
    post = pd.read_csv(DFIG / "empirical_atlas.csv", index_col=0).reindex(columns=FACTORS9).fillna(0.0)
    home = post.abs().values.argmax(1)
    order = sorted(range(len(post)), key=lambda i: (home[i], -abs(post.values[i, home[i]])))
    post = post.iloc[order]
    homes = [FACTORS9[h] for h in np.asarray(home)[order]]
    return post, homes


def _draw_atlas(ax, M, homes, title, vmax=0.95):
    im = ax.imshow(np.abs(M.values), aspect="auto", cmap=SEQ, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(FACTORS9)))
    ax.set_xticklabels([PRETTY[f] for f in FACTORS9], rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks([]); ax.set_title(title, fontsize=10)
    bnd = [i for i in range(1, len(homes)) if homes[i] != homes[i - 1]]
    for b in bnd:
        ax.axhline(b - 0.5, color="white", lw=1.4)
    for s in [0] + bnd:
        ax.text(-0.7, s + 0.4, PRETTY.get(homes[s], homes[s]), ha="right", va="top",
                fontsize=6.6, color=MUTE)
    return im


def _draw_phi(ax):
    P = pd.read_csv(REPORTS / "04_stage5_phi.csv", index_col=0)
    spec = [f for f in P.columns if f != "overall_severity"]
    M = P.loc[spec, spec].astype(float)
    labels = [PRETTY.get(f, f) for f in spec]
    im = ax.imshow(M.values, cmap=DIV, norm=TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.25))
    ax.set_xticks(range(len(spec))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(spec))); ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(spec)):
        for j in range(len(spec)):
            v = M.values[i, j]
            ax.text(j, i, "1" if i == j else f"{v:.2f}", ha="center", va="center", fontsize=7.2,
                    color=(MUTE if i == j else (INK if abs(v) < 0.16 else "white")))
    ax.set_title(r"Inter-dimension correlations $\Phi$", fontsize=10)
    return im


def fig2_map():
    post, homes = _atlas_matrix()
    fig = plt.figure(figsize=(10.4, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.28)
    ax0 = fig.add_subplot(gs[0]); im0 = _draw_atlas(ax0, post, homes, "Posterior loading atlas (the data)")
    cb = fig.colorbar(im0, ax=ax0, fraction=0.045, pad=0.02); cb.set_label("|loading|", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax1 = fig.add_subplot(gs[1]); im1 = _draw_phi(ax1)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04); cb1.ax.tick_params(labelsize=7)
    _panel(ax0, "a", dy=1.02); _panel(ax1, "b", dy=1.02)
    _save(fig, "fig2_map")


# ============================================================ FIG 3 — biology vs G
def fig3_biology_g():
    df = pd.read_csv(REPORTS / "07_corrG_phi.csv").sort_values("corrG_phi_with_G")
    y = np.arange(len(df))
    colors = [TEAL if b else ACCENT for b in df.domain.isin(["metabolic", "inflammatory"]).values]
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    ax.barh(y, df.corrG_phi_with_G, color=colors, height=0.62, zorder=3)
    ax.plot(df.bifactor_loading_on_G, y, "o", color=INK, ms=5, zorder=4)
    for yi, cg in zip(df.corrG_phi_with_G, y, strict=False):
        ax.text(yi + 0.012, cg, f"{yi:.2f}", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([PRETTY.get(d, d) for d in df.domain])
    ax.set_xlim(0, 0.52)
    ax.set_xlabel(r"correlation with the general burden factor $G$ (correlated-$G$ $\varphi$)")
    ax.set_title("Biology is the least severity-entangled domain", pad=10)
    handles = [Patch(color=TEAL, label="biology"), Patch(color=ACCENT, label="cognition / sleep"),
               plt.Line2D([], [], marker="o", ls="", color=INK, label=r"bifactor $|\lambda_G|$")]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    _save(fig, "fig3_biology_g")


# ============================================================ FIG 4 — continuum (PCA + gate + archetypes)
def _pca_2d():
    df = pd.read_parquet(M2 / "coordinates_full.parquet")
    X = df[[f"{f}__mean" for f in FACTORS9]].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    U, S, _ = np.linalg.svd(Xz - Xz.mean(0), full_matrices=False)
    pcs = U[:, :2] * S[:2]
    var = (S**2 / (S**2).sum())[:2]
    # orient for readability
    if np.corrcoef(pcs[:, 0], Xz[:, 0])[0, 1] < 0:
        pcs[:, 0] *= -1
    return pcs, var, df


def fig4_continuum():
    pcs, var, df = _pca_2d()
    coh = df["cohort"].str.lower().to_numpy()
    sev = df["overall_severity__mean"].to_numpy()
    infl = df["inflammatory__mean"].to_numpy()
    fig = plt.figure(figsize=(11.0, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.30)

    def scatter(ax, c, title, cmap=None, cbar=False, cat=False):
        if cat:
            for k, col in COH.items():
                m = coh == k
                ax.scatter(pcs[m, 0], pcs[m, 1], s=3, c=col, alpha=0.45, lw=0,
                           label=k.upper(), rasterized=True)
            ax.legend(markerscale=3, fontsize=7, loc="upper right")
        else:
            sc = ax.scatter(pcs[:, 0], pcs[:, 1], s=3, c=c, cmap=cmap, alpha=0.55, lw=0,
                            vmin=np.percentile(c, 2), vmax=np.percentile(c, 98), rasterized=True)
            if cbar:
                cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9.2); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"PC1 ({var[0]*100:.0f}%)", fontsize=7.5); ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)", fontsize=7.5)

    ax_a = fig.add_subplot(gs[0, 0]); scatter(ax_a, None, "by diagnosis (intermixed)", cat=True)
    ax_b = fig.add_subplot(gs[0, 1]); scatter(ax_b, sev, "by overall severity (G)", cmap=SEQ, cbar=True)
    ax_c = fig.add_subplot(gs[0, 2]); scatter(ax_c, infl, "by inflammatory load", cmap="cividis", cbar=True)
    _panel(ax_a, "a")

    # gate verdict
    ax_g = fig.add_subplot(gs[1, 0])
    gate = [("Gap statistic", "K = 1"), ("HDBSCAN", "0 clusters"), ("Hartigan dip (PC1)", "p ≈ 0.99"),
            ("Archetype scree", "no elbow"), ("Mixture BIC", "flat basin")]
    ax_g.axis("off"); ax_g.set_title("Structure gate → continuum", fontsize=9.2)
    for i, (m, v) in enumerate(gate):
        yy = 0.86 - i * 0.17
        ax_g.text(0.02, yy, m, fontsize=8.2, color=INK, va="center")
        ax_g.text(0.98, yy, v, fontsize=8.2, color=TEAL, fontweight="bold", va="center", ha="right")
        ax_g.axhline(yy - 0.085, xmin=0.02, xmax=0.98, color="#e6e6e6", lw=0.7)
    ax_g.text(0.5, -0.02, "no discrete biotypes", ha="center", fontsize=8, color=MUTE, style="italic",
              transform=ax_g.transAxes)
    _panel(ax_g, "b", dy=1.02)

    # archetype profiles
    ax_p = fig.add_subplot(gs[1, 1:])
    prof = pd.read_csv(M2 / "archetype_profiles.csv")
    prof = prof[prof.arm == "A_all9"].reset_index(drop=True)
    M = prof[FACTORS9].to_numpy(dtype=float)
    im = ax_p.imshow(M, aspect="auto", cmap=DIV, norm=TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2))
    ax_p.set_xticks(range(len(FACTORS9)))
    ax_p.set_xticklabels([PRETTY[f] for f in FACTORS9], rotation=45, ha="right", fontsize=7.5)
    ax_p.set_yticks(range(len(prof)))
    ax_p.set_yticklabels([f"A{i}" for i in range(len(prof))], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax_p.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.2,
                      color="white" if abs(M[i, j]) > 1.3 else INK)
    ax_p.set_title("Eight extreme phenotypes (archetypes); each the high pole of one axis", fontsize=9.2)
    cb = fig.colorbar(im, ax=ax_p, fraction=0.025, pad=0.02); cb.set_label("z", fontsize=8); cb.ax.tick_params(labelsize=7)
    _panel(ax_p, "c", dy=1.02)
    _save(fig, "fig4_continuum")


# ============================================================ FIG 5 — trait/state + spine-corner
def fig5_persistence():
    ts = pd.read_csv(REPORTS / "35_trait_state.csv")
    ts = ts[ts.axis != "substance"].copy()
    order = ts.sort_values("icc").axis.tolist()
    ts = ts.set_index("axis").reindex(order)
    cr = pd.read_csv(REPORTS / "36_change_rates.csv").set_index("axis").reindex(order)
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw={"wspace": 0.42})

    # panel a: ICC trait/state
    ax = axes[0]
    bio = {"metabolic", "inflammatory", "cognition"}
    cols = [TEAL if a in bio else (AMBER if ts.loc[a, "icc"] < 0.55 else ACCENT) for a in order]
    ax.barh(y, ts["icc"], color=cols, height=0.6, zorder=3)
    ax.errorbar(ts["icc"], y, xerr=[ts["icc"] - ts["icc_lo"], ts["icc_hi"] - ts["icc"]],
                fmt="none", ecolor=INK, lw=0.8, capsize=2, zorder=4)
    ax.axvline(0.5, color=MUTE, ls=":", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels([PRETTY.get(a, a) for a in order])
    ax.set_xlim(0, 1); ax.set_xlabel("individual rank-stability (ICC)")
    ax.set_title("Trait vs state: durable biology, moving symptoms", fontsize=9.6)
    ax.text(0.02, len(order) - 0.4, "← state", color=AMBER, fontsize=8, va="center")
    ax.text(0.98, 0.0, "trait →", color=TEAL, fontsize=8, va="center", ha="right")
    handles = [Patch(color=TEAL, label="biology / cognition (trait)"), Patch(color=AMBER, label="symptom (state)")]
    ax.legend(handles=handles, fontsize=7.6, loc="lower right")
    _panel(ax, "a", dy=1.03)

    # panel b: reliable-change (spine moves, corner holds)
    ax2 = axes[1]
    cols2 = [TEAL if a in bio else (ACCENT if a == "overall_severity" else MUTE) for a in order]
    ax2.barh(y, cr["frac_reliable"], color=cols2, height=0.6, zorder=3)
    for yi, a in zip(y, order, strict=False):
        ax2.text(cr.loc[a, "frac_reliable"] + 0.006, yi, f"{cr.loc[a, 'frac_reliable']*100:.0f}%",
                 va="center", fontsize=7.6, color=INK)
    ax2.set_yticks(y); ax2.set_yticklabels([PRETTY.get(a, a) for a in order])
    ax2.set_xlim(0, 0.62); ax2.set_xlabel("patients moving beyond measurement error (V0→V2)")
    ax2.set_title("Severity spine slides; biology corner holds", fontsize=9.6)
    handles2 = [Patch(color=ACCENT, label="severity (spine)"), Patch(color=TEAL, label="biology corner")]
    ax2.legend(handles=handles2, fontsize=7.6, loc="lower right")
    _panel(ax2, "b", dy=1.03)
    _save(fig, "fig5_persistence")


# ============================================================ FIG 6 — prognosis (atlas + dominance)
_M4_EP = [("egf_remission", "func.\nremission", "good"), ("egf_recovery", "func.\nrecovery", "good"),
          ("egf_deterioration", "deterior.", "poor"), ("egf_sustained_impair", "sustained\nimpair", "poor"),
          ("cgi_remission", "CGI\nremission", "good"), ("cgi_relapse", "relapse\nsurrog.", "poor"),
          ("cgi_sustained_severe", "sustained\nillness", "poor")]


def fig6_prognosis():
    a = pd.read_csv(REPORTS / "45_archetype_atlas.csv").sort_values("egf_remission", ascending=False)
    rows = list(a["archetype"]); cols = [e for e, _, _ in _M4_EP]
    rate = a.set_index("archetype")[cols].reindex(rows).values
    adv = rate.copy()
    for j, (_, _, pol) in enumerate(_M4_EP):
        if pol == "good":
            adv[:, j] = 1 - adv[:, j]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.3})
    ax = axes[0]
    im = ax.imshow(adv, cmap=DIV.reversed(), norm=TwoSlopeNorm(vmin=0.15, vcenter=0.5, vmax=0.85), aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([lbl for _, lbl, _ in _M4_EP], fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r}  (n={int(n)})" for r, n in zip(rows, a["n"], strict=False)], fontsize=7.6)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = rate[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                        color="white" if (adv[i, j] > 0.72 or adv[i, j] < 0.28) else INK)
    ax.set_title("Archetype prognostic atlas — 2-year endpoint rates", fontsize=9.6, pad=8)
    ax.text(0.5, -0.27, "blue favourable · red adverse; functional remission 14%–60%",
            transform=ax.transAxes, ha="center", fontsize=7.6, color=MUTE)
    _panel(ax, "a", dy=1.03)

    # dominance: co-informative
    ax2 = axes[1]
    h = pd.read_csv(REPORTS / "44_h2h_dsm5.csv")
    order = ["DSM-5 beyond foundation (A−D)", "map beyond foundation (C−D)",
             "map beyond DSM-5 (B−A)", "DSM-5 beyond map (B−C)"]
    short = ["DSM-5\nvs base", "map\nvs base", "map beyond\nDSM-5", "DSM-5 beyond\nmap"]
    sub = h[h.outcome == "egf"].set_index("contrast").reindex(order)
    vals, ses = sub["d_elpd"].values, sub["se"].values
    colors = [TEAL if (v - 2 * s) > 0 else MUTE for v, s in zip(vals, ses, strict=False)]
    ax2.bar(range(4), vals, color=colors, zorder=3)
    ax2.errorbar(range(4), vals, yerr=2 * ses, fmt="none", ecolor=INK, capsize=2, lw=0.8)
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(short, fontsize=7.2)
    ax2.set_ylabel(r"$\Delta$ELPD vs reference ($\pm$2SE)")
    ax2.set_title("Functioning: map and DSM-5 are co-informative", fontsize=9.6)
    _panel(ax2, "b", dy=1.03)
    _save(fig, "fig6_prognosis")


# ============================================================ ED FIG — M5 treatment (identification + moderation)
_M5_Q = {"lithium_bp": "lithium (BP)", "antipsychotic_bp": "antipsychotic (BP)", "clozapine_sz": "clozapine (SZ)"}
_M5_AX = ["cognition", "metabolic", "inflammatory"]


def edfig_m5():
    fig = plt.figure(figsize=(11.2, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.32)
    # identification
    ax = fig.add_subplot(gs[0])
    s = pd.read_csv(REPORTS / "55_propensity_summary.csv")
    s = s[(s["mode"] == "active_comparator") & s["question"].isin(_M5_Q)].set_index("question").reindex(_M5_Q)
    x = np.arange(len(s)); w = 0.36
    ax.bar(x - w / 2, s["max_smd_before"], w, label="before IPTW", color=AMBER, zorder=3)
    ax.bar(x + w / 2, s["max_smd_after"], w, label="after IPTW", color=ACCENT, zorder=3)
    ax.axhline(0.1, color=INK, ls=":", lw=0.8); ax.axhline(0.25, color=RED, ls="--", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([_M5_Q[q] for q in s.index], fontsize=8, rotation=12)
    ax.set_ylabel("max |SMD| (imbalance)")
    for xi, (_, r) in zip(x, s.iterrows(), strict=False):
        ax.text(xi, max(r["max_smd_before"], r["max_smd_after"]) + 0.02,
                f"{r['frac_in_support']:.0%} overlap", ha="center", fontsize=6.8, color=MUTE)
    ax.set_title("Identification: covariate balance", fontsize=9.4)
    ax.legend(fontsize=7.6)
    _panel(ax, "a", dy=1.03)

    # moderation forest
    m = pd.read_csv(REPORTS / "56_moderation.csv")
    outs = [("functioning", "functioning (EGF)"), ("cgi_response", "CGI response")]
    qs = [q for q in _M5_Q if (m.question == q).any()]
    gsr = gs[1].subgridspec(len(outs), len(qs), wspace=0.12, hspace=0.35)
    for ri, (oc, olab) in enumerate(outs):
        for ci, q in enumerate(qs):
            ax = fig.add_subplot(gsr[ri, ci])
            r = m[(m.question == q) & (m.outcome == oc)]
            if len(r):
                r = r.iloc[0]
                means = [r[f"int_{a}"] for a in _M5_AX]
                los = [r[f"int_{a}_lo"] for a in _M5_AX]; his = [r[f"int_{a}_hi"] for a in _M5_AX]
                cols = [RED if (lo > 0 or hi < 0) else MUTE for lo, hi in zip(los, his, strict=False)]
                for yi, (mn, lo, hi, c) in enumerate(zip(means, los, his, cols, strict=False)):
                    ax.plot([lo, hi], [yi, yi], color=c, lw=1.6, zorder=2)
                    ax.plot(mn, yi, "o", color=c, ms=4, zorder=3)
                ax.axvline(0, color=INK, lw=0.8)
            ax.set_yticks(range(len(_M5_AX)))
            ax.set_yticklabels(_M5_AX if ci == 0 else [], fontsize=7)
            if ri == 0:
                ax.set_title(_M5_Q[q], fontsize=8.2)
            if ci == 0:
                ax.set_ylabel(olab, fontsize=7.6)
            ax.tick_params(labelsize=6.5)
    fig.text(0.345, 1.0, "b", fontsize=12, fontweight="bold", color=INK, ha="right", va="bottom")
    fig.text(0.66, 0.99, "Does durable biology moderate treatment response? (treat × axis, 94% HDI)",
             ha="center", fontsize=9.2, fontweight="bold", color=ACCENTDK)
    fig.text(0.66, -0.02, "red = HDI excludes 0. Lithium: null. Antipsychotic: metabolic/inflammatory on "
             "functioning. Clozapine: channelled (uninterpretable).", ha="center", fontsize=6.8, color=MUTE)
    _save(fig, "edfig_m5_treatment")


if __name__ == "__main__":
    print(f"Generating article figures -> {OUT.relative_to(REPO)}")
    for fn in [fig1_overview, fig2_map, fig3_biology_g, fig4_continuum, fig5_persistence,
               fig6_prognosis, edfig_m5]:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  !! {fn.__name__} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("done.")
