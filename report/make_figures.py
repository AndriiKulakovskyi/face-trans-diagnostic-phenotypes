#!/usr/bin/env python3
"""Generate the FACE-ATLAS manuscript figures from committed AGGREGATE artifacts.

Reads only shareable aggregates (reports/*.csv, docs/figures/*.csv, configs/) — never per-patient
data — and writes PNGs into report/figures/. House style matches the report (deep-blue accent,
light->dark sequential blue, sans labels). Run:  python3 report/make_figures.py
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

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"
DFIG = REPO / "docs" / "figures"
CONFIGS = REPO / "configs"
OUT = REPO / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette (matches faceatlas.sty) ----
INK, MUTE, ACCENT, ACCENTDK = "#14181F", "#5B6573", "#2B4C8C", "#1E366B"
KR, CAV, OPEN, VIOLET = "#0F766E", "#B45309", "#B42318", "#6D28D9"
SEQ = LinearSegmentedColormap.from_list("face", ["#f7f7f7", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
DIV = LinearSegmentedColormap.from_list("facediv", ["#B42318", "#f6e9e6", "#f7f7f7", "#dbe5f3", "#2B4C8C"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9, "axes.edgecolor": MUTE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTE, "ytick.color": MUTE, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlecolor": ACCENTDK, "figure.dpi": 150, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
})

FACTORS9 = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep",
            "developmental_risk", "suicidality", "mania_activation", "substance"]
PRETTY = {"overall_severity": "G (severity)", "cognition": "cognition", "metabolic": "metabolic",
          "inflammatory": "inflammatory", "sleep": "sleep", "developmental_risk": "developmental",
          "suicidality": "suicidality", "mania_activation": "mania", "substance": "substance"}


def _save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {p.relative_to(REPO)}")


# ============================================================ 1. biology vs G (headline)
def fig_biology_g():
    df = pd.read_csv(REPORTS / "07_corrG_phi.csv")
    df = df.sort_values("corrG_phi_with_G")
    y = np.arange(len(df))
    is_bio = df.domain.isin(["metabolic", "inflammatory"]).values
    colors = [KR if b else ACCENT for b in is_bio]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.barh(y, df.corrG_phi_with_G, color=colors, height=0.62, zorder=3)
    ax.plot(df.bifactor_loading_on_G, y, "o", color=INK, ms=5, zorder=4)
    for yi, (cg, lab) in enumerate(zip(df.corrG_phi_with_G, df.domain, strict=False)):
        ax.text(cg + 0.012, yi, f"{cg:.2f}", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([PRETTY.get(d, d) for d in df.domain])
    ax.set_xlim(0, 0.52); ax.set_xlabel(r"correlation with the general burden factor $G$  (correlated-$G$ $\varphi$)")
    ax.set_title("Biology is the least severity-entangled domain", pad=12)
    from matplotlib.patches import Patch
    handles = [Patch(color=KR, label="biology"), Patch(color=ACCENT, label="cognition / sleep"),
               plt.Line2D([], [], marker="o", ls="", color=INK, label=r"bifactor $|\lambda_G|$")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    _save(fig, "fig_biology_g.png")


# ============================================================ 2. inter-dimension Phi
def fig_phi():
    P = pd.read_csv(REPORTS / "04_stage5_phi.csv", index_col=0)
    spec = [f for f in P.columns if f != "overall_severity"]
    M = P.loc[spec, spec].astype(float)
    labels = [PRETTY.get(f, f) for f in spec]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(M.values, cmap=DIV, norm=TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.25))
    ax.set_xticks(range(len(spec))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(spec))); ax.set_yticklabels(labels, fontsize=8.5)
    for i in range(len(spec)):
        for j in range(len(spec)):
            v = M.values[i, j]
            if i == j:
                ax.text(j, i, "1", ha="center", va="center", fontsize=8, color=MUTE)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color=INK if abs(v) < 0.16 else "white",
                        fontweight="bold" if abs(v) >= 0.18 else "normal")
    ax.set_title("Inter-dimension correlations $\\Phi$  (G held orthogonal)")
    off = M.values[~np.eye(len(spec), dtype=bool)]
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=7)
    ax.text(0.5, -0.30, f"mean |off-diagonal| = {np.abs(off).mean():.2f} — distinct axes",
            transform=ax.transAxes, ha="center", fontsize=8, color=MUTE)
    _save(fig, "fig_phi_heatmap.png")


# ============================================================ 3/4. atlas builder + prior->posterior
def _atlas_matrix():
    """Return (posterior, prior) item x 9-factor matrices on a shared home-grouped row order."""
    post = pd.read_csv(DFIG / "empirical_atlas.csv", index_col=0)
    post = post.reindex(columns=FACTORS9).fillna(0.0)
    home = post.abs().values.argmax(1)
    order = sorted(range(len(post)), key=lambda i: (home[i], -abs(post.values[i, home[i]])))
    post = post.iloc[order]
    # prior-permitted = |mean| + sd, from the prior matrix
    pm = pd.read_csv(CONFIGS / "prior_loading_matrix_v3.csv")
    pm["permit"] = pm.prior_mean.abs() + pm.prior_sd
    prior = pm.pivot_table(index="item", columns="factor", values="permit", aggfunc="max")
    prior = prior.reindex(index=post.index, columns=FACTORS9).fillna(0.0)
    homes = [FACTORS9[h] for h in np.asarray(home)[order]]
    return post, prior, homes


def _draw_atlas(ax, M, homes, title, vmax):
    im = ax.imshow(np.abs(M.values), aspect="auto", cmap=SEQ, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(FACTORS9)))
    ax.set_xticklabels([PRETTY[f] for f in FACTORS9], rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks([]); ax.set_title(title, fontsize=10)
    # separators between home-factor blocks
    bnd = [i for i in range(1, len(homes)) if homes[i] != homes[i - 1]]
    for b in bnd:
        ax.axhline(b - 0.5, color="white", lw=1.4)
    # block labels
    starts = [0] + bnd
    for s in starts:
        ax.text(-0.7, s + 0.4, PRETTY.get(homes[s], homes[s]), ha="right", va="top",
                fontsize=6.8, color=MUTE, rotation=0)
    return im


def fig_empirical_atlas():
    post, _, homes = _atlas_matrix()
    fig, ax = plt.subplots(figsize=(6.4, max(7.5, len(post) * 0.105)))
    im = _draw_atlas(ax, post, homes, "Empirical atlas — posterior loadings (the data)", 0.95)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("|posterior loading|", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    _save(fig, "fig_empirical_atlas.png")


def fig_prior_posterior():
    post, prior, homes = _atlas_matrix()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, max(7.5, len(post) * 0.105)))
    _draw_atlas(axes[0], prior, homes, "Prior atlas (theory expects)", 0.95)
    im = _draw_atlas(axes[1], post, homes, "Empirical atlas (data delivers)", 0.95)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("loading magnitude", fontsize=8); cb.ax.tick_params(labelsize=7)
    _save(fig, "fig_prior_posterior.png")


# ============================================================ 5. WAIC
def fig_waic():
    df = pd.read_csv(REPORTS / "05_waic.csv").sort_values("d_waic")
    short = {"bifactor (G + specifics)": "bifactor\n(G + specifics)",
             "correlated-factors (no G)": "correlated-factors\n(no G)",
             "unidimensional (G only)": "unidimensional\n(G only)"}
    y = np.arange(len(df))[::-1]
    colors = [KR, ACCENT, OPEN][:len(df)]
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    ax.barh(y, df.d_waic, color=colors, height=0.6, zorder=3)
    for yi, d in zip(y, df.d_waic, strict=False):
        ax.text(d + 800, yi, ("preferred" if d == 0 else f"+{d:,.0f}"), va="center",
                fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([short.get(m, m) for m in df.model], fontsize=8.5)
    ax.set_xlabel(r"$\Delta$WAIC vs the best model  (lower = better fit)")
    ax.set_xlim(0, 60000); ax.set_title("The bifactor structure is decisively preferred")
    _save(fig, "fig_waic.png")


# ============================================================ 6. invariance (2 panels)
def fig_invariance():
    cong = pd.read_csv(REPORTS / "06_congruence.csv")
    facs = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep"]
    pairs = ["BP–SZ", "BP–DR", "SZ–DR"]
    M = np.full((len(facs), 3), np.nan)
    for _, r in cong.iterrows():
        if r.factor in facs and r.pair in pairs:
            M[facs.index(r.factor), pairs.index(r.pair)] = r.phi_mean
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    im = ax.imshow(M, cmap=LinearSegmentedColormap.from_list("inv", ["#B42318", "#E0A800", "#0F766E"]),
                   vmin=0.7, vmax=1.0, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(pairs, fontsize=9)
    ax.set_yticks(range(len(facs))); ax.set_yticklabels([PRETTY[f] for f in facs], fontsize=9)
    for i in range(len(facs)):
        for j in range(3):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white", fontweight="bold")
    ax.set_title("Metric invariance — Tucker $\\varphi$")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=7)
    cb.set_label(r"$\varphi$ (≥0.95 invariant)", fontsize=7.5)
    # panel 2: mania + substance per-cohort loadings
    inv9 = pd.read_csv(REPORTS / "13_invariance9_loadings.csv")
    ax2 = axes[1]
    items = inv9.item.tolist()
    x = np.arange(len(items)); w = 0.26
    for k, (coh, col) in enumerate(zip(["BP", "SZ", "DR"], [ACCENT, VIOLET, CAV], strict=False)):
        vals = pd.to_numeric(inv9[coh], errors="coerce").fillna(0.0).values
        ax2.bar(x + (k - 1) * w, vals, w, label=coh, color=col, zorder=3)
    ax2.set_xticks(x); ax2.set_xticklabels(items, rotation=35, ha="right", fontsize=7.5)
    ax2.set_ylabel("loading"); ax2.set_title("New axes across cohorts (mania · substance)")
    ax2.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    ax2.annotate("Altman floors\nin DR (0.10)", xy=(1 + w, 0.10), xytext=(1.7, 0.52),
                 fontsize=7.3, color=OPEN, ha="center",
                 arrowprops=dict(arrowstyle="->", color=OPEN, lw=0.9))
    _save(fig, "fig_invariance.png")


# ============================================================ 7. mixed-block PPC
def fig_ppc():
    df = pd.read_csv(REPORTS / "12_mixed_ppc.csv")
    rate = df[df.type.isin(["binary", "ordinal"])].copy().sort_values(["home", "observed"])
    y = np.arange(len(rate))
    cmap = {"suicidality": ACCENT, "developmental_risk": KR, "substance": VIOLET}
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for yi, (_, r) in zip(y, rate.iterrows(), strict=False):
        ax.plot([r.pred_lo, r.pred_hi], [yi, yi], color=MUTE, lw=2.2, alpha=0.5, zorder=2)
        ax.plot(r.pred_mean, yi, "|", color=MUTE, ms=9, zorder=3)
        ax.plot(r.observed, yi, "o", color=cmap.get(r.home, ACCENT), ms=6, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(rate.item, fontsize=7.8)
    ax.set_xlabel("endorsement rate / mean — observed (dot) vs 90% posterior-predictive interval (bar)")
    ax.set_title("Absolute fit: 21/22 non-Gaussian indicators reproduce the data")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=PRETTY.get(h, h))
               for h, c in cmap.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    ax.text(0.015, 0.985, "Lone exception (not shown): isf09a, the suicide-attempt\n"
            "count (90.8% zeros) — a plain NegBinom over-predicts the tail.\n"
            "An item-level caveat; the factor is carried by its 7 binary ISF items.",
            transform=ax.transAxes, fontsize=6.9, color=OPEN, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=OPEN, lw=0.6, alpha=0.95))
    _save(fig, "fig_ppc.png")


# ============================================================ 8. reliability tiers
def fig_reliability():
    # aggregate counts from reports/07_scoring_report.md (committed, shareable)
    tiers = {  # well, partial, prior-dominated
        "G (severity)": (8606, 249, 158), "cognition": (6451, 56, 2506),
        "metabolic": (8515, 67, 431), "inflammatory": (7102, 227, 1684),
        "sleep": (7522, 122, 1369), "mania": (0, 8594, 419)}
    N = 9013
    labels = list(tiers)[::-1]
    well = np.array([tiers[k][0] for k in labels]) / N
    part = np.array([tiers[k][1] for k in labels]) / N
    prior = np.array([tiers[k][2] for k in labels]) / N
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.barh(y, well, color=KR, label="well-characterised (≥3 indicators)", zorder=3)
    ax.barh(y, part, left=well, color=CAV, label="partial (1–2)", zorder=3)
    ax.barh(y, prior, left=well + part, color=OPEN, alpha=0.85, label="prior-dominated (0)", zorder=3)
    for yi, k in zip(y, labels, strict=False):
        if tiers[k][2] >= 400:
            ax.text(0.995, yi, f"{tiers[k][2]:,} prior-dom.", va="center", ha="right",
                    fontsize=7.3, color="white", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1); ax.set_xlabel("fraction of 9,013 patients")
    ax.set_title("Per-patient reliability tiers — coverage is flagged, not assumed")
    ax.legend(frameon=False, fontsize=7.6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    _save(fig, "fig_reliability.png")


# ============================================================ 9. soft-prior shrinkage (concept)
def fig_soft_priors():
    x = np.linspace(-0.5, 1.6, 600)

    def npdf(m, s):
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    prim = npdf(0.6, 0.3) * (x > 0)                     # truncated normal+
    ax.plot(x, prim, color=KR, lw=2, label=r"primary  $\mathcal{N}^+(0.6,\,0.3^2)$")
    ax.fill_between(x, prim, color=KR, alpha=0.10)
    ax.plot(x, npdf(0, 0.25), color=ACCENT, lw=2, label=r"plausible cross  $\mathcal{N}(0,\,0.25^2)$")
    ax.plot(x, npdf(0, 0.05), color=MUTE, lw=2, ls="--", label=r"unlikely  $\mathcal{N}(0,\,0.05^2)$")
    ax.set_yticks([]); ax.set_xlabel(r"loading $\lambda_{jk}$")
    ax.set_title("Soft priors: theory proposes, the likelihood disposes")
    ax.axvline(0, color=MUTE, lw=0.7)
    ax.annotate("the data can pull any\nloading away from its prior", xy=(0.95, 0.4), xytext=(1.05, 1.6),
                fontsize=7.6, color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _save(fig, "fig_soft_priors.png")


# ============================================================ 10. coverage / missingness
def fig_coverage():
    cov = pd.read_csv(REPORTS / "01_coverage_by_indicator.csv")
    g = cov.groupby("home_factor")[["obs_bp", "obs_sz", "obs_dr"]].mean()
    g = g.reindex([f for f in FACTORS9 if f in g.index])
    y = np.arange(len(g)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for k, (c, col) in enumerate(zip(["obs_bp", "obs_sz", "obs_dr"], [ACCENT, VIOLET, CAV], strict=False)):
        ax.barh(y + (k - 1) * w, g[c].values, w, label=c[-2:].upper(), color=col, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([PRETTY.get(f, f) for f in g.index], fontsize=9)
    ax.set_xlim(0, 1); ax.set_xlabel("mean observed fraction (per cohort)")
    ax.set_title("Structured missingness: coverage varies by factor and cohort")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    _save(fig, "fig_coverage.png")


# ============================================================ M4. prognostic atlas (the "so what")
_M4_EP = [("egf_remission", "func. remission", "good"), ("egf_recovery", "func. recovery", "good"),
          ("egf_deterioration", "deterioration", "poor"), ("egf_sustained_impair", "sustained impair", "poor"),
          ("cgi_remission", "CGI remission", "good"), ("cgi_relapse", "relapse surrog.", "poor"),
          ("cgi_sustained_severe", "sustained illness", "poor")]


def fig_m4_atlas():
    a = pd.read_csv(REPORTS / "45_archetype_atlas.csv").sort_values("egf_remission", ascending=False)
    rows = list(a["archetype"])
    cols = [e for e, _, _ in _M4_EP]
    rate = a.set_index("archetype")[cols].reindex(rows).values
    adv = rate.copy()
    for j, (_, _, pol) in enumerate(_M4_EP):           # adversity: good -> 1-rate (red = adverse uniformly)
        if pol == "good":
            adv[:, j] = 1.0 - adv[:, j]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    im = ax.imshow(adv, cmap=DIV.reversed(), norm=TwoSlopeNorm(vmin=0.15, vcenter=0.5, vmax=0.85), aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{lbl}\n({'↑' if pol == 'good' else '↓'})" for _, lbl, pol in _M4_EP],
                       rotation=30, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r}  (n={int(n)})" for r, n in zip(rows, a["n"])], fontsize=8)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = rate[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if (adv[i, j] > 0.72 or adv[i, j] < 0.28) else INK)
    ax.set_title("The archetype prognostic atlas — 2-year clinical-endpoint rates", pad=10)
    ax.text(0.5, -0.42, "blue = favourable · red = adverse (polarity-aware); functional remission "
            "ranges 14%–60% across archetypes", transform=ax.transAxes, ha="center",
            fontsize=8, color=MUTE)
    _save(fig, "m4_atlas.png")


def fig_m4_dominance():
    h = pd.read_csv(REPORTS / "44_h2h_dsm5.csv")
    order = ["DSM-5 beyond foundation (A−D)", "map beyond foundation (C−D)",
             "map beyond DSM-5 (B−A)", "DSM-5 beyond map (B−C)"]
    short = ["DSM-5\nvs base", "map\nvs base", "map beyond\nDSM-5", "DSM-5 beyond\nmap"]
    outs = [("egf", "EGF (functioning)"), ("cgi_s", "CGI-S (severity)")]
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=False)
    for k, (oc, title) in enumerate(outs):
        sub = h[h.outcome == oc].set_index("contrast").reindex(order)
        vals, ses = sub["d_elpd"].values, sub["se"].values
        colors = [KR if (v - 2 * s) > 0 else MUTE for v, s in zip(vals, ses)]
        ax[k].bar(range(4), vals, color=colors, zorder=3)
        ax[k].errorbar(range(4), vals, yerr=2 * ses, fmt="none", ecolor=INK, capsize=2, lw=0.8)
        ax[k].axhline(0, color=INK, lw=0.8)
        ax[k].set_xticks(range(4)); ax[k].set_xticklabels(short, fontsize=7)
        ax[k].set_title(title, fontsize=9.5)
        if k == 0:
            ax[k].set_ylabel(r"$\Delta$ELPD vs reference ($\pm2$SE)")
    fig.suptitle("Map vs DSM-5: co-informative (each adds beyond the other)", y=1.02,
                 fontsize=10.5, fontweight="bold", color=ACCENTDK)
    _save(fig, "m4_dominance.png")


def fig_m4_value():
    c = pd.read_csv(REPORTS / "46_clinical_value.csv")
    c = c[c.model != "ΔAUC (map added)"]
    eps = ["egf_remission", "egf_deterioration", "egf_sustained_impair", "cgi_relapse"]
    models = ["DSM-5 only", "map only", "reference (DSM-5+severity)", "reference + map"]
    cmap = {"DSM-5 only": OPEN, "map only": "#9DBCE0", "reference (DSM-5+severity)": MUTE,
            "reference + map": ACCENT}
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    x = np.arange(len(eps)); w = 0.2
    for i, m in enumerate(models):
        vals = [float(c[(c.endpoint == e) & (c.model == m)]["auc"].iloc[0]) for e in eps]
        ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=cmap[m], zorder=3)
    ax.axhline(0.5, color=INK, lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([e.replace("egf_", "").replace("cgi_", "CGI ") for e in eps], fontsize=8)
    ax.set_ylim(0.45, 0.9); ax.set_ylabel("cross-validated AUC")
    ax.set_title("Clinical value: discrimination of 2-year endpoints (5-fold CV)", pad=10)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _save(fig, "m4_value.png")


_M5_Q = {"lithium_bp": "lithium (BP)", "antipsychotic_bp": "antipsychotic (BP)", "clozapine_sz": "clozapine (SZ)"}
_M5_AX = ["cognition", "metabolic", "inflammatory"]


def fig_m5_identification():
    s = pd.read_csv(REPORTS / "55_propensity_summary.csv")
    s = s[(s["mode"] == "active_comparator") & s["question"].isin(_M5_Q)].set_index("question").reindex(_M5_Q)
    x = np.arange(len(s)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - w / 2, s["max_smd_before"], w, label="before IPTW", color=OPEN, zorder=3)
    ax.bar(x + w / 2, s["max_smd_after"], w, label="after IPTW", color=ACCENT, zorder=3)
    ax.axhline(0.1, color=INK, ls=":", lw=0.8); ax.axhline(0.25, color=KR, ls="--", lw=0.9)
    ax.text(len(s) - 0.5, 0.26, "0.25 (imbalance)", ha="right", va="bottom", fontsize=6.5, color=KR)
    ax.set_xticks(x); ax.set_xticklabels([_M5_Q[q] for q in s.index], fontsize=8.5)
    ax.set_ylabel("max |SMD| (covariate imbalance)")
    for xi, (_, r) in zip(x, s.iterrows()):
        ax.text(xi, max(r["max_smd_before"], r["max_smd_after"]) + 0.015,
                f"overlap {r['frac_in_support']:.0%}", ha="center", fontsize=7, color=MUTE)
    ax.set_title("Identification gate: covariate balance before/after IPTW", pad=10)
    ax.text(0.5, -0.2, "lithium balances to ≈0; antipsychotic balances; clozapine cannot be balanced "
            "(channeling)", transform=ax.transAxes, ha="center", fontsize=7.5, color=MUTE)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "m5_identification.png")


def fig_m5_moderation():
    m = pd.read_csv(REPORTS / "56_moderation.csv")
    outs = [("functioning", "functioning (EGF)"), ("cgi_response", "CGI response")]
    qs = [q for q in _M5_Q if (m.question == q).any()]
    fig, axes = plt.subplots(len(outs), len(qs), figsize=(2.7 * len(qs), 4.4), squeeze=False)
    for ri, (oc, olab) in enumerate(outs):
        for ci, q in enumerate(qs):
            ax = axes[ri][ci]
            r = m[(m.question == q) & (m.outcome == oc)]
            if len(r):
                r = r.iloc[0]
                means = [r[f"int_{a}"] for a in _M5_AX]
                los = [r[f"int_{a}_lo"] for a in _M5_AX]; his = [r[f"int_{a}_hi"] for a in _M5_AX]
                cols = [KR if (lo > 0 or hi < 0) else MUTE for lo, hi in zip(los, his)]
                for yi, (mn, lo, hi, c) in enumerate(zip(means, los, his, cols)):
                    ax.plot([lo, hi], [yi, yi], color=c, lw=1.6, zorder=2)
                    ax.plot(mn, yi, "o", color=c, ms=4, zorder=3)
                ax.axvline(0, color=INK, lw=0.8)
            ax.set_yticks(range(len(_M5_AX)))
            ax.set_yticklabels(_M5_AX if ci == 0 else [], fontsize=7.5)
            if ri == 0:
                ax.set_title(_M5_Q[q], fontsize=8.5)
            if ci == 0:
                ax.set_ylabel(olab, fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("Does the durable biology moderate treatment response? (treat × axis, 94% HDI)",
                 y=1.0, fontsize=10, fontweight="bold", color=ACCENTDK)
    fig.text(0.5, -0.01, "red = HDI excludes 0. Lithium: null on both. Antipsychotic: metabolic/"
             "inflammatory on functioning. Clozapine: channeled (uninterpretable).",
             ha="center", fontsize=7, color=MUTE)
    _save(fig, "m5_moderation.png")


if __name__ == "__main__":
    print(f"Generating manuscript figures -> {OUT.relative_to(REPO)}")
    for fn in [fig_biology_g, fig_phi, fig_empirical_atlas, fig_prior_posterior, fig_waic,
               fig_invariance, fig_ppc, fig_reliability, fig_soft_priors, fig_coverage,
               fig_m4_atlas, fig_m4_dominance, fig_m4_value, fig_m5_identification, fig_m5_moderation]:
        try:
            fn()
        except Exception as e:
            print(f"  !! {fn.__name__} failed: {type(e).__name__}: {e}")
    print("done.")
