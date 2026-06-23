#!/usr/bin/env python3
"""Generate the FACE-ATLAS manuscript figures from committed AGGREGATE artifacts.

Reads only shareable aggregates (reports/*.csv, docs/figures/*.csv, configs/) — never per-patient
data — and writes PNGs into report/figures/. House style matches the report (deep-blue accent,
light->dark sequential blue, sans labels). Run:  python3 report/make_figures.py
"""
from __future__ import annotations

import json
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

# ---- Gaussian-copula result root (M2-M5 milestone figures are regenerated from here) ----
ROOP = REPO / "results" / "face"
STRATA = ROOP / "strata_oop"
TEMPORAL = ROOP / "temporal_oop"
PROGNOSIS = ROOP / "prognosis_oop"
TREATMENT = ROOP / "treatment_oop"

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


# ════════════════════════════════════════════════════════════════════════════════════════════
# MILESTONE FIGURES — regenerated from the Gaussian-copula result files (results/face/*_oop/)
# ════════════════════════════════════════════════════════════════════════════════════════════

# ---- ordered axis lists / archetype identities (shared by the milestone figures) ----
_ARCH_NAME = {0: "A0 biological", 1: "A1 low-burden", 2: "A2 severe non-biological",
              3: "A3 psychiatric symptom"}
_ARCH_SHORT = {0: "A0 biological", 1: "A1 low-burden", 2: "A2 severe-non-bio", 3: "A3 symptom"}
_VERDICT_COL = {"trait": KR, "mixed": CAV, "state": OPEN,
                "uninformative": MUTE, "invariant": KR}


def _pool_archetype_remission(a):
    """Pool the per-cohort archetype remission rates across cohorts, weighting each cohort by
    its observed denominator (n_rem = patients with the outcome observed). This coverage-weighted
    pool — not sum(n_rem)/sum(n) — reproduces the canonical archetype atlas (A1 60% > A3 43% >
    A2 32% > A0 27%); a naive sum(n_rem)/sum(n) would conflate the denominator with a numerator."""
    rows = []
    for k, g in a.groupby("archetype"):
        w = g["n_rem"].astype(float)
        rate = float((g["remission_rate"].astype(float) * w).sum() / w.sum())
        rows.append({"archetype": int(k), "rate": rate, "n": int(g["n"].sum())})
    out = pd.DataFrame(rows).sort_values("rate", ascending=False).reset_index(drop=True)
    return out


# ============================================================ M2.a structure-discovery (continuum)
def fig_m2_structure():
    """Silhouette-vs-K against the single-Gaussian falsification null -> a continuum, not clusters."""
    d = json.load(open(STRATA / "structure" / "data.json"))
    sil = d["diagnostics_A"]["silhouette"]["silhouette"]      # {"2": .., "3": .., ...}
    ks = sorted(int(k) for k in sil)
    vals = [sil[str(k)] for k in ks]
    peak = max(vals)
    fn = d["falsification_null"]                               # serialized single-Gaussian null (traceable)
    real_sil = float(fn["real"]["best_silhouette"])
    null_m = float(fn["null_mean"]["best_silhouette"])
    null_s = float(fn["null_sd"]["best_silhouette"])
    z = float(fn["z"]["best_silhouette"])

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    # null band
    ax.axhspan(null_m - null_s, null_m + null_s, color=OPEN, alpha=0.14, zorder=1)
    ax.axhline(null_m, color=OPEN, lw=1.0, ls="--", zorder=2)
    ax.text(ks[-1], null_m + null_s + 0.0015, f"single-Gaussian null  {null_m:.3f} ± {null_s:.3f}",
            ha="right", va="bottom", fontsize=7.4, color=OPEN)
    # real silhouette curve
    ax.plot(ks, vals, "-o", color=ACCENT, lw=1.8, ms=5, zorder=4, label="real coordinates")
    ax.plot(ks[0], vals[0], "o", color=ACCENTDK, ms=8, zorder=5)
    ax.annotate(f"real best {real_sil:.3f} ≈ null {null_m:.3f} ± {null_s:.3f}\n"
                f"(z = {z:.2f}) → continuum, not clusters",
                xy=(ks[0], vals[0]), xytext=(ks[0] + 1.6, vals[0] - 0.012),
                fontsize=7.6, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    ax.set_xticks(ks); ax.set_xlabel("number of clusters $K$")
    ax.set_ylabel("mean silhouette")
    ax.set_ylim(min(min(vals), null_m - null_s) - 0.006, max(peak, null_m + null_s) + 0.010)
    ax.set_title("Structure-discovery: a continuum, not clusters", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    # side panel: gate verdict from verdict_A.evidence
    ev = d["verdict_A"]["evidence"]
    ax2 = axes[1]; ax2.axis("off")
    rows = [("Hopkins (cluster tendency)", f"{ev['hopkins']:.3f}", "0.5 = random"),
            ("dip test  p (PC1)", f"{ev['dip_pc1_p']:.2f}", "unimodal"),
            ("HDBSCAN clusters", f"{int(ev['hdbscan_n'])}", "none found"),
            ("GMM K (under uncertainty)", "1", "single mode"),
            ("silhouette peak", f"{ev['silhouette_peak']:.3f}", "≈ null")]
    ax2.text(0.0, 1.0, "structure-gate verdict", fontsize=9.5, fontweight="bold",
             color=ACCENTDK, transform=ax2.transAxes, va="top")
    ax2.text(0.0, 0.90, d["verdict_A"]["label"].upper(), fontsize=11, fontweight="bold",
             color=KR, transform=ax2.transAxes, va="top")
    y = 0.74
    for name, val, note in rows:
        ax2.text(0.0, y, name, fontsize=7.6, color=INK, transform=ax2.transAxes, va="top")
        ax2.text(0.70, y, val, fontsize=7.6, color=ACCENT, fontweight="bold",
                 transform=ax2.transAxes, va="top", ha="right")
        ax2.text(0.76, y, note, fontsize=6.8, color=MUTE, transform=ax2.transAxes,
                 va="top", ha="left")
        y -= 0.135
    _save(fig, "m2_structure.png")


# ============================================================ M2.b archetype simplex profiles (A=4)
def fig_m2_archetypes():
    """The A=4 archetype simplex: 4x9 heatmap of mean axis profile per archetype."""
    df = pd.read_csv(STRATA / "consolidate" / "archetype_profiles.csv")
    df = df[df.arm == "A_all9"].sort_values("archetype")
    M = df[FACTORS9].astype(float).values                      # 4 x 9
    rows = [_ARCH_NAME[int(a)] for a in df.archetype]
    vmax = float(np.nanmax(np.abs(M)))
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    im = ax.imshow(M, cmap=DIV, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), aspect="auto")
    ax.set_xticks(range(len(FACTORS9)))
    ax.set_xticklabels([PRETTY[f] for f in FACTORS9], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.62 * vmax else INK)
    ax.set_title("The A = 4 archetype simplex — mean axis profile per corner", pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("mean coordinate (SD units)", fontsize=8); cb.ax.tick_params(labelsize=7)
    _save(fig, "m2_archetypes.png")


# ============================================================ M2.c per-axis variance / K-family
def fig_m2_regions():
    """The K=2 split is symptom-burden (suicidality) not severity/biology; finer K adds the gradient."""
    menu = pd.read_csv(STRATA / "consolidate" / "k_family_menu.csv").set_index("K")
    # per-axis eta2 at K=2
    axis_eta = {f: float(menu.loc[2, f"eta_{f}"]) for f in FACTORS9}
    order = sorted(FACTORS9, key=lambda f: axis_eta[f])
    yvals = [axis_eta[f] for f in order]
    bio = {"metabolic", "inflammatory"}
    colors = [KR if f in bio else (OPEN if f == "overall_severity" else ACCENT) for f in order]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    y = np.arange(len(order))
    ax.barh(y, yvals, color=colors, height=0.66, zorder=3)
    for yi, (f, v) in enumerate(zip(order, yvals)):
        ax.text(v + 0.006, yi, f"{v:.3f}", va="center", ha="left", fontsize=7.4, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([PRETTY[f] for f in order], fontsize=8.5)
    ax.set_xlim(0, max(yvals) * 1.18)
    ax.set_xlabel(r"variance explained by the $K{=}2$ split  ($\eta^2$ per axis)")
    ax.set_title("The split is symptom-burden, not severity or biology", pad=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ACCENT, label="symptom axes"),
                       Patch(color=OPEN, label="G (severity)"),
                       Patch(color=KR, label="biology")],
              loc="lower right", frameon=False, fontsize=7.6)

    # panel 2: severity/biology eta2 rising across K=2,3,4
    ax2 = axes[1]
    ks = [2, 3, 4]
    series = {"G (severity)": (OPEN, [menu.loc[k, "eta_overall_severity"] for k in ks]),
              "metabolic": (KR, [menu.loc[k, "eta_metabolic"] for k in ks]),
              "inflammatory": (CAV, [menu.loc[k, "eta_inflammatory"] for k in ks])}
    for lab, (col, vv) in series.items():
        ax2.plot(ks, vv, "-o", color=col, lw=1.7, ms=5, label=lab)
    ax2.set_xticks(ks); ax2.set_xlabel("K (nested family)")
    ax2.set_ylabel(r"$\eta^2$")
    ax2.set_title("finer K captures the gradient\nK = 2 discards", fontsize=9.5)
    ax2.legend(frameon=False, fontsize=7.4, loc="upper left")
    _save(fig, "m2_regions.png")


# ============================================================ M2.d assignment confidence (blends)
def fig_m2_confidence():
    """Honest 'most patients are blends': assignment entropy + boundary fraction across the K-family."""
    menu = pd.read_csv(STRATA / "consolidate" / "k_family_menu.csv")
    use = json.load(open(STRATA / "usefulness" / "data.json"))["assignment"]
    ks = menu["K"].astype(int).tolist()
    ent = menu["median_norm_entropy"].astype(float).tolist()
    conf = menu["confident_dominant_frac"].astype(float).tolist()
    bfrac = float(use["boundary_frac"])

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(ks)); w = 0.5
    ax.bar(x, ent, w, color=ACCENT, zorder=3, label="median normalised entropy (0 = crisp, 1 = maximal blend)")
    for xi, e in zip(x, ent):
        ax.text(xi, e + 0.012, f"{e:.2f}", ha="center", fontsize=7.6, color=INK)
    ax.axhline(0.5, color=MUTE, lw=0.7, ls=":", zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([f"K = {k}" for k in ks], fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("median assignment entropy")
    ax.set_title("Soft assignment: most patients are blends", pad=10)
    ax.legend(frameon=False, fontsize=7.4, loc="upper right")
    ax.text(0.5, 1.14, f"tessellation entropy stays high (median {ent[0]:.2f} at K=2); only {bfrac:.1%} of patients "
            "sit on a hard\nboundary, and the archetype simplex is softer still --- most have no dominant archetype.",
            transform=ax.transAxes, fontsize=7.2, color=MUTE, va="top", ha="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=MUTE, lw=0.5, alpha=0.95))
    _save(fig, "m2_confidence.png")


# ============================================================ M2.e embedding (viz-only PCA scatter)
def fig_m2_embedding():
    """Viz-only 2-D PCA of the 9-dim copula coordinate means, tinted by G — a smear, not islands."""
    f = STRATA / "coordinates" / "coordinates_full.parquet"
    if not f.exists():
        print("  -- m2_embedding skipped: no copula coordinate file")
        return
    df = pd.read_parquet(f)
    cols = [f"{a}__mean" for a in FACTORS9]
    X = df[cols].astype(float).values
    ok = np.isfinite(X).all(1)
    X = X[ok]
    g = df["overall_severity__mean"].astype(float).values[ok]
    # standardise + PCA via SVD (viz-only, never a clustering input)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    U, S, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
    pc = (Xs - Xs.mean(0)) @ Vt[:2].T
    ev = (S ** 2) / (S ** 2).sum()

    coh = df["cohort"].astype(str).values[ok]
    infl = df["inflammatory__mean"].astype(float).values[ok]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0))
    # panel 1: by cohort -- the transdiagnostic intermixing (no diagnostic territories)
    ax = axes[0]
    for c, col in {"bp": ACCENT, "sz": OPEN, "dr": KR}.items():
        m = coh == c
        ax.scatter(pc[m, 0], pc[m, 1], c=col, s=4, alpha=0.40, linewidths=0,
                   rasterized=True, label=c.upper())
    ax.set_title("by cohort --- fully intermixed", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.6, markerscale=2.5, loc="upper right")
    # panel 2: the severity (G) gradient
    sc1 = axes[1].scatter(pc[:, 0], pc[:, 1], c=g, cmap=SEQ, s=4, alpha=0.45,
                          linewidths=0, rasterized=True)
    axes[1].set_title("severity ($G$) gradient", fontsize=9.5)
    fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04).ax.tick_params(labelsize=6.5)
    # panel 3: the inflammatory gradient -- crosses the cloud in a different direction
    sc2 = axes[2].scatter(pc[:, 0], pc[:, 1], c=infl, cmap=SEQ, s=4, alpha=0.45,
                          linewidths=0, rasterized=True)
    axes[2].set_title("inflammatory gradient", fontsize=9.5)
    fig.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04).ax.tick_params(labelsize=6.5)
    for ax in axes:
        ax.set_xlabel(f"PC1 ({ev[0]:.0%})", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_ylabel(f"PC2 ({ev[1]:.0%})", fontsize=8)
    fig.suptitle(f"The continuum in one picture (PCA, viz-only · n = {len(pc):,}) --- "
                 "severity and biology cross in different directions",
                 y=1.03, fontsize=10.5, fontweight="bold", color=ACCENTDK)
    fig.tight_layout()
    _save(fig, "m2_embedding.png")


# ============================================================ M3.a trait/state (ICC per axis)
def fig_m3_traitstate():
    """Biology is trait, symptoms are state: ICC per axis with HDI bars, coloured by verdict."""
    df = pd.read_csv(TEMPORAL / "trait_state" / "trait_state.csv")
    df = df.sort_values("icc")                                  # low at bottom, high at top
    y = np.arange(len(df))
    icc = df["icc"].values
    lo = (icc - df["icc_lo"].values); hi = (df["icc_hi"].values - icc)
    cols = [_VERDICT_COL.get(v, MUTE) for v in df["verdict"]]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.barh(y, icc, color=cols, height=0.62, zorder=3)
    ax.errorbar(icc, y, xerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2, zorder=4)
    ax.axvline(0.5, color=INK, lw=0.8, ls="--", zorder=2)
    ax.text(0.5, len(df) - 0.4, "0.5", ha="center", va="bottom", fontsize=7, color=MUTE)
    for yi, (a, v) in enumerate(zip(df["axis"], icc)):
        ax.text(v + max(hi[yi], 0.0) + 0.012, yi, f"{v:.2f}", va="center", ha="left",
                fontsize=7.4, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([PRETTY.get(a, a) for a in df["axis"]], fontsize=8.5)
    ax.set_xlim(0, 1.0); ax.set_xlabel("intraclass correlation  (ICC, V0→V1→V2)")
    ax.set_title("Biology is trait, symptoms are state", pad=10)
    from matplotlib.patches import Patch
    seen = {}
    for v in df["verdict"]:
        seen.setdefault(v, _VERDICT_COL.get(v, MUTE))
    ax.legend(handles=[Patch(color=c, label=v) for v, c in seen.items()],
              loc="lower right", frameon=False, fontsize=7.6)
    _save(fig, "m3_traitstate.png")


# ============================================================ M3.b measurement invariance over time
def fig_m3_invariance():
    """The measurement holds over time: min Tucker phi per backbone axis vs the 0.95 invariance line."""
    df = pd.read_csv(TEMPORAL / "invariance" / "license.csv").sort_values("min_phi")
    y = np.arange(len(df))
    phi = df["min_phi"].values
    cols = [KR if p >= 0.95 else CAV for p in phi]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.barh(y, phi, color=cols, height=0.6, zorder=3)
    ax.axvline(0.95, color=INK, lw=0.9, ls="--", zorder=4)
    ax.text(0.95, len(df) - 0.4, "0.95 (invariant)", ha="center", va="bottom",
            fontsize=7, color=INK)
    for yi, p in zip(y, phi):
        ax.text(p - 0.004, yi, f"{p:.3f}", va="center", ha="right", fontsize=7.6,
                color="white", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([PRETTY.get(a, a) for a in df["axis"]], fontsize=9)
    ax.set_xlim(0.90, 1.0); ax.set_xlabel(r"minimum Tucker $\varphi$ across visits")
    ax.set_title("The map holds over time — all backbone axes invariant", pad=10)
    _save(fig, "m3_invariance.png")


# ============================================================ M3.c spine-corner persistence (G4)
def fig_m3_spinecorner():
    """Corners stay put, the spine slides: archetype-identity persistence + the G3<->G4 agreement."""
    pj = json.load(open(TEMPORAL / "persistence" / "persistence.json"))
    tt, mp, syn = pj["trajectory_types"], pj["membership_persistence"], pj["g3_g4_synthesis"]
    ts = pd.read_csv(TEMPORAL / "trait_state" / "trait_state.csv")[["axis", "icc", "verdict"]]
    rc = pd.read_csv(TEMPORAL / "persistence" / "reliable_change.csv")[["axis", "frac_reliable"]]
    m = ts.merge(rc, on="axis")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    # panel A: trajectory stability (the corner stays put)
    ax = axes[0]
    labels, vals, cols = ["stable", "drifting", "oscillating"], \
        [tt["stable"], tt["drifting"], tt["oscillating"]], [KR, CAV, OPEN]
    yb = np.arange(len(labels))[::-1]
    ax.barh(yb, vals, color=cols, height=0.6, zorder=3)
    for yi, v in zip(yb, vals):
        ax.text(v + 0.01, yi, f"{v:.0%}", va="center", ha="left", fontsize=8, color=INK)
    ax.set_yticks(yb); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 0.78); ax.set_xlabel("share of patients (V0→V2)")
    ax.set_title(f"Archetype identity persists\n(weight-cosine median {mp['cos_median']:.2f})", fontsize=9.3)
    # panel B: ICC vs reliable-change rate — the two routes agree (rho)
    ax = axes[1]
    ax.scatter(m["icc"], m["frac_reliable"], c=[_VERDICT_COL.get(v, MUTE) for v in m["verdict"]],
               s=46, zorder=3, edgecolor="white", linewidth=0.6)
    for _, r in m.iterrows():
        ax.annotate(PRETTY.get(r["axis"], r["axis"]), (r["icc"], r["frac_reliable"]),
                    fontsize=6.6, color=MUTE, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("trait fraction  (ICC, error-corrected)"); ax.set_ylabel("reliable-change rate")
    ax.set_title(f"Trait axes hold, state axes move\n(G3 ⟷ G4 agree, $\\rho={syn['spearman_rho']:.2f}$)",
                 fontsize=9.3)
    fig.suptitle("Corners stay put, the spine slides", y=1.04, fontsize=10.5,
                 fontweight="bold", color=ACCENTDK)
    fig.tight_layout()
    _save(fig, "m3_spinecorner.png")


# ============================================================ M4.a incremental prognostic value
def fig_m4_value():
    """Incremental ELPD for predicting functioning (egf): archetypes lead, operative K = none."""
    c = pd.read_csv(PROGNOSIS / "incremental" / "incremental_comparison.csv")
    c = c[(c.outcome == "egf") & (c.model != "R3y")].copy()      # drop the reference (d_elpd=0)
    c = c.sort_values("d_elpd_vs_ref")                           # smallest at bottom
    pretty = {"+archetypesA": "+ archetypes (all-9)", "+archetypesB": "+ archetypes (specifics)",
              "+specifics8": "+ 8 specifics", "+tess_k4": "+ tessellation K=4",
              "+tess_k3": "+ tessellation K=3", "+tess_k2": "+ tessellation K=2",
              "+durable": "+ durable trio (EIV)"}
    y = np.arange(len(c))
    d = c["d_elpd_vs_ref"].values; se = c["se_d_elpd"].values
    cols = [KR if v == "predictive" else MUTE for v in c["verdict"]]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.barh(y, d, color=cols, height=0.62, zorder=3)
    ax.errorbar(d, y, xerr=se, fmt="none", ecolor=INK, elinewidth=0.9, capsize=2, zorder=4)
    ax.axvline(0, color=INK, lw=0.8, zorder=2)
    for yi, (v, s) in enumerate(zip(d, se)):
        ax.text(v + s + 1.0, yi, f"+{v:.0f}", va="center", ha="left", fontsize=7.6, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty.get(m, m) for m in c["model"]], fontsize=8.5)
    ax.set_xlabel(r"$\Delta$ELPD vs reference (DSM-5 + severity + baseline outcome) $\pm$ SE")
    ax.set_title("Incremental prognostic value for functioning (operative K = none)", pad=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=KR, label="predictive"), Patch(color=MUTE, label="ambiguous")],
              loc="lower right", frameon=False, fontsize=7.6)
    _save(fig, "m4_value.png")


# ============================================================ M4.b archetype prognostic atlas
def fig_m4_atlas():
    """2-year functional remission by archetype, pooled across cohorts."""
    a = pd.read_csv(PROGNOSIS / "endpoints" / "archetype_atlas.csv")
    a = a[a.outcome == "egf"].copy()
    pooled = _pool_archetype_remission(a)
    rows = [_ARCH_SHORT[int(k)] for k in pooled["archetype"]]
    y = np.arange(len(pooled))[::-1]
    rate = pooled["rate"].values
    # tint by archetype identity (biology corner = worst)
    pal = {0: CAV, 1: KR, 2: ACCENT, 3: VIOLET}
    cols = [pal[int(k)] for k in pooled["archetype"]]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.barh(y, rate, color=cols, height=0.62, zorder=3)
    for yi, (r, n) in zip(y, zip(rate, pooled["n"])):
        ax.text(r + 0.008, yi, f"{r:.0%}  (n={int(n):,})", va="center", ha="left",
                fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(rows, fontsize=9)
    ax.set_xlim(0, max(rate) * 1.30); ax.set_xlabel("2-year functional remission rate (EGF)")
    ax.set_title("2-year functional remission by archetype", pad=10)
    ax.text(0.0, -0.30, "biological corner (A0) worst · low-burden (A1) best — "
            "a transdiagnostic prognostic gradient (coverage-weighted pool across cohorts)",
            transform=ax.transAxes, fontsize=7.2, color=MUTE, va="top", ha="left")
    _save(fig, "m4_atlas.png")


# ============================================================ M5. treatment moderation (earned boundary)
_M5_QLAB = {"lithium_bp": "lithium (BP)", "antipsychotic_bp": "antipsychotic (BP)",
            "clozapine_sz": "clozapine (SZ)"}


def fig_m5_moderation():
    """The earned boundary: ATE on functioning per treatment question (durable rep), with E-values."""
    m = pd.read_csv(TREATMENT / "moderation" / "moderation.csv")
    m = m[(m.representation == "durable") & (m.outcome == "functioning")].copy()
    order = ["lithium_bp", "antipsychotic_bp", "clozapine_sz"]
    m["__o"] = m["question"].map({q: i for i, q in enumerate(order)})
    m = m.sort_values("__o")
    y = np.arange(len(m))[::-1]
    ate = m["ate"].values; lo = m["ate_lo"].values; hi = m["ate_hi"].values
    excl = m["ate_excludes0"].values
    cols = [ACCENT if e else MUTE for e in excl]
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    for yi, (a, l, h, c) in zip(y, zip(ate, lo, hi, cols)):
        ax.plot([l, h], [yi, yi], color=c, lw=2.0, zorder=2)
        ax.plot(a, yi, "o", color=c, ms=6, zorder=3)
    ax.axvline(0, color=INK, lw=0.9, ls="-", zorder=1)
    for yi, (_, r) in zip(y, m.iterrows()):
        ax.text(max(r["ate_hi"], 0) + 0.02, yi,
                f"ATE {r['ate']:+.2f}  ·  E-value {r['e_value']:.2f}",
                va="center", ha="left", fontsize=7.6, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([_M5_QLAB[q] for q in m["question"]], fontsize=9)
    ax.set_xlabel("ATE on functioning  (doubly-robust, [ate_lo, ate_hi])")
    lim = max(np.nanmax(np.abs(np.r_[lo, hi])) * 1.55, 0.6)
    ax.set_xlim(-lim, lim)
    ax.set_title("Treatment moderation: the earned boundary\n"
                 "(lithium null, antipsychotic suggestive, clozapine channeled)",
                 pad=10, fontsize=10)
    _save(fig, "m5_moderation.png")


# ============================================================ SYNTHESIS (the copula vertical, one figure)
def fig_synthesis():
    """One multi-panel summary of the chain M2 (continuum + A=4) -> M3 (metabolic trait) -> M4 (A0 worst)."""
    # data
    d = json.load(open(STRATA / "structure" / "data.json"))
    sil = d["diagnostics_A"]["silhouette"]["silhouette"]
    ks = sorted(int(k) for k in sil); sv = [sil[str(k)] for k in ks]
    ts = pd.read_csv(TEMPORAL / "trait_state" / "trait_state.csv").sort_values("icc")
    atl = pd.read_csv(PROGNOSIS / "endpoints" / "archetype_atlas.csv")
    atl = atl[atl.outcome == "egf"]
    pooled = _pool_archetype_remission(atl)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))

    # panel 1: continuum (silhouette vs null)
    ax = axes[0]
    ax.axhspan(0.137, 0.143, color=OPEN, alpha=0.14)
    ax.axhline(0.140, color=OPEN, lw=1.0, ls="--")
    ax.plot(ks, sv, "-o", color=ACCENT, lw=1.7, ms=4)
    ax.set_xticks(ks); ax.set_xlabel("K"); ax.set_ylabel("silhouette")
    ax.set_title("M2 · continuum + A=4", fontsize=9.5)
    ax.text(0.5, 0.04, "peak ≈ single-Gaussian null", transform=ax.transAxes,
            ha="center", fontsize=7, color=OPEN)

    # panel 2: ICC bar (trait/state)
    ax = axes[1]
    y = np.arange(len(ts))
    cols = [_VERDICT_COL.get(v, MUTE) for v in ts["verdict"]]
    ax.barh(y, ts["icc"].values, color=cols, height=0.6, zorder=3)
    ax.axvline(0.5, color=INK, lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([PRETTY.get(a, a) for a in ts["axis"]], fontsize=7.4)
    ax.set_xlim(0, 1); ax.set_xlabel("ICC")
    ax.set_title("M3 · metabolic 0.91 trait", fontsize=9.5)

    # panel 3: remission gradient
    ax = axes[2]
    pal = {0: CAV, 1: KR, 2: ACCENT, 3: VIOLET}
    yy = np.arange(len(pooled))[::-1]
    ax.barh(yy, pooled["rate"].values, color=[pal[int(k)] for k in pooled["archetype"]],
            height=0.6, zorder=3)
    for yi, r in zip(yy, pooled["rate"].values):
        ax.text(r + 0.01, yi, f"{r:.0%}", va="center", ha="left", fontsize=7.4, color=INK)
    ax.set_yticks(yy)
    ax.set_yticklabels([_ARCH_SHORT[int(k)] for k in pooled["archetype"]], fontsize=7.4)
    ax.set_xlim(0, max(pooled["rate"]) * 1.3); ax.set_xlabel("func. remission")
    ax.set_title("M4 · A0 worst (27%)", fontsize=9.5)

    fig.suptitle("The Gaussian-copula vertical: a continuum, stratified on durable biology, "
                 "forecasting functioning", y=1.04, fontsize=10.5, fontweight="bold",
                 color=ACCENTDK)
    fig.tight_layout()
    _save(fig, "synthesis.png")


# ============================================================ M1.x biology⊥G confound sensitivity
def fig_biology_g_confound():
    """Φ(G, domain) across the confound-adjustment ladder: biology stays the least severity-entangled
    domain after age/sex/edu/site + antipsychotic adjustment (A3/BMI omitted — degenerate)."""
    f = REPO / "reports" / "12_biology_g_confound.csv"
    if not f.exists():
        print("  -- biology_g_confound skipped: reports/12_biology_g_confound.csv missing")
        return
    df = pd.read_csv(f).set_index("domain")
    arms = ["A0_unadjusted", "A1_demo_site", "A2_antipsychotic"]     # A3 degenerate -> not plotted
    labels = ["A0\nunadjusted", "A1\n+ demo + site", "A2\n+ antipsychotic"]
    x = np.arange(len(arms))
    style = {"metabolic": (KR, "-o", 2.2), "inflammatory": (VIOLET, "-o", 2.2),
             "cognition": (MUTE, "--s", 1.4), "sleep": (MUTE, "--^", 1.4)}
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    ax.axhspan(0, 0.15, color=KR, alpha=0.06, zorder=0)
    ax.text(0.04, 0.143, "biology band ($\\leq 0.15$)", fontsize=7, color=KR, va="top")
    for dom, (col, ls, lw) in style.items():
        if dom not in df.index:
            continue
        y = [float(df.loc[dom, a]) for a in arms]
        ax.plot(x, y, ls, color=col, lw=lw, ms=6, zorder=4, label=PRETTY.get(dom, dom))
        ax.annotate(f"{y[-1]:+.2f}", (x[-1], y[-1]), xytext=(7, 0), textcoords="offset points",
                    fontsize=7.8, color=col, va="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_xlim(-0.25, len(arms) - 0.35)
    ax.set_ylabel(r"$\Phi(G,\ \mathrm{domain})$  —  entanglement with severity")
    ax.set_ylim(-0.02, 0.46)
    ax.set_title("Biology $\\perp G$ survives medication + site adjustment", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.text(0.5, -0.22, "A3 (BMI as covariate) omitted — degenerate ($\\widehat{R}\\,1.83$): BMI is itself a "
            "metabolic indicator, so it cannot be partialled out of its own factor.",
            transform=ax.transAxes, fontsize=6.8, color=MUTE, ha="center")
    fig.tight_layout()
    _save(fig, "biology_g_confound.png")


# ════════════════════════════════════════════════════════════════════════════════════════════
_M1_FIGS = [fig_biology_g, fig_phi, fig_empirical_atlas, fig_prior_posterior, fig_waic,
            fig_invariance, fig_ppc, fig_reliability, fig_soft_priors, fig_coverage,
            fig_biology_g_confound]
_MILESTONE_FIGS = [fig_m2_structure, fig_m2_archetypes, fig_m2_regions, fig_m2_confidence,
                   fig_m2_embedding, fig_m3_traitstate, fig_m3_invariance, fig_m3_spinecorner, fig_m4_value,
                   fig_m4_atlas, fig_m5_moderation, fig_synthesis]

if __name__ == "__main__":
    print(f"Generating manuscript figures -> {OUT.relative_to(REPO)}")
    failures = []
    for fn in _M1_FIGS + _MILESTONE_FIGS:
        try:
            fn()
        except Exception as e:
            print(f"  !! {fn.__name__} failed: {type(e).__name__}: {e}")
            failures.append(fn.__name__)
    if failures:
        print(f"FAILED: {', '.join(failures)}")
    print("done.")
