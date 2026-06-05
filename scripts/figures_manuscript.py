"""Manuscript figures (v2) — regenerates all main figures from results/hfa/ artifacts.

Six publication-quality figures (300 dpi PNG -> results/reports/figures/):
  F1  design & analytic pipeline (schematic)
  F2  the four trans-diagnostic dimensions (top loadings + Phi_2 + no p-factor)
  F3  HEADLINE: symptom<->biology orthogonality heatmap + p-factor dissolution
  F4  dimensional, not categorical (silhouette real-vs-null, unimodal axes, overlap scatter)
  F5  predictive validity vs DSM (incremental forest + relapse-AUC narrative)
  F6  longitudinal coherence (structural invariance + score test-retest)

Reads only committed artifacts (no re-fit except the Study-B block heatmap, which reuses the
masked estimator on the committed Stage-2 scores). Masked / no-imputation throughout.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from trans_diag.axes import AXIS_NAMES, AXIS_SHORT

warnings.simplefilter("ignore")

HFA = ROOT / "results" / "hfa"
FIG = ROOT / "results" / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- style
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.family": "DejaVu Sans",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
})

# axis palette (K=3: internalizing / cognition / cardiometabolic)
AX_COL = {"internalizing": "#2C6FB5", "cognition": "#2E8B7A", "cardiometabolic": "#C24A4A"}
AX_ORDER = ["internalizing", "cognition", "cardiometabolic"]
AX_TITLE = {"internalizing": "Dim 1 · Internalizing", "cognition": "Dim 2 · Cognition",
            "cardiometabolic": "Dim 3 · Cardiometabolic"}
AX_ABBR = {"internalizing": "Int", "cognition": "Cog", "cardiometabolic": "CMet"}
COH_COL = {"bp": "#3B6FA0", "sz": "#B5562B", "dr": "#4E9A5B"}
POS, NEG = "#C24A4A", "#3B6FA0"   # red = higher pole, blue = reversed pole (bar charts)


def panel_tag(ax, s, dx=-0.02, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="right")


# human-readable construct labels for the loading bars
LABEL = {
    "qidsr": "QIDS depression", "madrs": "MADRS depression", "staya": "STAI anxiety",
    "fast": "FAST disability", "eq5d": "EQ-5D index (rev.)", "eq": "EQ-5D VAS (rev.)",
    "prism": "PRISM stigma", "egf": "GAF function (rev.)", "psqi": "PSQI sleep",
    "cgi_severity": "CGI severity", "mars": "MARS adherence (rev.)", "wurs": "WURS childhood ADHD",
    "csm": "Circadian morningness (rev.)", "ctq": "Childhood trauma (CTQ)",
    "hooccur_arret_travail_actuel": "Current work stoppage", "ess": "Epworth sleepiness",
    "executive": "Executive (TMT-B)", "psychomotor_speed": "Psychomotor (TMT-A)",
    "processing_speed": "Processing speed (WAIS)", "perceptual_reason": "Perceptual reasoning",
    "working_memory": "Working memory", "edulevel": "Education (rev.)",
    "nboccur_hospitalisation_lt": "Lifetime hospitalisations", "hodur_hospitalisation_lt": "Hospital days",
    "pregnn_rporres": "Pregnancy history", "agedebut_hospitalisation": "Age at 1st hospitalisation",
    "agetrt": "Age at 1st treatment", "agedebutpremier_episode": "Age at 1st episode",
    "attempt_history": "Suicide-attempt history", "lipids_hdl": "HDL/triglycerides",
    "inflammation": "Inflammation (CRP/WBC)", "autonomic_hr": "Autonomic heart rate",
    "adiposity": "Adiposity (BMI/waist)", "bio_lym_lbstresc": "Lymphocytes",
    "blood_pressure": "Blood pressure", "bio_qt": "QT interval (rev.)", "hepatic": "Hepatic enzymes",
    "cholesterol": "Cholesterol (LDL)", "bio_plat": "Platelets", "red_cell": "Red-cell mass",
    "bio_rr": "RR interval (rev.)", "glycemia": "Glycaemia (gluc/HbA1c)", "suicidal_ideation": "Suicidal ideation",
}


def lab(c):
    return LABEL.get(c, c.replace("_", " "))


# ============================================================================= F1
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333333", fs=8.4, tc="black", lw=1.1):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.4",
                                    fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=3)

    def arrow(x0, y0, x1, y1, color="#555555", lw=1.4):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                     lw=lw, color=color, zorder=1))

    ax.text(50, 97, "FACE trans-diagnostic phenotyping — analytic pipeline (V0 anchor, masked / no imputation)",
            ha="center", fontsize=11, fontweight="bold")

    # cohorts
    box(4, 84, 27, 9, "FACE-BP  (bipolar)\nn = 6,252", "#DCE6F2")
    box(36.5, 84, 27, 9, "FACE-SZ  (schizophrenia)\nn = 2,209", "#F2E2D8")
    box(69, 84, 27, 9, "FACE-DR  (depression)\nn = 552", "#DCEEDF")
    ax.text(50, 81.4, "N = 9,013 patients · baseline V0 → 4-year V4", ha="center", fontsize=8.6, style="italic")
    for x in (17.5, 50, 82.5):
        arrow(x, 84, 50 if x == 50 else (40 if x < 50 else 60), 78.5)

    # processing stages
    box(8, 69.5, 84, 8.6,
        "DATA PROCESSING (3 stages)   214-variable harmonised dictionary · per-variable sanity bounds (out-of-range → NaN)\n"
        "(1) native clinical scale   →   (2) type-aware scaling to [−1, 1]   →   (3) V0 item matrix",
        "#F4F4F2", fs=8.0)
    arrow(50, 69.5, 50, 64.5)

    # hierarchical measurement model band
    ax.add_patch(FancyBboxPatch((4, 28.5), 92, 35, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc="#FBFBF8", ec="#9AA7B0", lw=1.3, ls="--", zorder=0))
    ax.text(50, 61.6, "HIERARCHICAL / BIFACTOR MEASUREMENT MODEL  (hybrid: clinical anchors, data-revised)",
            ha="center", fontsize=8.8, fontweight="bold", color="#3a4a55")

    box(7, 50, 39, 8.4, "Stage 0 — item set\n188 V0 items (incl. recovered labs/vitals)", "#EAF1F8", fs=8.0)
    box(54, 50, 39, 8.4, "Stage 1 — exploratory EFA\n42 nameable first-order factors", "#EAF1F8", fs=8.0)
    arrow(46, 54.2, 54, 54.2)
    arrow(26.5, 50, 26.5, 45.2); arrow(73.5, 50, 50, 45.2)

    box(20, 36.4, 60, 8.4,
        "Stage 2 — first-order constructs\n88 constructs · within-construct masked 1-factor posterior scores  →  Φ₁",
        "#E3EDF6", fs=8.0)
    arrow(50, 36.4, 50, 31.6)
    box(14, 22.8, 72, 8.8,
        "Stage 3 — second-order dimensions   factor Φ₁ → promax → Φ₂ · Schmid–Leiman ECV · split-half Tucker K\n"
        "4 trans-diagnostic axes  +  2 orthogonal standalone constructs (mania, suicidality) · ECV 0.36 → no p-factor",
        "#D7E6F2", fs=8.0)

    # the four axes chips
    cx = [13.5, 35, 56.5, 78]
    for x, a in zip(cx, AX_ORDER, strict=False):
        box(x, 13.2, 20.5, 6.6, AX_TITLE[a].split(" · ")[1], AX_COL[a] + "33", ec=AX_COL[a], fs=8.0, lw=1.4)
        arrow(x + 10, 22.8, x + 10.25, 19.8, color=AX_COL[a])

    # two analysis arms + validation
    box(6, 2.2, 41, 8.0,
        "DIMENSIONAL ARM\nno p-factor · symptoms ⊥ biology (Study B)", "#EAF3EC", ec="#4E9A5B", fs=8.0, lw=1.3)
    box(53, 2.2, 41, 8.0,
        "STRATIFICATION ARM\ncontinuum · no discrete subtypes beyond DSM", "#F3ECEA", ec="#B5562B", fs=8.0, lw=1.3)
    arrow(20, 13.2, 20, 10.2); arrow(78, 13.2, 78, 10.2)
    ax.text(50, 0.4, "Validation:  A cohort-confound refuted · B symptom⊥biology / p-factor artifact · "
                     "C longitudinal coherence · D predictive validity vs DSM",
            ha="center", fontsize=7.8, style="italic", color="#444")

    fig.savefig(FIG / "fig1_pipeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  F1 pipeline -> fig1_pipeline.png")


# ============================================================================= F2
def fig2_axes():
    L = pd.read_csv(HFA / "stage3_loadings.csv", index_col=0)
    phi2 = pd.read_csv(HFA / "stage3_phi2.csv", index_col=0).to_numpy()
    dims = [c for c in L.columns if c.startswith("dim")]
    K = len(dims)

    fig = plt.figure(figsize=(11, 9.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 0.52], hspace=0.5, wspace=2.6,
                          left=0.22, right=0.97, top=0.93, bottom=0.07)
    # K axis panels, two per row across the top rows; a lone trailing panel is centred.
    cells = []
    for i in range(K):
        row = i // 2
        if i == K - 1 and K % 2 == 1:
            cells.append(gs[row, 1:4])
        else:
            cells.append(gs[row, 0:3] if i % 2 == 0 else gs[row, 3:6])
    for i, (d, a) in enumerate(zip(dims, AX_ORDER, strict=False)):
        ax = fig.add_subplot(cells[i])
        s = L[d].reindex(L[d].abs().sort_values(ascending=False).index)
        s = s[s.abs() > 0.30].head(8)[::-1]
        cols = [POS if v > 0 else NEG for v in s.values]
        ax.barh(range(len(s)), s.values, color=cols, edgecolor="white", height=0.74)
        ax.set_yticks(range(len(s))); ax.set_yticklabels([lab(c) for c in s.index], fontsize=7.6)
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_xlim(-1.0, 1.0); ax.set_title(AX_TITLE[a], color=AX_COL[a], fontsize=9.6, pad=4)
        ax.tick_params(axis="x", labelsize=7.2)
        ax.set_xlabel("second-order loading", fontsize=8)

    nwords = {2: "two", 3: "three", 4: "four", 5: "five"}.get(K, str(K))
    fig.suptitle(f"The {nwords} trans-diagnostic dimensions — defining constructs (|loading| > 0.30)",
                 fontsize=11.5, fontweight="bold", y=0.975)

    # Phi2 strip (inter-axis correlations) — own row, no label collisions
    axp = fig.add_subplot(gs[2, 1:3])
    im = axp.imshow(phi2, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    short = [AX_ABBR.get(a, a[:4]) for a in AX_ORDER[:K]]
    axp.set_xticks(range(K)); axp.set_yticks(range(K))
    axp.set_xticklabels(short, fontsize=7.4); axp.set_yticklabels(short, fontsize=7.4)
    for r in range(K):
        for c in range(K):
            axp.text(c, r, f"{phi2[r, c]:.2f}", ha="center", va="center", fontsize=7.0,
                     color="white" if abs(phi2[r, c]) > 0.33 else "black")
    axp.set_title("Φ₂  inter-axis correlations", fontsize=8.4, pad=4)
    axp.tick_params(length=0)
    mean_phi2 = float(np.abs(phi2[np.triu_indices(K, 1)]).mean())
    try:
        ecv = json.load(open(HFA / "stage3_meta.json"))["ecv"]      # Stage-3 Schmid–Leiman ECV
    except Exception:
        ecv = float("nan")
    fig.text(0.62, 0.135,
             f"Weakly correlated axes (mean |Φ₂| = {mean_phi2:.2f}).\n"
             f"Schmid–Leiman ECV = {ecv:.2f}  →  no dominant\ngeneral (p-)factor: the structure is genuinely\n"
             "multidimensional.   Mania, suicidality & substance\nuse are valid but orthogonal — not axes.",
             ha="left", va="center", fontsize=8.2, color="#33414b")
    fig.text(0.5, 0.018, "red = higher score is more pathological · blue = reverse-keyed construct",
             ha="center", fontsize=7.8, style="italic", color="#555")

    fig.savefig(FIG / "fig2_axes.png", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  F2 axes -> fig2_axes.png")


# ============================================================================= F3
def fig3_orthogonality():
    from collections import Counter

    from trans_diag.masked_fa import masked_correlation
    from trans_diag.variable import load_variables

    S = pd.read_pickle(HFA / "stage2_scores.pkl").set_index(["cohort", "patient_id"])
    fit = pd.read_csv(HFA / "stage2_construct_fit.csv").set_index("construct")
    item_sec = {v.canonical_name: v.section for v in load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))}
    SEC2BLOCK = {"AUTO-QUESTIONNAIRES": "symptom", "HETERO-QUESTIONNAIRES": "symptom",
                 "SUICIDE": "symptom", "EVALUATION MEDICALE": "symptom",
                 "BILAN BIOLOGIQUE": "biology", "CONSTANTES ET ECG": "biology",
                 "NEUROPSYCHOLOGIE": "cognition"}

    def block_of(con):
        secs = [item_sec.get(it) for it in str(fit.loc[con, "items"]).split(",") if it in item_sec]
        blocks = [SEC2BLOCK.get(s, "other") for s in secs]
        return Counter(blocks).most_common(1)[0][0] if blocks else "other"

    coh = S.index.get_level_values("cohort")
    Sbd = S[coh.isin(["bp", "dr"])]
    Z = (Sbd - Sbd.mean()) / Sbd.std()
    keep = [c for c in Z.columns if Z[c].notna().mean() >= 0.30 and Z[c].std() > 0]
    blk = {c: block_of(c) for c in keep}
    order_blocks = ["symptom", "biology", "cognition", "other"]
    ordered = [c for b in order_blocks for c in keep if blk[c] == b]
    R = pd.DataFrame(masked_correlation(Z[ordered], 100), index=ordered, columns=ordered)

    sb = json.load(open(HFA / "studyB_orthogonality.json"))["BP+DR"]
    o = sb["orthogonality"]; sets = sb["sets"]

    fig = plt.figure(figsize=(12.2, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.34,
                          left=0.07, right=0.96, top=0.86, bottom=0.14)

    # --- panel A: block-ordered correlation heatmap
    axA = fig.add_subplot(gs[0, 0])
    im = axA.imshow(R.to_numpy(), cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    bounds, pos, c0 = {}, 0, 0
    for b in order_blocks:
        n = sum(blk[c] == b for c in ordered)
        bounds[b] = (c0, c0 + n); c0 += n
    for b, (a0, a1) in bounds.items():
        for v in (a0, a1):
            axA.axhline(v - 0.5, color="black", lw=1.0); axA.axvline(v - 0.5, color="black", lw=1.0)
        mid = (a0 + a1) / 2 - 0.5
        axA.text(mid, -1.6, b.upper(), ha="center", va="bottom", fontsize=8.2, fontweight="bold")
        axA.text(-1.6, mid, b.upper(), ha="right", va="center", fontsize=8.2, fontweight="bold", rotation=90)
    axA.set_xticks([]); axA.set_yticks([])
    axA.set_title("Construct-correlation matrix, block-ordered (BP+DR)", fontsize=9.6, pad=18)
    cb = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.03); cb.set_label("masked r", fontsize=8)
    # annotate between-block means
    txt = (f"mean |r|   within symptom {o['symptom_symptom']:.2f} · within cognition {o['cognition_cognition']:.2f}\n"
           f"between symptom↔biology {o['biology_symptom']:.2f} · symptom↔cognition {o['cognition_symptom']:.2f} · "
           f"biology↔cognition {o['biology_cognition']:.2f}")
    axA.text(0.5, -0.14, txt, transform=axA.transAxes, ha="center", va="top", fontsize=7.8)
    panel_tag(axA, "a", dx=-0.04, dy=1.20)

    # --- panel B: p-factor dissolution
    axB = fig.add_subplot(gs[0, 1])
    order = ["symptom_only", "symptom+cognition", "symptom+biology", "full(all blocks)"]
    xlab = ["symptom\nonly", "+ cognition", "+ biology", "full\nintegrated"]
    pooled = json.load(open(HFA / "studyB_orthogonality.json"))["pooled"]["sets"]
    yb = [sets[k]["first_factor_share"] for k in order]
    yp = [pooled[k]["first_factor_share"] for k in order]
    x = np.arange(4)
    axB.plot(x, yb, "-o", color="#2C6FB5", lw=2.2, ms=8, label="BP+DR (clean)", zorder=3)
    axB.plot(x, yp, "--s", color="#9AA7B0", lw=1.6, ms=6, label="pooled (sensitivity)", zorder=2)
    for xi, v in zip(x, yb, strict=False):
        axB.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 11),
                     ha="center", fontsize=8.4, fontweight="bold", color="#2C6FB5")
    axB.fill_between(x, yb, min(yb) - 0.03, color="#2C6FB5", alpha=0.06)
    axB.set_xticks(x); axB.set_xticklabels(xlab, fontsize=8.2)
    axB.set_ylabel("first-factor share  λ₁ / Σλ\n(general-factor strength)", fontsize=8.6)
    axB.set_ylim(0.04, 0.38)
    axB.set_title("The p-factor is a symptom-only artifact", fontsize=9.6, pad=8)
    axB.legend(loc="upper right", fontsize=7.8)
    axB.annotate("admitting structured\nbiology + cognition\ndissolves the general factor",
                 xy=(3, yb[3]), xytext=(1.75, 0.27), fontsize=7.8, color="#444",
                 arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
    panel_tag(axB, "b", dx=-0.10, dy=1.13)

    fig.suptitle("Symptoms are orthogonal to biology — an integrated model has no dominant general factor",
                 fontsize=11.6, fontweight="bold", y=0.97)
    fig.savefig(FIG / "fig3_orthogonality.png", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  F3 orthogonality -> fig3_orthogonality.png")


# ============================================================================= F4
def fig4_continuum():
    ph = json.load(open(HFA / "phase5_structure.json"))["A"]
    F = pd.read_pickle(HFA / "stage3_scores.pkl")
    dims = [c for c in F.columns if c.startswith("dim")]

    fig = plt.figure(figsize=(13.2, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.32,
                          left=0.06, right=0.97, top=0.84, bottom=0.16)

    # panel A: silhouette real vs null
    axA = fig.add_subplot(gs[0, 0])
    gg = ph["gap_vs_gaussian"]
    ks = [d["k"] for d in gg]
    axA.plot(ks, [d["sil_real"] for d in gg], "-o", color="#2C6FB5", lw=2, ms=6, label="real data")
    axA.plot(ks, [d["sil_null"] for d in gg], "--o", color="#B0B0B0", lw=1.6, ms=5, label="Gaussian null")
    axA.fill_between(ks, [d["sil_real"] for d in gg], [d["sil_null"] for d in gg],
                     color="#2C6FB5", alpha=0.08)
    axA.set_xlabel("number of clusters k"); axA.set_ylabel("silhouette")
    axA.set_title("No cluster structure beyond a Gaussian blob", fontsize=9.4)
    axA.legend(loc="upper right")
    axA.text(0.5, 0.06, "HDBSCAN: 0 dense clusters (100% noise)\nk-means vs DSM: ARI ≈ 0.03",
             transform=axA.transAxes, ha="center", fontsize=7.8,
             bbox=dict(boxstyle="round", fc="#FBF3E8", ec="#D98E2B", lw=1))
    panel_tag(axA, "a", dx=-0.16, dy=1.14)

    # panel B: unimodal axis densities + Sarle
    axB = fig.add_subplot(gs[0, 1])

    def sarle(x):
        x = x[np.isfinite(x)]
        n = len(x); m = x.mean(); s = x.std()
        g1 = (((x - m) / s) ** 3).mean()
        g2 = (((x - m) / s) ** 4).mean() - 3.0
        return (g1 ** 2 + 1) / (g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))

    for d, a in zip(dims, AX_ORDER, strict=False):
        x = F[d].to_numpy(); x = x[np.isfinite(x)]
        x = (x - x.mean()) / x.std()
        kde_x = np.linspace(-4, 4, 200)
        h, edges = np.histogram(x, bins=60, range=(-4, 4), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        axB.plot(centers, np.convolve(h, np.ones(5) / 5, mode="same"),
                 color=AX_COL[a], lw=1.8, label=f"{AX_TITLE[a].split('· ')[1]}  (b={sarle(F[d].to_numpy()):.2f})")
    axB.set_xlabel("axis score (z)"); axB.set_ylabel("density")
    axB.set_title("Every axis is unimodal (Sarle b < 0.555)", fontsize=9.4)
    axB.legend(loc="upper left", fontsize=6.9)
    axB.set_xlim(-4, 4)
    panel_tag(axB, "b", dx=-0.16, dy=1.14)

    # panel C: overlap scatter
    axC = fig.add_subplot(gs[0, 2])
    sub = F.dropna(subset=["dim1", "dim2"]).sample(n=min(3200, len(F)), random_state=0)
    for c in ["bp", "sz", "dr"]:
        d = sub[sub["cohort"] == c]
        axC.scatter((d["dim1"] - F["dim1"].mean()) / F["dim1"].std(),
                    (d["dim2"] - F["dim2"].mean()) / F["dim2"].std(),
                    s=7, alpha=0.32, color=COH_COL[c], label=f"{c.upper()}", edgecolors="none")
    axC.set_xlabel("Internalizing (z)"); axC.set_ylabel("Cognition (z)")
    axC.set_title("Cohorts overlap in axis space (continuum)", fontsize=9.4)
    leg = axC.legend(loc="upper right", markerscale=2.2, fontsize=7.6)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    panel_tag(axC, "c", dx=-0.15, dy=1.14)

    fig.suptitle("Trans-diagnostic variation is dimensional, not categorical",
                 fontsize=11.6, fontweight="bold", y=0.98)
    fig.savefig(FIG / "fig4_continuum.png", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  F4 continuum -> fig4_continuum.png")


# ============================================================================= F5
def fig5_predictive():
    d = json.load(open(HFA / "studyD_predictive.json"))
    d2 = json.load(open(HFA / "studyD2_survival.json"))
    d4 = json.load(open(HFA / "studyD4_trajectory.json"))

    fig = plt.figure(figsize=(13.2, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.28,
                          left=0.31, right=0.96, top=0.85, bottom=0.13)

    # panel A: forest of incremental deltas
    axA = fig.add_subplot(gs[0, 0])
    rows = [
        ("GAF functioning — dims add over DSM (ΔR²)", *d["GAF@V2"]["delta_axes_add_dsm"], True),
        ("GAF functioning — cross-domain, non-circular (ΔR²)", *d["GAF@V2"]["delta_crossdomain_vs_base"], True),
        ("FAST disability — dims beat DSM (ΔR²)", *d["FAST@V2"]["delta_axes_vs_dsm"], True),
        ("FAST disability — cross-domain (ΔR²)", *d["FAST@V2"]["delta_crossdomain_vs_base"], True),
        ("Relapse, change-based — dims add (ΔAUC)", *d["relapse-by-V2"]["delta_axes_add_dsm"], False),
        ("Relapse, de-confounded — dims add, logistic (ΔAUC)", *d2["logistic"]["d_axes_add_dsm"], False),
        ("Relapse, de-confounded — dims add, gboost (ΔAUC)", *d2["gboost"]["d_axes_add_dsm"], False),
        ("Relapse, early-course — trajectory vs DSM (ΔAUC)", *d4["gboost"]["d_traj_vs_dsm"], False),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (name, est, ci, isR2) in zip(y, rows, strict=False):
        sig = ci[0] > 0
        col = "#2E8B7A" if sig else "#B0B0B0"
        axA.plot([ci[0], ci[1]], [yi, yi], color=col, lw=2.6, zorder=2)
        axA.plot(est, yi, "o", color=col, ms=8, zorder=3)
        axA.text(-0.02, yi, name, ha="right", va="center", fontsize=7.6,
                 transform=axA.get_yaxis_transform(), clip_on=False)
    axA.axvline(0, color="#333", lw=1.0, ls="--")
    axA.set_yticks([]); axA.set_ylim(-0.7, len(rows) - 0.3)
    axA.set_xlabel("incremental performance over DSM   (ΔR² for functioning · ΔAUC for relapse)")
    axA.set_xlim(-0.025, 0.105)
    axA.set_title("Modest but real increment over DSM diagnosis", fontsize=9.6)
    axA.text(0.98, 0.02, "green = 95% CI excludes 0", transform=axA.transAxes, ha="right",
             fontsize=7.4, color="#2E8B7A", style="italic")
    axA.text(-0.02, 1.07, "a", transform=axA.transAxes, fontsize=13, fontweight="bold", va="top", ha="right")

    # panel B: relapse-AUC narrative
    axB = fig.add_subplot(gs[0, 1])
    bars = [("change-based\nbaseline\n(confounded)", 0.765, "#C24A4A"),
            ("remission-based\nbaseline\n(de-confounded)", 0.578, "#8FB0C8"),
            ("+ DSM", d2["logistic"]["AUC"]["M1_+DSM"], "#6E94B4"),
            ("+ dimensions", d2["logistic"]["AUC"]["M2_+axes"], "#3B6FA0"),
            ("early-course\ntrajectory\n(V0+V1)", d4["logistic"]["AUC"]["+traj"], "#2E8B7A")]
    xb = np.arange(len(bars))
    axB.bar(xb, [b[1] for b in bars], color=[b[2] for b in bars], width=0.66, edgecolor="white")
    for xi, b in zip(xb, bars, strict=False):
        axB.text(xi, b[1] + 0.008, f"{b[1]:.2f}", ha="center", fontsize=8.4, fontweight="bold")
    axB.axhline(0.5, color="#999", lw=0.8, ls=":")
    axB.axhline(0.70, color="#2E8B7A", lw=1.0, ls="--")
    axB.text(-0.42, 0.704, "AUC 0.70", color="#2E8B7A", fontsize=7.2, va="bottom", ha="left")
    axB.set_xticks(xb); axB.set_xticklabels([b[0] for b in bars], fontsize=7.2)
    axB.set_ylabel("relapse prediction AUC"); axB.set_ylim(0.45, 0.83)
    axB.set_title("Relapse: the regression-to-the-mean confound, removed", fontsize=9.4)
    axB.annotate("", xy=(1, 0.60), xytext=(0, 0.747),
                 arrowprops=dict(arrowstyle="->", color="#C24A4A", lw=1.6))
    axB.text(0.62, 0.662, "−0.19\nregression\nto the mean", color="#C24A4A", fontsize=7.0, ha="left", va="top")
    panel_tag(axB, "b", dx=-0.12, dy=1.12)

    fig.suptitle("Predictive validity vs DSM — functioning robustly, relapse modestly (de-confounded)",
                 fontsize=11.4, fontweight="bold", y=0.96)
    fig.savefig(FIG / "fig5_predictive.png", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  F5 predictive -> fig5_predictive.png")


# ============================================================================= F6
def fig6_longitudinal():
    c = json.load(open(HFA / "studyC_longitudinal.json"))
    inv, stab = c["invariance"], c["stability"]
    order = list(AXIS_NAMES)
    short = [AXIS_SHORT[a] for a in order]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.22, wspace=0.26)

    x = np.arange(len(order)); w = 0.36
    # panel A: structural invariance
    v1 = [inv["V1"][a] for a in order]; v2 = [inv["V2"][a] for a in order]
    axA.bar(x - w / 2, v1, w, label="V1 vs V0", color=[AX_COL[a] for a in order], alpha=0.55, edgecolor="white")
    axA.bar(x + w / 2, v2, w, label="V2 vs V0", color=[AX_COL[a] for a in order], edgecolor="white")
    axA.axhline(0.85, color="#666", ls="--", lw=1.0)
    axA.text(len(order) - 1.6, 0.865, "0.85 threshold", fontsize=7, color="#666")
    for xi, (a, b) in enumerate(zip(v1, v2, strict=False)):
        axA.text(xi - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=7)
        axA.text(xi + w / 2, b + 0.015, f"{b:.2f}", ha="center", fontsize=7)
    axA.set_xticks(x); axA.set_xticklabels(short, fontsize=8, rotation=12)
    axA.set_ylabel("Tucker congruence vs V0"); axA.set_ylim(0, 1.08)
    axA.set_title("Structural invariance — the axes persist at follow-up", fontsize=9.4)
    axA.legend(loc="lower left", fontsize=7.6)
    axA.text(1, 0.06, "cognition 0 @V1: WAIS battery is\nbaseline-anchored (≈5% re-measured)",
             fontsize=6.8, color="#555", ha="center")
    panel_tag(axA, "a", dx=-0.10, dy=1.13)

    # panel B: score test-retest
    r1 = [stab[a]["rho_V0V1"] for a in order]; r2 = [stab[a]["rho_V0V2"] for a in order]
    axB.bar(x - w / 2, r1, w, label="V0↔V1", color=[AX_COL[a] for a in order], alpha=0.55, edgecolor="white")
    axB.bar(x + w / 2, r2, w, label="V0↔V2", color=[AX_COL[a] for a in order], edgecolor="white")
    for xi, (a, b) in enumerate(zip(r1, r2, strict=False)):
        axB.text(xi - w / 2, a + 0.012, f"{a:.2f}", ha="center", fontsize=7)
        axB.text(xi + w / 2, b + 0.012, f"{b:.2f}", ha="center", fontsize=7)
    axB.set_xticks(x); axB.set_xticklabels(short, fontsize=8, rotation=12)
    axB.set_ylabel("rank-order test–retest (Spearman ρ)"); axB.set_ylim(0, 0.8)
    axB.set_title("Score stability — trait (biology) vs state (mood)", fontsize=9.4)
    axB.legend(loc="upper right", fontsize=7.6)
    if "cardiometabolic" in order:
        ci = order.index("cardiometabolic")
        axB.annotate("cardiometabolic =\nmost trait-stable", xy=(ci, r1[ci]), xytext=(max(ci - 1.2, 0.2), 0.72),
                     fontsize=7, color="#C24A4A", arrowprops=dict(arrowstyle="->", color="#C24A4A", lw=1))
    panel_tag(axB, "b", dx=-0.10, dy=1.13)

    fig.suptitle("Longitudinal coherence (V0 → V1 → V2): structure persists; biology is the most measurement-robust axis",
                 fontsize=11, fontweight="bold", y=0.97)
    fig.savefig(FIG / "fig6_longitudinal.png", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  F6 longitudinal -> fig6_longitudinal.png")


# ============================================================================= S1
def figS1_bootstrap():
    b = json.load(open(HFA / "bootstrap_dimensionality.json"))
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13, 3.9))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.24, wspace=0.42)

    # (a) eigengaps with 95% CI — first gaps large, gap4 ~ 0 (degenerate)
    g = np.array(b["gap_mean"][:6]); lo = np.array(b["gap_ci"][0][:6]); hi = np.array(b["gap_ci"][1][:6])
    x = np.arange(len(g)); cols = ["#2C6FB5" if l > 0.15 else "#C24A4A" for l in lo]
    axA.bar(x, g, color=cols, edgecolor="white")
    axA.errorbar(x, g, yerr=[g - lo, hi - g], fmt="none", ecolor="#333", capsize=3, lw=1)
    axA.axhline(0, color="#888", lw=0.8)
    axA.set_xticks(x); axA.set_xticklabels([f"λ{i+1}–λ{i+2}" for i in range(len(g))], fontsize=7.4)
    axA.set_ylabel("eigenvalue gap (Φ₁)"); axA.set_title("Eigengaps (95% CI): 3 separated, then degenerate", fontsize=9)
    axA.annotate("gap₄ ≈ 0", xy=(3, hi[3]), xytext=(3.4, 0.9), fontsize=7.5, color="#C24A4A",
                 arrowprops=dict(arrowstyle="->", color="#C24A4A", lw=1))
    panel_tag(axA, "a", dx=-0.12, dy=1.17)

    # (b) distribution of the locked K — noisy
    K = b["K_dist"]; n = b["n_boot_K"]; ks = sorted(int(k) for k in K)
    axB.bar([str(k) for k in ks], [100 * K[str(k)] / n for k in ks], color="#2E8B7A", edgecolor="white")
    axB.set_xlabel("locked K (split-half rule)"); axB.set_ylabel("% of bootstraps"); axB.set_ylim(0, 100)
    axB.set_title("The count K is a noisy estimator", fontsize=9)
    panel_tag(axB, "b", dx=-0.18, dy=1.17)

    # (c) per-factor stability — high regardless
    names = {"qidsr": "internalizing", "cvlt_total_recall": "cognition", "lipids_hdl": "cardiometabolic",
             "agedebut_hospitalisation": "illness-course", "substance_use_disorder": "substance-use",
             "wurs": "childhood-adv."}
    st = b["stability_pct"]; labs = np.array([names.get(k, k) for k in b["ref_factors"]])
    vals = np.array([st[k] for k in b["ref_factors"]]); order = np.argsort(vals)
    axC.barh(labs[order], vals[order], color="#3B6FA0", edgecolor="white")
    axC.set_xlim(0, 108); axC.set_xlabel("% of resamples factor recovers (≥0.85)")
    axC.set_title("…but every factor is stable", fontsize=9)
    for i, v in enumerate(vals[order]):
        axC.text(v - 7, i, f"{v:.0f}", va="center", fontsize=7, color="white")
    panel_tag(axC, "c", dx=-0.30, dy=1.17)

    fig.suptitle("Figure S1 · Bootstrap robustness of dimensionality (50 cohort-stratified resamples): "
                 "the factors are stable; the count K is not", fontsize=10, fontweight="bold", y=0.99)
    fig.savefig(FIG / "figS1_bootstrap.png", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  S1 bootstrap -> figS1_bootstrap.png")


def main():
    print(f"Generating manuscript figures -> {FIG}")
    for fn in (fig1_pipeline, fig2_axes, fig3_orthogonality, fig4_continuum,
               fig5_predictive, fig6_longitudinal, figS1_bootstrap):
        try:
            fn()
        except Exception as e:  # noqa
            import traceback
            print(f"  !! {fn.__name__} FAILED: {e}")
            traceback.print_exc()
    print("done.")


if __name__ == "__main__":
    main()
