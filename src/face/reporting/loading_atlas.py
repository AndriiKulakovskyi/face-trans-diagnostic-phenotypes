"""Shared posterior-loading dot-atlas (article Figure 2 + technical-report empirical atlas).

The dot-atlas draws an indicator x factor matrix where dot size and colour encode the absolute
posterior median loading, an outline marks loadings whose 95% credible interval excludes zero
(a heavier ring marks the home-factor anchor), a left colour strip + alternating bands group rows
by home factor, and the general-factor (G) column carries shaded depression/anxiety ``windows``.

All house style (colormap, per-block colours, factor labels and short tags, fonts) is passed in, so
the article (mako / Okabe-Ito) and the technical report (sequential blue) share one implementation
rather than maintaining two copies.  Input is the CI-aware copula loadings table written by
:func:`face.measurement.synthetic.export_loadings_summary`
(``reports/copula_s5_9dim_loadings.csv``): one row per meaningful (item, factor) cell with columns
``item, factor, home, kind, loading, abs_loading, ci_low, ci_high, excludes_zero``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# Technical indicator code -> clinical display label.  Codes with no entry fall back to the raw
# code.  Suicidality (isf0*) and CTQ item-level labels are best-guess pending the instrument
# dictionary (no human-readable dictionary exists in the repo).
DISP = {
    # G: functioning / global severity anchors + depression/anxiety windows
    "cgi01": "CGI severity", "egf": "Global functioning (EGF)", "eq5d": "EQ-5D index",
    "eq5d0206": "EQ-5D VAS", "fast": "FAST total", "fast25": "FAST: autonomy",
    "fast26": "FAST: occupational", "fast27": "FAST: cognition", "fast28": "FAST: finances",
    "fast30": "FAST: leisure", "lvsbjind": "Subjective wellbeing",
    "madrs": "MADRS total", "qidsr120": "QIDS total", "staya": "STAI-state",
    # cognition
    "cvlt_total_recall": "CVLT total recall", "cvlt_short_delay_free_recall": "CVLT short-delay recall",
    "cvlt_long_delay_free_recall": "CVLT long-delay recall", "tmt_a_time_sec": "Trail Making A",
    "tmt_b_time_sec": "Trail Making B", "verbal_fluency_phonemic": "Verbal fluency (phonemic)",
    "verbal_fluency_semantic": "Verbal fluency (semantic)", "wais_code_std": "WAIS coding",
    "wais_digitspan_std": "WAIS digit span", "wais_ivt_index": "WAIS processing speed",
    "wais_similitudes_std": "WAIS similarities",
    # metabolic
    "bmi": "BMI", "hba1c": "HbA1c", "trig": "Triglycerides", "hdl": "HDL cholesterol",
    "ldl": "LDL cholesterol", "chol": "Total cholesterol", "cholhdl_lbstresc": "Chol/HDL ratio",
    "gluc": "Fasting glucose", "weight": "Weight", "wstcir": "Waist circumference", "urate": "Urate",
    "vitd": "Vitamin D", "sysbpsupine": "Systolic BP (supine)", "sysbpstanding": "Systolic BP (standing)",
    "diabpsupine": "Diastolic BP (supine)", "diabpstanding": "Diastolic BP (standing)",
    "hrsupine": "Heart rate (supine)", "hrstanding": "Heart rate (standing)", "eghrmn": "Heart rate (ECG)",
    "alp_lbstresc": "ALP", "alt_lbstresc": "ALT", "ast_lbstresc": "AST", "ggt_lbstresc": "GGT",
    "bili_lbstresc": "Bilirubin", "creat_lbstresc": "Creatinine", "urea_lbstresc": "Urea",
    "tsh_lbstresc": "TSH", "t3fr_lbstresc": "Free T3", "t4fr_lbstresc": "Free T4",
    # inflammatory
    "crp": "CRP", "neut": "Neutrophils", "lym_lbstresc": "Lymphocytes", "mono_lbstresc": "Monocytes",
    "eos": "Eosinophils", "baso_lbstresc": "Basophils", "wbc": "White blood cells", "plat": "Platelets",
    # sleep
    "psqi": "PSQI total", "psqi11": "PSQI: latency", "psqi12": "PSQI: duration",
    "psqi13": "PSQI: efficiency", "psqi14": "PSQI: disturbance", "psqi15": "PSQI: medication",
    "psqi17": "PSQI: daytime dysfunction", "ess0109": "Epworth sleepiness", "csm": "Chronotype (CSM)",
    # suicidality (best-guess item labels)
    "isf01": "Suicidal ideation", "isf02": "Active ideation", "isf03": "Ideation w/ method",
    "isf04": "Ideation w/ intent", "isf05": "Ideation w/ plan", "isf07": "Suicidal behaviour",
    "isf08": "Suicide attempt", "isf08a": "Attempts (count)", "isf09": "Aborted attempt",
    "isf09a": "Interrupted attempt (count)",
    # developmental / early-life
    "agepere": "Paternal age", "apgr0106_1min": "APGAR 1-min", "brthcirc": "Birth head circumference",
    "brthht": "Birth length", "gstabrth": "Gestational age", "prembrth": "Premature birth",
    "naisstyp": "Delivery type", "honeonat": "Neonatal hospitalisation",
    "traumacra_mhoccur": "Cranial trauma (history)", "autneuro_mhoccur": "Neurodevelopmental disorder",
    "epilepsie_mhoccur": "Epilepsy (history)", "mere_structure": "Maternal family structure",
    "pere_structure": "Paternal family structure", "wurs": "WURS (childhood ADHD)",
    "ctq29": "CTQ item 29", "ctq31": "CTQ item 31", "ctq33": "CTQ item 33", "ctq35": "CTQ item 35",
    "ctq37": "CTQ item 37", "ctq39": "CTQ item 39", "ctq40": "CTQ item 40", "ctq41": "CTQ item 41",
    # mania
    "altman": "Altman self-rating", "ymrs": "YMRS total",
    # substance
    "fagers": "Fagerstrom (FTND)", "sudose_cigarettes_lt": "Cigarettes (lifetime)",
    "suoccur_alcool": "Alcohol use", "suoccur_cannabis": "Cannabis use",
}

# Always-keep indicators so per-block capping never drops the G anchors or the dep/anx windows.
DEFAULT_PIN = {"overall_severity": {"fast", "egf", "cgi01", "madrs", "qidsr120", "staya"}}

DEFAULT_SUBTITLE = ("columns: G (burden backbone)  +  specific axes D1–D7    ·    "
                    "shaded = depression/anxiety windows on G")


def load_loadings(path) -> pd.DataFrame:
    """Read the CI-aware copula loadings table; coerce window items (no home) into the G block."""
    L = pd.read_csv(path)
    L["home"] = L["home"].fillna("overall_severity")
    if "abs_loading" not in L.columns:
        L["abs_loading"] = L["loading"].abs()
    if "excludes_zero" not in L.columns:
        L["excludes_zero"] = L["kind"].isin(["primary", "g_anchor"])
    return L


def atlas_size(a, s_min: float = 12.0, s_max: float = 320.0) -> float:
    """Marker AREA (pt^2) linear in |loading| (radius ~ sqrt|loading|), clipped at |loading|=1."""
    return s_min + (s_max - s_min) * float(min(abs(a), 1.0))


def atlas_rows(L, axes, max_per_block, *, pin=None, g_key: str = "overall_severity"):
    """Ordered (item, home) rows: blocks in ``axes`` order, items by |home loading| desc, capped.

    ``max_per_block=None`` keeps every indicator.  Window items (dep/anx) are placed last within the
    G block so the shaded G-window rectangle is contiguous.
    """
    pin = DEFAULT_PIN if pin is None else pin
    rows = []
    win = set(L[L["kind"] == "window"]["item"])
    for h in axes:
        rep = L[(L["home"] == h) & (L["factor"] == h)].copy()
        rep["a"] = rep["loading"].abs()
        items = list(rep.sort_values("a", ascending=False)["item"])
        if max_per_block:
            keep = set(items[:max_per_block]) | (pin.get(h, set()) & set(items))
            items = [it for it in items if it in keep]
        if h == g_key:
            items = [it for it in items if it not in win] + [it for it in items if it in win]
        rows += [(it, h) for it in items]
    return rows


def draw_dot_atlas(ax, L, axes, rows, *, cmap, block_colors, axlab, axtag,
                   window_color="#F0E442", label_fs=5.8, anchor_edge="#111111",
                   cross_edge="#555555", g_key="overall_severity", subtitle=DEFAULT_SUBTITLE,
                   disp=None):
    """Draw the indicator x factor dot-atlas on ``ax``; return the scatter handle (for the colorbar).

    Style is fully caller-supplied: ``cmap`` (sequential), ``block_colors`` (home factor -> colour),
    ``axlab`` (factor -> long label), ``axtag`` (factor -> short tag e.g. G / D1..D7).
    """
    disp = DISP if disp is None else disp
    key = {(r.item, r.factor): r for r in L.itertuples()}
    n = len(rows)
    row_of = {(it, h): i for i, (it, h) in enumerate(rows)}
    xs, ys, sz, cl, ec, lw = [], [], [], [], [], []
    for i, (it, h) in enumerate(rows):
        for j, ax_ in enumerate(axes):
            rec = key.get((it, ax_))
            if rec is None or abs(rec.loading) < 0.03:
                continue
            a = abs(rec.loading)
            xs.append(j); ys.append(i); sz.append(atlas_size(a)); cl.append(min(a, 1.0))
            if rec.kind in ("primary", "g_anchor"):
                ec.append(anchor_edge); lw.append(1.4)
            elif bool(rec.excludes_zero):
                ec.append(cross_edge); lw.append(0.8)
            else:
                ec.append("none"); lw.append(0.0)
    # contiguous home-factor blocks
    spans, y0, cur = [], 0, rows[0][1]
    for i, (it, h) in enumerate(rows):
        if h != cur:
            spans.append((cur, y0 - 0.5, i - 0.5)); y0, cur = i, h
    spans.append((cur, y0 - 0.5, n - 0.5))
    for k, (h, lo, hi) in enumerate(spans):
        if k % 2 == 0:
            ax.axhspan(lo, hi, color="#000000", alpha=0.035, zorder=0)
        ax.add_patch(Rectangle((-1.55, lo), 0.30, hi - lo, color=block_colors[h], clip_on=False, zorder=4))
        ax.text(-1.75, (lo + hi) / 2, axlab[h], rotation=90, ha="right", va="center",
                fontsize=7.2, fontweight="bold", color=block_colors[h])
    # G window shading (depression/anxiety windows behind the G column)
    gcol = list(axes).index(g_key)
    for it in set(L[L["kind"] == "window"]["item"]):
        if (it, g_key) in row_of:
            yr = row_of[(it, g_key)]
            ax.add_patch(Rectangle((gcol - 0.46, yr - 0.46), 0.92, 0.92, facecolor=window_color,
                                   alpha=0.30, edgecolor="none", zorder=1))
    sc = ax.scatter(xs, ys, s=sz, c=cl, cmap=cmap, vmin=0.0, vmax=1.0,
                    edgecolors=ec, linewidths=lw, zorder=3)
    for i, (it, h) in enumerate(rows):
        ax.text(len(axes) - 0.35, i, disp.get(it, it), ha="left", va="center",
                fontsize=label_fs, color="#333333")
    ax.axvline(gcol + 0.5, color="#bbbbbb", lw=0.8, zorder=2)
    ax.set_xticks(range(len(axes)))
    ax.set_xticklabels([axlab[a] for a in axes], rotation=32, ha="right", fontsize=6.8)
    for j, a in enumerate(axes):
        ax.text(j, -1.0, axtag[a], ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#222222")
    ax.set_ylim(n - 0.5, -2.4); ax.set_xlim(-3.2, len(axes) + 4.2)
    ax.set_yticks([]); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if subtitle:
        ax.text((len(axes) - 1) / 2.0, -2.15, subtitle, ha="center", va="bottom",
                fontsize=6.4, color="#666666", style="italic")
    return sc


def atlas_legends(fig, ax, sc, *, window_color="#F0E442", face="#4878a8"):
    """Colorbar (below) + dot-size and marker legends placed in the empty upper-right of the atlas."""
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.018, pad=0.045, aspect=46)
    cb.set_label("|posterior median loading|", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    size_h = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=face,
                     markeredgecolor="none", markersize=np.sqrt(atlas_size(v)),
                     label=f"|λ| = {v:g}") for v in (0.2, 0.4, 0.7)]
    leg1 = ax.legend(handles=size_h, loc="upper left", bbox_to_anchor=(0.50, 0.97),
                     bbox_transform=ax.transAxes, title="dot size", labelspacing=1.2,
                     borderpad=0.9, handletextpad=1.4, fontsize=7, title_fontsize=7.5)
    ax.add_artist(leg1)
    mk = lambda mew, lbl: Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#9ab",
                                 markeredgecolor="#111111" if mew else "none", markeredgewidth=mew,
                                 markersize=10, label=lbl)
    out_h = [mk(1.4, "home-factor anchor"), mk(0.8, "credible (95% CI ≠ 0)"), mk(0.0, "not credible"),
             Patch(facecolor=window_color, alpha=0.30, edgecolor="none", label="G window (dep./anx.)")]
    ax.legend(handles=out_h, loc="upper left", bbox_to_anchor=(0.50, 0.63),
              bbox_transform=ax.transAxes, title="markers", fontsize=7, title_fontsize=7.5,
              borderpad=0.8)


def draw_lollipop(ax, L, factor, *, color, axlab, axtag, g_key="overall_severity", disp=None, top=6):
    """One factor's interpretability lollipop: top home indicators, |loading| with 95% CI whisker."""
    disp = DISP if disp is None else disp
    if factor == g_key:
        sub = L[(L["factor"] == factor) & (L["kind"].isin(["g_anchor", "window"]))]
    else:
        sub = L[(L["home"] == factor) & (L["factor"] == factor) & (L["kind"] == "primary")]
    sub = sub.assign(a=sub["loading"].abs()).nlargest(top, "a").sort_values("a")
    y = np.arange(len(sub)); a = sub["a"].to_numpy()
    clo = np.abs(sub["ci_low"].to_numpy()); chi = np.abs(sub["ci_high"].to_numpy())
    lo_w, hi_w = np.minimum(clo, chi), np.maximum(clo, chi)
    ax.hlines(y, 0, a, color=color, lw=1.6, alpha=0.55, zorder=1)
    ax.hlines(y, lo_w, hi_w, color="#999999", lw=0.8, zorder=2)
    ax.scatter(a, y, s=30, color=color, edgecolor="white", lw=0.6, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([disp.get(i, i) for i in sub["item"]], fontsize=5.6)
    xmax = max(1.0, float(a.max()) * 1.15) if len(a) else 1.0
    ax.set_xlim(0, xmax)
    ax.set_title(f"{axtag[factor]} · {axlab[factor]}", fontsize=7.4, fontweight="bold", pad=3)
    ax.tick_params(axis="x", labelsize=6); ax.tick_params(length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
