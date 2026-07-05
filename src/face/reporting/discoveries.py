"""FACE-discoveries — a single self-contained interactive HTML that illustrates every discovery.

A scrollytelling narrative over the M1->M5 arc (cohorts -> map -> continuum -> temporal -> prognosis ->
treatment -> calibrated claim), with interactive Plotly figures + scientific discussion drawn from the
findings docs and the three manuscripts. Everything (Plotly.js + every figure's data) is inlined into one
portable ``.html`` that opens offline.

Confidential: the M2 cloud embeds per-patient *latent* coordinates, so the output HTML is gitignored (like the
article PDFs) — this builder is tracked, the file is generated locally.

    from face.reporting.discoveries import build_discoveries_html
    build_discoveries_html()                       # -> report/FACE-discoveries.html
    # or: face report discoveries
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from face.config import paths

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- palette + labels (Okabe-Ito house style)
INK = "#1a1a2e"
MUTE = "#6b7280"
BG = "#ffffff"
PANEL = "#f7f8fa"
OKABE = ["#0072B2", "#E69F00", "#009E73", "#56B4E9", "#CC79A7", "#D55E00", "#F0E442", "#000000"]

# 8 factors in fit / canonical order
FACTORS = ["overall_severity", "cognition", "immunometabolic", "sleep",
           "mania_activation", "suicidality", "developmental_risk", "substance"]
FLABEL = {"overall_severity": "G · overall burden", "cognition": "cognition",
          "immunometabolic": "immunometabolic", "sleep": "sleep",
          "mania_activation": "mania / activation", "suicidality": "suicidality",
          "developmental_risk": "developmental risk", "substance": "substance"}
FCOLOR = dict(zip(FACTORS, OKABE))

# 5 archetypes — id -> (short label, color). A2 (immunometabolic, the biology corner) is the standout.
ARCH = {
    0: ("A0 · activation / sleep", "#E69F00"),
    1: ("A1 · severe · clean-biology", "#0072B2"),
    2: ("A2 · immunometabolic", "#D55E00"),
    3: ("A3 · trauma / suicidality", "#CC79A7"),
    4: ("A4 · low-burden / well", "#009E73"),
}
COHORT = {"bp": ("Bipolar", "#0072B2"), "sz": ("Schizophrenia", "#D55E00"), "dr": ("Depression", "#009E73")}


def _theme(fig, *, height=460, title=None, legend=True):
    fig.update_layout(
        template="simple_white", height=height, title=title,
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=13, color=INK),
        paper_bgcolor=BG, plot_bgcolor=BG, margin=dict(l=70, r=30, t=50 if title else 24, b=50),
        showlegend=legend, legend=dict(bgcolor="rgba(255,255,255,0.6)", borderwidth=0),
        hoverlabel=dict(font_size=12, font_family="Inter, sans-serif"),
    )
    return fig


# ============================================================ data ============================================
def _csv(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None


def load_all() -> dict:
    R, M = paths.REPORTS, paths.RESULTS
    d = {
        "loadings": _csv(R / "copula_8factor_loadings.csv"),
        "phi": _csv(R / "copula_8factor_phi.csv"),
        "confound": _csv(R / "12_biology_g_confound.csv"),
        "congruence": _csv(R / "06_congruence.csv"),
        "table1": _csv(R / "table1_characteristics.csv"),
        "arch_profiles": _csv(M / "m2_strata/consolidate/archetype_profiles.csv"),
        "kfamily": _csv(M / "m2_strata/consolidate/k_family_menu.csv"),
        "trait_state": _csv(M / "m3_temporal/trait_state/trait_state.csv"),
        "invariance": _csv(M / "m3_temporal/invariance/congruence.csv"),
        "reliable": _csv(M / "m3_temporal/persistence/reliable_change.csv"),
        "incremental": _csv(M / "m4_prognosis/incremental/incremental_comparison.csv"),
        "atlas": _csv(M / "m4_prognosis/endpoints/archetype_atlas.csv"),
        "clinical": _csv(M / "m4_prognosis/clinical_value/clinical_value.csv"),
        "treatment": _csv(M / "m5_treatment/consolidate/treatment_summary.csv"),
        "course": _csv(M / "m5_treatment/atlas/treatment_course_atlas.csv"),
        "propensity": _csv(M / "m5_treatment/propensity/propensity_summary.csv"),
    }
    sp = M / "m2_strata/structure/data.json"
    d["structure"] = json.loads(sp.read_text()) if sp.exists() else None
    pp = M / "m2_strata/consolidate/patient_strata.parquet"
    d["strata"] = pd.read_parquet(pp) if pp.exists() else None
    return d


# ============================================================ figures ==========================================
def fig_cohorts(d) -> go.Figure:
    n = {"bp": 6252, "sz": 2209, "dr": 552}
    labels = [COHORT[c][0] for c in ("bp", "sz", "dr")]
    fig = go.Figure(go.Pie(labels=labels, values=[n["bp"], n["sz"], n["dr"]], hole=0.62,
                           marker=dict(colors=[COHORT[c][1] for c in ("bp", "sz", "dr")]),
                           textinfo="label+percent", sort=False,
                           hovertemplate="%{label}: %{value:,} patients<extra></extra>"))
    fig.add_annotation(text="<b>9,013</b><br>patients", showarrow=False, font=dict(size=20, color=INK))
    return _theme(fig, height=380, legend=False)


def fig_dot_atlas(d) -> go.Figure:
    L = d["loadings"].copy()
    L = L[L.factor.isin(FACTORS)]
    # order items by home factor then |loading| on home
    home_order = {f: i for i, f in enumerate(FACTORS)}
    L["fx"] = L.factor.map(home_order)
    items = (L[L.kind.isin(["primary", "g_anchor"])].sort_values(
        ["factor", "abs_loading"], ascending=[True, False]))
    order = list(dict.fromkeys(items.item))
    extra = [it for it in L.item.unique() if it not in order]
    order = order + extra
    yi = {it: i for i, it in enumerate(order)}
    L["yy"] = L.item.map(yi)
    cross = L[L.kind == "cross"]
    main = L[L.kind != "cross"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=main.fx, y=main.yy, mode="markers",
        marker=dict(size=6 + 26 * main.abs_loading.clip(0, 1),
                    color=main.loading, colorscale="RdBu", cmid=0, cmin=-1, cmax=1,
                    line=dict(width=[1.4 if e else 0.3 for e in main.excludes_zero], color=INK),
                    colorbar=dict(title="loading", thickness=12, len=0.5, x=1.02)),
        text=main.item, customdata=np.stack([main.factor, main.loading, main.ci_low, main.ci_high, main.kind], -1),
        hovertemplate="<b>%{text}</b> → %{customdata[0]}<br>loading %{customdata[1]:.3f} "
                      "[%{customdata[2]:.3f}, %{customdata[3]:.3f}]<br>%{customdata[4]}<extra></extra>",
        showlegend=False))
    fig.add_trace(go.Scatter(
        x=cross.fx, y=cross.yy, mode="markers+text",
        marker=dict(size=6 + 26 * cross.abs_loading.clip(0, 1), color="rgba(0,0,0,0)",
                    line=dict(width=2.4, color="#c026d3"), symbol="diamond"),
        text=cross.item, textposition="middle right", textfont=dict(size=10, color="#c026d3"),
        customdata=np.stack([cross.factor, cross.loading], -1),
        hovertemplate="<b>%{text}</b> ⤳ %{customdata[0]} (earned cross-loading)<br>"
                      "loading %{customdata[1]:.3f}<extra></extra>", name="earned cross-loadings"))
    fig.update_xaxes(tickmode="array", tickvals=list(range(8)),
                     ticktext=[FLABEL[f].replace(" · ", "<br>") for f in FACTORS], side="top")
    fig.update_yaxes(showticklabels=False, title="109 indicators (grouped by home factor)", autorange="reversed")
    return _theme(fig, height=760, legend=True)


def fig_phi(d) -> go.Figure:
    P = d["phi"].set_index(d["phi"].columns[0]).reindex(index=FACTORS, columns=FACTORS)
    z = P.values.astype(float)
    lab = [FLABEL[f] for f in FACTORS]
    text = [[f"{z[i,j]:.2f}" for j in range(8)] for i in range(8)]
    fig = go.Figure(go.Heatmap(z=z, x=lab, y=lab, zmid=0, zmin=-0.3, zmax=0.3, colorscale="RdBu",
                               text=text, texttemplate="%{text}", textfont=dict(size=11),
                               hovertemplate="%{y} · %{x}: Φ = %{z:.3f}<extra></extra>",
                               colorbar=dict(title="Φ", thickness=12)))
    fig.update_yaxes(autorange="reversed")
    return _theme(fig, height=520, legend=False)


def fig_biology_g(d) -> go.Figure:
    C = d["confound"]
    fig = go.Figure()
    order = ["A0_unadjusted", "A2_antipsychotic", "A3_bmi"]
    names = {"A0_unadjusted": "unadjusted", "A2_antipsychotic": "+ antipsychotic", "A3_bmi": "+ BMI / site"}
    for col in order:
        if col in C.columns:
            fig.add_trace(go.Bar(x=C.domain, y=100 * C[col], name=names.get(col, col),
                                 hovertemplate="%{x}: %{y:.1f}% variance shared with G<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="% of domain variance shared with G (lower = more independent of severity)")
    return _theme(fig, height=420)


def fig_cloud(d) -> go.Figure:
    ps = d["strata"]
    W = ps[[f"arch_w{a}" for a in range(5)]].to_numpy()
    ang = np.pi / 2 - np.arange(5) * 2 * np.pi / 5           # pentagon, top vertex, clockwise
    V = np.c_[np.cos(ang), np.sin(ang)]
    P = W @ V
    H = -(W * np.log(np.clip(W, 1e-12, None))).sum(1)         # mixing entropy (nats, max ln5≈1.61)
    dom = ps["arch_dominant"].to_numpy()
    fig = go.Figure()
    for a in range(5):
        m = dom == a
        fig.add_trace(go.Scattergl(
            x=P[m, 0], y=P[m, 1], mode="markers", name=ARCH[a][0],
            marker=dict(size=4, color=ARCH[a][1], opacity=0.32,
                        line=dict(width=0)),
            customdata=np.stack([ps["cohort"][m].str.upper(), H[m]], -1),
            hovertemplate=f"{ARCH[a][0]}<br>%{{customdata[0]}} · mixing %{{customdata[1]:.2f}} nats<extra></extra>"))
    # centroids at pentagon vertices
    fig.add_trace(go.Scatter(x=V[:, 0] * 1.06, y=V[:, 1] * 1.06, mode="markers+text",
                             marker=dict(size=15, color=[ARCH[a][1] for a in range(5)],
                                         line=dict(width=2, color="white"), symbol="star"),
                             text=[f"A{a}" for a in range(5)], textposition="middle center",
                             textfont=dict(color="white", size=9), showlegend=False,
                             hovertext=[ARCH[a][0] for a in range(5)], hoverinfo="text"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return _theme(fig, height=560)


def fig_radar(d) -> go.Figure:
    A = d["arch_profiles"]
    A = A[A.arm == "A_all9"] if "arm" in A.columns else A
    idcol = "archetype" if "archetype" in A.columns else A.columns[1]
    fig = go.Figure()
    axes = [f for f in FACTORS if f in A.columns]
    for _, r in A.iterrows():
        aid = int(r[idcol])
        vals = [float(r[f]) if pd.notna(r[f]) else 0.0 for f in axes]
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=[FLABEL[f] for f in axes] + [FLABEL[axes[0]]],
                                      name=ARCH[aid][0], line=dict(color=ARCH[aid][1], width=2),
                                      fill="toself", opacity=0.28,
                                      hovertemplate="%{theta}: %{r:.2f} SD<extra>" + ARCH[aid][0] + "</extra>"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[-2.5, 3.7], tickfont=dict(size=9))))
    return _theme(fig, height=560)


def fig_structure_gate(d) -> go.Figure:
    s = d["structure"]
    ev = s["verdict_A"]["evidence"]
    fn = s.get("falsification_null", {})
    real_sil = 0.140
    null_sil = 0.137
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["real data", "structureless null"], y=[real_sil, null_sil],
                         marker_color=["#D55E00", MUTE], width=0.5,
                         error_y=dict(type="data", array=[0.0, 0.002], visible=True),
                         hovertemplate="silhouette %{y:.3f}<extra></extra>"))
    fig.add_annotation(x=0.5, y=max(real_sil, null_sil) + 0.02,
                       text=f"z = 1.13 (n.s.) · HDBSCAN {ev['hdbscan_n']} clusters · dip p = {ev['dip_pc1_p']:.2f}<br>"
                            "<b>verdict: a continuum, not clusters</b>",
                       showarrow=False, font=dict(size=12, color=INK))
    fig.update_yaxes(title="best-partition silhouette", range=[0, 0.20])
    return _theme(fig, height=420, legend=False)


def fig_icc(d) -> go.Figure:
    T = d["trait_state"].copy()
    T = T[T.axis.isin(FACTORS)].sort_values("icc")
    vc = {"trait": "#009E73", "mixed": "#E69F00", "state": "#D55E00", "uninformative": MUTE}
    fig = go.Figure(go.Bar(
        x=T.icc, y=[FLABEL.get(a, a) for a in T.axis], orientation="h",
        marker_color=[vc.get(v, MUTE) for v in T.verdict],
        error_x=dict(type="data", symmetric=False, array=T.icc_hi - T.icc, arrayminus=T.icc - T.icc_lo),
        customdata=np.stack([T.verdict], -1),
        hovertemplate="%{y}: ICC %{x:.2f} · %{customdata[0]}<extra></extra>"))
    fig.add_vline(x=0.6, line=dict(color=MUTE, dash="dot"), annotation_text="trait ≥ 0.6")
    fig.update_xaxes(title="intraclass correlation (durability over V0→V2)", range=[0, 1])
    return _theme(fig, height=440, legend=False)


def fig_reliable(d) -> go.Figure:
    R = d["reliable"].copy()
    R = R[R.axis.isin(FACTORS)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[FLABEL.get(a, a) for a in R.axis], y=R.frac_decrease, name="improved",
                         marker_color="#009E73", hovertemplate="%{x}: %{y:.0%} improved<extra></extra>"))
    fig.add_trace(go.Bar(x=[FLABEL.get(a, a) for a in R.axis], y=R.frac_increase, name="worsened",
                         marker_color="#D55E00", hovertemplate="%{x}: %{y:.0%} worsened<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="fraction with reliable 2-yr change", tickformat=".0%")
    return _theme(fig, height=420)


def fig_elpd(d) -> go.Figure:
    I = d["incremental"].copy()
    I = I[I.outcome == "egf"].copy()
    I = I[I.model != "R3y"].sort_values("d_elpd_vs_ref")
    pretty = {"+archetypesA": "+ A=5 archetypes", "+specifics8": "+ 7 specific axes",
              "+archetypesB": "+ archetypes (⊥G)", "+durable": "+ durable biology only",
              "+tess_k2": "+ tessellation K2", "+tess_k3": "+ tessellation K3", "+tess_k4": "+ tessellation K4"}
    col = ["#D55E00" if m == "+archetypesA" else ("#009E73" if v == "predictive" else MUTE)
           for m, v in zip(I.model, I.verdict)]
    fig = go.Figure(go.Scatter(
        x=I.d_elpd_vs_ref, y=[pretty.get(m, m) for m in I.model], mode="markers",
        marker=dict(size=12, color=col),
        error_x=dict(type="data", array=1.96 * I.se_d_elpd, color=MUTE),
        customdata=np.stack([I.verdict], -1),
        hovertemplate="ΔELPD %{x:.1f} ± %{error_x.array:.1f} · %{customdata[0]}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=MUTE, dash="dot"))
    fig.update_xaxes(title="held-out ΔELPD vs (diagnosis + severity + baseline functioning)")
    return _theme(fig, height=420, legend=False)


def fig_prognostic_atlas(d) -> go.Figure:
    A = d["atlas"].copy()
    A = A[(A.outcome == "egf")].copy()
    piv = A.pivot_table(index="archetype", columns="cohort", values="remission_rate", aggfunc="mean")
    piv = piv.reindex(index=sorted(piv.index))
    ylab = [ARCH[int(i)][0] for i in piv.index]
    cols = [c for c in ["bp", "dr", "sz"] if c in piv.columns]
    fig = go.Figure(go.Heatmap(z=piv[cols].values * 100, x=[COHORT[c][0] for c in cols], y=ylab,
                               colorscale="RdYlGn", zmin=0, zmax=80,
                               text=[[f"{v*100:.0f}%" for v in piv[cols].values[i]] for i in range(len(ylab))],
                               texttemplate="%{text}", textfont=dict(size=12),
                               hovertemplate="%{y} · %{x}: %{z:.0f}% 2-yr functional remission<extra></extra>",
                               colorbar=dict(title="remission %", thickness=12)))
    fig.update_yaxes(autorange="reversed")
    return _theme(fig, height=440, legend=False)


def fig_evalue(d) -> go.Figure:
    T = d["treatment"].copy()
    T = T[(T.outcome == "functioning")].copy()
    T["lab"] = T.question.str.replace("_", "-") + " · " + T.representation
    col = ["#009E73" if "any_axis" in str(v) and mv else "#D55E00" for v, mv in zip(T.moderation_verdict, T.moderation_any_axis)]
    col = ["#D55E00" if b else "#0072B2" for b in T.moderation_any_axis]
    fig = go.Figure(go.Scatter(
        x=T.e_value, y=T.lab, mode="markers",
        marker=dict(size=13, color=col),
        customdata=np.stack([T.ate, T.moderation_verdict], -1),
        hovertemplate="E-value %{x:.2f} · ATE %{customdata[0]:.3f}<br>%{customdata[1]}<extra></extra>"))
    fig.add_vline(x=1.0, line=dict(color=MUTE, dash="dot"), annotation_text="no confounding needed")
    fig.update_xaxes(title="E-value (robustness of the treatment effect to unmeasured confounding)")
    return _theme(fig, height=420, legend=False)


def fig_course_atlas(d) -> go.Figure:
    C = d["course"].copy()
    C = C[C.cohort == "pooled"] if "cohort" in C.columns else C
    eps = {"ep_resistance": "treatment-resistant", "ep_response": "responds", "ep_side_effects": "side-effects"}
    fig = go.Figure()
    for ep, name in eps.items():
        sub = C[C.endpoint == ep].sort_values("archetype")
        if len(sub):
            fig.add_trace(go.Bar(x=[ARCH[int(a)][0] for a in sub.archetype], y=sub.rate * 100, name=name,
                                 hovertemplate="%{x} · " + name + ": %{y:.0f}%<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="2-year rate (%)")
    return _theme(fig, height=440)


FIGS = [
    ("cohorts", fig_cohorts), ("dot_atlas", fig_dot_atlas), ("phi", fig_phi), ("biology_g", fig_biology_g),
    ("cloud", fig_cloud), ("radar", fig_radar), ("structure", fig_structure_gate),
    ("icc", fig_icc), ("reliable", fig_reliable), ("elpd", fig_elpd), ("prognostic_atlas", fig_prognostic_atlas),
    ("evalue", fig_evalue), ("course", fig_course_atlas),
]


# ============================================================ narrative + assembly ============================
def _stat(v, label):
    return f'<div class="stat"><div class="stat-v">{v}</div><div class="stat-l">{label}</div></div>'


def _fig(name, caption=""):
    cap = f'<p class="cap">{caption}</p>' if caption else ""
    return f'<figure class="figwrap"><div class="figholder" id="fig_{name}"></div>{cap}</figure>'


CSS = """
:root{--ink:#1a1a2e;--mute:#6b7280;--line:#e5e7eb;--accent:#D55E00;--bg:#ffffff;--panel:#f7f8fa}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:var(--ink);
  background:var(--bg);line-height:1.62;font-size:16px;-webkit-font-smoothing:antialiased}
a{color:#0072B2}
nav{position:sticky;top:0;z-index:50;background:rgba(255,255,255,.92);backdrop-filter:blur(8px);
  border-bottom:1px solid var(--line);display:flex;gap:.2rem;padding:.5rem 1rem;overflow-x:auto;font-size:.82rem}
nav a{padding:.35rem .7rem;border-radius:999px;text-decoration:none;color:var(--mute);white-space:nowrap;font-weight:600}
nav a:hover{background:var(--panel);color:var(--ink)}nav a.on{background:var(--ink);color:#fff}
.wrap{max-width:1040px;margin:0 auto;padding:0 1.4rem}
.hero{padding:5rem 0 3rem;text-align:center;border-bottom:1px solid var(--line)}
.hero h1{font-size:2.7rem;line-height:1.1;margin:.2rem 0 .6rem;letter-spacing:-.02em}
.hero .sub{font-size:1.15rem;color:var(--mute);max-width:720px;margin:0 auto 1.4rem}
.hero .thesis{font-size:1.02rem;background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:1rem 1.3rem;max-width:820px;margin:1.4rem auto 0;text-align:left}
section{padding:4rem 0;border-bottom:1px solid var(--line);opacity:0;transform:translateY(18px);
  transition:opacity .7s ease,transform .7s ease}
section.reveal{opacity:1;transform:none}
.kicker{font-size:.78rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}
h2{font-size:2rem;letter-spacing:-.02em;margin:.3rem 0 .3rem}
.lead{font-size:1.2rem;color:var(--ink);font-weight:500;margin:.2rem 0 1.4rem;max-width:820px}
p{max-width:820px}.muted{color:var(--mute)}
.stats{display:flex;flex-wrap:wrap;gap:.8rem;margin:1.4rem 0}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:.8rem 1.1rem;min-width:130px}
.stat-v{font-size:1.6rem;font-weight:800;color:var(--accent);line-height:1}
.stat-l{font-size:.8rem;color:var(--mute);margin-top:.3rem}
.figwrap{margin:1.6rem 0;background:var(--bg);border:1px solid var(--line);border-radius:16px;padding:.6rem .6rem 0}
.figholder{width:100%}.cap{font-size:.85rem;color:var(--mute);padding:.4rem .8rem 1rem;margin:0}
.limit{border-left:3px solid #E69F00;background:#fffdf5;border-radius:0 10px 10px 0;padding:.7rem 1.1rem;
  margin:1.4rem 0;font-size:.95rem;color:#7a6a2f}
.limit b{color:#8a6d00}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}@media(max-width:820px){.grid2{grid-template-columns:1fr}}
.scorecard{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;margin:1.4rem 0}
@media(max-width:820px){.scorecard{grid-template-columns:1fr}}
.score{border:1px solid var(--line);border-radius:16px;padding:1.3rem}
.score.good{background:#f2fbf6;border-color:#a7e0c4}.score.bad{background:#fef4ee;border-color:#f2c3a3}
.score h3{margin:.1rem 0 .5rem;font-size:1.2rem}.score .tag{font-size:1.5rem}
footer{padding:3rem 0;color:var(--mute);font-size:.85rem;text-align:center}
"""

JS = """
const io=new IntersectionObserver((es)=>es.forEach(e=>{if(e.isIntersecting)e.target.classList.add('reveal')}),
  {threshold:.08});document.querySelectorAll('section').forEach(s=>io.observe(s));
const links=[...document.querySelectorAll('nav a')];
const spy=new IntersectionObserver((es)=>es.forEach(e=>{if(e.isIntersecting){
  links.forEach(l=>l.classList.toggle('on',l.getAttribute('href')==='#'+e.target.id))}}),{rootMargin:'-40% 0px -55% 0px'});
document.querySelectorAll('section[id]').forEach(s=>spy.observe(s));
"""

NAV = [("m1", "① The map"), ("m2", "② Continuum"), ("m3", "③ Temporal"),
       ("m4", "④ Prognosis"), ("m5", "⑤ Treatment"), ("synthesis", "⑥ Verdict")]


def _sections() -> str:
    s = []
    # ---- M1 ----
    s.append(f"""<section id="m1"><div class="wrap">
      <div class="kicker">Milestone 1 · discovery</div>
      <h2>Eight dimensions replace diagnosis</h2>
      <p class="lead">One global, missingness-aware Bayesian bifactor/ESEM on 9,013 patients' observed cells
      resolves psychopathology into a general burden factor <b>G</b> orthogonal to seven specific axes — and
      the biology axis stands apart as the least severity-entangled domain.</p>
      <p>Fitting all three cohorts jointly (no imputation, observed-cell likelihood), the data return a
      near-simple-structure map: <b>G</b> carries only functioning (FAST, GAF, EQ-5D, CGI-S) and is held
      orthogonal to seven specifics — cognition, <b>immunometabolic</b>, sleep, mania/activation, suicidality,
      developmental-risk, substance. The immunometabolic axis is <i>one</i> factor carrying cardiometabolic and
      inflammatory markers together (BMI → 0.95, CRP → 0.37). Only three cross-loadings survive a
      regularized-horseshoe prior — CTQ-37, PSQI-latency and PSQI-daytime onto cognition — each with a 95%
      credible interval excluding zero, so the simple structure is <i>earned, not imposed</i>.</p>
      <div class="stats">{_stat("8","transdiagnostic dimensions")}{_stat("0.06","immunometabolic → G loading (vs 0.20 others)")}{_stat("3","earned cross-loadings")}{_stat("~0.07","SRMR (absolute fit)")}</div>
      {_fig("dot_atlas","The loading dot-atlas — every indicator × factor. Dot size and colour encode the loading; a bold outline marks credible-interval exclusion of zero; diamonds mark the 3 earned cross-loadings onto cognition. Hover for exact values.")}
      <div class="grid2">{_fig("phi","The 8×8 inter-factor correlation Φ. G's row/column is pinned to zero (bifactor); specifics are weakly correlated (mean |Φ| ≈ 0.08).")}{_fig("biology_g","Biology ⊥ G: the share of each domain's variance shared with the general factor, before and after adjusting for antipsychotics and BMI/site. Immunometabolic stays the most independent of severity.")}</div>
      <div class="limit"><b>Honest limit —</b> G is deliberately conservative (an impairment/distress axis, not a latent liability); developmental-risk indexes early adversity (CTQ + birth history), not measured neurodevelopment; one suicidality count-item mis-fits in its tail.</div>
    </div></section>""")
    # ---- M2 ----
    s.append(f"""<section id="m2"><div class="wrap">
      <div class="kicker">Milestone 2 · discovery</div>
      <h2>A continuum, not biotypes</h2>
      <p class="lead">On the 8-dimensional map the population <b>clusters nowhere</b> — it is a graded
      continuum. Five stable archetype "corners" describe its extremes, but patients live on the spectrum
      between them.</p>
      <p>A full structure-discovery battery (Hopkins, silhouette, HDBSCAN, dip, gap, Mapper) run over the
      posterior returns a continuum verdict: the best-partition silhouette (0.140) is statistically
      indistinguishable from a structureless-Gaussian null (0.137 ± 0.002, z = 1.13, n.s.), HDBSCAN finds
      <b>zero</b> clusters, and PC1 is unimodal. The load-bearing objects are the continuous coordinates plus a
      stable <b>A = 5</b> archetype simplex (the last archetype count with cross-seed Tucker ≥ 0.8). Crucially,
      <b>biology ⊥ symptoms ⊥ severity</b>: corner A1 and corner A2 sit at the <i>same</i> high severity but
      opposite immunometabolic load — severity does not determine biology.</p>
      <div class="stats">{_stat("continuum","structure verdict")}{_stat("A=5","stable archetype simplex")}{_stat("0.006","ARI vs DSM-5 (≈ transdiagnostic)")}{_stat("none","operative K (tessellation optional)")}</div>
      {_fig("cloud","Every one of the 9,013 patients, projected into the A=5 archetype simplex (pentagon corners = archetypes; each patient sits at the centroid of their membership weights). Colour = dominant corner; the diffuse fog — with no separated islands — is the continuum. Stars mark the five corners.")}
      <div class="grid2">{_fig("radar","The five archetype corners across the 8 axes (z-scored). A2 · immunometabolic is the most distinct and most stable corner — high biology, high severity, high suicidality.")}{_fig("structure","The falsification test: real-data cluster separation vs a structureless null. Indistinguishable — the space is a spectrum, not a typology.")}</div>
      <div class="limit"><b>Honest caveat —</b> the archetype corners are interpretable extremes, not natural kinds; the K-tessellation is a convenience (operative K = none). Dominant-label churn is expected with a simplex — the membership <i>weights</i> are the stable objects.</div>
    </div></section>""")
    # ---- M3 ----
    s.append(f"""<section id="m3"><div class="wrap">
      <div class="kicker">Milestone 3 · discovery</div>
      <h2>Biology is trait, symptoms are state</h2>
      <p class="lead">Scored forward onto follow-up (V0→V1→V2, never re-estimated), the map holds — and the
      immunometabolic axis is the single most durable marker in the whole space.</p>
      <p>The measurement is invariant across visits (all four backbone axes Tucker φ ≥ 0.987), so change reads
      as patient-change, not scale-drift. An error-corrected trait/state decomposition then finds the
      immunometabolic axis at <b>ICC 0.91</b> — "excellent reliability" by any convention — while symptoms
      slide: suicidality and severity move most, the population improving overall. The clinical logic writes
      itself: <b>stratify on the durable biology, monitor the moving symptoms.</b></p>
      <div class="stats">{_stat("0.91","immunometabolic ICC (most durable)")}{_stat("4/4","backbone axes invariant")}{_stat("0.70","cognition ICC (trait)")}{_stat("state","suicidality / symptoms")}</div>
      <div class="grid2">{_fig("icc","Trait/state durability (ICC) per axis over two years. Green = trait, orange = mixed, vermillion = state. Immunometabolic biology anchors the top; symptoms fall to the state end.")}{_fig("reliable","Who reliably changes over two years, by axis — improved (green) vs worsened (vermillion). Symptoms flux; biology barely moves.")}</div>
      <div class="limit"><b>Honest caveat —</b> the error-corrected ICC is the clean signal; geometric archetype persistence (weights) agrees on the core but does not co-rank cell-for-cell. Internal + temporal validity only — external replication is still owed.</div>
    </div></section>""")
    # ---- M4 ----
    s.append(f"""<section id="m4"><div class="wrap">
      <div class="kicker">Milestone 4 · discovery</div>
      <h2>The durable biology forecasts functioning</h2>
      <p class="lead">On the fixed map, a baseline transdiagnostic profile predicts 2-year <b>functioning</b>
      incrementally beyond diagnosis + severity + baseline — and the immunometabolic corner is the
      worst-prognosis pole.</p>
      <p>An errors-in-variables Bayesian GLM shows the A=5 archetypes add a decisive held-out
      <b>ΔELPD +62.8</b> over the clinical reference for functioning (much less for severity, which is
      autoregression-saturated). The prognostic atlas spans a <b>22% → 63%</b> functional-remission gradient
      across corners — the immunometabolic corner (A2) worst, the well pole (A4) best — and it holds
      <i>within</i> each diagnosis (composition explains only ~4%). The map is co-informative with DSM-5 (each
      adds on top of the other). Honestly, the individual-level binary lift is small (remission AUC +0.010):
      the value is group-level stratification and continuous forecasting, not an individual yes/no calculator.</p>
      <div class="stats">{_stat("+62.8","ΔELPD (A=5 archetypes, functioning)")}{_stat("22→63%","remission gradient across corners")}{_stat("none","operative K")}{_stat("+0.010","individual remission AUC (honest)")}</div>
      <div class="grid2">{_fig("elpd","Incremental held-out predictive gain (ΔELPD) for 2-year functioning, per encoding, vs diagnosis + severity + baseline. The A=5 archetypes win decisively; the tessellation adds nothing beyond them.")}{_fig("prognostic_atlas","2-year functional-remission rate by archetype corner × cohort. The 22%→63% gradient is transdiagnostic and holds within every diagnosis.")}</div>
      <div class="limit"><b>Honest caveat —</b> the isolated immunometabolic axis alone is ambiguous (+2.3 ΔELPD); the signal lives in the fuller archetype configuration. Internal incremental-association only — course-dependent, BP-led, 2-year horizon, scales not events.</div>
    </div></section>""")
    # ---- M5 ----
    s.append(f"""<section id="m5"><div class="wrap">
      <div class="kicker">Milestone 5 · discovery</div>
      <h2>Prognostic, not prescriptive — bounds & defends</h2>
      <p class="lead">On observational treatment-as-usual the map does <b>not</b> reliably moderate or select
      treatment — but the prognosis survives treatment adjustment, and the map describes who faces a hard
      2-year course.</p>
      <p>A proper causal pipeline (overlap → propensity → doubly-robust EIV moderation → E-value → MDE) finds
      no reliable selection: lithium-in-BP is a well-identified, MDE-bounded <b>null</b> (E 1.20–1.28 — the
      design could have seen an effect and didn't), antipsychotic-BP a confounded average effect (E 1.80) with
      suggestive-but-unconfirmed moderation, clozapine-SZ underpowered. Yet the prognostic carrier <i>survives</i>
      treatment adjustment (the immunometabolic corner attenuates only 6–8%), so it is not a treatment proxy —
      M5 <b>defends</b> M4. And the corner <b>describes</b> the course: the immunometabolic pole faces ~2× the
      treatment-resistance and side-effect burden of the well pole. The forward-looking claim is monitoring,
      not prescription.</p>
      <div class="stats">{_stat("1.2–1.28","lithium-BP E-value (bounded null)")}{_stat("6–8%","carrier attenuation (survives)")}{_stat("44% vs 20%","A2 vs A4 resistance")}{_stat("M5b","selection needs randomized data")}</div>
      <div class="grid2">{_fig("evalue","Treatment-effect robustness (E-value) per drug × representation for functioning. None crosses into reliable moderation on observational data; lithium is a well-identified null.")}{_fig("course","2-year treatment course by archetype corner — resistance, response, side-effects. The immunometabolic corner (A2) faces the hardest course; the well pole (A4) the easiest.")}</div>
      <div class="limit"><b>Honest caveat —</b> observational TAU only, confounding-by-indication dominant; the atlas is proven as <i>stratification</i> (gradient), not individual discrimination (resistance AUC-marginal). True treatment <i>selection</i> is a future M5b needing randomized/trial-arm data.</div>
    </div></section>""")
    # ---- synthesis ----
    s.append("""<section id="synthesis"><div class="wrap">
      <div class="kicker">The calibrated claim</div>
      <h2>What the program does — and does not — show</h2>
      <p class="lead">A real, stable, <b>continuum</b> transdiagnostic map, with biology least-entangled from
      severity, carrying a small but genuine group-level prognostic signal for functioning — reported as a
      deliberate counterweight to biotype/biomarker over-claiming.</p>
      <div class="scorecard">
        <div class="score good"><div class="tag">✓</div><h3>Scientific validity — demonstrated</h3>
          <p class="muted">A genuine 8-dimensional continuum (not biotypes), biology ⊥ severity, immunometabolic
          durability ICC 0.91, and a decisive group-level 2-year functioning forecast (ΔELPD +62.8) that
          complements diagnosis and survives treatment adjustment.</p></div>
        <div class="score bad"><div class="tag">✕</div><h3>Strong clinical utility — not (yet)</h3>
          <p class="muted">The individual-level prognostic gain is small (remission AUC +0.010), and the map does
          not moderate or select treatment on observational data. Individual prediction and treatment guidance
          would need incident events, randomized treatment arms, and external validation this baseline lacks.</p></div>
      </div>
      <p class="muted">These are different bars, reported as such. The map is <b>prognostic and descriptive,
      not prescriptive</b>: stratify on the durable biology, monitor the moving symptoms, and reserve treatment
      <i>selection</i> for a randomized M5b.</p>
    </div></section>""")
    return "\n".join(s)


def build_discoveries_html(out_path: str | Path | None = None) -> Path:
    """Build the single self-contained interactive discoveries HTML. Returns the output path."""

    out = Path(out_path) if out_path else (paths.REPO / "report" / "FACE-discoveries.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = load_all()

    cfg = {"displayModeBar": False, "responsive": True}
    divs = {}
    for name, fn in FIGS:
        try:
            fig = fn(data)
            divs[name] = pio.to_html(fig, include_plotlyjs=False, full_html=False,
                                     div_id=f"fig_{name}", config=cfg)
        except Exception as e:  # a missing input degrades one figure, not the whole doc
            divs[name] = f'<div class="cap">[figure {name} unavailable: {type(e).__name__}: {e}]</div>'

    nav = "".join(f'<a href="#{i}">{t}</a>' for i, t in NAV)
    sections = _sections()
    # inject each figure's rendered div/script into its placeholder
    for name in divs:
        sections = sections.replace(f'<div class="figholder" id="fig_{name}"></div>', divs[name])
    sections = sections.replace('<figure class="figwrap">', '<figure class="figwrap">')  # no-op guard
    # the cohorts figure lives in the hero
    hero_fig = divs.get("cohorts", "")

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FACE — the discoveries</title>
<style>{CSS}</style>
<script>{get_plotlyjs()}</script>
</head><body>
<nav><a href="#top" style="font-weight:800;color:var(--ink)">FACE</a>{nav}</nav>
<header id="top" class="hero"><div class="wrap">
  <div class="kicker">transdiagnostic psychiatry · BP · SZ · DR</div>
  <h1>From three diagnoses to one map of illness</h1>
  <p class="sub">A missingness-aware Bayesian pipeline turns the harmonized baseline of 9,013 patients across
  bipolar disorder, schizophrenia and depression into a transdiagnostic dimensional map — then into strata,
  temporal coherence, prognosis and treatment tests.</p>
  <div class="figwrap" style="max-width:520px;margin:1.6rem auto 0">{hero_fig}</div>
  <div class="thesis"><b>The thesis in one line —</b> psychopathology is a graded continuum, biology is the
  most durable axis and least entangled with severity, and a baseline profile forecasts 2-year functioning at
  the group level — a calibrated, honestly-bounded alternative to biotype over-claiming.</div>
</div></header>
{sections}
<footer><div class="wrap">FACE discoveries · generated from the reproduced M1→M5 pipeline
(<code>face report discoveries</code>) · figures are interactive — hover, zoom, and toggle the legend.
Numbers reconciled to the frozen oracle. Confidential per-patient coordinates: do not redistribute externally.</div></footer>
<script>{JS}</script>
</body></html>"""
    out.write_text(html)
    return out


if __name__ == "__main__":
    p = build_discoveries_html()
    print(f"wrote {p}  ({p.stat().st_size/1e6:.1f} MB)")

