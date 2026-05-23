"""The DIMENSIONAL phenotype flow — companion to the discrete-cluster negative
result (Suppl. Fig S1). Where discrete cluster *labels* hop between visits
(~38% persistence), a patient's *band* on a continuous trans-diagnostic axis is
largely retained — the honest "flow" for a dimensional model.

Bands = V0 tertiles (Low/Mid/High), fixed and applied to every visit (display-only
discretization; the model itself stays continuous). We show a trait-like axis
(ADHD/impulsivity/trauma) and a state-like axis (mania/activation) side by side,
and report same-band persistence V0→V1 for all six axes (chance = 1/3).

Reads results/longitudinal_axes_scores.parquet.
Writes reports/figures/fig6_dimensional_flow.{png,svg}      (axis-band alluvial)
       reports/figures/fig6b_band_persistence.{png,svg}     (per-axis persistence)
Requires kaleido + plotly.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
RES = REPO / "results"
FIG = REPO / "reports" / "figures"

AXES = ["depression_severity", "later_onset", "mania_activation",
        "illness_burden", "metabolic", "adhd_impulsivity_trauma"]
PRETTY = {"depression_severity": "Depression", "later_onset": "Later onset",
          "mania_activation": "Mania", "illness_burden": "Illness burden",
          "metabolic": "Metabolic", "adhd_impulsivity_trauma": "ADHD/trauma"}
BAND_LABELS = ["Low", "Mid", "High"]
BAND_COLORS = ["#2ca02c", "#bdbdbd", "#d62728"]   # low / mid / high
DISCRETE_PERSIST = 0.388                            # discrete-cluster V0→V1 (Suppl. S1)

# DSM-5 subtypes ordered on the mood↔psychosis continuum (for the DSM → band panel)
SPECTRUM = {"Trouble dépressif majeur": 0, "Bipolaire de type 2": 1, "Bipolaire de type 1": 2,
            "Bipolaire non spécifié": 3, "Trouble schizo-affectif": 4,
            "Trouble schizophréniforme": 5, "Schizophrénie": 6}
DSM_SHORT = {"Trouble dépressif majeur": "MDD", "Bipolaire de type 2": "BP-II",
             "Bipolaire de type 1": "BP-I", "Bipolaire non spécifié": "BP-NOS",
             "Trouble schizo-affectif": "schizoaff.", "Trouble schizophréniforme": "schizophrenif.",
             "Schizophrénie": "schizophr."}
DSM_COLORS = ["#2c7bb6", "#5e9fc6", "#abd9e9", "#cccccc", "#fdae61", "#f46d43", "#d7191c"]


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Proportion of variance in an axis explained by DSM subtype (one-way ANOVA η²)."""
    m = np.isfinite(values)
    v, g = values[m], groups[m]
    grand = v.mean()
    ss_tot = float(((v - grand) ** 2).sum())
    ss_bet = float(sum(((v[g == lvl].mean() - grand) ** 2) * (g == lvl).sum()
                       for lvl in np.unique(g)))
    return ss_bet / ss_tot if ss_tot > 0 else float("nan")


def load_dsm_for(index) -> pd.Series:
    """patient (cohort, patient_id) → DSM-5 subtype, from the V0 frame."""
    from face_common import build_unified_dataframe
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO / "data", REPO / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
    v0 = df[df["visit"] == "V0"]
    dsm = pd.Series(v0["arm"].astype(str).to_numpy(),
                    index=pd.MultiIndex.from_arrays(
                        [v0["cohort"].str.lower().to_numpy(),
                         v0["usubjid_patients"].astype(str).to_numpy()],
                        names=["cohort", "patient_id"]))
    return dsm[~dsm.index.duplicated(keep="first")].reindex(index)


def band(series: pd.Series, edges: np.ndarray) -> pd.Series:
    """Assign each value to a V0-tertile band 0/1/2 (NaN preserved)."""
    out = pd.Series(np.digitize(series.to_numpy(float), edges), index=series.index)
    return out.where(series.notna(), np.nan)


def alluvial_traces(wide_band: pd.DataFrame, stages, title):
    nodes = [f"{s}·{BAND_LABELS[b]}" for s in stages for b in range(3)]
    nidx = {n: i for i, n in enumerate(nodes)}
    src, tgt, val = [], [], []
    for a, b in zip(stages, stages[1:]):
        pair = wide_band[[a, b]].dropna()
        ct = pd.crosstab(pair[a].astype(int), pair[b].astype(int))
        for i in ct.index:
            for j in ct.columns:
                if ct.loc[i, j] > 0:
                    src.append(nidx[f"{a}·{BAND_LABELS[int(i)]}"])
                    tgt.append(nidx[f"{b}·{BAND_LABELS[int(j)]}"])
                    val.append(int(ct.loc[i, j]))
    node_color = [BAND_COLORS[b] for _ in stages for b in range(3)]
    return go.Sankey(node=dict(label=nodes, color=node_color, pad=18, thickness=16),
                     link=dict(source=src, target=tgt, value=val)), title


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(RES / "longitudinal_axes_scores.parquet")
    v0 = s.xs("V0", level="visit")
    edges = {ax: np.quantile(v0[ax].dropna().to_numpy(), [1/3, 2/3]) for ax in AXES}

    # per-axis same-band persistence V0→V1
    print("Same-band persistence V0→V1 (chance = 0.333):")
    persist = {}
    for ax in AXES:
        b = (s[ax].groupby(level=["cohort", "patient_id", "visit"]).first()
             .reset_index().pivot_table(index=["cohort", "patient_id"],
                                        columns="visit", values=ax, aggfunc="first"))
        if "V0" in b and "V1" in b:
            bb = b[["V0", "V1"]].dropna()
            bv0 = band(bb["V0"], edges[ax]); bv1 = band(bb["V1"], edges[ax])
            persist[ax] = float((bv0.to_numpy() == bv1.to_numpy()).mean())
            print(f"  {PRETTY[ax]:14s} {persist[ax]:.3f}")

    # ── Fig 6: trait vs state axis-band alluvial V0→V1→V2 ──
    stages = ["V0", "V1", "V2"]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                        specs=[[{"type": "sankey"}, {"type": "sankey"}]],
                        subplot_titles=(
                            f"ADHD/trauma (trait) · {persist['adhd_impulsivity_trauma']*100:.0f}% stay in band",
                            f"Mania (state) · {persist['mania_activation']*100:.0f}% stay in band"))
    for col, ax in [(1, "adhd_impulsivity_trauma"), (2, "mania_activation")]:
        wb = pd.DataFrame({st: band(s.xs(st, level="visit")[ax], edges[ax]) for st in stages})
        sankey, _ = alluvial_traces(wb, stages, ax)
        fig.add_trace(sankey, row=1, col=col)
    fig.update_annotations(font_size=12, yshift=6)     # subplot titles
    fig.update_layout(
        title_text="Dimensional phenotype flow: continuous-axis band trajectories V0→V1→V2",
        title_y=0.97, height=500, width=1100, font=dict(size=11),
        margin=dict(t=104, l=10, r=10, b=10))
    for ext in ("png", "svg"):
        fig.write_image(str(FIG / f"fig6_dimensional_flow.{ext}"), scale=2)
    print("  wrote reports/figures/fig6_dimensional_flow.png/.svg")

    # ── Fig 6b: per-axis band persistence vs chance vs discrete ──
    order = sorted(persist, key=persist.get, reverse=True)
    bar = go.Figure(go.Bar(x=[PRETTY[a] for a in order], y=[persist[a] for a in order],
                           marker_color="#1f77b4",
                           text=[f"{persist[a]*100:.0f}%" for a in order], textposition="outside"))
    bar.add_hline(y=1/3, line_dash="dash", line_color="#888",
                  annotation_text="chance (3 bands = 33%)", annotation_position="bottom left")
    bar.add_hline(y=DISCRETE_PERSIST, line_dash="dot", line_color="#d62728",
                  annotation_text="discrete clusters (38%)", annotation_position="top left")
    bar.update_layout(title="Same-band persistence V0→V1 by dimension",
                      height=440, width=860, yaxis_title="P(same band V0→V1)",
                      yaxis_range=[0, 0.78], margin=dict(t=54, l=64, r=24, b=80))
    for ext in ("png", "svg"):
        bar.write_image(str(FIG / f"fig6b_band_persistence.{ext}"), scale=2)
    print("  wrote reports/figures/fig6b_band_persistence.png/.svg")

    # ── Fig 6c: DSM-5 → axis-band (cross-sectional, trans-diagnostic overlap) ──
    dsm = load_dsm_for(v0.index)
    valid = dsm.notna().to_numpy()
    g = dsm.to_numpy().astype(str)[valid]
    V = v0[AXES].to_numpy(float)[valid]
    # point η² + 95% bootstrap CI (B resamples of patients) per axis
    B, rng, n = 2000, np.random.default_rng(0), len(g)
    pt = np.array([eta_squared(V[:, j], g) for j in range(len(AXES))])
    boot = np.empty((B, len(AXES)))
    for b in range(B):
        idx = rng.integers(0, n, n); gb = g[idx]
        for j in range(len(AXES)):
            boot[b, j] = eta_squared(V[idx, j], gb)
    lo_a, hi_a = np.percentile(boot, [2.5, 97.5], axis=0)
    eta = dict(zip(AXES, pt)); lo = dict(zip(AXES, lo_a)); hi = dict(zip(AXES, hi_a))
    print(f"\nVariance in each axis explained by DSM-5 subtype (η² [95% bootstrap CI], B={B}, "
          f"n={n}):")
    for ax in AXES:
        print(f"  {PRETTY[ax]:14s} η²={eta[ax]:.3f}  [{lo[ax]:.3f}, {hi[ax]:.3f}]")
    pd.DataFrame({"axis": AXES, "eta_sq": [eta[a] for a in AXES],
                  "ci_lo": [lo[a] for a in AXES], "ci_hi": [hi[a] for a in AXES]}
                 ).to_csv(RES / "dimensional_dsm_eta_squared.csv", index=False)
    best = max(AXES, key=lambda a: eta[a])              # most diagnosis-linked axis
    bv0 = band(v0[best], edges[best])
    pair = pd.DataFrame({"dsm": dsm.to_numpy(), "band": bv0.to_numpy()}).dropna()
    dsm_order = [s for s in sorted(SPECTRUM, key=SPECTRUM.get) if s in set(pair["dsm"])]
    nodes = [DSM_SHORT[s] for s in dsm_order] + [f"{PRETTY[best]}·{b}" for b in BAND_LABELS]
    nidx = {n: i for i, n in enumerate(nodes)}
    ct = pd.crosstab(pair["dsm"], pair["band"].astype(int))
    src, tgt, val = [], [], []
    for s in dsm_order:
        for b in range(3):
            n = int(ct.loc[s, b]) if (s in ct.index and b in ct.columns) else 0
            if n > 0:
                src.append(nidx[DSM_SHORT[s]]); tgt.append(nidx[f"{PRETTY[best]}·{BAND_LABELS[b]}"])
                val.append(n)
    node_color = [DSM_COLORS[SPECTRUM[s]] for s in dsm_order] + BAND_COLORS
    figc = go.Figure(go.Sankey(node=dict(label=nodes, color=node_color, pad=16, thickness=16),
                               link=dict(source=src, target=tgt, value=val)))
    figc.update_layout(
        title=dict(text=f"DSM-5 → {PRETTY[best]} axis band  "
                        f"(η²={eta[best]:.2f}, 95% CI {lo[best]:.2f}–{hi[best]:.2f})",
                   font=dict(size=14), x=0.5, xanchor="center"),
        height=460, width=1040, font=dict(size=12), margin=dict(t=58, l=10, r=10, b=10))
    for ext in ("png", "svg"):
        figc.write_image(str(FIG / f"fig6c_dsm_axis_flow.{ext}"), scale=2)
    print(f"  wrote reports/figures/fig6c_dsm_axis_flow.png/.svg  (axis={best}, η²={eta[best]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
