"""Static export of the DSM-5 → phenotype flow (Supplementary figure) from saved
longitudinal artifacts — no pipeline re-run. Requires kaleido + plotly.

Reads results/longitudinal_assignments.csv, longitudinal_dsm_phenotype.csv,
longitudinal_meta.json; writes:
  reports/figures/figS1_dsm_phenotype_flow.{png,svg}   DSM-5 → V0 → V1 → V2 Sankey
  reports/figures/figS1b_dsm_composition.{png,svg}     DSM-5 composition per phenotype
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
FIG = REPO / "reports" / "figures"

SPECTRUM = {"Trouble dépressif majeur": 0, "Bipolaire de type 2": 1, "Bipolaire de type 1": 2,
            "Bipolaire non spécifié": 3, "Trouble schizo-affectif": 4,
            "Trouble schizophréniforme": 5, "Schizophrénie": 6}
DSM_SHORT = {"Trouble dépressif majeur": "MDD", "Bipolaire de type 2": "BP-II",
             "Bipolaire de type 1": "BP-I", "Bipolaire non spécifié": "BP-NOS",
             "Trouble schizo-affectif": "schizoaff.", "Trouble schizophréniforme": "schizophrenif.",
             "Schizophrénie": "schizophr."}
DSM_COLORS = ["#2c7bb6", "#5e9fc6", "#abd9e9", "#cccccc", "#fdae61", "#f46d43", "#d7191c"]
PALETTE = ["#3498db", "#e67e22", "#16a085", "#9b59b6", "#c0392b", "#7f8c8d"]


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    meta = json.loads((RES / "longitudinal_meta.json").read_text())
    k = int(meta["k"])
    ari = float(meta.get("dsm_phenotype_ari", float("nan")))
    a = pd.read_csv(RES / "longitudinal_assignments.csv")
    wide = a.pivot_table(index="patient_uid", columns="visit", values="phenotype", aggfunc="first")
    dsm_ct = pd.read_csv(RES / "longitudinal_dsm_phenotype.csv", index_col=0)  # dsm × V0 phenotype

    # ── Sankey: DSM-5 → V0 → V1 → V2 ──
    stages = [s for s in ["V0", "V1", "V2"] if s in wide.columns]
    dsm_order = [s for s in sorted(SPECTRUM, key=SPECTRUM.get) if s in dsm_ct.index]
    nodes = [DSM_SHORT[s] for s in dsm_order] + [f"{s}·C{c}" for s in stages for c in range(k)]
    nidx = {n: i for i, n in enumerate(nodes)}
    src, tgt, val = [], [], []
    for s in dsm_order:                                  # DSM-5 → V0
        for jc in dsm_ct.columns:
            n = int(dsm_ct.loc[s, jc])
            if n > 0:
                src.append(nidx[DSM_SHORT[s]]); tgt.append(nidx[f"V0·C{int(jc)}"]); val.append(n)
    for x, y in zip(stages, stages[1:]):                 # V0 → V1 → V2
        pair = wide[[x, y]].dropna()
        ct = pd.crosstab(pair[x].astype(int), pair[y].astype(int))
        for i in ct.index:
            for j in ct.columns:
                if ct.loc[i, j] > 0:
                    src.append(nidx[f"{x}·C{int(i)}"]); tgt.append(nidx[f"{y}·C{int(j)}"])
                    val.append(int(ct.loc[i, j]))
    node_color = [DSM_COLORS[SPECTRUM[s]] for s in dsm_order] + \
                 [PALETTE[c % len(PALETTE)] for _ in stages for c in range(k)]
    fig = go.Figure(go.Sankey(node=dict(label=nodes, color=node_color, pad=16, thickness=16),
                              link=dict(source=src, target=tgt, value=val)))
    fig.update_layout(
        title=f"DSM-5 → phenotype flow (V0→V1→V2)   ARI(DSM, phenotype) = {ari:.3f}",
        height=520, width=1040, font=dict(size=12), margin=dict(t=54, l=10, r=10, b=10))
    for ext in ("png", "svg"):
        fig.write_image(str(FIG / f"figS1_dsm_phenotype_flow.{ext}"), scale=2)
    print(f"  wrote reports/figures/figS1_dsm_phenotype_flow.png/.svg  (ARI={ari:.3f})")

    # ── heatmap: DSM-5 composition of each phenotype ──
    colnorm = dsm_ct.div(dsm_ct.sum(0).replace(0, 1), axis=1)
    fig2 = go.Figure(go.Heatmap(
        z=colnorm.to_numpy(), x=[f"C{c}" for c in colnorm.columns],
        y=[DSM_SHORT.get(s, s) for s in colnorm.index], text=dsm_ct.to_numpy(),
        texttemplate="%{text}", colorscale="Purples", zmin=0, zmax=1,
        colorbar=dict(title="col frac", thickness=12)))
    fig2.update_layout(title="DSM-5 composition of each V0 phenotype (n in cells)",
                       height=380, width=640, xaxis_title="V0 phenotype",
                       yaxis_title="DSM-5 (mood→psychosis)", margin=dict(t=50, l=120, b=46))
    for ext in ("png", "svg"):
        fig2.write_image(str(FIG / f"figS1b_dsm_composition.{ext}"), scale=2)
    print("  wrote reports/figures/figS1b_dsm_composition.png/.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
