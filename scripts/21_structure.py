#!/usr/bin/env python3
"""21 — M2.1 structure-discovery gate: cluster vs continuum vs branched (§3.1).

Characterizes the SHAPE of the M1 9-D coordinate cloud BEFORE any mixture is fit — the reported verdict
(clustered / continuum / branched) decides whether discrete strata are the right object and which view
(mixture vs archetypes) leads. Run on both G-arms (A: all 9; B: 8 specifics = pure profile) and
uncertainty-aware over M1 draws. The honest null (continuum) is permitted.

    python3 scripts/21_structure.py

Reads results/face/m2/{coordinates_full.parquet, coordinates_draws.npz, validation_table.parquet}.
Writes reports/21_structure.md + docs/figures/21_{selection,embedding,mapper}.png.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
M2 = REPO / "results" / "face" / "m2"
SEED = 20260609

CANON = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep", "mania_activation",
         "suicidality", "developmental_risk", "substance"]


def run_battery(X, axes, draws_cols=None, draws=None, label=""):
    from face.strata import structure as st
    print(f"  [{label}] hopkins / dip / gmm-bic / silhouette / gap / hdbscan ...", flush=True)
    diag = {
        "hopkins": st.hopkins(X, seed=SEED),
        "dip": st.dip_test(X, axes),
        "gmm_bic": st.gmm_bic_sweep(X, range(1, 13), seed=SEED),
        "silhouette": st.silhouette_sweep(X, range(2, 13), seed=SEED),
        "gap": st.gap_statistic(X, range(1, 13), B=10, seed=SEED),
        "hdbscan": st.hdbscan_summary(X),
    }
    if draws is not None and draws_cols is not None:
        print(f"  [{label}] uncertainty sweep over draws ...", flush=True)
        diag["uncertainty"] = st.uncertainty_sweep(draws, draws_cols, n_draw=20, seed=SEED)
    diag["verdict"] = st.verdict(diag)
    return diag


def main():
    df = pd.read_parquet(M2 / "coordinates_full.parquet")
    vt = pd.read_parquet(M2 / "validation_table.parquet")
    dz = np.load(M2 / "coordinates_draws.npz", allow_pickle=True)
    draws = dz["draws"]                                            # [S, N, 9]
    X9 = df[[f"{f}__mean" for f in CANON]].to_numpy()
    axesB = [a for a in CANON if a != "overall_severity"]
    colsB = [CANON.index(a) for a in axesB]
    X8 = X9[:, colsB]

    print("[A] Arm A — all 9 axes (severity × profile)", flush=True)
    A = run_battery(X9, CANON, draws_cols=list(range(9)), draws=draws, label="A")
    print("[B] Arm B — 8 specifics (pure profile, G removed)", flush=True)
    B = run_battery(X8, axesB, draws_cols=None, draws=None, label="B")   # uncertainty sweep on A only (cost)

    _fig_selection(A, B)
    _fig_embedding(X9, df, vt)
    mp_summary = _fig_mapper(X9, df)

    _report(A, B, mp_summary, df, vt)
    print("\n[done] wrote reports/21_structure.md + figures")


def _fig_selection(A, B):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for arm, d, c in [("A (9d)", A, "#2c7fb8"), ("B (8d)", B, "#d95f0e")]:
        bk = sorted(d["gmm_bic"]["bic"]); ks = sorted(d["gmm_bic"]["bic"])
        ax[0].plot(ks, [d["gmm_bic"]["bic"][k] for k in ks], "-o", label=arm, color=c)
        sk = sorted(d["silhouette"]["silhouette"])
        ax[1].plot(sk, [d["silhouette"]["silhouette"][k] for k in sk], "-o", label=arm, color=c)
        gk = sorted(d["gap"]["gap"])
        ax[2].plot(gk, [d["gap"]["gap"][k] for k in gk], "-o", label=arm, color=c)
    ax[0].set_title("GMM BIC (lower=better; flat/monotone ⇒ continuum)"); ax[0].set_xlabel("K")
    ax[1].set_title("KMeans silhouette (peak<0.15 ⇒ no clusters)"); ax[1].set_xlabel("K")
    ax[1].axhline(0.15, ls="--", c="grey", lw=0.8)
    ax[2].set_title("Gap statistic"); ax[2].set_xlabel("K")
    for a in ax:
        a.legend(fontsize=8)
    fig.suptitle("M2.1 structure-discovery — model-selection diagnostics", y=1.02)
    fig.tight_layout(); fig.savefig(FIGS / "21_selection.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def _fig_embedding(X9, df, vt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=SEED).fit_transform(X9)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    coh = vt["cohort"].to_numpy() if "cohort" in vt else df["cohort"].to_numpy()
    for c in np.unique(coh):
        mlab = coh == c
        ax[0].scatter(emb[mlab, 0], emb[mlab, 1], s=3, alpha=0.4, label=str(c))
    ax[0].set_title("UMAP — by cohort (Q3: mixed?)"); ax[0].legend(markerscale=3, fontsize=8)
    arm = vt["arm"].astype(str).to_numpy()
    for a in pd.unique(arm):
        mlab = arm == a
        ax[1].scatter(emb[mlab, 0], emb[mlab, 1], s=3, alpha=0.4, label=a[:18])
    ax[1].set_title("UMAP — by DSM-5 subtype"); ax[1].legend(markerscale=3, fontsize=6)
    for k, name in [(2, "overall_severity"), (3, "inflammatory")]:
        v = df[f"{name}__mean"].to_numpy()
        sc = ax[k].scatter(emb[:, 0], emb[:, 1], s=3, alpha=0.5, c=v, cmap="viridis",
                           vmin=np.percentile(v, 2), vmax=np.percentile(v, 98))
        ax[k].set_title(f"UMAP — by {name} (G vs biology)"); fig.colorbar(sc, ax=ax[k], shrink=0.7)
    fig.suptitle("M2.1 — UMAP embedding of the 9-D cloud (viz-only, not a clustering input)", y=1.02)
    fig.tight_layout(); fig.savefig(FIGS / "21_embedding.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def _fig_mapper(X9, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    from face.strata.structure import mapper_graph
    lens = df["overall_severity__mean"].to_numpy()                 # lens = severity (interpretable)
    G, members = mapper_graph(X9, lens, n_cubes=12, overlap=0.4, min_node=25, seed=SEED)
    if G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "components": 0}
    comps = nx.number_connected_components(G)
    inflam = df["inflammatory__mean"].to_numpy()
    pos = nx.spring_layout(G, seed=SEED, weight="weight")
    sizes = np.array([G.nodes[n]["size"] for n in G.nodes])
    ncol = np.array([float(inflam[list(members[n])].mean()) for n in G.nodes])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nd = nx.draw_networkx_nodes(G, pos, node_size=20 + 4 * np.sqrt(sizes), node_color=ncol,
                                cmap="magma", ax=ax)
    fig.colorbar(nd, ax=ax, shrink=0.7, label="node mean inflammatory")
    ax.set_title(f"M2.1 Mapper (lens=severity) — {G.number_of_nodes()} nodes, "
                 f"{comps} component(s)\nsingle chain ⇒ continuum · flares ⇒ branched · islands ⇒ clusters")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(FIGS / "21_mapper.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "components": comps}


def _diag_md(d, name):
    v = d["verdict"]
    e = v["evidence"]
    lines = [f"### Arm {name} — verdict: **{v['label']}** (clustered-signals {v['clustered_score']}/{v['n_signals']})",
             f"- Hopkins **{e['hopkins']:.2f}** (≈0.5 continuum · →1 clustered)",
             f"- GMM-BIC best K **{e['gmm_k_best']}**, ΔBIC(best vs K=1) {d['gmm_bic']['gain_over_k1']:.0f}, "
             f"monotone-decreasing **{e['gmm_monotone']}** (monotone ⇒ over-segmenting, no interior optimum)",
             f"- silhouette peak **{e['silhouette_peak']:.3f}** (<0.15 ⇒ no separation) · gap-stat K_opt **{e['gap_k_opt']}**",
             f"- dip PC1 p **{e['dip_pc1_p']:.3f}**, axes multimodal (p<.05) **{e['n_axes_multimodal']}**/{len(d['dip'])}",
             f"- HDBSCAN clusters **{e['hdbscan_n']}**, noise frac **{e['hdbscan_noise']:.2f}**"]
    if "uncertainty" in d:
        u = d["uncertainty"]
        lines.append(f"- uncertainty-aware (over draws): Hopkins {u['hopkins_mean']:.2f}±{u['hopkins_sd']:.2f}, "
                     f"GMM K_best mode **{u['k_best_mode']}**, distribution {u['k_best_distribution']}")
    return "\n".join(lines)


def _report(A, B, mp, df, vt):
    lead = "archetypes (continuum-honest soft view)" if "continuum" in A["verdict"]["label"] \
        else "mixture (discrete regions)"
    md = ["# 21 — M2.1 structure-discovery gate (cluster vs continuum vs branched)", "",
          "The reported SHAPE verdict for the M1 9-D coordinate cloud (§3.1), run on both G-arms and "
          "uncertainty-aware over M1 draws — *before* any mixture is fit. Coordinates on native latent "
          "z-scale; embeddings are viz-only (never a clustering input).", "",
          f"## Verdict — Arm A: **{A['verdict']['label']}** · Arm B: **{B['verdict']['label']}**",
          f"**Lead representation (per §3.1): {lead}.** The other view is reported alongside.", "",
          _diag_md(A, "A (all 9 — severity×profile)"), "",
          _diag_md(B, "B (8 specifics — pure profile)"), "",
          "## Mapper (lens = severity)",
          f"- {mp['nodes']} nodes · {mp['edges']} edges · **{mp['components']} connected component(s)** "
          "(1 chain ⇒ continuum; flares ⇒ branched; multiple islands ⇒ clusters). See figure.", "",
          "## Reading",
          "- **Continuum / weak-cluster** evidence ⇒ the coordinate space is graded, not a set of natural "
          "kinds. The mixture is then reported as a *soft tessellation* and **archetypes lead** (extreme "
          "phenotypes + simplex membership) — still a valid, actionable probabilistic decision-region object.",
          "- **Clustered** evidence ⇒ the mixture's discrete regions lead; archetypes complement.",
          "- This is a *precondition* check (§1.7): it does not by itself make strata 'better than DSM-5' — "
          "that is the M4/M5 predictive/treatment head-to-head.", "",
          "## Figures",
          "- `docs/figures/21_selection.png` — BIC / silhouette / gap vs K (both arms).",
          "- `docs/figures/21_embedding.png` — UMAP by cohort / DSM-5 subtype / severity / inflammatory.",
          "- `docs/figures/21_mapper.png` — Mapper graph (lens = severity, node colour = inflammatory)."]
    (REPORTS / "21_structure.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
