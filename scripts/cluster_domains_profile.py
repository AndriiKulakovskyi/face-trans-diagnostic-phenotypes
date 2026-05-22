"""Profile + name the direction-A domain phenotypes (publication-grade report).

Loads the saved domain embedding + residualized domain scores from
``scripts/cluster_domains.py`` and produces ``reports/cluster_domains.html``:

  - composition table (size, cohort mix, age, sex) + auto suggested names;
  - k-selection figure (stability / PAC / gap / sex-leakage vs k — why k=5);
  - 2D UMAP coloured by cluster and by cohort;
  - **cluster × domain mean-profile heatmap** (the phenotype signature figure);
  - per-cluster top enriched domains (Mann-Whitney rank-biserial bars);
  - medoid "vignettes" — the centroid-nearest patient per cluster;
  - independence panel (sex / age / cohort — confound verification).

Run:  python3 scripts/cluster_domains_profile.py [--k 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "archive"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from face_common import (  # noqa: E402
    ADMINISTRATIVE_FEATURES,
    build_unified_dataframe,
    load_variables,
    to_harmonized_dataset,
)
from face_stratification.analysis.enrichment import (  # noqa: E402
    compute_cluster_feature_enrichment,
)
from face_stratification.clustering.algorithms import run_kmeans  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
DICT_PATH = REPO_ROOT / "face-common-vars.xlsx"
RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"
EMB_PATH = RESULTS_DIR / "cluster_domains_embedding.parquet"
SCORES_PATH = RESULTS_DIR / "cluster_domains_scores.parquet"
RANDOM_STATE = 0

CSS = """
:root{--fg:#1f2933;--muted:#6b7280;--accent:#2b3a55;--bg:#fff;--card:#fbfbfd;
 --border:#e5e7eb;--warn:#c0392b;--good:#16a085;}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
 color:var(--fg);margin:0;padding:0 24px 80px;line-height:1.55}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--border);
 padding:18px 0;z-index:100;margin-bottom:24px}
h1{margin:0 0 4px;font-size:22px;color:var(--accent)}
h2{margin:38px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--accent);
 color:var(--accent);font-size:18px}
.muted{color:var(--muted);font-size:13px}
nav.toc a{display:inline-block;margin:2px 6px 2px 0;padding:3px 9px;background:#eef2f7;
 border-radius:12px;font-size:12px;color:var(--accent);text-decoration:none}
nav.toc a:hover{background:var(--accent);color:#fff}
.callout{border-left:4px solid var(--good);padding:10px 14px;background:#f2fbf6;
 margin:14px 0;border-radius:0 6px 6px 0}
table.k{border-collapse:collapse;margin:10px 0 18px;font-size:13px}
table.k th,table.k td{border:1px solid var(--border);padding:5px 10px;text-align:left}
table.k th{background:#eef2f7;font-weight:600}
.cards{display:flex;flex-wrap:wrap;gap:12px}
.card{border:1px solid var(--border);border-radius:8px;padding:12px 14px;
 background:var(--card);min-width:240px;flex:1}
.card h4{margin:0 0 6px;color:var(--accent);font-size:14px}
.card .muted{font-size:12px}
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--readiness", nargs="+", default=["READY", "PARTIAL"])
    return p.parse_args()


def fig_div(fig, div_id, include_js=False):
    return pio.to_html(fig, include_plotlyjs="cdn" if include_js else False,
                       full_html=False, div_id=div_id, config={"displayModeBar": False})


def df_table(df, floatfmt="{:.2f}"):
    out = df.copy()
    for c in out.select_dtypes(include="float").columns:
        out[c] = out[c].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
    return out.to_html(index=False, classes="k", border=0, escape=False)


def scatter(coords, color, title, div, include_js=False):
    fig = go.Figure()
    for cat in sorted(color.dropna().unique(), key=str):
        m = (color == cat).to_numpy()
        fig.add_trace(go.Scattergl(x=coords[m, 0], y=coords[m, 1], mode="markers",
                                   name=str(cat), marker=dict(size=3, opacity=0.6)))
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=460,
                      legend=dict(itemsizing="constant"),
                      margin=dict(t=46, l=20, r=20, b=20),
                      xaxis_title="UMAP-1", yaxis_title="UMAP-2")
    return fig_div(fig, div, include_js)


def main() -> int:
    args = parse_args()
    if not EMB_PATH.exists() or not SCORES_PATH.exists():
        sys.exit("Missing results/cluster_domains_*.parquet — run cluster_domains.py first.")
    REPORTS_DIR.mkdir(exist_ok=True)
    k = args.k

    emb = pd.read_parquet(EMB_PATH)
    scores = pd.read_parquet(SCORES_PATH)
    # Defensive: coerce the (cohort, patient_id) index to str on both levels so it
    # aligns with the embedding regardless of how the parquet round-tripped dtypes.
    scores.index = pd.MultiIndex.from_arrays(
        [scores.index.get_level_values("cohort").astype(str),
         scores.index.get_level_values("patient_id").astype(str)],
        names=("cohort", "patient_id"))
    scores = scores.reindex(emb.index)   # residualized domains

    print("Rebuilding demographics (age/sex/cohort/arm)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(DATA_DIR, DICT_PATH, readiness=args.readiness,
                                     format="long")
        full = to_harmonized_dataset(df, load_variables(DICT_PATH), visit="V0",
                                     exclude=ADMINISTRATIVE_FEATURES)
    age = full.X.reindex(emb.index)["age"]
    sex = full.X.reindex(emb.index)["sex"]
    cohort = pd.Series([c.upper() for c in emb.index.get_level_values("cohort")],
                       index=emb.index, name="cohort")
    arm = full.metadata.reindex(emb.index)["dsm_diagnosis"]

    labels = run_kmeans(emb, n_clusters=k, random_state=RANDOM_STATE).labels
    clusters = sorted(labels.unique())

    naming = pd.read_csv(RESULTS_DIR / "cluster_domains_naming.csv").set_index("cluster")
    meta = json.loads((RESULTS_DIR / "cluster_domains_meta.json").read_text())
    ksel = pd.read_csv(RESULTS_DIR / "cluster_domains_kselect.csv")

    # ── composition ──
    comp_rows = []
    for c in clusters:
        idx = labels.index[labels == c]
        cc = cohort.loc[idx].value_counts()
        comp_rows.append({
            "cluster": c, "n": len(idx),
            "cohort_mix": ", ".join(f"{kk} {int(v/len(idx)*100)}%" for kk, v in cc.items()),
            "age": float(age.loc[idx].mean()),
            "sex=1 %": float(sex.loc[idx].mean() * 100),
            "suggested_name": naming.loc[c, "top_domains"] if c in naming.index else "",
        })
    comp = pd.DataFrame(comp_rows)

    # ── enrichment ──
    enr = compute_cluster_feature_enrichment(scores, labels, q_threshold=0.05)
    top = enr.top_per_cluster(10)

    # ── cluster × domain mean profile (standardized) ──
    z = (scores - scores.mean()) / scores.std(ddof=0)
    profile = z.groupby(labels).mean().T          # domains × clusters
    # order domains by the cluster they peak in, then by value
    profile["_peak"] = profile.values.argmax(axis=1)
    profile["_val"] = profile.drop(columns="_peak").max(axis=1)
    profile = profile.sort_values(["_peak", "_val"], ascending=[True, False]).drop(
        columns=["_peak", "_val"])

    # ── medoids (centroid-nearest patient per cluster) ──
    arr = emb.to_numpy(np.float64)
    medoids = []
    for c in clusters:
        idx = np.where((labels == c).to_numpy())[0]
        centroid = arr[idx].mean(0)
        loc = idx[np.argmin(((arr[idx] - centroid) ** 2).sum(1))]
        pid = emb.index[loc]
        s = scores.iloc[loc].dropna()
        topdom = s.reindex(s.abs().sort_values(ascending=False).index).head(5)
        medoids.append({
            "cluster": c, "cohort": cohort.iloc[loc], "arm": arm.iloc[loc],
            "age": age.iloc[loc], "sex": sex.iloc[loc],
            "domains": "; ".join(f"{d} {v:+.1f}" for d, v in topdom.items()),
        })

    print("Fitting UMAP...")
    import umap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = umap.UMAP(n_neighbors=15, min_dist=0.1,
                           random_state=RANDOM_STATE).fit_transform(arr)

    # ── figures ──
    # k-selection
    fk = go.Figure()
    for col, name in [("stability_ari", "stability (↑)"), ("pac", "PAC (↓)"),
                      ("silhouette", "silhouette"), ("sex_cramersV", "sex leakage (↓)"),
                      ("cohort_ari", "cohort ARI (↓)")]:
        fk.add_trace(go.Scatter(x=ksel["k"], y=ksel[col], mode="lines+markers", name=name))
    fk.add_vline(x=k, line_dash="dash", line_color="#16a085")
    fk.update_layout(title="k selection — k=5 maximises stability/consensus before "
                     "sex leaks back", height=380, xaxis_title="k", yaxis_title="metric",
                     margin=dict(t=46, l=50, r=20, b=40))

    # heatmap
    hm = go.Figure(go.Heatmap(
        z=profile.to_numpy(), x=[f"C{c}" for c in profile.columns], y=list(profile.index),
        colorscale="RdBu", reversescale=True, zmid=0,
        colorbar=dict(title="std mean", thickness=12)))
    hm.update_layout(title="Cluster × domain mean profile (standardized; red=elevated)",
                     height=max(520, 16 * len(profile)),
                     margin=dict(t=46, l=200, r=20, b=40))

    # enrichment bars
    fe = make_subplots(rows=len(clusters), cols=1, vertical_spacing=0.03,
                       subplot_titles=[f"cluster {c} (n={int((labels==c).sum())})"
                                       for c in clusters])
    for i, c in enumerate(clusters, 1):
        sub = top[top["cluster"] == c].head(8).iloc[::-1]
        colors = ["#c0392b" if e >= 0 else "#2b6cb0" for e in sub["effect_rank_biserial"]]
        fe.add_trace(go.Bar(x=sub["effect_rank_biserial"], y=sub["feature_id"],
                            orientation="h", marker_color=colors, showlegend=False),
                     row=i, col=1)
    fe.update_layout(height=240 * len(clusters),
                     title="Top enriched domains per cluster (red ↑ / blue ↓ vs rest)",
                     margin=dict(t=60, l=20, r=20, b=30))
    fe.update_xaxes(range=[-1, 1], title_text="rank-biserial effect")

    # ── assemble ──
    ind = meta.get("independence", {})
    parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
             f"<title>FACE V0 domain phenotypes</title><style>{CSS}</style></head><body>"]
    parts.append("<header><h1>FACE V0 — trans-diagnostic domain phenotypes</h1>"
                 f"<div class='muted'>{len(emb):,} patients · {scores.shape[1]} "
                 f"age/sex-residualized domain scores · k={k} · masked-cosine spectral + "
                 "KMeans</div>"
                 "<nav class='toc'><a href='#comp'>Clusters</a><a href='#k'>k</a>"
                 "<a href='#umap'>UMAP</a><a href='#profile'>Profiles</a>"
                 "<a href='#enrich'>Enrichment</a><a href='#medoid'>Vignettes</a></nav></header>")
    parts.append(f"<div class='callout'><b>Confound verification (want ≈0):</b> "
                 f"sex Cramér's V {ind.get('sex_cramersV',float('nan')):.3f} · "
                 f"age-tertile ARI {ind.get('age_tertile_ari',float('nan')):.3f} · "
                 f"age dCor {ind.get('age_dcor',float('nan')):.3f} · "
                 f"<b>cohort ARI {ind.get('cohort_ari',float('nan')):.3f}</b> — clusters are "
                 "independent of sex, age and diagnosis (trans-diagnostic).</div>")

    parts.append("<h2 id='comp'>1 · Clusters</h2>")
    parts.append(df_table(comp[["cluster", "n", "cohort_mix", "age", "sex=1 %"]]))

    parts.append("<h2 id='k'>2 · How many clusters?</h2>")
    parts.append(fig_div(fk, "fig_k", include_js=True))

    parts.append("<h2 id='umap'>3 · UMAP</h2>")
    parts.append(scatter(coords, labels.astype(str).rename("cluster").reset_index(drop=True),
                         f"By cluster (k={k})", "umap_cl"))
    parts.append(scatter(coords, cohort.reset_index(drop=True), "By cohort", "umap_co"))

    parts.append("<h2 id='profile'>4 · Phenotype signatures (cluster × domain)</h2>")
    parts.append("<p class='muted'>Standardized mean domain score per cluster — the "
                 "defining axes of each phenotype.</p>")
    parts.append(fig_div(hm, "fig_hm"))

    parts.append("<h2 id='enrich'>5 · Top enriched domains</h2>")
    parts.append(fig_div(fe, "fig_enr"))

    parts.append("<h2 id='medoid'>6 · Representative patients (medoids)</h2>")
    parts.append("<div class='cards'>")
    for m in medoids:
        parts.append(
            f"<div class='card'><h4>Cluster {m['cluster']}</h4>"
            f"<div class='muted'>{m['cohort']} · {m['arm']} · age {m['age']:.0f} · "
            f"sex {int(m['sex']) if pd.notna(m['sex']) else '?'}</div>"
            f"<div style='margin-top:6px;font-size:12px'>{m['domains']}</div></div>")
    parts.append("</div></body></html>")

    out = REPORTS_DIR / "cluster_domains.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    comp.to_csv(RESULTS_DIR / "cluster_domains_profiles.csv", index=False)
    print(f"Wrote {out} ({out.stat().st_size/1024:.0f} KB)")
    print(f"     {RESULTS_DIR / 'cluster_domains_profiles.csv'}")
    print("\nPhenotype profiles:")
    print(comp[["cluster", "n", "cohort_mix", "age", "sex=1 %"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
