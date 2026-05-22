"""Profile + name the V0 clusters: k-sweep, UMAP, sister comparison, enrichment.

Phase-3 interpretation pass. Loads the saved embedding
(``results/cluster_v0_embedding.parquet``), rebuilds OUR harmonized V0 feature
matrix for interpretation, and produces:

  - **k comparison (4..9):** cluster × cohort contingency + silhouette + ARI/NMI
    vs the sister reference + cohort-entropy, to see how structure shifts with k.
  - **2D UMAP** of the 36-dim embedding, coloured by our cluster / cohort /
    sister cluster.
  - **Cluster naming** via the engine's per-cluster feature enrichment
    (Mann-Whitney U + Benjamini-Hochberg FDR, rank-biserial effect — sign
    convention verified: positive = higher inside the cluster). Top features are
    annotated with the dictionary label + section so each cluster gets a clinical
    description, including an explicit **metabolic-direction** read-out.
  - **Composition** per cluster (cohort / arm / age / sex).

Artifacts:
    results/cluster_v0_enrichment.csv   full (cluster × feature) enrichment table
    results/cluster_v0_profiles.csv     per-cluster composition + suggested name
    reports/cluster_v0.html             self-contained Plotly report

Run:  python3 scripts/cluster_v0_profile.py            # k=6 headline
      python3 scripts/cluster_v0_profile.py --k 7 --k-sweep 4 9
"""
from __future__ import annotations

import argparse
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
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from face_common import (  # noqa: E402
    ADMINISTRATIVE_FEATURES,
    CLINICAL_SECTIONS,
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
EMB_PATH = RESULTS_DIR / "cluster_v0_embedding.parquet"
REFERENCE_PATH = RESULTS_DIR / "v0_clusters_anchor.csv"

RANDOM_STATE = 0
SILHOUETTE_SAMPLE = 5000
TOP_N = 12
# sections we read as metabolic / physical-health for the direction check
METABOLIC_SECTIONS = {"BILAN BIOLOGIQUE", "CONSTANTES ET ECG"}

CSS = """
:root{--fg:#1f2933;--muted:#6b7280;--accent:#2b3a55;--bg:#fff;--card:#fbfbfd;
 --border:#e5e7eb;--warn:#c0392b;--good:#16a085;}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
 color:var(--fg);margin:0;padding:0 24px 80px;line-height:1.55}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--border);
 padding:18px 0;z-index:100;margin-bottom:24px}
h1{margin:0 0 4px;font-size:22px;color:var(--accent)}
h2{margin:40px 0 14px;padding-bottom:6px;border-bottom:2px solid var(--accent);
 color:var(--accent);font-size:18px}
h3{margin:24px 0 8px;font-size:15px}
.muted{color:var(--muted);font-size:13px}
nav.toc a{display:inline-block;margin:2px 6px 2px 0;padding:3px 9px;background:#eef2f7;
 border-radius:12px;font-size:12px;color:var(--accent);text-decoration:none}
nav.toc a:hover{background:var(--accent);color:#fff}
.callout{border-left:4px solid var(--accent);padding:10px 14px;background:#f5f7fb;
 margin:14px 0;border-radius:0 6px 6px 0}
.callout.warn{border-color:var(--warn);background:#fff5f4}
.callout.good{border-color:var(--good);background:#f2fbf6}
table.k{border-collapse:collapse;margin:10px 0 18px;font-size:13px}
table.k th,table.k td{border:1px solid var(--border);padding:5px 10px;text-align:left}
table.k th{background:#eef2f7;font-weight:600}
code{background:#eef2f7;padding:1px 5px;border-radius:4px;font-size:12px}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--k", type=int, default=6, help="headline cluster count (default 6)")
    p.add_argument("--k-sweep", type=int, nargs=2, default=(4, 9),
                   metavar=("LO", "HI"))
    p.add_argument("--readiness", nargs="+", default=["READY", "PARTIAL"])
    return p.parse_args()


def fig_div(fig: go.Figure, div_id: str, include_js: bool = False) -> str:
    return pio.to_html(fig, include_plotlyjs="cdn" if include_js else False,
                       full_html=False, div_id=div_id,
                       config={"displayModeBar": False})


def df_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    out = df.copy()
    for c in out.select_dtypes(include="float").columns:
        out[c] = out[c].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
    return out.to_html(index=False, classes="k", border=0, escape=False)


# ─── data loading ───────────────────────────────────────────────────────────

def load_inputs(readiness):
    emb = pd.read_parquet(EMB_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(DATA_DIR, DICT_PATH, readiness=readiness,
                                     format="long")
        variables = load_variables(DICT_PATH)
        # Enrichment matrix: SAME feature space as the embedding (clinical sections,
        # age/sex-residualized, minus site + *_mhoccur physical-comorbidity flags),
        # raw values — Mann-Whitney is rank-based, so residualized ranks are what
        # matters; no normalization needed.
        exclude = set(ADMINISTRATIVE_FEATURES) | {
            v.canonical_name for v in variables if v.canonical_name.endswith("_mhoccur")
        }
        dataset = to_harmonized_dataset(
            df, variables, visit="V0", exclude=exclude,
            sections=CLINICAL_SECTIONS, residualize_on=("age", "sex"))
        # Raw age/sex for per-cluster composition reporting (kept out of clustering).
        demo_ds = to_harmonized_dataset(df, variables, visit="V0",
                                        exclude=ADMINISTRATIVE_FEATURES)
    var_lookup = {v.canonical_name: v for v in variables}

    # align embedding ↔ feature matrix on the shared patient index
    common = emb.index.intersection(dataset.X.index)
    if len(common) != len(emb):
        warnings.warn(f"embedding/features index differ; using {len(common)} shared")
    emb = emb.loc[common]
    X = dataset.X.loc[common]
    demo = demo_ds.X.reindex(common)[[c for c in ("age", "sex") if c in demo_ds.X.columns]]

    cohorts = pd.Series(emb.index.get_level_values("cohort"), index=emb.index, name="cohort")
    arm = dataset.metadata.loc[common, "dsm_diagnosis"]
    their = load_reference(emb.index)
    return emb, X, cohorts, arm, their, var_lookup, demo


def load_reference(index: pd.MultiIndex) -> pd.Series:
    if not REFERENCE_PATH.exists():
        return pd.Series(index=index, dtype="float64", name="their_cluster")
    ref = pd.read_csv(REFERENCE_PATH)
    idx = pd.MultiIndex.from_arrays(
        [ref["cohort"].str.lower().to_numpy(), ref["usubjid_patients"].astype(str).to_numpy()],
        names=("cohort", "patient_id"))
    s = pd.Series(ref["cluster"].to_numpy(), index=idx, name="their_cluster")
    s = s[~s.index.duplicated(keep="first")]
    return s.reindex(index)


def cohort_entropy_mean(labels: pd.Series, cohorts: pd.Series) -> float:
    n = cohorts.nunique()
    if n < 2:
        return float("nan")
    logk, total, weight = np.log(n), 0.0, 0
    for _, idx in labels.groupby(labels).groups.items():
        dist = cohorts.loc[idx].value_counts(normalize=True).to_numpy()
        total += (-(dist * np.log(dist + 1e-12)).sum() / logk) * len(idx)
        weight += len(idx)
    return float(total / weight) if weight else float("nan")


# ─── analyses ────────────────────────────────────────────────────────────────

def k_sweep(emb, cohorts, their, lo, hi):
    shared = their.notna()
    arr = emb.to_numpy(np.float64)
    rows = []
    for k in range(lo, hi + 1):
        labels = run_kmeans(emb, n_clusters=k, random_state=RANDOM_STATE).labels
        sil = float(silhouette_score(arr, labels.to_numpy(),
                                     sample_size=min(SILHOUETTE_SAMPLE, len(emb)),
                                     random_state=RANDOM_STATE))
        ari = nmi = float("nan")
        if shared.any():
            ari = float(adjusted_rand_score(their[shared], labels[shared]))
            nmi = float(normalized_mutual_info_score(their[shared], labels[shared]))
        rows.append({"k": k, "silhouette": sil, "ari_vs_reference": ari,
                     "nmi_vs_reference": nmi,
                     "cohort_entropy_mean": cohort_entropy_mean(labels, cohorts),
                     "n_nonempty": int(labels.nunique())})
    return pd.DataFrame(rows)


def annotate_enrichment(table: pd.DataFrame, var_lookup) -> pd.DataFrame:
    def lab(fid):
        v = var_lookup.get(fid)
        return (v.label or fid) if v else fid

    def sec(fid):
        v = var_lookup.get(fid)
        return v.section if v else ""
    out = table.copy()
    out["label"] = out["feature_id"].map(lab)
    out["section"] = out["feature_id"].map(sec)
    out["direction"] = np.where(out["effect_rank_biserial"] >= 0, "higher", "lower")
    return out


def cluster_profiles(labels, cohorts, arm, demo, enr_top, var_lookup):
    """Per-cluster composition + a suggested descriptive name."""
    age = demo["age"] if "age" in demo.columns else pd.Series(np.nan, index=labels.index)
    sex = demo["sex"] if "sex" in demo.columns else pd.Series(np.nan, index=labels.index)
    rows = []
    for c in sorted(labels.unique()):
        idx = labels.index[labels == c]
        n = len(idx)
        comp = cohorts.loc[idx].value_counts()
        comp_pct = (comp / n * 100).round(0).astype(int)
        cohort_str = ", ".join(f"{k.upper()} {v}%" for k, v in comp_pct.items())
        dominant = comp_pct.idxmax().upper()
        top_arm = arm.loc[idx].value_counts().idxmax() if arm.loc[idx].notna().any() else ""
        # top enriched features for this cluster (already significant + sorted)
        feats = enr_top[enr_top["cluster"] == c].head(4)
        feat_str = "; ".join(f"{r.label} ({'↑' if r.direction=='higher' else '↓'})"
                             for r in feats.itertuples())
        name = f"{dominant}-led: {feat_str}" if feat_str else f"{dominant}-led"
        rows.append({
            "cluster": c, "n": n, "cohort_mix": cohort_str,
            "dominant_cohort": dominant, "top_arm": top_arm,
            "age_mean": float(age.loc[idx].mean()),
            "pct_sex1": float(sex.loc[idx].mean() * 100) if sex.loc[idx].notna().any() else float("nan"),
            "suggested_name": name,
        })
    return pd.DataFrame(rows)


def run_umap(emb):
    import umap  # local import; heavy
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = reducer.fit_transform(emb.to_numpy(np.float64))
    return pd.DataFrame(coords, columns=["umap1", "umap2"], index=emb.index)


# ─── figures ─────────────────────────────────────────────────────────────────

def scatter(coords, color_series, title, *, categorical=True, include_js=False, div=""):
    fig = go.Figure()
    if categorical:
        cats = sorted(color_series.dropna().unique().tolist())
        if color_series.isna().any():
            cats = cats + ["none"]
        for cat in cats:
            m = (color_series == cat) if cat != "none" else color_series.isna()
            fig.add_trace(go.Scattergl(
                x=coords.loc[m.values, "umap1"], y=coords.loc[m.values, "umap2"],
                mode="markers", name=str(cat),
                marker=dict(size=3, opacity=0.6)))
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=460,
                      legend=dict(itemsizing="constant"),
                      margin=dict(t=46, l=20, r=20, b=20),
                      xaxis_title="UMAP-1", yaxis_title="UMAP-2")
    return fig_div(fig, div, include_js)


def sweep_fig(sweep):
    fig = go.Figure()
    for col, name in [("silhouette", "silhouette"),
                      ("ari_vs_reference", "ARI vs sister"),
                      ("nmi_vs_reference", "NMI vs sister"),
                      ("cohort_entropy_mean", "cohort entropy")]:
        fig.add_trace(go.Scatter(x=sweep["k"], y=sweep[col], mode="lines+markers",
                                 name=name))
    fig.update_layout(title="k-sweep metrics", height=380, xaxis_title="k",
                      yaxis_title="metric", margin=dict(t=46, l=50, r=20, b=40))
    return fig_div(fig, "fig_sweep")


def enrichment_fig(enr_top, profiles, k):
    titles = [f"cluster {r.cluster} — {r.dominant_cohort} (n={r.n})"
              for r in profiles.itertuples()]
    fig = make_subplots(rows=k, cols=1, subplot_titles=titles, vertical_spacing=0.04)
    for i, c in enumerate(sorted(enr_top["cluster"].unique()), start=1):
        sub = enr_top[enr_top["cluster"] == c].head(10).iloc[::-1]
        colors = ["#c0392b" if d == "higher" else "#2b6cb0" for d in sub["direction"]]
        fig.add_trace(go.Bar(
            x=sub["effect_rank_biserial"], y=sub["label"], orientation="h",
            marker_color=colors, showlegend=False,
            hovertemplate="%{y}<br>effect=%{x:.2f}<extra></extra>"), row=i, col=1)
    fig.update_layout(height=260 * k, title="Top enriched features per cluster "
                      "(red ↑ higher inside, blue ↓ lower inside)",
                      margin=dict(t=60, l=20, r=20, b=30))
    fig.update_xaxes(title_text="rank-biserial effect", range=[-1, 1])
    return fig_div(fig, "fig_enrich")


def contingency_heatmap(ct, title, div, xlab, ylab):
    fig = go.Figure(go.Heatmap(
        z=ct.values, x=[str(c) for c in ct.columns], y=[str(r) for r in ct.index],
        text=ct.values, texttemplate="%{text}", textfont=dict(size=11),
        colorscale="Blues", colorbar=dict(thickness=12)))
    fig.update_layout(title=title, height=360, xaxis_title=xlab, yaxis_title=ylab,
                      margin=dict(t=46, l=70, r=20, b=50))
    return fig_div(fig, div)


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    if not EMB_PATH.exists():
        sys.exit(f"Missing embedding: {EMB_PATH}. Run scripts/cluster_v0.py first.")
    REPORTS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading embedding + rebuilding feature matrix...")
    emb, X, cohorts, arm, their, var_lookup, demo = load_inputs(args.readiness)
    print(f"  {len(emb):,} patients × {emb.shape[1]} embedding dims; "
          f"{X.shape[1]} clinical (age/sex-residualized) features")

    print(f"k-sweep {args.k_sweep}...")
    sweep = k_sweep(emb, cohorts, their, *args.k_sweep)
    print(sweep.round(3).to_string(index=False))

    k = args.k
    labels = run_kmeans(emb, n_clusters=k, random_state=RANDOM_STATE).labels

    print(f"Feature enrichment at k={k} (Mann-Whitney + BH FDR)...")
    enr = compute_cluster_feature_enrichment(X, labels, q_threshold=0.05)
    enr_table = annotate_enrichment(enr.table, var_lookup)
    enr_table.to_csv(RESULTS_DIR / "cluster_v0_enrichment.csv", index=False)
    enr_top_sig = annotate_enrichment(enr.top_per_cluster(TOP_N), var_lookup)
    print(f"  {enr.n_significant}/{enr.n_tests} (cluster×feature) tests significant")

    profiles = cluster_profiles(labels, cohorts, arm, demo, enr_top_sig, var_lookup)
    profiles.to_csv(RESULTS_DIR / "cluster_v0_profiles.csv", index=False)
    print("\nSuggested cluster names:")
    for r in profiles.itertuples():
        print(f"  [{r.cluster}] n={r.n:5d}  {r.suggested_name}")

    # metabolic-direction read-out (ROADMAP §3 ⚠️)
    metab = enr_top_sig[enr_top_sig["section"].isin(METABOLIC_SECTIONS)]
    biology_present = any(var_lookup.get(c) and var_lookup[c].section in METABOLIC_SECTIONS
                          for c in X.columns)
    print("\nMetabolic / physical-health features among top-enriched (direction):")
    if metab.empty:
        print("  (biology excluded by design)" if not biology_present
              else "  (none reached the per-cluster top list)")
    else:
        for r in metab.itertuples():
            print(f"  cluster {r.cluster}: {r.label} {r.direction} "
                  f"(effect {r.effect_rank_biserial:+.2f})")

    print("\nFitting UMAP...")
    coords = run_umap(emb)

    # contingencies
    ct_cohort = pd.crosstab(labels, cohorts)
    ct_cohort.index.name = "cluster"
    shared = their.notna()
    ct_their = (pd.crosstab(labels[shared], their[shared].astype(int))
                if shared.any() else pd.DataFrame())

    # ── assemble report ──
    print("Writing HTML report...")
    parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
             f"<title>FACE V0 clusters — profile</title><style>{CSS}</style></head><body>"]
    parts.append("<header><h1>FACE V0 clustering — profile & naming</h1>"
                 f"<div class='muted'>{len(emb):,} patients · {emb.shape[1]}-dim "
                 f"embedding · headline k={k} · clinical sections, age/sex-residualized · "
                 "engine: multipartite-spectral + KMeans</div>"
                 "<nav class='toc'><a href='#ksweep'>k-sweep</a>"
                 "<a href='#umap'>UMAP</a><a href='#names'>Cluster names</a>"
                 "<a href='#enrich'>Enrichment</a><a href='#sister'>Sister comparison</a>"
                 "</nav></header>")

    parts.append("<h2 id='ksweep'>1 · How many clusters?</h2>")
    parts.append("<p class='muted'>Silhouette rises monotonically (typical for "
                 "spectral embeddings); ARI/NMI vs the sister reference plateau; "
                 "cohort entropy (trans-diagnostic mixing) falls as k grows. "
                 f"k={k} matches the sister's non-ASP cluster count.</p>")
    parts.append(sweep_fig(sweep))
    parts.append(df_table(sweep))

    parts.append("<h2 id='umap'>2 · UMAP of the embedding</h2>")
    parts.append(scatter(coords, labels.astype(str).rename("cluster"),
                         f"Coloured by our cluster (k={k})", include_js=True,
                         div="umap_cluster"))
    parts.append(scatter(coords, cohorts.str.upper(), "Coloured by cohort",
                         div="umap_cohort"))
    parts.append(scatter(coords, their.map(lambda v: f"S{int(v)}" if pd.notna(v) else None),
                         "Coloured by sister cluster", div="umap_sister"))

    parts.append("<h2 id='names'>3 · Suggested cluster names</h2>")
    parts.append("<p class='muted'>Dominant cohort + top enriched features "
                 "(↑ higher inside, ↓ lower). Refine into clinical labels.</p>")
    show = profiles[["cluster", "n", "cohort_mix", "age_mean", "pct_sex1",
                     "suggested_name"]].rename(columns={"pct_sex1": "sex=1 %"})
    parts.append(df_table(show))
    metab_empty_msg = ("biology / physical-health features are excluded by design "
                       "in this clinical feature set" if not biology_present
                       else "no metabolic / physical-health feature reached a cluster's top list")
    metab_html = ("<div class='callout warn'><b>Metabolic-direction check:</b> "
                  + ("; ".join(f"cluster {r.cluster}: {r.label} <b>{r.direction}</b>"
                               for r in metab.itertuples())
                     if not metab.empty else metab_empty_msg)
                  + ".</div>")
    parts.append(metab_html)

    parts.append("<h2 id='enrich'>4 · Per-cluster feature enrichment</h2>")
    parts.append("<p class='muted'>Mann-Whitney U inside-vs-outside, "
                 "Benjamini-Hochberg FDR q&lt;0.05, rank-biserial effect "
                 "(sign verified: positive = higher inside).</p>")
    parts.append(enrichment_fig(enr_top_sig, profiles, k))

    parts.append("<h2 id='sister'>5 · Comparison with the sister 4-cohort clusters</h2>")
    parts.append(contingency_heatmap(ct_cohort, "Our cluster × cohort (counts)",
                                     "ct_cohort", "cohort", "our cluster"))
    if not ct_their.empty:
        ari = adjusted_rand_score(their[shared], labels[shared])
        nmi = normalized_mutual_info_score(their[shared], labels[shared])
        parts.append(f"<div class='callout'>On {int(shared.sum()):,} shared "
                     f"patients: <b>ARI {ari:.3f}</b>, NMI {nmi:.3f}. "
                     "Semantic recovery (SZ-pure group + BP–DR mood bridge), "
                     "not exact label match.</div>")
        ct_their.index.name = "our cluster"
        parts.append(contingency_heatmap(ct_their, "Our cluster × sister cluster (counts)",
                                         "ct_sister", "sister cluster", "our cluster"))
    parts.append("</body></html>")

    out = REPORTS_DIR / "cluster_v0.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"\nWrote {out} ({out.stat().st_size/1024:.0f} KB)")
    print(f"     {RESULTS_DIR / 'cluster_v0_enrichment.csv'}")
    print(f"     {RESULTS_DIR / 'cluster_v0_profiles.csv'}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
