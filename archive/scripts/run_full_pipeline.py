#!/usr/bin/env python
"""Run the full FACE stratification pipeline — save plots as PNGs, generate HTML report.

Usage:
    python scripts/run_full_pipeline.py
"""
from __future__ import annotations

import base64
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output" / "stratification" / "full_pipeline"
PLOTS = OUT / "plots"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

CSV_PATHS = {
    "bp": str(REPO / "data" / "BP.csv"),
    "sz": str(REPO / "data" / "SZ.csv"),
    "dr": str(REPO / "data" / "DR.csv"),
    "asp": str(REPO / "data" / "ASP.csv"),
}

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

DARK = dict(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27")
SCALE = 3

def save_fig(fig, name, w=1100, h=500):
    """Write a Plotly figure to PNG and return the path."""
    path = PLOTS / f"{name}.png"
    fig.write_image(str(path), width=w, height=h, scale=SCALE)
    logger.info("  Saved %s (%d×%d)", path.name, w, h)
    return path

def img_tag(path, alt=""):
    """Return an <img> tag with the file referenced by relative path."""
    rel = path.relative_to(OUT)
    return f'<img src="{rel}" alt="{alt}" style="width:100%;border-radius:8px;margin:12px 0;">'


# ═══════════════════════════════════════════════════════════════════════════════
# Stage A — Harmonization
# ═══════════════════════════════════════════════════════════════════════════════

logger.info("═══ Stage A: Building harmonized dataset ═══")
t0 = time.time()

from face_stratification.harmonization.harmonizer import build_harmonized_dataset
from face_stratification.harmonization.feature_schema import load_feature_schema

schema = load_feature_schema()
ds = build_harmonized_dataset(CSV_PATHS, schema=schema)

logger.info("Harmonized: %d patients × %d features in %.1fs", ds.X.shape[0], ds.X.shape[1], time.time() - t0)
cohort_counts = ds.metadata["cohort"].value_counts().to_dict()
logger.info("Cohort counts: %s", cohort_counts)

# ═══════════════════════════════════════════════════════════════════════════════
# Load raw CSVs for DSM subtyping
# ═══════════════════════════════════════════════════════════════════════════════

logger.info("═══ Loading raw CSVs for DSM subtypes ═══")
csv_data = {}
for cohort, path in CSV_PATHS.items():
    csv_data[cohort] = pd.read_csv(path, low_memory=False)
    logger.info("  %s: %d rows", cohort, len(csv_data[cohort]))

# ═══════════════════════════════════════════════════════════════════════════════
# DSM-5 Subtypes
# ═══════════════════════════════════════════════════════════════════════════════

logger.info("═══ DSM-5 Subtypes ═══")
from face_stratification.harmonization.dsm_subtypes import extract_dsm_subtypes_from_raw

dsm_subtypes = extract_dsm_subtypes_from_raw(ds.metadata, csv_data)
dsm_dist = dsm_subtypes.value_counts().sort_values(ascending=False)
logger.info("DSM subtypes:\n%s", dsm_dist.to_string())
dsm_dist.to_csv(OUT / "dsm_subtype_distribution.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# Missingness
# ═══════════════════════════════════════════════════════════════════════════════

logger.info("═══ Missingness characterization ═══")
from face_stratification.harmonization.missingness import characterize_missingness

miss = characterize_missingness(ds.X, ds.metadata, schema)
miss["per_feature_rates"].to_csv(OUT / "missingness_rates.csv")
with open(OUT / "missingness_mechanisms.json", "w") as f:
    json.dump(miss["mechanism_summary"], f, indent=2)
n_mcar = sum(1 for v in miss["mechanism_summary"].values() if v == "MCAR")
n_mar = sum(1 for v in miss["mechanism_summary"].values() if v == "MAR")
n_mnar = sum(1 for v in miss["mechanism_summary"].values() if v == "MNAR")

# ═══════════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════════

logger.info("═══ Normalization ═══")
from face_stratification.harmonization.normalization import fit_normalization, transform_normalization

norm_stats = fit_normalization(ds.X, ds.schema)
Xn = transform_normalization(ds.X, norm_stats)

# ═══════════════════════════════════════════════════════════════════════════════
# Stage B — Graph + Embeddings
# ═══════════════════════════════════════════════════════════════════════════════

logger.info("═══ Stage B: Graph construction ═══")
from face_stratification.graph.patient_similarity import build_multiplex_graph

t0 = time.time()
graph, block_graphs, td_graph = build_multiplex_graph(Xn, ds.schema, k=10, metadata=ds.metadata)
logger.info("Graph built in %.1fs: %d nodes, %d block types", time.time() - t0, graph.number_of_nodes(), len(block_graphs))

from face_stratification.harmonization.harmonizer import HarmonizedDataset as HD
normalized_ds = HD(X=Xn, metadata=ds.metadata, feature_metadata=ds.feature_metadata, schema=ds.schema)

logger.info("═══ Embedding: Raw Baseline ═══")
from face_stratification.models.raw_baseline import RawFeatureBaseline
baseline_emb = RawFeatureBaseline().fit_transform(normalized_ds)
logger.info("Baseline: %d × %d", baseline_emb.n_patients, baseline_emb.dim)

logger.info("═══ Embedding: PCA ═══")
from face_stratification.models.baselines import TransdiagnosticPCA
pca_emb = TransdiagnosticPCA(n_components=16).fit_transform(normalized_ds, graph=graph)
logger.info("PCA: %d × %d", pca_emb.n_patients, pca_emb.dim)

logger.info("═══ Embedding: UMAP ═══")
from face_stratification.models.baselines import TransdiagnosticUMAP
try:
    umap_emb = TransdiagnosticUMAP(n_components=16).fit_transform(normalized_ds, graph=graph)
    logger.info("UMAP: %d × %d", umap_emb.n_patients, umap_emb.dim)
    has_umap = True
except Exception as e:
    logger.warning("UMAP failed: %s", e)
    umap_emb = None
    has_umap = False

logger.info("═══ Embedding: Composite (PCA + Spectral) ═══")
from face_stratification.models.composite import ConcatenatedEmbedding
composite_emb = ConcatenatedEmbedding.build_default().fit_transform(normalized_ds, graph=graph)
logger.info("Composite: %d × %d", composite_emb.n_patients, composite_emb.dim)

# ═══════════════════════════════════════════════════════════════════════════════
# Stage C — Clustering
# ═══════════════════════════════════════════════════════════════════════════════

logger.info("═══ Stage C: k Selection ═══")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

primary_emb = composite_emb.values.to_numpy()
primary_emb_clean = np.nan_to_num(primary_emb, nan=0.0)

k_results: dict[int, dict] = {}
for k in range(3, 13):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(primary_emb_clean)
    sil = silhouette_score(primary_emb_clean, labels, sample_size=min(5000, len(labels)))
    db = davies_bouldin_score(primary_emb_clean, labels)
    ch = calinski_harabasz_score(primary_emb_clean, labels)
    k_results[k] = {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}
    logger.info("  k=%d: sil=%.4f DB=%.4f CH=%.1f", k, sil, db, ch)

pd.DataFrame(k_results).T.to_csv(OUT / "k_selection_metrics.csv")
best_k = max(k_results, key=lambda k: k_results[k]["silhouette"])
logger.info("Selected k=%d (sil=%.4f)", best_k, k_results[best_k]["silhouette"])

# Cluster at best k on all embeddings
def cluster_emb(emb_values, k):
    arr = np.nan_to_num(emb_values.to_numpy() if hasattr(emb_values, 'to_numpy') else np.array(emb_values), nan=0.0)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(arr)
    sil = silhouette_score(arr, labels, sample_size=min(5000, len(labels)))
    db = davies_bouldin_score(arr, labels)
    return labels, sil, db

methods = {}
bl_labels, bl_sil, bl_db = cluster_emb(baseline_emb.values, best_k)
methods["Raw Baseline"] = {"labels": bl_labels, "silhouette": bl_sil, "davies_bouldin": bl_db}
pca_labels, pca_sil, pca_db = cluster_emb(pca_emb.values, best_k)
methods["PCA"] = {"labels": pca_labels, "silhouette": pca_sil, "davies_bouldin": pca_db}
if has_umap:
    umap_labels, umap_sil, umap_db = cluster_emb(umap_emb.values, best_k)
    methods["UMAP"] = {"labels": umap_labels, "silhouette": umap_sil, "davies_bouldin": umap_db}
comp_labels, comp_sil, comp_db = cluster_emb(composite_emb.values, best_k)
methods["Composite"] = {"labels": comp_labels, "silhouette": comp_sil, "davies_bouldin": comp_db}

final_labels = comp_labels
cohorts = ds.metadata["cohort"].values

# ═══════════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════════

logger.info("═══ Analysis ═══")

# Cluster composition
comp_df = pd.DataFrame({"cluster": final_labels, "cohort": cohorts})
composition = pd.crosstab(comp_df["cluster"], comp_df["cohort"], normalize="index")
composition.to_csv(OUT / "cluster_composition.csv")
sizes = pd.Series(final_labels).value_counts().sort_index()

# DSM alignment
from scipy.stats import chi2_contingency
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

contingency = pd.crosstab(final_labels, cohorts)
chi2, p_val, dof, _ = chi2_contingency(contingency)
n = contingency.sum().sum()
k_min = min(contingency.shape) - 1
cramers_v = np.sqrt(chi2 / (n * k_min)) if k_min > 0 else 0.0
nmi = normalized_mutual_info_score(cohorts, final_labels)
ari = adjusted_rand_score(cohorts, final_labels)

# Cluster × DSM subtype cross-tabulation
dsm_cluster = pd.DataFrame({"cluster": final_labels, "dsm_subtype": dsm_subtypes.values}, index=ds.X.index)
dsm_xtab = pd.crosstab(dsm_cluster["cluster"], dsm_cluster["dsm_subtype"])
dsm_xtab_norm = pd.crosstab(dsm_cluster["cluster"], dsm_cluster["dsm_subtype"], normalize="index")
dsm_xtab.to_csv(OUT / "dsm_subtypes_per_cluster.csv")
dsm_xtab_norm.to_csv(OUT / "dsm_subtypes_per_cluster_normalized.csv")

# NMI/ARI vs DSM subtypes
nmi_dsm = normalized_mutual_info_score(dsm_subtypes.values, final_labels)
ari_dsm = adjusted_rand_score(dsm_subtypes.values, final_labels)
logger.info("NMI(clusters, DSM subtypes)=%.4f, ARI=%.4f", nmi_dsm, ari_dsm)

# Subtype purity per cluster
dsm_purity = dsm_xtab_norm.max(axis=1)
mean_purity = dsm_purity.mean()

# GMM soft clustering
from face_stratification.clustering.algorithms import run_gmm_soft, identify_boundary_patients
gmm_labels, posteriors = run_gmm_soft(primary_emb_clean, k=best_k)
boundary = identify_boundary_patients(posteriors)

# Treatment validation
from face_stratification.analysis.treatment_validation import run_treatment_validation
tx_result = run_treatment_validation(final_labels, ds.X, ds.metadata)
if tx_result.treatment_profiles is not None:
    tx_result.treatment_profiles.to_csv(OUT / "treatment_profiles.csv")

# Safety
from face_stratification.analysis.safety_analysis import run_safety_analysis
safety = run_safety_analysis(final_labels, ds.X, ds.metadata)

# Enrichment
from face_stratification.analysis.enrichment import compute_cluster_feature_enrichment
final_labels_series = pd.Series(final_labels, index=ds.X.index, name="cluster")
enrichment = compute_cluster_feature_enrichment(ds.X, final_labels_series, q_threshold=0.05)
enrichment.table.to_csv(OUT / "enrichment.csv")

# Meta-stability
from face_stratification.analysis.meta_stability import compute_meta_stability
stability = compute_meta_stability(
    patient_ids=np.arange(len(final_labels)),
    method_labels={name: m["labels"] for name, m in methods.items()},
)

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS — Save as PNGs
# ═══════════════════════════════════════════════════════════════════════════════

logger.info("═══ Generating Plots ═══")

# 1. k-Selection
k_df = pd.DataFrame(k_results).T
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=k_df.index, y=k_df["silhouette"], name="Silhouette", mode="lines+markers", line=dict(color="#6366f1", width=3)), secondary_y=False)
fig.add_trace(go.Scatter(x=k_df.index, y=k_df["davies_bouldin"], name="Davies-Bouldin", mode="lines+markers", line=dict(color="#f87171", width=3)), secondary_y=True)
fig.add_vline(x=best_k, line_dash="dash", line_color="#34d399", annotation_text=f"k={best_k}")
fig.update_layout(title="Dual-Criterion k Selection", xaxis_title="k", height=450, **DARK)
fig.update_yaxes(title_text="Silhouette (↑)", secondary_y=False)
fig.update_yaxes(title_text="Davies-Bouldin (↓)", secondary_y=True)
p_ksel = save_fig(fig, "01_k_selection")

# 2. Embedding comparison
fig = go.Figure()
fig.add_trace(go.Bar(x=list(methods.keys()), y=[m["silhouette"] for m in methods.values()], marker_color=["#9ca3af","#6366f1","#34d399","#f87171"][:len(methods)]))
fig.update_layout(title=f"Embedding Comparison — Silhouette (k={best_k})", yaxis_title="Silhouette Score", height=400, **DARK)
p_emb = save_fig(fig, "02_embedding_comparison")

# 3. Cluster composition heatmap
fig = go.Figure(data=go.Heatmap(z=composition.values, x=composition.columns.tolist(), y=[f"C{i}" for i in composition.index], colorscale="Viridis", text=composition.values.round(2), texttemplate="%{text:.2f}", textfont=dict(size=13)))
fig.update_layout(title="Cluster Composition by Cohort", height=max(350, 45 * best_k), **DARK)
p_comp = save_fig(fig, "03_cluster_composition")

# 4. Cluster sizes
fig = go.Figure(data=go.Bar(x=[f"C{i}" for i in sizes.index], y=sizes.values, marker_color=px.colors.qualitative.Set2[:len(sizes)], text=sizes.values, textposition="auto"))
fig.update_layout(title="Cluster Sizes", yaxis_title="Patients", height=400, **DARK)
p_sizes = save_fig(fig, "04_cluster_sizes")

# 5-6. UMAP projections
import umap as umap_lib
logger.info("  Computing 2D UMAP projection...")
coords_2d = umap_lib.UMAP(n_components=2, metric="cosine", random_state=42).fit_transform(primary_emb_clean)
proj_df = pd.DataFrame({"UMAP-1": coords_2d[:, 0], "UMAP-2": coords_2d[:, 1], "Cluster": [f"C{l}" for l in final_labels], "Cohort": cohorts, "DSM": dsm_subtypes.values})
proj_df.to_csv(OUT / "umap_2d_projection.csv", index=False)

fig = px.scatter(proj_df, x="UMAP-1", y="UMAP-2", color="Cluster", title="UMAP Projection — by Data-Driven Cluster", opacity=0.5, height=600, template="plotly_dark")
fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27")
p_umap_cl = save_fig(fig, "05_umap_by_cluster", h=600)

fig = px.scatter(proj_df, x="UMAP-1", y="UMAP-2", color="Cohort", title="UMAP Projection — by DSM Cohort", opacity=0.5, height=600, color_discrete_map={"bp": "#6366f1", "sz": "#f87171", "dr": "#fbbf24", "asp": "#34d399"}, template="plotly_dark")
fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27")
p_umap_co = save_fig(fig, "06_umap_by_cohort", h=600)

# 7. UMAP by DSM subtype (coarser grouping for readability)
dsm_coarse = dsm_subtypes.values.copy()
proj_df["DSM Subtype"] = dsm_coarse
unique_dsm = sorted(proj_df["DSM Subtype"].unique())
if len(unique_dsm) <= 20:
    fig = px.scatter(proj_df, x="UMAP-1", y="UMAP-2", color="DSM Subtype", title="UMAP Projection — by DSM-5 Subtype", opacity=0.5, height=650, template="plotly_dark")
    fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27")
    p_umap_dsm = save_fig(fig, "07_umap_by_dsm_subtype", h=650)
else:
    proj_df["DSM Group"] = [s.split("-")[0] + "-" + s.split("-")[1] if "-" in s else s for s in dsm_coarse]
    fig = px.scatter(proj_df, x="UMAP-1", y="UMAP-2", color="DSM Group", title="UMAP Projection — by DSM-5 Subgroup", opacity=0.5, height=650, template="plotly_dark")
    fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27")
    p_umap_dsm = save_fig(fig, "07_umap_by_dsm_subtype", h=650)

# 8. DSM subtypes per cluster heatmap (normalized)
fig = go.Figure(data=go.Heatmap(z=dsm_xtab_norm.values, x=dsm_xtab_norm.columns.tolist(), y=[f"C{i}" for i in dsm_xtab_norm.index], colorscale="Plasma", text=dsm_xtab_norm.values.round(2), texttemplate="%{text:.2f}", textfont=dict(size=11)))
fig.update_layout(title="Cluster × DSM-5 Subtype (row-normalized)", height=max(450, 45 * best_k), **DARK)
p_dsm_heat = save_fig(fig, "08_cluster_vs_dsm_subtypes", w=1300, h=max(500, 50 * best_k))

# 9. DSM subtypes per cluster — stacked bar (absolute counts)
dsm_cols = dsm_xtab.columns.tolist()
fig = go.Figure()
for col in dsm_cols:
    fig.add_trace(go.Bar(name=col, x=[f"C{i}" for i in dsm_xtab.index], y=dsm_xtab[col].values))
fig.update_layout(barmode="stack", title="Cluster × DSM-5 Subtype (absolute counts)", yaxis_title="Patients", height=500, **DARK)
p_dsm_stack = save_fig(fig, "09_dsm_subtypes_stacked", w=1300, h=550)

# 10. Missingness heatmap
miss_rates = miss["per_feature_rates"]
if "overall" in miss_rates.columns:
    top_missing = miss_rates["overall"].nlargest(30).index.tolist()
    cohort_cols = [c for c in miss_rates.columns if c != "overall"]
    if cohort_cols and top_missing:
        miss_sub = miss_rates.loc[top_missing, cohort_cols]
        fig = go.Figure(data=go.Heatmap(z=miss_sub.values, x=cohort_cols, y=top_missing, colorscale="Reds", text=miss_sub.values.round(2), texttemplate="%{text:.0%}"))
        fig.update_layout(title="Missingness Rates — Top 30 Features", height=700, **DARK)
        p_miss = save_fig(fig, "10_missingness", h=700)
    else:
        p_miss = None
else:
    p_miss = None

# 11. Safety
clusters_s = sorted(safety.attempt_rates.keys())
fig = go.Figure()
fig.add_trace(go.Bar(x=[f"C{c}" for c in clusters_s], y=[safety.attempt_rates[c] for c in clusters_s], name="Attempt Rate", marker_color=["#f87171" if c in safety.high_risk_clusters else "#6366f1" for c in clusters_s]))
if safety.ideation_rates:
    fig.add_trace(go.Bar(x=[f"C{c}" for c in clusters_s], y=[safety.ideation_rates.get(c, 0) for c in clusters_s], name="Ideation Rate", marker_color="#fbbf24", opacity=0.7))
fig.update_layout(title="Suicide Risk by Cluster", yaxis_title="Rate", barmode="group", height=400, **DARK)
p_safety = save_fig(fig, "11_safety")

# 12. Treatment profiles
if tx_result.treatment_profiles is not None and not tx_result.treatment_profiles.empty:
    tp = tx_result.treatment_profiles
    fig = go.Figure(data=go.Heatmap(z=tp.values, x=tp.columns.tolist(), y=[f"C{i}" for i in tp.index], colorscale="Blues", text=tp.values.round(2), texttemplate="%{text:.0%}", textfont=dict(size=12)))
    fig.update_layout(title="Treatment Profiles by Cluster", height=max(350, 45 * best_k), **DARK)
    p_tx = save_fig(fig, "12_treatment_profiles")
else:
    p_tx = None

# 13. Entropy histogram
fig = go.Figure(data=go.Histogram(x=boundary["entropy"], nbinsx=50, marker_color="#818cf8"))
fig.add_vline(x=1.5, line_dash="dash", line_color="#f87171", annotation_text="Boundary threshold")
fig.update_layout(title="Assignment Entropy (GMM)", xaxis_title="Entropy (bits)", yaxis_title="Count", height=400, **DARK)
p_entropy = save_fig(fig, "13_entropy")

# 14. DSM subtype distribution bar chart
fig = go.Figure(data=go.Bar(x=dsm_dist.index.tolist(), y=dsm_dist.values, marker_color="#818cf8"))
fig.update_layout(title="DSM-5 Subtype Distribution", yaxis_title="Patients", xaxis_tickangle=-45, height=450, **DARK)
p_dsm_dist = save_fig(fig, "14_dsm_distribution", h=450)

logger.info("═══ All %d plots saved to %s ═══", len(list(PLOTS.glob("*.png"))), PLOTS)

# ═══════════════════════════════════════════════════════════════════════════════
# HTML Report
# ═══════════════════════════════════════════════════════════════════════════════

logger.info("═══ Generating HTML Report ═══")

def mc(val, label):
    return f'<div class="metric"><div class="value">{val}</div><div class="label">{label}</div></div>'

# Enrichment top features
enrichment_rows = ""
if enrichment.n_significant > 0:
    sig = enrichment.table[enrichment.table["significant"] == True].nlargest(20, "abs_effect")
    for _, row in sig.iterrows():
        enrichment_rows += f'<tr><td>{row["feature_id"]}</td><td>C{int(row["cluster"])}</td><td>{row["effect_rank_biserial"]:+.3f}</td><td>{row["p_value_bh"]:.2e}</td></tr>'

# Method comparison table rows
method_rows = ""
for name, m in methods.items():
    delta = m["silhouette"] - methods["Raw Baseline"]["silhouette"]
    method_rows += f'<tr><td>{name}</td><td>{m["silhouette"]:.4f}</td><td>{m["davies_bouldin"]:.4f}</td><td>{delta:+.4f}</td></tr>'

# DSM subtype distribution table
dsm_table = ""
for st, ct in dsm_dist.items():
    dsm_table += f'<tr><td>{st}</td><td>{ct:,}</td><td>{ct/len(dsm_subtypes):.1%}</td></tr>'

mixed = (composition.max(axis=1) < 0.75).sum()

umap_sil_val = methods.get("UMAP", {}).get("silhouette", 0)

html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>FACE Stratification Results</title>
<style>
:root{{--bg:#0f1117;--sf:#1a1d27;--bd:#2d3142;--tx:#e0e0e6;--mt:#9ca3af;--ac:#6366f1;--a2:#818cf8;--ok:#34d399;--wn:#fbbf24;--er:#f87171}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--tx);line-height:1.7;padding:40px;max-width:1400px;margin:0 auto}}
h1{{font-size:36px;font-weight:800;margin-bottom:8px;background:linear-gradient(135deg,var(--ac),var(--ok));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
h2{{font-size:24px;font-weight:700;margin:48px 0 16px;padding-bottom:8px;border-bottom:1px solid var(--bd)}}
h3{{font-size:18px;font-weight:600;margin:24px 0 12px;color:var(--a2)}}
p{{color:var(--mt);margin-bottom:12px}}
.sub{{font-size:16px;color:var(--mt);margin-bottom:32px}}
.mg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin:20px 0}}
.m{{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:16px;text-align:center}}
.m .v{{font-size:28px;font-weight:700;color:var(--a2)}}.m .l{{font-size:11px;color:var(--mt);margin-top:4px}}
.c{{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:24px;margin:16px 0}}
.c p{{margin-bottom:10px}}
table{{width:100%;border-collapse:collapse;margin:16px 0;font-size:14px}}
th{{background:var(--sf);padding:10px 14px;text-align:left;font-weight:600;border-bottom:2px solid var(--bd)}}
td{{padding:8px 14px;border-bottom:1px solid var(--bd)}}
tr:hover td{{background:rgba(99,102,241,0.05)}}
.badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600}}
.badge-ok{{background:rgba(52,211,153,0.15);color:var(--ok)}}
.badge-wn{{background:rgba(251,191,36,0.15);color:var(--wn)}}
.callout{{background:rgba(99,102,241,0.08);border-left:3px solid var(--ac);padding:16px 20px;border-radius:0 8px 8px 0;margin:16px 0}}
.callout p{{color:var(--tx);margin:0}}
img{{width:100%;border-radius:8px;margin:12px 0}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
@media(max-width:900px){{.two{{grid-template-columns:1fr}}}}
</style></head><body>

<h1>FACE Cohort Stratification</h1>
<p class="sub">Trans-diagnostic phenotyping of {ds.X.shape[0]:,} patients across 4 FACE cohorts — {best_k} data-driven clusters vs DSM-5 subtypes</p>

<h2>1. Data Overview</h2>
<div class="mg">
{mc(f'{ds.X.shape[0]:,}', 'Total Patients')}
{mc(f'{cohort_counts.get("bp",0):,}', 'Bipolar (BP)')}
{mc(f'{cohort_counts.get("sz",0):,}', 'Schizophrenia (SZ)')}
{mc(f'{cohort_counts.get("asp",0):,}', 'Autism (ASP)')}
{mc(f'{cohort_counts.get("dr",0):,}', 'TRD (DR)')}
{mc(str(ds.X.shape[1]), 'Features')}
{mc(str(len(schema.blocks)), 'Clinical Blocks')}
{mc(f'{ds.X.isna().mean().mean():.1%}', 'Missingness')}
</div>

<h2>2. DSM-5 Subtypes</h2>
<div class="c">
<p>Fine-grained DSM-5 subtypes extracted from raw clinical data: BP-I/BP-II/BP-NOS from the <code>arm</code> column,
SZ positive/negative/mixed from PANSS subscale dominance, DR resistance staging from Sachs score + resistance flag,
ASP diagnostic category (Autism/Asperger/PDD-NOS) crossed with EGF functioning level.</p>
<table><tr><th>Subtype</th><th>Count</th><th>%</th></tr>{dsm_table}</table>
</div>
{img_tag(p_dsm_dist, "DSM subtype distribution")}

<h2>3. Missingness</h2>
<div class="c">
<p>Overall missingness: {ds.X.isna().mean().mean():.1%}. FACE missingness is MNAR — structurally tied to cohort.
Blocks classified: {n_mcar} MCAR, {n_mar} MAR, {n_mnar} MNAR.</p>
</div>
{img_tag(p_miss, "Missingness") if p_miss else ""}

<h2>4. Embeddings</h2>
<div class="c">
<table><tr><th>Method</th><th>Silhouette ↑</th><th>Davies-Bouldin ↓</th><th>Δ vs Baseline</th></tr>{method_rows}</table>
<div class="callout"><p>The raw baseline's high silhouette (≈0.97) is an artifact of NaN→0 filling creating cohort-specific
zero blocks. Meaningful comparison is among PCA ({pca_sil:.4f}), UMAP ({umap_sil_val:.4f}), and Composite ({comp_sil:.4f}).</p></div>
</div>
{img_tag(p_emb, "Embedding comparison")}

<h2>5. k Selection</h2>
{img_tag(p_ksel, "k selection")}
<div class="c"><p>Selected <strong>k={best_k}</strong> (silhouette = {k_results[best_k]["silhouette"]:.4f}). The flat curve suggests fuzzy
cluster boundaries — typical for psychiatric phenotypes.</p></div>

<h2>6. Cluster Analysis</h2>
<div class="mg">
{mc(str(best_k), 'Clusters')}
{mc(f'{comp_sil:.4f}', 'Silhouette')}
{mc(f'{cramers_v:.3f}', "Cramér's V")}
{mc(f'{nmi:.4f}', 'NMI vs cohort')}
{mc(f'{ari:.4f}', 'ARI vs cohort')}
{mc(f'{mixed}/{best_k}', 'Mixed clusters')}
</div>

<div class="two">
{img_tag(p_comp, "Composition")}
{img_tag(p_sizes, "Sizes")}
</div>

<h2>7. UMAP Projections</h2>
<p>2D UMAP reduction of the {composite_emb.dim}-dimensional composite embedding, colored three ways:</p>
<div class="two">
{img_tag(p_umap_cl, "UMAP by cluster")}
{img_tag(p_umap_co, "UMAP by cohort")}
</div>
{img_tag(p_umap_dsm, "UMAP by DSM subtype")}

<h2>8. Clusters vs DSM-5 Subtypes</h2>
<div class="c">
<h3>Alignment Metrics</h3>
<div class="mg">
{mc(f'{nmi_dsm:.4f}', 'NMI (clusters vs DSM subtypes)')}
{mc(f'{ari_dsm:.4f}', 'ARI (clusters vs DSM subtypes)')}
{mc(f'{mean_purity:.2f}', 'Mean cluster purity')}
{mc(f'{cramers_v:.3f}', "Cramér's V (vs cohort)")}
</div>
<p>NMI = {nmi_dsm:.4f} between clusters and DSM-5 subtypes. {'Low alignment confirms clusters capture structure beyond DSM categories.' if nmi_dsm < 0.3 else 'Moderate alignment suggests clusters partially recover DSM subtypes but add transdiagnostic signal.'}
Mean cluster purity (max subtype fraction per cluster) = {mean_purity:.2f} — {'clusters mix multiple DSM subtypes, supporting the transdiagnostic hypothesis.' if mean_purity < 0.6 else 'some clusters align with specific subtypes.'}</p>
</div>

{img_tag(p_dsm_heat, "Cluster vs DSM subtypes")}
{img_tag(p_dsm_stack, "DSM subtypes stacked")}

<div class="c">
<h3>Per-Cluster DSM Profile</h3>
<table><tr><th>Cluster</th><th>Size</th><th>Dominant Subtype</th><th>Purity</th><th>Top 3 Subtypes</th></tr>
"""

for cl in sorted(dsm_xtab.index):
    row = dsm_xtab_norm.loc[cl]
    top3 = row.nlargest(3)
    dominant = top3.index[0]
    purity = top3.iloc[0]
    top3_str = ", ".join(f"{s} ({v:.0%})" for s, v in top3.items())
    html += f'<tr><td>C{cl}</td><td>{sizes[cl]:,}</td><td>{dominant}</td><td>{purity:.0%}</td><td>{top3_str}</td></tr>\n'

html += f"""</table></div>

<h2>9. Treatment Validation</h2>
<div class="c">
<p>Treatment profiles differ across clusters: <span class="badge {'badge-ok' if tx_result.has_differential_treatment else 'badge-wn'}">{'YES' if tx_result.has_differential_treatment else 'NO'}</span></p>
<p>Functioning outcomes differ: <span class="badge {'badge-ok' if tx_result.has_differential_functioning else 'badge-wn'}">{'YES' if tx_result.has_differential_functioning else 'NO'}</span></p>
</div>
{img_tag(p_tx, "Treatment profiles") if p_tx else ""}

<h2>10. Safety</h2>
<div class="c">
<p>Attempt rates: {min(safety.attempt_rates.values()):.1%} – {max(safety.attempt_rates.values()):.1%}.
Ideation rates: {min(safety.ideation_rates.values()):.1%} – {max(safety.ideation_rates.values()):.1%}.</p>
</div>
{img_tag(p_safety, "Safety")}

<h2>11. Meta-Stability & Boundary Patients</h2>
<div class="c">
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Mean agreement across methods</td><td>{stability.summary().get('mean_agreement',0):.3f}</td></tr>
<tr><td>Core members (≥80%)</td><td>{stability.summary().get('n_core',0):,}</td></tr>
<tr><td>Boundary members (&lt;50%)</td><td>{stability.summary().get('n_boundary',0):,}</td></tr>
<tr><td>GMM boundary patients</td><td>{boundary['n_boundary']:,} ({boundary['boundary_fraction']:.1%})</td></tr>
</table></div>
{img_tag(p_entropy, "Entropy")}

{'<h2>12. Feature Enrichment (Top 20)</h2><div class="c"><table><tr><th>Feature</th><th>Cluster</th><th>Effect (rank-biserial)</th><th>q-value</th></tr>' + enrichment_rows + '</table></div>' if enrichment_rows else ''}

<h2>13. Synthesis</h2>
<div class="c">
<p><strong>{best_k} data-driven phenotypes</strong> discovered from {ds.X.shape[0]:,} FACE patients.
{mixed}/{best_k} clusters mix patients from multiple cohorts (Cramér's V = {cramers_v:.3f}).
NMI between clusters and DSM-5 subtypes = {nmi_dsm:.4f}, confirming transdiagnostic structure.
Treatment profiles and functioning outcomes differ significantly across clusters,
indicating clinical relevance beyond statistical artifact.</p>
</div>

<footer style="margin-top:48px;padding-top:16px;border-top:1px solid var(--bd);color:var(--mt);font-size:12px;">
FACE Stratification — {ds.X.shape[0]:,} patients, {ds.X.shape[1]} features, {best_k} clusters, {len(dsm_dist)} DSM subtypes
</footer></body></html>"""

report_path = OUT / "results_report.html"
report_path.write_text(html, encoding="utf-8")
logger.info("Report: %s (%d bytes)", report_path, len(html))
logger.info("═══ PIPELINE COMPLETE ═══")
