"""Publication-quality Plotly visualizations for Stage B.

All plots use the project's standard dark theme:
    template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27"
    scale=3 for high-DPI PNG export.

Cohort color scheme: bp=blue, sz=red, dr=green, asp=orange.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DARK = dict(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27")
SCALE = 3

COHORT_COLORS = {"bp": "#636EFA", "sz": "#EF553B", "dr": "#00CC96", "asp": "#FFA15A"}


def _save_fig(fig: Any, path: str | Path, width: int = 1100, height: int = 500) -> None:
    """Export figure to PNG via Kaleido."""
    try:
        fig.write_image(str(path), width=width, height=height, scale=SCALE)
        logger.info("Saved figure: %s", path)
    except Exception as exc:
        logger.warning("Failed to save %s: %s", path, exc)


# ─── Embedding comparison grid ──────────────────────────────────────────────


def plot_embedding_comparison_grid(
    projections: dict[str, np.ndarray],
    cohort_labels: np.ndarray,
    cluster_labels: dict[str, np.ndarray] | None = None,
    *,
    output_path: str | Path | None = None,
) -> Any:
    """2D UMAP projection grid for each embedding method, colored by cohort.

    Parameters
    ----------
    projections:
        {method_name: (N, 2) array of 2D coordinates}
    cohort_labels:
        (N,) cohort labels for coloring
    cluster_labels:
        Optional {method_name: (N,) labels} for cluster coloring subplot
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    methods = list(projections.keys())
    n_methods = len(methods)
    n_cols = min(4, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=methods,
        horizontal_spacing=0.04,
        vertical_spacing=0.06,
    )

    for idx, method in enumerate(methods):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        coords = projections[method]

        for cohort in sorted(set(cohort_labels)):
            mask = cohort_labels == cohort
            fig.add_trace(
                go.Scattergl(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode="markers",
                    marker=dict(size=2, color=COHORT_COLORS.get(cohort, "#888"), opacity=0.6),
                    name=cohort.upper(),
                    legendgroup=cohort,
                    showlegend=(idx == 0),
                ),
                row=row, col=col,
            )

    fig.update_layout(
        **DARK,
        height=300 * n_rows,
        width=300 * n_cols,
        title_text="Embedding Comparison — 2D UMAP Projections",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    if output_path:
        _save_fig(fig, output_path, width=300 * n_cols, height=300 * n_rows + 80)
    return fig


# ─── Metric comparison dashboard ────────────────────────────────────────────


def plot_metric_comparison_dashboard(
    results: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
    output_path: str | Path | None = None,
) -> Any:
    """Bar chart comparing methods across metrics with error bars.

    Parameters
    ----------
    results:
        DataFrame with columns: method, metric, value, ci_lower, ci_upper (optional)
    """
    import plotly.graph_objects as go

    if metrics is None:
        metrics = ["silhouette", "ari", "nmi", "transdiagnostic_score", "cramers_v"]

    available = [m for m in metrics if m in results.columns]
    if not available:
        logger.warning("No recognized metrics in results DataFrame")
        return go.Figure()

    methods = results.index if "method" not in results.columns else results["method"]

    fig = go.Figure()
    colors = ["#636EFA", "#EF553B", "#00CC96", "#FFA15A", "#AB63FA", "#FF6692"]

    for i, metric in enumerate(available):
        vals = results[metric].values
        fig.add_trace(go.Bar(
            name=metric,
            x=methods,
            y=vals,
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        **DARK,
        title="Method Comparison — Clustering Quality Metrics",
        barmode="group",
        xaxis_title="Method",
        yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=500,
        width=max(800, 100 * len(methods)),
    )

    if output_path:
        _save_fig(fig, output_path, width=max(800, 100 * len(methods)))
    return fig


# ─── Cluster × cohort contingency heatmap ───────────────────────────────────


def plot_cluster_cohort_contingency(
    cluster_labels: np.ndarray,
    cohort_labels: np.ndarray,
    *,
    normalize: str = "index",
    output_path: str | Path | None = None,
) -> Any:
    """Heatmap of cluster × cohort contingency table."""
    import plotly.graph_objects as go

    ct = pd.crosstab(
        pd.Series(cluster_labels, name="Cluster"),
        pd.Series(cohort_labels, name="Cohort"),
        normalize=normalize,
    )

    fig = go.Figure(data=go.Heatmap(
        z=ct.values,
        x=[str(c).upper() for c in ct.columns],
        y=[f"Cluster {c}" for c in ct.index],
        colorscale="Viridis",
        text=np.round(ct.values, 3),
        texttemplate="%{text:.2f}",
        textfont=dict(size=10),
        colorbar=dict(title="Proportion"),
    ))

    fig.update_layout(
        **DARK,
        title="Cluster × Cohort Distribution",
        xaxis_title="DSM Cohort",
        yaxis_title="Data-Driven Cluster",
        height=max(300, 50 * len(ct.index)),
        width=600,
    )

    if output_path:
        _save_fig(fig, output_path, width=600, height=max(300, 50 * len(ct.index)))
    return fig


# ─── Stability curves ──────────────────────────────────────────────────────


def plot_stability_curves(
    results: dict[str, pd.DataFrame],
    *,
    metric: str = "silhouette",
    output_path: str | Path | None = None,
) -> Any:
    """Mean metric vs k per method, with shaded confidence bands.

    Parameters
    ----------
    results:
        {method_name: DataFrame with columns [k, metric, ci_lower, ci_upper]}
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    colors = list(COHORT_COLORS.values()) + ["#AB63FA", "#FF6692", "#B6E880", "#FF97FF"]

    for i, (method, df) in enumerate(results.items()):
        if metric not in df.columns:
            continue
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=df["k"], y=df[metric],
            mode="lines+markers",
            name=method,
            line=dict(color=color),
            marker=dict(size=6),
        ))

        # Confidence band
        if "ci_lower" in df.columns and "ci_upper" in df.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([df["k"], df["k"][::-1]]),
                y=pd.concat([df["ci_upper"], df["ci_lower"][::-1]]),
                fill="toself",
                fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba"),
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
            ))

    fig.update_layout(
        **DARK,
        title=f"Stability Curves — {metric.replace('_', ' ').title()} vs k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title=metric.replace("_", " ").title(),
        height=500,
        width=800,
    )

    if output_path:
        _save_fig(fig, output_path, width=800, height=500)
    return fig


# ─── Feature enrichment heatmap ─────────────────────────────────────────────


def plot_feature_enrichment_heatmap(
    enrichment_df: pd.DataFrame,
    *,
    top_n: int = 15,
    output_path: str | Path | None = None,
) -> Any:
    """Heatmap of effect sizes for top enriched features per cluster.

    Parameters
    ----------
    enrichment_df:
        DataFrame with columns: feature, cluster, effect_size, significant
    """
    import plotly.graph_objects as go

    if enrichment_df.empty:
        return go.Figure()

    # Pivot to cluster × feature matrix
    if "effect_size" in enrichment_df.columns and "feature" in enrichment_df.columns:
        top_features = (
            enrichment_df.groupby("feature")["effect_size"]
            .apply(lambda x: x.abs().max())
            .nlargest(top_n)
            .index.tolist()
        )
        subset = enrichment_df[enrichment_df["feature"].isin(top_features)]
        pivot = subset.pivot_table(
            index="feature", columns="cluster", values="effect_size", fill_value=0
        )
    else:
        pivot = enrichment_df.iloc[:top_n]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"Cluster {c}" for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(pivot.values, 2),
        texttemplate="%{text:.2f}",
        textfont=dict(size=9),
        colorbar=dict(title="Effect Size"),
    ))

    fig.update_layout(
        **DARK,
        title=f"Feature Enrichment — Top {top_n} Features",
        xaxis_title="Cluster",
        yaxis_title="Feature",
        height=max(400, 25 * len(pivot.index)),
        width=max(600, 80 * len(pivot.columns)),
    )

    if output_path:
        _save_fig(fig, output_path, width=max(600, 80 * len(pivot.columns)),
                  height=max(400, 25 * len(pivot.index)))
    return fig


# ─── Permutation null distribution ──────────────────────────────────────────


def plot_permutation_null(
    observed: float,
    null_distribution: np.ndarray,
    metric_name: str,
    *,
    output_path: str | Path | None = None,
) -> Any:
    """Histogram of null distribution with vertical line at observed value."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=null_distribution,
        nbinsx=50,
        marker_color="#636EFA",
        opacity=0.7,
        name="Null distribution",
    ))
    fig.add_vline(
        x=observed,
        line_dash="dash",
        line_color="#EF553B",
        line_width=2,
        annotation_text=f"Observed: {observed:.4f}",
        annotation_position="top right",
    )

    p_value = float(np.mean(null_distribution >= observed))
    fig.update_layout(
        **DARK,
        title=f"Permutation Test — {metric_name} (p = {p_value:.4f})",
        xaxis_title=metric_name,
        yaxis_title="Count",
        height=400,
        width=700,
    )

    if output_path:
        _save_fig(fig, output_path, width=700, height=400)
    return fig


# ─── Method ranking radar ───────────────────────────────────────────────────


def plot_method_ranking_radar(
    results: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
    output_path: str | Path | None = None,
) -> Any:
    """Radar chart comparing methods across multiple metrics.

    Parameters
    ----------
    results:
        DataFrame indexed by method, columns = metrics.
    """
    import plotly.graph_objects as go

    if metrics is None:
        metrics = [c for c in results.columns if c in [
            "silhouette", "transdiagnostic_score", "stability",
            "calinski_harabasz_norm", "dunn_index_norm",
        ]]

    if not metrics:
        metrics = list(results.columns[:5])

    fig = go.Figure()
    colors = ["#636EFA", "#EF553B", "#00CC96", "#FFA15A", "#AB63FA", "#FF6692"]

    for i, (method, row) in enumerate(results.iterrows()):
        vals = [float(row.get(m, 0)) for m in metrics]
        vals.append(vals[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=metrics + [metrics[0]],
            name=str(method),
            fill="toself",
            opacity=0.4,
            line=dict(color=colors[i % len(colors)]),
        ))

    fig.update_layout(
        **DARK,
        title="Method Comparison — Multi-Metric Radar",
        polar=dict(
            bgcolor="#1a1d27",
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        height=600,
        width=700,
    )

    if output_path:
        _save_fig(fig, output_path, width=700, height=600)
    return fig


# ─── DSM subtype alluvial ───────────────────────────────────────────────────


def plot_dsm_subtype_mosaic(
    cluster_labels: np.ndarray,
    dsm_subtypes: np.ndarray,
    *,
    output_path: str | Path | None = None,
) -> Any:
    """Sankey/alluvial diagram: how DSM subtypes distribute across data-driven clusters."""
    import plotly.graph_objects as go

    # Build contingency
    ct = pd.crosstab(
        pd.Series(dsm_subtypes, name="DSM"),
        pd.Series(cluster_labels, name="Cluster"),
    )

    # Sankey: source = DSM subtypes, target = clusters
    dsm_labels = list(ct.index)
    cluster_labels_list = [f"Cluster {c}" for c in ct.columns]
    all_labels = dsm_labels + cluster_labels_list

    source, target, value = [], [], []
    for i, dsm in enumerate(dsm_labels):
        for j, cluster in enumerate(ct.columns):
            count = ct.loc[dsm, cluster]
            if count > 0:
                source.append(i)
                target.append(len(dsm_labels) + j)
                value.append(int(count))

    colors = list(COHORT_COLORS.values()) * 5

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=all_labels,
            color=colors[:len(all_labels)],
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color="rgba(150,150,150,0.3)",
        ),
    )])

    fig.update_layout(
        **DARK,
        title="DSM Subtypes → Data-Driven Clusters",
        height=500,
        width=900,
    )

    if output_path:
        _save_fig(fig, output_path, width=900, height=500)
    return fig


# ─── Train vs test comparison ───────────────────────────────────────────────


def plot_train_test_comparison(
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    *,
    output_path: str | Path | None = None,
) -> Any:
    """Side-by-side train vs test metrics to verify no overfitting."""
    import plotly.graph_objects as go

    metrics = sorted(set(train_metrics.keys()) & set(test_metrics.keys()))
    train_vals = [train_metrics[m] for m in metrics]
    test_vals = [test_metrics[m] for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Train", x=metrics, y=train_vals,
        marker_color="#636EFA",
    ))
    fig.add_trace(go.Bar(
        name="Test", x=metrics, y=test_vals,
        marker_color="#EF553B",
    ))

    fig.update_layout(
        **DARK,
        title="Train vs Test — Overfitting Check",
        barmode="group",
        xaxis_title="Metric",
        yaxis_title="Score",
        height=450,
        width=700,
    )

    if output_path:
        _save_fig(fig, output_path, width=700, height=450)
    return fig


# ─── LOCO stability heatmap ────────────────────────────────────────────────


def plot_loco_stability_heatmap(
    loco_results: dict[str, dict],
    *,
    output_path: str | Path | None = None,
) -> Any:
    """Heatmap of cohort-out × ARI stability."""
    import plotly.graph_objects as go

    cohorts = sorted(loco_results.keys())
    ari_vals = [
        loco_results[c].get("ari_vs_full", float("nan"))
        for c in cohorts
    ]

    fig = go.Figure(data=go.Heatmap(
        z=[ari_vals],
        x=[c.upper() for c in cohorts],
        y=["ARI vs Full"],
        colorscale="Viridis",
        text=[np.round(ari_vals, 3)],
        texttemplate="%{text:.3f}",
        textfont=dict(size=14),
        zmin=0, zmax=1,
        colorbar=dict(title="ARI"),
    ))

    fig.update_layout(
        **DARK,
        title="Leave-One-Cohort-Out Stability",
        xaxis_title="Held-Out Cohort",
        height=250,
        width=500,
    )

    if output_path:
        _save_fig(fig, output_path, width=500, height=250)
    return fig
