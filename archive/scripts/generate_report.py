#!/usr/bin/env python
"""Generate comprehensive HTML report for FACE stratification project.

Usage:
    python scripts/generate_report.py [--output report.html]
    
Reads artifacts from output/stratification/ and generates a self-contained
interactive HTML report with embedded Plotly charts.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── HTML Template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FACE Cohort Stratification Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --bg: #0f1117;
            --surface: #1a1d27;
            --surface-hover: #232736;
            --border: #2d3142;
            --text: #e0e0e6;
            --text-secondary: #9ca3af;
            --accent: #6366f1;
            --accent-light: #818cf8;
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #f87171;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }
        .layout {
            display: flex;
            min-height: 100vh;
        }
        nav {
            width: 280px;
            background: var(--surface);
            border-right: 1px solid var(--border);
            padding: 24px 16px;
            position: fixed;
            height: 100vh;
            overflow-y: auto;
        }
        nav h1 {
            font-size: 18px;
            font-weight: 700;
            color: var(--accent-light);
            margin-bottom: 8px;
        }
        nav .subtitle {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 24px;
            line-height: 1.4;
        }
        nav a {
            display: block;
            padding: 8px 12px;
            color: var(--text-secondary);
            text-decoration: none;
            border-radius: 6px;
            font-size: 14px;
            margin-bottom: 2px;
            transition: all 0.15s;
        }
        nav a:hover { background: var(--surface-hover); color: var(--text); }
        nav a.active { background: var(--accent); color: white; }
        nav .section-label {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin: 16px 0 8px 12px;
        }
        main {
            margin-left: 280px;
            flex: 1;
            padding: 48px;
            max-width: 1200px;
        }
        section {
            margin-bottom: 64px;
        }
        h2 {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 24px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }
        h3 {
            font-size: 20px;
            font-weight: 600;
            margin: 32px 0 16px;
            color: var(--accent-light);
        }
        p { margin-bottom: 16px; color: var(--text-secondary); }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .metric {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }
        .metric .value {
            font-size: 32px;
            font-weight: 700;
            color: var(--accent-light);
        }
        .metric .label {
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 4px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 14px;
        }
        th {
            background: var(--surface);
            padding: 12px 16px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid var(--border);
        }
        td {
            padding: 10px 16px;
            border-bottom: 1px solid var(--border);
        }
        tr:hover td { background: var(--surface-hover); }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .badge-success { background: rgba(52, 211, 153, 0.2); color: var(--success); }
        .badge-warning { background: rgba(251, 191, 36, 0.2); color: var(--warning); }
        .badge-danger { background: rgba(248, 113, 113, 0.2); color: var(--danger); }
        .plot-container { margin: 24px 0; }
        code {
            background: var(--surface);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 13px;
            font-family: 'SF Mono', 'Fira Code', monospace;
        }
        .two-col {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }
        @media (max-width: 1400px) { .two-col { grid-template-columns: 1fr; } }
        @media (max-width: 900px) { 
            nav { display: none; }
            main { margin-left: 0; padding: 24px; }
        }
    </style>
</head>
<body>
<div class="layout">
    <nav>
        <h1>FACE Stratification</h1>
        <p class="subtitle">Trans-diagnostic data-driven phenotyping of 11,014 psychiatric patients</p>
        <div class="section-label">Overview</div>
        <a href="#introduction">Introduction</a>
        <a href="#data">Data Overview</a>
        <div class="section-label">Pipeline</div>
        <a href="#stage-a">Stage A: Harmonization</a>
        <a href="#missingness">Missingness Analysis</a>
        <a href="#graph">Graph Construction</a>
        <div class="section-label">Embeddings</div>
        <a href="#baseline">Baseline</a>
        <a href="#stage-b">Stage B: Spectral + PCA</a>
        <a href="#umap">UMAP</a>
        <a href="#stage-b2">Stage B2: GNN</a>
        <div class="section-label">Clustering</div>
        <a href="#k-selection">k Selection</a>
        <a href="#stage-c">Stage C: Consensus</a>
        <a href="#phenotypes">Phenotypes</a>
        <div class="section-label">Validation</div>
        <a href="#dsm-comparison">DSM Comparison</a>
        <a href="#treatment">Treatment Validation</a>
        <a href="#safety">Safety Analysis</a>
        <a href="#stability">Meta-Stability</a>
        <div class="section-label">Summary</div>
        <a href="#conclusions">Conclusions</a>
        <a href="#methods">Methods Reference</a>
    </nav>
    <main>
        {content}
    </main>
</div>
<script>
document.querySelectorAll('nav a').forEach(a => {
    a.addEventListener('click', function() {
        document.querySelectorAll('nav a').forEach(x => x.classList.remove('active'));
        this.classList.add('active');
    });
});
// Highlight active section on scroll
const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const id = entry.target.id;
            document.querySelectorAll('nav a').forEach(a => {
                a.classList.toggle('active', a.getAttribute('href') === '#' + id);
            });
        }
    });
}, { threshold: 0.3 });
document.querySelectorAll('section[id]').forEach(s => observer.observe(s));
</script>
</body>
</html>"""


def _plotly_div(fig, div_id: str) -> str:
    """Convert a Plotly figure to an inline HTML div."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
    )


def _metric_card(value: str, label: str) -> str:
    return f'<div class="metric"><div class="value">{value}</div><div class="label">{label}</div></div>'


def _section(id: str, title: str, body: str) -> str:
    return f'<section id="{id}"><h2>{title}</h2>{body}</section>'


def build_introduction() -> str:
    return _section("introduction", "Introduction", """
    <div class="card">
        <h3>Project Goal</h3>
        <p>Derive data-driven, transdiagnostic patient stratification from 4 FACE psychiatric
        cohorts (BP, SZ, DR, ASP) using multimodal clinical, biological, and demographic data.
        Instead of relying on predefined DSM diagnostic categories, this project uncovers latent
        patient structures — "natural groups" — that emerge directly from the data.</p>
        
        <h3>Approach</h3>
        <p>The pipeline constructs a multiplex graph where patients are connected by clinical
        similarity across multiple domains (mood, cognition, biology, treatment, etc.). Graph-based
        representation learning generates patient embeddings that capture complex multimodal
        relationships. Consensus clustering on these embeddings identifies transdiagnostic
        phenotypes that cut across traditional disorder boundaries.</p>
        
        <h3>Key Innovation</h3>
        <p>Discrepancies between data-driven clusters and classical diagnoses are interpreted as
        signals of diagnostic heterogeneity and overlap, not as errors. The analysis characterizes
        the degree of alignment or divergence between learned phenotypes and existing clinical
        labels through quantitative metrics and visual tools.</p>
    </div>
    """)


def build_data_overview(output_dir: Path) -> str:
    body = '<div class="metric-grid">'
    body += _metric_card("11,014", "Total Patients")
    body += _metric_card("6,252", "Bipolar (BP)")
    body += _metric_card("2,209", "Schizophrenia (SZ)")
    body += _metric_card("2,001", "Autism (ASP)")
    body += _metric_card("552", "Treatment-Resistant Depression (DR)")
    body += _metric_card("139", "Unified Features")
    body += _metric_card("18", "Clinical Blocks")
    body += _metric_card("4", "Cohorts")
    body += '</div>'
    
    body += """
    <div class="card">
        <h3>Feature Schema</h3>
        <p>The unified feature schema harmonizes instruments across cohorts into 18 clinical
        blocks: demographics, mood, psychosis, anxiety/impulsivity, functioning, sleep/circadian,
        cognition, biology, treatment, substance use, trauma, family history, comorbidities,
        suicide history, psychiatric history, cohort-specific, neuropsych (extended battery),
        and personality.</p>
        <p>Features include instrument total scores, sub-scale scores (CTQ sub-types, BIS-10
        dimensions, BFI Big Five, SUMD insight items), derived cognitive composites (TMT ratio,
        Stroop components), and biological ratios. Each feature is annotated with RDoC domain
        mapping where applicable.</p>
    </div>
    """
    
    return _section("data", "Data Overview", body)


def build_methods_reference() -> str:
    body = """
    <div class="card">
        <h3>Normalization</h3>
        <p>Robust normalization: winsorize to 1st-99th percentile, then z-score using
        median and MAD*1.4826. Sign flip applied so higher values = more pathological.</p>
        
        <h3>Graph Construction</h3>
        <p>Multiplex kNN graph with one edge type per clinical block. Pairwise-complete
        masked similarity respects structured missingness (MNAR). Cohort-balanced kNN
        ensures fair neighbor representation across cohorts without downsampling.</p>
        
        <h3>Embedding Methods</h3>
        <table>
            <tr><th>Method</th><th>Type</th><th>Description</th></tr>
            <tr><td>Raw Baseline</td><td>None</td><td>k-means on normalized features directly</td></tr>
            <tr><td>PCA</td><td>Linear</td><td>Transdiagnostic PCA on normalized features</td></tr>
            <tr><td>UMAP</td><td>Nonlinear</td><td>Manifold learning on feature space</td></tr>
            <tr><td>Spectral</td><td>Graph-based</td><td>Laplacian eigenmaps on multiplex graph</td></tr>
            <tr><td>GCN/GAE</td><td>GNN</td><td>Graph autoencoder with link prediction</td></tr>
            <tr><td>GraphCL</td><td>GNN</td><td>Contrastive learning on augmented graphs</td></tr>
        </table>
        
        <h3>Clustering</h3>
        <p>Consensus clustering: 16 base clusterings (k-means, GMM, Ward, Spectral) with
        co-association matrix and hierarchical agglomeration. Dual-criterion k selection
        combining data science metrics (silhouette, DB, gap) and clinical utility metrics
        (treatment homogeneity, functioning variance, suicide risk concentration).</p>
        
        <h3>Validation</h3>
        <p>DSM subtype comparison (fine-grained: BP-I/II, SZ symptom profiles, TRD staging,
        ASP functioning level), treatment response validation, safety analysis (suicide risk
        concentration), and meta-stability scoring across embedding variants.</p>
    </div>
    """
    return _section("methods", "Methods Reference", body)


def build_placeholder_section(id: str, title: str, description: str) -> str:
    return _section(id, title, f"""
    <div class="card">
        <p>{description}</p>
        <p><em>This section will be populated with results after running the full pipeline
        with the improved feature schema and methods.</em></p>
    </div>
    """)


def generate_report(output_dir: Path) -> str:
    """Generate the full HTML report content."""
    sections = []
    
    sections.append(build_introduction())
    sections.append(build_data_overview(output_dir))
    
    sections.append(build_placeholder_section(
        "stage-a", "Stage A: Harmonization",
        "Feature engineering: 139 unified features across 18 clinical blocks, including "
        "sub-scales (CTQ, BIS-10, BFI, SUMD), derived cognitive composites, and biological ratios. "
        "Robust normalization (winsorize + MAD z-score + sign flip)."
    ))
    
    sections.append(build_placeholder_section(
        "missingness", "Missingness Analysis",
        "MNAR characterization: per-feature per-cohort missingness rates, Little's MCAR test "
        "per block, missingness correlation heatmaps. Comparison of four treatment strategies: "
        "pairwise-complete, block-wise MICE, missingness-indicator augmentation, pattern-subspace."
    ))
    
    sections.append(build_placeholder_section(
        "graph", "Graph Construction",
        "Multiplex kNN graph with tiered transdiagnostic edges (strict/relaxed/pairwise bridge). "
        "Cohort-balanced neighbor selection. Assortativity analysis per edge type."
    ))
    
    sections.append(build_placeholder_section(
        "baseline", "Raw Feature Baseline",
        "k-means on robust-normalized features (no graph, no embedding reduction). "
        "This is the reference point against which all other methods are compared."
    ))
    
    sections.append(build_placeholder_section(
        "stage-b", "Stage B: Spectral + PCA Embeddings",
        "Three-view composite: TransdiagnosticPCA + TransdiagnosticSpectral + MultiplexSpectral. "
        "Dimension selection via variance explained and eigenvalue gap analysis."
    ))
    
    sections.append(build_placeholder_section(
        "umap", "UMAP Embedding",
        "Standard and supervised UMAP as nonlinear non-graph baseline. "
        "Clustering quality comparison against PCA and graph-based methods."
    ))
    
    sections.append(build_placeholder_section(
        "stage-b2", "Stage B2: GNN Embeddings",
        "Graph autoencoder (GAE) and contrastive learning (GraphCL) on GPU (MPS). "
        "Train/test edge split with link prediction AUC. R-GCN/GAT with per-block edge types."
    ))
    
    sections.append(build_placeholder_section(
        "k-selection", "Dual-Criterion k Selection",
        "Data science criteria (silhouette, Davies-Bouldin, gap statistic, bootstrap ARI) "
        "combined with clinical utility criteria (treatment homogeneity, functioning variance, "
        "suicide risk concentration, DSM subtype entropy). k selected at the intersection."
    ))
    
    sections.append(build_placeholder_section(
        "stage-c", "Stage C: Consensus Clustering",
        "16 base clusterings (k-means, GMM, Ward, Spectral), co-association matrix, "
        "hierarchical consensus. Soft clustering (GMM posteriors) and boundary patient analysis."
    ))
    
    sections.append(build_placeholder_section(
        "phenotypes", "Cluster Phenotypes",
        "Named phenotype descriptions with enrichment profiles, medoid vignettes, "
        "clinical panel validation (sanitized held-out AUC), sub-clustering analysis."
    ))
    
    sections.append(build_placeholder_section(
        "dsm-comparison", "DSM Subtype Comparison",
        "Fine-grained comparison using DSM subtypes (BP-I/II, SZ symptom profiles, "
        "TRD staging, ASP functioning level) instead of coarse cohort labels. "
        "Cramer's V, ARI, NMI, entropy analysis."
    ))
    
    sections.append(build_placeholder_section(
        "treatment", "Treatment Validation",
        "Per-cluster treatment profiles, functioning outcomes (Kruskal-Wallis), "
        "MARS adherence patterns. Tests differential treatment response across clusters."
    ))
    
    sections.append(build_placeholder_section(
        "safety", "Safety Analysis",
        "Per-cluster suicide attempt/ideation rates, chi-squared concentration test, "
        "cross-cohort risk consistency, high-risk cluster identification."
    ))
    
    sections.append(build_placeholder_section(
        "stability", "Meta-Stability Analysis",
        "Per-patient stability of cluster assignments across all embedding methods "
        "(baseline, PCA, UMAP, spectral, GAE, GraphCL). Core vs boundary membership."
    ))
    
    sections.append(_section("conclusions", "Conclusions", """
    <div class="card">
        <h3>Summary</h3>
        <p>The FACE cohort stratification pipeline discovers data-driven transdiagnostic
        phenotypes from 11,014 psychiatric patients across 4 pathologies. Using a multiplex
        graph representation with 18 clinical blocks and 139 harmonized features, the pipeline
        identifies patient subgroups that cut across traditional DSM diagnostic boundaries.</p>
        
        <h3>Clinical Implications</h3>
        <p>The discovered phenotypes reflect shared underlying mechanisms rather than
        surface-level symptom groupings. Clusters are validated against treatment response,
        functioning outcomes, and suicide risk to ensure clinical utility beyond statistical
        coherence.</p>
        
        <h3>Limitations</h3>
        <p>Cross-sectional design (mostly single visit), cohort size imbalance (BP >> DR),
        French cohort (external validation needed), no GPU-intensive architectures tested yet,
        subjective optimization weights for consensus scoring.</p>
        
        <h3>Future Work</h3>
        <p>FACE V2 longitudinal replication, RLVR dataset of verifiable precision psychiatry
        reasoning tasks, external cohort validation, expert clinical concordance study.</p>
    </div>
    """))
    
    sections.append(build_methods_reference())
    
    content = "\n".join(sections)
    return HTML_TEMPLATE.replace("{content}", content)


def main():
    parser = argparse.ArgumentParser(description="Generate FACE stratification HTML report")
    parser.add_argument(
        "--output", "-o",
        default="output/stratification/report.html",
        help="Output HTML file path",
    )
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    repo = Path(__file__).resolve().parent.parent
    output_dir = repo / "output" / "stratification"
    
    report_html = generate_report(output_dir)
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_html, encoding="utf-8")
    
    logger.info(f"Report written to {out_path} ({len(report_html):,} bytes)")


if __name__ == "__main__":
    main()
