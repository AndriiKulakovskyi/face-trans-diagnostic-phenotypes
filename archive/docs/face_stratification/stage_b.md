# Stage B — Transdiagnostic Patient Embedding, Clustering, and Validation

> **Version 2.0** — Complete rewrite consolidating the former `stage_b.md`, `stage_b2.md`, and `stage_b2_5.md`.

## 1. Scientific Objective

Learn patient embeddings that capture **multimodal clinical similarity** across 4 FACE psychiatric cohorts (BP, SZ, DR, ASP), comparing graph-based and non-graph methods against a formal raw-feature baseline. Cluster these embeddings to discover **data-driven transdiagnostic phenotypes** that cut across DSM-5 diagnostic boundaries, and validate them with rigorous statistical and clinical evaluation.

## 2. Data Description

| Cohort | Pathology | N | Features |
|--------|-----------|---|----------|
| BP | Bipolar Disorder | ~5,400 | 2,229 CSV columns |
| SZ | Schizophrenia | ~2,200 | 1,713 CSV columns |
| DR | Treatment-Resistant Depression | ~350 | 2,217 CSV columns |
| ASP | Autism Spectrum Disorder | ~1,300 | 4,326 CSV columns |

After harmonization: ~9,250 patients, ~215 unified features across 21 clinical blocks. Heavy structural missingness (cohort-specific instruments). Only ~8-9 features with >= 50% coverage in all 4 cohorts (the "transdiagnostic" set).

**Key constraint:** No imputation on the default path. All pairwise similarities use pairwise-complete masked computation.

## 3. Experimental Design

### 3.1 Train/Test Split

**Stratified by cohort + DSM subtype** using `StratifiedShuffleSplit` on the combined key `"{cohort}_{dsm_diagnosis}"`. Default: 80% train / 20% test. DR retains ~70 test patients (minimum for stable silhouette estimation).

**Leakage prevention protocol:**
1. `fit_normalization()` on train split only
2. `transform_normalization(test, train_stats)` for test
3. Graph built on train patients only; test patients get test-to-train edges
4. GNN inductive inference: frozen encoder + test-to-train edges
5. Clustering fitted on train; test assigned to nearest centroid

### 3.2 Cross-Validation

Repeated 5-fold stratified CV (3 repeats) for hyperparameter selection, applied to the **train split only** (never on test).

### 3.3 Leave-One-Cohort-Out (LOCO)

4 splits, each holding out one entire cohort. For **stability assessment** only — tests whether the clustering structure survives removal of an entire diagnostic category.

## 4. Embedding Methods (16 methods across 5 families)

### 4.1 Feature Baselines (no graph)

| # | Method | Mechanism | Dimension |
|---|--------|-----------|-----------|
| 1 | **RawFeatureBaseline** | k-means on normalized features directly | d_features |
| 2 | **TransdiagnosticPCA** | Linear PCA on ~8 transdiagnostic features | 8 |
| 3 | **KernelPCA (RBF)** | Non-linear feature interactions via RBF kernel | 16 |
| 4 | **TransdiagnosticUMAP** | Manifold learning on transdiagnostic features | 16 |

### 4.2 Deep Feature Baselines (no graph, PyTorch + MPS)

| # | Method | Architecture | Why |
|---|--------|-------------|-----|
| 5 | **VanillaAE** | d -> 128 -> 64 -> 32 -> 64 -> 128 -> d, MSE | Deep baseline without graph |
| 6 | **VAE** | Same + KL regularization (beta-VAE) | Smoother latent space |

NaN handling: fill with 0 + binary missingness mask as additional input channels.

### 4.3 Graph Spectral

| # | Method | Input | Captures |
|---|--------|-------|----------|
| 7 | **TransdiagnosticSpectral** | Transdiagnostic subgraph only | Trusted-edge community structure |
| 8 | **MultiplexSpectral** | All 17 block edge types | Full multi-relational geometry |
| 9 | **DiffusionMap** | Multiplex adjacency, diffusion time t=2 | Multi-scale structure |

### 4.4 GNN Methods (PyTorch, MPS-accelerated)

| # | Method | Objective | Architecture |
|---|--------|-----------|-------------|
| 10 | **GAE** | Link prediction (BCE) | 2-layer GCN encoder + inner product decoder |
| 11 | **VGAE** | Link prediction + KL | Variational GCN (mu, log_sigma) |
| 12 | **GraphCL** | Contrastive (NT-Xent) | Edge-drop + feature-mask augmentations |
| 13 | **GAT** | Link prediction (BCE) | Multi-head sparse attention |
| 14 | **DGI** | Mutual information maximization | Bilinear discriminator, node vs graph |
| 15 | **R-GCN** | Link prediction (BCE) | Per-relation weights, basis decomposition, **sparse ops** |

All GNN models: auto-detect device (`MPS > CUDA > CPU`), 20% edge holdout for GAE/VGAE evaluation. R-GCN rewritten with sparse operations (O(E) memory, not O(R x N^2)).

### 4.5 Multi-View Composite

| # | Method | Mechanism |
|---|--------|-----------|
| 16 | **WeightedConcatenated** | PCA(8) + TransSpectral(16) + MultiplexSpectral(32) with learned per-view weights |

Default composite: 56-dim. Each view L2-normalized before concatenation, then row-wise L2-normalized.

## 5. Clustering Methods (10 algorithms)

| Family | Method | Key Property |
|--------|--------|-------------|
| Partitional | **k-means** | Standard baseline |
| | **k-medoids (PAM)** | Produces actual patient prototypes for vignette retrieval |
| | **Mini-batch k-means** | Fast variant for permutation tests |
| Probabilistic | **GMM (full/tied/diag)** | Soft assignments, different covariance assumptions |
| | **Bayesian GMM** | Automatic k via Dirichlet process prior |
| Density | **HDBSCAN** | Arbitrary shapes, identifies noise/outliers |
| Kernel | **Spectral clustering** | Kernel-based in embedding space |
| Hierarchical | **Ward / complete / average** | Agglomerative with different linkage criteria |

k-selection: sweep k in {3..12} with dual-criterion selection (silhouette + clinical utility).

## 6. Validation Framework

### 6.1 Internal Validation

- **Silhouette** (cosine, subsampled) — per-point cohesion vs separation
- **Calinski-Harabasz** — ratio of between to within variance
- **Davies-Bouldin** — worst-case cluster pair overlap
- **Dunn index** — min inter-cluster / max intra-cluster distance

### 6.2 External Validation (vs DSM)

- **ARI, NMI, AMI** — agreement with cohort labels (adjusted for chance)
- **V-measure** — harmonic mean of homogeneity + completeness
- **Cramer's V** — strength of DSM alignment (want: moderate, not extreme)
- **Chi-square** — statistical significance of cluster-cohort association

### 6.3 Information-Theoretic

- **H(clusters)** — Shannon entropy of cluster distribution
- **H(cohort|cluster)** — conditional entropy: high = transdiagnostic
- **I(cluster; cohort)** — mutual information
- **Transdiagnostic score** — mean per-cluster cohort entropy / log2(n_cohorts)
- **Per-feature information gain** — ANOVA F-statistic per feature against clusters

### 6.4 Permutation Testing

Two null types:
- **Label permutation null** (silhouette): shuffle cluster labels, recompute
- **Reference permutation null** (ARI, NMI, Cramer's V): shuffle DSM labels

n=1000 permutations. Reports p-value + 95% CI for each metric.

### 6.5 Stability Analysis

- **Bootstrap stability** (n=100): resample -> re-cluster -> pairwise ARI + Jaccard
- **Perturbation stability**: Gaussian noise (sigma=0.01) -> re-cluster -> ARI
- **LOCO stability**: hold out one cohort -> re-train -> measure agreement

### 6.6 Clinical Validation

- **Treatment profiles**: per-cluster medication class proportions, Kruskal-Wallis test
- **Suicide risk**: per-cluster attempt/ideation rates, chi-squared concentration
- **Biomarkers**: per-cluster BMI, metabolic syndrome, lipids (Kruskal-Wallis)
- **Safety**: no cluster should disproportionately concentrate high-risk patients

### 6.7 Cohort Fairness

- **Entropy ratio**: H(cohort|cluster) / H(cohort) -> 1 = transdiagnostic
- **Max cluster imbalance**: worst-case cohort over-representation ratio
- **Moran's I propagation**: which features are visible/invisible to the graph

## 7. Interpretability

- **Feature importance** (permutation-based): ANOVA F per feature against cluster labels
- **Dimension analysis**: what each latent dimension correlates with
- **Block ablation**: remove one clinical block -> re-run pipeline -> measure metric change

## 8. Pipeline Orchestration

```python
from face_stratification.stage_b.pipeline import StageBPipeline

pipeline = StageBPipeline(config_path="config/face_stratification/stage_b_config.yaml")
result = pipeline.run(dataset)
summary = pipeline.summarize(result)
```

Steps: split -> normalize (train) -> graph (train) -> embed (16 methods) -> cluster (10 methods x k) -> validate (internal + external + info-theoretic + permutation + stability) -> interpret -> select best -> save.

## 9. Configuration

All hyperparameters in `config/face_stratification/stage_b_config.yaml`. No magic numbers in Python.

## 10. Visualization

All figures: Plotly dark theme (`paper_bgcolor="#0f1117"`, `plot_bgcolor="#1a1d27"`, scale=3). Includes:
- Embedding comparison grid (2D UMAP projections)
- Metric comparison dashboard (bar chart with error bars)
- Cluster x cohort contingency heatmap (Viridis)
- Stability curves (ARI vs k with confidence bands)
- Feature enrichment heatmap (RdBu_r diverging)
- Permutation null distributions
- Method ranking radar chart
- DSM subtype alluvial/Sankey diagram
- Train vs test overfitting check
- LOCO stability heatmap

## 11. Reproducibility

- Fixed seeds (42 for split, 0 for clustering)
- Deterministic operations where possible
- Full config versioned in YAML
- `PatientEmbedding.save()` / `.load()` for embedding persistence
- `StageBResult` captures all intermediate artifacts

## 12. File Inventory

### New Files (created by this rewrite)

| File | Purpose |
|------|---------|
| `evaluation/__init__.py` | Evaluation subpackage |
| `evaluation/split.py` | Train/test split + CV |
| `evaluation/leakage_guard.py` | Data leakage prevention |
| `evaluation/validation.py` | Internal + external + info-theoretic metrics |
| `evaluation/permutation.py` | Permutation testing |
| `evaluation/stability.py` | Bootstrap + perturbation + LOCO |
| `evaluation/clinical_validation.py` | Treatment, suicide, biomarker validation |
| `evaluation/interpretability.py` | Feature importance + dimension analysis + block ablation |
| `evaluation/fairness.py` | Cohort-fair clustering metrics |
| `evaluation/graph_analysis.py` | Moran's I feature propagation analysis |
| `models/kernel_methods.py` | KernelPCA + DiffusionMap |
| `models/deep_baselines.py` | VanillaAE + VAE |
| `stage_b2/vgae.py` | Variational GAE |
| `stage_b2/gat.py` | Graph Attention Network |
| `stage_b2/dgi.py` | Deep Graph Infomax |
| `stage_b/pipeline.py` | Master orchestrator |
| `stage_b/method_registry.py` | Method enumeration from YAML |
| `visualization/stage_b_plots.py` | All Plotly figures |
| `config/face_stratification/stage_b_config.yaml` | All hyperparameters |

### Modified Files

| File | Change |
|------|--------|
| `stage_b2/rgcn.py` | Rewritten with sparse ops, implements BaseEmbeddingModel |
| `models/composite.py` | Added WeightedConcatenatedEmbedding |
| `clustering/algorithms.py` | Added 7 new clustering methods |
| `clustering/metrics.py` | Added Calinski-Harabasz, Davies-Bouldin, Cramer's V, information-theoretic |
| `__init__.py` | Updated exports for all new classes/functions |
| `stage_b2/__init__.py` | Updated for VGAE, GAT, DGI, sparse R-GCN |
