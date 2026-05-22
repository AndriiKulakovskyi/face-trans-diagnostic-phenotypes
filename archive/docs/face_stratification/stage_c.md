# Stage C — Clustering, k Selection, and Validation

## Objective

Discover data-driven transdiagnostic phenotypes through consensus clustering, with scientifically valid k selection using both statistical and clinical criteria.

## Method

### Consensus Clustering
- 16 base clusterings: 4 algorithms (k-means, GMM, Ward, Spectral) × 4 embedding views.
- Co-association matrix: fraction of times each patient pair is co-clustered.
- Hierarchical agglomeration on the consensus matrix.

### Dual-Criterion k Selection
**Data science criteria:**
- Silhouette score, Davies-Bouldin index, Calinski-Harabasz score
- Gap statistic (vs uniform reference)
- Bootstrap stability (25 bootstrap samples, mean ARI)

**Clinical utility criteria:**
- Within-cluster treatment profile homogeneity (Shannon entropy)
- Within-cluster functioning score variance (FAST/PSP/EGF)
- Suicide risk concentration (chi-squared test of attempt rates)
- DSM subtype entropy per cluster

Selected k at the intersection of acceptable data science metrics and optimal clinical utility.

### Soft Clustering
- GMM posteriors give per-patient probability of belonging to each cluster.
- Per-patient assignment entropy identifies boundary patients (entropy > 1.5 bits).
- Boundary patients analyzed as potential dual-phenotype or transitional cases.

### Meta-Stability
- Each patient's cluster assignment under all embedding methods (baseline, PCA, UMAP, spectral, GAE, GraphCL, R-GCN).
- Agreement fraction: fraction of methods assigning the same cluster.
- Core members (agreement ≥ 80%) vs boundary patients (agreement < 50%).

### Validation
- **DSM subtype comparison**: Cramer's V, ARI, NMI, entropy against fine-grained subtypes.
- **Treatment validation**: per-cluster treatment profiles, functioning outcomes (Kruskal-Wallis), MARS adherence.
- **Safety analysis**: per-cluster suicide attempt/ideation rates, chi-squared concentration, high-risk cluster identification.

## Results

*To be populated after running the full pipeline.*

## Limitations

- Consensus clustering is sensitive to the choice of base clusterings.
- Clinical utility metrics may favor clinically obvious groupings (e.g., by severity) over novel transdiagnostic subtypes.
- Safety analysis is observational — cluster membership does not imply causal risk.

## Next Steps

- Run pipeline with all improvements and generate cluster phenotype cards.
- Expert clinical validation using medoid vignettes.
