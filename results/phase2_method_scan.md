# Phase 2 — ML methods scan (2024–2026) for trans-diagnostic patient clustering

> Performed before locking the primary clustering method for `methodology-v1`.
> Goal: confirm that SNF + Leiden is still current, surface any 2024–2026
> developments we should incorporate, and flag what to leave out.

## Scope of the scan

Multi-axis literature search across:

- Trans-diagnostic psychiatric subtyping (clinical-feature clustering across
  DSM-5 categories).
- Post-SNF developments (miss-SNF, metasnf, ANF, SM-netFusion).
- Multi-modal graph neural networks for patient stratification.
- Self-supervised / contrastive learning on EHR / clinical data.
- Foundation models for structured clinical events.
- Variational autoencoders for mixed-type subtype discovery.
- Latent growth / latent class trajectory clustering for longitudinal data.
- Community detection (Louvain, Leiden) and cluster stability (Hennig 2007).

## Findings by family

### 1. SNF lineage (the primary candidate)

| Method | Year / venue | What it adds | Fit for FACE |
|---|---|---|---|
| **SNF** (Wang et al.) | 2014, Nature Methods | Multi-modal similarity fusion via random walks; foundational. | ✓ Reference primary. |
| **snfpy** (Markello) | 2017+, MIT | Faithful Python port; scikit-learn compatible. | ✓ Use this. |
| **ANF** (Affinity Network Fusion) | ~2018 | View-wise weighting; faster; circumvents SNF normalization. | Sensitivity check candidate. |
| **miss-SNF** | 2025, Bioinformatics | Handles *completely missing* modalities via self-loops. | Not applicable — our missingness is within-modality NaN, not whole-modality. |
| **metasnf** | 2024, R / CRAN | Meta-clustering across an SNF hyperparameter grid (σ, K, T). Robust meta-cluster solution. | ✓ Adopt the *principle* in Python: sweep then commit. |
| **SM-netFusion** | ~2023 | Supervised cross-diffusion using class labels. | Not applicable — we are unsupervised. |

### 2. Graph neural networks for patient stratification

| Source | Verdict |
|---|---|
| Multimodal GNN review, *Frontiers AI* 2025 | Intermediate fusion (81% of systems) + attention dominates. Top performers (FC-HGNN for depression AUC 0.95-1.00, CsAGP for AD AUC 0.99-1.00) are **supervised diagnosis tasks**, not unsupervised subtyping. |
| Causality-aware GNNs, *npj Systems Biology* 2025 | Pathway-level functional stratification — different problem. |
| HePGR (heterogeneous patient graph), *Springer* 2025 | Graph attention + contrastive for EMR; supervised. |

**Verdict for unsupervised trans-diagnostic clustering at N=6K**: GNNs do
not have a clear advantage over SNF + community detection. Reserve for
Phase 7+ if a deep-learning sensitivity check is required.

### 3. Self-supervised / contrastive learning on EHR

| Source | Verdict |
|---|---|
| GAME (federated EHR rep. learning), 2024+ | k-means clustering on learned embeddings, demonstrated on AD + suicide risk. |
| Hypergraph contrastive learning for EHRs, 2024 | Improves supervised tasks (sepsis, diagnosis). |
| FairEHR-CLP, 2024 | Fairness-aware contrastive; supervised tasks. |

**Verdict**: Contrastive EHR representations are designed for *event-sequence*
data, not yearly cross-sectional snapshots. Not applicable to FACE without
substantial re-engineering.

### 4. Foundation models for structured EHR

| Source | Verdict |
|---|---|
| Systematic review of EHR foundation models, 2025 | Architecture: transformers with long context windows, next-event prediction. |
| PRISM (transformer LM of clinical events), 2025 | Tokens = events on a longitudinal timeline. |
| EHRSHOT, Generative trajectory FM, 2024–2025 | Trained on millions of event sequences. |

**Verdict**: Wrong data shape for FACE. These models presuppose dense
event streams (medication orders, lab time points, codes). FACE has 5
yearly visits with curated cross-sectional snapshots. Not applicable.

### 5. Variational autoencoders / deep generative subtyping

| Source | Verdict |
|---|---|
| MoVAE (multi-omics VAE), 2024 | Multi-omics integration + cancer subtype detection. |
| Deconfounding VAEs for subtyping, *Briefings in Bioinformatics* 2024 | VAE + adversarial deconfounding for batch/age effects. |
| Joint clinical+molecular subtyping VAE for COPD, 2023 | Mixed-data VAE. |

**Verdict**: VAEs are credible at N≥10⁴ with very high-dimensional features
(omics). At N=6K, p=59, the inductive bias is wrong — we'd over-parameterize.
Worth a sensitivity check on the 31 continuous-only features but not
primary.

### 6. Trans-diagnostic psychiatric clustering precedents

| Source | Method | N | Trans-diagnostic outcome |
|---|---|---|---|
| Wen et al. 2021, *Neuropsychopharmacology* | k-means on dimensional clinical features | 1,250 (DEP/BIP/SZ/SZA + controls) | 5 mixed clusters ordered by severity; PGS validation. |
| 2024 medrxiv | k-means on symptoms + functional connectivity | trans-diagnostic inpatients | clinical + neural signature clusters. |
| 2025 bioRxiv (Aug) | EEG features + clustering | 1,701 youth | trans-diagnostic electrophysiological subtypes. |
| Drysdale 2017 *Nat Med* | Hierarchical clustering on connectivity | depression | 4 subtypes; influential precedent. |

**Verdict**: Classical clustering (k-means, hierarchical) on dimensional
features is the dominant paradigm. SNF + community detection would be a
modest methodological step forward, defensible because our data is genuinely
multi-modal mixed-type.

### 7. Longitudinal trajectory clustering (relevant to Phase 4)

| Method | When to use |
|---|---|
| LGMM (latent growth mixture model) | Continuous outcome over time; gold standard for psychiatric trajectories. |
| GBTM / LCGA (group-based / latent class growth) | Categorical or count outcomes over time. |
| KmL (k-means longitudinal) | Distance-based, mixed-type via mods. |
| Latent Transition Analysis (LTA) | Movement between latent states across visits — directly relevant to our V0→V4 transition story. |

**Verdict for Phase 4**: Re-cluster + match by patient ID (already
pre-registered) is simpler and avoids LGMM's parametric trajectory
assumptions. Add LTA as a *secondary* trajectory analysis if reviewers
push for explicit transition modelling.

### 8. Cluster stability / community detection

- **Hennig 2007 cluster bootstrap** remains the standard. Thresholds:
  ≥ 0.85 highly stable, ≤ 0.5 dissolve.
- **Leiden** > Louvain: better-connected communities, faster (Traag et al.
  2019). Use `leidenalg` in Python.
- **Bootstrap n**: 100 sufficient for triage; 1000 for publication.

## Comparison table — primary clustering candidates

| Method | Mixed-type | NaN-native | Multi-modal | Interpretable | Nature pedigree | Code maturity |
|---|---|---|---|---|---|---|
| **SNF + Leiden** | ✓ via per-modality networks | ✓ partial similarities | **✓ native** | Moderate (post-hoc by feature) | Wang 2014 *Nat Methods*; SNF used in multiple cancer subtype papers | snfpy mature |
| Hierarchical + Gower | ✓ via Gower | ✓ partial Gower | ✗ single distance | High (dendrogram) | Drysdale 2017 *Nat Med* | scipy mature |
| LCA / mixture models | ✓ if categorical | Limited | ✗ single likelihood | High (latent profiles) | Wen 2021 *Neuropsychopharm* | stepmix mature |
| K-means / KMeans++ | ✗ Euclidean | ✗ requires imputation | ✗ | Moderate | classic | scikit-learn |
| GMM | ✗ on binaries | ✗ requires imputation | ✗ | Moderate | classic | scikit-learn |
| HDBSCAN | ✓ flexible distance | Limited | ✗ | Low (no centres) | classic | hdbscan |
| GNN + attention | ✓ | Architecture-dependent | ✓ | Low | Supervised tasks dominant | PyG, DGL |
| VAE / contrastive | ✓ | ✗ requires imputation or masking | ✓ | Low | Multi-omics subtype literature | pytorch |
| EHR foundation models | n/a | n/a | n/a | Low | precision-medicine event prediction | EHRSHOT, PRISM |

## Final recommendation (revises ROADMAP §3 D3)

**PRIMARY clustering**: SNF (Wang 2014) + Leiden community detection (Traag 2019).

- Build modality-specific similarity networks: biology / med-history / sleep
  / mood-severity / suicide history / demographics.
- Fuse via standard SNF (snfpy).
- Apply Leiden on the fused affinity graph.
- Pre-register σ, K (kNN), T (diffusion iters), resolution before any
  primary run. Sweep on the V0 anchor in a separate phase-2 script, then
  commit (metasnf-style robustness over hyperparameters).

**ROBUSTNESS CHECKS** (run on same V0 anchor, report concordance via
ARI / NMI vs primary):

1. **Hierarchical (Ward) + Gower distance** — interpretable baseline.
2. **Latent Class Analysis (LCA)** — probabilistic mixture for mixed-type
   data, principled for our categorical features (stepmix in Python).
3. **Affinity Network Fusion (ANF)** — SNF variant; tests sensitivity to
   the specific fusion algorithm.

**STABILITY** (primary):

- Hennig 2007 bootstrap Jaccard, 1000 resamples, on the primary SNF
  clusters. Dissolved clusters (mean Jaccard < 0.5) flagged and excluded
  from H1 testing.

**WHAT WE EXPLICITLY DO NOT INCLUDE** (and why):

- GMM as a primary or robustness method — Gaussian likelihood on 28 binary
  features is ill-posed (covariance degenerate).
- HDBSCAN as a primary or sensitivity — density-based, no characterizable
  centres, struggles at our N.
- GNN, VAE, or contrastive embedding pipelines — SOTA for supervised
  tasks, no clear gain for unsupervised subtyping at N=6K, and add
  substantial implementation/dependency cost. Reserve for Phase 7+.
- EHR foundation models / next-event LLMs — wrong data shape for our
  yearly-visit cross-sectional snapshots.
- LGMM as primary trajectory model — over-parametric for the V0-anchor +
  re-cluster matching strategy; reserve as a secondary trajectory analysis.

## Key sources

- **SNF**: Wang et al., *Nature Methods* 2014. <https://www.nature.com/articles/nmeth.2810>
- **snfpy**: Markello, GitHub. <https://github.com/rmarkello/snfpy>
- **miss-SNF**: 2025, *Bioinformatics*. <https://academic.oup.com/bioinformatics/article/41/4/btaf150/8106484>
- **metasnf**: 2024, CRAN. <https://cran.r-project.org/package=metasnf>
- **Multimodal GNN review**: *Frontiers in AI* 2025. <https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1716706/full>
- **Causality-aware GNNs**: *npj Systems Biology* 2025. <https://www.nature.com/articles/s41540-025-00567-1>
- **Deconfounding VAEs**: *Briefings in Bioinformatics* 2024. <https://academic.oup.com/bib/article/25/6/bbae512/7824239>
- **Trans-diagnostic clusters (5 mixed)**: Wen et al., *Neuropsychopharmacology* 2021. <https://www.nature.com/articles/s41386-021-01051-0>
- **2025 EEG transdiagnostic youth**: bioRxiv. <https://www.biorxiv.org/content/10.1101/2025.08.01.668189v1.full>
- **Functional-connectivity transdiagnostic**: medRxiv 2024. <https://www.medrxiv.org/content/10.1101/2024.12.30.24319777.full.pdf>
- **Leiden algorithm**: Traag et al., *Scientific Reports* 2019. <https://www.nature.com/articles/s41598-019-41695-z>
- **Hennig bootstrap stability**: clusterboot/fpc, 2007. <https://rdrr.io/cran/fpc/man/clusterboot.html>
- **Longitudinal clustering review (2025)**: *International Statistical Review*. <https://onlinelibrary.wiley.com/doi/full/10.1111/insr.12588>
- **Modern ML for precision psychiatry**: *Patterns* 2022. <https://www.cell.com/patterns/fulltext/S2666-3899(22)00227-6>
