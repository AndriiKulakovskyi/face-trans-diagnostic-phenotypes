# Stage B2 — GNN Embeddings (GPU-Accelerated)

## Objective

Learn patient embeddings that integrate graph structure with node features using neural message passing on the multiplex patient-similarity graph. Compare against non-graph baselines.

## Method

### GPU Acceleration
- Auto-detection: MPS (Apple Silicon) → CUDA → CPU via `get_device()`.
- All GNN models (GCN, GAE, R-GCN, GraphCL) run on the best available device.

### GCN Encoder
- 2-layer sparse GCN (Kipf & Welling 2017) operating on the merged multiplex adjacency.
- Symmetric normalization: D^{-1/2} A D^{-1/2}.

### Graph Autoencoder (GAE)
- GCN encoder + inner product decoder for link prediction.
- **Train/test edge split**: 20% held-out edges for link prediction AUC.
- Reports train AUC and test AUC to detect overfitting.

### R-GCN (Relational GCN)
- Preserves per-block edge types as separate relation matrices.
- Basis decomposition (n_bases ≤ 8) limits parameter count.
- 3-layer architecture with residual connections.
- One weight matrix per clinical block (17 relations), enabling the model to learn that mood-similarity edges carry different information than biology-similarity edges.

### GraphCL (Contrastive Learning)
- NT-Xent contrastive loss on augmented graph views.
- Domain-aware augmentation: edge dropping and feature masking within blocks.
- Device-agnostic implementation with MPS/CUDA support.

## Results

| Model | Embed Dim | Train AUC | Test AUC | Silhouette | vs Baseline |
|-------|-----------|-----------|----------|------------|-------------|
| GCN | — | — | — | — | — |
| GAE | — | — | — | — | — |
| R-GCN | — | — | — | — | — |
| GraphCL | — | — | — | — | — |

*To be populated after running the full pipeline.*

## Limitations

- Full-batch training (all patients at once) — not a bottleneck at n=11K but would need mini-batching at larger scale.
- R-GCN basis decomposition may underfit if the number of bases is too small relative to the number of distinct relation types.

## Next Steps

- Tune hyperparameters (hidden_dim, n_layers, n_bases, learning rate).
- Compare R-GCN vs GAT (attention-based) architecture.
