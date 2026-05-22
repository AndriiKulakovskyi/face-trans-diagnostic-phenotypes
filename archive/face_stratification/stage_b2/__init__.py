"""Stage B2 — deep GNN embeddings on the Stage A multiplex graph.

This package adds torch-based graph neural network views to the Stage B
composite embedding. Three unsupervised architectures are implemented:

- :class:`StageB2GAE` — Kipf-Welling Graph Autoencoder trained via
  link prediction (binary cross-entropy on positive edges vs negative
  samples).
- :class:`StageB2GraphContrastive` — GraphCL-style self-supervised
  contrastive learning with edge drop + feature masking augmentations
  and NT-Xent loss.
- :class:`StageB2RGCN` — Relational GCN that preserves per-block edge
  type information using relation-specific weight matrices with basis
  decomposition.

All models implement :class:`BaseEmbeddingModel` so they plug directly
into the Stage B :class:`ConcatenatedEmbedding` factory and Stage C's
pipeline without any surface changes.

Dependencies: plain PyTorch with sparse COO tensors. **No
torch-geometric required** — the sparse GCN layer and adjacency
normalization are implemented in :mod:`face_stratification.stage_b2.gcn`.
"""

from face_stratification.stage_b2.gcn import (
    GCNEncoder,
    SparseGCNLayer,
    build_multiplex_adjacency_from_nx,
    get_device,
    normalize_adjacency,
)
from face_stratification.stage_b2.gae import StageB2GAE
from face_stratification.stage_b2.rgcn import SparseRGCNEncoder, StageB2RGCN
from face_stratification.stage_b2.contrastive import (
    StageB2GraphContrastive,
    _drop_edges,
    _mask_features,
    _nt_xent_loss,
)
from face_stratification.stage_b2.vgae import StageB2VGAE
from face_stratification.stage_b2.gat import StageB2GAT
from face_stratification.stage_b2.dgi import StageB2DGI
from face_stratification.stage_b2.sweep import (
    SweepConfig,
    SweepResult,
    compute_transdiagnostic_score,
    evaluate_config,
    pick_best_transdiagnostic_config,
    run_sweep,
)

__all__ = [
    "GCNEncoder",
    "SparseRGCNEncoder",
    "SparseGCNLayer",
    "StageB2GAE",
    "StageB2GraphContrastive",
    "StageB2RGCN",
    "build_multiplex_adjacency_from_nx",
    "get_device",
    "normalize_adjacency",
    "StageB2VGAE",
    "StageB2GAT",
    "StageB2DGI",
    # Sweep API
    "SweepConfig",
    "SweepResult",
    "compute_transdiagnostic_score",
    "evaluate_config",
    "pick_best_transdiagnostic_config",
    "run_sweep",
]
