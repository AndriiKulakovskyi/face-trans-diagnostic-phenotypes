"""Method registry: enumerates all 16 embedding + 10 clustering methods.

Reads hyperparameters from YAML config. Single point of truth for which
methods are enabled and with what parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from face_stratification.models.base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).parents[3] / "config" / "face_stratification" / "stage_b_config.yaml"


def load_stage_b_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the Stage B YAML config."""
    p = Path(path) if path else _DEFAULT_CONFIG
    with open(p) as f:
        return yaml.safe_load(f)


def get_embedding_methods(
    config: dict[str, Any] | None = None,
) -> dict[str, BaseEmbeddingModel]:
    """Instantiate all enabled embedding methods from config.

    Returns ``{method_name: instantiated_model}`` for methods where
    ``enabled: true`` in the YAML config.
    """
    if config is None:
        config = load_stage_b_config()

    methods: dict[str, BaseEmbeddingModel] = {}
    em = config.get("embedding_methods", {})

    # ── Feature baselines ──
    if em.get("raw_baseline", {}).get("enabled", False):
        from face_stratification.models.raw_baseline import RawFeatureBaseline
        methods["raw_baseline"] = RawFeatureBaseline()

    if em.get("pca", {}).get("enabled", False):
        from face_stratification.models.baselines import TransdiagnosticPCA
        c = em["pca"]
        methods["pca"] = TransdiagnosticPCA(
            n_components=c.get("n_components", 8),
            l2_normalize=True,
        )

    if em.get("kernel_pca", {}).get("enabled", False):
        from face_stratification.models.kernel_methods import KernelPCAEmbedding
        c = em["kernel_pca"]
        methods["kernel_pca"] = KernelPCAEmbedding(
            n_components=c.get("n_components", 16),
            kernel=c.get("kernel", "rbf"),
            gamma=c.get("gamma", "auto"),
        )

    if em.get("umap", {}).get("enabled", False):
        from face_stratification.models.baselines import TransdiagnosticUMAP
        c = em["umap"]
        methods["umap"] = TransdiagnosticUMAP(
            n_components=c.get("n_components", 16),
            n_neighbors=c.get("n_neighbors", 30),
            min_dist=c.get("min_dist", 0.2),
            metric=c.get("metric", "cosine"),
        )

    # ── Deep feature baselines ──
    if em.get("vanilla_ae", {}).get("enabled", False):
        from face_stratification.models.deep_baselines import VanillaAE
        c = em["vanilla_ae"]
        methods["vanilla_ae"] = VanillaAE(
            bottleneck_dim=c.get("bottleneck_dim", 32),
            hidden_dims=c.get("hidden_dims", [128, 64]),
            n_epochs=c.get("n_epochs", 200),
            lr=c.get("lr", 1e-3),
            batch_size=c.get("batch_size", 256),
        )

    if em.get("vae", {}).get("enabled", False):
        from face_stratification.models.deep_baselines import VAE
        c = em["vae"]
        methods["vae"] = VAE(
            bottleneck_dim=c.get("bottleneck_dim", 32),
            hidden_dims=c.get("hidden_dims", [128, 64]),
            n_epochs=c.get("n_epochs", 200),
            lr=c.get("lr", 1e-3),
            beta=c.get("beta", 1.0),
            batch_size=c.get("batch_size", 256),
        )

    # ── Graph spectral ──
    if em.get("transdiagnostic_spectral", {}).get("enabled", False):
        from face_stratification.models.spectral import TransdiagnosticSpectral
        c = em["transdiagnostic_spectral"]
        methods["transdiagnostic_spectral"] = TransdiagnosticSpectral(
            n_components=c.get("n_components", 16),
            l2_normalize=True,
        )

    if em.get("multiplex_spectral", {}).get("enabled", False):
        from face_stratification.models.spectral import MultiplexSpectral
        c = em["multiplex_spectral"]
        methods["multiplex_spectral"] = MultiplexSpectral(
            n_components=c.get("n_components", 32),
            l2_normalize=True,
        )

    if em.get("diffusion_map", {}).get("enabled", False):
        from face_stratification.models.kernel_methods import DiffusionMapEmbedding
        c = em["diffusion_map"]
        methods["diffusion_map"] = DiffusionMapEmbedding(
            n_components=c.get("n_components", 16),
            diffusion_time=c.get("diffusion_time", 2),
            alpha=c.get("alpha", 0.5),
        )

    # ── GNN methods ──
    if em.get("gae", {}).get("enabled", False):
        from face_stratification.stage_b2.gae import StageB2GAE
        c = em["gae"]
        methods["gae"] = StageB2GAE(
            hidden_dim=c.get("hidden_dim", 64),
            out_dim=c.get("out_dim", 32),
            n_layers=c.get("n_layers", 2),
            n_epochs=c.get("n_epochs", 200),
            lr=c.get("lr", 0.01),
        )

    if em.get("vgae", {}).get("enabled", False):
        from face_stratification.stage_b2.vgae import StageB2VGAE
        c = em["vgae"]
        methods["vgae"] = StageB2VGAE(
            hidden_dim=c.get("hidden_dim", 64),
            out_dim=c.get("out_dim", 32),
            n_layers=c.get("n_layers", 2),
            beta=c.get("beta", 1.0),
            n_epochs=c.get("n_epochs", 200),
            lr=c.get("lr", 0.01),
        )

    if em.get("graphcl", {}).get("enabled", False):
        from face_stratification.stage_b2.contrastive import StageB2GraphContrastive
        c = em["graphcl"]
        methods["graphcl"] = StageB2GraphContrastive(
            hidden_dim=c.get("hidden_dim", 64),
            out_dim=c.get("out_dim", 32),
            n_layers=c.get("n_layers", 3),
            n_epochs=c.get("n_epochs", 300),
            lr=c.get("lr", 1e-3),
            temperature=c.get("temperature", 0.5),
            p_edge=c.get("p_edge_drop", 0.2),
            p_feat=c.get("p_feature_mask", 0.2),
            include_edge_types=(["transdiagnostic"] if c.get("edge_filter") == "transdiagnostic_only" else None),
        )

    if em.get("gat", {}).get("enabled", False):
        from face_stratification.stage_b2.gat import StageB2GAT
        c = em["gat"]
        methods["gat"] = StageB2GAT(
            hidden_dim=c.get("hidden_dim", 64),
            out_dim=c.get("out_dim", 32),
            n_heads=c.get("n_heads", 4),
            n_layers=c.get("n_layers", 2),
            n_epochs=c.get("n_epochs", 200),
            lr=c.get("lr", 0.005),
        )

    if em.get("dgi", {}).get("enabled", False):
        from face_stratification.stage_b2.dgi import StageB2DGI
        c = em["dgi"]
        methods["dgi"] = StageB2DGI(
            hidden_dim=c.get("hidden_dim", 64),
            out_dim=c.get("out_dim", 32),
            n_layers=c.get("n_layers", 2),
            n_epochs=c.get("n_epochs", 300),
            lr=c.get("lr", 1e-3),
        )

    if em.get("rgcn", {}).get("enabled", False):
        from face_stratification.stage_b2.rgcn import StageB2RGCN
        c = em["rgcn"]
        methods["rgcn"] = StageB2RGCN(
            hidden_dim=c.get("hidden_dim", 64),
            out_dim=c.get("out_dim", 32),
            n_layers=c.get("n_layers", 2),
            n_bases=c.get("n_bases", 8),
            n_epochs=c.get("n_epochs", 200),
            lr=c.get("lr", 0.01),
        )

    # ── Multi-view ──
    if em.get("weighted_concatenated", {}).get("enabled", False):
        from face_stratification.models.composite import WeightedConcatenatedEmbedding
        methods["weighted_concatenated"] = WeightedConcatenatedEmbedding.build_default()

    logger.info("Loaded %d embedding methods from config", len(methods))
    return methods


def get_clustering_methods(
    config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return clustering method specs from config.

    Returns ``{method_name: {params}}`` for enabled methods.
    Methods are invoked via the functions in ``clustering.algorithms``.
    """
    if config is None:
        config = load_stage_b_config()

    cm = config.get("clustering_methods", {})
    methods: dict[str, dict[str, Any]] = {}

    for name, spec in cm.items():
        if spec.get("enabled", False):
            methods[name] = {k: v for k, v in spec.items() if k != "enabled"}

    logger.info("Loaded %d clustering methods from config", len(methods))
    return methods
