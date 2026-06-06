"""V3 priors module — build the soft-prior loading matrix from configs."""
from .build_matrix import build_prior_matrix, load_configs, resolve_ontology

__all__ = ["build_prior_matrix", "load_configs", "resolve_ontology"]
