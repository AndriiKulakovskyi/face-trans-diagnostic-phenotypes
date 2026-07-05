"""Configuration layer: canonical paths, logical-name config registry, typed loaders.

    from face.config import paths, registry, loader
    matrix_path = registry.path("loading_matrix.primary")
    df = loader.matrix("loading_matrix")
    out = paths.results("m1")            # -> <repo>/results/m1_measurement
"""
from __future__ import annotations

from . import loader, paths, registry

__all__ = ["paths", "registry", "loader"]
