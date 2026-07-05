"""Logical-name -> config-file resolution — one table instead of scattered ``v3`` path literals.

Renaming a config is one edit here, not a grep across ~10 modules. Names carry *method* meaning
(``immunometabolic`` = the biology merge; ``immunometabolic_crossload`` = merge + the 3 earned
cross-loadings), never a version. ``loading_matrix.primary`` is the operative matrix M2/M3 read.
"""
from __future__ import annotations

from .paths import CONFIGS

_CONFIGS: dict[str, str] = {
    # item x factor prior loading matrices
    "loading_matrix": "loading_matrix.csv",                                       # M1 base (9-factor prior)
    "loading_matrix.immunometabolic": "loading_matrix.immunometabolic.csv",       # biology merge (8-factor)
    "loading_matrix.immunometabolic_crossload": "loading_matrix.immunometabolic_crossload.csv",  # merge + 3 cross-loadings
    "loading_matrix.primary": "loading_matrix.immunometabolic_crossload.csv",     # operative map M2/M3 read
    "loading_priors": "loading_priors.csv",
    # ontology + likelihood + prior specs
    "ontology": "dimensions.yaml",
    "ontology_candidates": "ontology_candidates.yaml",
    "likelihoods": "likelihoods.yaml",
    "likelihood_map": "likelihood_map.yaml",
    "priors": "priors.yaml",
    # milestone outcome definitions
    "prognosis_outcomes": "prognosis_outcomes.yaml",
    "treatment_outcomes": "treatment_outcomes.yaml",
}


def path(name: str):
    """Resolve a logical config name to an absolute ``Path`` under ``configs/``."""
    try:
        return CONFIGS / _CONFIGS[name]
    except KeyError as e:
        raise KeyError(f"unknown config '{name}'; known: {sorted(_CONFIGS)}") from e
