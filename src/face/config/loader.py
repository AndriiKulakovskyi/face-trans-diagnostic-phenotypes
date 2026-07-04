"""Typed accessors over the config registry — read the ontology / matrices / outcomes by logical name."""
from __future__ import annotations

from functools import cache
from typing import Any

import pandas as pd
import yaml

from . import registry

_OUTCOME_CONFIG = {"m4": "prognosis_outcomes", "prognosis": "prognosis_outcomes",
                   "m5": "treatment_outcomes", "treatment": "treatment_outcomes"}


def matrix(name: str = "loading_matrix.primary") -> pd.DataFrame:
    """Load an item x factor loading matrix by logical name."""
    return pd.read_csv(registry.path(name))


@cache
def yaml_config(name: str) -> dict[str, Any]:
    """Load a YAML config by logical name (``ontology``, ``priors``, ``likelihoods``, ...)."""
    with open(registry.path(name)) as fh:
        return yaml.safe_load(fh)


def ontology() -> dict[str, Any]:
    return yaml_config("ontology")


def outcomes(milestone: str) -> dict[str, Any]:
    """Outcome definitions for a milestone key (``m4``/``prognosis`` or ``m5``/``treatment``)."""
    return yaml_config(_OUTCOME_CONFIG[milestone])
