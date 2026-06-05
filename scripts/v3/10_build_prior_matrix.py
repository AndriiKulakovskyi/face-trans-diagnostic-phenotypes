#!/usr/bin/env python3
"""Thin CLI: expand the V3 config ontology into the soft-prior loading matrix.

    python3 scripts/v3/10_build_prior_matrix.py

Reads  configs/dimensions.yaml + priors.yaml + likelihood_map_v3.yaml
Writes configs/prior_loading_matrix_v3.csv + configs/likelihoods.yaml

This is the config-first replacement for the hard-coded model SPEC: the matrix it
produces is the single source the Bayesian ESEM-bifactor engine reads to build Lambda.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from v3.priors import build_prior_matrix  # noqa: E402

if __name__ == "__main__":
    build_prior_matrix()
