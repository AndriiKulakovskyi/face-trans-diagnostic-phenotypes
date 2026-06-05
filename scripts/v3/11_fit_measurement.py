#!/usr/bin/env python3
"""Thin CLI: fit one stage of the V3 soft-prior ESEM-bifactor measurement model.

    python3 scripts/v3/11_fit_measurement.py --stage 1
    python3 scripts/v3/11_fit_measurement.py --stage 0 --smoke

Stages are defined in configs/bayesian_model.yaml; the engine lives in
src/v3/latent_models/bayesian. Outputs -> results/v3/bayesian/stage{S}/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from v3.latent_models.bayesian.fit import main  # noqa: E402

if __name__ == "__main__":
    main()
