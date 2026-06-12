"""Frozen archetype membership per visit — project follow-up coordinates onto the M2 archetypes.

Arm B (G-residualized, the specific-axis subspace) is the PRIMARY persistence vehicle (docs/TEMPORAL_MODEL.md
§1.4/§6): it measures corner identity *independent of* severity. Arm A (all-9) is the contextual view. Pure
reuse of the M2 simplex projector (`project_to_Z` / `project_draws`) — no re-fit, no re-discovery.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from face.strata.archetypes import project_draws, project_to_Z


def _entropy(W: np.ndarray) -> np.ndarray:
    """Normalized Shannon entropy of each patient's simplex weights (0 = pure archetype, 1 = uniform)."""
    A = W.shape[1]
    P = np.clip(W, 1e-12, 1.0)
    return -(P * np.log(P)).sum(1) / np.log(A)


def archetype_membership(X_mean: np.ndarray, draws: np.ndarray, cols, Z: np.ndarray,
                         names: list[str], *, prefix: str, index, n_draw: int = 40, seed: int = 0):
    """Project coordinates onto the fixed archetypes `Z` → per-patient simplex weights + uncertainty.

    `X_mean` [N, D] are the coordinates on the D axes Z is defined on; `draws` [S, N, D_full] are the
    posterior draws; `cols` selects the D axes within `draws`. Returns a DataFrame (indexed by `index`)
    of weights `{prefix}_w{a}` (+ `_sd`), `{prefix}_dominant`/`_dominant_name`/`_entropy`.
    """
    W = project_to_Z(X_mean, Z)                                  # [N, A] simplex weights
    Wsd = project_draws(Z, draws, cols, n_draw=n_draw, seed=seed)["sd"]   # [N, A] draw-wise SD
    A = W.shape[1]
    dom = W.argmax(1)
    out = {f"{prefix}_w{a}": np.round(W[:, a], 4) for a in range(A)}
    out.update({f"{prefix}_w{a}_sd": np.round(Wsd[:, a], 4) for a in range(A)})
    out[f"{prefix}_dominant"] = dom
    out[f"{prefix}_dominant_name"] = [names[d] for d in dom]
    out[f"{prefix}_entropy"] = np.round(_entropy(W), 4)
    return pd.DataFrame(out, index=index)
