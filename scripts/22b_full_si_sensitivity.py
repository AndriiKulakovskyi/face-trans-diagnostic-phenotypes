#!/usr/bin/env python3
"""22b — diagonal-S vs full-S_i tessellation sensitivity (issue P2-04).

The Extreme-Deconvolution mixture (scripts/22) deconvolves each coordinate's measurement error S_i. The
headline equation writes S_i as a full per-patient covariance, but the reported fit used the DIAGONAL
(marginal SDs). The coherent scorer (step 3.1) now exports the FULL per-patient covariance
(``coordinates_cov.npz``), so this re-runs the K=4 tessellation under BOTH and reports whether the full
cross-dimension uncertainty changes the regions — the reviewer's diagonal-vs-full sensitivity, which
either justifies the diagonal approximation or adopts the full S_i.

    python3 scripts/22b_full_si_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.strata.mixture import xd_em  # noqa: E402

M2 = REPO / "results" / "face" / "m2"
REPORTS = REPO / "reports"
SEED, K = 20260609, 4
CANON = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep", "mania_activation",
         "suicidality", "developmental_risk", "substance"]


def _align(mu_a, mu_b):
    """Hungarian alignment of B's components to A's by nearest mean; returns the B-index per A-index."""
    from scipy.optimize import linear_sum_assignment
    cost = np.linalg.norm(mu_a[:, None] - mu_b[None, :], axis=2)
    _, ci = linear_sum_assignment(cost)
    return ci


def main():
    from sklearn.metrics import adjusted_rand_score

    df = pd.read_parquet(M2 / "coordinates_full.parquet")
    X = df[[f"{f}__mean" for f in CANON]].to_numpy()
    Sdiag = df[[f"{f}__sd" for f in CANON]].to_numpy() ** 2
    cov = np.load(M2 / "coordinates_cov.npz")["cov"].astype("float64")   # [N, 9, 9] coherent S_i
    Sfull = cov.copy()
    bad = ~np.isfinite(Sfull).all((1, 2))                                 # prior-dominated guard -> diag
    if bad.any():
        Sfull[bad] = 0.0
        Sfull[bad, np.arange(len(CANON)), np.arange(len(CANON))] = Sdiag[bad]
    off = np.abs(cov[~bad][:, np.triu_indices(len(CANON), 1)[0], np.triu_indices(len(CANON), 1)[1]])

    print("[1/2] XD at K=4 under diagonal S_i...", flush=True)
    fd = xd_em(X, Sdiag, K, seed=SEED)
    print("[2/2] XD at K=4 under full S_i...", flush=True)
    ff = xd_em(X, Sfull, K, seed=SEED)

    map_d, map_f = fd["resp"].argmax(1), ff["resp"].argmax(1)
    ari = float(adjusted_rand_score(map_d, map_f))
    perm = _align(fd["mu"], ff["mu"])
    mean_shift = np.linalg.norm(fd["mu"] - ff["mu"][perm], axis=1)
    share_d = np.bincount(map_d, minlength=K) / len(X)
    share_f = np.bincount(map_f, minlength=K) / len(X)

    tab = pd.DataFrame({
        "metric": ["BIC", "max |component mean shift|", "MAP partition ARI (diag vs full)",
                   "mean |off-diagonal S_i| (typical cross-dim uncertainty)"],
        "diagonal_S": [round(fd["bic"], 1), "", "", ""],
        "full_S": [round(ff["bic"], 1), round(float(mean_shift.max()), 3), round(ari, 3),
                   round(float(off.mean()), 4)],
    })
    tab.to_csv(REPORTS / "22b_full_si_sensitivity.csv", index=False)

    negligible = ari > 0.9 and mean_shift.max() < 0.15
    md = ["# 22b — diagonal vs full S_i tessellation sensitivity (P2-04)", "",
          "The reported tessellation deconvolves the **diagonal** measurement error (marginal SDs). The "
          "coherent scorer now exports the **full** per-patient covariance S_i, so we re-run the K=4 XD "
          "tessellation under both. If the regions are unchanged, the diagonal approximation is justified.",
          "", "## Diagonal vs full S_i (K=4)", tab.to_markdown(index=False), "",
          (f"- **Diagonal S_i is a justified approximation:** the MAP partitions agree (ARI {ari:.3f}), "
           f"component means move ≤ {mean_shift.max():.3f}, and the typical off-diagonal coordinate "
           f"uncertainty is small ({off.mean():.4f}). The reported diagonal-S tessellation stands; the "
           "full-S_i arm is available."
           if negligible else
           f"- **Full S_i changes the tessellation** (ARI {ari:.3f}, max mean shift {mean_shift.max():.3f}); "
           "adopt the full-S_i fit as primary and update the strata."),
          "", "- population shares — diagonal "
          f"{[round(float(v), 3) for v in share_d]} · full {[round(float(v), 3) for v in share_f]}.", ""]
    (REPORTS / "22b_full_si_sensitivity_report.md").write_text("\n".join(md))
    print("\n".join(md))
    print("\nwrote reports/22b_full_si_sensitivity_report.md (+ .csv)")


if __name__ == "__main__":
    main()
