#!/usr/bin/env python
"""One-vs-two biology factors: does the data prefer metabolic + inflammatory as TWO factors (the
certified map) or as ONE 'immunometabolic' factor (the prior ontology / FACE soft-prior candidate #5)?

Fits both structures on the SAME cohort-balanced subsample (default N=1000, cold-started so good R-hat is
honest, not init-stuck), then compares them by WAIC/LOO on the CONTINUOUS block — computed offline from the
posterior Λ/Φ/σ, because that block's likelihood is a pm.Potential (Woodbury) that pymc's log_likelihood
does not capture. Same patients, same items, same z data; only the factor structure differs, so the ELPD
difference is a clean fit-vs-parsimony verdict for the biology block.

    HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_biomerge_test.py --smoke   # wiring (tiny draws)
    python3 scripts/run_job.py biomerge -- env HDF5_USE_FILE_LOCKING=FALSE \
        python notebooks/run_biomerge_test.py --n 1000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def _find_repo_root(start: Path) -> Path:
    for c in (start, *start.parents):
        if (c / "src" / "face" / "models").exists() and (c / "pyproject.toml").exists():
            return c
    raise RuntimeError("repo root not found")


REPO = _find_repo_root(Path(__file__).resolve())
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
for m in [n for n in sys.modules if n == "face" or n.startswith("face.")]:
    f = getattr(sys.modules[m], "__file__", None)
    if f and SRC not in Path(f).resolve().parents:
        del sys.modules[m]

import arviz as az  # noqa: E402
from face.models.bayesian.measurement_model_oop import (  # noqa: E402
    DEFAULT_EXPLICIT_FACTORS, S5_FACTORS, MeasurementConfig, MeasurementDataset, StageDefinition, StageRunner,
)

MERGED_MATRIX = REPO / "configs" / "prior_loading_matrix_v3_biomerge.csv"
MERGED_FACTORS = ["overall_severity", "cognition", "immunometabolic", "sleep",
                  "suicidality", "developmental_risk", "mania_activation", "substance"]


def continuous_block_elpd(idata, M, *, psi_floor: float, n_draws: int, seed: int = 0):
    """Per-patient WAIC ELPD on the marginalized continuous block: z_i|O ~ N(0, (ΛΦΛ'+Ψ)_O).

    Returns (elpd_i, lppd_i, p_waic_i) over patients, where Σ = Λ Φ Λ' + diag((psi_floor+σ)^2).
    Computed offline from the posterior because the block's likelihood is a pm.Potential.
    """
    P = idata.posterior
    Lam = P["Lam"].stack(s=("chain", "draw")).transpose("s", ...).values        # (S,J,F)
    Phi = P["Phi"].stack(s=("chain", "draw")).transpose("s", ...).values        # (S,F,F)
    sig = P["sigma"].stack(s=("chain", "draw")).transpose("s", ...).values      # (S,J)
    S = Lam.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(S, size=min(n_draws, S), replace=False)
    mask = np.isfinite(M)
    patterns, inv = np.unique(mask, axis=0, return_inverse=True)
    N = M.shape[0]
    ll = np.zeros((len(idx), N))
    for d, s in enumerate(idx):
        Sig = Lam[s] @ Phi[s] @ Lam[s].T + np.diag((psi_floor + sig[s]) ** 2)
        for p in range(patterns.shape[0]):
            cols = np.flatnonzero(patterns[p])
            rows = np.flatnonzero(inv == p)
            if cols.size == 0:
                continue
            Sp = Sig[np.ix_(cols, cols)]
            X = np.nan_to_num(M[np.ix_(rows, cols)])
            ll[d, rows] = multivariate_normal.logpdf(X, mean=np.zeros(cols.size), cov=Sp, allow_singular=True)
    lppd = logsumexp(ll, axis=0) - np.log(ll.shape[0])
    p_waic = ll.var(axis=0)
    return lppd - p_waic, lppd, p_waic


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000, help="balanced subsample size (default 1000)")
    p.add_argument("--draws", type=int, default=800)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--elpd-draws", type=int, default=400, help="posterior draws used for the WAIC ELPD")
    p.add_argument("--smoke", action="store_true", help="tiny wiring check (40 draws, 2 chains)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.smoke:
        a.n, a.draws, a.tune, a.chains, a.elpd_draws = 400, 40, 40, 2, 40

    base_cfg = MeasurementConfig().with_gaussian_copula()
    base_cfg = replace(base_cfg, output_dir=base_cfg.output_dir / "copula" / "biomerge",
                       figure_dir=base_cfg.figure_dir / "copula" / "biomerge")
    split_cfg = base_cfg
    merge_cfg = replace(base_cfg, prior_matrix=MERGED_MATRIX)

    common = dict(correlated=True, windows=True, mixed=True, explicit_factors=list(DEFAULT_EXPLICIT_FACTORS),
                  min_cohorts=2, n_subsample=a.n, balanced=True, draws=a.draws, tune=a.tune,
                  chains=a.chains, target_accept=0.95, seed=20260605)
    tag = f"n{a.n}"
    split_stage = StageDefinition(f"split_2factor_{tag}", S5_FACTORS, **common)
    merge_stage = StageDefinition(f"merge_1factor_{tag}", MERGED_FACTORS, **common)

    print(f"[biomerge] split = {len(S5_FACTORS)} factors (metabolic + inflammatory)", flush=True)
    print(f"[biomerge] merge = {len(MERGED_FACTORS)} factors (immunometabolic)", flush=True)
    print(f"[biomerge] N={a.n} draws={a.draws} tune={a.tune} chains={a.chains} (cold-start, same subsample)", flush=True)

    results = {}
    for cfg, stage, label in [(split_cfg, split_stage, "split_2factor"), (merge_cfg, merge_stage, "merge_1factor")]:
        print(f"\n=== fitting {label} ({stage.name}) ===", flush=True)
        t0 = time.time()
        idata, manifest = StageRunner(cfg).run_stage(stage, overwrite=a.overwrite, prev_stage=None)
        diag = manifest.get("diagnostics", {})
        print(f"[{label}] diagnostics={diag}  elapsed={round(time.time()-t0)}s", flush=True)
        results[label] = {"cfg": cfg, "stage": stage, "idata": idata, "diag": diag}

    # same continuous z-data for both (matrix-independent); compute once
    M = MeasurementDataset(split_cfg).mixed(
        S5_FACTORS, explicit_factors=list(DEFAULT_EXPLICIT_FACTORS), min_cohorts=2,
        balanced=True, n_subsample=a.n, seed=20260605).base.M
    psi = float(split_cfg.psi_floor)

    print("\n=== WAIC on the continuous block (offline; higher ELPD = better fit) ===", flush=True)
    elpd = {}
    for label in ("split_2factor", "merge_1factor"):
        ei, lppd, pw = continuous_block_elpd(results[label]["idata"], M, psi_floor=psi, n_draws=a.elpd_draws)
        elpd[label] = ei
        print(f"  {label:14s} elpd_waic={ei.sum():.1f}  (lppd={lppd.sum():.1f}, p_waic={pw.sum():.1f}, "
              f"N={len(ei)})", flush=True)
    diff = elpd["split_2factor"] - elpd["merge_1factor"]
    d_sum = float(diff.sum())
    d_se = float(np.sqrt(len(diff)) * diff.std())
    winner = "split (TWO factors)" if d_sum > 0 else "merge (ONE factor)"
    print(f"\n[verdict] Δelpd (split - merge) = {d_sum:.1f} ± {d_se:.1f} (SE)", flush=True)
    print(f"[verdict] {abs(d_sum):.1f} / {d_se:.1f} = {abs(d_sum)/d_se:.1f} SE favouring {winner}", flush=True)
    print(f"[verdict] decisive if |Δ|/SE > ~2-3; otherwise the simpler ONE-factor model is preferred (parsimony)",
          flush=True)

    out = base_cfg.output_dir / f"biomerge_verdict_{tag}.json"
    out.write_text(json.dumps({
        "N": a.n, "draws": a.draws, "chains": a.chains, "elpd_draws": a.elpd_draws,
        "diagnostics": {k: results[k]["diag"] for k in results},
        "elpd_waic": {k: float(elpd[k].sum()) for k in elpd},
        "delta_elpd_split_minus_merge": d_sum, "se": d_se, "se_units": abs(d_sum) / d_se if d_se else None,
        "winner": winner,
    }, indent=2))
    print(f"[biomerge] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
