#!/usr/bin/env python
"""One-vs-two biology factors: does the data prefer metabolic + inflammatory as TWO factors (the
certified map) or as ONE 'immunometabolic' factor (the prior ontology / FACE soft-prior candidate #5)?

The metabolic + inflammatory indicators are 100% continuous, so this question lives entirely in the
CONTINUOUS BACKBONE (the certified continuous core, S1: G + cognition + biology + sleep) — which is the
marginalized Woodbury fit that converges cleanly cold (R-hat ~1.01), unlike the full mixed model (which
needs the staged warm-start and otherwise mixes at R-hat ~1.7 and would make the comparison untrustworthy).

We fit both structures on the SAME cohort-balanced subsample, then compare by WAIC on the continuous block,
computed OFFLINE from the posterior Λ/Φ/σ (the block's likelihood is a pm.Potential pymc's log_likelihood
does not capture). Same patients, same items, same z data; only the biology factor structure differs.

    HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_biomerge_test.py --smoke   # wiring (tiny draws)
    python3 scripts/run_job.py biomerge -- env HDF5_USE_FILE_LOCKING=FALSE \
        python notebooks/run_biomerge_test.py --n 2000
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

from face.measurement.engine import (  # noqa: E402
    S1_FACTORS,
    MeasurementConfig,
    MeasurementDataset,
    StageDefinition,
    StageRunner,
)

MERGED_MATRIX = REPO / "configs" / "loading_matrix.immunometabolic.csv"
# Continuous core (S1) with the biology block as TWO factors (split) vs ONE (merge).
SPLIT_FACTORS = list(S1_FACTORS)                                              # G, cognition, metabolic, inflammatory, sleep
MERGE_FACTORS = ["overall_severity", "cognition", "immunometabolic", "sleep"]  # biology collapsed to one


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
    p.add_argument("--n", type=int, default=2000, help="balanced subsample size (default 2000)")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--elpd-draws", type=int, default=400, help="posterior draws used for the WAIC ELPD")
    p.add_argument("--smoke", action="store_true", help="tiny wiring check (60 draws, 2 chains)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.smoke:
        a.n, a.draws, a.tune, a.chains, a.elpd_draws = 800, 60, 60, 2, 60

    base_cfg = MeasurementConfig().with_gaussian_copula()
    base_cfg = replace(base_cfg, output_dir=base_cfg.output_dir / "copula" / "biomerge",
                       figure_dir=base_cfg.figure_dir / "copula" / "biomerge")
    split_cfg = base_cfg
    merge_cfg = replace(base_cfg, prior_matrix=MERGED_MATRIX)

    # Continuous backbone (mixed=False = marginalized Woodbury; converges cleanly cold).
    common = dict(correlated=True, windows=True, mixed=False, min_cohorts=2, n_subsample=a.n,
                  balanced=True, draws=a.draws, tune=a.tune, chains=a.chains, target_accept=0.95, seed=20260605)
    tag = f"n{a.n}"
    split_stage = StageDefinition(f"split_2factor_{tag}", SPLIT_FACTORS, **common)
    merge_stage = StageDefinition(f"merge_1factor_{tag}", MERGE_FACTORS, **common)

    print(f"[biomerge] split = {len(SPLIT_FACTORS)} factors {SPLIT_FACTORS}", flush=True)
    print(f"[biomerge] merge = {len(MERGE_FACTORS)} factors {MERGE_FACTORS}", flush=True)
    print(f"[biomerge] continuous backbone (copula, marginalized); N={a.n} draws={a.draws} "
          f"tune={a.tune} chains={a.chains}", flush=True)

    results = {}
    for cfg, stage, label in [(split_cfg, split_stage, "split_2factor"), (merge_cfg, merge_stage, "merge_1factor")]:
        print(f"\n=== fitting {label} ({stage.name}) ===", flush=True)
        t0 = time.time()
        idata, manifest = StageRunner(cfg).run_stage(stage, overwrite=a.overwrite, prev_stage=None)
        diag = manifest.get("diagnostics", {})
        print(f"[{label}] diagnostics={diag}  elapsed={round(time.time()-t0)}s", flush=True)
        results[label] = {"cfg": cfg, "stage": stage, "idata": idata, "diag": diag}

    # same continuous z-data for both; verify the two cores hold identical items in identical order
    # (only the biology factor *labels* differ), so the shared M aligns with both models' Lam rows.
    split_core = MeasurementDataset(split_cfg).core(
        SPLIT_FACTORS, correlated=True, windows=True, balanced=True, n_subsample=a.n, seed=20260605)
    merge_core = MeasurementDataset(merge_cfg).core(
        MERGE_FACTORS, correlated=True, windows=True, balanced=True, n_subsample=a.n, seed=20260605)
    assert list(split_core.items) == list(merge_core.items), "split/merge cores differ in items — comparison invalid"
    assert np.allclose(np.nan_to_num(split_core.M), np.nan_to_num(merge_core.M)), "split/merge M differ"
    M = split_core.M
    psi = float(split_cfg.psi_floor)
    print(f"[biomerge] continuous block: {M.shape[0]} patients x {M.shape[1]} items (identical in both fits)", flush=True)

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
    print("[verdict] decisive if |Δ|/SE > ~2-3; otherwise the simpler ONE-factor model is preferred (parsimony)",
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
