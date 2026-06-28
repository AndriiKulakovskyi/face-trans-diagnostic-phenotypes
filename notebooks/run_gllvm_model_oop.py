#!/usr/bin/env python
"""Variational GLLVM atlas engine — driver for the 8-factor operational map.

Fits the variational mixed-likelihood GLLVM (PyTorch SVI) on the same data contract and
ontology as the certified NUTS M1, but trained by stochastic variational optimization.  An
exploration / acceleration arm — congruent-with / authority-defers-to the copula NUTS map.

    # fast wiring check (tiny epochs, balanced subsample, single rung):
    HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_gllvm_model_oop.py --mode smoke
    # full fit (long; run detached):
    python3 scripts/run_job.py gllvm -- python notebooks/run_gllvm_model_oop.py \
        --mode production --consolidate

The JAX XLA_FLAGS gotcha does NOT apply here (this is torch, not numpyro).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for c in (start, *start.parents):
        if (c / "src" / "face" / "models").exists() and (c / "pyproject.toml").exists():
            return c
    raise RuntimeError(f"Could not locate FACE repository root from {start}")


REPO = _find_repo_root(Path(__file__).resolve())
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
for m in [n for n in sys.modules if n == "face" or n.startswith("face.")]:
    f = getattr(sys.modules[m], "__file__", None)
    if f and SRC not in Path(f).resolve().parents:
        del sys.modules[m]

from face.models.variational.gllvm_model_oop import GLLVMConfig, GLLVMRunner  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "medium", "production"], default="smoke",
                   help="smoke: tiny epochs + balanced subsample, single rung; "
                        "medium: full-N, fewer epochs; production: full 2-rung plan.")
    p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    p.add_argument("--epochs", type=int, default=None, help="override the epoch count")
    p.add_argument("--lr", type=float, default=None, help="override the learning rate")
    p.add_argument("--seed", type=int, default=20260605)
    p.add_argument("--stage", default=None, help="run a single stage by name (default: full plan)")
    p.add_argument("--overwrite", action="store_true", help="refit even if a stage is cached")
    p.add_argument("--consolidate", action="store_true", help="write the consolidate/ hand-off")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> GLLVMConfig:
    config = GLLVMConfig()
    config = replace(config, device=args.device, seed=args.seed)
    if args.mode == "smoke":
        config = config.with_smoke_defaults()
    elif args.mode == "medium":
        config = replace(config, epochs=1500)
    # production: defaults (epochs=4000, full 2-rung plan)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.lr is not None:
        config = replace(config, lr=args.lr)
    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)
    runner = GLLVMRunner(config)
    print(f"[gllvm] mode={args.mode} device={config.device} factors={list(config.factors)}", flush=True)
    print(f"[gllvm] prior_matrix={config.prior_matrix.name} orthogonal={config.orthogonal_factors} "
          f"specific_cross={config.specific_cross} likelihood={config.likelihood_mode}", flush=True)

    t0 = time.time()
    if args.stage is not None:
        stage = next((s for s in config.stage_plan if s.name == args.stage), None)
        if stage is None:
            raise SystemExit(f"unknown stage {args.stage}; have {[s.name for s in config.stage_plan]}")
        fit = runner.run_stage(stage, overwrite=args.overwrite)
    else:
        fit = runner.run_plan(overwrite=args.overwrite)
    elapsed = time.time() - t0

    if args.consolidate:
        runner.consolidate(fit)

    hist = fit["history"]
    summary = {
        "stage": fit["stage"],
        "factors": fit["factor_cols"],
        "elapsed_sec": round(elapsed, 1),
        "final": {k: round(float(hist[-1][k]), 3) for k in ("loss", "nll", "kl", "penalty")} if hist else {},
        "out": str(config.output_dir / fit["stage"]),
    }
    print(json.dumps(summary, indent=2), flush=True)

    # Cheap sanity checks (the gate report).
    import numpy as np

    model, data = fit["model"], fit["data"]
    phi = model.phi_matrix()
    fc = data.factor_cols
    for f in ("overall_severity", *config.orthogonal_factors):
        if f in fc:
            i = fc.index(f)
            offmax = float(np.abs(np.delete(phi[i], i)).max())
            print(f"[gllvm] Phi[{f}] off-diagonal max = {offmax:.6f} (expect 0 — orthogonal)", flush=True)
    lam, free = model.loadings(), data.ontology.free_mask
    print(f"[gllvm] max |loading| on hard-zero cells = {float(np.abs(lam[~free]).max()):.6f} (expect 0)", flush=True)
    ncross = sum(1 for v in data.ontology.kind.values() if v == "cross")
    print(f"[gllvm] freed cross-loadings = {ncross} (expect 3: ctq37/psqi11/psqi17 -> cognition)", flush=True)
    print(f"[gllvm] DONE -> {config.output_dir / fit['stage']}", flush=True)
    print("[gllvm] next (after gate): validate vs NUTS with "
          "`python notebooks/run_gllvm_validation.py`", flush=True)


if __name__ == "__main__":
    main()
