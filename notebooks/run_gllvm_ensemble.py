#!/usr/bin/env python
"""Seed ensemble for the variational GLLVM — loading credible bands + Phi recovery check.

Runs ``--seeds`` independent fits (default low-rank q, which is the only thing that moves Phi),
stacks the loading matrices to a per-cell **seed-ensemble CI** (mean ± 1.96·SD across seeds),
and writes a CI-aware ``loadings_summary.csv`` in the NUTS export schema (so the dot-atlas and
any downstream consumer get populated ``ci_low``/``ci_high``/``excludes_zero``).  The
ensemble-mean Phi is the low-rank Phi-recovery estimate (vs the mean-field baseline 0.083).

Caveat: a seed ensemble captures **optimization / initialization variability**, a lower bound
on the true posterior width — it is an identifiability/stability signal, not a calibrated
Bayesian CI.  NUTS remains the uncertainty authority.

    HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_gllvm_ensemble.py --seeds 6 --q-rank 2
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyses.variational_gllvm import validate as V  # noqa: E402
from analyses.variational_gllvm.engine import (  # noqa: E402
    G_KEY,
    GLLVMConfig,
    GLLVMProjector,
    GLLVMRunner,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=6, help="number of ensemble fits")
    p.add_argument("--base-seed", type=int, default=20260605)
    p.add_argument("--q-rank", type=int, default=2, help="low-rank q (0 = mean-field)")
    p.add_argument("--full-cov", action="store_true", help="full per-patient covariance q (overrides q-rank)")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--keep-seed-dirs", action="store_true", help="don't delete per-seed outputs")
    return p.parse_args()


def ci_loadings_summary(data, lam_stack: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """CI-aware loadings_summary from a (k, J, F) loading stack: mean ± 1.96·SD per free cell."""
    mean = lam_stack.mean(0)
    sd = lam_stack.std(0, ddof=1) if lam_stack.shape[0] > 1 else np.zeros_like(mean)
    lo, hi = mean - 1.96 * sd, mean + 1.96 * sd
    kind = data.ontology.kind
    fidx = {f: i for i, f in enumerate(data.factor_cols)}
    rows = []
    for j, item in enumerate(data.items):
        h = data.home[j]
        fam = data.likelihood_families[j]
        block = data.blocks[j]
        for c, factor in enumerate(data.factor_cols):
            if not bool(data.ontology.free_mask[j, c]):
                continue
            kk = kind.get((j, c)) or ("primary" if factor == h else
                                     "bifactor_G" if factor == G_KEY else "cross")
            m = float(mean[j, c])
            rows.append(dict(item=item, factor=factor, home=h or "", block=block,
                             likelihood_family=fam, kind=kk, loading=m, abs_loading=abs(m),
                             ci_low=float(lo[j, c]), ci_high=float(hi[j, c]),
                             excludes_zero=bool(lo[j, c] > 0 or hi[j, c] < 0)))
    df = pd.DataFrame(rows)
    df["__f"] = df["factor"].map(fidx)
    return df.sort_values(["item", "__f"]).drop(columns="__f").reset_index(drop=True), mean


def main() -> None:
    args = parse_args()
    base = GLLVMConfig()
    ens_dir = base.output_dir / ("ensemble_fullcov" if args.full_cov else "ensemble")
    ens_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.base_seed + i for i in range(args.seeds)]
    qdesc = "full_cov" if args.full_cov else f"q_rank={args.q_rank}"
    print(f"[ens] {args.seeds} fits, {qdesc}, epochs={args.epochs}, seeds={seeds}", flush=True)

    lam_list, phi_list, coord_list, seed_dirs = [], [], [], []
    ref_fit = None
    for s in seeds:
        cfg = replace(base, seed=s, q_rank=0 if args.full_cov else args.q_rank,
                      full_cov=args.full_cov, epochs=args.epochs,
                      output_dir=ens_dir / f"seed_{s}")
        runner = GLLVMRunner(cfg)
        fit = runner.run_plan(overwrite=True)
        lam_list.append(fit["model"].loadings())
        phi_list.append(fit["model"].phi_matrix())
        coord_list.append(GLLVMProjector(cfg).coordinates_frame(fit))
        seed_dirs.append(cfg.output_dir)
        ref_fit = fit
        print(f"[ens] seed {s} done (final -ELBO {fit['history'][-1]['loss']:.0f})", flush=True)

    lam_stack = np.stack(lam_list)  # (k, J, F)
    phi_mean = np.mean(phi_list, 0)
    data = ref_fit["data"]
    factors = data.factor_cols

    loadings_df, lam_mean = ci_loadings_summary(data, lam_stack)
    loadings_df.to_csv(ens_dir / "loadings_summary.csv", index=False)
    pd.DataFrame(phi_mean, index=factors, columns=factors).to_csv(ens_dir / "phi.csv")
    # Ensemble-mean coordinates (mean of per-fit means; sd = mean of per-fit posterior sds).
    coords = sum(c[[col for col in c.columns if col.endswith("__mean")]] for c in coord_list) / len(coord_list)
    base_coords = coord_list[0].copy()
    for col in coords.columns:
        base_coords[col] = coords[col]
    base_coords.to_parquet(ens_dir / "coordinates.parquet")
    np.savez_compressed(ens_dir / "ensemble_loadings.npz", lam_stack=lam_stack,
                        items=np.array(data.items), factors=np.array(factors))

    # Validate the ensemble mean vs the cached NUTS targets (no idata reload).
    nuts_loadings = pd.read_csv(base.output_dir / "nuts_targets" / "loadings_summary.csv")
    nuts_phi = pd.read_csv(base.output_dir / "nuts_targets" / "phi.csv", index_col=0)
    nuts_coords = pd.read_parquet(
        REPO / "results/m2_strata/coordinates/coordinates_full.parquet"
    ).astype({"cohort": str, "patient_id": str}).set_index(["cohort", "patient_id"])
    rep = V.run_congruence(ens_dir, nuts_loadings, nuts_phi, nuts_coords=nuts_coords,
                           out_csv=ens_dir / "validation_report.csv")

    corr = [f for f in factors if f not in ("overall_severity", "substance")]
    import itertools
    vm = float(np.mean([abs(phi_mean[factors.index(a), factors.index(b)])
                        for a, b in itertools.combinations(corr, 2)]))
    nm = float(np.mean([abs(nuts_phi.loc[a, b]) for a, b in itertools.combinations(corr, 2)]))
    n_credible = int(loadings_df["excludes_zero"].sum())
    summary = {
        "seeds": seeds, "q_rank": 0 if args.full_cov else args.q_rank, "full_cov": args.full_cov,
        "verdict": rep["verdict"],
        "phi_mean_offdiag_VI": round(vm, 4), "phi_mean_offdiag_NUTS": round(nm, 4),
        "phi_shrinkage": round(1 - vm / nm, 3), "phi_max_diff": rep["phi"]["max_abs_offdiag_diff"],
        "loadings_credible": f"{n_credible}/{len(loadings_df)}",
    }
    (ens_dir / "ensemble_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n=== ENSEMBLE SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2, default=float), flush=True)
    print("\nTucker:\n", rep["tucker"][["factor", "tucker", "pass"]].to_string(index=False), flush=True)
    print("\nCoord:\n", rep["coordinates"][["factor", "r", "pass"]].to_string(index=False), flush=True)
    print(f"\nPhi: VI mean|offdiag|={vm:.3f} vs NUTS {nm:.3f} "
          f"(shrinkage {1 - vm / nm:.0%}; baseline mean-field was 0.083 / 21%)", flush=True)
    for cell in ("cognition~sleep", "cognition~developmental_risk", "cognition~suicidality"):
        a, b = cell.split("~")
        print(f"  {cell}: VI={phi_mean[factors.index(a), factors.index(b)]:+.3f} "
              f"NUTS={nuts_phi.loc[a, b]:+.3f}", flush=True)

    if not args.keep_seed_dirs:
        for d in seed_dirs:
            shutil.rmtree(d, ignore_errors=True)
    print(f"\n[ens] DONE -> {ens_dir}", flush=True)


if __name__ == "__main__":
    main()
