#!/usr/bin/env python
"""Validate the variational GLLVM map against the NUTS copula M1 fit.

Reads the VI ``consolidate/`` exports and compares them to the canonical 8-factor copula
NUTS fit (loadings / Phi exported on the fly from its idata).  Reports Tucker congruence per
factor, Phi agreement (immunometabolic block + the orthogonal G/substance rows), and — if a
NUTS coordinate parquet is supplied — per-factor coordinate correlation.

    HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_gllvm_validation.py \
        --nuts-idata results/m1_measurement/primary/idata.nc

Worded "congruent", not "certified": NUTS remains the inferential authority.
"""
from __future__ import annotations

import argparse
import sys
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

import pandas as pd  # noqa: E402

from analyses.variational_gllvm import validate as V  # noqa: E402
from analyses.variational_gllvm.engine import GLLVMConfig  # noqa: E402

DEFAULT_NUTS_IDATA = (
    REPO / "results" / "face" / "oop_measurement" / "copula" / "weighted_8d" / "hs_s5_merged_xc" / "idata.nc"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vi-dir", default=str(GLLVMConfig().output_dir / "consolidate"),
                   help="VI consolidate/ (or stage) directory with the exports")
    p.add_argument("--nuts-idata", default=str(DEFAULT_NUTS_IDATA),
                   help="cached 8-factor copula NUTS idata to export targets from")
    p.add_argument("--nuts-coords",
                   default=str(REPO / "results" / "face" / "strata_oop" / "coordinates" / "coordinates_full.parquet"),
                   help="NUTS coordinate parquet ({factor}__mean/__n_obs schema); default = the "
                        "strata copula coordinates")
    p.add_argument("--out", default=str(GLLVMConfig().output_dir / "validation_report.csv"))
    return p.parse_args()


def main() -> None:
    import json

    args = parse_args()
    idata_path = Path(args.nuts_idata)
    if not idata_path.exists():
        raise SystemExit(
            f"NUTS idata not found: {idata_path}\n"
            "Run the 8-factor copula map first (notebooks/run_horseshoe_map.py) or point "
            "--nuts-idata at the cached fit."
        )

    gcfg = GLLVMConfig()
    mconfig = gcfg.measurement_config()
    targets_dir = gcfg.output_dir / "nuts_targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    print(f"[validate] exporting NUTS targets from {idata_path} ...", flush=True)
    nuts_loadings, nuts_phi = V.nuts_targets_from_idata(
        idata_path, mconfig, factors=list(gcfg.factors),
        explicit_factors=list(gcfg.explicit_factors), specific_cross=gcfg.specific_cross,
    )
    # Cache the NUTS targets so the figures driver doesn't reload the 3.5 GB idata.
    nuts_loadings.to_csv(targets_dir / "loadings_summary.csv", index=False)
    nuts_phi.to_csv(targets_dir / "phi.csv")

    nuts_coords = None
    if args.nuts_coords and Path(args.nuts_coords).exists():
        nuts_coords = pd.read_parquet(args.nuts_coords)
        # Align to the VI coordinates' (cohort, patient_id) MultiIndex (strata stores them as columns).
        if {"cohort", "patient_id"}.issubset(nuts_coords.columns):
            nuts_coords = nuts_coords.astype({"cohort": str, "patient_id": str}).set_index(
                ["cohort", "patient_id"])
        print(f"[validate] NUTS coordinates: {args.nuts_coords} {nuts_coords.shape}", flush=True)

    report = V.run_congruence(
        args.vi_dir, nuts_loadings, nuts_phi, nuts_coords=nuts_coords, out_csv=args.out
    )
    print("\n=== Tucker congruence per factor ===", flush=True)
    print(report["tucker"].to_string(index=False), flush=True)
    print("\n=== Phi agreement ===", flush=True)
    print(f"  frobenius(offdiag) = {report['phi']['frobenius_offdiag']:.4f}", flush=True)
    print(f"  max |offdiag diff| = {report['phi']['max_abs_offdiag_diff']:.4f} "
          f"(pass={report['phi']['pass']})", flush=True)
    for cell, vals in report["phi"]["key_cells"].items():
        print(f"    {cell}: VI={vals['vi']:+.3f}  NUTS={vals['nuts']:+.3f}", flush=True)
    if report["coordinates"] is not None:
        print("\n=== Coordinate correlation per factor ===", flush=True)
        print(report["coordinates"].to_string(index=False), flush=True)
    print(f"\n[validate] VERDICT: {report['verdict']}", flush=True)

    # Structured metrics for the figures driver / docs.
    metrics = {
        "verdict": report["verdict"],
        "tucker": report["tucker"].to_dict("records") if report["tucker"] is not None else [],
        "phi": report["phi"],
        "coordinates": report["coordinates"].to_dict("records") if report["coordinates"] is not None else [],
    }
    metrics_path = gcfg.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=float))
    print(f"[validate] report -> {args.out} · metrics -> {metrics_path} · "
          f"targets -> {targets_dir}", flush=True)


if __name__ == "__main__":
    main()
