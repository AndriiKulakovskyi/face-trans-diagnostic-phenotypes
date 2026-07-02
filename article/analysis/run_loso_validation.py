#!/usr/bin/env python
"""Leave-one-site-out (LOSO) external-validity proxy for the FACE-ATLAS measurement map.

For each recruitment site: write a filtered copy of data/processed with that site's rows
removed, refit the VI/GLLVM map on the remainder, and compare to the full-data map by
(P1) Tucker loading congruence and (P2) score-level immuno-severity decoupling. Runs on the
VI path so the folds are tractable relative to the full-N NUTS map.

    # wiring check on one small fold, smoke epochs (~10 s):
    KMP_DUPLICATE_LIB_OK=TRUE PYTENSOR_FLAGS="cxx=" python notebooks/run_loso_validation.py \
        --mode smoke --min-site-n 100000 --only-site 20 --out results/face/loso_smoke

    # production: 15 large-site folds at scientific scale (use a GPU/many-core box):
    KMP_DUPLICATE_LIB_OK=TRUE python notebooks/run_loso_validation.py \
        --mode production --min-site-n 100 --device mps --out results/face/loso

Design + endpoints: see the LOSO validation plan (out/loso_validation_plan.md).

*** This runner was wiring-checked end-to-end on the smoke path (both fits ran, exports
    parsed, congruence + score-decoupling computed). The only #VERIFY left is the held-out
    SCORING step (S1) — see score_holdout() — which needs the projection entrypoint confirmed
    on your data; P1 and P2 are fully tested. ***
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + MKL both ship libomp on macOS
os.environ.setdefault("OMP_NUM_THREADS", "4")
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

def _find_repo(start: Path) -> Path:
    """Walk up from a starting point until we find the repo (has data/processed + src/face).
    Robust to the script living in notebooks/ OR being run from an out-of-tree workspace."""
    for base in [start, *start.parents]:
        if (base / "data" / "processed" / "site_v0.parquet").exists() and (base / "src" / "face").is_dir():
            return base
    # explicit fallback for this project
    cand = Path("/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr")
    if (cand / "data" / "processed" / "site_v0.parquet").exists():
        return cand
    raise SystemExit("could not locate the face repo root (data/processed + src/face)")


REPO = _find_repo(Path(__file__).resolve().parent)
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from face.models.variational.gllvm_model_oop import GLLVMConfig, GLLVMRunner   # noqa: E402
from face.models.variational import validate as vv                            # noqa: E402

PROC = REPO / "data" / "processed"
ALIGNED_PARQUETS = ("baseline_v0.parquet", "site_v0.parquet", "covariates_v0.parquet")
# G is orthogonal by construction in the bifactor map, so the decoupling estimand is measured
# at the SCORE level (immuno vs severity posterior means), NOT off Phi (which is structurally 0).
SEV, IMMUNO = "overall_severity", "immunometabolic"


def make_filtered_proc(dst: Path, drop_site: float) -> tuple[int, int]:
    """Copy processed_dir to dst, dropping one site's rows from the 3 index-aligned parquets.
    (All three share the exact (cohort, patient_id) MultiIndex in the same order — verified.)"""
    dst.mkdir(parents=True, exist_ok=True)
    site = pd.read_parquet(PROC / "site_v0.parquet")
    keep = (site.iloc[:, 0] != drop_site).to_numpy()
    for f in PROC.iterdir():
        if f.name in ALIGNED_PARQUETS:
            pd.read_parquet(f).iloc[keep].to_parquet(dst / f.name)
        else:
            shutil.copy2(f, dst / f.name)   # metadata / standardization spec: copy unchanged
    return int(keep.sum()), int((~keep).sum())


def run_map(processed_dir: Path, output_dir: Path, *, mode: str, device: str) -> Path:
    """Fit the VI map from processed_dir; return the export dir holding loadings_summary.csv."""
    cfg = GLLVMConfig()
    cfg = replace(cfg, processed_dir=Path(processed_dir), output_dir=Path(output_dir), device=device)
    if mode == "smoke":
        cfg = cfg.with_smoke_defaults()          # epochs=120 — wiring only, NOT scientific
    elif mode == "medium":
        cfg = replace(cfg, epochs=1500)
    # production: defaults (epochs=4000, full 2-rung plan)
    fit = GLLVMRunner(cfg).run_plan(overwrite=True)
    # Select the FINAL stage explicitly. The 2-rung plan writes s1_backbone (G + continuous
    # backbone: cognition/immuno/sleep only) AND s8_full (all 8 factors). rglob walk order is
    # NOT consistent across directories, so taking rglob-first silently mixes stages across
    # folds. Prefer the full-model stage so all 8 factors are comparable.
    cands = {p.parent.name: p.parent for p in Path(output_dir).rglob("loadings_summary.csv")}
    if not cands:
        raise RuntimeError(f"no loadings_summary.csv under {output_dir}")
    for pref in ("s8_full", "smoke_s8_full"):
        if pref in cands:
            return cands[pref]
    return sorted(cands.values(), key=lambda p: p.name)[-1]  # last stage by name


def score_decoupling(export_dir: Path) -> dict:
    """P2 estimand at the score level: |corr(immuno_mean, severity_mean)| and the reference
    ordering vs cognition/sleep. coordinates.parquet columns are '<factor>__mean' etc."""
    c = pd.read_parquet(Path(export_dir) / "coordinates.parquet")
    m = lambda f: c[f + "__mean"].to_numpy(float)
    out = {"decouple_immuno": abs(np.corrcoef(m(IMMUNO), m(SEV))[0, 1])}
    for ref in ("cognition", "sleep"):
        if ref + "__mean" in c.columns:
            out[f"corr_{ref}"] = abs(np.corrcoef(m(ref), m(SEV))[0, 1])
    out["immuno_is_lowest"] = out["decouple_immuno"] == min(
        v for k, v in out.items() if k.startswith(("decouple_", "corr_")))
    return out


def score_holdout(loso_export: Path, full_export: Path, held_ids) -> float:
    """S1 (#VERIFY): score the held-out patients on the LOSO map and correlate their
    coordinates with the full-data map. The VI runner scores the fit sample only, so the
    held-out rows must be projected through the LOSO posterior via the measurement model's
    projection entrypoint (MeasurementModel.projection_frame, measurement_model_oop.py:1806).
    Wire that in once and return median per-factor r; returns nan until then."""
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["smoke", "medium", "production"], default="production")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    ap.add_argument("--min-site-n", type=int, default=100,
                    help="hold out only sites with >= this many patients (primary analysis: 100)")
    ap.add_argument("--only-site", type=float, default=None, help="run a single fold (wiring)")
    ap.add_argument("--out", type=Path, default=REPO / "results/face/loso")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    site = pd.read_parquet(PROC / "site_v0.parquet").iloc[:, 0]
    counts = site.value_counts()
    if args.only_site is not None:
        folds = [args.only_site]
    else:
        folds = sorted(counts[counts >= args.min_site_n].index.tolist())
    print(f"[loso] mode={args.mode} device={args.device} folds={len(folds)} "
          f"sites={[float(x) for x in folds]}", flush=True)

    # full-data reference map (fit once)
    full_exp = run_map(PROC, args.out / "full", mode=args.mode, device=args.device)
    full = vv.load_vi(full_exp)
    F8 = list(GLLVMConfig().factors)

    rows = []
    for s in folds:
        fold_proc = args.out / f"proc_drop{int(s)}"
        nkeep, nheld = make_filtered_proc(fold_proc, s)
        t0 = time.time()
        loso_exp = run_map(fold_proc, args.out / f"loso_drop{int(s)}", mode=args.mode, device=args.device)
        loso = vv.load_vi(loso_exp)
        tuck = vv.tucker_congruence_per_factor(loso["loadings"], full["loadings"], F8)
        dec = score_decoupling(loso_exp)
        held_ids = site.index[(site == s).to_numpy()]
        row = dict(site=float(s), n_keep=nkeep, n_held=nheld,
                   immuno_tucker=float(tuck.loc[tuck.factor == IMMUNO, "tucker"].iloc[0]),
                   min_factor_tucker=float(tuck["tucker"].min()),
                   n_factor_scored=int(tuck["tucker"].notna().sum()),   # factors comparable in both fits
                   n_factor_pass=int(tuck["pass"].sum()),
                   decouple_immuno=dec["decouple_immuno"],
                   immuno_is_lowest=bool(dec["immuno_is_lowest"]),
                   held_coord_r=score_holdout(loso_exp, full_exp, held_ids),
                   elapsed_s=round(time.time() - t0, 1))
        rows.append(row)
        shutil.rmtree(fold_proc, ignore_errors=True)              # filtered copy is disposable
        shutil.rmtree(args.out / f"loso_drop{int(s)}", ignore_errors=True)  # heavy fit artifacts; metrics captured
        pd.DataFrame(rows).to_csv(args.out / "loso_summary.csv", index=False)  # checkpoint each fold
        print(f"[loso] site {s:>6}: immuno_phi={row['immuno_tucker']:.3f} "
              f"min_phi={row['min_factor_tucker']:.3f} decouple={row['decouple_immuno']:.3f} "
              f"({row['elapsed_s']}s)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out / "loso_summary.csv", index=False)
    print("\n=== LOSO headline (mode=%s) ===" % args.mode, flush=True)
    print(f"immuno loading congruence: min Tucker phi = {out.immuno_tucker.min():.3f} "
          f"(median {out.immuno_tucker.median():.3f}) across {len(out)} folds", flush=True)
    print(f"score-level decoupling |corr(immuno,severity)|: "
          f"{out.decouple_immuno.min():.3f}-{out.decouple_immuno.max():.3f}; "
          f"immuno lowest-correlated factor in {int(out.immuno_is_lowest.sum())}/{len(out)} folds", flush=True)
    print(f"-> {args.out/'loso_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
