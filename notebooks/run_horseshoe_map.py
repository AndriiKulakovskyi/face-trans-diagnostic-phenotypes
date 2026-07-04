#!/usr/bin/env python
"""The 8-factor sparse-ESEM map: metabolic + inflammatory merged into one immunometabolic factor,
with the off-home specific cross-loadings freed under a REGULARIZED HORSESHOE.

The horseshoe is a sparsity-inducing prior — a sharp spike at 0 crushes the many noise cross-loadings
(so a low-instrument factor's column is not diluted and its identity is protected: substance n=4, mania
n=2) while heavy tails let a GENUINELY supported small cross-loading escape and become credible. This is
the principled way to let instruments load weakly on several axes without collapsing thin factors.

Ladder (each rung warm-starts the next):
  hs_s1_merged  continuous core (G, cognition, immunometabolic, sleep)        — hard-zero, clean backbone
  hs_s3_merged  + developmental_risk, mania_activation (continuous)           — hard-zero
  hs_s5_merged  full 8-factor MIXED map (+ suicidality, substance explicit)   — HORSESHOE cross-loadings,
                warm-started from the clean hard-zero backbone (identified, unlike free cross-loadings).

    HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_horseshoe_map.py --smoke
    python3 scripts/run_job.py horseshoe -- env HDF5_USE_FILE_LOCKING=FALSE \
        python notebooks/run_horseshoe_map.py
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path


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
    DEFAULT_EXPLICIT_FACTORS,
    MeasurementConfig,
    StageDefinition,
    StageRunner,
)

MERGED_MATRIX = REPO / "configs" / "loading_matrix.immunometabolic.csv"
F1 = ["overall_severity", "cognition", "immunometabolic", "sleep"]
F3 = F1 + ["developmental_risk", "mania_activation"]
F8 = ["overall_severity", "cognition", "immunometabolic", "sleep",
      "suicidality", "developmental_risk", "mania_activation", "substance"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2000, help="balanced subsample (default 2000)")
    p.add_argument("--tau0", type=float, default=0.05, help="horseshoe global-shrinkage scale")
    p.add_argument("--slab-c", type=float, default=0.30, help="horseshoe slab width (caps escaped |λ|)")
    p.add_argument("--smoke", action="store_true", help="tiny wiring check")
    p.add_argument("--final", choices=["horseshoe", "hardzero"], default="horseshoe",
                   help="cross-loading treatment for the final 8-factor MIXED map: horseshoe (sparse ESEM) "
                        "or hardzero (the operational map for M2-M5, converges like the certified fit).")
    p.add_argument("--fold", action="store_true",
                   help="final operational map = hard-zero + the 6 sparse-ESEM-SELECTED cross-loadings "
                        "(configs/loading_matrix.immunometabolic_crossload.csv, freed via specific_cross). "
                        "A small identified refinement between well-separated factors.")
    p.add_argument("--weighted", action="store_true",
                   help="fit at FULL N with cohort-weighting (the operational map M2-M5 consume, analogue of "
                        "copula/weighted/s5_9dim_mixed). Output -> copula/weighted_8d/. ~4h for the mixed stage.")
    p.add_argument("--salvage", action="store_true",
                   help="re-fit ONLY the weighted mixed stage with substance pinned ORTHOGONAL (kills the "
                        "immunometabolic<->substance rotation that broke the full-N fit) + warm-started from the "
                        "converged balanced map (anchors substance loadings at 0.585). Implies --weighted --fold.")
    p.add_argument("--exclude", default="",
                   help="comma-separated indicators to DROP entirely (sensitivity arm), e.g. "
                        "'bmi,weight,wstcir' for the immunometabolic minus-anthropometry refit. The output "
                        "dir gets an '_excl-<items>' suffix so it never collides with the canonical map.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.salvage:
        a.weighted = True
        a.fold = True
        a.final = "hardzero"
    # --weighted => full-N cohort-weighted (the operational estimand M2-M5 consume); else balanced subsample.
    samp = dict(balanced=False, n_subsample=None) if a.weighted else dict(balanced=True, n_subsample=a.n)
    cont = dict(correlated=True, windows=True, mixed=False, seed=20260605, **samp)
    mixed = dict(correlated=True, windows=True, mixed=True, explicit_factors=list(DEFAULT_EXPLICIT_FACTORS),
                 min_cohorts=2, seed=20260605, **samp)
    dc, dt, mc_, mt = (1000, 1000, 1500, 2000) if not a.smoke else (40, 40, 40, 40)
    ch = 4 if not a.smoke else 2
    s5name = {"horseshoe": "hs_s5_merged", "hardzero": "hs_s5_merged_hz"}[a.final]
    if a.fold:
        s5name = "hs_s5_merged_xc"
    s1 = StageDefinition("hs_s1_merged", F1, draws=dc, tune=dt, chains=ch, target_accept=0.95, **cont)
    s3 = StageDefinition("hs_s3_merged", F3, draws=dc, tune=dt, chains=ch, target_accept=0.95, **cont)
    s5 = StageDefinition(s5name, F8, draws=mc_, tune=mt, chains=ch, target_accept=0.95,
                         specific_cross=a.fold, cross_sd_scale=1.0, **mixed)

    excl = tuple(x.strip() for x in a.exclude.split(",") if x.strip())
    sub = "weighted_8d" if a.weighted else "horseshoe_8d"
    if excl:
        sub += "_excl-" + "-".join(excl)
    base = MeasurementConfig().with_gaussian_copula()
    base = replace(base, prior_matrix=MERGED_MATRIX, cohort_weighted=a.weighted, exclude_items=excl,
                   output_dir=base.output_dir / "copula" / sub,
                   figure_dir=base.figure_dir / "copula" / sub)
    hs = base.with_horseshoe(tau0=a.tau0, slab_c=a.slab_c)

    if a.fold:
        # operational map + the 6 data-selected cross-loadings (specific_cross frees exactly those cells)
        xc = REPO / "configs" / "loading_matrix.immunometabolic_crossload.csv"
        final_cfg = replace(base, prior_matrix=xc)
        import pandas as _pd
        n_xc = int((_pd.read_csv(xc)["rationale"].astype(str).str.contains("folded")).sum())
        final_mode = f"FOLDED (hard-zero + {n_xc} selected cross-loadings)"
    else:
        final_cfg, final_mode = (hs, "HORSESHOE") if a.final == "horseshoe" else (base, "hard-zero")
    print(f"[hs-map] 8-factor merged map; final={a.final} (tau0={a.tau0} slab_c={a.slab_c}); "
          f"N={'full' if a.weighted else a.n}"
          f"{'; EXCLUDING ' + ','.join(excl) if excl else ''}", flush=True)
    print(f"[hs-map] factors: {F8}", flush=True)

    if a.salvage:
        # Re-fit ONLY the weighted mixed stage: substance pinned orthogonal (kills the immunometabolic<->
        # substance rotation) + warm-started from the converged BALANCED map (anchors loadings, incl substance).
        import shutil
        bal_src = REPO / "results" / "face" / "oop_measurement" / "copula" / "horseshoe_8d" / "hs_s5_merged_xc" / "idata.nc"
        bal_dir = base.output_dir / "bal_seed"
        bal_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(bal_src, bal_dir / "idata.nc")
        bal_seed = StageDefinition("bal_seed", F8, draws=mc_, tune=mt, chains=ch, target_accept=0.95,
                                   specific_cross=True, cross_sd_scale=1.0, **mixed)
        final_cfg = final_cfg.with_substance_orthogonal()
        print(f"[hs-map] SALVAGE: substance pinned orthogonal {final_cfg.orthogonal_factors}; "
              f"warm-start s5 from the converged balanced map", flush=True)
        t0 = time.time()
        _idata, man = StageRunner(final_cfg).run_stage(s5, overwrite=True, prev_stage=bal_seed)
        print(f"[{s5.name}] diagnostics={man.get('diagnostics', {})}  elapsed={round(time.time()-t0)}s", flush=True)
        out = final_cfg.output_dir / s5.name
        print(f"[hs-map] SALVAGE DONE -> {out}/idata.nc", flush=True)
        return

    # backbone always hard-zero (clean, fast); final stage per --final, all same output_dir for warm-start
    runner_hz = StageRunner(base)
    prev = None
    for st, runner, mode in [(s1, runner_hz, "hard-zero"), (s3, runner_hz, "hard-zero"),
                             (s5, StageRunner(final_cfg), final_mode)]:
        print(f"\n=== {st.name} ({mode}, {len(st.factors)} factors{' MIXED' if st.mixed else ''}) ===", flush=True)
        t0 = time.time()
        _idata, man = runner.run_stage(st, overwrite=a.overwrite, prev_stage=prev)
        print(f"[{st.name}] diagnostics={man.get('diagnostics', {})}  elapsed={round(time.time()-t0)}s", flush=True)
        prev = st

    out = base.output_dir / s5.name
    print(f"\n[hs-map] DONE -> {out}/idata.nc", flush=True)
    print(f"[hs-map] next: export CI-aware loadings: "
          f"`python notebooks/run_export_loadings.py --idata {out}/idata.nc`", flush=True)


if __name__ == "__main__":
    main()
