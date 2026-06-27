#!/usr/bin/env python
"""Export CI-aware posterior factor loadings + Phi from the canonical Gaussian-copula M1 fit.

The copula s5_9dim_mixed fit stores loadings in two places — the continuous ``Lam`` matrix and the
per-explicit-item ``lh_``/``lg_`` scalars — and only as a NetCDF posterior; no CSV / credible-interval
summary has ever been exported.  This driver rebuilds the prepared mixed dataset DETERMINISTICALLY
(so the item ordering aligns with the saved ``Lam``), reads only the loading variables out of the
786 MB idata (never materializing the patient-score blocks ``z_e`` / ``f_e``), and writes a tidy long
loadings table with equal-tailed 95% credible intervals plus the 9x9 Phi matrix.

    PYTHONPATH=$PWD/src HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_export_loadings.py

Outputs (default):
* results/face/oop_measurement/copula/s5_9dim_mixed/loadings_summary.csv   (canonical)
* results/face/oop_measurement/copula/s5_9dim_mixed/phi.csv                (canonical)
* reports/copula_s5_9dim_loadings.csv     (article-facing; supersedes stale native 11_s5_9dim_loadings.csv)
* reports/copula_s5_9dim_phi.csv          (article-facing; supersedes stale native 04_stage5_phi.csv)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[reportMissingImports]
import xarray as xr


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "face" / "models").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Could not locate FACE repository root from {start}")


REPO = _find_repo_root(Path(__file__).resolve())
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

loaded_face = sys.modules.get("face")
loaded_face_file = getattr(loaded_face, "__file__", None) if loaded_face is not None else None
if loaded_face is not None and (
    loaded_face_file is None or SRC not in Path(loaded_face_file).resolve().parents
):
    for module_name in [name for name in sys.modules if name == "face" or name.startswith("face.")]:
        del sys.modules[module_name]

from face.models.bayesian.measurement_model_oop import (  # noqa: E402
    DEFAULT_EXPLICIT_FACTORS,
    MeasurementConfig,
    MeasurementDataset,
)
from face.models.bayesian.synthetic import (  # noqa: E402
    export_loadings_summary,
    export_phi,
)

# Canonical M1 map: the weighted full-N 8-factor operational fit (immunometabolic merge + 3 cross-loadings
# + substance orthogonal). This is the MAIN map the whole vertical + the article are built on.
DEFAULT_IDATA = REPO / "results" / "face" / "oop_measurement" / "copula" / "weighted_8d" / "hs_s5_merged_xc" / "idata.nc"
FOLDED_MATRIX = REPO / "configs" / "prior_loading_matrix_v3_biomerge_xc.csv"
F8_FIT = ["overall_severity", "cognition", "immunometabolic", "sleep", "suicidality",
          "developmental_risk", "mania_activation", "substance"]


class _IData:
    """Minimal idata shim exposing ``.posterior`` for the synthetic-module exporters."""

    def __init__(self, posterior: xr.Dataset):
        self.posterior = posterior


def _load_loading_posterior(idata_path: Path) -> _IData:
    """Open ONLY the loading variables (Lam/Phi/sigma + lh_/lg_ scalars) from the posterior group.

    The full idata is ~786 MB, dominated by per-patient latent blocks (``z_e``, ``f_e``); those are
    never needed here, so we select the loading vars and ``.load()`` just those into memory (~40 MB).
    """
    post = xr.open_dataset(idata_path, group="posterior")
    needed = ["Lam", "Phi", "sigma"]
    needed += [v for v in post.data_vars if str(v).startswith(("lh_", "lg_"))]
    keep = [v for v in needed if v in post.data_vars]
    sub = post[keep].load()
    post.close()
    return _IData(sub)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--idata", type=Path, default=DEFAULT_IDATA,
                   help="Path to the copula s5_9dim_mixed idata.nc (default: canonical run).")
    p.add_argument("--reports-dir", type=Path, default=REPO / "reports",
                   help="Where to write the article-facing CSV copies.")
    p.add_argument("--hdi-prob", type=float, default=0.95,
                   help="Equal-tailed credible-interval mass (default 0.95 -> 2.5/97.5 quantiles).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    idata_path = args.idata.resolve()
    if not idata_path.exists():
        raise FileNotFoundError(f"copula idata not found: {idata_path}")
    manifest_path = idata_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    stage = manifest.get("stage_spec", {})

    # --- deterministic rebuild of the prepared mixed dataset (must match the 8-factor fit) ---
    cfg_sig = manifest.get("config_sig", {})
    config = MeasurementConfig().with_gaussian_copula()
    config = replace(config, prior_matrix=FOLDED_MATRIX,
                     cohort_weighted=bool(cfg_sig.get("cohort_weighted", True)),
                     output_dir=config.output_dir / "copula" / "weighted_8d")
    if "substance" in (cfg_sig.get("orthogonal_factors") or ["substance"]):
        config = config.with_substance_orthogonal()
    ds = MeasurementDataset(config)
    factors = stage.get("factors") or F8_FIT
    explicit_factors = stage.get("explicit_factors") or list(DEFAULT_EXPLICIT_FACTORS)
    mixed = ds.mixed(
        factors,
        explicit_factors=explicit_factors,
        min_cohorts=int(stage.get("min_cohorts", 2)),
        balanced=bool(stage.get("balanced", False)),
        n_subsample=stage.get("n_subsample", None),
        seed=int(stage.get("seed", 20260605)),
    )

    base = mixed.base
    print(f"[rebuild] N={base.M.shape[0]} J={base.M.shape[1]} factors={base.factor_cols}")

    # --- alignment + config asserts (catch any data/config drift vs the saved fit) ---
    J_manifest = int(manifest.get("J", base.M.shape[1]))
    assert base.M.shape[1] == J_manifest, f"J mismatch: rebuild {base.M.shape[1]} vs manifest {J_manifest}"
    if stage.get("factors"):
        assert list(base.factor_cols) == list(stage["factors"]), "factor_cols != manifest stage factors"
    # the e_cols indirection must resolve explicit homes correctly (suicidality stays explicit on the 8-factor map)
    assert base.factor_cols[mixed.e_cols[mixed.ng_home["isf01"]]] == "suicidality", "isf01 home != suicidality"

    # --- load loadings posterior (lean) + export ---
    idata = _load_loading_posterior(idata_path)
    assert idata.posterior["Lam"].sizes[list(idata.posterior["Lam"].dims)[-2]] == base.M.shape[1], \
        "Lam item-axis != rebuilt J"

    # Cross-loading arm: the manifest records specific_cross when the fit freed the plausible_cross
    # cells, so the export must build its kind-map the same way to surface those cross cells.
    xcross = bool(stage.get("specific_cross"))
    cross_sd_scale = float(stage.get("cross_sd_scale", 0.25))
    loadings = export_loadings_summary(idata, mixed, config, hdi_prob=args.hdi_prob,
                                       specific_cross=xcross, cross_sd_scale=cross_sd_scale)
    phi = export_phi(idata, base.factor_cols)
    if xcross:
        n_cross = int((loadings["kind"] == "cross").sum())
        n_cross_cred = int(((loadings["kind"] == "cross") & loadings["excludes_zero"]).sum())
        print(f"[arm] cross cells: {n_cross} emitted, {n_cross_cred} with 95% CI excluding zero")

    # --- spot-checks (verified expectations) ---
    def cell(item: str, factor: str):
        m = loadings[(loadings["item"] == item) & (loadings["factor"] == factor)]
        return None if m.empty else m.iloc[0]

    # 8-factor spot-checks: biology indicators home to immunometabolic; suicidality anchor; an earned cross-loading
    bmi = cell("bmi", "immunometabolic"); crp = cell("crp", "immunometabolic"); isf = cell("isf01", "suicidality")
    assert bmi is not None and bmi["abs_loading"] > 0.5 and bool(bmi["excludes_zero"]), "bmi->immunometabolic spot-check"
    assert crp is not None and crp["loading"] != 0.0, "crp->immunometabolic spot-check"
    assert isf is not None and isf["abs_loading"] > 1.0 and bool(isf["excludes_zero"]), "isf01->suicidality spot-check"
    assert cell("bmi", "cognition") is None, "bmi->cognition should be a dropped hard-zero"
    ctq = cell("ctq37", "cognition")   # one of the 3 earned cross-loadings (childhood adversity -> cognition)
    print(f"[spot] bmi/immunometabolic={bmi['loading']:.3f}  crp/immunometabolic={crp['loading']:.3f}  "
          f"isf01/suicidality={isf['loading']:.3f}  ctq37/cognition(cross)="
          f"{ctq['loading']:.3f} excl0={bool(ctq['excludes_zero'])}" if ctq is not None else "[spot] ctq37 cross not emitted")

    # --- write outputs ---
    out_dir = idata_path.parent
    reports = args.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    # The 8-factor map is canonical -> article-facing CSVs carry the 8factor name.
    prefix = "copula_8factor"
    targets = {
        out_dir / "loadings_summary.csv": (loadings, False),
        out_dir / "phi.csv": (phi, True),
        reports / f"{prefix}_loadings.csv": (loadings, False),
        reports / f"{prefix}_phi.csv": (phi, True),
    }
    for path, (frame, index) in targets.items():
        frame.to_csv(path, index=index)
        print(f"[write] {path}  ({frame.shape[0]}x{frame.shape[1]})")

    n_excl = int(loadings["excludes_zero"].sum())
    print(f"[done] {loadings.shape[0]} loading cells, {n_excl} with 95% CI excluding zero; "
          f"Phi {phi.shape[0]}x{phi.shape[1]}")


if __name__ == "__main__":
    main()
