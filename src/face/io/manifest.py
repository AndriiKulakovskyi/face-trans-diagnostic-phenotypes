"""Exact fit metadata + patient-index persistence (issue P2-03).

Every fit should write a machine-readable manifest and persist the EXACT sampled patient index next
to its ``idata.nc``, so downstream scoring/QC loads the real index instead of reconstructing the
subsample from a seed + balanced-sampling logic (fragile to data-ordering / NumPy-version drift).

    from face.io import manifest
    manifest.write_manifest("s5_cert9_s1", out_dir=out, N=base.M.shape[0],
                            index=base.index, cohort=base.cohort, seed=seed, diagnostics=diag)
    idx = manifest.load_index(out)            # later, in scoring
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PACKAGES = ["numpy", "scipy", "pandas", "scikit-learn", "pymc", "pytensor",
             "numpyro", "jax", "jaxlib", "arviz", "h5netcdf"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def package_versions() -> dict:
    import importlib.metadata as md
    import sys

    out: dict = {"python": sys.version.split()[0]}
    for pkg in _PACKAGES:
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = None
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root()), text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _to_frame(index_like) -> pd.DataFrame:
    """Coerce a pandas Index / MultiIndex / DataFrame / array into a flat DataFrame of the index."""
    if isinstance(index_like, pd.DataFrame):
        return index_like.reset_index(drop=True)
    if isinstance(index_like, pd.MultiIndex):
        return index_like.to_frame(index=False)
    if isinstance(index_like, pd.Index):
        return index_like.to_frame(index=False)
    return pd.DataFrame({"row": np.asarray(index_like)})


def index_hash(index_like) -> str:
    df = _to_frame(index_like)
    payload = df.to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def save_index(out_dir, index_like, cohort=None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = _to_frame(index_like)
    if cohort is not None and "cohort" not in df.columns:
        df = df.assign(cohort=np.asarray(cohort))
    p = out / "index.parquet"
    df.to_parquet(p)
    return p


def load_index(out_dir) -> pd.DataFrame | None:
    p = Path(out_dir) / "index.parquet"
    return pd.read_parquet(p) if p.exists() else None


def cohort_counts(cohort) -> dict | None:
    if cohort is None:
        return None
    vals, cnts = np.unique(np.asarray(cohort).astype(str), return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, cnts, strict=False)}


def write_manifest(stage: str, *, out_dir=None, N=None, index=None, cohort=None, seed=None,
                   diagnostics=None, extra=None) -> Path:
    """Write ``results/manifests/<stage>_manifest.json`` (shareable: counts + index HASH, no rows) and,
    if ``out_dir``+``index`` are given, persist the exact ``index.parquet`` beside the fit's idata."""
    mdir = repo_root() / "results" / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    man = {
        "stage": stage,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "N": int(N) if N is not None else (len(_to_frame(index)) if index is not None else None),
        "cohort_counts": cohort_counts(cohort),
        "seed": seed,
        "patient_index_hash": index_hash(index) if index is not None else None,
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "diagnostics": diagnostics,
    }
    if extra:
        man.update(extra)
    p = mdir / f"{stage}_manifest.json"
    p.write_text(json.dumps(man, indent=2, default=str))
    if out_dir is not None and index is not None:
        save_index(out_dir, index, cohort=cohort)
    return p
