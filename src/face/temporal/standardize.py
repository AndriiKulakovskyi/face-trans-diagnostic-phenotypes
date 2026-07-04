"""The V0 standardization spec — the load-bearing piece that lets follow-up be scored on the FIXED
M1 scale (docs/TEMPORAL_MODEL.md §3.1).

`prepare()` z-scores each indicator *in-sample* off `baseline_v0.parquet`, so the certified loadings live
on the V0 moments. To score V1/V2 on the SAME scale we must reuse the V0 per-item transform
(family / sign / lognormal log-min / mean / sd) — **never** recompute it per visit, which would re-centre
genuine improvement to ~0 and erase the change. The spec is captured at the source via
`prepare(emit_moments=True)`, the exact loop that builds the fit's matrix, so it can never drift from the
fit. `apply_spec` then reproduces that transform on any visit's raw cells (a value outside V0's lognormal
support → NaN, counted, never imputed). The invariance refit (G1) deliberately does NOT use this — it
z-scores in-sample per visit because Tucker congruence is scale-invariant.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from face.measurement.kernel import S5_FACTORS, prepare


@dataclass
class V0StdSpec:
    """Per-item V0 standardization moments (the frozen scale). `logmin` is the V0 lognormal min-shift
    (None for non-lognormal items); `mean`/`sd` are of the log+sign-oriented values."""
    items: list[str]
    family: dict[str, str]
    sign: dict[str, int]
    logmin: dict[str, float | None]
    mean: dict[str, float]
    sd: dict[str, float]


def capture_v0_spec(factors: list[str] = S5_FACTORS) -> V0StdSpec:
    """Capture the V0 transform from the certified scoring config (`prepare(correlated, windows)` — the
    exact standardization M2.0 scored on). Returns a `V0StdSpec` over `prep.items`."""
    prep, mom = prepare(factors, correlated=True, windows=True, emit_moments=True)
    items = list(prep.items)
    return V0StdSpec(
        items=items,
        family={it: str(mom[it][0]) for it in items},
        sign={it: int(mom[it][1]) for it in items},
        logmin={it: (None if mom[it][2] is None else float(mom[it][2])) for it in items},
        mean={it: float(mom[it][3]) for it in items},
        sd={it: float(mom[it][4]) for it in items},
    )


def save_spec(spec: V0StdSpec, path: str | Path) -> None:
    Path(path).write_text(json.dumps({
        "items": spec.items, "family": spec.family, "sign": spec.sign,
        "logmin": spec.logmin, "mean": spec.mean, "sd": spec.sd}))


def load_spec(path: str | Path) -> V0StdSpec:
    d = json.loads(Path(path).read_text())
    return V0StdSpec(items=list(d["items"]), family=d["family"],
                     sign={k: int(v) for k, v in d["sign"].items()},
                     logmin=d["logmin"], mean=d["mean"], sd=d["sd"])


def apply_spec(spec: V0StdSpec, B_visit: pd.DataFrame) -> pd.DataFrame:
    """Standardize a visit's RAW harmonized cells onto the frozen V0 scale (reproduces `prepare()`'s
    transform with V0-frozen stats). Columns = `spec.items` (an item not collected at this visit →
    all-NaN column); a cell outside V0's lognormal support → NaN. NaN = missing, never imputed."""
    idx = B_visit.index
    out: dict[str, pd.Series] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for it in spec.items:
            if it in B_visit.columns:
                v = pd.to_numeric(B_visit[it], errors="coerce").astype(float)
            else:
                v = pd.Series(np.nan, index=idx)
            if spec.family[it] == "lognormal":
                mn = spec.logmin[it]
                v = np.log1p(v - mn + 1e-6) if (mn is not None and mn <= 0) else np.log(v)
            v = spec.sign[it] * v
            sd = spec.sd[it]
            col = (v - spec.mean[it]) / sd if sd and sd > 0 else v * 0.0
            out[it] = col.where(np.isfinite(col))   # log of 0/neg → ±inf → NaN (outside V0 support; not imputed)
    return pd.DataFrame(out, index=idx)[spec.items]


def prep_visit_continuous(spec: V0StdSpec, B_visit: pd.DataFrame, factors: list[str] = S5_FACTORS):
    """A `CorePrep` carrying the V0 measurement STRUCTURE (items / home / factor order / loading cells)
    but the visit's V0-scaled `M` — feeds the M1 scorers unchanged at stage 34. No re-fit, no re-discovery."""
    base = prepare(factors, correlated=True, windows=True)
    M = apply_spec(spec, B_visit)[base.items].to_numpy()
    cohort = np.asarray(B_visit.index.get_level_values("cohort"))
    return replace(base, M=M, index=B_visit.index, cohort=cohort)


def prep_visit_mixed(spec: V0StdSpec, mp_v0, B_visit: pd.DataFrame, *, cert_index, B_v0: pd.DataFrame):
    """A `MixedPrep` with the certified V0 STRUCTURE (item sets / factor columns / homes / cutpoint
    coding) but the visit's data: continuous block V0-scaled (`apply_spec`), non-Gaussian cells read raw
    from the visit, ordinals re-coded to the CERTIFIED V0 category map (so the fixed cutpoints stay valid).
    Feeds `project_explicit_full_n` unchanged at stage 34 — no re-fit. ``cert_index`` is the certified fit
    subsample index (defines the ordinal coding); ``B_v0`` is `baseline_v0` (where that coding is read)."""
    items = mp_v0.base.items
    Mvis = apply_spec(spec, B_visit)[items].to_numpy()
    cohort = np.asarray(B_visit.index.get_level_values("cohort"))
    base_vis = replace(mp_v0.base, M=Mvis, index=B_visit.index, cohort=cohort)

    def grab(cols):
        return pd.DataFrame({c: (pd.to_numeric(B_visit[c], errors="coerce") if c in B_visit.columns
                                 else np.nan) for c in cols}, index=B_visit.index).to_numpy().astype(float)

    mp_vis = replace(mp_v0, base=base_vis, Bin=grab(mp_v0.bin_items), Cnt=grab(mp_v0.cnt_items),
                     Ord=grab(mp_v0.ord_items), ord_K=list(mp_v0.ord_K))
    # re-code ordinals to the certified V0 coding (top/bottom absorption), matching align_ordinals_to_fit
    for k, it in enumerate(mp_v0.ord_items):
        uniq = np.sort(pd.to_numeric(B_v0.loc[cert_index][it], errors="coerce").dropna().unique())
        remap = {float(v): i for i, v in enumerate(uniq)}
        K = len(uniq)
        raw = mp_vis.Ord[:, k]
        code = np.full(len(raw), np.nan)
        for i, v in enumerate(raw):
            if np.isnan(v):
                continue
            if v in remap:
                code[i] = remap[v]
            elif v > uniq[-1]:
                code[i] = K - 1
            elif v < uniq[0]:
                code[i] = 0
            else:
                code[i] = remap[float(uniq[uniq <= v].max())]
        mp_vis.Ord[:, k] = code
        mp_vis.ord_K[k] = K
    return mp_vis
