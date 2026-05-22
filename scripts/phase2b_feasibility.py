"""Phase 2b — empirical feasibility of the staged discovery design.

Answers, on the READY (fully-shared) core at V0:

  1. Per-cohort completeness distribution (per-patient fraction of non-NaN
     READY features).
  2. How many patients per cohort exceed completeness floors
     {0.80, 0.85, 0.90, 0.95, 1.00} — and the max BALANCED discovery-set
     size achievable at each floor (= min across cohorts).
  3. READY core composition: per-section, per-dtype, near-constant features.
  4. Selection-bias check: are high-completeness (>=95%) patients different
     from the rest on age and available severity scales? (Welch t-test +
     standardized mean difference.)

Writes results/phase2b_feasibility.json and prints a readable summary.

Run:  python3 scripts/phase2b_feasibility.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from face_common import (  # noqa: E402
    IDENTIFIER_COLUMNS,
    build_unified_dataframe,
    load_variables,
)

DATA_DIR = REPO_ROOT / "data"
DICT_PATH = REPO_ROOT / "face-common-vars.xlsx"
RESULTS_DIR = REPO_ROOT / "results"

COHORTS = ["BP", "SZ", "DR"]
FLOORS = [0.80, 0.85, 0.90, 0.95, 1.00]


def near_constant_share(series: pd.Series) -> float:
    nn = series.dropna()
    if len(nn) == 0:
        return 1.0
    return float(nn.value_counts(normalize=True).iloc[0])


def welch_smd(a: pd.Series, b: pd.Series) -> dict:
    """Welch t-test approximation + standardized mean difference (Cohen's d)."""
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 5 or len(b) < 5:
        return {"n_high": int(len(a)), "n_rest": int(len(b)),
                "mean_high": None, "mean_rest": None, "smd": None}
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled_sd = np.sqrt((va + vb) / 2) if (va + vb) > 0 else np.nan
    smd = (ma - mb) / pooled_sd if pooled_sd and not np.isnan(pooled_sd) else None
    return {"n_high": int(len(a)), "n_rest": int(len(b)),
            "mean_high": float(ma), "mean_rest": float(mb),
            "smd": float(smd) if smd is not None else None}


def main() -> int:
    print("Loading READY-core unified frame (readiness=['READY'], long)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(
            DATA_DIR, DICT_PATH, readiness=["READY"], format="long",
        )
    variables = load_variables(DICT_PATH)
    var_lookup = {v.canonical_name: v for v in variables}

    v0 = df[df["visit"] == "V0"].copy()
    feature_cols = [c for c in v0.columns if c not in IDENTIFIER_COLUMNS]
    print(f"  READY feature columns: {len(feature_cols)}")
    print(f"  V0 patients: {v0['usubjid_patients'].nunique():,}")

    # ---- 0. column-level completeness + informative-core definition -------
    # The full READY core is sparse per-patient because it bundles rarely-
    # collected items (suicide attempt detail, rare-disease binaries). Define
    # an INFORMATIVE CORE: drop near-constant binaries and any feature whose
    # own column-completeness < 0.70. Patient completeness is then measured on
    # this informative subset, which is what we'd actually cluster on.
    col_completeness = 1.0 - v0[feature_cols].isna().mean()
    informative = []
    for col in feature_cols:
        var = var_lookup.get(col)
        dt = var.dtype if var else ""
        is_nc = (dt in ("int8 binary", "int8 categorical", "int8 ordinal")
                 and near_constant_share(v0[col]) > 0.95)
        if is_nc:
            continue
        if col_completeness[col] < 0.70:
            continue
        informative.append(col)
    print(f"  informative core (drop near-constant + col-completeness<0.70): "
          f"{len(informative)} features")

    # ---- 1+2. completeness distribution & floor feasibility --------------
    # Report on BOTH the full READY core and the informative core.
    results_by_set = {}
    for set_name, cols in (("full_ready", feature_cols),
                           ("informative_core", informative)):
        comp = 1.0 - v0[cols].isna().mean(axis=1)
        v0[f"_comp_{set_name}"] = comp
        dist = {}
        for c in COHORTS:
            s = comp[v0["cohort"] == c]
            dist[c] = {"n": int(len(s)), "mean": float(s.mean()),
                       **{f"p{p}": float(np.percentile(s, p))
                          for p in (10, 25, 50, 75, 90)}}
        floor_table = {}
        for fl in FLOORS:
            counts = {c: int((comp[v0["cohort"] == c] >= fl).sum())
                      for c in COHORTS}
            floor_table[f"{fl:.2f}"] = {"per_cohort": counts,
                                        "balanced_N": min(counts.values())}
        results_by_set[set_name] = {"distribution": dist,
                                    "floor_feasibility": floor_table}
        print(f"\n=== [{set_name}] per-cohort V0 completeness "
              f"({len(cols)} features) ===")
        for c in COHORTS:
            d = dist[c]
            print(f"  {c}: n={d['n']:>5}  mean={d['mean']:.3f}  "
                  f"median={d['p50']:.3f}  p25={d['p25']:.3f}  p10={d['p10']:.3f}")
        print(f"  {'floor':>6} " + " ".join(f"{c:>6}" for c in COHORTS) +
              f" {'balanced':>9}")
        for fl in FLOORS:
            ft = floor_table[f"{fl:.2f}"]
            print(f"  {fl:>6.2f} " +
                  " ".join(f"{ft['per_cohort'][c]:>6}" for c in COHORTS) +
                  f" {ft['balanced_N']:>9}")

    # legacy aliases for downstream code (use informative core as primary)
    dist = results_by_set["informative_core"]["distribution"]
    floor_table = results_by_set["informative_core"]["floor_feasibility"]
    v0 = v0.assign(_completeness=v0["_comp_informative_core"])

    print("\n=== Per-cohort V0 completeness on the READY core ===")
    dist = {}
    for c in COHORTS:
        s = v0.loc[v0["cohort"] == c, "_completeness"]
        pct = {f"p{p}": float(np.percentile(s, p)) for p in (10, 25, 50, 75, 90)}
        dist[c] = {"n": int(len(s)), "mean": float(s.mean()), **pct}
        print(f"  {c}: n={len(s):>5}  mean={s.mean():.3f}  "
              f"median={pct['p50']:.3f}  p25={pct['p25']:.3f}  p10={pct['p10']:.3f}")

    print("\n=== Patients exceeding completeness floors (per cohort) ===")
    print(f"  {'floor':>6} " + " ".join(f"{c:>6}" for c in COHORTS) +
          f" {'balanced_N':>11}")
    floor_table = {}
    for fl in FLOORS:
        counts = {c: int((v0.loc[v0['cohort'] == c, '_completeness'] >= fl).sum())
                  for c in COHORTS}
        balanced = min(counts.values())
        floor_table[f"{fl:.2f}"] = {"per_cohort": counts, "balanced_N": balanced}
        print(f"  {fl:>6.2f} " + " ".join(f"{counts[c]:>6}" for c in COHORTS) +
              f" {balanced:>11}")

    # ---- 3. READY core composition ---------------------------------------
    print("\n=== READY core composition ===")
    sec_counts: dict[str, int] = {}
    dtype_counts: dict[str, int] = {}
    near_const = []
    for col in feature_cols:
        var = var_lookup.get(col)
        sec = var.section if var else "—"
        dt = var.dtype if var else "—"
        sec_counts[sec] = sec_counts.get(sec, 0) + 1
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
        if dt in ("int8 binary", "int8 categorical", "int8 ordinal"):
            ms = near_constant_share(v0[col])
            if ms > 0.95:
                near_const.append((col, sec, round(ms, 3)))
    print("  by section:")
    for s, n in sorted(sec_counts.items(), key=lambda x: -x[1]):
        print(f"    {s:<28} {n}")
    print("  by dtype:")
    for d, n in sorted(dtype_counts.items(), key=lambda x: -x[1]):
        print(f"    {d:<28} {n}")
    print(f"  near-constant (modal >95%): {len(near_const)}")
    for col, sec, ms in near_const:
        print(f"    {col:<22} [{sec}]  modal={ms}")

    # Categorize biology vs psychopathology
    biology = {"BILAN BIOLOGIQUE", "CONSTANTES ET ECG", "ANTECEDENTS",
               "PERINATALITE"}
    psych = {"AUTO-QUESTIONNAIRES", "HETERO-QUESTIONNAIRES",
             "NEUROPSYCHOLOGIE", "EVALUATION MEDICALE", "SUICIDE"}
    n_bio = sum(n for s, n in sec_counts.items() if s in biology)
    n_psy = sum(n for s, n in sec_counts.items() if s in psych)
    n_other = len(feature_cols) - n_bio - n_psy
    print(f"\n  biology-leaning sections : {n_bio}")
    print(f"  psychopathology sections : {n_psy}")
    print(f"  other (PATIENT/SOCIAL/…) : {n_other}")

    # ---- 4. selection-bias confound check --------------------------------
    # Use a data-driven floor: the highest floor that still leaves a usable
    # high-completeness group on the informative core (>= ~150 patients in
    # the smallest cohort). Falls back to the cohort-specific 75th percentile.
    usable_floor = None
    for fl in (0.95, 0.90, 0.85, 0.80, 0.75):
        if floor_table[f"{fl:.2f}"]["balanced_N"] >= 100:
            usable_floor = fl
            break
    if usable_floor is None:
        usable_floor = 0.80
    print(f"\n=== Selection-bias check: informative-core completeness "
          f">= {usable_floor:.2f} vs rest ===")
    high_mask = v0["_completeness"] >= usable_floor
    print(f"  high-completeness group: {int(high_mask.sum())} patients "
          f"({high_mask.mean()*100:.1f}% of V0)")
    # candidate clinical comparison variables present in READY
    candidates = [c for c in ("age", "madrs", "ymrs", "cgi01", "psqi",
                              "bmi", "imc") if c in v0.columns]
    confound = {}
    for col in candidates:
        res = {}
        for c in COHORTS:
            sub = v0[v0["cohort"] == c]
            res[c] = welch_smd(sub.loc[high_mask, col], sub.loc[~high_mask, col])
        confound[col] = res
        print(f"  {col}:")
        for c in COHORTS:
            r = res[c]
            if r["smd"] is None:
                print(f"    {c}: insufficient data")
            else:
                flag = "  <-- |SMD|>0.2" if abs(r["smd"]) > 0.2 else ""
                print(f"    {c}: mean_high={r['mean_high']:.2f} "
                      f"mean_rest={r['mean_rest']:.2f} SMD={r['smd']:+.3f}{flag}")

    # ---- persist ---------------------------------------------------------
    out = {
        "readiness": ["READY"],
        "n_ready_features": len(feature_cols),
        "n_informative_features": len(informative),
        "informative_features": informative,
        "n_v0_patients": int(v0["usubjid_patients"].nunique()),
        "results_by_feature_set": results_by_set,
        "section_counts": sec_counts,
        "dtype_counts": dtype_counts,
        "near_constant_features": [
            {"name": c, "section": s, "modal_share": m} for c, s, m in near_const
        ],
        "biology_vs_psych": {"biology": n_bio, "psych": n_psy, "other": n_other},
        "selection_bias_floor": usable_floor,
        "selection_bias_check": confound,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "phase2b_feasibility.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")

    # ---- verdict ---------------------------------------------------------
    bal95 = floor_table["0.95"]["balanced_N"]
    bal90 = floor_table["0.90"]["balanced_N"]
    print("\n=== VERDICT ===")
    print(f"  Max balanced discovery set at >=95% floor: {bal95} per cohort "
          f"({bal95 * 3} total)")
    print(f"  Max balanced discovery set at >=90% floor: {bal90} per cohort "
          f"({bal90 * 3} total)")
    if bal95 >= 200:
        print("  -> 200x3 discovery set IS feasible at the 95% floor.")
    else:
        print(f"  -> 200x3 NOT feasible at 95%. DR ceiling = "
              f"{floor_table['0.95']['per_cohort']['DR']}. "
              f"Consider floor=0.90 (DR={floor_table['0.90']['per_cohort']['DR']}) "
              f"or balanced N={bal95}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
