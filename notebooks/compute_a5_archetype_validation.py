"""Copula A=5 archetype-level validation metrics (M2).

Computes the archetype-level transdiagnostic / not-just-severity / descriptive-fit
metrics for the REPORTED A=5 simplex (arm A) on the reported 8-factor copula map
(coord_source=copula_weighted_8d, model_version strata_oop_2026_06_26_v2_8factor).

Background: the legacy ``reports/23b_archetype_compare.md`` and
``reports/24_validation.md`` reported these numbers on a SUPERSEDED native 9-d
A=8 fit (separate metabolic+inflammatory axes). This driver recomputes them on
the reported copula A=5 dominant assignment so the docs cite the right object.

Reads:
  - results/m2_strata/consolidate/patient_strata.parquet  (arch_dominant, arm=DSM dx, cohort)
  - results/m2_strata/coordinates/coordinates_full.parquet (per-axis posterior-mean coords)
Writes:
  - results/m2_strata/usefulness/a5_archetype_validation.json
  - results/m2_strata/usefulness/a5_archetype_validation.csv  (eta^2 per axis x grouping)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

ROOT = Path(__file__).resolve().parents[1]
STRATA = ROOT / "results" / "face" / "strata_oop"

AXES = [
    "overall_severity", "cognition", "immunometabolic", "sleep",
    "mania_activation", "suicidality", "developmental_risk", "substance",
]
SPECIFICS = [a for a in AXES if a != "overall_severity"]

ARCH_LABEL = {  # arch_dominant int -> short name (from archetype_profiles.csv signs)
    0: "A0 activation/sleep",
    1: "A1 severe clean-biology",
    2: "A2 immunometabolic corner",
    3: "A3 trauma/suicidality",
    4: "A4 low-burden/well",
}


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramer's V between two categorical labelings."""
    cm = contingency_matrix(a, b)
    chi2 = _chi2(cm)
    n = cm.sum()
    r, k = cm.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def _chi2(cm: np.ndarray) -> float:
    cm = cm.astype(float)
    n = cm.sum()
    exp = cm.sum(1, keepdims=True) * cm.sum(0, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(exp > 0, (cm - exp) ** 2 / exp, 0.0)
    return float(terms.sum())


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way ANOVA eta^2 = SS_between / SS_total for one axis."""
    grand = values.mean()
    ss_total = ((values - grand) ** 2).sum()
    ss_between = 0.0
    for g in np.unique(groups):
        v = values[groups == g]
        ss_between += len(v) * (v.mean() - grand) ** 2
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def main() -> None:
    ps = pd.read_parquet(STRATA / "consolidate" / "patient_strata.parquet")
    co = pd.read_parquet(STRATA / "coordinates" / "coordinates_full.parquet")
    df = ps.merge(
        co[["cohort", "patient_id"] + [f"{a}__mean" for a in AXES]],
        on=["cohort", "patient_id"], how="inner", validate="1:1",
    )
    assert len(df) == 9013, f"expected 9013 rows, got {len(df)}"

    arch = df["arch_dominant"].to_numpy()
    dsm5 = df["arm"].astype("category")
    cohort = df["cohort"].astype("category")

    # --- transdiagnostic: ARI + Cramer's V of A=5 dominant vs DSM-5 / cohort ---
    transdiagnostic = {
        "ari_dsm5": float(adjusted_rand_score(dsm5, arch)),
        "ari_cohort": float(adjusted_rand_score(cohort, arch)),
        "cramers_v_dsm5": cramers_v(arch, dsm5),
        "cramers_v_cohort": cramers_v(arch, cohort),
        "n_dsm5_levels": int(dsm5.nunique()),
        "n_cohort_levels": int(cohort.nunique()),
    }

    # --- eta^2 per axis for each grouping (archetype A=5 vs DSM-5 vs cohort) ---
    rows = []
    eta_by_group = {}
    for gname, gvals in [("archetype_A5", arch),
                          ("dsm5", dsm5.cat.codes.to_numpy()),
                          ("cohort", cohort.cat.codes.to_numpy())]:
        per_axis = {a: eta_squared(df[f"{a}__mean"].to_numpy(), gvals) for a in AXES}
        eta_by_group[gname] = per_axis
        for a, e in per_axis.items():
            rows.append({"grouping": gname, "axis": a, "eta_sq": round(e, 4)})
    eta_csv = pd.DataFrame(rows)

    arch_eta = eta_by_group["archetype_A5"]
    not_just_severity = {
        "eta_G_overall_severity": round(arch_eta["overall_severity"], 4),
        "mean_eta_specifics": round(float(np.mean([arch_eta[a] for a in SPECIFICS])), 4),
        "eta_per_axis": {a: round(arch_eta[a], 4) for a in AXES},
        "top_axes": sorted(arch_eta.items(), key=lambda kv: -kv[1])[:3],
    }

    # --- descriptive separation: mean eta^2 over axes, archetype vs DSM-5 vs cohort ---
    tighter_than_dsm5 = {
        "mean_eta_all_axes_archetype_A5": round(
            float(np.mean(list(arch_eta.values()))), 4),
        "mean_eta_all_axes_dsm5": round(
            float(np.mean(list(eta_by_group["dsm5"].values()))), 4),
        "mean_eta_all_axes_cohort": round(
            float(np.mean(list(eta_by_group["cohort"].values()))), 4),
    }
    tighter_than_dsm5["archetype_over_dsm5_ratio"] = round(
        tighter_than_dsm5["mean_eta_all_axes_archetype_A5"]
        / tighter_than_dsm5["mean_eta_all_axes_dsm5"], 2)

    # --- population shares + cohort composition ---
    counts = df["arch_dominant"].value_counts().sort_index()
    shares = {ARCH_LABEL[int(k)]: {"n": int(v), "share": round(v / len(df), 4)}
              for k, v in counts.items()}
    comp = pd.crosstab(df["arch_dominant"], df["cohort"], normalize="columns").round(3)
    comp.index = [ARCH_LABEL[int(i)] for i in comp.index]
    cohort_composition = comp.to_dict()

    out = {
        "object": "copula A=5 archetype simplex (arm A, A_all9)",
        "model_version": "strata_oop_2026_06_26_v2_8factor",
        "coord_source": "copula_weighted_8d",
        "N": int(len(df)),
        "note": ("Archetype-level validation recomputed on the REPORTED copula A=5 "
                 "dominant assignment. Supersedes the native 9-d A=8 numbers in "
                 "legacy reports/23b_archetype_compare.md and reports/24_validation.md."),
        "transdiagnostic": transdiagnostic,
        "not_just_severity": not_just_severity,
        "tighter_than_dsm5_descriptive": tighter_than_dsm5,
        "population_shares": shares,
        "cohort_composition": cohort_composition,
    }

    out_dir = STRATA / "usefulness"
    (out_dir / "a5_archetype_validation.json").write_text(json.dumps(out, indent=2))
    eta_csv.to_csv(out_dir / "a5_archetype_validation.csv", index=False)

    print(json.dumps(out, indent=2))
    print("\nwrote:", out_dir / "a5_archetype_validation.json")
    print("wrote:", out_dir / "a5_archetype_validation.csv")


if __name__ == "__main__":
    main()
