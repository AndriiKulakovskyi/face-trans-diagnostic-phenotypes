"""60 — Manuscript Table 1: sample characteristics by cohort (AGGREGATE-ONLY).

Reads the harmonized per-patient demographics (cohort, age, sex, education, DSM-5
arm, site) for the V0 baseline and the longitudinal retention counts, and writes
**only aggregate** summaries to reports/ + article/tables/. No per-patient row is
ever written out — confidential-safe by construction.

Sources (confidential, gitignored):
  results/face/m2/validation_table.parquet   age/sex/education_years/arm/site/cohort, N=9013
  reports/30_retention.csv                    V1/V2 retention per cohort (already aggregate)

Sex coding (src/face/data/rules.py): 0=male, 1=female.

Outputs (shareable aggregates):
  reports/table1_characteristics.csv          tidy long aggregate
  reports/table1_characteristics.md           formatted Table 1 + DSM-5 subtype panel
  article/tables/table1_characteristics.md    copy for the manuscript
Run:  python3 scripts/60_table1.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
VALID = ROOT / "results/face/m2/validation_table.parquet"
RETENTION = ROOT / "reports/30_retention.csv"
OUT_CSV = ROOT / "reports/table1_characteristics.csv"
OUT_MD = ROOT / "reports/table1_characteristics.md"
OUT_MD_ART = ROOT / "article/tables/table1_characteristics.md"

COHORTS = ["bp", "sz", "dr"]
LABEL = {"bp": "Bipolar (BP)", "sz": "Schizophrenia (SZ)", "dr": "Depression (DR)"}
# Source DSM-5 arm labels are French; translate for the manuscript.
ARM_EN = {
    "Bipolaire de type 1": "Bipolar I",
    "Bipolaire de type 2": "Bipolar II",
    "Bipolaire non spécifié": "Bipolar, unspecified",
    "Schizophrénie": "Schizophrenia",
    "Trouble schizo-affectif": "Schizoaffective disorder",
    "Trouble schizophréniforme": "Schizophreniform disorder",
    "Trouble dépressif majeur": "Major depressive disorder",
}


def _msd(x: pd.Series) -> str:
    x = x.dropna()
    return f"{x.mean():.1f} ({x.std():.1f})" if len(x) else "—"


def _med_iqr(x: pd.Series) -> str:
    x = x.dropna()
    if not len(x):
        return "—"
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f"{med:.0f} [{q1:.0f}–{q3:.0f}]"


def _pct(n: int, d: int) -> str:
    return f"{n} ({100 * n / d:.1f}%)" if d else "—"


def _miss(x: pd.Series) -> str:
    n = int(x.isna().sum())
    return f"{n} ({100 * n / len(x):.1f}%)" if len(x) else "—"


def main() -> None:
    df = pd.read_parquet(VALID)
    df["cohort"] = df["cohort"].str.lower()
    ret = pd.read_csv(RETENTION)
    ret["cohort"] = ret["cohort"].str.lower()

    def ret_frac(cohort: str, visit: str) -> float:
        sub = ret[(ret.cohort == cohort) & (ret.visit == visit)]
        return float(sub.frac_of_v0.iloc[0]) if len(sub) else np.nan

    groups = {c: df[df.cohort == c] for c in COHORTS}
    groups["overall"] = df
    order = COHORTS + ["overall"]

    rows = []

    def add(characteristic: str, fn) -> None:
        rows.append({"characteristic": characteristic, **{g: fn(groups[g]) for g in order}})

    add("N", lambda g: f"{len(g)}")
    add("Age, years — mean (SD)", lambda g: _msd(g.age))
    add("Age, years — median [IQR]", lambda g: _med_iqr(g.age))
    add("Female sex — n (%)", lambda g: _pct(int((g.sex == 1).sum()), int(g.sex.notna().sum())))
    add("Education, years — mean (SD)", lambda g: _msd(g.education_years))
    add("Recruitment sites — n", lambda g: f"{g.siteid_city.nunique(dropna=True)}")
    add("Age missing — n (%)", lambda g: _miss(g.age))
    add("Sex missing — n (%)", lambda g: _miss(g.sex))
    add("Education missing — n (%)", lambda g: _miss(g.education_years))

    # retention rows (per-cohort from 30_retention.csv; overall = N-weighted)
    n_tot = len(df)
    for visit, lab in [("V1", "Retained at V1 (12 mo) — % of V0"),
                       ("V2", "Retained at V2 (24 mo) — % of V0")]:
        vals = {}
        for c in COHORTS:
            f = ret_frac(c, visit)
            vals[c] = f"{100 * f:.1f}%" if not np.isnan(f) else "—"
        ov = np.nansum([ret_frac(c, visit) * len(groups[c]) for c in COHORTS]) / n_tot
        vals["overall"] = f"{100 * ov:.1f}%"
        rows.append({"characteristic": lab, **vals})

    tab = pd.DataFrame(rows)[["characteristic", *order]]

    # DSM-5 subtype panel (per cohort), aggregate counts only
    sub_rows = []
    for c in COHORTS:
        g = groups[c]
        vc = g.arm.value_counts(dropna=False)
        for name, n in vc.items():
            if pd.isna(name) or str(name).strip() == "":
                label = "Not recorded"
            else:
                label = ARM_EN.get(str(name), str(name))
            sub_rows.append({"cohort": LABEL[c], "DSM-5 subtype": label,
                             "n": int(n), "pct_of_cohort": round(100 * n / len(g), 1)})
    subtab = pd.DataFrame(sub_rows)

    # ---- write tidy CSV (aggregate only) ----
    tab_long = tab.melt(id_vars="characteristic", var_name="group", value_name="value")
    tab_long.to_csv(OUT_CSV, index=False)

    # ---- write formatted markdown ----
    hdr = "Bipolar (BP) | Schizophrenia (SZ) | Depression (DR) | Overall"
    lines = []
    lines.append("# Table 1 — Sample characteristics by cohort (FACE V0 baseline)")
    lines.append("")
    lines.append("Aggregate-only summary of the harmonized V0 baseline (N = 9,013). "
                 "Sex coded 0 = male, 1 = female. Per-patient data are confidential "
                 "(Fondation FondaMental) and never reproduced; this table is reproducible via "
                 "`scripts/60_table1.py`.")
    lines.append("")
    lines.append(f"| Characteristic | {hdr} |")
    lines.append("|" + "---|" * 5)
    for _, r in tab.iterrows():
        lines.append(f"| {r['characteristic']} | {r['bp']} | {r['sz']} | {r['dr']} | {r['overall']} |")
    lines.append("")
    lines.append("## DSM-5 subtype distribution (within cohort)")
    lines.append("")
    lines.append("| Cohort | DSM-5 subtype | n | % of cohort |")
    lines.append("|---|---|---:|---:|")
    for _, r in subtab.iterrows():
        lines.append(f"| {r['cohort']} | {r['DSM-5 subtype']} | {r['n']} | {r['pct_of_cohort']} |")
    lines.append("")
    lines.append("> Notes. Age/education are years. Retention rows are % of the V0 cohort retained at "
                 "the 12- and 24-month visits (source `reports/30_retention.csv`); the overall value is "
                 "N-weighted across cohorts. Recruitment-site counts are distinct sites contributing "
                 "patients per cohort. DSM-5 arm is validation/metadata only — never a model input.")
    md = "\n".join(lines) + "\n"
    OUT_MD.write_text(md)
    OUT_MD_ART.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD_ART.write_text(md)

    print(f"wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    print(f"wrote {OUT_MD_ART.relative_to(ROOT)}")
    print()
    print(md)


if __name__ == "__main__":
    main()
