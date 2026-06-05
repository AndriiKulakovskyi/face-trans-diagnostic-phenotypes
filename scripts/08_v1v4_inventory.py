"""V1-V4 inventory + LOCKED relapse-outcome derivation (v2) — reproducibility for Studies C/D.

Two jobs:
  1. INVENTORY — per-visit attrition by cohort + coverage of candidate longitudinal outcomes
     (documents that the longitudinal arm is BP+SZ, V1-V2 horizon; DR collapses by V3).
  2. RELAPSE — the derived outcome locked in docs/legacy_v2/planning/VALIDATION_PLAN_v2.md (verified, not asserted):
       * REJECTED hospitalization-count relapse (`nboccur_hospitalisation_lt` lifetime count is
         non-monotone: 41% of consecutive pairs DECREASE -> recall noise, untrustworthy).
       * PRIMARY  CGI-S relapse by V2 = CGI-S (`cgi01`) rises >=2 OR crosses <4 -> >=4 in the
         V0->V1 or V1->V2 interval (clinician-rated event; 20% prevalence; BP 23/SZ 14/DR 8).
       * SENSITIVITY mood-syndromal = MADRS crosses <20->>=20 OR YMRS crosses <12->>=12.
Saves per-patient relapse to results/hfa/relapse.csv (keyed by `patient_uid` = COHORT::usubjid;
join to V0 dimension scores via cohort.upper()+'::'+patient_id — the case gotcha from Study A).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from trans_diag import build_unified_dataframe

warnings.simplefilter("ignore")
OUT = ROOT / "results" / "hfa"
OUT.mkdir(parents=True, exist_ok=True)
VISITS = [f"V{i}" for i in range(5)]


def derive_relapse(df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient relapse-by-V2 (CGI primary + mood sensitivity). Available-case, no imputation."""
    d = df[df["visit"].isin(VISITS)].drop_duplicates(["patient_uid", "visit"]).copy()
    for c in ("cgi01", "madrs", "ymrs"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    pkey = d.drop_duplicates("patient_uid").set_index("patient_uid")["cohort"]
    idx = pkey.index    # common patient index — the 3 pivots otherwise have different indices
    cgi = d.pivot_table(index="patient_uid", columns="visit", values="cgi01", aggfunc="first").reindex(idx)
    mad = d.pivot_table(index="patient_uid", columns="visit", values="madrs", aggfunc="first").reindex(idx)
    ymr = d.pivot_table(index="patient_uid", columns="visit", values="ymrs", aggfunc="first").reindex(idx)
    for w in (cgi, mad, ymr):                  # ensure all 5 visit columns exist
        for v in VISITS:
            if v not in w.columns:
                w[v] = np.nan

    def cgi_iv(a, b):
        m = (cgi[a].notna() & cgi[b].notna()).fillna(False)
        r = (((cgi[b] - cgi[a]) >= 2) | ((cgi[a] < 4) & (cgi[b] >= 4))).fillna(False)
        return m, (r & m)

    def mood_iv(a, b):
        m = ((mad[a].notna() & mad[b].notna()) | (ymr[a].notna() & ymr[b].notna())).fillna(False)
        r = (((mad[a] < 20) & (mad[b] >= 20)) | ((ymr[a] < 12) & (ymr[b] >= 12))).fillna(False)
        return m, (r & m)

    cm01, cr01 = cgi_iv("V0", "V1"); cm12, cr12 = cgi_iv("V1", "V2")
    mm01, mr01 = mood_iv("V0", "V1"); mm12, mr12 = mood_iv("V1", "V2")
    out = pd.DataFrame(index=cgi.index)
    out["cohort"] = pkey
    out["cgi_evaluable"] = (cm01 | cm12)
    out["relapse_cgi_byV2"] = (cr01 | cr12).astype(int)
    out["mood_evaluable"] = (mm01 | mm12)
    out["relapse_mood_byV2"] = (mr01 | mr12).astype(int)
    return out


def main() -> None:
    df = build_unified_dataframe(str(ROOT / "data"), str(ROOT / "data" / "face-common-vars.xlsx"),
                                 readiness=["READY", "PARTIAL", "NOT USABLE", "ID"], format="long")

    print("=== attrition: patients per visit by cohort ===")
    vis = df[df.visit.isin(VISITS)].groupby(["visit", "cohort"])["patient_uid"].nunique().unstack(fill_value=0)
    print(vis.to_string())

    print("\n=== outcome coverage at V1/V2 (% non-NaN among patients at visit) ===")
    for c in ["egf", "fast", "madrs", "ymrs", "cgi01", "eq5d", "isf05"]:
        cov = {v: df.loc[df.visit == v, c].notna().mean() * 100 for v in ("V0", "V1", "V2")}
        print(f"  {c:10s} " + "  ".join(f"{v}={cov[v]:.0f}%" for v in cov))

    rel = derive_relapse(df)
    ev = rel[rel.cgi_evaluable]
    print("\n=== LOCKED relapse outcome (CGI-S by V2) ===")
    print(f"  evaluable n={len(ev)}  overall relapse {ev.relapse_cgi_byV2.mean()*100:.0f}%")
    for c in ["BP", "SZ", "DR"]:
        s = ev[ev.cohort == c]
        print(f"    {c}: n={len(s):5d}  relapse_by_V2={s.relapse_cgi_byV2.mean()*100:.0f}%")
    print(f"  mood-syndromal (sensitivity): evaluable n={int(rel.mood_evaluable.sum())}  "
          f"relapse {rel[rel.mood_evaluable].relapse_mood_byV2.mean()*100:.0f}%")

    rel.to_csv(OUT / "relapse.csv")
    print(f"\nsaved -> {OUT}/relapse.csv  (per-patient; gitignored)")


if __name__ == "__main__":
    main()
