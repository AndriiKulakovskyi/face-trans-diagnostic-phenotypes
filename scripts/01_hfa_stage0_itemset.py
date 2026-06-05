"""Stage 0 (v2) — freeze the hierarchical-FA item set, masked correlation, factorability.

Implements Stage 0 of docs/legacy_v2/planning/HIERARCHICAL_FA_PLAN.md. V0 anchor; masked / no-imputation.

Loads the FULL dictionary (incl. rows currently flagged NOT USABLE / ID) so that "add every
valid measurement" is honoured, then applies one explicit, documented exclude list:
  - identifiers / administrative;
  - covariates routed to residualization (age, sex) or dropped (education_years, cclin01=age dup);
  - degenerate columns (brthdtc = date->1e18 artifact / PII);
  - cohort-incomparable confounds (hcg = pregnancy, clozapin = SZ treatment marker);
  - D8 branching suicide-attempt method/lethality items (ltsv*/ltsg*: <6% observed, 0 complete
    cases, every pair below the masked min_pair=100 floor -> inert);
  - by-construction collinear TMT B-A (tmtba01 = TMT-B - TMT-A, both already in).

Emits the frozen item set + robust-z matrices (residualized primary + raw sensitivity, per D7)
+ factorability diagnostics (KMO/MSA, eigenvalues, coverage) for Stage 1.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset
from trans_diag.domains import _robust_z
from trans_diag.masked_fa import masked_correlation

warnings.simplefilter("ignore")

DATA = ROOT / "data"
DICT = DATA / "face-common-vars.xlsx"
OUT = ROOT / "results" / "hfa"
OUT.mkdir(parents=True, exist_ok=True)
MIN_PAIR = 100

# ---- explicit exclude list (D6-D9 locked; see docs/legacy_v2/planning/HIERARCHICAL_FA_PLAN.md) ----
IDENT = {"usubjid_patients", "fondacode", "cohort", "arm", "armcd", "visit", "visitnum", "siteid_city"}
COVAR = {"age", "sex", "education_years", "cclin01"}     # age,sex -> residualize_on; edu/cclin01 dropped
DEGEN = {"brthdtc"}
CONFOUND = {"hcg_lbstresc", "clozapin", "oxcarbaz"}   # pregnancy + drug-level treatment markers
BRANCHING = {f"ltsv0{i}" for i in range(1, 10)} | {f"ltsg0{i}" for i in range(1, 7)}
COLLINEAR = {"tmtba01"}
EXCLUDE_ALL = IDENT | COVAR | DEGEN | CONFOUND | BRANCHING | COLLINEAR
RESID = ("age", "sex")


def factorability(R: np.ndarray, Z: pd.DataFrame, min_pair: int, lam: float = 0.05):
    """Factorability diagnostics robust to a near-singular masked correlation.

    Plain KMO inverts R; the masked + nearest-PD matrix is near-singular (instrument
    totals co-present with their sub-scores, collinear labs, + the 1e-8 eigen-clip), so its
    inverse explodes and KMO/MSA collapse to ~0 — an artifact, not unfactorability. We therefore:
      - report KMO/MSA on a *ridge-shrunk* matrix (1-lam)R + lam*I (stable inverse);
      - report a robust per-item fitness = mean |off-diagonal r| (no inverse);
      - count near-zero eigenvalues of the *pre-repair* pairwise correlation = redundancies to
        prune in Stage 1/2. The eigenvalue scree (leading factors) is the real factorability signal.
    """
    Rs = (1.0 - lam) * R + lam * np.eye(R.shape[0])
    Ri = np.linalg.inv(Rs)
    d = np.sqrt(np.clip(np.diag(Ri), 1e-12, None))
    P = -Ri / np.outer(d, d)
    np.fill_diagonal(P, 0.0)
    Roff = R.copy()
    np.fill_diagonal(Roff, 0.0)
    r2c, p2c = (Roff ** 2).sum(0), (P ** 2).sum(0)
    msa = r2c / (r2c + p2c)
    kmo_shrunk = float((Roff ** 2).sum() / ((Roff ** 2).sum() + (P ** 2).sum()))
    # per-item fitness = STRONGEST correlation with any other item (mean-over-all under-rates
    # items that belong to a small tight cluster, e.g. labs). max|r|<0.15 => truly isolated.
    max_abs_r = np.abs(Roff).max(0)

    w = np.linalg.eigvalsh(R)
    cond = float(w[-1] / max(w[0], 1e-12))
    n_floor = int((w < 1e-7).sum())     # eigenvalues at the nearest-PD floor (rank deficiency)
    return dict(kmo_shrunk=kmo_shrunk, msa=msa, max_abs_r=max_abs_r,
                cond=cond, n_eig_floor=n_floor)


def main() -> None:
    df = build_unified_dataframe(
        str(DATA), str(DICT),
        readiness=["READY", "PARTIAL", "NOT USABLE", "ID"], format="long",
    )
    variables = load_variables(str(DICT))
    by = {v.canonical_name: v for v in variables}

    def build(residualize: bool):
        return to_harmonized_dataset(
            df, variables, visit="V0", sections=None,
            exclude=(EXCLUDE_ALL - set(RESID)) if residualize else EXCLUDE_ALL,
            residualize_on=RESID if residualize else None,
            normalize=False,
        )

    ds = build(True)          # primary: residualized on age+sex
    ds_raw = build(False)     # sensitivity: un-residualized (age/sex still excluded)

    cols = [c for c in ds.X.columns if c in ds_raw.X.columns]
    Xr, Xu = ds.X[cols], ds_raw.X[cols]
    Zr, Zu = Xr.apply(_robust_z), Xu.apply(_robust_z)
    meta = ds.metadata.loc[Xr.index]
    coh = meta["cohort"]

    print(f"Frozen item set: {len(cols)} items | V0 patients: {len(Xr)} "
          f"{ {k: int(v) for k, v in coh.value_counts().items()} }")

    # per-item coverage
    rows = []
    for c in cols:
        s = Xr[c]
        rec = {"item": c, "section": by[c].section, "dtype": by[c].dtype,
               "readiness": by[c].cluster_readiness.split("—")[0].strip()}
        for cc in ("bp", "sz", "dr"):
            rec[f"n_{cc}"] = int(s[(coh == cc).values].notna().sum())
        rec["obs_frac"] = float(s.notna().mean())
        rows.append(rec)
    items_df = pd.DataFrame(rows)

    R = masked_correlation(Zr, MIN_PAIR)
    fac = factorability(R, Zr, MIN_PAIR)
    items_df["msa_shrunk"] = fac["msa"]
    items_df["max_abs_r"] = fac["max_abs_r"]
    ev = np.sort(np.linalg.eigvalsh(R))[::-1]

    newly = sorted(c for c in cols if not by[c].cluster_readiness.startswith(("READY", "PARTIAL")))
    excluded_present = sorted(EXCLUDE_ALL & set(by))
    isolated = sorted(items_df.loc[items_df.max_abs_r < 0.15, "item"])   # correlate with ~nothing
    lowcov = sorted(items_df.loc[items_df.obs_frac < 0.10, "item"])

    print("factorability (robust to near-singularity):")
    print(f"  eigenvalues >1: {int((ev > 1).sum())}  | top12: {np.round(ev[:12], 2)}  "
          f"(clear leading factors -> factorable)")
    print(f"  ridge-shrunk KMO (lam=0.05) = {fac['kmo_shrunk']:.3f}  | cond(R)={fac['cond']:.1e}, "
          f"{fac['n_eig_floor']} eigenvalues at PD-floor (plain KMO undefined -> use scree)")
    print(f"  isolated items (max|r|<0.15, load on ~nothing): {len(isolated)}  {isolated}")
    print(f"  low-coverage items (obs_frac<0.10): {len(lowcov)}  {lowcov[:14]}")
    print(f"newly-included (were NOT USABLE/ID): {len(newly)}  {newly}")
    print(f"excluded canonicals (present in dict): {len(excluded_present)}")

    # ---- save artifacts ----
    items_df.sort_values(["section", "item"]).to_csv(OUT / "stage0_items.csv", index=False)
    np.savez(OUT / "stage0_corr_resid.npz", R=R, cols=np.array(cols, dtype=object))
    Zr.to_pickle(OUT / "stage0_Z_resid.pkl")
    Zu.to_pickle(OUT / "stage0_Z_raw.pkl")
    meta.to_pickle(OUT / "stage0_meta.pkl")
    json.dump(
        {"n_items": len(cols), "n_patients": int(len(Xr)),
         "cohort_n": {k: int(v) for k, v in coh.value_counts().items()},
         "kmo_shrunk": fac["kmo_shrunk"], "cond_R": fac["cond"],
         "n_eig_floor": fac["n_eig_floor"], "isolated_items": isolated, "low_coverage_items": lowcov,
         "eig_gt1": int((ev > 1).sum()), "eig_top20": [float(x) for x in ev[:20]],
         "newly_included": newly, "min_pair": MIN_PAIR},
        open(OUT / "stage0_diagnostics.json", "w"), indent=2,
    )
    print(f"\nsaved -> {OUT}/stage0_*.*")


if __name__ == "__main__":
    main()
