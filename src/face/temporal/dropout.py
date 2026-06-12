"""G6 — attrition / retention from the harmonized long frame (no imputation, no raw read).

`retention_table` is the M3.0 deliverable consumed by the inventory (stage 30) and the
informative-dropout model (stage 31). It counts, per cohort × visit, the unique patients with a
record at that visit and the fraction of that cohort's V0 roster — the attrition curve. Pure pandas
on the output of `build_unified_dataframe(..., format="long")`; a visit row that exists but carries
no modeled cell still counts as *retained* (the visit occurred), which is the standard attrition
denominator. The raw dropout-reason extractor (`lost_to_follow_up_*`, BP-only `chdiag`) and the
logistic informative-dropout model land here at stage 31.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

_COHORT_FILES: dict[str, str] = {"bp": "bipolar.csv", "sz": "schizophrenia.csv", "dr": "depression.csv"}

# Primary lost-to-follow-up REASON field per cohort (free text for BP/SZ; numeric codes for DR) and the
# death-DATE field. Deaths use a SENTINEL date (1900-01-01 / 1000-01-01 = "no death"), so a real death is a
# parsed date with year > 1901 — counting mere non-null cells over-counts wildly. Diagnosis-change exits
# ("Changement de diagnostic") are the only internal trace of DSM-5 instability (§A): present in BP and SZ
# reason text (sparse), coded in DR. Administrative columns, read directly (not in the modeled dictionary).
_REASON_COL: dict[str, str] = {"bp": "lost_to_followup_patient", "sz": "lost_to_follow_up", "dr": "lost_to_follow_up"}
_DEATH_COL: dict[str, str] = {"bp": "dthdtc_patient", "sz": "lost_to_follow_up_dcd", "dr": "dthdtc"}
_REASON_CODED: frozenset[str] = frozenset({"dr"})        # DR reason is a numeric code, not decoded here
_REASON_MAP: tuple[tuple[str, str], ...] = (             # accent-stripped substring -> category
    ("refus", "refusal"),
    ("chang", "diagnosis_change"),                       # "Changement de diagnostic"
    ("demenag", "moved"), ("demen", "moved"),
    ("deced", "deceased"), ("deces", "deceased"),
    ("ne sai", "unknown"),
    ("autre", "other"),
)


def _norm(s: object) -> str:
    """Lowercase + strip accents, for robust French free-text matching."""
    t = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in t if not unicodedata.combining(c)).strip().lower()


def _classify_reason(value: object) -> str | None:
    n = _norm(value)
    if n in ("", "nan", "none"):
        return None
    for token, cat in _REASON_MAP:
        if token in n:
            return cat
    return "other"


def _visit_order(v: str) -> int:
    """Sort key for visit labels 'V0'..'V10' (numeric suffix)."""
    try:
        return int(str(v)[1:])
    except (ValueError, IndexError):
        return 9999


def retention_table(long_df: pd.DataFrame, visits: list[str] | None = None) -> pd.DataFrame:
    """Per cohort × visit: unique patients present, and fraction of that cohort's V0 roster.

    Parameters
    ----------
    long_df : output of ``build_unified_dataframe(..., format="long")`` — one row per
        ``patient_uid`` × ``visit``; must carry ``patient_uid``, ``cohort``, ``visit``.
    visits : optional explicit visit list/order; default = every visit present, sorted V0..V10.

    Returns a tidy frame ``[cohort, visit, n_patients, frac_of_v0]`` (cohort-lowercased),
    visits ordered V0..V10. Retention = a record exists at that visit (no modeled-cell requirement).
    """
    need = {"patient_uid", "cohort", "visit"}
    missing = need - set(long_df.columns)
    if missing:
        raise ValueError(f"long_df missing columns {sorted(missing)}")

    d = (long_df[["patient_uid", "cohort", "visit"]]
         .dropna(subset=["visit"])
         .assign(cohort=lambda x: x["cohort"].astype(str).str.lower())
         .drop_duplicates())

    rows: list[dict] = []
    for c in sorted(d["cohort"].unique()):
        dc = d[d["cohort"] == c]
        n0 = dc.loc[dc["visit"] == "V0", "patient_uid"].nunique()
        vs = visits if visits is not None else sorted(dc["visit"].unique(), key=_visit_order)
        for v in vs:
            n = dc.loc[dc["visit"] == v, "patient_uid"].nunique()
            rows.append({"cohort": c, "visit": v, "n_patients": int(n),
                         "frac_of_v0": round(n / n0, 3) if n0 else float("nan")})
    return pd.DataFrame(rows)


def patient_retention(long_df: pd.DataFrame, visits: tuple[str, ...] = ("V1", "V2")) -> pd.DataFrame:
    """Per-patient retention flags among the V0 roster (from the long frame; no raw read, no imputation).

    Returns a frame indexed by ``(cohort, patient_id)`` — cohort lowercased, patient_id = str(usubjid),
    matching the M2 artifact keys — with ``retained_{v}`` (0/1) for each requested visit, ``n_visits``,
    and ``last_visit``. Only patients with a V0 record (the roster) are included.
    """
    need = {"cohort", "usubjid_patients", "visit"}
    missing = need - set(long_df.columns)
    if missing:
        raise ValueError(f"long_df missing columns {sorted(missing)}")
    d = long_df[["cohort", "usubjid_patients", "visit"]].dropna(subset=["visit"]).copy()
    d["cohort"] = d["cohort"].astype(str).str.lower()
    d["patient_id"] = d["usubjid_patients"].astype(str)
    seen = d.groupby(["cohort", "patient_id"])["visit"].agg(set)
    seen = seen[seen.apply(lambda s: "V0" in s)]                 # V0 roster only
    out = pd.DataFrame(index=seen.index)
    for v in visits:
        out[f"retained_{v}"] = seen.apply(lambda s: int(v in s))
    out["n_visits"] = seen.apply(len)
    out["last_visit"] = seen.apply(lambda s: max(s, key=_visit_order))
    out.index.names = ["cohort", "patient_id"]
    return out


def extract_dropout(data_dir: str | Path = "data") -> pd.DataFrame:
    """Per-patient dropout reason + death flag from the raw cohort CSVs (the only raw read in M3).

    Parses the free-text reason field (BP/SZ) into categories {refusal, moved, diagnosis_change, deceased,
    other, unknown}; DR reasons are numeric codes → "coded" (decode in M4). Deaths come from the death-date
    field with the 1900/1000 sentinel excluded (year > 1901). Descriptive only in M3 — captured chiefly for
    M4 (§A). Returns one row per patient: ``cohort, patient_id, reason, deceased, lost_flag``.
    """
    data_dir = Path(data_dir)
    parts: list[pd.DataFrame] = []
    for cohort, fname in _COHORT_FILES.items():
        rcol, dcol = _REASON_COL.get(cohort), _DEATH_COL.get(cohort)
        header = pd.read_csv(data_dir / fname, nrows=0).columns
        use = ["usubjid_patients"] + [c for c in (rcol, dcol) if c and c in header]
        raw = pd.read_csv(data_dir / fname, usecols=use, low_memory=False, encoding="utf-8-sig")
        g = pd.DataFrame({"cohort": cohort, "patient_id": raw["usubjid_patients"].astype(str)})
        if rcol in raw.columns and cohort in _REASON_CODED:
            v = raw[rcol].astype(str).str.strip()
            g["reason"] = (v.ne("") & ~v.str.lower().isin(["nan", "none"])).map({True: "coded", False: None})
        elif rcol in raw.columns:
            g["reason"] = raw[rcol].map(_classify_reason)
        else:
            g["reason"] = None
        date_death = pd.Series(False, index=raw.index)
        if dcol in raw.columns:
            dt = pd.to_datetime(raw[dcol], errors="coerce", format="ISO8601")
            date_death = dt.notna() & (dt.dt.year > 1901)            # exclude 1900/1000 sentinels
        g["deceased"] = (g["reason"].eq("deceased") | date_death).astype(int)   # text OR date
        parts.append(g)
    allg = pd.concat(parts, ignore_index=True)
    agg = allg.groupby(["cohort", "patient_id"]).agg(
        reason=("reason", lambda s: next((x for x in s if pd.notna(x)), None)),
        deceased=("deceased", "max"))
    agg["lost_flag"] = (agg["reason"].notna() | (agg["deceased"] == 1)).astype(int)
    return agg.reset_index()
