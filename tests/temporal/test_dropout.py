"""M3 G6 — dropout/retention unit tests (synthetic frames + a guarded raw-data smoke test)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from face.temporal.dropout import (_classify_reason, _norm, _visit_order, patient_retention,
                                   retention_table)

_DATA = Path(__file__).resolve().parents[2] / "data"


def _long():
    # 3 BP patients (p1 V0/V1/V2, p2 V0/V1, p3 V0) + 1 SZ patient (V0/V1). Duplicate rows allowed.
    rows = [
        ("BP::1", "BP", "V0"), ("BP::1", "BP", "V1"), ("BP::1", "BP", "V2"),
        ("BP::2", "BP", "V0"), ("BP::2", "BP", "V1"),
        ("BP::3", "BP", "V0"), ("BP::3", "BP", "V0"),     # dup V0 row → counted once
        ("SZ::1", "SZ", "V0"), ("SZ::1", "SZ", "V1"),
    ]
    return pd.DataFrame(rows, columns=["patient_uid", "cohort", "visit"])


def test_visit_order():
    assert [_visit_order(v) for v in ("V0", "V2", "V10")] == [0, 2, 10]
    assert _visit_order("Vx") == 9999          # unparseable → sorts last


def test_retention_counts_and_fractions():
    r = retention_table(_long())
    bp = r[r.cohort == "bp"].set_index("visit")
    assert int(bp.loc["V0", "n_patients"]) == 3     # p1,p2,p3 (dup collapsed)
    assert int(bp.loc["V1", "n_patients"]) == 2     # p1,p2
    assert int(bp.loc["V2", "n_patients"]) == 1     # p1
    assert bp.loc["V0", "frac_of_v0"] == 1.0
    assert bp.loc["V1", "frac_of_v0"] == round(2 / 3, 3)
    assert bp.loc["V2", "frac_of_v0"] == round(1 / 3, 3)


def test_retention_cohort_lowercased_and_visits_ordered():
    r = retention_table(_long())
    assert set(r.cohort.unique()) == {"bp", "sz"}    # cohort labels lowercased
    sz = r[r.cohort == "sz"]
    assert list(sz.visit) == sorted(sz.visit, key=_visit_order)   # V0 before V1


def test_retention_explicit_visit_window():
    r = retention_table(_long(), visits=["V0", "V1", "V2"])
    # SZ has no V2 record → n_patients 0, frac 0.0 (the window is honored, not silently dropped)
    sz = r[r.cohort == "sz"].set_index("visit")
    assert int(sz.loc["V2", "n_patients"]) == 0
    assert sz.loc["V2", "frac_of_v0"] == 0.0


def test_retention_missing_columns_raises():
    with pytest.raises(ValueError, match="missing columns"):
        retention_table(pd.DataFrame({"patient_uid": ["x"], "visit": ["V0"]}))


# ---- patient_retention (per-patient flags) ----

def _long_full():
    rows = [("BP", 1, "V0"), ("BP", 1, "V1"), ("BP", 1, "V2"),
            ("BP", 2, "V0"), ("BP", 2, "V1"),
            ("BP", 3, "V0"),
            ("SZ", 1, "V0"), ("SZ", 1, "V2")]            # present V0,V2 — skipped V1
    df = pd.DataFrame(rows, columns=["cohort", "usubjid_patients", "visit"])
    df["patient_uid"] = df.cohort + "::" + df.usubjid_patients.astype(str)
    return df


def test_patient_retention_flags():
    r = patient_retention(_long_full(), visits=("V1", "V2"))
    assert int(r.loc[("bp", "1"), "retained_V1"]) == 1 and int(r.loc[("bp", "1"), "retained_V2"]) == 1
    assert int(r.loc[("bp", "2"), "retained_V1"]) == 1 and int(r.loc[("bp", "2"), "retained_V2"]) == 0
    assert int(r.loc[("bp", "3"), "retained_V1"]) == 0
    assert int(r.loc[("sz", "1"), "retained_V1"]) == 0 and int(r.loc[("sz", "1"), "retained_V2"]) == 1
    assert int(r.loc[("bp", "1"), "n_visits"]) == 3
    assert r.loc[("sz", "1"), "last_visit"] == "V2"


def test_patient_retention_v0_roster_only():
    df = pd.DataFrame([["SZ", 9, "V1"]], columns=["cohort", "usubjid_patients", "visit"])
    df["patient_uid"] = "SZ::9"
    assert ("sz", "9") not in patient_retention(df).index   # no V0 record → not in the roster


# ---- reason parsing (accent-robust French free text) ----

def test_norm_strips_accents():
    assert _norm("Décédé") == "decede"
    assert _norm("  Refus du Patient ") == "refus du patient"


@pytest.mark.parametrize("text,cat", [
    ("Changement de diagnostic", "diagnosis_change"),
    ("Refus du patient", "refusal"),
    ("Déménagement", "moved"),
    ("Patient décédé", "deceased"),
    ("Ne sais pas", "unknown"),
    ("Autre", "other"),
    ("quelque chose inattendu", "other"),
])
def test_classify_reason(text, cat):
    assert _classify_reason(text) == cat


def test_classify_reason_empty():
    assert _classify_reason("") is None and _classify_reason(float("nan")) is None


# ---- extract_dropout: guarded raw-data smoke test (regression guard for the death-sentinel bug) ----

@pytest.mark.skipif(not (_DATA / "bipolar.csv").exists(), reason="raw cohort CSVs not present")
def test_extract_dropout_schema_and_plausibility():
    from face.temporal.dropout import extract_dropout
    d = extract_dropout(str(_DATA))
    assert {"cohort", "patient_id", "reason", "deceased", "lost_flag"}.issubset(d.columns)
    assert int(d.duplicated(["cohort", "patient_id"]).sum()) == 0          # one row per patient
    death_rate = d.groupby("cohort")["deceased"].mean()
    assert (death_rate < 0.10).all()      # sentinel-corrected: would FAIL at 0.44 (the 966-deaths bug)
    dc_cohorts = set(d.loc[d.reason == "diagnosis_change", "cohort"])
    assert {"bp", "sz"} <= dc_cohorts     # diagnosis-change exits in BP and SZ reason text (§A)
