"""M5.1 — the treatment-exposure harmonization layer (pure-logic tests)."""
from __future__ import annotations

import pandas as pd

from face.treatment.medications import (
    CLASSES,
    atc_to_classes,
    classstr_to_classes,
    extract_exposures,
)


def test_atc_longest_prefix_and_classes():
    # lithium (N05AN01) must win over the antipsychotic prefix (N05A)
    assert atc_to_classes("['N05AN01']") == {"lithium"}
    assert atc_to_classes(["N05AH02"]) == {"antipsychotic"}          # clozapine
    assert atc_to_classes("['N06AB06']") == {"antidepressant"}       # an SSRI
    assert atc_to_classes(["N03AG01"]) == {"mood_stabilizer"}        # valproate
    assert atc_to_classes(["N05BA04"]) == {"anxiolytic"}             # a benzodiazepine
    # a real multi-drug list -> the union of classes
    cell = "['N05AH02', 'N06AB06', 'N03AX09', 'N05BA04']"
    assert atc_to_classes(cell) == {"antipsychotic", "antidepressant", "mood_stabilizer", "anxiolytic"}
    assert atc_to_classes(None) == set() and atc_to_classes("[]") == set()


def test_classstr_french_vocab_complete():
    # the DR vocab the M5.0 audit missed must now map
    assert classstr_to_classes("['NORMOTHYMIQUES']") == {"mood_stabilizer"}
    assert classstr_to_classes("['TRANQUILLISANTS']") == {"anxiolytic"}
    assert classstr_to_classes("['ANTIPSYCHOTIQUES ATYPIQUES', 'AUTRES ANTIDEPRESSEURS']") == {
        "antipsychotic", "antidepressant"}


def test_extract_bp_lifetime_flags_and_nan_honesty():
    raw = pd.DataFrame({
        "usubjid_patients": ["1", "2", "3"],
        "visit": ["V0", "V0", "V0"],
        "cmoccur_lithi": ["Y", "N", None],          # patient 3 has no record -> NaN, not 0
        "cmoccur_antip": ["N", "Y", "N"], "cmoccur_neuro": ["N", "N", "N"],
        "cmoccur_antid": ["Y", "Y", "Y"], "cmoccur_thymo": ["N", "N", "N"],
        "cmoccur_benzo": ["N", "N", "N"], "lithiumplasma": [0.8, None, None],
    })
    ex = extract_exposures("bp", raw, visit="V0")
    assert ex.loc["1", "on_lithium"] == 1.0 and ex.loc["2", "on_lithium"] == 0.0
    assert pd.isna(ex.loc["3", "on_lithium"])               # no imputation
    assert ex.loc["2", "on_antipsychotic"] == 1.0           # cmoccur_antip OR cmoccur_neuro
    assert ex.loc["1", "lithium_plasma"] == 0.8
    assert (ex["temporality"] == "lifetime").all()


def test_extract_sz_atc_current_and_clozapine():
    raw = pd.DataFrame({
        "usubjid_patients": ["10", "11"],
        "visit": ["V0", "V0"],
        "med_psy_code_atc": ["['N05AH02', 'N06AB06']", "['N05BA04']"],
        "rad_clozapine": ["Oui", "Non"],
    })
    ex = extract_exposures("sz", raw, visit="V0")
    assert ex.loc["10", "on_antipsychotic"] == 1.0 and ex.loc["10", "on_antidepressant"] == 1.0
    assert ex.loc["10", "on_clozapine"] == 1.0 and ex.loc["11", "on_clozapine"] == 0.0
    assert ex.loc["11", "on_anxiolytic"] == 1.0 and ex.loc["11", "on_antipsychotic"] == 0.0
    assert set(f"on_{c}" for c in CLASSES).issubset(ex.columns) and (ex["temporality"] == "current").all()
