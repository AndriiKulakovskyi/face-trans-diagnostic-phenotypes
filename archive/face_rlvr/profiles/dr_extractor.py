"""Structured data extraction from DR.csv rows.

Extracts treatment-resistant depression-specific data: treatment resistance
staging, CSSRS binary ideation, DR-specific biology, plus shared clinical
data (demographics, biology, treatments, family history, suicide).

Reuses generic dataclasses and helpers from bp_extractor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from face_rlvr.profiles.common_extractors import (
    # Generic dataclasses
    Demographics,
    LabValue,
    BiologicalPanel,
    SubstanceUse,
    FamilyHistory,
    RelativeHistory,
    SuicideHistory,
    HospitalizationHistory,
    CognitiveProfile,
    CurrentEpisodeCriteria,
    MostRecentEpisode,
    # Helpers
    _safe_float,
    _safe_str,
    _safe_int,
    _is_yes,
    _decode,
    _col_yes_or_none,
    # Lookup dicts
    MARITAL_STATUS_CODES,
    EDUCATION_LEVEL_CODES,
    EMPLOYMENT_CODES,
    # Reusable extractors
    extract_substance_use as _extract_substance_use,
    extract_suicide_indicators as _extract_suicide_indicators,
    extract_suicide_history as _extract_suicide_history,
)
from face_rlvr.profiles.common_instruments import ScoreInterpretation, interpret_score
from face_rlvr.profiles.glossary_loader import get_cohort_column_map
from face_rlvr.profiles.dr_instruments import (
    DR_INSTRUMENTS,
    DR_DEPRESSION_INSTRUMENTS,
    DR_MOOD_INSTRUMENTS,
    DR_GLOBAL_INSTRUMENTS,
    DR_FUNCTIONING_INSTRUMENTS,
    DR_ANXIETY_INSTRUMENTS,
    DR_SLEEP_INSTRUMENTS,
    DR_ADHERENCE_INSTRUMENTS,
    DR_TRAUMA_INSTRUMENTS,
    DR_SUBSTANCE_INSTRUMENTS,
    DR_SELF_ESTEEM_INSTRUMENTS,
    DR_PERSONALITY_INSTRUMENTS,
    DR_SCREENING_INSTRUMENTS,
    DR_IMPULSIVITY_INSTRUMENTS,
)


# ─── DR-specific dataclasses ────────────────────────────────────────────────


def _load_dr_codes() -> dict[str, str]:
    from face_rlvr.profiles.glossary_loader import get_cohort_categorical_codes
    return get_cohort_categorical_codes("dr")["RESISTANCE_LEVEL_CODES"]

RESISTANCE_LEVEL_CODES = _load_dr_codes()


@dataclass
class TreatmentResistance:
    """Treatment resistance staging for DR patients."""
    is_resistant: bool | None = None  # epi_resist (0=no, 1=partial, 2=resistant)
    resistance_level: str | None = None  # decoded from epi_resist
    current_episode_number: int | None = None  # epi_num_epi
    current_episode_duration_months: float | None = None  # epi_duree
    antidepressant_response: str | None = None  # epi_vir_antid
    episode_treated: bool | None = None  # epi_traite
    age_first_treatment: int | None = None  # agetrt
    total_treatment_duration_months: float | None = None  # duree_glob_trt
    has_psychotic_features: bool | None = None  # epi_car_psycho
    achieved_complete_remission: bool | None = None  # epi_remi_comp
    sachs_score: float | None = None  # sachs_ (Sachs treatment resistance staging)


@dataclass
class DRTreatmentProfile:
    """DR-specific treatment profile."""
    lithium_level: float | None = None  # lithemie
    valproate_level: float | None = None  # valprate
    has_ect: bool = False  # proccur_ect
    medication_adherence: ScoreInterpretation | None = None


@dataclass
class DRSuicideAssessment:
    """DR uses cssrs01-cssrs05 (not css0101-css0106 like BP)."""
    wish_to_die: bool | None = None  # cssrs01
    nonspecific_ideation: bool | None = None  # cssrs02
    ideation_with_method: bool | None = None  # cssrs03
    ideation_with_intent: bool | None = None  # cssrs04
    ideation_with_plan: bool | None = None  # cssrs05
    highest_ideation_level: int | None = None  # derived: highest True among 1-5


@dataclass
class PsychiatricHistoryDR:
    """Simplified psychiatric history for DR (fewer fields than BP)."""
    age_first_episode: int | None = None
    illness_duration_years: float | None = None
    current_episode_severity: str | None = None


@dataclass
class DRPatientData:
    """Complete structured patient data extracted from a DR.csv row."""

    patient_id: str
    demographics: Demographics
    psychiatric_history: PsychiatricHistoryDR
    treatment_resistance: TreatmentResistance
    current_episode_criteria: CurrentEpisodeCriteria
    most_recent_episode: MostRecentEpisode

    # Interpreted instrument scores (grouped by clinical domain)
    depression_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    mood_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    global_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    functioning_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    anxiety_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    sleep_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    adherence_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    trauma_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    substance_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    self_esteem_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    personality_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    screening_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    # Newly added
    impulsivity_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)

    # Non-instrument data
    cognitive_profile: CognitiveProfile = field(default_factory=CognitiveProfile)
    biology: BiologicalPanel = field(default_factory=BiologicalPanel)
    treatments: DRTreatmentProfile = field(default_factory=DRTreatmentProfile)
    substance_use: SubstanceUse = field(default_factory=SubstanceUse)
    family_history: FamilyHistory = field(default_factory=FamilyHistory)
    hospitalization: HospitalizationHistory = field(default_factory=HospitalizationHistory)
    somatic_comorbidities: list[str] = field(default_factory=list)
    psychiatric_comorbidities: list[str] = field(default_factory=list)
    suicide_indicators: dict[str, Any] = field(default_factory=dict)
    suicide_history: SuicideHistory = field(default_factory=SuicideHistory)
    cssrs_assessment: DRSuicideAssessment = field(default_factory=DRSuicideAssessment)
    # Episode type counts
    episode_counts: dict[str, int | None] = field(default_factory=dict)
    # Birth/neonatal
    birth_data: dict[str, Any] = field(default_factory=dict)
    # Extended somatic history (additional flags)
    extended_somatic_history: list[str] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION FUNCTION
# ═════════════════════════════════════════════════════════════════════════════


def extract_dr_patient(row: pd.Series, patient_id: str | None = None) -> DRPatientData:
    """Extract structured patient data from a DR.csv row.

    All CSV column names come from ``config/glossary/dr/column_map.yaml``.
    """
    cm = get_cohort_column_map("dr")
    pid = patient_id or _safe_str(row.get(cm.patient_id_column)) or "unknown"

    return DRPatientData(
        patient_id=pid,
        demographics=_extract_demographics(row, cm),
        psychiatric_history=_extract_dr_psych_history(row, cm),
        treatment_resistance=_extract_treatment_resistance(row, cm),
        current_episode_criteria=_extract_current_episode_criteria(row, cm),
        most_recent_episode=_extract_most_recent_episode(row, cm),
        depression_scores=_extract_instrument_scores(row, DR_DEPRESSION_INSTRUMENTS),
        mood_scores=_extract_instrument_scores(row, DR_MOOD_INSTRUMENTS),
        global_scores=_extract_instrument_scores(row, DR_GLOBAL_INSTRUMENTS),
        functioning_scores=_extract_instrument_scores(row, DR_FUNCTIONING_INSTRUMENTS),
        anxiety_scores=_extract_instrument_scores(row, DR_ANXIETY_INSTRUMENTS),
        sleep_scores=_extract_instrument_scores(row, DR_SLEEP_INSTRUMENTS),
        adherence_scores=_extract_instrument_scores(row, DR_ADHERENCE_INSTRUMENTS),
        trauma_scores=_extract_instrument_scores(row, DR_TRAUMA_INSTRUMENTS),
        substance_scores=_extract_instrument_scores(row, DR_SUBSTANCE_INSTRUMENTS),
        self_esteem_scores=_extract_instrument_scores(row, DR_SELF_ESTEEM_INSTRUMENTS),
        personality_scores=_extract_instrument_scores(row, DR_PERSONALITY_INSTRUMENTS),
        screening_scores=_extract_instrument_scores(row, DR_SCREENING_INSTRUMENTS),
        impulsivity_scores=_extract_instrument_scores(row, DR_IMPULSIVITY_INSTRUMENTS),
        cognitive_profile=_extract_cognitive_profile(row, cm),
        biology=_extract_dr_biology(row, cm),
        treatments=_extract_dr_treatments(row, cm),
        substance_use=_extract_substance_use(row, cm),
        family_history=_extract_family_history(row, cm),
        hospitalization=_extract_dr_hospitalization(row, cm),
        somatic_comorbidities=_extract_dr_somatic_comorbidities(row, cm),
        psychiatric_comorbidities=_extract_dr_psychiatric_comorbidities(row, cm),
        suicide_indicators=_extract_suicide_indicators(row, cm),
        suicide_history=_extract_suicide_history(row, cm),
        cssrs_assessment=_extract_dr_cssrs(row, cm),
        episode_counts=_extract_episode_counts(row, cm),
        birth_data=_extract_birth_data(row, cm),
        extended_somatic_history=_extract_extended_somatic_history(row, cm),
    )


# ─── Sub-extractors ──────────────────────────────────────────────────────────


def _extract_demographics(row: pd.Series, cm) -> Demographics:
    d = cm.demographics
    sex_raw = _safe_str(row.get(d.sex))
    sex = None
    sex_label_fr = None
    if sex_raw:
        s = sex_raw.lower()
        if s in ("m", "masculin", "1", "male"):
            sex, sex_label_fr = "M", "homme"
        elif s in ("f", "feminin", "feminin", "2", "female"):
            sex, sex_label_fr = "F", "femme"

    return Demographics(
        age=_safe_int(row.get(d.age)),
        sex=sex,
        sex_label_fr=sex_label_fr,
        site_id=_safe_str(row.get(d.site_id)),
        arm=_safe_str(row.get(d.arm)),
        marital_status=_decode(row.get(d.marital_status), MARITAL_STATUS_CODES) if d.marital_status else None,
        education_level=_decode(row.get(d.education_level), EDUCATION_LEVEL_CODES),
        employment=_decode(row.get(d.employment), EMPLOYMENT_CODES) if d.employment else None,
    )


def _extract_instrument_scores(
    row: pd.Series, instrument_names: list[str],
) -> dict[str, ScoreInterpretation]:
    results = {}
    for name in instrument_names:
        inst = DR_INSTRUMENTS.get(name)
        if inst is None:
            continue
        raw = _safe_float(row.get(inst.total_column))
        subscales: dict[str, float | None] = {}
        for sub_name, sub_col in inst.subscale_columns.items():
            subscales[sub_name] = _safe_float(row.get(sub_col))
        results[name] = interpret_score(inst, raw, subscales)
    return results


def _extract_dr_psych_history(row: pd.Series, cm) -> PsychiatricHistoryDR:
    ph = cm.psychiatric_history.model_dump() if cm.psychiatric_history else {}
    return PsychiatricHistoryDR(
        age_first_episode=_safe_int(row.get(ph.get("age_first_episode"))) if ph.get("age_first_episode") else None,
        illness_duration_years=_safe_float(row.get(ph.get("illness_duration_years"))) if ph.get("illness_duration_years") else None,
        current_episode_severity=_safe_str(row.get(ph.get("current_episode_severity"))) if ph.get("current_episode_severity") else None,
    )


def _extract_treatment_resistance(row: pd.Series, cm) -> TreatmentResistance:
    tr = cm.model_extra.get("treatment_resistance", {}) if cm.model_extra else {}
    resist_col = tr.get("resistance_level")
    resist_raw = _safe_str(row.get(resist_col)) if resist_col else None
    is_resistant = None
    resistance_level = None
    if resist_raw is not None:
        resistance_level = _decode(row.get(resist_col), RESISTANCE_LEVEL_CODES)
        try:
            val = int(float(resist_raw))
            is_resistant = val >= 1
        except (ValueError, TypeError):
            pass

    return TreatmentResistance(
        is_resistant=is_resistant,
        resistance_level=resistance_level,
        current_episode_number=_safe_int(row.get(tr.get("current_episode_number"))) if tr.get("current_episode_number") else None,
        current_episode_duration_months=_safe_float(row.get(tr.get("current_episode_duration_months"))) if tr.get("current_episode_duration_months") else None,
        antidepressant_response=_safe_str(row.get(tr.get("antidepressant_response"))) if tr.get("antidepressant_response") else None,
        episode_treated=_col_yes_or_none(row, tr.get("episode_treated")),
        age_first_treatment=_safe_int(row.get(tr.get("age_first_treatment"))) if tr.get("age_first_treatment") else None,
        total_treatment_duration_months=_safe_float(row.get(tr.get("total_treatment_duration_months"))) if tr.get("total_treatment_duration_months") else None,
        has_psychotic_features=_col_yes_or_none(row, tr.get("has_psychotic_features")),
        achieved_complete_remission=_col_yes_or_none(row, tr.get("achieved_complete_remission")),
        sachs_score=_safe_float(row.get(tr.get("sachs_score"))) if tr.get("sachs_score") else None,
    )


def _extract_dr_cssrs(row: pd.Series, cm) -> DRSuicideAssessment:
    """Extract C-SSRS binary items from the column map."""
    cssrs = cm.model_extra.get("cssrs", {}) if cm.model_extra else {}
    prefix = cssrs.get("col_prefix", "cssrs0")
    item_numbers = cssrs.get("items", [1, 2, 3, 4, 5])

    items: dict[int, bool | None] = {}
    for i in item_numbers:
        col = f"{prefix}{i}"
        raw = _safe_str(row.get(col))
        items[i] = _is_yes(raw) if raw else None

    highest = None
    for level in range(max(item_numbers), 0, -1):
        if items.get(level) is True:
            highest = level
            break

    return DRSuicideAssessment(
        wish_to_die=items.get(1),
        nonspecific_ideation=items.get(2),
        ideation_with_method=items.get(3),
        ideation_with_intent=items.get(4),
        ideation_with_plan=items.get(5),
        highest_ideation_level=highest,
    )


def _extract_current_episode_criteria(row: pd.Series, cm) -> CurrentEpisodeCriteria:
    ce = cm.current_episode_criteria
    if ce is None:
        return CurrentEpisodeCriteria()
    kwargs: dict[str, Any] = {}
    for field_name, col in ce.depressive_items.items():
        kwargs[field_name] = _col_yes_or_none(row, col)
    for field_name, col in ce.manic_items.items():
        kwargs[field_name] = _col_yes_or_none(row, col)
    if ce.depressive_total:
        kwargs["depressive_symptom_count"] = _safe_int(row.get(ce.depressive_total))
    if ce.manic_total:
        kwargs["manic_symptom_count"] = _safe_int(row.get(ce.manic_total))
    return CurrentEpisodeCriteria(**kwargs)


def _extract_most_recent_episode(row: pd.Series, cm) -> MostRecentEpisode:
    mre = cm.most_recent_episode
    if mre is None:
        return MostRecentEpisode()
    return MostRecentEpisode(
        episode_type=_safe_str(row.get(mre.episode_type)) if mre.episode_type else None,
        severity=_safe_str(row.get(mre.severity)) if mre.severity else None,
        start_date=_safe_str(row.get(mre.start_date)) if mre.start_date else None,
        end_date=_safe_str(row.get(mre.end_date)) if mre.end_date else None,
        chronicity=_safe_str(row.get(mre.chronicity)) if mre.chronicity else None,
        postpartum=_col_yes_or_none(row, mre.postpartum),
    )


def _extract_cognitive_profile(row: pd.Series, cm) -> CognitiveProfile:
    cp = cm.cognitive_profile.model_dump() if cm.cognitive_profile else {}

    def f(key: str):
        col = cp.get(key)
        return _safe_float(row.get(col)) if col else None

    tmt_a = f("tmt_a_seconds") or f("tmt_a_seconds_alt")
    tmt_b = f("tmt_b_seconds") or f("tmt_b_seconds_alt")
    tmt_ba = (tmt_b - tmt_a) if (tmt_a is not None and tmt_b is not None) else None
    tmt_ratio = (tmt_b / tmt_a) if (tmt_a and tmt_a > 0 and tmt_b is not None) else None

    return CognitiveProfile(
        tmt_a_seconds=tmt_a,
        tmt_b_seconds=tmt_b,
        tmt_b_minus_a=round(tmt_ba, 1) if tmt_ba is not None else None,
        tmt_ratio_ba=round(tmt_ratio, 2) if tmt_ratio is not None else None,
        stroop_interference=f("stroop_interference"),
        cvlt_total_learning=f("cvlt_total_learning"),
        cvlt_long_delay_free=f("cvlt_long_delay_free"),
        cvlt_recognition=f("cvlt_recognition"),
        phonemic_fluency=f("phonemic_fluency"),
        wais_similarities_std=f("wais_similarities_std"),
        wais_vocabulary_std=f("wais_vocabulary_std"),
        wais_working_memory_std=f("wais_working_memory_std"),
    )


def _extract_dr_biology(row: pd.Series, cm) -> BiologicalPanel:
    """DR lab values + vitals + ECG, all columns from column_map.yaml."""
    from face_rlvr.profiles.glossary_loader import get_cohort_lab_ranges

    values = []
    for lab in get_cohort_lab_ranges("dr"):
        val = _safe_float(row.get(lab.csv_col))
        if val is None:
            continue
        nrange = tuple(lab.normal_range) if lab.normal_range else None
        is_abn = False
        abn = None
        if nrange:
            lo, hi = nrange
            if val < lo:
                is_abn, abn = True, "low"
            elif val > hi:
                is_abn, abn = True, "high"
        values.append(LabValue(
            name=lab.name, name_fr=lab.name_fr, value=val, unit=lab.unit,
            normal_range=nrange, is_abnormal=is_abn, abnormality=abn,
        ))

    vitals_map = cm.vitals.model_dump() if cm.vitals else {}
    vitals = {k: _safe_float(row.get(col)) for k, col in vitals_map.items()}
    vitals = {k: v for k, v in vitals.items() if v is not None}

    ecg_map = cm.ecg.model_dump() if cm.ecg else {}
    ecg = {k: _safe_float(row.get(col)) for k, col in ecg_map.items()}
    ecg = {k: v for k, v in ecg.items() if v is not None}

    return BiologicalPanel(values=values, vitals=vitals, ecg=ecg)


def _extract_dr_treatments(row: pd.Series, cm) -> DRTreatmentProfile:
    t = cm.treatments.model_dump() if cm.treatments else {}
    lithium = _safe_float(row.get(t.get("lithium_level"))) if t.get("lithium_level") else None
    valproate = _safe_float(row.get(t.get("valproate_level"))) if t.get("valproate_level") else None
    has_ect = _is_yes(row.get(t.get("ect_flag"))) if t.get("ect_flag") else False

    mars_interp = None
    adherence_key = t.get("adherence_instrument_key")
    if adherence_key:
        inst = DR_INSTRUMENTS.get(adherence_key)
        if inst is not None:
            mars_interp = interpret_score(inst, _safe_float(row.get(inst.total_column)))

    return DRTreatmentProfile(
        lithium_level=lithium,
        valproate_level=valproate,
        has_ect=has_ect,
        medication_adherence=mars_interp,
    )


def _extract_family_history(row: pd.Series, cm) -> FamilyHistory:
    """Same pedigree logic as BP/SZ, driven by cm.family_history."""
    fh = cm.family_history
    if fh is None:
        return FamilyHistory()

    mat_trouble = _safe_str(row.get(fh.maternal_psychiatric)) if fh.maternal_psychiatric else None
    pat_trouble = _safe_str(row.get(fh.paternal_psychiatric)) if fh.paternal_psychiatric else None
    keyword = fh.bipolar_keyword.lower()
    family_bp = any(t and keyword in t.lower() for t in (mat_trouble, pat_trouble))

    sfx = fh.relative_suffixes
    relatives: list[RelativeHistory] = []
    for item in fh.relatives:
        prefix = item["key"]
        trouble = _safe_str(row.get(f"{prefix}{sfx.get('psychiatric_disorder', '_trouble')}"))
        structure = _safe_str(row.get(f"{prefix}{sfx.get('structure', '_structure')}"))
        if not trouble and not structure:
            continue
        relatives.append(RelativeHistory(
            relation=prefix,
            relation_fr=item["label_fr"],
            psychiatric_disorder=trouble,
            suicide=_is_yes(row.get(f"{prefix}{sfx.get('suicide', '_suicide')}")),
            substance_use=_is_yes(row.get(f"{prefix}{sfx.get('substance_use', '_substance')}")),
            anxiety=_is_yes(row.get(f"{prefix}{sfx.get('anxiety', '_anx')}")),
            dementia=_is_yes(row.get(f"{prefix}{sfx.get('dementia', '_dem')}")),
        ))
        if trouble and keyword in trouble.lower():
            family_bp = True

    for item in fh.siblings:
        prefix = item["key"]
        trouble = _safe_str(row.get(f"{prefix}{sfx.get('psychiatric_disorder', '_trouble')}"))
        if not trouble:
            continue
        relatives.append(RelativeHistory(
            relation=prefix,
            relation_fr=item["label_fr"],
            psychiatric_disorder=trouble,
            suicide=_is_yes(row.get(f"{prefix}{sfx.get('suicide', '_suicide')}")),
            substance_use=_is_yes(row.get(f"{prefix}{sfx.get('substance_use', '_substance')}")),
        ))
        if keyword in trouble.lower():
            family_bp = True

    def _count(col):
        return _safe_int(row.get(col)) if col else None

    n_brothers = _count(fh.brothers_count_col)
    n_sisters = _count(fh.sisters_count_col)
    n_siblings = ((n_brothers or 0) + (n_sisters or 0)) if (n_brothers is not None or n_sisters is not None) else None
    brothers_aff = _count(fh.brothers_affected_col)
    sisters_aff = _count(fh.sisters_affected_col)
    n_siblings_aff = ((brothers_aff or 0) + (sisters_aff or 0)) if (brothers_aff is not None or sisters_aff is not None) else None
    n_sons = _count(fh.sons_count_col)
    n_daughters = _count(fh.daughters_count_col)
    n_children = ((n_sons or 0) + (n_daughters or 0)) if (n_sons is not None or n_daughters is not None) else None
    sons_aff = _count(fh.sons_affected_col)
    daughters_aff = _count(fh.daughters_affected_col)
    n_children_aff = ((sons_aff or 0) + (daughters_aff or 0)) if (sons_aff is not None or daughters_aff is not None) else None

    return FamilyHistory(
        maternal_psychiatric=mat_trouble,
        paternal_psychiatric=pat_trouble,
        maternal_substance=_is_yes(row.get(fh.maternal_substance)) if fh.maternal_substance else False,
        paternal_substance=_is_yes(row.get(fh.paternal_substance)) if fh.paternal_substance else False,
        maternal_suicide=_is_yes(row.get(fh.maternal_suicide)) if fh.maternal_suicide else False,
        paternal_suicide=_is_yes(row.get(fh.paternal_suicide)) if fh.paternal_suicide else False,
        family_bipolar=family_bp,
        relatives=relatives,
        n_siblings=n_siblings,
        n_siblings_affected=n_siblings_aff,
        n_children=n_children,
        n_children_affected=n_children_aff,
    )


def _extract_dr_hospitalization(row: pd.Series, cm) -> HospitalizationHistory:
    h = cm.hospitalization
    if h is None:
        return HospitalizationHistory()
    n_lt = _safe_int(row.get(h.n_lifetime)) if h.n_lifetime else None
    return HospitalizationHistory(
        ever_hospitalized=(n_lt is not None and n_lt > 0),
        n_hospitalizations_lifetime=n_lt,
        n_hospitalizations_last_year=_safe_int(row.get(h.n_last_year)) if h.n_last_year else None,
        duration_last_hospitalization=_safe_float(row.get(h.duration_last)) if h.duration_last else None,
        work_absences=_is_yes(row.get(h.work_absences_flag)) if h.work_absences_flag else False,
        n_work_absences=_safe_int(row.get(h.n_work_absences)) if h.n_work_absences else None,
    )


def _extract_dr_somatic_comorbidities(row: pd.Series, cm) -> list[str]:
    if cm.comorbidities is None:
        return []
    return [
        item.label_fr for item in cm.comorbidities.somatic
        if item.col and _is_yes(row.get(item.col))
    ]


def _extract_dr_psychiatric_comorbidities(row: pd.Series, cm) -> list[str]:
    c = cm.comorbidities
    if c is None:
        return []
    comorbidities: list[str] = []
    for item in c.psychiatric:
        if item.col and _is_yes(row.get(item.col)):
            comorbidities.append(item.label_fr)
    if c.general_anxiety_flag and _is_yes(row.get(c.general_anxiety_flag)) \
            and not any("anxieux" in x.lower() for x in comorbidities):
        comorbidities.append(c.general_anxiety_label_fr)
    if c.substance_use_flag and _is_yes(row.get(c.substance_use_flag)):
        comorbidities.append(c.substance_use_label_fr)
    return comorbidities


def _extract_episode_counts(row: pd.Series, cm) -> dict[str, int | None]:
    """Episode type counts from the cohort-specific column map."""
    ec = cm.model_extra.get("episode_counts", {}) if cm.model_extra else {}
    return {k: _safe_int(row.get(col)) for k, col in ec.items()}


def _extract_birth_data(row: pd.Series, cm) -> dict[str, Any]:
    """Birth/neonatal data (limited in DR)."""
    bd = cm.model_extra.get("birth_data", {}) if cm.model_extra else {}
    data: dict[str, Any] = {}
    for k, col in bd.items():
        val = _safe_float(row.get(col))
        if val is not None:
            data[k] = val
    return data


def _extract_extended_somatic_history(row: pd.Series, cm) -> list[str]:
    """Additional somatic flags beyond the standard comorbidity list."""
    items = cm.model_extra.get("extended_somatic_history", []) if cm.model_extra else []
    return [item["label_fr"] for item in items if _is_yes(row.get(item["col"]))]
