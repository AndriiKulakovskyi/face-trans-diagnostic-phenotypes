"""Structured data extraction from SZ.csv rows.

Extracts schizophrenia-specific data: PANSS, Calgary, PSP, AIMS, BARS,
SUMD, psychotic symptoms, plus shared clinical data (demographics, biology,
treatments, family history, suicide).

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
    AdditionalNeuropsych,
    NonPharmTreatment,
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
from face_rlvr.profiles.sz_instruments import (
    SZ_INSTRUMENTS,
    SZ_PSYCHOSIS_INSTRUMENTS,
    SZ_DEPRESSION_INSTRUMENTS,
    SZ_GLOBAL_INSTRUMENTS,
    SZ_FUNCTIONING_INSTRUMENTS,
    SZ_MOVEMENT_INSTRUMENTS,
    SZ_MOOD_INSTRUMENTS,
    SZ_SLEEP_INSTRUMENTS,
    SZ_ADHERENCE_INSTRUMENTS,
    SZ_TRAUMA_INSTRUMENTS,
    SZ_SUBSTANCE_INSTRUMENTS,
    SZ_SCREENING_INSTRUMENTS,
)


# ─── SZ-specific dataclasses ─────────────────────────────────────────────────


@dataclass
class PsychoticHistory:
    """Schizophrenia-specific psychiatric history."""
    has_psychotic_disorder: bool = False
    n_psychotic_episodes: int | None = None
    age_onset_sz: int | None = None
    age_onset_mood: int | None = None
    symptom_evolution_mode: str | None = None
    n_psychotic_episodes_lifetime: str | None = None  # evnum_tbpsy_lt (stored as str in CSV)
    n_suicide_events_last_year: str | None = None


@dataclass
class PsychoticSymptoms:
    """Current psychotic symptom phenomenology from ceoccur* columns."""
    # Positive symptoms
    hallucinations_auditory: bool = False
    hallucinations_visual: bool = False
    hallucinations_somatic: bool = False
    hallucinations_intrapsychic: bool = False
    delusions_persecution: bool = False
    delusions_reference: bool = False
    delusions_grandiosity: bool = False
    delusions_erotomanic: bool = False
    delusions_jealousy: bool = False
    delusions_mystical: bool = False
    delusions_somatic: bool = False
    thought_control: bool = False
    thought_broadcasting: bool = False
    # Disorganization
    disorganized_speech: bool = False
    disorganized_behavior: bool = False
    bizarre_behavior: bool = False
    # Negative symptoms
    avolition: bool = False
    alogia: bool = False
    # Catatonia
    catatonia: bool = False

    @property
    def positive_symptoms(self) -> list[str]:
        symptoms = []
        if self.hallucinations_auditory:
            symptoms.append("hallucinations auditives")
        if self.hallucinations_visual:
            symptoms.append("hallucinations visuelles")
        if self.hallucinations_somatic:
            symptoms.append("hallucinations cénesthésiques")
        if self.hallucinations_intrapsychic:
            symptoms.append("hallucinations intrapsychiques")
        if self.delusions_persecution:
            symptoms.append("idées de persécution")
        if self.delusions_reference:
            symptoms.append("idées de référence")
        if self.delusions_grandiosity:
            symptoms.append("idées de grandeur")
        if self.delusions_erotomanic:
            symptoms.append("érotomanie")
        if self.delusions_jealousy:
            symptoms.append("jalousie délirante")
        if self.delusions_mystical:
            symptoms.append("thématique mystique")
        if self.delusions_somatic:
            symptoms.append("thématique somatique")
        if self.thought_control:
            symptoms.append("syndrome d'influence")
        if self.thought_broadcasting:
            symptoms.append("diffusion de la pensée")
        return symptoms

    @property
    def negative_symptoms(self) -> list[str]:
        symptoms = []
        if self.avolition:
            symptoms.append("avolition")
        if self.alogia:
            symptoms.append("alogie")
        return symptoms

    @property
    def disorganization_symptoms(self) -> list[str]:
        symptoms = []
        if self.disorganized_speech:
            symptoms.append("discours désorganisé")
        if self.disorganized_behavior:
            symptoms.append("comportement désorganisé")
        if self.bizarre_behavior:
            symptoms.append("comportement bizarre")
        return symptoms


@dataclass
class InsightAssessment:
    """SUMD — Scale to Assess Unawareness of Mental Disorder."""
    awareness_of_illness: float | None = None  # sumd01
    awareness_of_medication_effect: float | None = None  # sumd02
    awareness_of_social_consequences: float | None = None  # sumd03
    awareness_of_hallucinations: float | None = None  # sumd04
    awareness_of_delusions: float | None = None  # sumd05
    awareness_of_thought_disorder: float | None = None  # sumd06
    awareness_of_flat_affect: float | None = None  # sumd07
    awareness_of_anhedonia: float | None = None  # sumd08
    awareness_of_asociality: float | None = None  # sumd09


@dataclass
class SZTreatmentProfile:
    """SZ-specific treatment profile."""
    on_clozapine: bool = False
    clozapine_plasma: float | None = None
    n_antipsychotics: int | None = None
    n_anticholinergics: int | None = None
    n_anxiolytics: int | None = None
    n_hypnotics: int | None = None
    n_mood_stabilizers: int | None = None
    n_antidepressants: int | None = None
    medication_adherence: ScoreInterpretation | None = None


@dataclass
class SZPatientData:
    """Complete structured patient data extracted from an SZ.csv row."""

    patient_id: str
    demographics: Demographics
    psychotic_history: PsychoticHistory
    psychotic_symptoms: PsychoticSymptoms
    insight: InsightAssessment

    # Interpreted instrument scores
    psychosis_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    depression_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    global_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    functioning_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    movement_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    mood_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    sleep_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    adherence_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    trauma_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    substance_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    screening_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)

    # Non-instrument data
    cognitive_profile: CognitiveProfile = field(default_factory=CognitiveProfile)
    additional_neuropsych: AdditionalNeuropsych = field(default_factory=AdditionalNeuropsych)
    biology: BiologicalPanel = field(default_factory=BiologicalPanel)
    treatments: SZTreatmentProfile = field(default_factory=SZTreatmentProfile)
    substance_use: SubstanceUse = field(default_factory=SubstanceUse)
    family_history: FamilyHistory = field(default_factory=FamilyHistory)
    hospitalization: HospitalizationHistory = field(default_factory=HospitalizationHistory)
    somatic_comorbidities: list[str] = field(default_factory=list)
    psychiatric_comorbidities: list[str] = field(default_factory=list)
    suicide_indicators: dict[str, Any] = field(default_factory=dict)
    suicide_history: SuicideHistory = field(default_factory=SuicideHistory)

    # PANSS Wallwork 5-factor scores (pre-computed in SZ.csv)
    panss_wallwork_positive: float | None = None
    panss_wallwork_negative: float | None = None
    panss_wallwork_disorganized: float | None = None
    panss_wallwork_excited: float | None = None
    panss_wallwork_depressed: float | None = None


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION FUNCTION
# ═════════════════════════════════════════════════════════════════════════════


def extract_sz_patient(row: pd.Series, patient_id: str | None = None) -> SZPatientData:
    """Extract structured patient data from an SZ.csv row.

    All CSV column names come from ``config/glossary/sz/column_map.yaml``.
    """
    cm = get_cohort_column_map("sz")
    pid = patient_id or _safe_str(row.get(cm.patient_id_column)) or "unknown"

    return SZPatientData(
        patient_id=pid,
        demographics=_extract_demographics(row, cm),
        psychotic_history=_extract_psychotic_history(row, cm),
        psychotic_symptoms=_extract_psychotic_symptoms(row, cm),
        insight=_extract_insight(row, cm),
        psychosis_scores=_extract_instrument_scores(row, SZ_PSYCHOSIS_INSTRUMENTS),
        depression_scores=_extract_instrument_scores(row, SZ_DEPRESSION_INSTRUMENTS),
        global_scores=_extract_instrument_scores(row, SZ_GLOBAL_INSTRUMENTS),
        functioning_scores=_extract_instrument_scores(row, SZ_FUNCTIONING_INSTRUMENTS),
        movement_scores=_extract_instrument_scores(row, SZ_MOVEMENT_INSTRUMENTS),
        mood_scores=_extract_instrument_scores(row, SZ_MOOD_INSTRUMENTS),
        sleep_scores=_extract_instrument_scores(row, SZ_SLEEP_INSTRUMENTS),
        adherence_scores=_extract_instrument_scores(row, SZ_ADHERENCE_INSTRUMENTS),
        trauma_scores=_extract_instrument_scores(row, SZ_TRAUMA_INSTRUMENTS),
        substance_scores=_extract_instrument_scores(row, SZ_SUBSTANCE_INSTRUMENTS),
        screening_scores=_extract_instrument_scores(row, SZ_SCREENING_INSTRUMENTS),
        cognitive_profile=_extract_cognitive_profile(row, cm),
        additional_neuropsych=_extract_additional_neuropsych(row, cm),
        biology=_extract_sz_biology(row, cm),
        treatments=_extract_sz_treatments(row, cm),
        substance_use=_extract_substance_use(row, cm),
        family_history=_extract_family_history(row, cm),
        hospitalization=_extract_sz_hospitalization(row, cm),
        somatic_comorbidities=_extract_sz_somatic_comorbidities(row, cm),
        psychiatric_comorbidities=_extract_sz_psychiatric_comorbidities(row, cm),
        suicide_indicators=_extract_suicide_indicators(row, cm),
        suicide_history=_extract_suicide_history(row, cm),
        # PANSS Wallwork 5-factor scores (pre-computed columns in SZ.csv)
        panss_wallwork_positive=_safe_float(row.get("pansspow")),
        panss_wallwork_negative=_safe_float(row.get("panssnew")),
        panss_wallwork_disorganized=_safe_float(row.get("panssdiw")),
        panss_wallwork_excited=_safe_float(row.get("panssexw")),
        panss_wallwork_depressed=_safe_float(row.get("panssdew")),
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
        elif s in ("f", "feminin", "féminin", "2", "female"):
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
        inst = SZ_INSTRUMENTS.get(name)
        if inst is None:
            continue
        raw = _safe_float(row.get(inst.total_column))
        subscales: dict[str, float | None] = {}
        for sub_name, sub_col in inst.subscale_columns.items():
            subscales[sub_name] = _safe_float(row.get(sub_col))
        results[name] = interpret_score(inst, raw, subscales)
    return results


def _extract_psychotic_history(row: pd.Series, cm) -> PsychoticHistory:
    ph = cm.model_extra.get("psychotic_history", {}) if cm.model_extra else {}
    psychotic_col = ph.get("has_psychotic_disorder")
    psychotic = _safe_int(row.get(psychotic_col)) if psychotic_col else None
    return PsychoticHistory(
        has_psychotic_disorder=psychotic == 1 if psychotic is not None else False,
        n_psychotic_episodes=_safe_int(row.get(ph.get("n_psychotic_episodes"))) if ph.get("n_psychotic_episodes") else None,
        age_onset_sz=_safe_int(row.get(ph.get("age_onset_sz"))) if ph.get("age_onset_sz") else None,
        age_onset_mood=_safe_int(row.get(ph.get("age_onset_mood"))) if ph.get("age_onset_mood") else None,
        symptom_evolution_mode=_safe_str(row.get(ph.get("symptom_evolution_mode"))) if ph.get("symptom_evolution_mode") else None,
        n_psychotic_episodes_lifetime=_safe_str(row.get(ph.get("n_psychotic_episodes_lifetime"))) if ph.get("n_psychotic_episodes_lifetime") else None,
        n_suicide_events_last_year=_safe_str(row.get(ph.get("n_suicide_events_last_year"))) if ph.get("n_suicide_events_last_year") else None,
    )


def _extract_psychotic_symptoms(row: pd.Series, cm) -> PsychoticSymptoms:
    ps = cm.model_extra.get("psychotic_symptoms", {}) if cm.model_extra else {}
    return PsychoticSymptoms(
        **{field_name: _is_yes(row.get(col)) for field_name, col in ps.items()}
    )


def _extract_insight(row: pd.Series, cm) -> InsightAssessment:
    ins = cm.model_extra.get("insight", {}) if cm.model_extra else {}
    return InsightAssessment(
        **{field_name: _safe_float(row.get(col)) for field_name, col in ins.items()}
    )


def _extract_cognitive_profile(row: pd.Series, cm) -> CognitiveProfile:
    cp = cm.cognitive_profile.model_dump() if cm.cognitive_profile else {}

    def f(key: str):
        col = cp.get(key)
        return _safe_float(row.get(col)) if col else None

    # SZ cognitive columns have primary + alt (tmta01_ vs tmt0101, similstd_wais4 vs _wais3)
    tmt_a = f("tmt_a_seconds") or f("tmt_a_seconds_alt")
    tmt_b = f("tmt_b_seconds") or f("tmt_b_seconds_alt")
    tmt_ba = (tmt_b - tmt_a) if (tmt_a is not None and tmt_b is not None) else None
    tmt_ratio = (tmt_b / tmt_a) if (tmt_a and tmt_a > 0 and tmt_b is not None) else None

    wais_sim_std = f("wais_similarities_std") or f("wais_similarities_std_alt")

    cobra_interp = None
    cobra_key = cp.get("cobra_instrument_key")
    if cobra_key:
        cobra_inst = SZ_INSTRUMENTS.get(cobra_key)
        if cobra_inst is not None:
            cobra_interp = interpret_score(cobra_inst, _safe_float(row.get(cobra_inst.total_column)))

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
        wais_similarities_std=wais_sim_std,
        wais_vocabulary_std=f("wais_vocabulary_std"),
        cobra_interpretation=cobra_interp,
    )


def _extract_additional_neuropsych(row: pd.Series, cm) -> AdditionalNeuropsych:
    an = cm.additional_neuropsych.model_dump() if cm.additional_neuropsych else {}
    def f(key: str):
        col = an.get(key)
        return _safe_float(row.get(col)) if col else None
    return AdditionalNeuropsych(
        matrices_raw=f("matrices_raw"),
        matrices_std=f("matrices_std"),
        code_raw=f("code_raw"),
        code_std=f("code_std"),
        digit_span_forward_total=f("digit_span_forward_total"),
        digit_span_forward_std=f("digit_span_forward_std"),
        digit_span_backward_total=f("digit_span_backward_total"),
        digit_span_backward_std=f("digit_span_backward_std"),
        digit_span_total_raw=f("digit_span_total_raw"),
        digit_span_total_std=f("digit_span_total_std"),
    )


def _extract_sz_biology(row: pd.Series, cm) -> BiologicalPanel:
    """SZ lab values + vitals + ECG (columns all from column_map.yaml)."""
    from face_rlvr.profiles.glossary_loader import get_cohort_lab_ranges

    values = []
    for lab in get_cohort_lab_ranges("sz"):
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


def _extract_sz_treatments(row: pd.Series, cm) -> SZTreatmentProfile:
    t = cm.treatments.model_dump() if cm.treatments else {}

    cloz_plasma = _safe_float(row.get(t.get("clozapine_plasma"))) if t.get("clozapine_plasma") else None
    cloz_flag = t.get("clozapine_flag")
    on_cloz = (_is_yes(row.get(cloz_flag)) if cloz_flag else False) or (cloz_plasma is not None and cloz_plasma > 0)

    mars_interp = None
    adherence_key = t.get("adherence_instrument_key")
    if adherence_key:
        inst = SZ_INSTRUMENTS.get(adherence_key)
        if inst is not None:
            mars_interp = interpret_score(inst, _safe_float(row.get(inst.total_column)))

    def i(key: str):
        col = t.get(key)
        return _safe_int(row.get(col)) if col else None

    return SZTreatmentProfile(
        on_clozapine=on_cloz,
        clozapine_plasma=cloz_plasma if on_cloz else None,
        n_antipsychotics=i("n_antipsychotics"),
        n_anticholinergics=i("n_anticholinergics"),
        n_anxiolytics=i("n_anxiolytics"),
        n_hypnotics=i("n_hypnotics"),
        n_mood_stabilizers=i("n_mood_stabilizers"),
        n_antidepressants=i("n_antidepressants"),
        medication_adherence=mars_interp,
    )


def _extract_family_history(row: pd.Series, cm) -> FamilyHistory:
    """Same pedigree logic as BP, driven by cm.family_history."""
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


def _extract_sz_hospitalization(row: pd.Series, cm) -> HospitalizationHistory:
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
    )


def _extract_sz_somatic_comorbidities(row: pd.Series, cm) -> list[str]:
    if cm.comorbidities is None:
        return []
    return [
        item.label_fr for item in cm.comorbidities.somatic
        if item.col and _is_yes(row.get(item.col))
    ]


def _extract_sz_psychiatric_comorbidities(row: pd.Series, cm) -> list[str]:
    if cm.comorbidities is None:
        return []
    return [
        item.label_fr for item in cm.comorbidities.psychiatric
        if item.col and _is_yes(row.get(item.col))
    ]
