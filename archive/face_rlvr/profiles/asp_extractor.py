"""Structured data extraction from ASP.csv (TSASDI) rows.

Extracts autism-specific data: diagnosis profile, developmental history,
repetitive behaviors, plus limited shared clinical data (demographics,
minimal biology, treatments, comorbidities).

Key differences from BP/SZ:
- Patient ID column: usubjid_identite (not usubjid_patients)
- No MADRS, YMRS, PANSS, FAST, STAI
- Now extracts: PSQI, ESS, MARS, Fagerstrom, PRISM (data exists in ASP.csv)
- No ISF/C-SSRS suicide instruments — uses BDI-II item 9 as partial mitigation
- No mere_trouble/pere_trouble family psychiatric history
- No maristat (uses empjob1/empjob2 instead of empjob)
- Very sparse data — most instruments <25% coverage
- Biology: only bmi, gluc, hdl, chol, trig

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
    # Helpers
    _safe_float,
    _safe_str,
    _safe_int,
    _is_yes,
    _decode,
    # Lookup dicts
    EDUCATION_LEVEL_CODES,
    EMPLOYMENT_CODES,
    # Clinical utilities
    compute_bmi_category,
    detect_metabolic_syndrome,
    compute_data_completeness,
)
from face_rlvr.profiles.common_instruments import ScoreInterpretation, interpret_score
from face_rlvr.profiles.glossary_loader import get_cohort_column_map
from face_rlvr.profiles.asp_instruments import (
    ASP_INSTRUMENTS,
    ASP_DEPRESSION_INSTRUMENTS,
    ASP_GLOBAL_INSTRUMENTS,
    ASP_FUNCTIONING_INSTRUMENTS,
    ASP_REPETITIVE_INSTRUMENTS,
    ASP_COGNITIVE_INSTRUMENTS,
    ASP_EXECUTIVE_INSTRUMENTS,
    ASP_AUTISM_SCREENING_INSTRUMENTS,
    ASP_ANXIETY_INSTRUMENTS,
    ASP_ADHD_INSTRUMENTS,
    ASP_TRAUMA_INSTRUMENTS,
    ASP_SLEEP_INSTRUMENTS,
    ASP_ADHERENCE_INSTRUMENTS,
    ASP_SUBSTANCE_INSTRUMENTS,
)


# ─── DSM type lookup ────────────────────────────────────────────────────────

def _load_asp_codes() -> dict[str, dict[str, str]]:
    from face_rlvr.profiles.glossary_loader import get_cohort_categorical_codes
    return get_cohort_categorical_codes("asp")

DSM_TYPE_CODES = _load_asp_codes()["DSM_TYPE_CODES"]


# ─── ASP-specific dataclasses ───────────────────────────────────────────────


@dataclass
class AutismDiagnosis:
    """Autism diagnostic profile from DSM-5, ADI-R, and ADOS."""

    dsm_type: float | None = None  # dsmtype (1=autism, 2=Asperger, etc.)
    dsm_type_label: str | None = None  # decoded French label
    dsm_domain1_met: bool | None = None  # dsmv1 (social communication)
    dsm_domain2_met: bool | None = None  # dsmv2 (restricted behaviors)
    adi_diagnostic: float | None = None  # adi_diag
    ados_exam: str | None = None  # adosexam
    # DSM-5 individual criteria
    dsm_criteria: dict[str, bool] = field(default_factory=dict)  # dsmaut01-dsmaut11


@dataclass
class DevelopmentalHistory:
    """Developmental milestones, neonatal history, and parental ages."""

    age_motor_milestones: float | None = None  # agemots (months)
    age_first_phrases: float | None = None  # agephras (months)
    mother_age: float | None = None  # agemere
    father_age: float | None = None  # agepere
    # Neonatal / perinatal (binary flags, 94-96% coverage)
    psychomotor_delay: bool | None = None  # retpsy
    language_delay: bool | None = None  # retlang
    fetal_distress: bool | None = None  # souffet
    neonatal_complications: bool | None = None  # neonat
    resuscitation: bool | None = None  # reanim
    pregnancy_pathology: bool | None = None  # pathgro
    fetal_pathology: bool | None = None  # pathfeta
    neonatal_illness: bool | None = None  # affecbb
    feeding_difficulties: bool | None = None  # tralim
    sleep_difficulties: bool | None = None  # trsomm
    twin: bool | None = None  # gemel
    seizures: bool | None = None  # crisconv
    birth_weight_g: float | None = None  # naispoid
    birth_height_cm: float | None = None  # naistail
    apgar_score: float | None = None  # apgar1
    # Language
    language_expression: str | None = None  # langx
    current_language: str | None = None  # langact


@dataclass
class LearningDisabilities:
    """Learning disabilities (70% coverage in ASP)."""

    dyslexia: bool | None = None  # neuro_dyslex
    dysorthography: bool | None = None  # neuro_dysor
    dyscalculia: bool | None = None  # neuro_dyscal
    dysphasia: bool | None = None  # neuro_dysph
    dyspraxia: bool | None = None  # neuro_dyspra
    speech_disorder: bool | None = None  # neuro_parol
    stuttering: bool | None = None  # neuro_begai


@dataclass
class MCDDProfile:
    """Multiple Complex Developmental Disorder criteria (63% coverage)."""

    criteria_met: list[int] = field(default_factory=list)  # which mcdd1-15 are 1
    total_criteria_met: int = 0
    total_criteria_assessed: int = 0


@dataclass
class MedicalAntecedents:
    """Medical history flags (97-99% coverage in ASP)."""

    cardiac: bool | None = None  # antcardi
    endocrine: bool | None = None  # antendo
    neurological: bool | None = None  # antneuro
    ent: bool | None = None  # antorl (ORL)
    pulmonary: bool | None = None  # antpulm
    rheumatological: bool | None = None  # antrhum
    hepatic: bool | None = None  # anthepa
    cancer: bool | None = None  # antcanc
    genetic_disorder: bool | None = None  # malgen
    other_condition: bool | None = None  # autrmal


@dataclass
class ASPClinicalStatus:
    """Current clinical and care status."""

    age_at_diagnosis_years: float | None = None  # adiagea
    age_at_diagnosis_months: float | None = None  # adiagem
    in_psychiatric_care: bool | None = None  # contpsy
    currently_hospitalized: bool | None = None  # hospicour
    currently_treated: bool | None = None  # trtcour
    has_insomnia: bool | None = None  # insom
    has_hypersomnia: bool | None = None  # hypers
    executive_function_impairment: bool | None = None  # fctexe
    social_cognition_impairment: bool | None = None  # cogsoc
    school_level: str | None = None  # nivmaxc (decoded)
    school_type: str | None = None  # scolass (decoded)


@dataclass
class PregnancyToxicology:
    """Pregnancy and toxicology data (94-97% coverage)."""

    toxicology_exposure: bool | None = None  # tox
    folic_acid_supplementation: bool | None = None  # gros_fol
    bleeding_during_pregnancy: bool | None = None  # saign
    infection_viral: bool | None = None  # infevir


@dataclass
class ASPTreatmentProfile:
    """ASP-specific treatment profile."""

    on_antidepressant: bool = False  # antidepc
    on_antipsychotic: bool = False  # antipsyc
    on_lamotrigine: bool = False  # lamotrig
    non_pharm_treatments: list[str] = field(default_factory=list)  # trt_non_parmaco_1 through _15


@dataclass
class ASPPatientData:
    """Complete structured patient data extracted from an ASP.csv row."""

    patient_id: str
    demographics: Demographics
    autism_diagnosis: AutismDiagnosis
    developmental_history: DevelopmentalHistory

    # Interpreted instrument scores (grouped by clinical domain)
    depression_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    global_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    functioning_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    repetitive_behavior_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    cognitive_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    executive_function_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    autism_screening_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    anxiety_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    adhd_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    trauma_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    # Newly added domains (data exists in ASP.csv)
    sleep_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    adherence_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    substance_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)

    # Non-instrument data
    substance_use: SubstanceUse = field(default_factory=SubstanceUse)
    bdi_item9_suicidal_thoughts: int | None = None  # BDI-II item 9 (0-3): partial suicide risk mitigation
    biology: BiologicalPanel = field(default_factory=BiologicalPanel)
    treatments: ASPTreatmentProfile = field(default_factory=ASPTreatmentProfile)
    somatic_comorbidities: list[str] = field(default_factory=list)
    psychiatric_comorbidities: list[str] = field(default_factory=list)
    learning_disabilities: LearningDisabilities = field(default_factory=LearningDisabilities)
    mcdd_profile: MCDDProfile = field(default_factory=MCDDProfile)
    medical_antecedents: MedicalAntecedents = field(default_factory=MedicalAntecedents)
    clinical_status: ASPClinicalStatus = field(default_factory=ASPClinicalStatus)
    pregnancy_toxicology: PregnancyToxicology = field(default_factory=PregnancyToxicology)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION FUNCTION
# ═════════════════════════════════════════════════════════════════════════════


def extract_asp_patient(row: pd.Series, patient_id: str | None = None) -> ASPPatientData:
    """Extract structured patient data from an ASP.csv row.

    All CSV column names come from ``config/glossary/asp/column_map.yaml``.
    """
    cm = get_cohort_column_map("asp")
    pid = patient_id or _safe_str(row.get(cm.patient_id_column)) or "unknown"
    extras = cm.model_extra or {}

    return ASPPatientData(
        patient_id=pid,
        demographics=_extract_demographics(row, cm),
        autism_diagnosis=_extract_autism_diagnosis(row, cm),
        developmental_history=_extract_developmental_history(row, cm),
        depression_scores=_extract_instrument_scores(row, ASP_DEPRESSION_INSTRUMENTS),
        global_scores=_extract_instrument_scores(row, ASP_GLOBAL_INSTRUMENTS),
        functioning_scores=_extract_instrument_scores(row, ASP_FUNCTIONING_INSTRUMENTS),
        repetitive_behavior_scores=_extract_instrument_scores(row, ASP_REPETITIVE_INSTRUMENTS),
        cognitive_scores=_extract_instrument_scores(row, ASP_COGNITIVE_INSTRUMENTS),
        executive_function_scores=_extract_instrument_scores(row, ASP_EXECUTIVE_INSTRUMENTS),
        autism_screening_scores=_extract_instrument_scores(row, ASP_AUTISM_SCREENING_INSTRUMENTS),
        anxiety_scores=_extract_instrument_scores(row, ASP_ANXIETY_INSTRUMENTS),
        adhd_scores=_extract_instrument_scores(row, ASP_ADHD_INSTRUMENTS),
        trauma_scores=_extract_instrument_scores(row, ASP_TRAUMA_INSTRUMENTS),
        sleep_scores=_extract_instrument_scores(row, ASP_SLEEP_INSTRUMENTS),
        adherence_scores=_extract_instrument_scores(row, ASP_ADHERENCE_INSTRUMENTS),
        substance_scores=_extract_instrument_scores(row, ASP_SUBSTANCE_INSTRUMENTS),
        substance_use=_extract_asp_substance_use(row, cm),
        bdi_item9_suicidal_thoughts=_safe_int(row.get(extras.get("bdi_item9", "bdi0209"))),
        biology=_extract_asp_biology(row, cm),
        treatments=_extract_asp_treatments(row, cm),
        somatic_comorbidities=_extract_asp_somatic_comorbidities(row, cm),
        psychiatric_comorbidities=_extract_psychiatric_comorbidities(row, cm),
        learning_disabilities=_extract_learning_disabilities(row, cm),
        mcdd_profile=_extract_mcdd(row, cm),
        medical_antecedents=_extract_medical_antecedents(row, cm),
        clinical_status=_extract_clinical_status(row, cm),
        pregnancy_toxicology=_extract_pregnancy_toxicology(row, cm),
    )


# ─── Sub-extractors ──────────────────────────────────────────────────────────


def _extract_demographics(row: pd.Series, cm) -> Demographics:
    """ASP uses usubjid_identite + empjob1/empjob2, no maristat."""
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

    employment1 = _decode(row.get(d.employment), EMPLOYMENT_CODES) if d.employment else None
    employment2 = _decode(row.get(d.employment_alt), EMPLOYMENT_CODES) if d.employment_alt else None

    return Demographics(
        age=_safe_int(row.get(d.age)),
        sex=sex,
        sex_label_fr=sex_label_fr,
        site_id=_safe_str(row.get(d.site_id)),
        arm=_safe_str(row.get(d.arm)),
        marital_status=None,  # ASP has no maristat
        education_level=_decode(row.get(d.education_level), EDUCATION_LEVEL_CODES) if d.education_level else None,
        employment=employment1 or employment2,
    )


def _extract_instrument_scores(
    row: pd.Series, instrument_names: list[str],
) -> dict[str, ScoreInterpretation]:
    """Generic instrument score extractor using ASP_INSTRUMENTS registry."""
    results = {}
    for name in instrument_names:
        inst = ASP_INSTRUMENTS.get(name)
        if inst is None:
            continue
        raw = _safe_float(row.get(inst.total_column))
        subscales: dict[str, float | None] = {}
        for sub_name, sub_col in inst.subscale_columns.items():
            subscales[sub_name] = _safe_float(row.get(sub_col))
        results[name] = interpret_score(inst, raw, subscales)
    return results


def _flag_or_none(row, col: str | None) -> bool | None:
    """ASP pattern: 0/1/9 → True/False/None (9 = unknown)."""
    if not col:
        return None
    val = _safe_float(row.get(col))
    if val is None or val == 9:
        return None
    return val == 1


def _extract_autism_diagnosis(row: pd.Series, cm) -> AutismDiagnosis:
    """Extract autism diagnostic profile using column map."""
    ad = cm.model_extra.get("autism_diagnosis", {}) if cm.model_extra else {}

    dsm_type_col = ad.get("dsm_type")
    dsm_type_raw = _safe_float(row.get(dsm_type_col)) if dsm_type_col else None
    dsm_type_label = _decode(row.get(dsm_type_col), DSM_TYPE_CODES) if dsm_type_col else None

    dsmv1 = _safe_float(row.get(ad.get("dsm_domain1"))) if ad.get("dsm_domain1") else None
    dsmv2 = _safe_float(row.get(ad.get("dsm_domain2"))) if ad.get("dsm_domain2") else None

    dsm_criteria: dict[str, bool] = {}
    for col, label in (ad.get("dsm_criteria") or {}).items():
        val = _safe_float(row.get(col))
        if val is not None:
            dsm_criteria[label] = val == 1.0

    return AutismDiagnosis(
        dsm_type=dsm_type_raw,
        dsm_type_label=dsm_type_label,
        dsm_domain1_met=dsmv1 == 1.0 if dsmv1 is not None else None,
        dsm_domain2_met=dsmv2 == 1.0 if dsmv2 is not None else None,
        adi_diagnostic=_safe_float(row.get(ad.get("adi_diagnostic"))) if ad.get("adi_diagnostic") else None,
        ados_exam=_safe_str(row.get(ad.get("ados_exam"))) if ad.get("ados_exam") else None,
        dsm_criteria=dsm_criteria,
    )


def _extract_developmental_history(row: pd.Series, cm) -> DevelopmentalHistory:
    """Developmental milestones, neonatal history, parental ages — from column map."""
    dh = cm.model_extra.get("developmental_history", {}) if cm.model_extra else {}

    def f(key: str):
        col = dh.get(key)
        return _safe_float(row.get(col)) if col else None

    def s(key: str):
        col = dh.get(key)
        return _safe_str(row.get(col)) if col else None

    return DevelopmentalHistory(
        age_motor_milestones=f("age_motor_milestones"),
        age_first_phrases=f("age_first_phrases"),
        mother_age=f("mother_age"),
        father_age=f("father_age"),
        psychomotor_delay=_flag_or_none(row, dh.get("psychomotor_delay")),
        language_delay=_flag_or_none(row, dh.get("language_delay")),
        fetal_distress=_flag_or_none(row, dh.get("fetal_distress")),
        neonatal_complications=_flag_or_none(row, dh.get("neonatal_complications")),
        resuscitation=_flag_or_none(row, dh.get("resuscitation")),
        pregnancy_pathology=_flag_or_none(row, dh.get("pregnancy_pathology")),
        fetal_pathology=_flag_or_none(row, dh.get("fetal_pathology")),
        neonatal_illness=_flag_or_none(row, dh.get("neonatal_illness")),
        feeding_difficulties=_flag_or_none(row, dh.get("feeding_difficulties")),
        sleep_difficulties=_flag_or_none(row, dh.get("sleep_difficulties")),
        twin=_flag_or_none(row, dh.get("twin")),
        seizures=_flag_or_none(row, dh.get("seizures")),
        birth_weight_g=f("birth_weight_g"),
        birth_height_cm=f("birth_height_cm"),
        apgar_score=f("apgar_score"),
        language_expression=s("language_expression"),
        current_language=s("current_language"),
    )


def _extract_asp_biology(row: pd.Series, cm) -> BiologicalPanel:
    """Extract minimal biology panel. Lab definitions loaded from config/glossary/asp/lab_ranges.yaml."""
    from face_rlvr.profiles.glossary_loader import get_cohort_lab_ranges

    values: list[LabValue] = []
    vitals: dict[str, float | None] = {}

    # Vitals from column_map (ASP currently only has bmi)
    vitals_map = cm.vitals.model_dump() if cm.vitals else {}
    for k, col in vitals_map.items():
        v = _safe_float(row.get(col))
        if v is not None:
            vitals[k] = v

    # Lab values from lab_ranges.yaml
    for lab in get_cohort_lab_ranges("asp"):
        val = _safe_float(row.get(lab.csv_col))
        if val is None:
            continue
        nrange = tuple(lab.normal_range) if lab.normal_range else None
        is_abnormal = False
        abnormality = None
        if nrange:
            lo, hi = nrange
            if val < lo:
                is_abnormal, abnormality = True, "low"
            elif val > hi:
                is_abnormal, abnormality = True, "high"
        values.append(LabValue(
            name=lab.name,
            name_fr=lab.name_fr,
            value=val,
            unit=lab.unit,
            normal_range=nrange,
            is_abnormal=is_abnormal,
            abnormality=abnormality,
        ))

    return BiologicalPanel(values=values, vitals=vitals)


def _extract_asp_treatments(row: pd.Series, cm) -> ASPTreatmentProfile:
    """Extract ASP treatment profile — columns come from column_map."""
    t = cm.model_extra.get("treatments", {}) if cm.model_extra else {}

    non_pharm: list[str] = []
    prefix = t.get("non_pharm_prefix", "trt_non_parmaco_")
    count = t.get("non_pharm_count", 15)
    for i in range(1, count + 1):
        val = _safe_str(row.get(f"{prefix}{i}"))
        if val and val.lower() not in ("nan", "none", ""):
            non_pharm.append(val)

    return ASPTreatmentProfile(
        on_antidepressant=_is_yes(row.get(t.get("antidepressant"))) if t.get("antidepressant") else False,
        on_antipsychotic=_is_yes(row.get(t.get("antipsychotic"))) if t.get("antipsychotic") else False,
        on_lamotrigine=_is_yes(row.get(t.get("lamotrigine"))) if t.get("lamotrigine") else False,
        non_pharm_treatments=non_pharm,
    )


def _extract_asp_somatic_comorbidities(row: pd.Series, cm) -> list[str]:
    """Somatic comorbidities — list from column_map."""
    if cm.comorbidities is None:
        return []
    return [
        item.label_fr for item in cm.comorbidities.somatic
        if item.col and _is_yes(row.get(item.col))
    ]


def _extract_psychiatric_comorbidities(row: pd.Series, cm) -> list[str]:
    """MINI-based psychiatric comorbidities (values 0/1/9) — list from column_map."""
    if cm.comorbidities is None:
        return []
    result: list[str] = []
    for item in cm.comorbidities.psychiatric:
        if not item.col:
            continue
        val = _safe_float(row.get(item.col))
        if val is not None and val == 1:
            result.append(item.label_fr)
    return result


def _extract_learning_disabilities(row: pd.Series, cm) -> LearningDisabilities:
    """Learning disabilities — column names come from column_map."""
    ld = cm.model_extra.get("learning_disabilities", {}) if cm.model_extra else {}
    def flag(key: str):
        return _flag_or_none(row, ld.get(key))
    return LearningDisabilities(
        dyslexia=flag("dyslexia"),
        dysorthography=flag("dysorthography"),
        dyscalculia=flag("dyscalculia"),
        dysphasia=flag("dysphasia"),
        dyspraxia=flag("dyspraxia"),
        speech_disorder=flag("speech_disorder"),
        stuttering=flag("stuttering"),
    )


def _extract_mcdd(row: pd.Series, cm) -> MCDDProfile:
    """MCDD criteria — prefix + count come from column_map."""
    mcdd_cfg = cm.model_extra.get("mcdd", {}) if cm.model_extra else {}
    prefix = mcdd_cfg.get("col_prefix", "mcdd")
    count = mcdd_cfg.get("count", 15)

    criteria_met: list[int] = []
    total_assessed = 0
    for i in range(1, count + 1):
        val = _safe_float(row.get(f"{prefix}{i}"))
        if val is not None:
            total_assessed += 1
            if val == 1:
                criteria_met.append(i)

    return MCDDProfile(
        criteria_met=criteria_met,
        total_criteria_met=len(criteria_met),
        total_criteria_assessed=total_assessed,
    )


# ─── School level lookup ─────────────────────────────────────────────────────

SCHOOL_LEVEL_CODES = _load_asp_codes()["SCHOOL_LEVEL_CODES"]
SCHOOL_TYPE_CODES = _load_asp_codes()["SCHOOL_TYPE_CODES"]


def _extract_medical_antecedents(row: pd.Series, cm) -> MedicalAntecedents:
    """Medical history flags — column names come from column_map."""
    ma = cm.model_extra.get("medical_antecedents", {}) if cm.model_extra else {}
    def flag(key: str):
        return _flag_or_none(row, ma.get(key))
    return MedicalAntecedents(
        cardiac=flag("cardiac"),
        endocrine=flag("endocrine"),
        neurological=flag("neurological"),
        ent=flag("ent"),
        pulmonary=flag("pulmonary"),
        rheumatological=flag("rheumatological"),
        hepatic=flag("hepatic"),
        cancer=flag("cancer"),
        genetic_disorder=flag("genetic_disorder"),
        other_condition=flag("other_condition"),
    )


def _extract_clinical_status(row: pd.Series, cm) -> ASPClinicalStatus:
    """Current clinical/care status + school level — columns from column_map."""
    cs = cm.model_extra.get("clinical_status", {}) if cm.model_extra else {}
    def flag(key: str):
        return _flag_or_none(row, cs.get(key))
    return ASPClinicalStatus(
        age_at_diagnosis_years=_safe_float(row.get(cs.get("age_at_diagnosis_years"))) if cs.get("age_at_diagnosis_years") else None,
        age_at_diagnosis_months=_safe_float(row.get(cs.get("age_at_diagnosis_months"))) if cs.get("age_at_diagnosis_months") else None,
        in_psychiatric_care=flag("in_psychiatric_care"),
        currently_hospitalized=flag("currently_hospitalized"),
        currently_treated=flag("currently_treated"),
        has_insomnia=flag("has_insomnia"),
        has_hypersomnia=flag("has_hypersomnia"),
        executive_function_impairment=flag("executive_function_impairment"),
        social_cognition_impairment=flag("social_cognition_impairment"),
        school_level=_decode(row.get(cs.get("school_level")), SCHOOL_LEVEL_CODES) if cs.get("school_level") else None,
        school_type=_decode(row.get(cs.get("school_type")), SCHOOL_TYPE_CODES) if cs.get("school_type") else None,
    )


def _extract_pregnancy_toxicology(row: pd.Series, cm) -> PregnancyToxicology:
    """Pregnancy and toxicology data — columns from column_map."""
    pt = cm.model_extra.get("pregnancy_toxicology", {}) if cm.model_extra else {}
    def flag(key: str):
        return _flag_or_none(row, pt.get(key))
    return PregnancyToxicology(
        toxicology_exposure=flag("toxicology_exposure"),
        folic_acid_supplementation=flag("folic_acid_supplementation"),
        bleeding_during_pregnancy=flag("bleeding_during_pregnancy"),
        infection_viral=flag("infection_viral"),
    )


def _extract_asp_substance_use(row: pd.Series, cm) -> SubstanceUse:
    """Substance use (ASP uses tabac + PRISM item columns) — all from column_map.

    ``prism_items`` in YAML maps canonical substance categories to their
    column/label pair. ``alcohol`` and ``cannabis`` are bucketed into their
    own flags; everything else goes into ``other_substances``.
    """
    su = cm.substance_use
    if su is None:
        return SubstanceUse()

    tobacco = _is_yes(row.get(su.tobacco)) if su.tobacco else False
    cpd = _safe_float(row.get(su.cigarettes_per_day)) if su.cigarettes_per_day else None

    # PRISM items — reach into the loose dict field
    prism_items = (cm.model_extra.get("substance_use", {}) or {}).get("prism_items", {}) if cm.model_extra else {}

    alcohol = False
    cannabis = False
    other: list[str] = []

    def _item_col_label(v):
        """Normalise {col, label_fr} dict OR bare col string."""
        if isinstance(v, dict):
            return v.get("col"), v.get("label_fr")
        return v, None

    for key, spec in prism_items.items():
        col, label = _item_col_label(spec)
        if not col:
            continue
        val = _safe_float(row.get(col))
        if val is None or val < 1:
            continue
        if key == "alcohol":
            alcohol = True
        elif key == "cannabis":
            cannabis = True
        else:
            other.append(label or key)

    return SubstanceUse(
        tobacco_current=tobacco,
        tobacco_cpd=cpd,
        alcohol_current=alcohol,
        cannabis_current=cannabis,
        other_substances=other,
    )
