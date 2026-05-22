"""Structured data extraction from BP.csv rows.

Extracts:
- Questionnaire SCORES only (no raw items)
- Demographics, biology, vitals, treatments, comorbidities, history
- Applies clinical interpretations to each score

All shared dataclasses and helpers come from common_extractors.
Only BP-specific data structures and logic live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from face_rlvr.profiles.bp_instruments import (
    BP_INSTRUMENTS,
    MOOD_INSTRUMENTS,
    FUNCTIONAL_INSTRUMENTS,
    ANXIETY_IMPULSIVITY_INSTRUMENTS,
    SLEEP_INSTRUMENTS,
    COGNITIVE_INSTRUMENTS,
    ADHERENCE_INSTRUMENTS,
    TRAUMA_INSTRUMENTS,
    SUICIDE_INSTRUMENTS,
    TREATMENT_RESPONSE_INSTRUMENTS,
    SUBSTANCE_INSTRUMENTS,
    SCREENING_INSTRUMENTS,
)
from face_rlvr.profiles.common_instruments import (
    InstrumentDefinition,
    ScoreInterpretation,
    interpret_score,
)
from face_rlvr.profiles.common_extractors import (
    # Shared dataclasses
    Demographics,
    LabValue,
    BiologicalPanel,
    SubstanceUse,
    RelativeHistory,
    FamilyHistory,
    SuicideHistory,
    HospitalizationHistory,
    CognitiveProfile,
    AdditionalNeuropsych,
    NonPharmTreatment,
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
    # Shared sub-extractors
    extract_substance_use as _extract_substance_use,
    extract_suicide_indicators as _extract_suicide_indicators,
    extract_suicide_history as _extract_suicide_history,
    # Data completeness helper
    compute_data_completeness,
)
from face_rlvr.profiles.glossary_loader import get_cohort_column_map


# ─── BP-specific dataclasses ──────────────────────────────────────────────────


@dataclass
class PsychiatricHistory:
    age_first_episode: int | None = None
    illness_duration_years: float | None = None
    rapid_cycling: bool | None = None
    current_episode_type: str | None = None
    current_episode_severity: str | None = None

    # Lifetime episode counts
    n_depressive_episodes_lifetime: int | None = None
    n_manic_episodes_lifetime: int | None = None
    n_hypomanic_episodes_lifetime: int | None = None
    n_mixed_episodes_lifetime: int | None = None

    # Last year episode counts
    n_depressive_episodes_last_year: int | None = None
    n_manic_episodes_last_year: int | None = None
    n_mixed_episodes_last_year: int | None = None

    # Psychotic features
    n_psychotic_depressive_lifetime: int | None = None
    n_psychotic_manic_lifetime: int | None = None


@dataclass
class TreatmentProfile:
    current_medications: list[str] = field(default_factory=list)
    mood_stabilizers: dict[str, Any] = field(default_factory=dict)  # drug -> {plasma_level, duration, etc.}
    on_lithium: bool = False
    lithium_plasma: float | None = None
    on_valproate: bool = False
    valproate_plasma: float | None = None
    on_carbamazepine: bool = False
    on_lamotrigine: bool = False
    on_antidepressant: bool = False
    on_antipsychotic: bool = False
    on_benzodiazepine: bool = False
    on_thymoregulator: bool = False
    medication_adherence: ScoreInterpretation | None = None


@dataclass
class LifetimeMedications:
    """Lifetime medication history (ever-exposed classes)."""
    antidepressant_ever: bool = False
    antidepressant_duration_months: float | None = None
    antipsychotic_ever: bool = False
    antipsychotic_duration_months: float | None = None
    neuroleptic_ever: bool = False
    neuroleptic_duration_months: float | None = None
    lithium_ever: bool = False
    lithium_duration_months: float | None = None
    benzodiazepine_ever: bool = False
    benzodiazepine_duration_months: float | None = None
    thymoregulator_ever: bool = False
    thymoregulator_duration_months: float | None = None


@dataclass
class CircadianRhythm:
    """Circadian rhythm assessment (BP-specific)."""
    flexibility_score: float | None = None
    languid_vigorous_score: float | None = None


@dataclass
class DivaADHD:
    """DIVA structured ADHD interview (DSM-IV criteria, BP-specific)."""
    attention_adult_count: int | None = None  # divaaa (0-9)
    hyperactivity_adult_count: int | None = None  # divaahi (0-9)
    attention_childhood_count: int | None = None  # divaea (0-9)
    hyperactivity_childhood_count: int | None = None  # divaehi (0-9)


@dataclass
class BPPatientData:
    """Complete structured patient data extracted from a BP.csv row."""

    patient_id: str
    demographics: Demographics
    psychiatric_history: PsychiatricHistory

    # Interpreted instrument scores (grouped by clinical domain)
    mood_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    functional_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    anxiety_impulsivity_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    sleep_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    screening_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    adherence_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    trauma_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    suicide_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    treatment_response_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)
    substance_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)

    # Cognitive (combination of instruments + neuropsych tests)
    cognitive_profile: CognitiveProfile = field(default_factory=CognitiveProfile)
    additional_neuropsych: AdditionalNeuropsych = field(default_factory=AdditionalNeuropsych)

    # Biology, treatments, comorbidities
    biology: BiologicalPanel = field(default_factory=BiologicalPanel)
    treatments: TreatmentProfile = field(default_factory=TreatmentProfile)
    lifetime_medications: LifetimeMedications = field(default_factory=LifetimeMedications)
    non_pharm_treatments: NonPharmTreatment = field(default_factory=NonPharmTreatment)
    substance_use: SubstanceUse = field(default_factory=SubstanceUse)
    family_history: FamilyHistory = field(default_factory=FamilyHistory)
    hospitalization: HospitalizationHistory = field(default_factory=HospitalizationHistory)

    # Current episode features
    current_episode_criteria: CurrentEpisodeCriteria = field(default_factory=CurrentEpisodeCriteria)
    most_recent_episode: MostRecentEpisode = field(default_factory=MostRecentEpisode)

    # Somatic comorbidities (list of condition names)
    somatic_comorbidities: list[str] = field(default_factory=list)
    psychiatric_comorbidities: list[str] = field(default_factory=list)

    # Suicide-related indicators (summary flags)
    suicide_indicators: dict[str, Any] = field(default_factory=dict)
    # Detailed suicide history (Columbia + ISF)
    suicide_history: SuicideHistory = field(default_factory=SuicideHistory)

    # DIVA ADHD interview
    diva_adhd: DivaADHD = field(default_factory=DivaADHD)

    # Circadian rhythm
    circadian: CircadianRhythm = field(default_factory=CircadianRhythm)

    # V1 follow-up scores (only MADRS and YMRS have _n1 columns in BP.csv)
    v1_mood_scores: dict[str, ScoreInterpretation] = field(default_factory=dict)

    # Data completeness metrics
    data_completeness: dict[str, float] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTION FUNCTION
# ═════════════════════════════════════════════════════════════════════════════


def extract_bp_patient(row: pd.Series, patient_id: str | None = None) -> BPPatientData:
    """Extract structured patient data from a BP.csv row.

    All CSV column names come from ``config/glossary/bp/column_map.yaml``.
    Instrument scores come from ``config/glossary/bp/instruments.yaml``.
    Lab ranges come from ``config/glossary/bp/lab_ranges.yaml``.
    """
    cm = get_cohort_column_map("bp")
    pid = patient_id or _safe_str(row.get(cm.patient_id_column)) or "unknown"

    return BPPatientData(
        patient_id=pid,
        demographics=_extract_demographics(row, cm),
        psychiatric_history=_extract_psychiatric_history(row, cm),
        mood_scores=_extract_instrument_scores(row, MOOD_INSTRUMENTS),
        functional_scores=_extract_instrument_scores(row, FUNCTIONAL_INSTRUMENTS),
        anxiety_impulsivity_scores=_extract_instrument_scores(row, ANXIETY_IMPULSIVITY_INSTRUMENTS),
        sleep_scores=_extract_instrument_scores(row, SLEEP_INSTRUMENTS),
        screening_scores=_extract_instrument_scores(row, SCREENING_INSTRUMENTS),
        adherence_scores=_extract_instrument_scores(row, ADHERENCE_INSTRUMENTS),
        trauma_scores=_extract_instrument_scores(row, TRAUMA_INSTRUMENTS),
        suicide_scores=_extract_instrument_scores(row, SUICIDE_INSTRUMENTS),
        treatment_response_scores=_extract_instrument_scores(row, TREATMENT_RESPONSE_INSTRUMENTS),
        substance_scores=_extract_instrument_scores(row, SUBSTANCE_INSTRUMENTS),
        cognitive_profile=_extract_cognitive_profile(row, cm),
        additional_neuropsych=_extract_additional_neuropsych(row, cm),
        biology=_extract_biology(row, cm),
        treatments=_extract_treatments(row, cm),
        lifetime_medications=_extract_lifetime_medications(row, cm),
        non_pharm_treatments=_extract_non_pharm_treatments(row, cm),
        substance_use=_extract_substance_use(row, cm),
        family_history=_extract_family_history(row, cm),
        hospitalization=_extract_hospitalization(row, cm),
        current_episode_criteria=_extract_current_episode_criteria(row, cm),
        most_recent_episode=_extract_most_recent_episode(row, cm),
        somatic_comorbidities=_extract_somatic_comorbidities(row, cm),
        psychiatric_comorbidities=_extract_psychiatric_comorbidities(row, cm),
        suicide_indicators=_extract_suicide_indicators(row, cm),
        suicide_history=_extract_suicide_history(row, cm),
        diva_adhd=_extract_diva(row, cm),
        circadian=_extract_circadian(row, cm),
        v1_mood_scores=_extract_v1_mood_scores(row, cm),
        data_completeness=_compute_completeness_bp(row),
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
        social_protection=_safe_str(row.get(d.social_protection)) if d.social_protection else None,
    )


def _extract_psychiatric_history(row: pd.Series, cm) -> PsychiatricHistory:
    ph = cm.psychiatric_history
    # psychiatric_history is loosely typed (extra="allow"); read via dict access
    pd_dict = ph.model_dump() if ph else {}

    def col(key: str):
        return pd_dict.get(key)

    rc_col = col("rapid_cycling")
    return PsychiatricHistory(
        age_first_episode=_safe_int(row.get(col("age_first_episode"))) if col("age_first_episode") else None,
        illness_duration_years=_safe_float(row.get(col("illness_duration_years"))) if col("illness_duration_years") else None,
        rapid_cycling=(_is_yes(row.get(rc_col)) if rc_col and _safe_str(row.get(rc_col)) else None),
        current_episode_type=_safe_str(row.get(col("current_episode_type"))) if col("current_episode_type") else None,
        current_episode_severity=_safe_str(row.get(col("current_episode_severity"))) if col("current_episode_severity") else None,
        n_depressive_episodes_lifetime=_safe_int(row.get(col("n_depressive_episodes_lifetime"))) if col("n_depressive_episodes_lifetime") else None,
        n_manic_episodes_lifetime=_safe_int(row.get(col("n_manic_episodes_lifetime"))) if col("n_manic_episodes_lifetime") else None,
        n_hypomanic_episodes_lifetime=_safe_int(row.get(col("n_hypomanic_episodes_lifetime"))) if col("n_hypomanic_episodes_lifetime") else None,
        n_mixed_episodes_lifetime=_safe_int(row.get(col("n_mixed_episodes_lifetime"))) if col("n_mixed_episodes_lifetime") else None,
        n_depressive_episodes_last_year=_safe_int(row.get(col("n_depressive_episodes_last_year"))) if col("n_depressive_episodes_last_year") else None,
        n_manic_episodes_last_year=_safe_int(row.get(col("n_manic_episodes_last_year"))) if col("n_manic_episodes_last_year") else None,
        n_mixed_episodes_last_year=_safe_int(row.get(col("n_mixed_episodes_last_year"))) if col("n_mixed_episodes_last_year") else None,
        n_psychotic_depressive_lifetime=_safe_int(row.get(col("n_psychotic_depressive_lifetime"))) if col("n_psychotic_depressive_lifetime") else None,
        n_psychotic_manic_lifetime=_safe_int(row.get(col("n_psychotic_manic_lifetime"))) if col("n_psychotic_manic_lifetime") else None,
    )


def _extract_instrument_scores(
    row: pd.Series, instrument_names: list[str],
) -> dict[str, ScoreInterpretation]:
    """Extract and interpret scores for a list of instruments."""
    results = {}
    for name in instrument_names:
        inst = BP_INSTRUMENTS.get(name)
        if inst is None:
            continue

        raw = _safe_float(row.get(inst.total_column))

        # Extract subscale values
        subscales: dict[str, float | None] = {}
        for sub_name, sub_col in inst.subscale_columns.items():
            subscales[sub_name] = _safe_float(row.get(sub_col))

        results[name] = interpret_score(inst, raw, subscales)

    return results


def _extract_cognitive_profile(row: pd.Series, cm) -> CognitiveProfile:
    cp = cm.cognitive_profile.model_dump() if cm.cognitive_profile else {}

    def f(key: str):
        col = cp.get(key)
        return _safe_float(row.get(col)) if col else None

    tmt_a = f("tmt_a_seconds")
    tmt_b = f("tmt_b_seconds")
    tmt_ba = (tmt_b - tmt_a) if (tmt_a is not None and tmt_b is not None) else None
    tmt_ratio = (tmt_b / tmt_a) if (tmt_a and tmt_a > 0 and tmt_b is not None) else None

    # COBRA interpretation (reads the instrument from the YAML registry)
    cobra_interp = None
    cobra_key = cp.get("cobra_instrument_key")
    if cobra_key:
        cobra_inst = BP_INSTRUMENTS.get(cobra_key)
        if cobra_inst is not None:
            cobra_raw = _safe_float(row.get(cobra_inst.total_column))
            cobra_interp = interpret_score(cobra_inst, cobra_raw)

    return CognitiveProfile(
        tmt_a_seconds=tmt_a,
        tmt_b_seconds=tmt_b,
        tmt_b_minus_a=round(tmt_ba, 1) if tmt_ba is not None else None,
        tmt_ratio_ba=round(tmt_ratio, 2) if tmt_ratio is not None else None,
        stroop_word=f("stroop_word"),
        stroop_color=f("stroop_color"),
        stroop_color_word=f("stroop_color_word"),
        stroop_interference=f("stroop_interference"),
        cvlt_total_learning=f("cvlt_total_learning"),
        cvlt_short_delay_free=f("cvlt_short_delay_free"),
        cvlt_long_delay_free=f("cvlt_long_delay_free"),
        cvlt_recognition=f("cvlt_recognition"),
        phonemic_fluency=f("phonemic_fluency"),
        wais_similarities_std=f("wais_similarities_std"),
        wais_vocabulary_std=f("wais_vocabulary_std"),
        wais_working_memory_std=f("wais_working_memory_std"),
        cobra_interpretation=cobra_interp,
    )


def _extract_biology(row: pd.Series, cm) -> BiologicalPanel:
    """Extract lab values + vitals + ECG. All columns come from the column_map."""
    from face_rlvr.profiles.glossary_loader import get_cohort_lab_ranges

    values = []
    for lab in get_cohort_lab_ranges("bp"):
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


def _extract_treatments(row: pd.Series, cm) -> TreatmentProfile:
    t = cm.treatments.model_dump() if cm.treatments else {}
    mood_stab_defs = t.get("mood_stabilizers", {})
    class_cols = t.get("classes", {})

    # Plasma levels for mood stabilizers (BP-specific)
    levels: dict[str, float | None] = {}
    for drug_key, d in mood_stab_defs.items():
        levels[drug_key] = _safe_float(row.get(d["plasma"]))

    on_li = levels.get("lithium") is not None and levels["lithium"] > 0
    on_vpa = levels.get("valproate") is not None and levels["valproate"] > 0
    on_cbz = levels.get("carbamazepine") is not None and levels["carbamazepine"] > 0
    on_lam = levels.get("lamotrigine") is not None and levels["lamotrigine"] > 0

    mood_stab: dict[str, Any] = {}
    for drug_key, val in levels.items():
        if val is not None and val > 0:
            d = mood_stab_defs[drug_key]
            mood_stab[drug_key] = {"plasma_level": val, "unit": d.get("unit", "")}

    on_antid = _is_yes(row.get(class_cols.get("antidepressant"))) if class_cols.get("antidepressant") else False
    on_antip = _is_yes(row.get(class_cols.get("antipsychotic"))) if class_cols.get("antipsychotic") else False
    on_benzo = _is_yes(row.get(class_cols.get("benzodiazepine"))) if class_cols.get("benzodiazepine") else False
    on_thymo = _is_yes(row.get(class_cols.get("thymoregulator"))) if class_cols.get("thymoregulator") else False

    # Medication adherence via the named instrument (MARS)
    mars_interp = None
    adherence_key = t.get("adherence_instrument_key")
    if adherence_key:
        inst = BP_INSTRUMENTS.get(adherence_key)
        if inst is not None:
            mars_raw = _safe_float(row.get(inst.total_column))
            mars_interp = interpret_score(inst, mars_raw)

    return TreatmentProfile(
        mood_stabilizers=mood_stab,
        on_lithium=on_li,
        lithium_plasma=levels.get("lithium"),
        on_valproate=on_vpa,
        valproate_plasma=levels.get("valproate"),
        on_carbamazepine=on_cbz,
        on_lamotrigine=on_lam,
        on_antidepressant=on_antid,
        on_antipsychotic=on_antip,
        on_benzodiazepine=on_benzo,
        on_thymoregulator=on_thymo,
        medication_adherence=mars_interp,
    )


def _extract_family_history(row: pd.Series, cm) -> FamilyHistory:
    fh = cm.family_history
    if fh is None:
        return FamilyHistory()

    mat_trouble = _safe_str(row.get(fh.maternal_psychiatric)) if fh.maternal_psychiatric else None
    pat_trouble = _safe_str(row.get(fh.paternal_psychiatric)) if fh.paternal_psychiatric else None
    mat_substance = _is_yes(row.get(fh.maternal_substance)) if fh.maternal_substance else False
    pat_substance = _is_yes(row.get(fh.paternal_substance)) if fh.paternal_substance else False
    mat_suicide = _is_yes(row.get(fh.maternal_suicide)) if fh.maternal_suicide else False
    pat_suicide = _is_yes(row.get(fh.paternal_suicide)) if fh.paternal_suicide else False

    keyword = fh.bipolar_keyword.lower()
    family_bp = False
    for trouble in (mat_trouble, pat_trouble):
        if trouble and keyword in trouble.lower():
            family_bp = True
            break

    # Extended pedigree (grandparents + siblings)
    relatives: list[RelativeHistory] = []
    sfx = fh.relative_suffixes

    def _build_relative(prefix: str, label_fr: str, include_cardio: bool) -> RelativeHistory | None:
        trouble = _safe_str(row.get(f"{prefix}{sfx.get('psychiatric_disorder', '_trouble')}"))
        structure_col = f"{prefix}{sfx.get('structure', '_structure')}"
        structure = _safe_str(row.get(structure_col))
        if not trouble and not structure:
            return None
        rel = RelativeHistory(
            relation=prefix,
            relation_fr=label_fr,
            psychiatric_disorder=trouble,
            suicide=_is_yes(row.get(f"{prefix}{sfx.get('suicide', '_suicide')}")),
            substance_use=_is_yes(row.get(f"{prefix}{sfx.get('substance_use', '_substance')}")),
            anxiety=_is_yes(row.get(f"{prefix}{sfx.get('anxiety', '_anx')}")),
            dementia=_is_yes(row.get(f"{prefix}{sfx.get('dementia', '_dem')}")),
            cardiovascular_risk=(
                _is_yes(row.get(f"{prefix}{sfx.get('cardiovascular_risk', '_risque')}"))
                if include_cardio else False
            ),
        )
        return rel

    for item in fh.relatives:
        rel = _build_relative(item["key"], item["label_fr"], include_cardio=True)
        if rel:
            relatives.append(rel)
            if rel.psychiatric_disorder and keyword in rel.psychiatric_disorder.lower():
                family_bp = True

    for item in fh.siblings:
        prefix = item["key"]
        trouble = _safe_str(row.get(f"{prefix}{sfx.get('psychiatric_disorder', '_trouble')}"))
        if not trouble:
            continue
        rel = RelativeHistory(
            relation=prefix,
            relation_fr=item["label_fr"],
            psychiatric_disorder=trouble,
            suicide=_is_yes(row.get(f"{prefix}{sfx.get('suicide', '_suicide')}")),
            substance_use=_is_yes(row.get(f"{prefix}{sfx.get('substance_use', '_substance')}")),
            anxiety=_is_yes(row.get(f"{prefix}{sfx.get('anxiety', '_anx')}")),
            dementia=_is_yes(row.get(f"{prefix}{sfx.get('dementia', '_dem')}")),
        )
        relatives.append(rel)
        if keyword in trouble.lower():
            family_bp = True

    # Sibling + children counts
    def _count(col: str | None) -> int | None:
        return _safe_int(row.get(col)) if col else None

    n_brothers = _count(fh.brothers_count_col)
    n_sisters = _count(fh.sisters_count_col)
    n_siblings = None
    if n_brothers is not None or n_sisters is not None:
        n_siblings = (n_brothers or 0) + (n_sisters or 0)

    brothers_aff = _count(fh.brothers_affected_col)
    sisters_aff = _count(fh.sisters_affected_col)
    n_siblings_aff = None
    if brothers_aff is not None or sisters_aff is not None:
        n_siblings_aff = (brothers_aff or 0) + (sisters_aff or 0)

    n_sons = _count(fh.sons_count_col)
    n_daughters = _count(fh.daughters_count_col)
    n_children = None
    if n_sons is not None or n_daughters is not None:
        n_children = (n_sons or 0) + (n_daughters or 0)

    sons_aff = _count(fh.sons_affected_col)
    daughters_aff = _count(fh.daughters_affected_col)
    n_children_aff = None
    if sons_aff is not None or daughters_aff is not None:
        n_children_aff = (sons_aff or 0) + (daughters_aff or 0)

    return FamilyHistory(
        maternal_psychiatric=mat_trouble,
        paternal_psychiatric=pat_trouble,
        maternal_substance=mat_substance,
        paternal_substance=pat_substance,
        maternal_suicide=mat_suicide,
        paternal_suicide=pat_suicide,
        family_bipolar=family_bp,
        relatives=relatives,
        n_siblings=n_siblings,
        n_siblings_affected=n_siblings_aff,
        n_children=n_children,
        n_children_affected=n_children_aff,
    )


def _extract_hospitalization(row: pd.Series, cm) -> HospitalizationHistory:
    h = cm.hospitalization
    if h is None:
        return HospitalizationHistory()

    n_lt = _safe_int(row.get(h.n_lifetime)) if h.n_lifetime else None
    return HospitalizationHistory(
        ever_hospitalized=(n_lt is not None and n_lt > 0),
        n_hospitalizations_lifetime=n_lt,
        n_hospitalizations_last_year=_safe_int(row.get(h.n_last_year)) if h.n_last_year else None,
        duration_last_hospitalization=_safe_float(row.get(h.duration_last)) if h.duration_last else None,
        er_visits_recent=_is_yes(row.get(h.er_visits_flag)) if h.er_visits_flag else False,
        n_er_visits=_safe_int(row.get(h.n_er_visits)) if h.n_er_visits else None,
        work_absences=_is_yes(row.get(h.work_absences_flag)) if h.work_absences_flag else False,
        n_work_absences=_safe_int(row.get(h.n_work_absences)) if h.n_work_absences else None,
    )


def _extract_somatic_comorbidities(row: pd.Series, cm) -> list[str]:
    """Somatic comorbidities. Column list comes from config/glossary/bp/column_map.yaml."""
    if cm.comorbidities is None:
        return []
    result = []
    for item in cm.comorbidities.somatic:
        col = item.col or (f"{item.key}{cm.comorbidities.somatic_suffix}" if item.key else None)
        if col and _is_yes(row.get(col)):
            result.append(item.label_fr)
    return result


def _extract_psychiatric_comorbidities(row: pd.Series, cm) -> list[str]:
    """Psychiatric comorbidities + general anxiety + SUD flags from YAML."""
    c = cm.comorbidities
    if c is None:
        return []
    comorbidities: list[str] = []
    for item in c.psychiatric:
        col = item.col
        if col and _is_yes(row.get(col)):
            comorbidities.append(item.label_fr)

    if c.general_anxiety_flag and _is_yes(row.get(c.general_anxiety_flag)) \
            and not any("anxieux" in x.lower() for x in comorbidities):
        comorbidities.append(c.general_anxiety_label_fr)

    if c.substance_use_flag and _is_yes(row.get(c.substance_use_flag)):
        comorbidities.append(c.substance_use_label_fr)

    return comorbidities


def _extract_current_episode_criteria(row: pd.Series, cm) -> CurrentEpisodeCriteria:
    """DSM symptom criteria for current episodes — column names come from YAML."""
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


def _extract_lifetime_medications(row: pd.Series, cm) -> LifetimeMedications:
    lm = cm.lifetime_medications
    if lm is None:
        return LifetimeMedications()
    classes = lm.classes
    def pair(name: str) -> tuple[bool, float | None]:
        d = classes.get(name, {})
        return (
            _is_yes(row.get(d.get("ever"))) if d.get("ever") else False,
            _safe_float(row.get(d.get("duration"))) if d.get("duration") else None,
        )
    ad_ever, ad_dur = pair("antidepressant")
    ap_ever, ap_dur = pair("antipsychotic")
    nl_ever, nl_dur = pair("neuroleptic")
    li_ever, li_dur = pair("lithium")
    bz_ever, bz_dur = pair("benzodiazepine")
    ty_ever, ty_dur = pair("thymoregulator")
    return LifetimeMedications(
        antidepressant_ever=ad_ever, antidepressant_duration_months=ad_dur,
        antipsychotic_ever=ap_ever, antipsychotic_duration_months=ap_dur,
        neuroleptic_ever=nl_ever, neuroleptic_duration_months=nl_dur,
        lithium_ever=li_ever, lithium_duration_months=li_dur,
        benzodiazepine_ever=bz_ever, benzodiazepine_duration_months=bz_dur,
        thymoregulator_ever=ty_ever, thymoregulator_duration_months=ty_dur,
    )


def _extract_non_pharm_treatments(row: pd.Series, cm) -> NonPharmTreatment:
    npt = cm.non_pharm_treatments.model_dump() if cm.non_pharm_treatments else {}

    def yes_of(key: str) -> bool:
        col = npt.get(key)
        return _is_yes(row.get(col)) if col else False

    def int_of(key: str) -> int | None:
        col = npt.get(key)
        return _safe_int(row.get(col)) if col else None

    return NonPharmTreatment(
        has_non_pharm_lifetime=yes_of("has_any"),
        ect_lifetime=yes_of("ect_ever"),
        ect_sessions=int_of("ect_sessions"),
        tms_lifetime=yes_of("tms_ever"),
        tms_sessions=int_of("tms_sessions"),
        cbt_lifetime=yes_of("cbt_ever"),
        ipsrt_lifetime=yes_of("ipsrt_ever"),
        psychoeducation_lifetime=yes_of("psychoeducation_ever"),
    )


def _extract_additional_neuropsych(row: pd.Series, cm) -> AdditionalNeuropsych:
    an = cm.additional_neuropsych.model_dump() if cm.additional_neuropsych else {}

    def f(key: str):
        col = an.get(key)
        return _safe_float(row.get(col)) if col else None

    def y(key: str):
        col = an.get(key)
        return _col_yes_or_none(row, col)

    return AdditionalNeuropsych(
        matrices_raw=f("matrices_raw"),
        matrices_std=f("matrices_std"),
        code_raw=f("code_raw"),
        code_std=f("code_std"),
        symbol_raw=f("symbol_raw"),
        symbol_std=f("symbol_std"),
        digit_span_forward_total=f("digit_span_forward_total"),
        digit_span_forward_std=f("digit_span_forward_std"),
        digit_span_backward_total=f("digit_span_backward_total"),
        digit_span_backward_std=f("digit_span_backward_std"),
        digit_span_total_raw=f("digit_span_total_raw"),
        digit_span_total_std=f("digit_span_total_std"),
        cpt_omissions=f("cpt_omissions"),
        cpt_commissions=f("cpt_commissions"),
        cpt_hit_rt=f("cpt_hit_rt"),
        cpt_variability=f("cpt_variability"),
        cpt_detectability=f("cpt_detectability"),
        cpt_perseverations=f("cpt_perseverations"),
        dyslexia=y("dyslexia"),
        dysorthographia=y("dysorthographia"),
        dyscalculia=y("dyscalculia"),
        dyspraxia=y("dyspraxia"),
    )


def _extract_diva(row: pd.Series, cm) -> DivaADHD:
    dv = cm.diva_adhd.model_dump() if cm.diva_adhd else {}
    def i(key: str):
        col = dv.get(key)
        return _safe_int(row.get(col)) if col else None
    return DivaADHD(
        attention_adult_count=i("attention_adult_count"),
        hyperactivity_adult_count=i("hyperactivity_adult_count"),
        attention_childhood_count=i("attention_childhood_count"),
        hyperactivity_childhood_count=i("hyperactivity_childhood_count"),
    )


def _extract_circadian(row: pd.Series, cm) -> CircadianRhythm:
    cr = cm.circadian.model_dump() if cm.circadian else {}
    def f(key: str):
        col = cr.get(key)
        return _safe_float(row.get(col)) if col else None
    return CircadianRhythm(
        flexibility_score=f("flexibility_score"),
        languid_vigorous_score=f("languid_vigorous_score"),
    )


# ─── V1 Follow-up extraction ────────────────────────────────────────────────


def _extract_v1_mood_scores(row: pd.Series, cm) -> dict[str, ScoreInterpretation]:
    """Extract V1 (follow-up) mood scores from columns listed in the column map.

    BP is the only cohort with _n1 follow-up columns. The instrument keys and
    their V1 column names come from ``config/glossary/bp/column_map.yaml``.
    """
    from copy import copy

    results: dict[str, ScoreInterpretation] = {}
    v1_cfg = cm.v1_followup
    if v1_cfg is None:
        return results
    for key, v1_col in v1_cfg.instruments.items():
        raw = _safe_float(row.get(v1_col))
        if raw is None:
            continue
        v1_def = copy(BP_INSTRUMENTS[key])
        v1_def.total_column = v1_col
        results[f"{key}_V1"] = interpret_score(v1_def, raw)
    return results


def _compute_completeness_bp(row: pd.Series) -> dict[str, float]:
    """Compute data completeness for BP patient."""
    # Extract scores first to check availability
    score_dicts = {
        "mood": _extract_instrument_scores(row, MOOD_INSTRUMENTS),
        "functional": _extract_instrument_scores(row, FUNCTIONAL_INSTRUMENTS),
        "anxiety_impulsivity": _extract_instrument_scores(row, ANXIETY_IMPULSIVITY_INSTRUMENTS),
        "sleep": _extract_instrument_scores(row, SLEEP_INSTRUMENTS),
        "cognitive": _extract_instrument_scores(row, ["COBRA"]),
        "adherence": _extract_instrument_scores(row, ADHERENCE_INSTRUMENTS),
        "trauma": _extract_instrument_scores(row, TRAUMA_INSTRUMENTS),
        "suicide": _extract_instrument_scores(row, SUICIDE_INSTRUMENTS),
        "treatment_response": _extract_instrument_scores(row, TREATMENT_RESPONSE_INSTRUMENTS),
        "substance": _extract_instrument_scores(row, SUBSTANCE_INSTRUMENTS),
        "screening": _extract_instrument_scores(row, SCREENING_INSTRUMENTS),
    }
    return compute_data_completeness(score_dicts)
