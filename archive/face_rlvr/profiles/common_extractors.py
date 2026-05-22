"""Shared data extraction helpers, dataclasses, and clinical utility functions.

Used by all cohort extractors (BP, SZ, DR, ASP).
Contains:
- Safe value extraction helpers
- Categorical code lookups
- Shared dataclasses (Demographics, LabValue, BiologicalPanel, etc.)
- Clinical computation utilities (BMI, metabolic syndrome, Framingham, etc.)
- Quality control functions (floor/ceiling, completeness, RCI)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from face_rlvr.profiles.common_instruments import ScoreInterpretation, InstrumentDefinition


# ─── Safe value extraction helpers ───────────────────────────────────────────


def _safe_float(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_str(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip()
    return s if s and s.lower() != "nan" else None


def _safe_int(val: Any) -> int | None:
    f = _safe_float(val)
    return int(f) if f is not None else None


def _is_yes(val: Any) -> bool:
    s = _safe_str(val)
    if s is None:
        return False
    return s.lower() in ("y", "yes", "oui", "1", "o", "true")


def _decode(val: Any, lookup: dict[str, str]) -> str | None:
    """Decode a raw value using a lookup dictionary."""
    s = _safe_str(val)
    if s is None:
        return None
    return lookup.get(s) or lookup.get(s.replace(".0", "")) or s


# ─── Categorical code lookups ────────────────────────────────────────────────

# Categorical code lookups loaded from config/glossary/common/categorical_codes.yaml
def _load_common_codes() -> dict[str, dict[str, str]]:
    from face_rlvr.profiles.glossary_loader import get_common_categorical_codes
    return get_common_categorical_codes()

MARITAL_STATUS_CODES = _load_common_codes()["MARITAL_STATUS_CODES"]
EDUCATION_LEVEL_CODES = _load_common_codes()["EDUCATION_LEVEL_CODES"]
EMPLOYMENT_CODES = _load_common_codes()["EMPLOYMENT_CODES"]


# ─── Shared dataclasses ──────────────────────────────────────────────────────


@dataclass
class Demographics:
    age: int | None = None
    sex: str | None = None  # "M" or "F"
    sex_label_fr: str | None = None  # "homme" or "femme"
    site_id: str | None = None
    arm: str | None = None
    marital_status: str | None = None
    education_level: str | None = None
    employment: str | None = None
    social_protection: str | None = None


@dataclass
class LabValue:
    name: str
    name_fr: str
    value: float | None
    unit: str
    normal_range: tuple[float, float] | None = None
    is_abnormal: bool = False
    abnormality: str | None = None  # "high" or "low"


@dataclass
class BiologicalPanel:
    values: list[LabValue] = field(default_factory=list)
    vitals: dict[str, float | None] = field(default_factory=dict)
    ecg: dict[str, float | None] = field(default_factory=dict)


@dataclass
class SubstanceUse:
    tobacco_current: bool = False
    tobacco_cpd: float | None = None  # cigarettes per day
    alcohol_current: bool = False
    alcohol_type: str | None = None
    cannabis_current: bool = False
    other_substances: list[str] = field(default_factory=list)
    substance_use_disorder: bool = False


@dataclass
class RelativeHistory:
    """Psychiatric history for a single family member."""
    relation: str
    relation_fr: str
    psychiatric_disorder: str | None = None
    suicide: bool = False
    substance_use: bool = False
    anxiety: bool = False
    dementia: bool = False
    cardiovascular_risk: bool = False


@dataclass
class FamilyHistory:
    maternal_psychiatric: str | None = None
    paternal_psychiatric: str | None = None
    maternal_substance: bool = False
    paternal_substance: bool = False
    maternal_suicide: bool = False
    paternal_suicide: bool = False
    family_bipolar: bool = False
    relatives: list[RelativeHistory] = field(default_factory=list)
    n_siblings: int | None = None
    n_siblings_affected: int | None = None
    n_children: int | None = None
    n_children_affected: int | None = None


@dataclass
class SuicideHistory:
    """Detailed suicide attempt history (Columbia + ISF)."""
    ever_felt_life_not_worth: bool | None = None
    ever_wished_dead: bool | None = None
    ever_thought_suicide: bool | None = None
    ever_planned_suicide: bool | None = None
    ever_attempted: bool | None = None
    n_attempts: int | None = None
    has_violent_attempts: bool | None = None
    n_violent_attempts: int | None = None
    has_serious_attempts: bool | None = None
    n_serious_attempts: int | None = None
    most_serious_method: str | None = None
    most_violent_method: str | None = None
    most_serious_trigger: str | None = None


@dataclass
class HospitalizationHistory:
    ever_hospitalized: bool = False
    n_hospitalizations_lifetime: int | None = None
    n_hospitalizations_last_year: int | None = None
    duration_last_hospitalization: float | None = None
    er_visits_recent: bool = False
    n_er_visits: int | None = None
    work_absences: bool = False
    n_work_absences: int | None = None


@dataclass
class CognitiveProfile:
    """Neuropsychological test results (scores, not raw items)."""
    tmt_a_seconds: float | None = None
    tmt_b_seconds: float | None = None
    tmt_b_minus_a: float | None = None
    tmt_ratio_ba: float | None = None
    stroop_word: float | None = None
    stroop_color: float | None = None
    stroop_color_word: float | None = None
    stroop_interference: float | None = None
    cvlt_total_learning: float | None = None
    cvlt_short_delay_free: float | None = None
    cvlt_long_delay_free: float | None = None
    cvlt_recognition: float | None = None
    phonemic_fluency: float | None = None
    semantic_fluency: float | None = None
    wais_similarities_std: float | None = None
    wais_vocabulary_std: float | None = None
    wais_working_memory_std: float | None = None
    cobra_interpretation: ScoreInterpretation | None = None


@dataclass
class AdditionalNeuropsych:
    """Additional neuropsych tests."""
    matrices_raw: float | None = None
    matrices_std: float | None = None
    code_raw: float | None = None
    code_std: float | None = None
    symbol_raw: float | None = None
    symbol_std: float | None = None
    digit_span_forward_total: float | None = None
    digit_span_forward_std: float | None = None
    digit_span_backward_total: float | None = None
    digit_span_backward_std: float | None = None
    digit_span_total_raw: float | None = None
    digit_span_total_std: float | None = None
    cpt_omissions: float | None = None
    cpt_commissions: float | None = None
    cpt_hit_rt: float | None = None
    cpt_variability: float | None = None
    cpt_detectability: float | None = None
    cpt_perseverations: float | None = None
    dyslexia: bool | None = None
    dysorthographia: bool | None = None
    dyscalculia: bool | None = None
    dyspraxia: bool | None = None


@dataclass
class NonPharmTreatment:
    """Non-pharmacological treatment history."""
    has_non_pharm_lifetime: bool = False
    ect_lifetime: bool = False
    ect_sessions: int | None = None
    tms_lifetime: bool = False
    tms_sessions: int | None = None
    cbt_lifetime: bool = False
    ipsrt_lifetime: bool = False
    psychoeducation_lifetime: bool = False


@dataclass
class CurrentEpisodeCriteria:
    """DSM criteria for current depressive and manic episodes."""
    depressed_mood: bool | None = None
    anhedonia: bool | None = None
    weight_change: bool | None = None
    sleep_disturbance: bool | None = None
    psychomotor_change: bool | None = None
    fatigue: bool | None = None
    worthlessness: bool | None = None
    concentration_difficulty: bool | None = None
    suicidal_thoughts: bool | None = None
    depressive_symptom_count: int | None = None
    elevated_mood: bool | None = None
    irritable_mood: bool | None = None
    grandiosity: bool | None = None
    decreased_sleep_need: bool | None = None
    pressured_speech: bool | None = None
    flight_of_ideas: bool | None = None
    distractibility: bool | None = None
    goal_directed_activity: bool | None = None
    risky_behavior: bool | None = None
    manic_symptom_count: int | None = None


@dataclass
class MostRecentEpisode:
    """Characteristics of the most recent mood episode."""
    episode_type: str | None = None
    severity: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    chronicity: str | None = None
    postpartum: bool | None = None


# ═════════════════════════════════════════════════════════════════════════════
# CLINICAL UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════


def _get_clinical_constants():
    """Lazy accessor for loaded clinical constants YAML."""
    from face_rlvr.profiles.glossary_loader import get_clinical_constants
    return get_clinical_constants()


def compute_bmi_category(bmi: float | None) -> tuple[str, str] | None:
    """WHO BMI classification. Returns (code, label_fr) or None if missing.

    Thresholds loaded from config/glossary/common/clinical_constants.yaml.
    """
    if bmi is None:
        return None
    for band in _get_clinical_constants().bmi_categories:
        if band.max is None or bmi < band.max:
            return (band.code, band.label_fr)
    # Defensive fallback (should be unreachable — last band has max: null)
    last = _get_clinical_constants().bmi_categories[-1]
    return (last.code, last.label_fr)


def detect_metabolic_syndrome(
    waist_cm: float | None,
    sex: str | None,
    trig: float | None,
    hdl: float | None,
    sbp: float | None,
    dbp: float | None,
    glucose: float | None,
) -> tuple[bool, list[str]]:
    """IDF/ATP-III metabolic syndrome detection.

    Thresholds loaded from config/glossary/common/clinical_constants.yaml.
    Returns (is_metabolic_syndrome, list_of_met_criteria_fr).
    """
    ms = _get_clinical_constants().metabolic_syndrome
    criteria_met: list[str] = []

    # 1. Abdominal obesity
    if waist_cm is not None and sex is not None:
        waist_thresh = ms.abdominal_obesity_cm.M if sex == "M" else ms.abdominal_obesity_cm.F
        if waist_cm >= waist_thresh:
            criteria_met.append(f"Obésité abdominale (tour de taille {waist_cm:.0f} cm)")

    # 2. Hypertriglyceridemia
    if trig is not None and trig >= ms.hypertriglyceridemia_mmol_l:
        criteria_met.append(f"Hypertriglycéridémie ({trig:.2f} mmol/L)")

    # 3. Low HDL
    if hdl is not None and sex is not None:
        hdl_thresh = ms.hdl_low_mmol_l.M if sex == "M" else ms.hdl_low_mmol_l.F
        if hdl < hdl_thresh:
            criteria_met.append(f"HDL bas ({hdl:.2f} mmol/L)")

    # 4. Hypertension
    if sbp is not None and dbp is not None:
        if sbp >= ms.hypertension.sbp_mmhg or dbp >= ms.hypertension.dbp_mmhg:
            criteria_met.append(f"Hypertension ({sbp:.0f}/{dbp:.0f} mmHg)")

    # 5. Hyperglycemia
    if glucose is not None and glucose >= ms.hyperglycemia_mmol_l:
        criteria_met.append(f"Hyperglycémie à jeun ({glucose:.2f} mmol/L)")

    return (len(criteria_met) >= ms.minimum_criteria_met, criteria_met)


# Sex-specific laboratory reference ranges built from YAML at import time.
def _build_sex_specific_ranges() -> dict[str, dict[str, tuple[float, float]]]:
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for canonical, row in _get_clinical_constants().sex_specific_lab_ranges.items():
        mapping = {"M": tuple(row.M), "F": tuple(row.F)}
        out[canonical] = mapping
        for alias in row.aliases:
            out[alias] = mapping
    return out


SEX_SPECIFIC_RANGES: dict[str, dict[str, tuple[float, float]]] = _build_sex_specific_ranges()


def get_sex_specific_lab_range(
    lab_name: str, sex: str | None
) -> tuple[float, float] | None:
    """Return sex-specific normal range for labs with sex-dependent ranges."""
    ranges = SEX_SPECIFIC_RANGES.get(lab_name)
    if ranges is None or sex is None:
        return None
    return ranges.get(sex)


def _match_point_row(rows, value: float) -> int:
    """Walk through point rows top-down; return pts of first match.

    A row matches if value >= row.min (when min is set) and/or value <= row.max
    (when max is set). Returns 0 if no row matches.
    """
    for row in rows:
        min_v = row.min
        max_v = row.max
        if min_v is not None and value < min_v:
            continue
        if max_v is not None and value > max_v:
            continue
        return row.pts
    return 0


def compute_framingham_risk(
    age: int | None,
    sex: str | None,
    total_chol: float | None,
    hdl: float | None,
    sbp: float | None,
    on_bp_treatment: bool = False,
    smoking: bool = False,
    diabetes: bool = False,
) -> tuple[float | None, str]:
    """Simplified Framingham 10-year cardiovascular risk score.

    Point tables loaded from config/glossary/common/clinical_constants.yaml.
    Returns (risk_percent, category_fr).
    """
    if age is None or sex is None or total_chol is None or hdl is None or sbp is None:
        return (None, "Données insuffisantes")

    fr = _get_clinical_constants().framingham
    lo_age, hi_age = fr.age_bounds
    if age < lo_age or age > hi_age:
        return (None, f"Âge hors limites Framingham ({lo_age}-{hi_age})")

    risk_factors = 0

    # Age points (sex-specific)
    age_rows = fr.age_points.get(sex, [])
    risk_factors += _match_point_row(age_rows, age)

    # Cholesterol points
    risk_factors += _match_point_row(fr.cholesterol_points, total_chol)

    # HDL points
    risk_factors += _match_point_row(fr.hdl_points, hdl)

    # SBP points (treatment-dependent)
    sbp_rows = fr.sbp_points.treated if on_bp_treatment else fr.sbp_points.untreated
    risk_factors += _match_point_row(sbp_rows, sbp)

    # Smoking
    if smoking:
        risk_factors += fr.smoking_pts

    # Diabetes
    if diabetes:
        risk_factors += fr.diabetes_pts.get(sex, 0)

    # Risk table: find first row where risk_factors <= max_points
    risk_pct = 30.0
    for row in fr.risk_table:
        if row.max_points is None or risk_factors <= row.max_points:
            risk_pct = row.risk_pct
            break

    # Category: find first row where risk_pct < max_pct
    category = "Risque cardiovasculaire très élevé"
    for cat in fr.categories:
        if cat.max_pct is None or risk_pct < cat.max_pct:
            category = cat.label_fr
            break

    return (risk_pct, category)


# ─── Medication monitoring alerts ─────────────────────────────────────────────


def check_medication_lab_alerts(
    treatments: dict[str, Any],
    lab_values: list[LabValue],
    sex: str | None = None,
) -> list[str]:
    """Check lab values against medication-specific monitoring requirements.

    Rules are loaded from config/glossary/common/clinical_constants.yaml.

    Args:
        treatments: dict with medication flags (on_lithium, on_valproate, etc.)
        lab_values: list of LabValue objects from biology extraction
        sex: patient sex (currently unused in alert logic, kept for API compat)

    Returns:
        List of French alert strings for medication-lab interactions.
    """
    alerts: list[str] = []
    lab_dict = {lv.name: lv for lv in lab_values}

    for rule in _get_clinical_constants().medication_lab_alerts:
        # Treatment flag gating
        if rule.treatment_flag:
            if not treatments.get(rule.treatment_flag):
                continue
        elif rule.treatment_flags_any:
            if not any(treatments.get(flag) for flag in rule.treatment_flags_any):
                continue
        else:
            continue  # rule must have at least one treatment flag

        # Find first matching lab name
        for lab_name in rule.lab_names:
            lv = lab_dict.get(lab_name)
            if lv is None:
                continue

            matched = False
            if rule.absolute_threshold is not None:
                at = rule.absolute_threshold
                if at.max is not None and lv.value is not None and lv.value < at.max:
                    matched = True
                elif at.min is not None and lv.value is not None and lv.value > at.min:
                    matched = True
            elif rule.direction is not None:
                if lv.is_abnormal and lv.abnormality == rule.direction:
                    matched = True

            if matched:
                # Substitute {liver_name} placeholder for the valproate rule
                alert = rule.alert_fr.replace("{liver_name}", lab_name)
                alerts.append(alert)
                if rule.stop_after_first_match:
                    break

    return alerts


# ─── Drug-drug interaction flagging ──────────────────────────────────────────

# Backward-compat constant (built from YAML at import time)
def _build_drug_interactions():
    return [
        (r.drug1, r.drug2, r.severity, r.alert_fr)
        for r in _get_clinical_constants().drug_interactions
    ]

DRUG_INTERACTIONS: list[tuple[str, str, str, str]] = _build_drug_interactions()


def check_drug_interactions(treatment_flags: dict[str, Any]) -> list[str]:
    """Check for known critical drug-drug interactions.

    Rules loaded from config/glossary/common/clinical_constants.yaml.
    """
    cc = _get_clinical_constants()
    alerts: list[str] = []
    for rule in cc.drug_interactions:
        if treatment_flags.get(rule.drug1) and treatment_flags.get(rule.drug2):
            prefix = cc.severity_prefixes.get(rule.severity, rule.severity)
            alerts.append(f"[{prefix}] {rule.alert_fr}")
    return alerts


# ─── Reliable Change Index (RCI) ────────────────────────────────────────────


def compute_rci(
    v0_score: float | None,
    v1_score: float | None,
    se_measurement: float,
) -> tuple[float | None, str]:
    """Jacobson & Truax (1991) Reliable Change Index.

    Args:
        v0_score: baseline score
        v1_score: follow-up score
        se_measurement: standard error of measurement for the instrument

    Returns:
        (rci_value, interpretation_fr)
        |RCI| > 1.96 = reliable change at p < .05
    """
    if v0_score is None or v1_score is None or se_measurement <= 0:
        return (None, "Données insuffisantes pour le calcul du RCI")

    s_diff = math.sqrt(2 * se_measurement ** 2)
    if s_diff == 0:
        return (None, "Erreur de mesure nulle")

    rci = (v1_score - v0_score) / s_diff

    if abs(rci) > 1.96:
        if rci < 0:
            interp = f"Amélioration statistiquement fiable (RCI = {rci:.2f}, p < 0.05)"
        else:
            interp = f"Détérioration statistiquement fiable (RCI = {rci:.2f}, p < 0.05)"
    else:
        interp = f"Changement non statistiquement significatif (RCI = {rci:.2f})"

    return (rci, interp)


# ─── Quality control functions ───────────────────────────────────────────────


def detect_floor_ceiling_effects(
    scores: dict[str, ScoreInterpretation],
    instruments: dict[str, InstrumentDefinition],
) -> list[str]:
    """Detect instruments at minimum or maximum of their range.

    Returns list of French alert strings.
    """
    effects: list[str] = []
    for name, interp in scores.items():
        if not interp.score_available or interp.raw_score is None:
            continue
        inst = instruments.get(name)
        if inst is None:
            continue
        lo, hi = inst.score_range
        if interp.raw_score == lo:
            effects.append(f"{name} : effet plancher (score = {interp.raw_score})")
        elif interp.raw_score == hi:
            effects.append(f"{name} : effet plafond (score = {interp.raw_score})")
    return effects


def compute_data_completeness(
    score_dicts: dict[str, dict[str, ScoreInterpretation]],
) -> dict[str, float]:
    """Compute per-domain data completeness (proportion of available scores).

    Args:
        score_dicts: mapping domain_name -> {instrument_name -> ScoreInterpretation}

    Returns:
        mapping domain_name -> completeness ratio (0.0 to 1.0)
    """
    result: dict[str, float] = {}
    for domain, scores in score_dicts.items():
        total = len(scores)
        if total == 0:
            result[domain] = 0.0
            continue
        available = sum(1 for s in scores.values() if s.score_available)
        result[domain] = available / total
    return result


# ─── Cognitive normative tables (loaded from YAML) ──────────────────────────

def _build_cognitive_norms_dict(table_key: str) -> dict[int, dict[str, tuple[float, float]]]:
    cc = _get_clinical_constants()
    src = getattr(cc.cognitive_norms, table_key)["by_decade"]
    return {
        int(decade): {test: tuple(vals) for test, vals in tests.items()}
        for decade, tests in src.items()
    }

TMT_NORMS: dict[int, dict[str, tuple[float, float]]] = _build_cognitive_norms_dict("tmt")
STROOP_NORMS: dict[int, dict[str, tuple[float, float]]] = _build_cognitive_norms_dict("stroop")


def compute_cognitive_z_score(
    test: str,
    raw_score: float | None,
    age: int | None,
) -> tuple[float | None, str]:
    """Compute age-adjusted z-score for neuropsychological tests.

    Normative tables loaded from config/glossary/common/clinical_constants.yaml.
    """
    if raw_score is None or age is None:
        return (None, "Données insuffisantes")

    # Find age decade (clamped to [20, 70])
    age_decade = min(max((age // 10) * 10, 20), 70)

    # Look up norms
    norms = TMT_NORMS.get(age_decade, {}).get(test)
    if norms is None:
        norms = STROOP_NORMS.get(age_decade, {}).get(test)
    if norms is None:
        return (None, f"Normes non disponibles pour {test}")

    mean, sd = norms
    if sd == 0:
        return (None, "Écart-type nul")

    # For TMT, higher = worse, so z-score is reversed
    if test.startswith("TMT"):
        z = (mean - raw_score) / sd  # negative z = worse performance
    else:
        z = (raw_score - mean) / sd

    # Interpretation from YAML-loaded z_score_bands (first match wins on ascending max)
    label_fr = "Performance supérieure à la norme"
    for band in _get_clinical_constants().cognitive_norms.z_score_bands:
        if band.max is None or z <= band.max:
            label_fr = band.label_fr
            break
    interp = f"{label_fr} (z = {z:.1f})"

    return (z, interp)


# ─── Instrument score extraction helper ──────────────────────────────────────


def extract_instrument_scores(
    row: Any,
    instrument_names: list[str],
    instruments_dict: dict[str, InstrumentDefinition],
) -> dict[str, ScoreInterpretation]:
    """Extract and interpret scores for a list of instruments.

    Args:
        row: pandas Series (CSV row)
        instrument_names: list of instrument keys in instruments_dict
        instruments_dict: the cohort's instrument registry

    Returns:
        dict mapping instrument name to ScoreInterpretation
    """
    from face_rlvr.profiles.common_instruments import interpret_score as _interpret

    results: dict[str, ScoreInterpretation] = {}
    for name in instrument_names:
        inst = instruments_dict.get(name)
        if inst is None:
            continue
        raw = _safe_float(row.get(inst.total_column))
        subscales: dict[str, float | None] = {}
        for sub_name, sub_col in inst.subscale_columns.items():
            subscales[sub_name] = _safe_float(row.get(sub_col))
        results[name] = _interpret(inst, raw, subscales)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Shared sub-extractors used by multiple cohorts (BP/SZ/DR)
# ═════════════════════════════════════════════════════════════════════════════


def extract_substance_use(row, cm) -> SubstanceUse:
    """Extract substance use data using the cohort's column map.

    ``cm`` must be a ``CohortColumnMap`` (or at least expose ``cm.substance_use``).
    """
    su = cm.substance_use
    if su is None:
        return SubstanceUse()

    tobacco = _is_yes(row.get(su.tobacco)) if su.tobacco and _safe_str(row.get(su.tobacco)) else False
    alcohol = _is_yes(row.get(su.alcohol)) if su.alcohol else False
    cannabis = _is_yes(row.get(su.cannabis)) if su.cannabis else False

    others: list[str] = []
    for label, col in su.other_substances.items():
        if _is_yes(row.get(col)):
            others.append(label)

    has_sud = False
    if su.substance_use_disorder:
        sud = _safe_str(row.get(su.substance_use_disorder))
        has_sud = _is_yes(sud) if sud else False

    cpd = _safe_float(row.get(su.cigarettes_per_day)) if su.cigarettes_per_day else None

    return SubstanceUse(
        tobacco_current=tobacco,
        tobacco_cpd=cpd,
        alcohol_current=alcohol,
        alcohol_type=_safe_str(row.get(su.alcohol_type)) if su.alcohol_type else None,
        cannabis_current=cannabis,
        other_substances=others,
        substance_use_disorder=has_sud,
    )


def extract_suicide_indicators(row, cm) -> dict[str, Any]:
    """Extract suicide-related summary indicators using the cohort column map."""
    indicators: dict[str, Any] = {}
    si = cm.suicide_indicators
    if si is None:
        return indicators

    if si.madrs_item10:
        madrs10 = _safe_float(row.get(si.madrs_item10))
        if madrs10 is not None:
            indicators["madrs_suicide_item"] = madrs10
            indicators["madrs_suicide_elevated"] = madrs10 >= si.madrs_elevated_threshold

    if si.lifetime_attempts:
        v = _safe_float(row.get(si.lifetime_attempts))
        if v is not None:
            indicators["lifetime_suicide_attempts"] = int(v)

    if si.lifetime_self_harm:
        v = _safe_float(row.get(si.lifetime_self_harm))
        if v is not None:
            indicators["lifetime_self_harm"] = int(v)

    return indicators


def _col_yes_or_none(row, col: Optional[str]):
    """_is_yes(row[col]) if col is set AND value is present, else None."""
    if not col:
        return None
    if _safe_str(row.get(col)) is None:
        return None
    return _is_yes(row.get(col))


def extract_suicide_history(row, cm) -> SuicideHistory:
    """Extract detailed suicide history from ISF and Columbia instruments."""
    sh = cm.suicide_history
    if sh is None:
        return SuicideHistory()

    return SuicideHistory(
        ever_felt_life_not_worth=_col_yes_or_none(row, sh.ever_felt_life_not_worth),
        ever_wished_dead=_col_yes_or_none(row, sh.ever_wished_dead),
        ever_thought_suicide=_col_yes_or_none(row, sh.ever_thought_suicide),
        ever_planned_suicide=_col_yes_or_none(row, sh.ever_planned_suicide),
        ever_attempted=_col_yes_or_none(row, sh.ever_attempted),
        n_attempts=_safe_int(row.get(sh.n_attempts)) if sh.n_attempts else None,
        has_violent_attempts=_col_yes_or_none(row, sh.has_violent_attempts),
        n_violent_attempts=_safe_int(row.get(sh.n_violent_attempts)) if sh.n_violent_attempts else None,
        has_serious_attempts=_col_yes_or_none(row, sh.has_serious_attempts),
        n_serious_attempts=_safe_int(row.get(sh.n_serious_attempts)) if sh.n_serious_attempts else None,
        most_serious_method=_safe_str(row.get(sh.most_serious_method)) if sh.most_serious_method else None,
        most_violent_method=_safe_str(row.get(sh.most_violent_method)) if sh.most_violent_method else None,
        most_serious_trigger=_safe_str(row.get(sh.most_serious_trigger)) if sh.most_serious_trigger else None,
    )
