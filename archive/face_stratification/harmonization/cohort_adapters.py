"""Cohort adapters: ``PatientProfile`` → flat unified feature dict (V1 only).

Each ``adapt_<cohort>_profile`` function takes the cohort-specific
``*PatientData`` dataclass produced by ``face_rlvr.profiles.extract_*_patient``
and returns ``dict[feature_id, value]`` where:

- ``feature_id`` matches a ``UnifiedFeature.id`` from
  ``config/face_stratification/feature_schema.yaml``,
- missing / not-measured values are emitted as ``None`` (never raise),
- booleans are emitted as ``0.0`` / ``1.0`` so downstream numeric matrices stay
  homogeneous,
- only baseline-visit (V1) fields are read. BP's ``v1_mood_scores`` (the
  ``_n1`` follow-up) is **explicitly never touched**.

Adapters never import anything from ``face_rlvr.profiles`` beyond the
cohort-specific ``PatientData`` dataclass — they are pure functions over
already-extracted structures.

Design note
-----------
A single adapter is easier to audit than a metaclass / declarative mapper, and
changes to the feature schema require matching Python edits anyway. The
redundancy is intentional.
"""

from __future__ import annotations

from typing import Any

from face_rlvr.profiles import (
    ASPPatientData,
    BPPatientData,
    DRPatientData,
    ScoreInterpretation,
    SZPatientData,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _b(val: bool | None) -> float | None:
    """Coerce an optional bool to 0.0/1.0/None (for numeric matrices)."""
    if val is None:
        return None
    return 1.0 if bool(val) else 0.0


def _f(val: float | int | None) -> float | None:
    """Coerce optional numeric to float (or None)."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _score(scores: dict[str, ScoreInterpretation] | None, key: str) -> float | None:
    """Pull a raw score out of a dict of ``ScoreInterpretation`` objects.

    Returns ``None`` if the dict is missing, the key is absent, or the score
    was flagged as unavailable / suspect.
    """
    if not scores:
        return None
    s = scores.get(key)
    if s is None or not s.score_available or s.suspect_value:
        return None
    return _f(s.raw_score)


def _subscale(
    scores: dict[str, ScoreInterpretation] | None,
    instrument_key: str,
    subscale_name: str,
    *,
    ignore_suspect: bool = False,
) -> float | None:
    """Pull a sub-scale value from a ScoreInterpretation's subscales dict.

    Returns None if the dict is missing, the key is absent, the score
    was flagged as unavailable, or the subscale is not present.

    When *ignore_suspect* is True, the ``suspect_value`` flag on the
    **total** score is bypassed.  Use this for instruments whose domain
    subscales are independently valid even when the total trips the
    suspect check (e.g. ADI-R, BRIEF, RBS-R).
    """
    if not scores:
        return None
    s = scores.get(instrument_key)
    if s is None or not s.score_available:
        return None
    if s.suspect_value and not ignore_suspect:
        return None
    if not s.subscales:
        return None
    val = s.subscales.get(subscale_name)
    return _f(val)


def _education_ordinal(level: str | None) -> str | None:
    """Map the French education label from glossary to a canonical ordinal token."""
    if level is None:
        return None
    norm = level.strip().lower()
    # Keep coarse categories that match allowed_values in the schema.
    if "aucun" in norm or norm == "none":
        return "none"
    if "primaire" in norm:
        return "primaire"
    if "collège" in norm or "college" in norm:
        return "college"
    if "lycée" in norm or "lycee" in norm:
        return "lycee"
    if "bac+2" in norm or "bac + 2" in norm or "bts" in norm or "dut" in norm:
        return "bac_plus_2"
    if "bac+5" in norm or "bac + 5" in norm or "master" in norm or "doctorat" in norm:
        return "bac_plus_5_plus"
    if "bac" in norm:
        return "bac"
    return None


def _marital_partnered(status: str | None) -> bool | None:
    if status is None:
        return None
    norm = status.strip().lower()
    if not norm:
        return None
    partnered_markers = ("marié", "marie", "concubin", "pacs", "union", "couple")
    return any(m in norm for m in partnered_markers)


def _employed(status: str | None) -> bool | None:
    if status is None:
        return None
    norm = status.strip().lower()
    if not norm:
        return None
    if "activité" in norm or "actif" in norm or "emploi" in norm or "travail" in norm:
        return True
    if "sans" in norm or "chômage" in norm or "chomage" in norm or "inactif" in norm:
        return False
    return None


def _get_lab(biology: Any, *names: str) -> float | None:
    """Look up a lab value by canonical English name in a ``BiologicalPanel``.

    ``face_rlvr`` stores labs as a list of ``LabValue`` objects with a ``name``
    field; we match case-insensitively and return ``None`` if none match.
    """
    if biology is None or not getattr(biology, "values", None):
        return None
    wanted = {n.lower() for n in names}
    for lab in biology.values:
        lname = (lab.name or "").lower()
        if lname in wanted:
            return _f(lab.value)
    return None


def _count_items(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return len(value)
    except TypeError:
        return None


def _compute_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Compute a safe ratio, returning None if either value is missing or denominator is zero."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def _fh_any_relative(fam: Any, attr: str) -> bool | None:
    """Any relative flags ``attr`` in a ``FamilyHistory``."""
    if fam is None:
        return None
    parental = None
    for parent_attr in (f"maternal_{attr.split('_')[0]}", f"paternal_{attr.split('_')[0]}"):
        v = getattr(fam, parent_attr, None)
        if v is not None:
            parental = bool(parental) or bool(v)
    rel_flag = any(getattr(r, attr, False) for r in getattr(fam, "relatives", []) or [])
    if parental is None and not rel_flag:
        return None
    return bool(parental) or bool(rel_flag)


def _fh_bipolar(fam: Any) -> bool | None:
    if fam is None:
        return None
    flag = getattr(fam, "family_bipolar", None)
    if flag:
        return True
    # fall back to parental psychiatric string mentions
    for parental_attr in ("maternal_psychiatric", "paternal_psychiatric"):
        s = getattr(fam, parental_attr, None)
        if s and "bipolair" in s.lower():
            return True
    return flag if flag is not None else None


def _fh_suicide(fam: Any) -> bool | None:
    if fam is None:
        return None
    parts = [
        getattr(fam, "maternal_suicide", None),
        getattr(fam, "paternal_suicide", None),
    ]
    rel = any(getattr(r, "suicide", False) for r in getattr(fam, "relatives", []) or [])
    if all(p is None for p in parts) and not rel:
        return None
    return any(p for p in parts) or rel


def _fh_substance(fam: Any) -> bool | None:
    if fam is None:
        return None
    parts = [
        getattr(fam, "maternal_substance", None),
        getattr(fam, "paternal_substance", None),
    ]
    rel = any(getattr(r, "substance_use", False) for r in getattr(fam, "relatives", []) or [])
    if all(p is None for p in parts) and not rel:
        return None
    return any(p for p in parts) or rel


def _fh_n_affected(fam: Any) -> int | None:
    """Rough count of affected relatives across siblings, children, and extended pedigree."""
    if fam is None:
        return None
    parts: list[int] = []
    if fam.n_siblings_affected is not None:
        parts.append(int(fam.n_siblings_affected))
    if fam.n_children_affected is not None:
        parts.append(int(fam.n_children_affected))
    parts.append(sum(1 for r in fam.relatives or [] if r.psychiatric_disorder))
    if not parts:
        return None
    return sum(parts)


# ─── Derived feature helpers ──────────────────────────────────────────────────

import math


def _polypharmacy_index(*flags: float | None) -> float | None:
    """Count non-None True (1.0) treatment flags."""
    present = [f for f in flags if f is not None]
    if not present:
        return None
    return float(sum(1 for f in present if f == 1.0))


def _onset_category(age_first: float | None) -> float | None:
    """Ordinal onset category: 0=early (<18), 1=typical (18-25), 2=late (>25)."""
    if age_first is None:
        return None
    if age_first < 18:
        return 0.0
    elif age_first <= 25:
        return 1.0
    else:
        return 2.0


def _illness_burden(
    duration: float | None, n_episodes: float | None, n_hosp: float | None,
) -> float | None:
    """log1p(duration * episodes * (1 + hospitalizations)). Log-scaled cumulative burden."""
    if duration is None:
        return None
    eps = n_episodes if n_episodes is not None else 1.0
    hosp = n_hosp if n_hosp is not None else 0.0
    raw = duration * eps * (1.0 + hosp)
    if raw < 0:
        return None
    return math.log1p(raw)


def _waist_height_ratio(bio: Any) -> float | None:
    """Waist circumference / height from vitals dict."""
    if bio is None or not bio.vitals:
        return None
    waist = bio.vitals.get("waist_cm") or bio.vitals.get("waist")
    height = bio.vitals.get("height_cm") or bio.vitals.get("height")
    if waist is None or height is None:
        return None
    w = _f(waist)
    h = _f(height)
    if w is None or h is None or h <= 0:
        return None
    return w / h


# ─── BP adapter ───────────────────────────────────────────────────────────────


def adapt_bp_profile(data: BPPatientData) -> dict[str, float | None]:
    """Map a fully-extracted BP patient to the unified feature dict (V1 only).

    BP's ``v1_mood_scores`` attribute is **intentionally never touched** — that
    is the ``_n1`` follow-up and is out of scope for this sub-project.
    """
    demo = data.demographics
    bio = data.biology
    tx = data.treatments
    fam = data.family_history
    sub = data.substance_use
    sui = data.suicide_history
    hist = data.psychiatric_history
    cog = data.cognitive_profile

    out: dict[str, float | None] = {
        # Demographics
        "demo_age_years": _f(demo.age),
        "demo_sex_male": 1.0 if demo.sex == "M" else (0.0 if demo.sex == "F" else None),
        "demo_education_years_ordinal": _encode_ordinal(
            _education_ordinal(demo.education_level)
        ),
        "demo_marital_partnered": _b(_marital_partnered(demo.marital_status)),
        "demo_employed": _b(_employed(demo.employment)),
        # Mood instruments
        "inst_madrs_total": _score(data.mood_scores, "MADRS"),
        "inst_ymrs_total": _score(data.mood_scores, "YMRS"),
        "inst_cgis_total": _score(data.mood_scores, "CGI-S"),
        "inst_qids_total": _score(data.mood_scores, "QIDS-SR16"),
        "inst_mathys_total": _score(data.mood_scores, "MAThyS"),
        "inst_asrm_total": _score(data.mood_scores, "ASRM"),
        # Anxiety / impulsivity
        "inst_stai_ya_total": _score(data.anxiety_impulsivity_scores, "STAI-YA"),
        "inst_bis10_total": _score(data.anxiety_impulsivity_scores, "BIS-10"),
        "inst_bdhi_total": _score(data.anxiety_impulsivity_scores, "BDHI"),
        "inst_als_total": _score(data.anxiety_impulsivity_scores, "ALS"),
        # Functioning
        "inst_fast_total": _score(data.functional_scores, "FAST"),
        "inst_eq5d_total": _score(data.functional_scores, "EQ-5D"),
        # Sleep / circadian
        "inst_psqi_total": _score(data.sleep_scores, "PSQI"),
        "inst_ess_total": _score(data.sleep_scores, "ESS"),
        "inst_csm_total": _score(data.sleep_scores, "CSM"),
        # Cognition
        "cog_tmt_a_seconds": _f(cog.tmt_a_seconds),
        "cog_tmt_b_seconds": _f(cog.tmt_b_seconds),
        "cog_stroop_interference": _f(cog.stroop_interference),
        "cog_cvlt_total_learning": _f(cog.cvlt_total_learning),
        "cog_cvlt_long_delay_free": _f(cog.cvlt_long_delay_free),
        "cog_phonemic_fluency": _f(cog.phonemic_fluency),
        "cog_semantic_fluency": _f(cog.semantic_fluency),
        "cog_wais_similarities_std": _f(cog.wais_similarities_std),
        "cog_wais_vocabulary_std": _f(cog.wais_vocabulary_std),
        "cog_wais_working_memory_std": _f(cog.wais_working_memory_std),
        # Biology
        "bio_bmi": _f(bio.vitals.get("bmi")),
        "bio_waist_cm": _f(bio.vitals.get("waist_cm")),
        "bio_sbp_mmhg": _f(bio.vitals.get("sbp_supine")),
        "bio_dbp_mmhg": _f(bio.vitals.get("dbp_supine")),
        "bio_hr_bpm": _f(bio.vitals.get("hr_supine")),
        "bio_qtc_ms": _f(bio.ecg.get("qtc") if bio.ecg else None),
        # Treatment (BP-specific profile)
        "tx_on_antidepressant": _b(tx.on_antidepressant),
        "tx_on_antipsychotic": _b(tx.on_antipsychotic),
        "tx_on_mood_stabilizer": _b(tx.on_thymoregulator or tx.on_lithium or tx.on_valproate),
        "tx_on_benzodiazepine": _b(tx.on_benzodiazepine),
        "tx_on_lithium": _b(tx.on_lithium),
        "inst_mars_total": _score(data.adherence_scores, "MARS"),
        # Substance use
        "sub_tobacco_current": _b(sub.tobacco_current),
        "sub_tobacco_cpd": _f(sub.tobacco_cpd),
        "sub_alcohol_current": _b(sub.alcohol_current),
        "sub_cannabis_current": _b(sub.cannabis_current),
        "sub_use_disorder": _b(sub.substance_use_disorder),
        # Trauma
        "inst_ctq_total": _score(data.trauma_scores, "CTQ"),
        # Family history
        "fh_bipolar_any": _b(_fh_bipolar(fam)),
        "fh_suicide_any": _b(_fh_suicide(fam)),
        "fh_substance_any": _b(_fh_substance(fam)),
        "fh_n_affected_relatives": _f(_fh_n_affected(fam)),
        # Comorbidities
        "cm_n_somatic": _f(_count_items(data.somatic_comorbidities)),
        "cm_n_psychiatric": _f(_count_items(data.psychiatric_comorbidities)),
        # Suicide history
        "sui_ever_ideation": _b(sui.ever_thought_suicide),
        "sui_ever_attempt": _b(sui.ever_attempted),
        "sui_n_attempts": _f(sui.n_attempts),
        "sui_any_violent_attempt": _b(sui.has_violent_attempts),
        # Psychiatric history
        "psyh_age_first_episode": _f(hist.age_first_episode),
        "psyh_illness_duration_years": _f(hist.illness_duration_years),
        "psyh_n_depressive_episodes_lifetime": _f(hist.n_depressive_episodes_lifetime),
        "psyh_n_manic_episodes_lifetime": _f(hist.n_manic_episodes_lifetime),
        "psyh_rapid_cycling": _b(hist.rapid_cycling),
        "psyh_n_hospitalizations_lifetime": _f(data.hospitalization.n_hospitalizations_lifetime),
        # ── Sub-scales ──
        # CTQ sub-scales (French keys from glossary YAML)
        "inst_ctq_emotional_abuse": _subscale(data.trauma_scores, "CTQ", "Abus émotionnel"),
        "inst_ctq_physical_abuse": _subscale(data.trauma_scores, "CTQ", "Abus physique"),
        "inst_ctq_sexual_abuse": _subscale(data.trauma_scores, "CTQ", "Abus sexuel"),
        "inst_ctq_emotional_neglect": _subscale(data.trauma_scores, "CTQ", "Négligence émotionnelle"),
        "inst_ctq_physical_neglect": _subscale(data.trauma_scores, "CTQ", "Négligence physique"),
        # BIS-10 sub-scales
        "inst_bis10_attentional": _subscale(data.anxiety_impulsivity_scores, "BIS-10", "Impulsivité attentionnelle"),
        "inst_bis10_motor": _subscale(data.anxiety_impulsivity_scores, "BIS-10", "Impulsivité motrice"),
        "inst_bis10_nonplanning": _subscale(data.anxiety_impulsivity_scores, "BIS-10", "Impulsivité de planification"),
        # BDHI 9 sub-scales (hostility dimensions)
        "inst_bdhi_assault": _subscale(data.anxiety_impulsivity_scores, "BDHI", "Agressivité (Assault)"),
        "inst_bdhi_indirect_hostility": _subscale(data.anxiety_impulsivity_scores, "BDHI", "Hostilité indirecte"),
        "inst_bdhi_irritability": _subscale(data.anxiety_impulsivity_scores, "BDHI", "Irritabilité"),
        "inst_bdhi_negativism": _subscale(data.anxiety_impulsivity_scores, "BDHI", "Négativisme"),
        "inst_bdhi_resentment": _subscale(data.anxiety_impulsivity_scores, "BDHI", "Ressentiment"),
        "inst_bdhi_suspicion": _subscale(data.anxiety_impulsivity_scores, "BDHI", "Suspicion"),
        "inst_bdhi_verbal_hostility": _subscale(data.anxiety_impulsivity_scores, "BDHI", "Hostilité verbale"),
        "inst_bdhi_guilt": _subscale(data.anxiety_impulsivity_scores, "BDHI", "Culpabilité"),
        "inst_bdhi_attitudinal": _subscale(data.anxiety_impulsivity_scores, "BDHI", "Composante attitudinale"),
        # Cognition derived
        "cog_tmt_b_minus_a": _f(cog.tmt_b_minus_a),
        "cog_tmt_ratio_ba": _f(cog.tmt_ratio_ba),
        "cog_stroop_word": _f(cog.stroop_word),
        "cog_stroop_color": _f(cog.stroop_color),
        "cog_stroop_color_word": _f(cog.stroop_color_word),
        "cog_cvlt_short_delay_free": _f(cog.cvlt_short_delay_free),
        "cog_cvlt_recognition": _f(cog.cvlt_recognition),
        # AdditionalNeuropsych
        "np_wais_matrices_std": _f(data.additional_neuropsych.matrices_std) if data.additional_neuropsych else None,
        "np_wais_code_std": _f(data.additional_neuropsych.code_std) if data.additional_neuropsych else None,
        "np_wais_symbol_std": _f(data.additional_neuropsych.symbol_std) if data.additional_neuropsych else None,
        "np_digit_span_forward_std": _f(data.additional_neuropsych.digit_span_forward_std) if data.additional_neuropsych else None,
        "np_digit_span_backward_std": _f(data.additional_neuropsych.digit_span_backward_std) if data.additional_neuropsych else None,
        "np_digit_span_total_std": _f(data.additional_neuropsych.digit_span_total_std) if data.additional_neuropsych else None,
        "np_cpt_omissions": _f(data.additional_neuropsych.cpt_omissions) if data.additional_neuropsych else None,
        "np_cpt_commissions": _f(data.additional_neuropsych.cpt_commissions) if data.additional_neuropsych else None,
        "np_cpt_hit_rt": _f(data.additional_neuropsych.cpt_hit_rt) if data.additional_neuropsych else None,
        "np_cpt_variability": _f(data.additional_neuropsych.cpt_variability) if data.additional_neuropsych else None,
        # Treatment plasma levels
        "tx_lithium_level": _f(tx.lithium_level if hasattr(tx, 'lithium_level') else None),
        "tx_valproate_level": _f(tx.valproate_level if hasattr(tx, 'valproate_level') else None),
    }
    # Derived composites (must be computed after dict is built)
    out["tx_polypharmacy_index"] = _polypharmacy_index(
        out.get("tx_on_antidepressant"), out.get("tx_on_antipsychotic"),
        out.get("tx_on_mood_stabilizer"), out.get("tx_on_benzodiazepine"),
        out.get("tx_on_lithium"),
    )
    out["bio_waist_height_ratio"] = _waist_height_ratio(bio)
    out["psyh_onset_category"] = _onset_category(out.get("psyh_age_first_episode"))
    n_episodes_bp = (
        _f((hist.n_depressive_episodes_lifetime or 0) + (hist.n_manic_episodes_lifetime or 0))
        if hist.n_depressive_episodes_lifetime is not None or hist.n_manic_episodes_lifetime is not None
        else None
    )
    out["psyh_illness_burden"] = _illness_burden(
        out.get("psyh_illness_duration_years"), n_episodes_bp,
        out.get("psyh_n_hospitalizations_lifetime"),
    )
    return out


# ─── SZ adapter ───────────────────────────────────────────────────────────────


def adapt_sz_profile(data: SZPatientData) -> dict[str, float | None]:
    demo = data.demographics
    bio = data.biology
    tx = data.treatments
    fam = data.family_history
    sub = data.substance_use
    sui = data.suicide_history
    cog = data.cognitive_profile

    # SUMD mean insight — average over non-None items
    sumd_items = [
        getattr(data.insight, f, None)
        for f in (
            "awareness_of_illness",
            "awareness_of_medication_effect",
            "awareness_of_social_consequences",
            "awareness_of_hallucinations",
            "awareness_of_delusions",
            "awareness_of_thought_disorder",
            "awareness_of_flat_affect",
            "awareness_of_anhedonia",
            "awareness_of_asociality",
        )
    ]
    sumd_vals = [v for v in sumd_items if v is not None]
    sumd_mean = sum(sumd_vals) / len(sumd_vals) if sumd_vals else None

    out: dict[str, float | None] = {
        # Demographics
        "demo_age_years": _f(demo.age),
        "demo_sex_male": 1.0 if demo.sex == "M" else (0.0 if demo.sex == "F" else None),
        "demo_education_years_ordinal": _encode_ordinal(
            _education_ordinal(demo.education_level)
        ),
        "demo_marital_partnered": _b(_marital_partnered(demo.marital_status)),
        "demo_employed": _b(_employed(demo.employment)),
        # Mood instruments (SZ has YMRS + Calgary + CGI-S)
        "inst_ymrs_total": _score(data.mood_scores, "YMRS"),
        "inst_cgis_total": _score(data.global_scores, "CGI-S"),
        "inst_calgary_total": _score(data.depression_scores, "Calgary"),
        # Psychosis
        "inst_panss_total": _score(data.psychosis_scores, "PANSS"),
        "inst_panss_p": _score(data.psychosis_scores, "PANSS-P"),
        "inst_panss_n": _score(data.psychosis_scores, "PANSS-N"),
        "inst_panss_g": _score(data.psychosis_scores, "PANSS-G"),
        "inst_aims_total": _score(data.movement_scores, "AIMS"),
        "inst_bars_total": _score(data.movement_scores, "BARS"),
        # Functioning
        "inst_psp_total": _score(data.functioning_scores, "PSP"),
        "inst_eq5d_total": _score(data.functioning_scores, "EQ-5D"),
        # Sleep
        "inst_psqi_total": _score(data.sleep_scores, "PSQI"),
        # Cognition
        "cog_tmt_a_seconds": _f(cog.tmt_a_seconds),
        "cog_tmt_b_seconds": _f(cog.tmt_b_seconds),
        "cog_stroop_interference": _f(cog.stroop_interference),
        "cog_cvlt_total_learning": _f(cog.cvlt_total_learning),
        "cog_cvlt_long_delay_free": _f(cog.cvlt_long_delay_free),
        "cog_phonemic_fluency": _f(cog.phonemic_fluency),
        "cog_semantic_fluency": _f(cog.semantic_fluency),
        "cog_wais_similarities_std": _f(cog.wais_similarities_std),
        "cog_wais_vocabulary_std": _f(cog.wais_vocabulary_std),
        "cog_wais_working_memory_std": _f(cog.wais_working_memory_std),
        # Biology
        "bio_bmi": _f(bio.vitals.get("bmi")),
        "bio_fasting_glucose": _get_lab(bio, "glucose", "fasting_glucose", "glycémie"),
        "bio_total_cholesterol": _get_lab(bio, "cholesterol", "total_cholesterol", "cholestérol total"),
        "bio_hdl_cholesterol": _get_lab(bio, "hdl", "hdl cholesterol"),
        "bio_triglycerides": _get_lab(bio, "triglycerides", "triglycérides"),
        # Treatment
        "tx_on_antidepressant": _f(None if tx.n_antidepressants is None else (1.0 if tx.n_antidepressants > 0 else 0.0)),
        "tx_on_antipsychotic": _f(None if tx.n_antipsychotics is None else (1.0 if tx.n_antipsychotics > 0 else 0.0)),
        "tx_on_mood_stabilizer": _f(None if tx.n_mood_stabilizers is None else (1.0 if tx.n_mood_stabilizers > 0 else 0.0)),
        "tx_on_clozapine": _b(tx.on_clozapine),
        "inst_mars_total": _score(data.adherence_scores, "MARS"),
        # Substance use
        "sub_tobacco_current": _b(sub.tobacco_current),
        "sub_tobacco_cpd": _f(sub.tobacco_cpd),
        "sub_alcohol_current": _b(sub.alcohol_current),
        "sub_cannabis_current": _b(sub.cannabis_current),
        "sub_use_disorder": _b(sub.substance_use_disorder),
        # Trauma
        "inst_ctq_total": _score(data.trauma_scores, "CTQ"),
        # Family history
        "fh_bipolar_any": _b(_fh_bipolar(fam)),
        "fh_suicide_any": _b(_fh_suicide(fam)),
        "fh_substance_any": _b(_fh_substance(fam)),
        "fh_n_affected_relatives": _f(_fh_n_affected(fam)),
        # Comorbidities
        "cm_n_somatic": _f(_count_items(data.somatic_comorbidities)),
        "cm_n_psychiatric": _f(_count_items(data.psychiatric_comorbidities)),
        # Suicide history
        "sui_ever_ideation": _b(sui.ever_thought_suicide),
        "sui_ever_attempt": _b(sui.ever_attempted),
        "sui_n_attempts": _f(sui.n_attempts),
        "sui_any_violent_attempt": _b(sui.has_violent_attempts),
        # Psychiatric history
        "psyh_age_first_episode": _f(data.psychotic_history.age_onset_sz),
        "psyh_illness_duration_years": (
            _f(_safe_illness_duration(demo.age, data.psychotic_history.age_onset_sz))
        ),
        "psyh_n_hospitalizations_lifetime": _f(data.hospitalization.n_hospitalizations_lifetime),
        # Cohort-specific
        "sz_insight_sumd_mean": _f(sumd_mean),
        # PANSS Wallwork 5-factor model
        "inst_panss_wallwork_positive": _f(data.panss_wallwork_positive),
        "inst_panss_wallwork_negative": _f(data.panss_wallwork_negative),
        "inst_panss_wallwork_disorganized": _f(data.panss_wallwork_disorganized),
        "inst_panss_wallwork_excited": _f(data.panss_wallwork_excited),
        "inst_panss_wallwork_depressed": _f(data.panss_wallwork_depressed),
        # ── Sub-scales ──
        # CTQ sub-scales (French keys from glossary YAML)
        "inst_ctq_emotional_abuse": _subscale(data.trauma_scores, "CTQ", "Abus émotionnel"),
        "inst_ctq_physical_abuse": _subscale(data.trauma_scores, "CTQ", "Abus physique"),
        "inst_ctq_sexual_abuse": _subscale(data.trauma_scores, "CTQ", "Abus sexuel"),
        "inst_ctq_emotional_neglect": _subscale(data.trauma_scores, "CTQ", "Négligence émotionnelle"),
        "inst_ctq_physical_neglect": _subscale(data.trauma_scores, "CTQ", "Négligence physique"),
        # Cognition derived
        "cog_tmt_b_minus_a": _f(cog.tmt_b_minus_a),
        "cog_tmt_ratio_ba": _f(cog.tmt_ratio_ba),
        "cog_stroop_word": _f(cog.stroop_word),
        "cog_stroop_color": _f(cog.stroop_color),
        "cog_stroop_color_word": _f(cog.stroop_color_word),
        "cog_cvlt_short_delay_free": _f(cog.cvlt_short_delay_free),
        "cog_cvlt_recognition": _f(cog.cvlt_recognition),
        # AdditionalNeuropsych
        "np_wais_matrices_std": _f(data.additional_neuropsych.matrices_std) if data.additional_neuropsych else None,
        "np_wais_code_std": _f(data.additional_neuropsych.code_std) if data.additional_neuropsych else None,
        "np_wais_symbol_std": _f(data.additional_neuropsych.symbol_std) if data.additional_neuropsych else None,
        "np_digit_span_forward_std": _f(data.additional_neuropsych.digit_span_forward_std) if data.additional_neuropsych else None,
        "np_digit_span_backward_std": _f(data.additional_neuropsych.digit_span_backward_std) if data.additional_neuropsych else None,
        "np_digit_span_total_std": _f(data.additional_neuropsych.digit_span_total_std) if data.additional_neuropsych else None,
        "np_cpt_omissions": _f(data.additional_neuropsych.cpt_omissions) if data.additional_neuropsych else None,
        "np_cpt_commissions": _f(data.additional_neuropsych.cpt_commissions) if data.additional_neuropsych else None,
        "np_cpt_hit_rt": _f(data.additional_neuropsych.cpt_hit_rt) if data.additional_neuropsych else None,
        "np_cpt_variability": _f(data.additional_neuropsych.cpt_variability) if data.additional_neuropsych else None,
        # SUMD individual items (9 items total)
        "sz_sumd_awareness_illness": _f(data.insight.awareness_of_illness) if data.insight else None,
        "sz_sumd_awareness_medication": _f(data.insight.awareness_of_medication_effect) if data.insight else None,
        "sz_sumd_awareness_social": _f(data.insight.awareness_of_social_consequences) if data.insight else None,
        "sz_sumd_awareness_hallucinations": _f(data.insight.awareness_of_hallucinations) if data.insight else None,
        "sz_sumd_awareness_delusions": _f(data.insight.awareness_of_delusions) if data.insight else None,
        "sz_sumd_awareness_thought_disorder": _f(data.insight.awareness_of_thought_disorder) if data.insight else None,
        "sz_sumd_awareness_flat_affect": _f(data.insight.awareness_of_flat_affect) if data.insight else None,
        "sz_sumd_awareness_anhedonia": _f(data.insight.awareness_of_anhedonia) if data.insight else None,
        "sz_sumd_awareness_asociality": _f(data.insight.awareness_of_asociality) if data.insight else None,
        # Treatment plasma levels
        "tx_clozapine_level": _f(tx.clozapine_level if hasattr(tx, 'clozapine_level') else None),
        # Bio derived ratio
        "bio_tg_hdl_ratio": _compute_ratio(
            _get_lab(bio, "triglycerides", "triglycérides"),
            _get_lab(bio, "hdl", "hdl cholesterol"),
        ),
    }
    # Derived composites
    out["tx_polypharmacy_index"] = _polypharmacy_index(
        out.get("tx_on_antidepressant"), out.get("tx_on_antipsychotic"),
        out.get("tx_on_mood_stabilizer"), out.get("tx_on_clozapine"),
    )
    out["psyh_onset_category"] = _onset_category(out.get("psyh_age_first_episode"))
    out["psyh_illness_burden"] = _illness_burden(
        out.get("psyh_illness_duration_years"), None,
        out.get("psyh_n_hospitalizations_lifetime"),
    )
    return out


def _safe_illness_duration(age: int | None, age_onset: int | None) -> float | None:
    if age is None or age_onset is None:
        return None
    dur = age - age_onset
    return float(dur) if dur >= 0 else None


# ─── DR adapter ───────────────────────────────────────────────────────────────


def adapt_dr_profile(data: DRPatientData) -> dict[str, float | None]:
    demo = data.demographics
    bio = data.biology
    tx = data.treatments
    fam = data.family_history
    sub = data.substance_use
    sui = data.suicide_history
    hist = data.psychiatric_history
    tr = data.treatment_resistance
    cog = data.cognitive_profile

    out: dict[str, float | None] = {
        # Demographics
        "demo_age_years": _f(demo.age),
        "demo_sex_male": 1.0 if demo.sex == "M" else (0.0 if demo.sex == "F" else None),
        "demo_education_years_ordinal": _encode_ordinal(
            _education_ordinal(demo.education_level)
        ),
        "demo_marital_partnered": _b(_marital_partnered(demo.marital_status)),
        "demo_employed": _b(_employed(demo.employment)),
        # Mood
        "inst_madrs_total": _score(data.depression_scores, "MADRS"),
        "inst_ymrs_total": _score(data.mood_scores, "YMRS"),
        "inst_cgis_total": _score(data.global_scores, "CGI-S"),
        "inst_qids_total": _score(data.depression_scores, "QIDS"),
        "inst_mathys_total": _score(data.mood_scores, "MAThyS"),
        "inst_shaps_total": _score(data.depression_scores, "SHAPS"),
        # Anxiety / impulsivity
        "inst_stai_ya_total": _score(data.anxiety_scores, "STAI-YA"),
        "inst_bis10_total": _score(data.impulsivity_scores, "BIS-10"),
        # Functioning
        "inst_fast_total": _score(data.functioning_scores, "FAST"),
        "inst_egf_total": _score(data.functioning_scores, "EGF"),
        # Sleep
        "inst_psqi_total": _score(data.sleep_scores, "PSQI"),
        "inst_ess_total": _score(data.sleep_scores, "ESS"),
        "inst_csm_total": _score(data.screening_scores, "CSM"),
        # Cognition
        "cog_tmt_a_seconds": _f(cog.tmt_a_seconds),
        "cog_tmt_b_seconds": _f(cog.tmt_b_seconds),
        "cog_stroop_interference": _f(cog.stroop_interference),
        "cog_cvlt_total_learning": _f(cog.cvlt_total_learning),
        "cog_cvlt_long_delay_free": _f(cog.cvlt_long_delay_free),
        "cog_phonemic_fluency": _f(cog.phonemic_fluency),
        "cog_semantic_fluency": _f(cog.semantic_fluency),
        "cog_wais_similarities_std": _f(cog.wais_similarities_std),
        "cog_wais_vocabulary_std": _f(cog.wais_vocabulary_std),
        "cog_wais_working_memory_std": _f(cog.wais_working_memory_std),
        # Biology
        "bio_bmi": _f(bio.vitals.get("bmi")),
        "bio_waist_cm": _f(bio.vitals.get("waist_cm")),
        "bio_sbp_mmhg": _f(bio.vitals.get("sbp_supine")),
        "bio_dbp_mmhg": _f(bio.vitals.get("dbp_supine")),
        "bio_fasting_glucose": _get_lab(bio, "glucose", "fasting_glucose", "glycémie"),
        "bio_total_cholesterol": _get_lab(bio, "cholesterol", "total_cholesterol", "cholestérol total"),
        "bio_hdl_cholesterol": _get_lab(bio, "hdl", "hdl cholesterol"),
        "bio_triglycerides": _get_lab(bio, "triglycerides", "triglycérides"),
        # Treatment
        "tx_on_antidepressant": None,  # not extracted directly in DRTreatmentProfile
        "tx_on_antipsychotic": None,
        "tx_on_mood_stabilizer": _b(tx.lithium_level is not None or tx.valproate_level is not None),
        "tx_on_lithium": _b(tx.lithium_level is not None),
        "inst_mars_total": _score(data.adherence_scores, "MARS"),
        # Substance use
        "sub_tobacco_current": _b(sub.tobacco_current),
        "sub_tobacco_cpd": _f(sub.tobacco_cpd),
        "sub_alcohol_current": _b(sub.alcohol_current),
        "sub_cannabis_current": _b(sub.cannabis_current),
        "sub_use_disorder": _b(sub.substance_use_disorder),
        # Trauma
        "inst_ctq_total": _score(data.trauma_scores, "CTQ"),
        "inst_pcl5_total": _score(data.trauma_scores, "PCL-5"),
        # Family history
        "fh_bipolar_any": _b(_fh_bipolar(fam)),
        "fh_suicide_any": _b(_fh_suicide(fam)),
        "fh_substance_any": _b(_fh_substance(fam)),
        "fh_n_affected_relatives": _f(_fh_n_affected(fam)),
        # Comorbidities
        "cm_n_somatic": _f(_count_items(data.somatic_comorbidities)),
        "cm_n_psychiatric": _f(_count_items(data.psychiatric_comorbidities)),
        # Suicide history
        "sui_ever_ideation": _b(sui.ever_thought_suicide),
        "sui_ever_attempt": _b(sui.ever_attempted),
        "sui_n_attempts": _f(sui.n_attempts),
        "sui_any_violent_attempt": _b(sui.has_violent_attempts),
        # Psychiatric history
        "psyh_age_first_episode": _f(hist.age_first_episode),
        "psyh_illness_duration_years": _f(hist.illness_duration_years),
        "psyh_n_hospitalizations_lifetime": _f(data.hospitalization.n_hospitalizations_lifetime),
        # Cohort-specific DR
        "dr_treatment_resistant": _b(tr.is_resistant),
        "dr_sachs_score": _f(tr.sachs_score),
        # ── Sub-scales ──
        # CTQ sub-scales (French keys from glossary YAML)
        "inst_ctq_emotional_abuse": _subscale(data.trauma_scores, "CTQ", "Abus émotionnel"),
        "inst_ctq_physical_abuse": _subscale(data.trauma_scores, "CTQ", "Abus physique"),
        "inst_ctq_sexual_abuse": _subscale(data.trauma_scores, "CTQ", "Abus sexuel"),
        "inst_ctq_emotional_neglect": _subscale(data.trauma_scores, "CTQ", "Négligence émotionnelle"),
        "inst_ctq_physical_neglect": _subscale(data.trauma_scores, "CTQ", "Négligence physique"),
        # BIS-10 sub-scales
        "inst_bis10_attentional": _subscale(data.impulsivity_scores, "BIS-10", "Impulsivité attentionnelle"),
        "inst_bis10_motor": _subscale(data.impulsivity_scores, "BIS-10", "Impulsivité motrice"),
        "inst_bis10_nonplanning": _subscale(data.impulsivity_scores, "BIS-10", "Impulsivité de planification"),
        # BFI personality — ignore_suspect because total_column is one subscale (bfi_extr),
        # not a real total, so the BFI always trips the suspect check; subscales are valid.
        "inst_bfi_openness": _subscale(data.personality_scores, "BFI", "Ouverture", ignore_suspect=True),
        "inst_bfi_conscientiousness": _subscale(data.personality_scores, "BFI", "Conscienciosité", ignore_suspect=True),
        "inst_bfi_extraversion": _subscale(data.personality_scores, "BFI", "Extraversion", ignore_suspect=True),
        "inst_bfi_agreeableness": _subscale(data.personality_scores, "BFI", "Agréabilité", ignore_suspect=True),
        "inst_bfi_neuroticism": _subscale(data.personality_scores, "BFI", "Névrosisme", ignore_suspect=True),
        # Cognition derived
        "cog_tmt_b_minus_a": _f(cog.tmt_b_minus_a),
        "cog_tmt_ratio_ba": _f(cog.tmt_ratio_ba),
        "cog_stroop_word": _f(cog.stroop_word),
        "cog_stroop_color": _f(cog.stroop_color),
        "cog_stroop_color_word": _f(cog.stroop_color_word),
        "cog_cvlt_short_delay_free": _f(cog.cvlt_short_delay_free),
        "cog_cvlt_recognition": _f(cog.cvlt_recognition),
        # DR-specific instruments
        "dr_cssrs_highest_ideation": _f(data.cssrs_assessment.highest_ideation_level) if data.cssrs_assessment else None,
        "dr_rosenberg_total": _score(data.self_esteem_scores, "Rosenberg"),
        "dr_erd_total": _score(data.depression_scores, "ERD"),
        "dr_leaps_total": _score(data.functioning_scores, "LEAPS"),
        # Treatment plasma levels
        "tx_lithium_level": _f(tx.lithium_level),
        "tx_valproate_level": _f(tx.valproate_level if hasattr(tx, 'valproate_level') else None),
        # Bio derived ratio
        "bio_tg_hdl_ratio": _compute_ratio(
            _get_lab(bio, "triglycerides", "triglycérides"),
            _get_lab(bio, "hdl", "hdl cholesterol"),
        ),
    }
    # Derived composites
    out["tx_polypharmacy_index"] = _polypharmacy_index(
        out.get("tx_on_antidepressant"), out.get("tx_on_antipsychotic"),
        out.get("tx_on_mood_stabilizer"), out.get("tx_on_lithium"),
    )
    out["bio_waist_height_ratio"] = _waist_height_ratio(bio)
    out["psyh_onset_category"] = _onset_category(out.get("psyh_age_first_episode"))
    out["psyh_illness_burden"] = _illness_burden(
        out.get("psyh_illness_duration_years"), None,
        out.get("psyh_n_hospitalizations_lifetime"),
    )
    return out


# ─── ASP adapter ──────────────────────────────────────────────────────────────


def adapt_asp_profile(data: ASPPatientData) -> dict[str, float | None]:
    demo = data.demographics
    bio = data.biology
    tx = data.treatments
    sub = data.substance_use
    dia = data.autism_diagnosis
    dev = data.developmental_history

    # ASP has no FamilyHistory / SuicideHistory / PsychiatricHistory shared
    # dataclass; many features will remain NaN (expected).
    out: dict[str, float | None] = {
        # Demographics
        "demo_age_years": _f(demo.age),
        "demo_sex_male": 1.0 if demo.sex == "M" else (0.0 if demo.sex == "F" else None),
        "demo_education_years_ordinal": _encode_ordinal(
            _education_ordinal(demo.education_level)
        ),
        "demo_marital_partnered": _b(_marital_partnered(demo.marital_status)),
        "demo_employed": _b(_employed(demo.employment)),
        # Mood instruments
        "inst_bdi2_total": _score(data.depression_scores, "BDI-II"),
        "inst_cgis_total": _score(data.global_scores, "CGI-C"),
        # Anxiety / impulsivity
        "inst_hama_total": _score(data.anxiety_scores, "HAM-A"),
        "inst_lsas_total": _score(data.anxiety_scores, "LSAS"),
        # Functioning
        "inst_eq5d_total": _score(data.functioning_scores, "EQ-5D"),
        "inst_egf_total": _score(data.functioning_scores, "EGF"),
        # Sleep
        "inst_psqi_total": _score(data.sleep_scores, "PSQI"),
        "inst_ess_total": _score(data.sleep_scores, "ESS"),
        # Biology — ASP rarely has labs/vitals; use BMI if present
        "bio_bmi": _f((bio.vitals or {}).get("bmi")),
        # Treatment
        "tx_on_antidepressant": _b(tx.on_antidepressant),
        "tx_on_antipsychotic": _b(tx.on_antipsychotic),
        "inst_mars_total": _score(data.adherence_scores, "MARS"),
        # Substance use
        "sub_tobacco_current": _b(sub.tobacco_current),
        "sub_tobacco_cpd": _f(sub.tobacco_cpd),
        "sub_alcohol_current": _b(sub.alcohol_current),
        "sub_cannabis_current": _b(sub.cannabis_current),
        "sub_use_disorder": _b(sub.substance_use_disorder),
        # Trauma
        "inst_ctq_total": _score(data.trauma_scores, "CTQ"),
        # Comorbidities
        "cm_n_somatic": _f(_count_items(data.somatic_comorbidities)),
        "cm_n_psychiatric": _f(_count_items(data.psychiatric_comorbidities)),
        # Family history (all None — ASP doesn't extract it)
        "fh_bipolar_any": None,
        "fh_suicide_any": None,
        "fh_substance_any": None,
        "fh_n_affected_relatives": None,
        # Suicide history (partial via BDI item 9)
        "sui_ever_ideation": _b((data.bdi_item9_suicidal_thoughts or 0) > 0) if data.bdi_item9_suicidal_thoughts is not None else None,
        "sui_ever_attempt": None,
        # Cohort-specific ASP
        "asp_dsm_domain1_met": _b(dia.dsm_domain1_met),
        "asp_dsm_domain2_met": _b(dia.dsm_domain2_met),
        "asp_age_language_months": _f(dev.age_first_phrases),
        # ── Sub-scales ──
        # CTQ sub-scales (French keys from glossary YAML)
        "inst_ctq_emotional_abuse": _subscale(data.trauma_scores, "CTQ", "Abus émotionnel"),
        "inst_ctq_physical_abuse": _subscale(data.trauma_scores, "CTQ", "Abus physique"),
        "inst_ctq_sexual_abuse": _subscale(data.trauma_scores, "CTQ", "Abus sexuel"),
        "inst_ctq_emotional_neglect": _subscale(data.trauma_scores, "CTQ", "Négligence émotionnelle"),
        "inst_ctq_physical_neglect": _subscale(data.trauma_scores, "CTQ", "Négligence physique"),
        # ASP-specific instruments — RBS-R total + 6 subscales
        "asp_rbs_r_total": _score(data.repetitive_behavior_scores, "RBS-R"),
        "asp_rbs_r_stereotypies": _subscale(data.repetitive_behavior_scores, "RBS-R", "Stéréotypies"),
        "asp_rbs_r_self_injury": _subscale(data.repetitive_behavior_scores, "RBS-R", "Auto-mutilation"),
        "asp_rbs_r_compulsive": _subscale(data.repetitive_behavior_scores, "RBS-R", "Comportements compulsifs"),
        "asp_rbs_r_rituals": _subscale(data.repetitive_behavior_scores, "RBS-R", "Rituels"),
        "asp_rbs_r_sameness": _subscale(data.repetitive_behavior_scores, "RBS-R", "Mêmeté"),
        "asp_rbs_r_restricted": _subscale(data.repetitive_behavior_scores, "RBS-R", "Comportements restreints"),
        # BRIEF GEC + 9 subscales
        "asp_brief_gec": _subscale(data.executive_function_scores, "BRIEF", "GEC"),
        "asp_brief_inhibition": _subscale(data.executive_function_scores, "BRIEF", "Inhibition"),
        "asp_brief_flexibility": _subscale(data.executive_function_scores, "BRIEF", "Flexibilité"),
        "asp_brief_emotional_control": _subscale(data.executive_function_scores, "BRIEF", "Contrôle émotionnel"),
        "asp_brief_self_control": _subscale(data.executive_function_scores, "BRIEF", "Auto-contrôle"),
        "asp_brief_initiative": _subscale(data.executive_function_scores, "BRIEF", "Initiative"),
        "asp_brief_working_memory": _subscale(data.executive_function_scores, "BRIEF", "Mémoire de travail"),
        "asp_brief_planning": _subscale(data.executive_function_scores, "BRIEF", "Planification/Organisation"),
        "asp_brief_task_monitoring": _subscale(data.executive_function_scores, "BRIEF", "Contrôle de la tâche"),
        "asp_brief_organization": _subscale(data.executive_function_scores, "BRIEF", "Organisation matérielle"),
        # WAIS-IV 4 indices
        "asp_wais4_icv": _subscale(data.cognitive_scores, "WAIS-IV", "Compréhension verbale (ICV)"),
        "asp_wais4_iri": _subscale(data.cognitive_scores, "WAIS-IV", "Raisonnement perceptif (IRP)"),
        "asp_wais4_imt": _subscale(data.cognitive_scores, "WAIS-IV", "Mémoire de travail (IMT)"),
        "asp_wais4_ivt": _subscale(data.cognitive_scores, "WAIS-IV", "Vitesse de traitement (IVT)"),
        # ADI-R 4 domains (ignore_suspect: diagnostic subscales valid independently)
        "asp_adir_social": _subscale(data.autism_screening_scores, "ADI-R", "Interaction sociale (A)", ignore_suspect=True),
        "asp_adir_communication": _subscale(data.autism_screening_scores, "ADI-R", "Communication (B)", ignore_suspect=True),
        "asp_adir_restricted": _subscale(data.autism_screening_scores, "ADI-R", "Comportements restreints (C)", ignore_suspect=True),
        "asp_adir_development": _subscale(data.autism_screening_scores, "ADI-R", "Développement anormal < 36 mois (D)", ignore_suspect=True),
        # LSAS 2 subscales
        "asp_lsas_anxiety": _subscale(data.anxiety_scores, "LSAS", "Anxiété"),
        "asp_lsas_avoidance": _subscale(data.anxiety_scores, "LSAS", "Évitement"),
        # ADHD-RS 2 subscales
        "asp_adhd_rs_inattention": _subscale(data.adhd_scores, "ADHD-RS", "Inattention"),
        "asp_adhd_rs_hyperactivity": _subscale(data.adhd_scores, "ADHD-RS", "Hyperactivité-Impulsivité"),
        # Derived composites
        "tx_polypharmacy_index": _polypharmacy_index(
            _b(tx.on_antidepressant), _b(tx.on_antipsychotic),
        ),
    }
    return out


# ─── Ordinal encoding helper ─────────────────────────────────────────────────

_EDUCATION_ORDINAL_INDEX: dict[str, int] = {
    "none": 0,
    "primaire": 1,
    "college": 2,
    "lycee": 3,
    "bac": 4,
    "bac_plus_2": 5,
    "bac_plus_5_plus": 6,
}


def _encode_ordinal(token: str | None) -> float | None:
    if token is None:
        return None
    idx = _EDUCATION_ORDINAL_INDEX.get(token)
    return float(idx) if idx is not None else None
