"""Build comprehensive patient profiles and French clinical vignettes.

Takes structured BPPatientData and produces:
- Clinical synthesis header (key findings in 2-3 sentences)
- Section-by-section clinical narrative in French
- Clinical discordance notes when instrument scores conflict
- Full vignette combining all sections

Section ordering follows standard French psychiatric consultation structure:
Identity -> History -> Episode -> Risk/Suicide -> Mood -> Functioning ->
Cognition -> Biology -> Treatments -> Comorbidities -> Family
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_rlvr.profiles.bp_extractor import (
    BPPatientData,
    CognitiveProfile,
    AdditionalNeuropsych,
    ScoreInterpretation,
)
from face_rlvr.profiles.sz_extractor import SZPatientData
from face_rlvr.profiles.dr_extractor import DRPatientData
from face_rlvr.profiles.asp_extractor import ASPPatientData
from face_rlvr.profiles.bp_instruments import BP_INSTRUMENTS
from face_rlvr.profiles.common_extractors import (
    compute_bmi_category,
    detect_metabolic_syndrome,
    compute_framingham_risk,
    check_medication_lab_alerts,
    check_drug_interactions,
    detect_floor_ceiling_effects,
    compute_data_completeness,
    compute_cognitive_z_score,
)


@dataclass
class PatientProfile:
    """Complete patient profile with structured sections and full vignette."""

    synthesis_section: str = ""
    demographics_section: str = ""
    history_section: str = ""
    episode_criteria_section: str = ""
    suicide_section: str = ""
    clinical_notes_section: str = ""
    mood_section: str = ""
    functional_section: str = ""
    anxiety_impulsivity_section: str = ""
    sleep_section: str = ""
    cognitive_section: str = ""
    additional_neuropsych_section: str = ""
    biology_section: str = ""
    treatment_section: str = ""
    lifetime_medication_section: str = ""
    non_pharm_treatment_section: str = ""
    treatment_response_section: str = ""
    comorbidity_section: str = ""
    substance_section: str = ""
    trauma_section: str = ""
    screening_section: str = ""
    diva_section: str = ""
    extended_family_section: str = ""
    risk_section: str = ""

    full_vignette: str = ""
    data: BPPatientData | SZPatientData | DRPatientData | ASPPatientData | None = None


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _fmt(val: float | None, decimals: int = 0, unit: str = "") -> str:
    if val is None:
        return "non disponible"
    if decimals == 0 and val == int(val):
        return f"{int(val)}{(' ' + unit) if unit else ''}"
    return f"{val:.{decimals}f}{(' ' + unit) if unit else ''}"


def _score_line(interp: ScoreInterpretation) -> str:
    if not interp.score_available:
        return ""
    return f"  - {interp.clinical_interpretation_fr}"


def _available_scores(scores: dict[str, ScoreInterpretation]) -> list[ScoreInterpretation]:
    return [s for s in scores.values() if s.score_available]


def _get_score(scores: dict[str, ScoreInterpretation], key: str) -> float | None:
    interp = scores.get(key)
    return interp.raw_score if interp and interp.score_available else None


def _get_severity(scores: dict[str, ScoreInterpretation], key: str) -> str:
    interp = scores.get(key)
    return interp.severity_code if interp and interp.score_available else "missing"


def _get_label_fr(scores: dict[str, ScoreInterpretation], key: str) -> str:
    """Get the French severity label from a scores dict."""
    interp = scores.get(key)
    if interp and interp.score_available:
        return interp.severity_label_fr.lower()
    return ""


def _is_female(data: BPPatientData) -> bool:
    return data.demographics.sex == "F"


def _agree(word_m: str, word_f: str, data: BPPatientData) -> str:
    """Gender-agree a French word."""
    return word_f if _is_female(data) else word_m


# ═════════════════════════════════════════════════════════════════════════════
# SYNTHESIS HEADER
# ═════════════════════════════════════════════════════════════════════════════


def _build_synthesis(data: BPPatientData) -> str:
    """2-3 sentence clinical summary placed first in the vignette."""
    parts = []

    # Current mood state
    madrs = _get_score(data.mood_scores, "MADRS")
    ymrs = _get_score(data.mood_scores, "YMRS")
    madrs_sev = _get_severity(data.mood_scores, "MADRS")
    ymrs_sev = _get_severity(data.mood_scores, "YMRS")

    arm = data.demographics.arm or "trouble bipolaire"
    state = "euthymique"
    if madrs is not None and ymrs is not None:
        if ymrs >= 13 and madrs >= 20:
            state = "en épisode mixte"
        elif ymrs >= 20:
            state = "en épisode maniaque"
        elif ymrs >= 13:
            state = "en hypomanie"
        elif madrs >= 20:
            state = f"en dépression {_get_severity(data.mood_scores, 'MADRS').replace('_', ' ')}"
        elif madrs >= 7:
            state = "en dépression légère"
    elif madrs is not None:
        if madrs >= 20:
            state = "en dépression modérée à sévère"
        elif madrs >= 7:
            state = "en dépression légère"

    sent1 = f"Patient {_agree('suivi', 'suivie', data)} pour {arm}, actuellement {state}"
    if madrs is not None:
        sent1 += f" (MADRS = {_fmt(madrs)})"
    sent1 += "."

    # Suicide risk
    cssrs = _get_score(data.suicide_scores, "C-SSRS")
    cssrs_sev = _get_severity(data.suicide_scores, "C-SSRS")
    si = data.suicide_indicators
    if cssrs is not None and cssrs >= 2:
        cssrs_label = data.suicide_scores["C-SSRS"].severity_label_fr
        sent1 += f" Risque suicidaire : {cssrs_label.lower()} (C-SSRS = {_fmt(cssrs)})."
    elif si.get("madrs_suicide_elevated"):
        sent1 += " Item suicidaire MADRS élevé."

    parts.append(sent1)

    # Notable findings (pick top concerns)
    concerns = []
    als_sev = _get_severity(data.anxiety_impulsivity_scores, "ALS")
    if als_sev == "high":
        concerns.append("labilité affective marquée")
    bis_sev = _get_severity(data.anxiety_impulsivity_scores, "BIS-10")
    if bis_sev == "high":
        concerns.append("impulsivité élevée")
    ctq_sev = _get_severity(data.trauma_scores, "CTQ")
    if ctq_sev in ("moderate_severe", "severe_extreme"):
        concerns.append("antécédents traumatiques significatifs")
    psqi_sev = _get_severity(data.sleep_scores, "PSQI")
    if psqi_sev in ("poor", "very_poor"):
        concerns.append("troubles du sommeil")
    if concerns:
        parts.append("Facteurs notables : " + ", ".join(concerns) + ".")

    # Current treatment
    t = data.treatments
    if t.mood_stabilizers:
        drug = list(t.mood_stabilizers.keys())[0]
        info = t.mood_stabilizers[drug]
        pl = info.get("plasma_level")
        unit = info.get("unit", "")
        parts.append(f"Traitement par {drug} (taux = {_fmt(pl, 1)} {unit}).")

    return "Synthèse clinique : " + " ".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION BUILDERS
# ═════════════════════════════════════════════════════════════════════════════


def _build_demographics(data: BPPatientData) -> str:
    d = data.demographics
    sex = d.sex_label_fr or "sexe non précisé"
    arm = d.arm or "trouble bipolaire"
    age = d.age or "?"
    line1 = f"Patient {data.patient_id}, {sex} de {age} ans, {_agree('suivi', 'suivie', data)} pour {arm}."

    details = []
    if d.marital_status:
        details.append(d.marital_status)
    if d.education_level:
        details.append(f"niveau d'études : {d.education_level}")
    if d.employment:
        details.append(d.employment)
    if details:
        return line1 + "\n" + ", ".join(details).capitalize() + "."
    return line1


def _build_history(data: BPPatientData) -> str:
    h = data.psychiatric_history
    lines = []

    if h.age_first_episode is not None:
        lines.append(f"  - Début du trouble à {h.age_first_episode} ans")
    if h.illness_duration_years is not None:
        lines.append(f"  - Durée d'évolution : {_fmt(h.illness_duration_years)} ans")

    ep_parts = []
    if h.n_depressive_episodes_lifetime is not None:
        ep_parts.append(f"{h.n_depressive_episodes_lifetime} EDM")
    if h.n_manic_episodes_lifetime is not None:
        ep_parts.append(f"{h.n_manic_episodes_lifetime} épisode(s) maniaque(s)")
    if h.n_hypomanic_episodes_lifetime is not None:
        ep_parts.append(f"{h.n_hypomanic_episodes_lifetime} épisode(s) hypomaniaque(s)")
    if h.n_mixed_episodes_lifetime is not None and h.n_mixed_episodes_lifetime > 0:
        ep_parts.append(f"{h.n_mixed_episodes_lifetime} épisode(s) mixte(s)")
    if ep_parts:
        lines.append(f"  - Épisodes (vie entière) : {', '.join(ep_parts)}")

    psy_parts = []
    if h.n_psychotic_depressive_lifetime and h.n_psychotic_depressive_lifetime > 0:
        psy_parts.append(f"{h.n_psychotic_depressive_lifetime} EDM psychotique(s)")
    if h.n_psychotic_manic_lifetime and h.n_psychotic_manic_lifetime > 0:
        psy_parts.append(f"{h.n_psychotic_manic_lifetime} manie(s) psychotique(s)")
    if psy_parts:
        lines.append(f"  - Caractéristiques psychotiques : {', '.join(psy_parts)}")

    if h.rapid_cycling:
        lines.append("  - Cycles rapides : oui")

    if h.current_episode_type:
        sev = f" ({h.current_episode_severity})" if h.current_episode_severity else ""
        lines.append(f"  - Épisode actuel : {h.current_episode_type}{sev}")

    hosp = data.hospitalization
    if hosp.ever_hospitalized and hosp.n_hospitalizations_lifetime:
        hosp_line = f"  - Hospitalisations : {hosp.n_hospitalizations_lifetime} (vie entière)"
        if hosp.n_hospitalizations_last_year:
            hosp_line += f", dont {hosp.n_hospitalizations_last_year} dans l'année écoulée"
        lines.append(hosp_line)

    if not lines:
        return "Antécédents psychiatriques : données non disponibles."
    return "Antécédents psychiatriques :\n" + "\n".join(lines)


def _build_episode_criteria(data: BPPatientData) -> str:
    ec = data.current_episode_criteria
    lines = []

    dep_items = [
        (ec.depressed_mood, "humeur dépressive"),
        (ec.anhedonia, "anhédonie"),
        (ec.weight_change, "modification pondérale"),
        (ec.sleep_disturbance, "troubles du sommeil"),
        (ec.psychomotor_change, "modification psychomotrice"),
        (ec.fatigue, "fatigue"),
        (ec.worthlessness, "culpabilité/dévalorisation"),
        (ec.concentration_difficulty, "difficultés de concentration"),
        (ec.suicidal_thoughts, "idées de mort/suicide"),
    ]
    dep_present = [label for val, label in dep_items if val]
    if dep_present or ec.depressive_symptom_count is not None:
        count = ec.depressive_symptom_count or len(dep_present)
        lines.append(f"  Critères dépressifs : {count}/9 présents")
        if dep_present:
            lines.append(f"    ({', '.join(dep_present)})")

    man_items = [
        (ec.elevated_mood, "humeur élevée"),
        (ec.irritable_mood, "irritabilité"),
        (ec.grandiosity, "idées de grandeur"),
        (ec.decreased_sleep_need, "réduction du besoin de sommeil"),
        (ec.pressured_speech, "logorrhée"),
        (ec.flight_of_ideas, "fuite des idées"),
        (ec.distractibility, "distractibilité"),
        (ec.goal_directed_activity, "activité augmentée"),
        (ec.risky_behavior, "activités à risque"),
    ]
    man_present = [label for val, label in man_items if val]
    if man_present or ec.manic_symptom_count is not None:
        count = ec.manic_symptom_count or len(man_present)
        lines.append(f"  Critères maniaques : {count}/9 présents")
        if man_present:
            lines.append(f"    ({', '.join(man_present)})")

    mre = data.most_recent_episode
    if mre.episode_type:
        parts = [mre.episode_type]
        if mre.severity:
            parts.append(mre.severity)
        if mre.chronicity:
            parts.append(f"chronicité : {mre.chronicity}")
        if mre.postpartum:
            parts.append("post-partum")
        lines.append(f"  Épisode le plus récent : {', '.join(parts)}")

    if not lines:
        return ""
    return "Épisode actuel (critères DSM) :\n" + "\n".join(lines)


def _build_suicide(data: BPPatientData) -> str:
    """Suicide assessment — placed early in the vignette for safety visibility."""
    available = _available_scores(data.suicide_scores)
    si = data.suicide_indicators
    sh = data.suicide_history
    lines = []

    # C-SSRS
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
        if interp.instrument == "C-SSRS" and interp.subscales:
            for sub_name, sub_val in interp.subscales.items():
                if sub_val is not None:
                    lines.append(f"    {sub_name} : {_fmt(sub_val)}/5")

    # MADRS suicide item
    if si.get("madrs_suicide_elevated"):
        lines.append(f"  - Item suicidaire MADRS élevé ({si.get('madrs_suicide_item', '?')}/6)")

    # ISF history
    if sh.ever_attempted:
        n = sh.n_attempts
        lines.append(f"  - Antécédent de tentative(s) de suicide : {n if n else 'oui'}")
        if sh.has_violent_attempts and sh.n_violent_attempts:
            lines.append(f"    dont {sh.n_violent_attempts} TS violente(s)")
        if sh.has_serious_attempts and sh.n_serious_attempts:
            lines.append(f"    dont {sh.n_serious_attempts} TS grave(s)")
    elif sh.ever_attempted is False and sh.ever_thought_suicide:
        lines.append("  - Antécédent d'idéation suicidaire au cours de la vie (sans passage à l'acte)")

    # Columbia lethality
    if sh.most_serious_method:
        lines.append(f"  - Méthode TS la plus grave : {sh.most_serious_method}")

    if not lines:
        return ""
    return "Évaluation suicidaire :\n" + "\n".join(lines)


def _build_clinical_notes(data: BPPatientData) -> str:
    """Detect and flag clinical discordances between instruments."""
    notes = []

    madrs = _get_score(data.mood_scores, "MADRS")
    madrs_sev = _get_severity(data.mood_scores, "MADRS")
    qids = _get_score(data.mood_scores, "QIDS-SR16")
    qids_sev = _get_severity(data.mood_scores, "QIDS-SR16")
    ymrs = _get_score(data.mood_scores, "YMRS")
    ymrs_sev = _get_severity(data.mood_scores, "YMRS")
    asrm = _get_score(data.mood_scores, "ASRM")
    asrm_sev = _get_severity(data.mood_scores, "ASRM")
    cgi = _get_score(data.mood_scores, "CGI-S")

    # 1. MADRS vs QIDS-SR16 discordance
    if madrs is not None and qids is not None:
        dep_levels = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3, "very_severe": 4}
        m_level = dep_levels.get(madrs_sev, -1)
        q_level = dep_levels.get(qids_sev, -1)
        if m_level != q_level and m_level >= 0 and q_level >= 0:
            higher = "auto-évaluation (QIDS)" if q_level > m_level else "hétéro-évaluation (MADRS)"
            notes.append(
                f"Discordance MADRS ({_fmt(madrs)}, {_get_label_fr(data.mood_scores, 'MADRS')}) / "
                f"QIDS-SR16 ({_fmt(qids)}, {_get_label_fr(data.mood_scores, 'QIDS-SR16')}) : "
                f"la {higher} est plus sévère."
            )

    # 2. ASRM vs YMRS discordance
    if asrm is not None and ymrs is not None:
        if asrm_sev == "positive" and ymrs_sev in ("normal", "mild"):
            notes.append(
                f"Discordance ASRM ({_fmt(asrm)}, positif) / "
                f"YMRS ({_fmt(ymrs)}, {ymrs_sev}) : "
                f"le patient rapporte des symptômes hypomaniaques non confirmés par le clinicien."
            )
        elif asrm_sev == "negative" and ymrs_sev in ("moderate", "severe"):
            notes.append(
                f"Discordance ASRM ({_fmt(asrm)}, négatif) / "
                f"YMRS ({_fmt(ymrs)}, {ymrs_sev}) : "
                f"manie clinique non perçue par le patient (possible anosognosie)."
            )

    # 3. CGI vs symptom scales
    if cgi is not None and cgi >= 4 and madrs is not None and ymrs is not None:
        if madrs < 20 and ymrs < 13:
            notes.append(
                f"Discordance CGI-S ({_fmt(cgi)}) avec MADRS ({_fmt(madrs)}) et "
                f"YMRS ({_fmt(ymrs)}) : la sévérité globale excède les scores symptomatiques."
            )

    # 4. Sleep + sedating medications
    ess = _get_score(data.sleep_scores, "ESS")
    ess_sev = _get_severity(data.sleep_scores, "ESS")
    t = data.treatments
    if ess is not None and ess_sev in ("excessive", "severe"):
        sedating = []
        if t.on_lithium:
            sedating.append("lithium")
        if t.on_benzodiazepine:
            sedating.append("benzodiazépine")
        if t.on_antipsychotic:
            sedating.append("antipsychotique")
        if sedating:
            notes.append(
                f"Somnolence diurne excessive (ESS = {_fmt(ess)}) chez un patient sous "
                f"{', '.join(sedating)} : effet sédatif iatrogène à évaluer."
            )

    # 5. FAST vs EQ-5D functional discordance
    fast = _get_score(data.functional_scores, "FAST")
    fast_sev = _get_severity(data.functional_scores, "FAST")
    eq5d = _get_score(data.functional_scores, "EQ-5D")
    eq5d_sev = _get_severity(data.functional_scores, "EQ-5D")

    if fast is not None and eq5d is not None:
        eq5d_interp = data.functional_scores.get("EQ-5D")
        if not (eq5d_interp and eq5d_interp.suspect_value):
            if fast_sev in ("none", "mild") and eq5d_sev == "poor":
                notes.append(
                    f"Discordance FAST ({_fmt(fast)}, {_get_label_fr(data.functional_scores, 'FAST')}) / "
                    f"EQ-5D ({eq5d:.2f}, {_get_label_fr(data.functional_scores, 'EQ-5D')}) : "
                    f"le retentissement subjectif excède l'altération fonctionnelle objectivée."
                )
            elif fast_sev in ("moderate", "severe") and eq5d_sev == "good":
                notes.append(
                    f"Discordance FAST ({_fmt(fast)}, {_get_label_fr(data.functional_scores, 'FAST')}) / "
                    f"EQ-5D ({eq5d:.2f}, {_get_label_fr(data.functional_scores, 'EQ-5D')}) : "
                    f"l'altération fonctionnelle objectivée ne se reflète pas dans la qualité de vie perçue."
                )

    # 6. Floor/ceiling effect detection
    all_scores: dict[str, ScoreInterpretation] = {}
    for domain_scores in (
        data.mood_scores,
        data.functional_scores,
        data.anxiety_impulsivity_scores,
        data.sleep_scores,
        data.suicide_scores,
        data.screening_scores,
        data.trauma_scores,
    ):
        all_scores.update(domain_scores)
    fc_effects = detect_floor_ceiling_effects(all_scores, BP_INSTRUMENTS)
    for effect in fc_effects:
        notes.append(f"Qualité des données — {effect}")

    if not notes:
        return ""
    lines = ["Notes cliniques :"]
    for note in notes:
        lines.append(f"  * {note}")
    return "\n".join(lines)


def _build_mood(data: BPPatientData) -> str:
    available = [s for s in _available_scores(data.mood_scores) if not s.suspect_value]
    if not available:
        return ""

    # Sub-group by evaluation type
    hetero = []
    auto = []
    for interp in available:
        inst = BP_INSTRUMENTS.get(interp.instrument)
        if inst and inst.evaluation_type == "auto":
            auto.append(interp)
        else:
            hetero.append(interp)

    lines = ["État thymique actuel :"]

    if hetero and auto:
        # Both groups present — use sub-headers
        lines.append("  Hétéro-évaluation :")
        for interp in hetero:
            line = _score_line(interp)
            if line:
                lines.append(f"  {line}")
        lines.append("  Auto-évaluation :")
        for interp in auto:
            line = _score_line(interp)
            if line:
                lines.append(f"  {line}")
    else:
        # Only one group — flat list
        for interp in available:
            line = _score_line(interp)
            if line:
                lines.append(line)

    return "\n".join(lines)


def _build_functional(data: BPPatientData) -> str:
    available = [s for s in _available_scores(data.functional_scores) if not s.suspect_value]
    if not available:
        return ""
    lines = ["Fonctionnement :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_anxiety_impulsivity(data: BPPatientData) -> str:
    available = _available_scores(data.anxiety_impulsivity_scores)
    if not available:
        return ""
    lines = ["Anxiété et impulsivité :"]
    for interp in available:
        if interp.suspect_value:
            continue
        line = _score_line(interp)
        if line:
            lines.append(line)
        # BIS subscales only when impulsivity is elevated
        if interp.instrument == "BIS-10" and interp.severity_code == "high" and interp.subscales:
            for sub_name, sub_val in interp.subscales.items():
                if sub_val is not None:
                    lines.append(f"    - {sub_name} : {_fmt(sub_val)}")
    return "\n".join(lines)


def _build_sleep(data: BPPatientData) -> str:
    available = _available_scores(data.sleep_scores)
    if not available:
        return ""
    lines = ["Sommeil :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_cognitive(data: BPPatientData) -> str:
    cp = data.cognitive_profile
    lines = []

    if cp.cobra_interpretation and cp.cobra_interpretation.score_available:
        lines.append(_score_line(cp.cobra_interpretation))

    age = data.demographics.age

    if cp.tmt_a_seconds is not None or cp.tmt_b_seconds is not None:
        tmt_parts = []
        if cp.tmt_a_seconds is not None:
            tmt_parts.append(f"TMT-A = {_fmt(cp.tmt_a_seconds)} s")
            z_a, z_a_interp = compute_cognitive_z_score("TMT-A", cp.tmt_a_seconds, age)
            if z_a is not None:
                tmt_parts.append(f"z = {z_a:.1f}")
        if cp.tmt_b_seconds is not None:
            tmt_parts.append(f"TMT-B = {_fmt(cp.tmt_b_seconds)} s")
            z_b, z_b_interp = compute_cognitive_z_score("TMT-B", cp.tmt_b_seconds, age)
            if z_b is not None:
                tmt_parts.append(f"z = {z_b:.1f}")
        if cp.tmt_b_minus_a is not None:
            tmt_parts.append(f"B-A = {_fmt(cp.tmt_b_minus_a, 1)} s")
        lines.append(f"  - Trail Making Test : {', '.join(tmt_parts)}")
        # Add z-score interpretations
        if cp.tmt_a_seconds is not None:
            z_a, z_a_interp = compute_cognitive_z_score("TMT-A", cp.tmt_a_seconds, age)
            if z_a is not None:
                lines.append(f"    TMT-A : {z_a_interp}")
        if cp.tmt_b_seconds is not None:
            z_b, z_b_interp = compute_cognitive_z_score("TMT-B", cp.tmt_b_seconds, age)
            if z_b is not None:
                lines.append(f"    TMT-B : {z_b_interp}")
        if cp.tmt_b_minus_a is not None:
            if cp.tmt_b_minus_a > 90:
                lines.append("    Flexibilité cognitive altérée (B-A > 90 s)")
            elif cp.tmt_b_minus_a > 60:
                lines.append("    Flexibilité cognitive limite (B-A 60-90 s)")

    if cp.stroop_interference is not None:
        z_stroop, z_stroop_interp = compute_cognitive_z_score(
            "Stroop_interference", cp.stroop_interference, age,
        )
        if z_stroop is not None:
            lines.append(f"  - Stroop interférence : {_fmt(cp.stroop_interference)} — {z_stroop_interp}")
        else:
            lines.append(f"  - Stroop interférence : {_fmt(cp.stroop_interference)}")

    if cp.cvlt_total_learning is not None:
        cvlt_parts = [f"apprentissage = {_fmt(cp.cvlt_total_learning)}"]
        if cp.cvlt_long_delay_free is not None:
            cvlt_parts.append(f"rappel différé = {_fmt(cp.cvlt_long_delay_free)}")
        lines.append(f"  - CVLT : {', '.join(cvlt_parts)}")

    if cp.phonemic_fluency is not None:
        lines.append(f"  - Fluence phonémique : {_fmt(cp.phonemic_fluency)}")

    wais_parts = []
    if cp.wais_similarities_std is not None:
        wais_parts.append(f"similitudes = {_fmt(cp.wais_similarities_std)}")
    if cp.wais_vocabulary_std is not None:
        wais_parts.append(f"vocabulaire = {_fmt(cp.wais_vocabulary_std)}")
    if cp.wais_working_memory_std is not None:
        wais_parts.append(f"mémoire de travail = {_fmt(cp.wais_working_memory_std)}")
    if wais_parts:
        lines.append(f"  - WAIS (notes standard) : {', '.join(wais_parts)}")

    if not lines:
        return ""
    return "Profil cognitif :\n" + "\n".join(lines)


def _build_additional_neuropsych(data: BPPatientData) -> str:
    an = data.additional_neuropsych
    lines = []

    if an.matrices_std is not None:
        lines.append(f"  - Matrices (WAIS) : note standard = {_fmt(an.matrices_std)}")
    if an.code_std is not None:
        lines.append(f"  - Code (vitesse de traitement) : note standard = {_fmt(an.code_std)}")
    if an.symbol_std is not None:
        lines.append(f"  - Symboles : note standard = {_fmt(an.symbol_std)}")
    if an.digit_span_total_std is not None:
        parts = [f"total = {_fmt(an.digit_span_total_std)}"]
        if an.digit_span_forward_std is not None:
            parts.append(f"direct = {_fmt(an.digit_span_forward_std)}")
        if an.digit_span_backward_std is not None:
            parts.append(f"inverse = {_fmt(an.digit_span_backward_std)}")
        lines.append(f"  - Empan de chiffres (notes standard) : {', '.join(parts)}")

    cpt_parts = []
    if an.cpt_omissions is not None:
        cpt_parts.append(f"omissions = {_fmt(an.cpt_omissions)}")
    if an.cpt_commissions is not None:
        cpt_parts.append(f"commissions = {_fmt(an.cpt_commissions)}")
    if an.cpt_hit_rt is not None:
        cpt_parts.append(f"TR = {_fmt(an.cpt_hit_rt)}")
    if an.cpt_detectability is not None:
        cpt_parts.append(f"d' = {_fmt(an.cpt_detectability, 2)}")
    if cpt_parts:
        lines.append(f"  - CPT-III (attention soutenue) : {', '.join(cpt_parts)}")

    ld = []
    if an.dyslexia:
        ld.append("dyslexie")
    if an.dysorthographia:
        ld.append("dysorthographie")
    if an.dyscalculia:
        ld.append("dyscalculie")
    if an.dyspraxia:
        ld.append("dyspraxie")
    if ld:
        lines.append(f"  - Troubles des apprentissages : {', '.join(ld)}")

    if not lines:
        return ""
    return "Bilan neuropsychologique complémentaire :\n" + "\n".join(lines)


def _build_biology(data: BPPatientData) -> str:
    bio = data.biology
    if not bio.values and not bio.vitals:
        return ""

    lines = []

    # Vitals
    vital_parts = []
    v = bio.vitals
    bmi = v.get("bmi")
    if bmi is not None:
        bmi_result = compute_bmi_category(bmi)
        bmi_cat = f" ({bmi_result[1]})" if bmi_result else ""
        vital_parts.append(f"IMC = {_fmt(bmi, 1)} kg/m²{bmi_cat}")
    if "weight_kg" in v:
        vital_parts.append(f"poids = {_fmt(v['weight_kg'], 1)} kg")
    if "waist_cm" in v:
        vital_parts.append(f"tour de taille = {_fmt(v['waist_cm'])} cm")
    if "sbp_standing" in v and "dbp_standing" in v:
        vital_parts.append(f"PA = {_fmt(v['sbp_standing'])}/{_fmt(v['dbp_standing'])} mmHg")
    if "hr_standing" in v:
        vital_parts.append(f"FC = {_fmt(v['hr_standing'])} bpm")
    if vital_parts:
        lines.append("  Constantes : " + ", ".join(vital_parts))

    if bio.ecg:
        ecg_parts = []
        if "qtc" in bio.ecg:
            qtc = bio.ecg["qtc"]
            flag = " (allongé)" if qtc > 450 else ""
            ecg_parts.append(f"QTc = {_fmt(qtc)} ms{flag}")
        if ecg_parts:
            lines.append("  ECG : " + ", ".join(ecg_parts))

    # Lab values — compressed display
    abnormal = [lv for lv in bio.values if lv.is_abnormal]
    normal = [lv for lv in bio.values if not lv.is_abnormal]

    if abnormal:
        lines.append("  Anomalies biologiques :")
        for lv in abnormal:
            arrow = "↑" if lv.abnormality == "high" else "↓"
            lines.append(f"    {lv.name_fr} : {_fmt(lv.value, 2)} {lv.unit} ({arrow})")

    if normal:
        lines.append(f"  {len(normal)} autres paramètres biologiques dans les normes.")
    elif not abnormal and bio.values:
        lines.append(f"  Bilan biologique sans anomalie ({len(bio.values)} paramètres dans les normes).")

    # ── Metabolic syndrome detection ──
    sex = data.demographics.sex
    # Extract lab values needed for metabolic syndrome
    lab_dict = {lv.name: lv for lv in bio.values}
    trig_lv = lab_dict.get("Triglycérides") or lab_dict.get("Triglycerides")
    hdl_lv = lab_dict.get("HDL") or lab_dict.get("HDL-cholestérol")
    glucose_lv = lab_dict.get("Glycémie") or lab_dict.get("Glucose") or lab_dict.get("Glycémie à jeun")

    sbp = v.get("sbp_standing") or v.get("sbp_supine")
    dbp = v.get("dbp_standing") or v.get("dbp_supine")

    met_positive, met_criteria = detect_metabolic_syndrome(
        waist_cm=v.get("waist_cm"),
        sex=sex,
        trig=trig_lv.value if trig_lv else None,
        hdl=hdl_lv.value if hdl_lv else None,
        sbp=sbp,
        dbp=dbp,
        glucose=glucose_lv.value if glucose_lv else None,
    )
    if met_criteria:
        if met_positive:
            lines.append("  Syndrome métabolique (IDF/ATP-III) : OUI")
        else:
            lines.append(f"  Critères métaboliques ({len(met_criteria)}/5, seuil = 3) :")
        for c in met_criteria:
            lines.append(f"    - {c}")

    # ── Framingham cardiovascular risk ──
    total_chol_lv = lab_dict.get("Cholestérol total") or lab_dict.get("Total cholesterol")
    age = data.demographics.age
    smoking = data.substance_use.tobacco_current if data.substance_use else False
    # Check if diabetes is listed in somatic comorbidities
    has_diabetes = any("diabète" in c.lower() or "diabetes" in c.lower() for c in data.somatic_comorbidities)
    t = data.treatments
    on_bp_treatment = any("antihypertenseur" in m.lower() for m in t.current_medications) if t.current_medications else False

    fram_risk, fram_cat = compute_framingham_risk(
        age=age,
        sex=sex,
        total_chol=total_chol_lv.value if total_chol_lv else None,
        hdl=hdl_lv.value if hdl_lv else None,
        sbp=sbp,
        on_bp_treatment=on_bp_treatment,
        smoking=smoking,
        diabetes=has_diabetes,
    )
    if fram_risk is not None:
        lines.append(f"  Risque cardiovasculaire (Framingham) : {fram_risk:.0f}% à 10 ans — {fram_cat}")

    # ── Medication-lab alerts ──
    treatment_dict = {
        "on_lithium": t.on_lithium,
        "on_valproate": t.on_valproate,
        "on_carbamazepine": t.on_carbamazepine,
        "on_antipsychotic": t.on_antipsychotic,
    }
    med_alerts = check_medication_lab_alerts(treatment_dict, bio.values, sex)
    if med_alerts:
        lines.append("  Alertes médicamenteuses :")
        for alert in med_alerts:
            lines.append(f"    ⚠ {alert}")

    if not lines:
        return ""
    return "Bilan somatique :\n" + "\n".join(lines)


def _build_treatment(data: BPPatientData) -> str:
    t = data.treatments
    lines = []

    for drug, info in t.mood_stabilizers.items():
        pl = info.get("plasma_level")
        unit = info.get("unit", "")
        if pl is not None:
            range_note = ""
            if drug == "lithium":
                if pl < 0.6:
                    range_note = " (sous-thérapeutique)"
                elif pl <= 0.8:
                    range_note = " (zone thérapeutique basse)"
                elif pl <= 1.0:
                    range_note = " (zone thérapeutique)"
                else:
                    range_note = " (supra-thérapeutique)"
            elif drug == "valproate":
                if pl < 50:
                    range_note = " (sous-thérapeutique)"
                elif pl <= 100:
                    range_note = " (zone thérapeutique)"
                else:
                    range_note = " (supra-thérapeutique)"
            lines.append(f"  - {drug.capitalize()} : {_fmt(pl, 1)} {unit}{range_note}")
        else:
            lines.append(f"  - {drug.capitalize()}")

    other_meds = []
    if t.on_antidepressant:
        other_meds.append("antidépresseur")
    if t.on_antipsychotic:
        other_meds.append("antipsychotique")
    if t.on_benzodiazepine:
        other_meds.append("benzodiazépine")
    if other_meds:
        lines.append(f"  - Autres : {', '.join(other_meds)}")

    if t.medication_adherence and t.medication_adherence.score_available:
        lines.append(f"  - Observance : {t.medication_adherence.clinical_interpretation_fr}")

    # ── Drug-drug interaction alerts ──
    treatment_flags = {
        "on_lithium": t.on_lithium,
        "on_valproate": t.on_valproate,
        "on_carbamazepine": t.on_carbamazepine,
        "on_lamotrigine": t.on_lamotrigine,
        "on_antipsychotic": t.on_antipsychotic,
        "on_antidepressant": t.on_antidepressant,
        "on_benzodiazepine": t.on_benzodiazepine,
    }
    interaction_alerts = check_drug_interactions(treatment_flags)
    if interaction_alerts:
        lines.append("  Interactions médicamenteuses :")
        for alert in interaction_alerts:
            lines.append(f"    {alert}")

    if not lines:
        return ""
    return "Traitement actuel :\n" + "\n".join(lines)


def _build_lifetime_medications(data: BPPatientData) -> str:
    lm = data.lifetime_medications
    lines = []
    meds = [
        (lm.lithium_ever, "Lithium", lm.lithium_duration_months),
        (lm.thymoregulator_ever, "Thymorégulateur", lm.thymoregulator_duration_months),
        (lm.antidepressant_ever, "Antidépresseur", lm.antidepressant_duration_months),
        (lm.antipsychotic_ever, "Antipsychotique", lm.antipsychotic_duration_months),
        (lm.neuroleptic_ever, "Neuroleptique", lm.neuroleptic_duration_months),
        (lm.benzodiazepine_ever, "Benzodiazépine", lm.benzodiazepine_duration_months),
    ]
    for ever, name, duration in meds:
        if ever:
            dur_str = f" ({_fmt(duration)} mois)" if duration else ""
            lines.append(f"  - {name}{dur_str}")
    if not lines:
        return ""
    return "Historique médicamenteux (vie entière) :\n" + "\n".join(lines)


def _build_non_pharm(data: BPPatientData) -> str:
    np_t = data.non_pharm_treatments
    if not np_t.has_non_pharm_lifetime:
        return ""
    lines = []
    if np_t.ect_lifetime:
        sessions = f" ({np_t.ect_sessions} séances)" if np_t.ect_sessions else ""
        lines.append(f"  - Sismothérapie (ECT){sessions}")
    if np_t.tms_lifetime:
        sessions = f" ({np_t.tms_sessions} séances)" if np_t.tms_sessions else ""
        lines.append(f"  - TMS{sessions}")
    if np_t.cbt_lifetime:
        lines.append("  - TCC")
    if np_t.ipsrt_lifetime:
        lines.append("  - IPSRT")
    if np_t.psychoeducation_lifetime:
        lines.append("  - Psychoéducation")
    if not lines:
        lines.append("  - Traitement non pharmacologique (non précisé)")
    return "Traitements non pharmacologiques :\n" + "\n".join(lines)


def _build_treatment_response(data: BPPatientData) -> str:
    available = _available_scores(data.treatment_response_scores)
    if not available:
        return ""
    lines = ["Réponse au traitement :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
        if interp.instrument == "ALDA" and interp.subscales:
            for sub_name, sub_val in interp.subscales.items():
                if sub_val is not None:
                    lines.append(f"    {sub_name} : {_fmt(sub_val)}")
    return "\n".join(lines)


def _build_comorbidity(data: BPPatientData) -> str:
    lines = []
    if data.psychiatric_comorbidities:
        lines.append("  Psychiatriques : " + ", ".join(data.psychiatric_comorbidities))
    if data.somatic_comorbidities:
        lines.append("  Somatiques : " + ", ".join(data.somatic_comorbidities))

    su = data.substance_use
    su_parts = []
    if su.tobacco_current:
        su_parts.append("tabac")
    if su.alcohol_current:
        su_parts.append("alcool")
    if su.cannabis_current:
        su_parts.append("cannabis")
    if su.other_substances:
        su_parts.extend(su.other_substances)
    if su_parts:
        lines.append("  Usage de substances : " + ", ".join(su_parts))
    if su.substance_use_disorder:
        lines.append("  Trouble lié à l'usage de substances diagnostiqué")

    if not lines:
        return ""
    return "Comorbidités :\n" + "\n".join(lines)


def _build_substance_assessment(data: BPPatientData) -> str:
    available = _available_scores(data.substance_scores)
    if not available:
        return ""
    lines = ["Évaluation addictologique :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_trauma(data: BPPatientData) -> str:
    available = _available_scores(data.trauma_scores)
    if not available:
        return ""
    lines = ["Traumatismes de l'enfance :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
        if interp.instrument == "CTQ" and interp.subscales:
            for sub_name, sub_val in interp.subscales.items():
                if sub_val is not None:
                    level = ""
                    if sub_val <= 8:
                        level = " (aucun/minimal)"
                    elif sub_val <= 12:
                        level = " (léger à modéré)"
                    elif sub_val <= 15:
                        level = " (modéré à sévère)"
                    else:
                        level = " (sévère à extrême)"
                    lines.append(f"    {sub_name} : {_fmt(sub_val)}{level}")
    return "\n".join(lines)


def _build_screening(data: BPPatientData) -> str:
    available = _available_scores(data.screening_scores)
    if not available:
        return ""
    lines = ["Dépistages :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_diva(data: BPPatientData) -> str:
    diva = data.diva_adhd
    if diva.attention_adult_count is None and diva.hyperactivity_adult_count is None:
        return ""
    lines = []
    if diva.attention_adult_count is not None:
        lines.append(f"  - Inattention adulte : {diva.attention_adult_count}/9")
    if diva.hyperactivity_adult_count is not None:
        lines.append(f"  - Hyperactivité adulte : {diva.hyperactivity_adult_count}/9")
    if diva.attention_childhood_count is not None:
        lines.append(f"  - Inattention enfance : {diva.attention_childhood_count}/9")
    if diva.hyperactivity_childhood_count is not None:
        lines.append(f"  - Hyperactivité enfance : {diva.hyperactivity_childhood_count}/9")
    return "DIVA (TDAH — DSM-IV) :\n" + "\n".join(lines)


def _build_extended_family(data: BPPatientData) -> str:
    fh = data.family_history
    lines = []

    gp_with_disorder = [r for r in fh.relatives if r.psychiatric_disorder and r.relation.startswith("g")]
    gp_with_suicide = [r for r in fh.relatives if r.suicide and r.relation.startswith("g")]
    if gp_with_disorder:
        for r in gp_with_disorder:
            lines.append(f"  - {r.relation_fr} : {r.psychiatric_disorder}")
    if gp_with_suicide:
        for r in gp_with_suicide:
            lines.append(f"  - {r.relation_fr} : suicide")

    if fh.n_siblings is not None:
        affected = f", dont {fh.n_siblings_affected} atteint(s)" if fh.n_siblings_affected else ""
        lines.append(f"  - Fratrie : {fh.n_siblings} membre(s){affected}")
    sibling_rels = [r for r in fh.relatives if r.relation in ("frere1", "soeur1")]
    for r in sibling_rels:
        if r.psychiatric_disorder:
            lines.append(f"    {r.relation_fr} : {r.psychiatric_disorder}")

    if fh.n_children is not None and fh.n_children > 0:
        affected = f", dont {fh.n_children_affected} atteint(s)" if fh.n_children_affected else ""
        lines.append(f"  - Enfants : {fh.n_children}{affected}")

    if not lines:
        return ""
    return "Antécédents familiaux élargis :\n" + "\n".join(lines)


def _build_risk(data: BPPatientData) -> str:
    """Aggregated risk factors — placed last as a synthesis of risk elements."""
    lines = []

    # Suicide risk (cross-referenced from C-SSRS + ISF)
    cssrs_sev = _get_severity(data.suicide_scores, "C-SSRS")
    if cssrs_sev not in ("missing", "wish_to_be_dead"):
        cssrs_label = data.suicide_scores.get("C-SSRS")
        if cssrs_label and cssrs_label.score_available:
            lines.append(f"  - Risque suicidaire : {cssrs_label.severity_label_fr.lower()}")

    ec = data.current_episode_criteria
    if ec.suicidal_thoughts:
        lines.append("  - Idées de mort/suicide présentes (critère DSM)")

    sh = data.suicide_history
    if sh.ever_attempted:
        lines.append("  - Antécédent de tentative de suicide")

    # Impulsivity / hostility / lability
    bis = data.anxiety_impulsivity_scores.get("BIS-10")
    if bis and bis.score_available and bis.severity_code == "high":
        lines.append("  - Impulsivité élevée")

    bdhi = data.anxiety_impulsivity_scores.get("BDHI")
    if bdhi and bdhi.score_available and bdhi.severity_code == "high":
        lines.append("  - Hostilité élevée")

    als = data.anxiety_impulsivity_scores.get("ALS")
    if als and als.score_available and als.severity_code == "high":
        lines.append("  - Labilité affective marquée")

    ctq = data.trauma_scores.get("CTQ")
    if ctq and ctq.score_available and ctq.severity_code in ("moderate_severe", "severe_extreme"):
        lines.append("  - Traumatismes de l'enfance significatifs")

    fh = data.family_history
    if fh.maternal_suicide or fh.paternal_suicide:
        lines.append("  - Antécédents familiaux de suicide")
    if fh.family_bipolar:
        lines.append("  - Antécédents familiaux de trouble bipolaire")
    fam_parts = []
    if fh.maternal_psychiatric:
        fam_parts.append(f"mère : {fh.maternal_psychiatric}")
    if fh.paternal_psychiatric:
        fam_parts.append(f"père : {fh.paternal_psychiatric}")
    if fam_parts:
        lines.append(f"  - ATCD psychiatriques familiaux : {', '.join(fam_parts)}")

    if not lines:
        return ""
    return "Facteurs de risque :\n" + "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN BUILDER
# ═════════════════════════════════════════════════════════════════════════════


def build_bp_profile(data: BPPatientData) -> PatientProfile:
    """Build a comprehensive patient profile from structured BP data."""
    # Section order follows standard psychiatric consultation structure
    sections = {
        "synthesis_section": _build_synthesis(data),
        "demographics_section": _build_demographics(data),
        "history_section": _build_history(data),
        "episode_criteria_section": _build_episode_criteria(data),
        "suicide_section": _build_suicide(data),
        "clinical_notes_section": _build_clinical_notes(data),
        "mood_section": _build_mood(data),
        "functional_section": _build_functional(data),
        "anxiety_impulsivity_section": _build_anxiety_impulsivity(data),
        "sleep_section": _build_sleep(data),
        "cognitive_section": _build_cognitive(data),
        "additional_neuropsych_section": _build_additional_neuropsych(data),
        "biology_section": _build_biology(data),
        "treatment_section": _build_treatment(data),
        "lifetime_medication_section": _build_lifetime_medications(data),
        "non_pharm_treatment_section": _build_non_pharm(data),
        "treatment_response_section": _build_treatment_response(data),
        "comorbidity_section": _build_comorbidity(data),
        "substance_section": _build_substance_assessment(data),
        "trauma_section": _build_trauma(data),
        "screening_section": _build_screening(data),
        "diva_section": _build_diva(data),
        "extended_family_section": _build_extended_family(data),
        "risk_section": _build_risk(data),
    }

    vignette_parts = [s for s in sections.values() if s]
    full_vignette = "\n\n".join(vignette_parts)

    return PatientProfile(
        **sections,
        full_vignette=full_vignette,
        data=data,
    )
