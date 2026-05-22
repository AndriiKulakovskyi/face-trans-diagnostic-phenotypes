"""Build comprehensive patient profiles and French clinical vignettes for SZ.

Follows the same structure as bp_profile_builder but adapted for schizophrenia:
- Psychotic symptoms replace mood episode criteria
- PANSS/Calgary replace MADRS/YMRS as primary instruments
- Movement disorders and insight sections added
- Clinical discordance checks are SZ-specific
- Metabolic syndrome flagging (critical for antipsychotic-treated patients)
"""

from __future__ import annotations

from face_rlvr.profiles.bp_profile_builder import (
    PatientProfile,
    _fmt,
    _score_line,
    _available_scores,
    _get_score,
    _get_severity,
    _is_female,
    _agree,
)
from face_rlvr.profiles.common_instruments import ScoreInterpretation
from face_rlvr.profiles.common_extractors import (
    compute_bmi_category,
    detect_metabolic_syndrome,
    compute_framingham_risk,
    check_medication_lab_alerts,
    check_drug_interactions,
    detect_floor_ceiling_effects,
    compute_cognitive_z_score,
)
from face_rlvr.profiles.sz_extractor import SZPatientData
from face_rlvr.profiles.sz_instruments import SZ_INSTRUMENTS


# ─── Helpers ──────────────────────────────────────────────────────────────────

_UNKNOWN_VALUES = {"ne sais pas", "ne sait pas", "unknown", "inconnu", "nsp", "?", "999", "999.0"}


def _is_unknown(val: str | None) -> bool:
    """Check if a value is a 'don't know' placeholder."""
    if val is None:
        return True
    return val.strip().lower() in _UNKNOWN_VALUES


def _get_label_fr(scores: dict[str, ScoreInterpretation], key: str) -> str:
    """Get the French severity label from a scores dict."""
    interp = scores.get(key)
    if interp and interp.score_available:
        return interp.severity_label_fr.lower()
    return ""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION BUILDERS
# ═════════════════════════════════════════════════════════════════════════════


def _build_synthesis(data: SZPatientData) -> str:
    parts = []

    panss = _get_score(data.psychosis_scores, "PANSS")
    calgary = _get_score(data.depression_scores, "Calgary")
    calgary_sev = _get_severity(data.depression_scores, "Calgary")

    arm = data.demographics.arm or "schizophrénie"
    state_parts = []
    if panss is not None:
        label = _get_label_fr(data.psychosis_scores, "PANSS")
        state_parts.append(f"symptômes psychotiques — {label} (PANSS = {_fmt(panss)})")
    if calgary is not None and calgary_sev != "none":
        label = _get_label_fr(data.depression_scores, "Calgary")
        state_parts.append(f"{label} (Calgary = {_fmt(calgary)})")

    sent1 = f"Patient {_agree('suivi', 'suivie', data)} pour {arm}"
    if state_parts:
        sent1 += f", {', '.join(state_parts)}"
    sent1 += "."

    si = data.suicide_indicators
    sh = data.suicide_history
    if sh.ever_attempted:
        sent1 += " ATCD de tentative de suicide."
    elif si.get("madrs_suicide_elevated"):
        sent1 += " Item suicidaire élevé."

    parts.append(sent1)

    # Key findings
    concerns = []
    psp = _get_score(data.functioning_scores, "PSP")
    if psp is not None and psp <= 30:
        concerns.append("fonctionnement sévèrement altéré")
    aims = _get_score(data.movement_scores, "AIMS")
    if aims is not None and aims >= 1:
        concerns.append("dyskinésie tardive")
    bars_sev = _get_severity(data.movement_scores, "BARS")
    if bars_sev in ("moderate", "severe"):
        concerns.append("akathisie")
    insight = data.insight
    if insight.awareness_of_illness is not None and insight.awareness_of_illness >= 2:
        concerns.append("insight altéré")
    # Metabolic risk in synthesis
    bio = data.biology
    bmi = bio.vitals.get("bmi")
    if bmi is not None and bmi >= 30:
        concerns.append("obésité (syndrome métabolique à évaluer)")
    if concerns:
        parts.append("Facteurs notables : " + ", ".join(concerns) + ".")

    t = data.treatments
    if t.on_clozapine:
        parts.append(f"Sous clozapine (taux = {_fmt(t.clozapine_plasma)} ng/mL)." if t.clozapine_plasma else "Sous clozapine.")

    return "Synthèse clinique : " + " ".join(parts)


def _build_demographics(data: SZPatientData) -> str:
    d = data.demographics
    sex = d.sex_label_fr or "sexe non précisé"
    arm = d.arm or "schizophrénie"
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


def _build_history(data: SZPatientData) -> str:
    h = data.psychotic_history
    lines = []

    # Fix 6: Always show age of onset when available
    if h.age_onset_sz is not None and h.age_onset_sz < 900:  # filter 999 placeholder
        lines.append(f"  - Début du trouble à {h.age_onset_sz} ans")
    if h.n_psychotic_episodes is not None:
        lines.append(f"  - Nombre d'épisodes psychotiques : {h.n_psychotic_episodes}")
    # Fix 4: Only show lifetime episodes if different from n_psychotic_episodes
    if h.n_psychotic_episodes_lifetime:
        try:
            lt_val = int(h.n_psychotic_episodes_lifetime)
            if h.n_psychotic_episodes is not None and lt_val != h.n_psychotic_episodes:
                lines.append(f"  - ⚠ Épisodes psychotiques (vie entière, source alternative) : {lt_val}")
            elif h.n_psychotic_episodes is None:
                lines.append(f"  - Épisodes psychotiques (vie entière) : {lt_val}")
        except (ValueError, TypeError):
            lines.append(f"  - Épisodes psychotiques (vie entière) : {h.n_psychotic_episodes_lifetime}")

    if h.symptom_evolution_mode:
        lines.append(f"  - Mode évolutif : {h.symptom_evolution_mode}")

    hosp = data.hospitalization
    if hosp.ever_hospitalized and hosp.n_hospitalizations_lifetime:
        hosp_line = f"  - Hospitalisations : {hosp.n_hospitalizations_lifetime} (vie entière)"
        if hosp.n_hospitalizations_last_year:
            hosp_line += f", dont {hosp.n_hospitalizations_last_year} dans l'année écoulée"
        lines.append(hosp_line)

    if not lines:
        return "Antécédents psychiatriques : données non disponibles."
    return "Antécédents psychiatriques :\n" + "\n".join(lines)


def _build_psychotic_symptoms(data: SZPatientData) -> str:
    ps = data.psychotic_symptoms
    lines = []

    positive = ps.positive_symptoms
    if positive:
        lines.append(f"  Symptômes positifs : {', '.join(positive)}")

    negative = ps.negative_symptoms
    if negative:
        lines.append(f"  Symptômes négatifs : {', '.join(negative)}")

    disorg = ps.disorganization_symptoms
    if disorg:
        lines.append(f"  Désorganisation : {', '.join(disorg)}")

    if ps.catatonia:
        lines.append("  Catatonie : présente")

    if not lines:
        return ""
    return "Symptômes psychotiques actuels :\n" + "\n".join(lines)


def _build_psychosis_scores(data: SZPatientData) -> str:
    available = [s for s in _available_scores(data.psychosis_scores) if not s.suspect_value]
    if not available:
        return ""
    lines = ["Évaluation de la psychose (PANSS) :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
        if interp.instrument == "PANSS" and interp.subscales:
            for sub_name, sub_val in interp.subscales.items():
                if sub_val is not None:
                    lines.append(f"    {sub_name} : {_fmt(sub_val)}")
    return "\n".join(lines)


def _build_depression_and_mood(data: SZPatientData) -> str:
    """Fix 2: Combine Calgary depression + YMRS mood into one integrated section."""
    dep_available = _available_scores(data.depression_scores)
    mood_available = _available_scores(data.mood_scores)

    if not dep_available and not mood_available:
        return ""

    lines = ["Évaluation thymique :"]
    for interp in dep_available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    for interp in mood_available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_suicide(data: SZPatientData) -> str:
    si = data.suicide_indicators
    sh = data.suicide_history
    lines = []

    if si.get("madrs_suicide_elevated"):
        lines.append(f"  - Item suicidaire élevé ({si.get('madrs_suicide_item', '?')}/6)")

    if sh.ever_attempted:
        n = sh.n_attempts
        lines.append(f"  - Antécédent de tentative(s) de suicide : {n if n else 'oui'}")
        if sh.has_violent_attempts and sh.n_violent_attempts:
            lines.append(f"    dont {sh.n_violent_attempts} TS violente(s)")
    elif sh.ever_attempted is False and sh.ever_thought_suicide:
        lines.append("  - Antécédent d'idéation suicidaire au cours de la vie (sans passage à l'acte)")

    if not lines:
        return ""
    return "Évaluation suicidaire :\n" + "\n".join(lines)


def _build_clinical_notes(data: SZPatientData) -> str:
    notes = []

    panss = _get_score(data.psychosis_scores, "PANSS")
    panss_sev = _get_severity(data.psychosis_scores, "PANSS")
    cgi = _get_score(data.global_scores, "CGI-S")
    calgary = _get_score(data.depression_scores, "Calgary")
    calgary_sev = _get_severity(data.depression_scores, "Calgary")
    panss_n = _get_score(data.psychosis_scores, "PANSS-N")
    panss_n_sev = _get_severity(data.psychosis_scores, "PANSS-N")

    # 1. CGI vs PANSS discordance
    if cgi is not None and panss is not None:
        if cgi >= 4 and panss_sev == "mild":
            notes.append(
                f"Discordance CGI-S ({_fmt(cgi)}) avec PANSS ({_fmt(panss)}, {_get_label_fr(data.psychosis_scores, 'PANSS')}) : "
                f"la sévérité globale excède les scores symptomatiques."
            )

    # 2. Calgary depression vs PANSS negative
    if calgary is not None and panss_n is not None:
        if calgary_sev in ("mild", "moderate_severe") and panss_n_sev in ("moderate", "severe"):
            notes.append(
                f"Calgary ({_fmt(calgary)}, {_get_label_fr(data.depression_scores, 'Calgary')}) et "
                f"PANSS-N ({_fmt(panss_n)}, {_get_label_fr(data.psychosis_scores, 'PANSS-N')}) "
                f"tous deux élevés : distinguer la dépression des symptômes négatifs."
            )

    # 3. Insight vs symptom severity
    insight = data.insight
    if insight.awareness_of_illness is not None and panss is not None:
        if insight.awareness_of_illness >= 2 and panss_sev in ("mild",):
            notes.append(
                "Insight altéré malgré des symptômes psychotiques légers : "
                "évaluer l'adhésion thérapeutique et le risque de rechute."
            )

    # 4. Movement disorders
    aims = _get_score(data.movement_scores, "AIMS")
    bars_sev = _get_severity(data.movement_scores, "BARS")
    if aims is not None and aims >= 1:
        notes.append(
            "Dyskinésie tardive détectée (AIMS positif) : "
            "réévaluer le rapport bénéfice/risque du traitement antipsychotique."
        )
    if bars_sev in ("moderate", "severe"):
        notes.append(
            f"Akathisie {_get_label_fr(data.movement_scores, 'BARS')} (BARS) : "
            f"risque de mauvaise observance et de détresse."
        )

    # Fix 3: S-QoL vs EQ-5D discordance
    sqol = _get_score(data.functioning_scores, "S-QoL")
    sqol_sev = _get_severity(data.functioning_scores, "S-QoL")
    eq5d = _get_score(data.functioning_scores, "EQ-5D")
    eq5d_sev = _get_severity(data.functioning_scores, "EQ-5D")

    if sqol is not None and eq5d is not None:
        eq5d_interp = data.functioning_scores.get("EQ-5D")
        if not (eq5d_interp and eq5d_interp.suspect_value):
            if sqol_sev == "good" and eq5d_sev == "poor":
                notes.append(
                    f"Discordance S-QoL ({sqol:.1f}, {_get_label_fr(data.functioning_scores, 'S-QoL')}) / "
                    f"EQ-5D ({eq5d:.2f}, {_get_label_fr(data.functioning_scores, 'EQ-5D')}) : "
                    f"les deux mesures de qualité de vie divergent."
                )
            elif sqol_sev == "poor" and eq5d_sev == "good":
                notes.append(
                    f"Discordance S-QoL ({sqol:.1f}, {_get_label_fr(data.functioning_scores, 'S-QoL')}) / "
                    f"EQ-5D ({eq5d:.2f}, {_get_label_fr(data.functioning_scores, 'EQ-5D')}) : "
                    f"les deux mesures de qualité de vie divergent."
                )

    # PSP vs S-QoL discordance
    psp = _get_score(data.functioning_scores, "PSP")
    psp_sev = _get_severity(data.functioning_scores, "PSP")
    if psp is not None and sqol is not None:
        if psp_sev in ("marked", "severe") and sqol_sev == "good":
            notes.append(
                f"Discordance PSP ({_fmt(psp)}, {_get_label_fr(data.functioning_scores, 'PSP')}) / "
                f"S-QoL ({sqol:.1f}, {_get_label_fr(data.functioning_scores, 'S-QoL')}) : "
                f"le retentissement fonctionnel objectivé ne se reflète pas dans la qualité de vie perçue."
            )

    # Fix 4: Contradictory episode counts
    h = data.psychotic_history
    if h.n_psychotic_episodes is not None and h.n_psychotic_episodes_lifetime:
        try:
            lt_val = int(h.n_psychotic_episodes_lifetime)
            if lt_val != h.n_psychotic_episodes:
                notes.append(
                    f"Incohérence : {h.n_psychotic_episodes} épisodes (psychotic_nb) vs "
                    f"{lt_val} (evnum_tbpsy_lt). Vérifier les sources de données."
                )
        except (ValueError, TypeError):
            pass

    # Phase 6: Floor/ceiling effects across all SZ score dicts
    all_scores: dict[str, ScoreInterpretation] = {}
    for score_dict in [
        data.psychosis_scores, data.depression_scores, data.global_scores,
        data.functioning_scores, data.movement_scores, data.mood_scores,
        data.sleep_scores, data.adherence_scores, data.trauma_scores,
        data.substance_scores, data.screening_scores,
    ]:
        all_scores.update(score_dict)
    fc_effects = detect_floor_ceiling_effects(all_scores, SZ_INSTRUMENTS)
    if fc_effects:
        notes.append(
            "Effets plancher/plafond détectés : " + " ; ".join(fc_effects) + "."
        )

    # Phase 8: PANSS-N elevated items — clinical note when negative symptoms are prominent
    if panss_n is not None and panss_n_sev in ("moderate", "severe"):
        notes.append(
            f"PANSS-N élevée ({_fmt(panss_n)}, {_get_label_fr(data.psychosis_scores, 'PANSS-N')}) : "
            f"évaluer les symptômes négatifs primaires vs secondaires "
            f"(dépression, sédation médicamenteuse, retrait social lié à l'environnement)."
        )

    # Drug interaction flags
    treatment_flags = {
        "on_clozapine": data.treatments.on_clozapine,
        "on_antipsychotic": data.treatments.n_antipsychotics > 0 if data.treatments.n_antipsychotics else False,
    }
    drug_alerts = check_drug_interactions(treatment_flags)
    if drug_alerts:
        for alert in drug_alerts:
            notes.append(alert)

    if not notes:
        return ""
    return "Notes cliniques :\n" + "\n".join(f"  * {n}" for n in notes)


def _build_functioning(data: SZPatientData) -> str:
    available = [s for s in _available_scores(data.functioning_scores) if not s.suspect_value]
    if not available:
        return ""
    lines = ["Fonctionnement :"]
    for interp in available:
        lines.append(_score_line(interp))
    return "\n".join(line for line in lines if line)


def _build_insight_section(data: SZPatientData) -> str:
    ins = data.insight
    items = [
        (ins.awareness_of_illness, "Conscience du trouble"),
        (ins.awareness_of_medication_effect, "Conscience de l'effet du traitement"),
        (ins.awareness_of_social_consequences, "Conscience des conséquences sociales"),
    ]
    lines = []
    for val, label in items:
        if val is not None:
            level = "absent" if val == 0 else ("partiel" if val == 1 else ("altéré" if val == 2 else "très altéré"))
            lines.append(f"  - {label} : {level} ({_fmt(val)}/3)")

    if not lines:
        return ""
    return "Insight (SUMD) :\n" + "\n".join(lines)


def _build_movement_disorders(data: SZPatientData) -> str:
    available = _available_scores(data.movement_scores)
    if not available:
        return ""
    lines = ["Troubles du mouvement :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_sleep(data: SZPatientData) -> str:
    available = _available_scores(data.sleep_scores)
    if not available:
        return ""
    lines = ["Sommeil :"]
    for interp in available:
        lines.append(_score_line(interp))
    return "\n".join(line for line in lines if line)


def _build_cognitive(data: SZPatientData) -> str:
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
        if cp.tmt_b_minus_a is not None:
            if cp.tmt_b_minus_a > 90:
                lines.append("    Flexibilité cognitive altérée (B-A > 90 s)")
            elif cp.tmt_b_minus_a > 60:
                lines.append("    Flexibilité cognitive limite (B-A 60-90 s)")
        # Z-score interpretations
        if cp.tmt_a_seconds is not None:
            z_a, z_a_interp = compute_cognitive_z_score("TMT-A", cp.tmt_a_seconds, age)
            if z_a is not None:
                lines.append(f"    TMT-A : {z_a_interp}")
        if cp.tmt_b_seconds is not None:
            z_b, z_b_interp = compute_cognitive_z_score("TMT-B", cp.tmt_b_seconds, age)
            if z_b is not None:
                lines.append(f"    TMT-B : {z_b_interp}")

    if cp.stroop_interference is not None:
        stroop_z, stroop_interp = compute_cognitive_z_score("Stroop_interference", cp.stroop_interference, age)
        stroop_extra = ""
        if stroop_z is not None:
            stroop_extra = f" (z = {stroop_z:.1f}, {stroop_interp})"
        lines.append(f"  - Stroop interférence : {_fmt(cp.stroop_interference)}{stroop_extra}")

    if cp.cvlt_total_learning is not None:
        cvlt_parts = [f"apprentissage = {_fmt(cp.cvlt_total_learning)}"]
        if cp.cvlt_long_delay_free is not None:
            cvlt_parts.append(f"rappel différé = {_fmt(cp.cvlt_long_delay_free)}")
        lines.append(f"  - CVLT : {', '.join(cvlt_parts)}")

    wais_parts = []
    if cp.wais_similarities_std is not None:
        wais_parts.append(f"similitudes = {_fmt(cp.wais_similarities_std)}")
    if cp.wais_vocabulary_std is not None:
        wais_parts.append(f"vocabulaire = {_fmt(cp.wais_vocabulary_std)}")
    if wais_parts:
        lines.append(f"  - WAIS (notes standard) : {', '.join(wais_parts)}")

    if not lines:
        return ""
    return "Profil cognitif :\n" + "\n".join(lines)


def _build_biology(data: SZPatientData) -> str:
    bio = data.biology
    if not bio.values and not bio.vitals:
        return ""
    lines = []

    vital_parts = []
    v = bio.vitals
    if "bmi" in v:
        bmi = v["bmi"]
        bmi_result = compute_bmi_category(bmi)
        bmi_label = f" ({bmi_result[1]})" if bmi_result else ""
        vital_parts.append(f"IMC = {_fmt(bmi, 1)} kg/m²{bmi_label}")
    if "weight_kg" in v:
        vital_parts.append(f"poids = {_fmt(v['weight_kg'], 1)} kg")
    if "waist_cm" in v:
        vital_parts.append(f"tour de taille = {_fmt(v['waist_cm'])} cm")
    if "sbp_supine" in v and "dbp_supine" in v:
        vital_parts.append(f"PA = {_fmt(v['sbp_supine'])}/{_fmt(v['dbp_supine'])} mmHg")
    if vital_parts:
        lines.append("  Constantes : " + ", ".join(vital_parts))

    if bio.ecg and "qtc" in bio.ecg:
        qtc = bio.ecg["qtc"]
        flag = " (allongé)" if qtc > 450 else ""
        lines.append(f"  ECG : QTc = {_fmt(qtc)} ms{flag}")

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

    # Metabolic syndrome detection (critical for antipsychotic-treated patients)
    sex = data.demographics.sex
    sbp = v.get("sbp_supine")
    dbp = v.get("dbp_supine")
    lab_dict = {lv.name: lv for lv in bio.values}
    trig_lv = lab_dict.get("Triglycerides") or lab_dict.get("Triglycérides")
    hdl_lv = lab_dict.get("HDL") or lab_dict.get("HDL-cholestérol")
    glucose_lv = lab_dict.get("Glucose") or lab_dict.get("Glycémie à jeun")
    ms_positive, ms_criteria = detect_metabolic_syndrome(
        waist_cm=v.get("waist_cm"),
        sex=sex,
        trig=trig_lv.value if trig_lv else None,
        hdl=hdl_lv.value if hdl_lv else None,
        sbp=sbp,
        dbp=dbp,
        glucose=glucose_lv.value if glucose_lv else None,
    )
    if ms_positive:
        lines.append(f"  Syndrome métabolique (IDF/ATP-III) : OUI ({len(ms_criteria)}/5 critères)")
        for crit in ms_criteria:
            lines.append(f"    - {crit}")
    elif ms_criteria:
        lines.append(f"  Syndrome métabolique : NON ({len(ms_criteria)}/5 critères, seuil = 3)")

    # Framingham cardiovascular risk
    chol_lv = lab_dict.get("Total cholesterol") or lab_dict.get("Cholestérol total")
    smoking = data.substance_use.tobacco_current
    framingham_risk, framingham_cat = compute_framingham_risk(
        age=data.demographics.age,
        sex=sex,
        total_chol=chol_lv.value if chol_lv else None,
        hdl=hdl_lv.value if hdl_lv else None,
        sbp=sbp,
        on_bp_treatment=False,
        smoking=smoking,
        diabetes=False,
    )
    if framingham_risk is not None:
        lines.append(f"  Risque cardiovasculaire Framingham : {framingham_risk:.0f}% ({framingham_cat})")

    # Medication-lab alerts (especially clozapine monitoring)
    treatment_flags = {
        "on_clozapine": data.treatments.on_clozapine,
        "on_antipsychotic": data.treatments.n_antipsychotics > 0 if data.treatments.n_antipsychotics else False,
    }
    med_alerts = check_medication_lab_alerts(treatment_flags, bio.values, sex)
    if med_alerts:
        lines.append("  Alertes médicamenteuses :")
        for alert in med_alerts:
            lines.append(f"    - {alert}")

    if not lines:
        return ""
    return "Bilan somatique :\n" + "\n".join(lines)


def _build_treatment(data: SZPatientData) -> str:
    t = data.treatments
    lines = []

    if t.on_clozapine:
        if t.clozapine_plasma:
            lines.append(f"  - Clozapine : taux plasmatique = {_fmt(t.clozapine_plasma)} ng/mL")
        else:
            lines.append("  - Clozapine")

    counts = [
        (t.n_anticholinergics, "anticholinergique(s)"),
        (t.n_anxiolytics, "anxiolytique(s)"),
        (t.n_hypnotics, "hypnotique(s)"),
        (t.n_mood_stabilizers, "thymorégulateur(s)"),
        (t.n_antidepressants, "antidépresseur(s)"),
    ]
    for n, label in counts:
        if n is not None and n > 0:
            lines.append(f"  - {n} {label}")

    if t.medication_adherence and t.medication_adherence.score_available:
        lines.append(f"  - Observance : {t.medication_adherence.clinical_interpretation_fr}")

    if not lines:
        return ""
    return "Traitement actuel :\n" + "\n".join(lines)


def _build_comorbidity(data: SZPatientData) -> str:
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
    # Fix 8: Add tobacco from Fagerström when elevated
    fag = data.substance_scores.get("Fagerström")
    if fag and fag.score_available and fag.severity_code not in ("low",) and "tabac" not in su_parts:
        su_parts.append("tabac (dépendance nicotinique)")
    if su_parts:
        lines.append("  Usage de substances : " + ", ".join(su_parts))

    if not lines:
        return ""
    return "Comorbidités :\n" + "\n".join(lines)


def _build_substance(data: SZPatientData) -> str:
    available = _available_scores(data.substance_scores)
    if not available:
        return ""
    lines = ["Évaluation addictologique :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_trauma(data: SZPatientData) -> str:
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


def _build_screening(data: SZPatientData) -> str:
    available = _available_scores(data.screening_scores)
    if not available:
        return ""
    lines = ["Dépistages :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_family(data: SZPatientData) -> str:
    fh = data.family_history
    lines = []

    # Fix 7: Filter "Ne sais pas" / unknown values
    fam_parts = []
    if fh.maternal_psychiatric and not _is_unknown(fh.maternal_psychiatric):
        fam_parts.append(f"mère : {fh.maternal_psychiatric}")
    if fh.paternal_psychiatric and not _is_unknown(fh.paternal_psychiatric):
        fam_parts.append(f"père : {fh.paternal_psychiatric}")
    if fam_parts:
        lines.append(f"  - Parents : {', '.join(fam_parts)}")

    gp_with_disorder = [
        r for r in fh.relatives
        if r.psychiatric_disorder and r.relation.startswith("g") and not _is_unknown(r.psychiatric_disorder)
    ]
    for r in gp_with_disorder:
        lines.append(f"  - {r.relation_fr} : {r.psychiatric_disorder}")

    if fh.n_siblings is not None:
        affected = f", dont {fh.n_siblings_affected} atteint(s)" if fh.n_siblings_affected else ""
        lines.append(f"  - Fratrie : {fh.n_siblings} membre(s){affected}")

    if fh.maternal_suicide or fh.paternal_suicide:
        lines.append("  - Antécédents familiaux de suicide")

    if not lines:
        return ""
    return "Antécédents familiaux élargis :\n" + "\n".join(lines)


def _build_risk(data: SZPatientData) -> str:
    lines = []

    sh = data.suicide_history
    if sh.ever_attempted:
        lines.append("  - Antécédent de tentative de suicide")

    insight = data.insight
    if insight.awareness_of_illness is not None and insight.awareness_of_illness >= 2:
        lines.append("  - Insight altéré (risque de non-adhésion)")

    mars = data.adherence_scores.get("MARS")
    if mars and mars.score_available and mars.severity_code == "poor":
        lines.append("  - Mauvaise observance thérapeutique")

    aims = _get_score(data.movement_scores, "AIMS")
    if aims is not None and aims >= 1:
        lines.append("  - Dyskinésie tardive")

    # Fix 5: Metabolic syndrome flag
    bio = data.biology
    bmi = bio.vitals.get("bmi")
    waist = bio.vitals.get("waist_cm")
    metabolic_flags = 0
    if bmi is not None and bmi >= 30:
        metabolic_flags += 1
    if waist is not None and waist > 102:  # male cutoff; female = 88
        metabolic_flags += 1
    abnormal_lipids = any(
        lv.is_abnormal and lv.name in ("Triglycerides", "HDL", "Glucose", "Total cholesterol")
        for lv in bio.values
    )
    if abnormal_lipids:
        metabolic_flags += 1
    if metabolic_flags >= 2:
        lines.append("  - Syndrome métabolique probable (obésité + anomalies lipidiques/glycémiques)")
    elif bmi is not None and bmi >= 30:
        lines.append("  - Obésité sous antipsychotique (risque métabolique)")

    ctq = data.trauma_scores.get("CTQ")
    if ctq and ctq.score_available and ctq.severity_code in ("moderate_severe", "severe_extreme"):
        lines.append("  - Traumatismes de l'enfance significatifs")

    fh = data.family_history
    if fh.maternal_suicide or fh.paternal_suicide:
        lines.append("  - Antécédents familiaux de suicide")
    # Fix 7: Filter unknown family psychiatric history from risk
    fam_parts = []
    if fh.maternal_psychiatric and not _is_unknown(fh.maternal_psychiatric):
        fam_parts.append(f"mère : {fh.maternal_psychiatric}")
    if fh.paternal_psychiatric and not _is_unknown(fh.paternal_psychiatric):
        fam_parts.append(f"père : {fh.paternal_psychiatric}")
    if fam_parts:
        lines.append(f"  - ATCD psychiatriques familiaux : {', '.join(fam_parts)}")

    if not lines:
        return ""
    return "Facteurs de risque :\n" + "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN BUILDER
# ═════════════════════════════════════════════════════════════════════════════


def build_sz_profile(data: SZPatientData) -> PatientProfile:
    """Build a comprehensive SZ patient profile.

    Section order follows standard psychiatric consultation structure,
    harmonised with BP vignette style. The PatientProfile field names
    are reused as generic slots — the vignette order is controlled by
    the ordered_sections list, not by field names.
    """
    # Build all sections
    all_sections = {
        "synthesis": _build_synthesis(data),
        "demographics": _build_demographics(data),
        "history": _build_history(data),
        "psychotic_symptoms": _build_psychotic_symptoms(data),
        "suicide": _build_suicide(data),
        "clinical_notes": _build_clinical_notes(data),
        "psychosis_scores": _build_psychosis_scores(data),
        "depression_mood": _build_depression_and_mood(data),
        "functioning": _build_functioning(data),
        "insight": _build_insight_section(data),
        "movement": _build_movement_disorders(data),
        "sleep": _build_sleep(data),
        "cognitive": _build_cognitive(data),
        "biology": _build_biology(data),
        "treatment": _build_treatment(data),
        "comorbidity": _build_comorbidity(data),
        "substance": _build_substance(data),
        "trauma": _build_trauma(data),
        "screening": _build_screening(data),
        "family": _build_family(data),
        "risk": _build_risk(data),
    }

    # Build vignette in correct order
    ordered_keys = [
        "synthesis", "demographics", "history", "psychotic_symptoms",
        "suicide", "clinical_notes", "psychosis_scores", "depression_mood",
        "functioning", "insight", "movement", "sleep", "cognitive",
        "biology", "treatment", "comorbidity", "substance", "trauma",
        "screening", "family", "risk",
    ]
    vignette_parts = [all_sections[k] for k in ordered_keys if all_sections.get(k)]
    full_vignette = "\n\n".join(vignette_parts)

    # Map to PatientProfile fields (reused as generic slots)
    return PatientProfile(
        synthesis_section=all_sections.get("synthesis", ""),
        demographics_section=all_sections.get("demographics", ""),
        history_section=all_sections.get("history", ""),
        episode_criteria_section=all_sections.get("psychotic_symptoms", ""),
        suicide_section=all_sections.get("suicide", ""),
        clinical_notes_section=all_sections.get("clinical_notes", ""),
        mood_section=all_sections.get("psychosis_scores", ""),
        functional_section=all_sections.get("depression_mood", ""),
        anxiety_impulsivity_section=all_sections.get("functioning", ""),
        sleep_section=all_sections.get("insight", ""),
        cognitive_section=all_sections.get("movement", ""),
        additional_neuropsych_section=all_sections.get("sleep", ""),
        biology_section=all_sections.get("cognitive", ""),
        treatment_section=all_sections.get("biology", ""),
        lifetime_medication_section=all_sections.get("treatment", ""),
        comorbidity_section=all_sections.get("comorbidity", ""),
        substance_section=all_sections.get("substance", ""),
        trauma_section=all_sections.get("trauma", ""),
        screening_section=all_sections.get("screening", ""),
        extended_family_section=all_sections.get("family", ""),
        risk_section=all_sections.get("risk", ""),
        full_vignette=full_vignette,
        data=data,
    )
