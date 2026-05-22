"""Build comprehensive patient profiles and French clinical vignettes for DR.

Follows the same structure as bp_profile_builder but adapted for
treatment-resistant depression:
- Treatment resistance staging replaces bipolar episode history
- MADRS/QIDS/ERD/SHAPS as primary depression instruments
- CSSRS binary ideation (cssrs01-cssrs05) instead of BP's css0101-css0106
- BAS/SPIN anxiety instruments added
- LEAPS occupational functioning added
- Clinical discordance checks are DR-specific
"""

from __future__ import annotations

from face_rlvr.profiles.bp_profile_builder import (
    PatientProfile,
    _fmt,
    _score_line,
    _available_scores,
    _get_score,
    _get_severity,
    _get_label_fr,
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
from face_rlvr.profiles.dr_extractor import DRPatientData
from face_rlvr.profiles.dr_instruments import DR_INSTRUMENTS


# ─── Helpers ──────────────────────────────────────────────────────────────────

_UNKNOWN_VALUES = {"ne sais pas", "ne sait pas", "unknown", "inconnu", "nsp", "?", "999", "999.0"}


def _is_unknown(val: str | None) -> bool:
    """Check if a value is a 'don't know' placeholder."""
    if val is None:
        return True
    return val.strip().lower() in _UNKNOWN_VALUES


# ═════════════════════════════════════════════════════════════════════════════
# SECTION BUILDERS
# ═════════════════════════════════════════════════════════════════════════════


def _build_synthesis(data: DRPatientData) -> str:
    """2-3 sentence clinical summary placed first in the vignette."""
    parts = []

    madrs = _get_score(data.depression_scores, "MADRS")
    madrs_sev = _get_severity(data.depression_scores, "MADRS")

    # Current depression severity
    severity_fr = ""
    if madrs_sev == "severe":
        severity_fr = "severe"
    elif madrs_sev == "moderate":
        severity_fr = "moderee"
    elif madrs_sev == "mild":
        severity_fr = "legere"
    elif madrs_sev == "normal":
        severity_fr = "en remission"
    else:
        severity_fr = "non evaluee"

    sent1 = f"Patient {_agree('suivi', 'suivie', data)} pour depression resistante, actuellement en depression {severity_fr}"
    if madrs is not None:
        sent1 += f" (MADRS = {_fmt(madrs)})"
    sent1 += "."

    # Treatment resistance info
    tr = data.treatment_resistance
    if tr.resistance_level:
        sent1 += f" Niveau de resistance : {tr.resistance_level}."

    # Suicide risk
    cssrs = data.cssrs_assessment
    si = data.suicide_indicators
    if cssrs.highest_ideation_level is not None and cssrs.highest_ideation_level >= 3:
        levels_fr = {
            3: "ideation avec methode",
            4: "ideation avec intention",
            5: "ideation avec plan precis",
        }
        label = levels_fr.get(cssrs.highest_ideation_level, "ideation suicidaire")
        sent1 += f" Risque suicidaire : {label} (C-SSRS niveau {cssrs.highest_ideation_level})."
    elif si.get("madrs_suicide_elevated"):
        sent1 += " Item suicidaire MADRS eleve."

    parts.append(sent1)

    # Notable findings
    concerns = []
    ctq_sev = _get_severity(data.trauma_scores, "CTQ")
    if ctq_sev in ("moderate_severe", "severe_extreme"):
        concerns.append("antecedents traumatiques significatifs")
    psqi_sev = _get_severity(data.sleep_scores, "PSQI")
    if psqi_sev in ("poor", "very_poor"):
        concerns.append("troubles du sommeil")
    shaps = data.depression_scores.get("SHAPS")
    if shaps and shaps.score_available and shaps.severity_code == "positive":
        concerns.append("anhedonie significative")
    erd_sev = _get_severity(data.depression_scores, "ERD")
    if erd_sev in ("moderate", "severe"):
        concerns.append("ralentissement psychomoteur")
    if concerns:
        parts.append("Facteurs notables : " + ", ".join(concerns) + ".")

    # Treatment info
    t = data.treatments
    treat_parts = []
    if t.lithium_level is not None:
        treat_parts.append(f"lithium (lithemie = {_fmt(t.lithium_level, 1)} mEq/L)")
    if t.valproate_level is not None:
        treat_parts.append(f"valproate (taux = {_fmt(t.valproate_level, 1)} ug/mL)")
    if t.has_ect:
        treat_parts.append("ECT")
    if treat_parts:
        parts.append("Traitement : " + ", ".join(treat_parts) + ".")

    return "Synthese clinique : " + " ".join(parts)


def _build_demographics(data: DRPatientData) -> str:
    d = data.demographics
    sex = d.sex_label_fr or "sexe non precise"
    arm = d.arm or "depression resistante"
    age = d.age or "?"
    line1 = f"Patient {data.patient_id}, {sex} de {age} ans, {_agree('suivi', 'suivie', data)} pour {arm}."

    details = []
    if d.marital_status:
        details.append(d.marital_status)
    if d.education_level:
        details.append(f"niveau d'etudes : {d.education_level}")
    if d.employment:
        details.append(d.employment)
    if details:
        return line1 + "\n" + ", ".join(details).capitalize() + "."
    return line1


def _build_history(data: DRPatientData) -> str:
    h = data.psychiatric_history
    tr = data.treatment_resistance
    lines = []

    if h.age_first_episode is not None:
        lines.append(f"  - Debut du trouble a {h.age_first_episode} ans")
    if h.illness_duration_years is not None:
        lines.append(f"  - Duree d'evolution : {_fmt(h.illness_duration_years)} ans")

    # Treatment resistance staging
    if tr.resistance_level:
        lines.append(f"  - Niveau de resistance au traitement : {tr.resistance_level}")
    if tr.current_episode_number is not None:
        lines.append(f"  - Episode actuel : numero {tr.current_episode_number}")
    if tr.current_episode_duration_months is not None:
        lines.append(f"  - Duree de l'episode actuel : {_fmt(tr.current_episode_duration_months)} mois")
    if tr.age_first_treatment is not None:
        lines.append(f"  - Age au premier traitement : {tr.age_first_treatment} ans")
    if tr.total_treatment_duration_months is not None:
        lines.append(f"  - Duree totale de traitement : {_fmt(tr.total_treatment_duration_months)} mois")
    if tr.has_psychotic_features:
        lines.append("  - Caracteristiques psychotiques presentes")
    if tr.achieved_complete_remission is not None:
        if tr.achieved_complete_remission:
            lines.append("  - Remission complete atteinte au moins une fois")
        else:
            lines.append("  - Jamais de remission complete")
    if tr.sachs_score is not None:
        sachs_val = tr.sachs_score
        if sachs_val <= 2:
            sachs_stage = "Résistance non établie"
        elif sachs_val == 3:
            sachs_stage = "Stade I (échec d'un essai adéquat)"
        elif sachs_val <= 5:
            sachs_stage = "Stade II (échec de deux essais adéquats)"
        elif sachs_val <= 7:
            sachs_stage = "Stade III (résistance sévère)"
        else:
            sachs_stage = "Stade IV-V (résistance extrême)"
        lines.append(f"  - Score de Sachs (staging de resistance) : {_fmt(sachs_val)} — {sachs_stage}")
    if tr.antidepressant_response:
        lines.append(f"  - Reponse aux antidepresseurs : {tr.antidepressant_response}")

    if h.current_episode_severity:
        lines.append(f"  - Severite de l'episode actuel : {h.current_episode_severity}")

    # Episode counts
    ec = data.episode_counts
    if ec:
        ep_parts = []
        for ep_type, count in [("depressive", "depressifs"), ("manic", "maniaques"),
                                ("hypomanic", "hypomaniaques"), ("mixed", "mixtes")]:
            val = ec.get(ep_type)
            if val is not None and val > 0:
                ep_parts.append(f"{val} {count}")
        if ep_parts:
            lines.append(f"  - Episodes au cours de la vie : {', '.join(ep_parts)}")

    hosp = data.hospitalization
    if hosp.ever_hospitalized and hosp.n_hospitalizations_lifetime:
        hosp_line = f"  - Hospitalisations : {hosp.n_hospitalizations_lifetime} (vie entiere)"
        if hosp.n_hospitalizations_last_year:
            hosp_line += f", dont {hosp.n_hospitalizations_last_year} dans l'annee ecoulee"
        lines.append(hosp_line)

    if not lines:
        return "Antecedents psychiatriques : donnees non disponibles."
    return "Antecedents psychiatriques :\n" + "\n".join(lines)


def _build_episode_criteria(data: DRPatientData) -> str:
    ec = data.current_episode_criteria
    lines = []

    dep_items = [
        (ec.depressed_mood, "humeur depressive"),
        (ec.anhedonia, "anhedonie"),
        (ec.weight_change, "modification ponderale"),
        (ec.sleep_disturbance, "troubles du sommeil"),
        (ec.psychomotor_change, "modification psychomotrice"),
        (ec.fatigue, "fatigue"),
        (ec.worthlessness, "culpabilite/devalorisation"),
        (ec.concentration_difficulty, "difficultes de concentration"),
        (ec.suicidal_thoughts, "idees de mort/suicide"),
    ]
    dep_present = [label for val, label in dep_items if val]
    if dep_present or ec.depressive_symptom_count is not None:
        count = ec.depressive_symptom_count or len(dep_present)
        lines.append(f"  Criteres depressifs : {count}/9 presents")
        if dep_present:
            lines.append(f"    ({', '.join(dep_present)})")

    man_items = [
        (ec.elevated_mood, "humeur elevee"),
        (ec.irritable_mood, "irritabilite"),
        (ec.grandiosity, "idees de grandeur"),
        (ec.decreased_sleep_need, "reduction du besoin de sommeil"),
        (ec.pressured_speech, "logorrhee"),
        (ec.flight_of_ideas, "fuite des idees"),
        (ec.distractibility, "distractibilite"),
        (ec.goal_directed_activity, "activite augmentee"),
        (ec.risky_behavior, "activites a risque"),
    ]
    man_present = [label for val, label in man_items if val]
    if man_present or ec.manic_symptom_count is not None:
        count = ec.manic_symptom_count or len(man_present)
        lines.append(f"  Criteres maniaques : {count}/9 presents")
        if man_present:
            lines.append(f"    ({', '.join(man_present)})")

    mre = data.most_recent_episode
    if mre.episode_type:
        parts = [mre.episode_type]
        if mre.severity:
            parts.append(mre.severity)
        if mre.chronicity:
            parts.append(f"chronicite : {mre.chronicity}")
        if mre.postpartum:
            parts.append("post-partum")
        lines.append(f"  Episode le plus recent : {', '.join(parts)}")

    if not lines:
        return ""
    return "Episode actuel (criteres DSM) :\n" + "\n".join(lines)


def _build_suicide(data: DRPatientData) -> str:
    """Suicide assessment — ISF history + CSSRS binary ideation."""
    si = data.suicide_indicators
    sh = data.suicide_history
    cssrs = data.cssrs_assessment
    lines = []

    # C-SSRS binary ideation level
    if cssrs.highest_ideation_level is not None:
        levels_fr = {
            1: "Desir de mort",
            2: "Ideation suicidaire non specifique",
            3: "Ideation avec methode",
            4: "Ideation avec intention",
            5: "Ideation avec plan precis",
        }
        label = levels_fr.get(cssrs.highest_ideation_level, "Non classe")
        lines.append(f"  - C-SSRS niveau {cssrs.highest_ideation_level}/5 : {label}")

    # MADRS suicide item
    if si.get("madrs_suicide_elevated"):
        lines.append(f"  - Item suicidaire MADRS eleve ({si.get('madrs_suicide_item', '?')}/6)")

    # ISF history
    if sh.ever_attempted:
        n = sh.n_attempts
        lines.append(f"  - Antecedent de tentative(s) de suicide : {n if n else 'oui'}")
        if sh.has_violent_attempts and sh.n_violent_attempts:
            lines.append(f"    dont {sh.n_violent_attempts} TS violente(s)")
        if sh.has_serious_attempts and sh.n_serious_attempts:
            lines.append(f"    dont {sh.n_serious_attempts} TS grave(s)")
    elif sh.ever_attempted is False and sh.ever_thought_suicide:
        lines.append("  - Antecedent d'ideation suicidaire au cours de la vie (sans passage a l'acte)")

    if sh.most_serious_method:
        lines.append(f"  - Methode TS la plus grave : {sh.most_serious_method}")

    if not lines:
        return ""
    return "Evaluation suicidaire :\n" + "\n".join(lines)


def _build_clinical_notes(data: DRPatientData) -> str:
    """Detect and flag clinical discordances between instruments."""
    notes = []

    madrs = _get_score(data.depression_scores, "MADRS")
    madrs_sev = _get_severity(data.depression_scores, "MADRS")
    qids = _get_score(data.depression_scores, "QIDS")
    qids_sev = _get_severity(data.depression_scores, "QIDS")

    # 1. MADRS vs QIDS discordance
    if madrs is not None and qids is not None:
        dep_levels = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3, "very_severe": 4}
        m_level = dep_levels.get(madrs_sev, -1)
        q_level = dep_levels.get(qids_sev, -1)
        if m_level != q_level and m_level >= 0 and q_level >= 0:
            higher = "auto-evaluation (QIDS)" if q_level > m_level else "hetero-evaluation (MADRS)"
            notes.append(
                f"Discordance MADRS ({_fmt(madrs)}, {_get_label_fr(data.depression_scores, 'MADRS')}) / "
                f"QIDS ({_fmt(qids)}, {_get_label_fr(data.depression_scores, 'QIDS')}) : "
                f"la {higher} est plus severe."
            )

    # 2. ERD vs MADRS discordance — high retardation with low depression
    erd = _get_score(data.depression_scores, "ERD")
    erd_sev = _get_severity(data.depression_scores, "ERD")
    if erd is not None and madrs is not None:
        if erd_sev in ("moderate", "severe") and madrs_sev in ("normal", "mild"):
            notes.append(
                f"Discordance ERD ({_fmt(erd)}, {_get_label_fr(data.depression_scores, 'ERD')}) / "
                f"MADRS ({_fmt(madrs)}, {_get_label_fr(data.depression_scores, 'MADRS')}) : "
                f"ralentissement psychomoteur important malgre une depression legere "
                f"(composante psychomotrice predominante)."
            )

    # 3. Treatment resistance + poor adherence
    tr = data.treatment_resistance
    mars = data.adherence_scores.get("MARS")
    if tr.is_resistant and mars and mars.score_available and mars.severity_code == "poor":
        notes.append(
            "Depression etiquetee resistante avec mauvaise observance therapeutique (MARS) : "
            "la pseudo-resistance par non-observance doit etre exclue avant de conclure "
            "a une resistance vraie."
        )

    # 4. EGF vs FAST discordance
    egf = _get_score(data.functioning_scores, "EGF")
    egf_sev = _get_severity(data.functioning_scores, "EGF")
    fast = _get_score(data.functioning_scores, "FAST")
    fast_sev = _get_severity(data.functioning_scores, "FAST")
    if egf is not None and fast is not None:
        # EGF high (good functioning) but FAST high (poor functioning) or vice versa
        egf_good = egf_sev in ("superior", "absent_minimal", "slight")
        fast_poor = fast_sev in ("moderate", "severe")
        egf_poor = egf_sev in ("serious", "major_impairment", "serious_impairment", "some_danger", "persistent_danger")
        fast_good = fast_sev in ("none", "mild")
        if (egf_good and fast_poor) or (egf_poor and fast_good):
            notes.append(
                f"Discordance EGF ({_fmt(egf)}, {_get_label_fr(data.functioning_scores, 'EGF')}) / "
                f"FAST ({_fmt(fast)}, {_get_label_fr(data.functioning_scores, 'FAST')}) : "
                f"evaluations divergentes du fonctionnement global."
            )

    # 5. CGI vs symptom scales
    cgi = _get_score(data.global_scores, "CGI-S")
    if cgi is not None and cgi >= 4 and madrs is not None:
        if madrs < 20:
            notes.append(
                f"Discordance CGI-S ({_fmt(cgi)}) avec MADRS ({_fmt(madrs)}) : "
                f"la severite globale excede le score depressif."
            )

    # 6. Floor/ceiling effects across all DR score dicts
    all_scores: dict[str, ScoreInterpretation] = {}
    for score_dict in [
        data.depression_scores, data.mood_scores, data.global_scores,
        data.functioning_scores, data.anxiety_scores, data.sleep_scores,
        data.adherence_scores, data.trauma_scores, data.substance_scores,
        data.self_esteem_scores, data.impulsivity_scores,
    ]:
        all_scores.update(score_dict)
    fc_effects = detect_floor_ceiling_effects(all_scores, DR_INSTRUMENTS)
    if fc_effects:
        notes.append(
            "Effets plancher/plafond detectes : " + " ; ".join(fc_effects) + "."
        )

    # 7. BIS-10 vs MADRS discordance: high impulsivity with low depression = trait impulsivity
    bis_sev = _get_severity(data.impulsivity_scores, "BIS-10")
    if bis_sev == "high" and madrs_sev in ("normal", "mild"):
        bis_score = _get_score(data.impulsivity_scores, "BIS-10")
        notes.append(
            f"Discordance BIS-10 ({_fmt(bis_score)}, impulsivite elevee) / "
            f"MADRS ({_fmt(madrs)}, {_get_label_fr(data.depression_scores, 'MADRS')}) : "
            f"impulsivite-trait probable (independante de l'episode depressif)."
        )

    # 8. ERD vs BDI/QIDS severity discordance
    if erd is not None and qids is not None:
        if erd_sev in ("moderate", "severe") and qids_sev in ("normal", "mild"):
            notes.append(
                f"Discordance ERD ({_fmt(erd)}, {_get_label_fr(data.depression_scores, 'ERD')}) / "
                f"QIDS ({_fmt(qids)}, {_get_label_fr(data.depression_scores, 'QIDS')}) : "
                f"ralentissement objectif sans plainte depressive subjective proportionnelle."
            )
        elif erd_sev in ("normal",) and qids_sev in ("severe", "very_severe"):
            notes.append(
                f"Discordance ERD ({_fmt(erd)}, {_get_label_fr(data.depression_scores, 'ERD')}) / "
                f"QIDS ({_fmt(qids)}, {_get_label_fr(data.depression_scores, 'QIDS')}) : "
                f"plainte depressive subjective elevee sans ralentissement psychomoteur objectif."
            )

    if not notes:
        return ""
    return "Notes cliniques :\n" + "\n".join(f"  * {n}" for n in notes)


def _build_depression(data: DRPatientData) -> str:
    """Depression scores with hetero/auto sub-groups."""
    available = [s for s in _available_scores(data.depression_scores) if not s.suspect_value]
    if not available:
        return ""

    hetero = []
    auto = []
    for interp in available:
        inst = DR_INSTRUMENTS.get(interp.instrument)
        if inst and inst.evaluation_type == "auto":
            auto.append(interp)
        else:
            hetero.append(interp)

    lines = ["Evaluation depressive :"]

    if hetero and auto:
        lines.append("  Hetero-evaluation :")
        for interp in hetero:
            line = _score_line(interp)
            if line:
                lines.append(f"  {line}")
        lines.append("  Auto-evaluation :")
        for interp in auto:
            line = _score_line(interp)
            if line:
                lines.append(f"  {line}")
    else:
        for interp in available:
            line = _score_line(interp)
            if line:
                lines.append(line)

    return "\n".join(lines)


def _build_functioning(data: DRPatientData) -> str:
    available = [s for s in _available_scores(data.functioning_scores) if not s.suspect_value]
    if not available:
        return ""
    lines = ["Fonctionnement :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_anxiety(data: DRPatientData) -> str:
    available = _available_scores(data.anxiety_scores)
    if not available:
        return ""
    lines = ["Anxiete :"]
    for interp in available:
        if interp.suspect_value:
            continue
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_sleep(data: DRPatientData) -> str:
    available = _available_scores(data.sleep_scores)
    if not available:
        return ""
    lines = ["Sommeil :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_cognitive(data: DRPatientData) -> str:
    cp = data.cognitive_profile
    lines = []

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
                lines.append("    Flexibilite cognitive alteree (B-A > 90 s)")
            elif cp.tmt_b_minus_a > 60:
                lines.append("    Flexibilite cognitive limite (B-A 60-90 s)")
        # Add z-score interpretation for TMT
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
        lines.append(f"  - Stroop interference : {_fmt(cp.stroop_interference)}{stroop_extra}")

    if cp.cvlt_total_learning is not None:
        cvlt_parts = [f"apprentissage = {_fmt(cp.cvlt_total_learning)}"]
        if cp.cvlt_long_delay_free is not None:
            cvlt_parts.append(f"rappel differe = {_fmt(cp.cvlt_long_delay_free)}")
        lines.append(f"  - CVLT : {', '.join(cvlt_parts)}")

    if cp.phonemic_fluency is not None:
        lines.append(f"  - Fluence phonemique : {_fmt(cp.phonemic_fluency)}")

    wais_parts = []
    if cp.wais_similarities_std is not None:
        wais_parts.append(f"similitudes = {_fmt(cp.wais_similarities_std)}")
    if cp.wais_vocabulary_std is not None:
        wais_parts.append(f"vocabulaire = {_fmt(cp.wais_vocabulary_std)}")
    if cp.wais_working_memory_std is not None:
        wais_parts.append(f"memoire de travail = {_fmt(cp.wais_working_memory_std)}")
    if wais_parts:
        lines.append(f"  - WAIS (notes standard) : {', '.join(wais_parts)}")

    if not lines:
        return ""
    return "Profil cognitif :\n" + "\n".join(lines)


def _build_biology(data: DRPatientData) -> str:
    bio = data.biology
    if not bio.values and not bio.vitals:
        return ""
    lines = []

    vital_parts = []
    v = bio.vitals
    bmi = v.get("bmi")
    if bmi is not None:
        bmi_result = compute_bmi_category(bmi)
        bmi_label = f" ({bmi_result[1]})" if bmi_result else ""
        vital_parts.append(f"IMC = {_fmt(bmi, 1)} kg/m2{bmi_label}")
    if "weight_kg" in v:
        vital_parts.append(f"poids = {_fmt(v['weight_kg'], 1)} kg")
    if "waist_cm" in v:
        vital_parts.append(f"tour de taille = {_fmt(v['waist_cm'])} cm")
    sbp = v.get("sbp_supine")
    dbp = v.get("dbp_supine")
    if sbp is not None and dbp is not None:
        vital_parts.append(f"PA = {_fmt(sbp)}/{_fmt(dbp)} mmHg")
    if vital_parts:
        lines.append("  Constantes : " + ", ".join(vital_parts))

    if bio.ecg and "qtc" in bio.ecg:
        qtc = bio.ecg["qtc"]
        flag = " (allonge)" if qtc > 450 else ""
        lines.append(f"  ECG : QTc = {_fmt(qtc)} ms{flag}")

    abnormal = [lv for lv in bio.values if lv.is_abnormal]
    normal = [lv for lv in bio.values if not lv.is_abnormal]

    if abnormal:
        lines.append("  Anomalies biologiques :")
        for lv in abnormal:
            arrow = "^" if lv.abnormality == "high" else "v"
            lines.append(f"    {lv.name_fr} : {_fmt(lv.value, 2)} {lv.unit} ({arrow})")

    if normal:
        lines.append(f"  {len(normal)} autres parametres biologiques dans les normes.")
    elif not abnormal and bio.values:
        lines.append(f"  Bilan biologique sans anomalie ({len(bio.values)} parametres dans les normes).")

    # Metabolic syndrome detection
    sex = data.demographics.sex
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
        lines.append(f"  Syndrome metabolique (IDF/ATP-III) : OUI ({len(ms_criteria)}/5 criteres)")
        for crit in ms_criteria:
            lines.append(f"    - {crit}")
    elif ms_criteria:
        lines.append(f"  Syndrome metabolique : NON ({len(ms_criteria)}/5 criteres, seuil = 3)")

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

    # Medication-lab alerts
    treatment_flags = {
        "on_lithium": data.treatments.lithium_level is not None,
        "on_valproate": data.treatments.valproate_level is not None,
    }
    med_alerts = check_medication_lab_alerts(treatment_flags, bio.values, sex)
    drug_alerts = check_drug_interactions(treatment_flags)
    all_alerts = med_alerts + drug_alerts
    if all_alerts:
        lines.append("  Alertes medicamenteuses :")
        for alert in all_alerts:
            lines.append(f"    - {alert}")

    if not lines:
        return ""
    return "Bilan somatique :\n" + "\n".join(lines)


def _build_treatment(data: DRPatientData) -> str:
    t = data.treatments
    lines = []

    if t.lithium_level is not None:
        range_note = ""
        if t.lithium_level < 0.6:
            range_note = " (sous-therapeutique)"
        elif t.lithium_level <= 0.8:
            range_note = " (zone therapeutique basse)"
        elif t.lithium_level <= 1.0:
            range_note = " (zone therapeutique)"
        else:
            range_note = " (supra-therapeutique)"
        lines.append(f"  - Lithemie : {_fmt(t.lithium_level, 1)} mEq/L{range_note}")

    if t.valproate_level is not None:
        range_note = ""
        if t.valproate_level < 50:
            range_note = " (sous-therapeutique)"
        elif t.valproate_level <= 100:
            range_note = " (zone therapeutique)"
        else:
            range_note = " (supra-therapeutique)"
        lines.append(f"  - Valproate : {_fmt(t.valproate_level, 1)} ug/mL{range_note}")

    if t.has_ect:
        lines.append("  - Electroconvulsivotherapie (ECT) en cours")

    if t.medication_adherence and t.medication_adherence.score_available:
        lines.append(f"  - Observance : {t.medication_adherence.clinical_interpretation_fr}")

    if not lines:
        return ""
    return "Traitement actuel :\n" + "\n".join(lines)


def _build_comorbidity(data: DRPatientData) -> str:
    lines = []
    if data.psychiatric_comorbidities:
        lines.append("  Psychiatriques : " + ", ".join(data.psychiatric_comorbidities))
    all_somatic = list(data.somatic_comorbidities)
    if data.extended_somatic_history:
        all_somatic.extend(data.extended_somatic_history)
    if all_somatic:
        lines.append("  Somatiques : " + ", ".join(all_somatic))

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
        lines.append("  Trouble lie a l'usage de substances diagnostique")

    if not lines:
        return ""
    return "Comorbidites :\n" + "\n".join(lines)


def _build_substance(data: DRPatientData) -> str:
    available = _available_scores(data.substance_scores)
    if not available:
        return ""
    lines = ["Evaluation addictologique :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_trauma(data: DRPatientData) -> str:
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
                        level = " (leger a modere)"
                    elif sub_val <= 15:
                        level = " (modere a severe)"
                    else:
                        level = " (severe a extreme)"
                    lines.append(f"    {sub_name} : {_fmt(sub_val)}{level}")
    return "\n".join(lines)


def _build_self_esteem(data: DRPatientData) -> str:
    available = _available_scores(data.self_esteem_scores)
    if not available:
        return ""
    lines = ["Estime de soi :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_personality(data: DRPatientData) -> str:
    available = _available_scores(data.personality_scores)
    if not available:
        return ""
    lines = ["Profil de personnalite (BFI) :"]
    for interp in available:
        if interp.instrument == "BFI" and interp.subscales:
            for sub_name, sub_val in interp.subscales.items():
                if sub_val is not None:
                    lines.append(f"  - {sub_name} : {_fmt(sub_val, 1)}/5")
    if len(lines) <= 1:
        return ""
    return "\n".join(lines)


def _build_screening(data: DRPatientData) -> str:
    available = _available_scores(data.screening_scores)
    if not available:
        return ""
    lines = ["Depistages :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_family(data: DRPatientData) -> str:
    fh = data.family_history
    lines = []

    fam_parts = []
    if fh.maternal_psychiatric and not _is_unknown(fh.maternal_psychiatric):
        fam_parts.append(f"mere : {fh.maternal_psychiatric}")
    if fh.paternal_psychiatric and not _is_unknown(fh.paternal_psychiatric):
        fam_parts.append(f"pere : {fh.paternal_psychiatric}")
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
        lines.append("  - Antecedents familiaux de suicide")

    if not lines:
        return ""
    return "Antecedents familiaux elargis :\n" + "\n".join(lines)


def _build_impulsivity(data: DRPatientData) -> str:
    """Impulsivity — BIS-10 (newly extracted from DR.csv)."""
    available = _available_scores(data.impulsivity_scores)
    if not available:
        return ""
    lines = ["Impulsivité :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_risk(data: DRPatientData) -> str:
    lines = []

    sh = data.suicide_history
    if sh.ever_attempted:
        lines.append("  - Antecedent de tentative de suicide")

    cssrs = data.cssrs_assessment
    if cssrs.highest_ideation_level is not None and cssrs.highest_ideation_level >= 3:
        lines.append(f"  - Ideation suicidaire active (C-SSRS niveau {cssrs.highest_ideation_level})")

    mars = data.adherence_scores.get("MARS")
    if mars and mars.score_available and mars.severity_code == "poor":
        lines.append("  - Mauvaise observance therapeutique")

    tr = data.treatment_resistance
    if tr.is_resistant:
        lines.append("  - Depression resistante au traitement")
    if tr.has_psychotic_features:
        lines.append("  - Caracteristiques psychotiques")
    if tr.achieved_complete_remission is False:
        lines.append("  - Jamais de remission complete")

    # Metabolic syndrome flag
    bio = data.biology
    bmi = bio.vitals.get("bmi")
    waist = bio.vitals.get("waist_cm")
    metabolic_flags = 0
    if bmi is not None and bmi >= 30:
        metabolic_flags += 1
    if waist is not None and waist > 102:
        metabolic_flags += 1
    abnormal_lipids = any(
        lv.is_abnormal and lv.name in ("Triglycerides", "HDL", "Glucose", "Total cholesterol")
        for lv in bio.values
    )
    if abnormal_lipids:
        metabolic_flags += 1
    if metabolic_flags >= 2:
        lines.append("  - Syndrome metabolique probable (obesite + anomalies lipidiques/glycemiques)")
    elif bmi is not None and bmi >= 30:
        lines.append("  - Obesite (risque metabolique)")

    ctq = data.trauma_scores.get("CTQ")
    if ctq and ctq.score_available and ctq.severity_code in ("moderate_severe", "severe_extreme"):
        lines.append("  - Traumatismes de l'enfance significatifs")

    fh = data.family_history
    if fh.maternal_suicide or fh.paternal_suicide:
        lines.append("  - Antecedents familiaux de suicide")
    fam_parts = []
    if fh.maternal_psychiatric and not _is_unknown(fh.maternal_psychiatric):
        fam_parts.append(f"mere : {fh.maternal_psychiatric}")
    if fh.paternal_psychiatric and not _is_unknown(fh.paternal_psychiatric):
        fam_parts.append(f"pere : {fh.paternal_psychiatric}")
    if fam_parts:
        lines.append(f"  - ATCD psychiatriques familiaux : {', '.join(fam_parts)}")

    if not lines:
        return ""
    return "Facteurs de risque :\n" + "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN BUILDER
# ═════════════════════════════════════════════════════════════════════════════


def build_dr_profile(data: DRPatientData) -> PatientProfile:
    """Build a comprehensive DR patient profile.

    Section order follows standard psychiatric consultation structure,
    harmonised with BP/SZ vignette style. The PatientProfile field names
    are reused as generic slots -- the vignette order is controlled by
    the ordered_sections list, not by field names.
    """
    # Build all sections
    all_sections = {
        "synthesis": _build_synthesis(data),
        "demographics": _build_demographics(data),
        "history": _build_history(data),
        "episode_criteria": _build_episode_criteria(data),
        "suicide": _build_suicide(data),
        "clinical_notes": _build_clinical_notes(data),
        "depression": _build_depression(data),
        "functioning": _build_functioning(data),
        "anxiety": _build_anxiety(data),
        "impulsivity": _build_impulsivity(data),
        "sleep": _build_sleep(data),
        "self_esteem": _build_self_esteem(data),
        "personality": _build_personality(data),
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
        "synthesis", "demographics", "history", "episode_criteria",
        "suicide", "clinical_notes", "depression", "functioning",
        "anxiety", "impulsivity", "sleep", "self_esteem", "personality",
        "cognitive", "biology", "treatment",
        "comorbidity", "substance", "trauma", "screening",
        "family", "risk",
    ]
    vignette_parts = [all_sections[k] for k in ordered_keys if all_sections.get(k)]
    full_vignette = "\n\n".join(vignette_parts)

    # Map to PatientProfile fields (reused as generic slots)
    return PatientProfile(
        synthesis_section=all_sections.get("synthesis", ""),
        demographics_section=all_sections.get("demographics", ""),
        history_section=all_sections.get("history", ""),
        episode_criteria_section=all_sections.get("episode_criteria", ""),
        suicide_section=all_sections.get("suicide", ""),
        clinical_notes_section=all_sections.get("clinical_notes", ""),
        mood_section=all_sections.get("depression", ""),
        functional_section=all_sections.get("functioning", ""),
        anxiety_impulsivity_section=all_sections.get("anxiety", ""),
        sleep_section=all_sections.get("sleep", ""),
        cognitive_section=all_sections.get("cognitive", ""),
        biology_section=all_sections.get("biology", ""),
        treatment_section=all_sections.get("treatment", ""),
        comorbidity_section=all_sections.get("comorbidity", ""),
        substance_section=all_sections.get("substance", ""),
        trauma_section=all_sections.get("trauma", ""),
        screening_section=all_sections.get("screening", ""),
        extended_family_section=all_sections.get("family", ""),
        risk_section=all_sections.get("risk", ""),
        full_vignette=full_vignette,
        data=data,
    )
