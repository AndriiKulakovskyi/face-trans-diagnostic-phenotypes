"""Build comprehensive patient profiles and French clinical vignettes for ASP.

Follows the same structure as bp/sz_profile_builder but adapted for autism:
- Autism diagnosis profile replaces mood/psychotic episode criteria
- BDI-II replaces MADRS/Calgary as primary depression instrument
- RBS-R for repetitive behaviors (ASP-specific)
- Developmental history section (motor/language milestones)
- No suicide instruments (ISF/C-SSRS not available)
- No family psychiatric history (mere_trouble/pere_trouble absent)
- Minimal biology (bmi, gluc, hdl, chol, trig only)
"""

from __future__ import annotations

from face_rlvr.profiles.bp_profile_builder import (
    PatientProfile,
    _fmt,
    _score_line,
    _available_scores,
    _get_score,
    _get_severity,
    _get_label_fr as _bp_get_label_fr,
    _is_female,
    _agree,
)
from face_rlvr.profiles.common_instruments import ScoreInterpretation
from face_rlvr.profiles.asp_extractor import ASPPatientData
from face_rlvr.profiles.asp_instruments import ASP_INSTRUMENTS
from face_rlvr.profiles.common_extractors import (
    compute_bmi_category,
    detect_metabolic_syndrome,
    detect_floor_ceiling_effects,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _get_label_fr(scores: dict[str, ScoreInterpretation], key: str) -> str:
    """Get the French severity label from a scores dict."""
    interp = scores.get(key)
    if interp and interp.score_available:
        return interp.severity_label_fr.lower()
    return ""


def _dsm_type_to_label(dsm_type: float | None) -> str:
    """Convert DSM type code to French label for synthesis."""
    if dsm_type is None:
        return "trouble du spectre de l'autisme"
    t = int(dsm_type) if dsm_type == int(dsm_type) else dsm_type
    if t == 1:
        return "autisme infantile"
    elif t == 2:
        return "syndrome d'Asperger"
    elif t == 3:
        return "TED non spécifié"
    return "trouble du spectre de l'autisme"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION BUILDERS
# ═════════════════════════════════════════════════════════════════════════════


def _build_synthesis(data: ASPPatientData) -> str:
    """2-3 sentence clinical summary placed first in the vignette."""
    parts = []

    diag = data.autism_diagnosis
    diag_label = _dsm_type_to_label(diag.dsm_type)
    sent1 = f"Patient {_agree('suivi', 'suivie', data)} pour {diag_label}"

    # Depression mention
    bdi = _get_score(data.depression_scores, "BDI-II")
    bdi_sev = _get_severity(data.depression_scores, "BDI-II")
    if bdi is not None and bdi_sev not in ("minimal", "missing"):
        label = _get_label_fr(data.depression_scores, "BDI-II")
        sent1 += f", {label} (BDI-II = {_fmt(bdi)})"

    sent1 += "."
    parts.append(sent1)

    # IQ summary in first sentence if available
    wais_sev = _get_severity(data.cognitive_scores, "WAIS-IV")
    wais_qi = _get_score(data.cognitive_scores, "WAIS-IV")
    if wais_qi is not None:
        qi_label = _get_label_fr(data.cognitive_scores, "WAIS-IV")
        sent1 = sent1.rstrip(".") + f", QI total = {_fmt(wais_qi)} ({qi_label})."

    # Key findings
    concerns = []
    rbsr_sev = _get_severity(data.repetitive_behavior_scores, "RBS-R")
    if rbsr_sev == "high":
        concerns.append("comportements répétitifs fréquents")
    elif rbsr_sev == "moderate":
        concerns.append("comportements répétitifs modérés")

    # Executive function
    brief_sev = _get_severity(data.executive_function_scores, "BRIEF")
    if brief_sev in ("potentially_clinical", "clinically_significant"):
        concerns.append("difficultés exécutives")

    # ADHD
    adhd_sev = _get_severity(data.adhd_scores, "ADHD-RS")
    if adhd_sev in ("moderate", "severe"):
        concerns.append("symptômes de TDAH")

    ctq_sev = _get_severity(data.trauma_scores, "CTQ")
    if ctq_sev in ("moderate_severe", "severe_extreme"):
        concerns.append("antécédents traumatiques significatifs")

    eq5d_sev = _get_severity(data.functioning_scores, "EQ-5D")
    if eq5d_sev == "poor":
        concerns.append("qualité de vie altérée")

    bio = data.biology
    bmi = bio.vitals.get("bmi")
    if bmi is not None and bmi >= 30:
        concerns.append("obésité")

    if concerns:
        parts.append("Facteurs notables : " + ", ".join(concerns) + ".")

    # Treatment
    t = data.treatments
    tx_parts = []
    if t.on_antipsychotic:
        tx_parts.append("antipsychotique")
    if t.on_antidepressant:
        tx_parts.append("antidépresseur")
    if t.on_lamotrigine:
        tx_parts.append("lamotrigine")
    if tx_parts:
        parts.append(f"Traitement actuel : {', '.join(tx_parts)}.")

    return "Synthèse clinique : " + " ".join(parts)


def _build_demographics(data: ASPPatientData) -> str:
    """Demographics — adapted for ASP (no maristat)."""
    d = data.demographics
    sex = d.sex_label_fr or "sexe non précisé"
    diag_label = _dsm_type_to_label(data.autism_diagnosis.dsm_type)
    age = d.age or "?"
    line1 = f"Patient {data.patient_id}, {sex} de {age} ans, {_agree('suivi', 'suivie', data)} pour {diag_label}."

    details = []
    if d.education_level:
        details.append(f"niveau d'études : {d.education_level}")
    if d.employment:
        details.append(d.employment)
    if details:
        return line1 + "\n" + ", ".join(details).capitalize() + "."
    return line1


def _build_autism_diagnosis(data: ASPPatientData) -> str:
    """Autism diagnostic profile."""
    diag = data.autism_diagnosis
    lines = []

    if diag.dsm_type_label:
        lines.append(f"  - Type diagnostique : {diag.dsm_type_label}")

    if diag.adi_diagnostic is not None:
        adi_label = "positif" if diag.adi_diagnostic == 1.0 else "négatif"
        lines.append(f"  - ADI-R diagnostique : {adi_label}")

    if diag.ados_exam:
        lines.append(f"  - ADOS : {diag.ados_exam}")

    # DSM-5 domains
    domain_parts = []
    if diag.dsm_domain1_met is not None:
        status = "rempli" if diag.dsm_domain1_met else "non rempli"
        domain_parts.append(f"domaine 1 (communication sociale) : {status}")
    if diag.dsm_domain2_met is not None:
        status = "rempli" if diag.dsm_domain2_met else "non rempli"
        domain_parts.append(f"domaine 2 (comportements restreints) : {status}")
    if domain_parts:
        lines.append(f"  - Critères DSM-5 : {', '.join(domain_parts)}")

    # Individual criteria met
    criteria_met = [label for label, met in diag.dsm_criteria.items() if met]
    if criteria_met:
        lines.append(f"  - Critères individuels présents ({len(criteria_met)}) :")
        for c in criteria_met:
            lines.append(f"    - {c}")

    # ADI-R domain scores
    adir = data.autism_screening_scores.get("ADI-R")
    if adir and adir.score_available:
        line = _score_line(adir)
        if line:
            lines.append(line)
        if adir.subscales:
            adi_thresholds = {
                "Interaction sociale (A)": 10,
                "Communication (B)": 8,
                "Comportements restreints (C)": 3,
                "Développement anormal < 36 mois (D)": 1,
            }
            for sub_name, sub_val in adir.subscales.items():
                if sub_val is not None:
                    threshold = adi_thresholds.get(sub_name)
                    above = ""
                    if threshold is not None:
                        above = " (≥ seuil)" if sub_val >= threshold else " (< seuil)"
                    lines.append(f"    {sub_name} : {_fmt(sub_val)}{above}")

    # AQ-24 screening
    aq = data.autism_screening_scores.get("AQ-24")
    if aq and aq.score_available:
        line = _score_line(aq)
        if line:
            lines.append(line)

    # MCDD criteria
    mcdd = data.mcdd_profile
    if mcdd.total_criteria_assessed > 0:
        lines.append(f"  - MCDD : {mcdd.total_criteria_met}/{mcdd.total_criteria_assessed} critères remplis")

    if not lines:
        return ""
    return "Profil diagnostique autistique :\n" + "\n".join(lines)


def _build_developmental(data: ASPPatientData) -> str:
    """Developmental milestones, neonatal history, learning disabilities."""
    dev = data.developmental_history
    lines = []

    # Milestones
    if dev.age_motor_milestones is not None:
        lines.append(f"  - Acquisitions motrices : {_fmt(dev.age_motor_milestones)} mois")
    if dev.age_first_phrases is not None:
        lines.append(f"  - Premières phrases : {_fmt(dev.age_first_phrases)} mois")

    # Delays
    delays = []
    if dev.psychomotor_delay:
        delays.append("retard psychomoteur")
    if dev.language_delay:
        delays.append("retard de langage")
    if delays:
        lines.append(f"  - Retards : {', '.join(delays)}")

    # Neonatal/perinatal
    perinatal = []
    if dev.pregnancy_pathology:
        perinatal.append("pathologie de grossesse")
    if dev.fetal_distress:
        perinatal.append("souffrance fœtale")
    if dev.fetal_pathology:
        perinatal.append("pathologie fœtale")
    if dev.neonatal_complications:
        perinatal.append("complications néonatales")
    if dev.resuscitation:
        perinatal.append("réanimation")
    if dev.neonatal_illness:
        perinatal.append("affection néonatale")
    if perinatal:
        lines.append(f"  - Périnatalité : {', '.join(perinatal)}")

    # Birth data
    birth_parts = []
    if dev.birth_weight_g is not None:
        birth_parts.append(f"poids = {_fmt(dev.birth_weight_g)} g")
    if dev.birth_height_cm is not None:
        birth_parts.append(f"taille = {_fmt(dev.birth_height_cm)} cm")
    if dev.apgar_score is not None:
        birth_parts.append(f"Apgar = {_fmt(dev.apgar_score)}")
    if birth_parts:
        lines.append(f"  - Naissance : {', '.join(birth_parts)}")

    if dev.twin:
        lines.append("  - Gémellité")

    # Early difficulties
    early = []
    if dev.feeding_difficulties:
        early.append("alimentation")
    if dev.sleep_difficulties:
        early.append("sommeil")
    if dev.seizures:
        early.append("crises convulsives")
    if early:
        lines.append(f"  - Difficultés précoces : {', '.join(early)}")

    # Parental ages
    parent_parts = []
    if dev.mother_age is not None:
        parent_parts.append(f"mère = {_fmt(dev.mother_age)} ans")
    if dev.father_age is not None:
        parent_parts.append(f"père = {_fmt(dev.father_age)} ans")
    if parent_parts:
        lines.append(f"  - Âge des parents à la naissance : {', '.join(parent_parts)}")

    # Learning disabilities
    ld = data.learning_disabilities
    disabilities = []
    if ld.dyslexia:
        disabilities.append("dyslexie")
    if ld.dysorthography:
        disabilities.append("dysorthographie")
    if ld.dyscalculia:
        disabilities.append("dyscalculie")
    if ld.dysphasia:
        disabilities.append("dysphasie")
    if ld.dyspraxia:
        disabilities.append("dyspraxie")
    if ld.speech_disorder:
        disabilities.append("trouble de la parole")
    if ld.stuttering:
        disabilities.append("bégaiement")
    if disabilities:
        lines.append(f"  - Troubles des apprentissages : {', '.join(disabilities)}")

    if not lines:
        return ""
    return "Histoire développementale :\n" + "\n".join(lines)


def _build_clinical_status(data: ASPPatientData) -> str:
    """Clinical status — age at diagnosis, care, sleep, school, exec function."""
    cs = data.clinical_status
    lines = []

    if cs.age_at_diagnosis_years is not None:
        lines.append(f"  - Âge au diagnostic : {_fmt(cs.age_at_diagnosis_years)} ans")
    elif cs.age_at_diagnosis_months is not None:
        lines.append(f"  - Âge au diagnostic : {_fmt(cs.age_at_diagnosis_months)} mois")

    if cs.school_level:
        school = cs.school_level
        if cs.school_type:
            school += f" ({cs.school_type})"
        lines.append(f"  - Niveau scolaire maximal : {school}")

    care_parts = []
    if cs.in_psychiatric_care:
        care_parts.append("suivi psychiatrique en cours")
    if cs.currently_hospitalized:
        care_parts.append("hospitalisé actuellement")
    if cs.currently_treated:
        care_parts.append("traitement en cours")
    if care_parts:
        lines.append(f"  - Prise en charge : {', '.join(care_parts)}")

    sleep_parts = []
    if cs.has_insomnia:
        sleep_parts.append("insomnie")
    if cs.has_hypersomnia:
        sleep_parts.append("hypersomnie")
    if sleep_parts:
        lines.append(f"  - Troubles du sommeil : {', '.join(sleep_parts)}")

    neuro_parts = []
    if cs.executive_function_impairment:
        neuro_parts.append("atteinte des fonctions exécutives")
    if cs.social_cognition_impairment:
        neuro_parts.append("atteinte de la cognition sociale")
    if neuro_parts:
        lines.append(f"  - Profil neurocognitif : {', '.join(neuro_parts)}")

    if not lines:
        return ""
    return "Statut clinique actuel :\n" + "\n".join(lines)


def _build_medical_history(data: ASPPatientData) -> str:
    """Medical antecedents and pregnancy data."""
    ma = data.medical_antecedents
    lines = []

    antecedents = []
    if ma.cardiac:
        antecedents.append("cardiaque")
    if ma.endocrine:
        antecedents.append("endocrinien")
    if ma.neurological:
        antecedents.append("neurologique")
    if ma.ent:
        antecedents.append("ORL")
    if ma.pulmonary:
        antecedents.append("pulmonaire")
    if ma.rheumatological:
        antecedents.append("rhumatologique")
    if ma.hepatic:
        antecedents.append("hépatique")
    if ma.cancer:
        antecedents.append("carcinologique")
    if ma.genetic_disorder:
        antecedents.append("maladie génétique")
    if ma.other_condition:
        antecedents.append("autre pathologie")
    if antecedents:
        lines.append(f"  - Antécédents médicaux : {', '.join(antecedents)}")

    # Pregnancy / toxicology
    pt = data.pregnancy_toxicology
    preg_parts = []
    if pt.toxicology_exposure:
        preg_parts.append("exposition toxicologique")
    if pt.bleeding_during_pregnancy:
        preg_parts.append("saignements pendant la grossesse")
    if pt.infection_viral:
        preg_parts.append("infection virale")
    if pt.folic_acid_supplementation:
        preg_parts.append("supplémentation en acide folique")
    if preg_parts:
        lines.append(f"  - Grossesse : {', '.join(preg_parts)}")

    if not lines:
        return ""
    return "Antécédents médicaux :\n" + "\n".join(lines)


def _build_depression(data: ASPPatientData) -> str:
    """Depression section — BDI-II if available."""
    available = _available_scores(data.depression_scores)
    if not available:
        return ""
    lines = ["Évaluation dépressive :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_functioning(data: ASPPatientData) -> str:
    """Functioning section — EQ-5D and EGF."""
    available = [s for s in _available_scores(data.functioning_scores) if not s.suspect_value]
    if not available:
        return ""
    lines = ["Fonctionnement et qualité de vie :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_repetitive_behaviors(data: ASPPatientData) -> str:
    """Repetitive behaviors — RBS-R total + subscales."""
    available = _available_scores(data.repetitive_behavior_scores)
    if not available:
        return ""
    lines = ["Comportements répétitifs (RBS-R) :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
        if interp.instrument == "RBS-R" and interp.subscales:
            for sub_name, sub_val in interp.subscales.items():
                if sub_val is not None:
                    lines.append(f"    {sub_name} : {_fmt(sub_val)}")
    return "\n".join(lines)


def _build_cognitive(data: ASPPatientData) -> str:
    """Cognitive / IQ section — WAIS-IV indices."""
    available = _available_scores(data.cognitive_scores)
    if not available:
        return ""
    lines = ["Profil cognitif (WAIS-IV) :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
        # Display index subscales
        if interp.instrument == "WAIS-IV" and interp.subscales:
            for sub_name, sub_val in interp.subscales.items():
                if sub_val is not None:
                    level = ""
                    if sub_val < 70:
                        level = " (extrêmement bas)"
                    elif sub_val < 80:
                        level = " (zone limite)"
                    elif sub_val < 90:
                        level = " (moyenne inférieure)"
                    elif sub_val <= 109:
                        level = " (moyen)"
                    elif sub_val <= 119:
                        level = " (moyenne supérieure)"
                    else:
                        level = " (supérieur)"
                    lines.append(f"    {sub_name} = {_fmt(sub_val)}{level}")
    return "\n".join(lines)


def _build_executive_function(data: ASPPatientData) -> str:
    """Executive function — BRIEF subscales."""
    available = _available_scores(data.executive_function_scores)
    if not available:
        return ""
    lines = ["Fonctions exécutives (BRIEF) :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
        # Display subscales with clinical thresholds
        if interp.instrument == "BRIEF" and interp.subscales:
            elevated = []
            normal = []
            for sub_name, sub_val in interp.subscales.items():
                if sub_val is not None:
                    if sub_val >= 70:
                        elevated.append(f"{sub_name} = {_fmt(sub_val)} (clinique)")
                    elif sub_val >= 65:
                        elevated.append(f"{sub_name} = {_fmt(sub_val)} (limite)")
                    else:
                        normal.append(sub_name)
            if elevated:
                lines.append("    Sous-échelles élevées :")
                for e in elevated:
                    lines.append(f"      {e}")
            if normal:
                lines.append(f"    Dans les normes : {', '.join(normal)}")
    return "\n".join(lines)


def _build_anxiety(data: ASPPatientData) -> str:
    """Anxiety section — HAM-A, LSAS."""
    available = _available_scores(data.anxiety_scores)
    if not available:
        return ""
    lines = ["Anxiété :"]
    for interp in available:
        if interp.suspect_value:
            continue
        line = _score_line(interp)
        if line:
            lines.append(line)
        # LSAS subscales
        if interp.instrument == "LSAS" and interp.subscales:
            for sub_name, sub_val in interp.subscales.items():
                if sub_val is not None:
                    lines.append(f"    {sub_name} : {_fmt(sub_val)}")
    return "\n".join(lines)


def _build_adhd(data: ASPPatientData) -> str:
    """ADHD assessment — ADHD-RS."""
    available = _available_scores(data.adhd_scores)
    if not available:
        return ""
    lines = ["Évaluation du TDAH :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
        if interp.instrument == "ADHD-RS" and interp.subscales:
            for sub_name, sub_val in interp.subscales.items():
                if sub_val is not None:
                    lines.append(f"    {sub_name} : {_fmt(sub_val)}")
    return "\n".join(lines)


def _build_clinical_notes(data: ASPPatientData) -> str:
    """Detect and flag clinical discordances between instruments."""
    notes = []

    # 1. WAIS-IV index scatter (>15 pts between highest and lowest index)
    wais = data.cognitive_scores.get("WAIS-IV")
    if wais and wais.score_available and wais.subscales:
        indices = [v for v in wais.subscales.values() if v is not None]
        if len(indices) >= 2:
            scatter = max(indices) - min(indices)
            if scatter > 15:
                notes.append(
                    f"Hétérogénéité significative du profil cognitif WAIS-IV "
                    f"(écart inter-indices = {_fmt(scatter)} points > 15) : "
                    f"le QI total doit être interprété avec prudence."
                )

    # 2. EGF vs EQ-5D discordance
    egf = _get_score(data.functioning_scores, "EGF")
    egf_sev = _get_severity(data.functioning_scores, "EGF")
    eq5d = _get_score(data.functioning_scores, "EQ-5D")
    eq5d_sev = _get_severity(data.functioning_scores, "EQ-5D")
    if egf is not None and eq5d is not None:
        egf_good = egf_sev in ("superior", "absent_minimal", "slight")
        eq5d_poor = eq5d_sev == "poor"
        egf_poor = egf_sev in ("serious", "major_impairment", "serious_impairment")
        eq5d_good = eq5d_sev == "good"
        if (egf_good and eq5d_poor) or (egf_poor and eq5d_good):
            notes.append(
                f"Discordance EGF ({_fmt(egf)}) / EQ-5D ({_fmt(eq5d, 2)}) : "
                f"évaluations divergentes du fonctionnement."
            )

    # 3. ADHD-RS vs BRIEF inhibition
    adhd = _get_score(data.adhd_scores, "ADHD-RS")
    adhd_sev = _get_severity(data.adhd_scores, "ADHD-RS")
    brief = data.executive_function_scores.get("BRIEF")
    if adhd is not None and brief and brief.subscales:
        inhib = brief.subscales.get("Inhibition")
        if inhib is not None:
            adhd_elevated = adhd_sev in ("moderate", "severe")
            inhib_normal = inhib < 65
            if adhd_elevated and inhib_normal:
                notes.append(
                    f"Discordance ADHD-RS ({_fmt(adhd)}, {_get_label_fr(data.adhd_scores, 'ADHD-RS')}) / "
                    f"BRIEF Inhibition ({_fmt(inhib)}, normal) : "
                    f"les symptômes de TDAH ne se reflètent pas dans le score d'inhibition exécutive."
                )

    # 4. Floor/ceiling effects across all instrument scores
    all_scores: dict[str, ScoreInterpretation] = {}
    for score_dict in (
        data.depression_scores, data.functioning_scores,
        data.repetitive_behavior_scores, data.cognitive_scores,
        data.executive_function_scores, data.anxiety_scores,
        data.adhd_scores, data.trauma_scores, data.sleep_scores,
        data.adherence_scores, data.substance_scores,
        data.global_scores, data.autism_screening_scores,
    ):
        all_scores.update(score_dict)
    fc_effects = detect_floor_ceiling_effects(all_scores, ASP_INSTRUMENTS)
    for effect in fc_effects:
        notes.append(f"Contrôle qualité — {effect}")

    # 5. WAIS-IV index discrepancy interpretation (>= 12 points)
    wais = data.cognitive_scores.get("WAIS-IV")
    if wais and wais.score_available and wais.subscales:
        sub = wais.subscales
        icv = sub.get("Compréhension verbale (ICV)")
        irp = sub.get("Raisonnement perceptif (IRP)")
        imt = sub.get("Mémoire de travail (IMT)")
        ivt = sub.get("Vitesse de traitement (IVT)")
        indices = {"ICV": icv, "IRP": irp, "IMT": imt, "IVT": ivt}
        available_indices = {k: v for k, v in indices.items() if v is not None}

        if len(available_indices) >= 2:
            # Check all pairs for >= 12 point differences
            discrepancies: list[str] = []
            keys = list(available_indices.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    diff = abs(available_indices[keys[i]] - available_indices[keys[j]])
                    if diff >= 12:
                        discrepancies.append(
                            f"{keys[i]}–{keys[j]} = {_fmt(diff)} points"
                        )

            if discrepancies:
                notes.append(
                    f"Dissociation significative WAIS-IV (≥ 12 pts) : "
                    + ", ".join(discrepancies) + "."
                )
                # Interpret specific patterns
                if icv is not None and ivt is not None and icv - ivt >= 12:
                    notes.append(
                        "Profil typique du TSA avec déficit de vitesse de traitement "
                        f"(ICV = {_fmt(icv)} vs IVT = {_fmt(ivt)})."
                    )
                if irp is not None and icv is not None and irp - icv >= 12:
                    notes.append(
                        "Dissociation verbo-perceptive, atypique "
                        f"(IRP = {_fmt(irp)} vs ICV = {_fmt(icv)})."
                    )
                if imt is not None:
                    other_vals = [v for k, v in available_indices.items() if k != "IMT" and v is not None]
                    if other_vals and all(imt <= v - 12 for v in other_vals):
                        notes.append(
                            f"Déficit de la mémoire de travail (IMT = {_fmt(imt)}), "
                            "retentissement fonctionnel probable."
                        )

    if not notes:
        return ""
    return "Notes cliniques :\n" + "\n".join(f"  * {n}" for n in notes)


def _build_biology(data: ASPPatientData) -> str:
    """Minimal biology — bmi, gluc, hdl, chol, trig."""
    bio = data.biology
    if not bio.values and not bio.vitals:
        return ""
    lines = []

    # BMI with WHO category
    bmi = bio.vitals.get("bmi")
    if bmi is not None:
        cat = compute_bmi_category(bmi)
        bmi_label = f" ({cat[1]})" if cat else ""
        lines.append(f"  IMC = {_fmt(bmi, 1)} kg/m²{bmi_label}")

    # Metabolic syndrome detection
    sex_code = "M" if not _is_female(data) else "F"
    waist = bio.vitals.get("waist")
    trig_val = next((lv.value for lv in bio.values if lv.name in ("Triglycerides", "Triglycérides")), None)
    hdl_val = next((lv.value for lv in bio.values if lv.name == "HDL"), None)
    gluc_val = next((lv.value for lv in bio.values if lv.name in ("Glucose", "Glycémie")), None)
    sbp = bio.vitals.get("sbp")
    dbp = bio.vitals.get("dbp")
    has_ms, ms_criteria = detect_metabolic_syndrome(waist, sex_code, trig_val, hdl_val, sbp, dbp, gluc_val)
    if has_ms:
        lines.append(f"  Syndrome métabolique ({len(ms_criteria)} critères) :")
        for c in ms_criteria:
            lines.append(f"    - {c}")
    elif ms_criteria:
        lines.append(f"  Critères métaboliques présents ({len(ms_criteria)}/5, < seuil diagnostic) :")
        for c in ms_criteria:
            lines.append(f"    - {c}")

    # Lab values
    abnormal = [lv for lv in bio.values if lv.is_abnormal]
    normal = [lv for lv in bio.values if not lv.is_abnormal]

    if abnormal:
        lines.append("  Anomalies biologiques :")
        for lv in abnormal:
            arrow = "↑" if lv.abnormality == "high" else "↓"
            lines.append(f"    {lv.name_fr} : {_fmt(lv.value, 2)} {lv.unit} ({arrow})")

    if normal:
        lines.append(f"  {len(normal)} paramètre(s) biologique(s) dans les normes.")
    elif not abnormal and bio.values:
        lines.append(f"  Bilan biologique sans anomalie ({len(bio.values)} paramètres dans les normes).")

    if not lines:
        return ""
    return "Bilan somatique :\n" + "\n".join(lines)


def _build_treatment(data: ASPPatientData) -> str:
    """Treatment section — antidepressant, antipsychotic, lamotrigine, non-pharm."""
    t = data.treatments
    lines = []

    if t.on_antidepressant:
        lines.append("  - Antidépresseur")
    if t.on_antipsychotic:
        lines.append("  - Antipsychotique")
    if t.on_lamotrigine:
        lines.append("  - Lamotrigine")

    if t.non_pharm_treatments:
        lines.append(f"  - Traitements non pharmacologiques : {', '.join(t.non_pharm_treatments)}")

    if not lines:
        return ""
    return "Traitement actuel :\n" + "\n".join(lines)


def _build_comorbidity(data: ASPPatientData) -> str:
    """Somatic and psychiatric comorbidities."""
    lines = []
    if data.psychiatric_comorbidities:
        lines.append("  Psychiatriques : " + ", ".join(data.psychiatric_comorbidities))
    if data.somatic_comorbidities:
        lines.append("  Somatiques : " + ", ".join(data.somatic_comorbidities))
    if not lines:
        return ""
    return "Comorbidités :\n" + "\n".join(lines)


def _build_trauma(data: ASPPatientData) -> str:
    """Childhood trauma — CTQ if available."""
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


def _build_global(data: ASPPatientData) -> str:
    """Global clinical impression — CGI-C if available."""
    available = _available_scores(data.global_scores)
    if not available:
        return ""
    lines = ["Évaluation globale :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_sleep(data: ASPPatientData) -> str:
    """Sleep assessment — PSQI + ESS (newly extracted from ASP.csv)."""
    available = _available_scores(data.sleep_scores)
    if not available:
        return ""
    lines = ["Évaluation du sommeil :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_adherence(data: ASPPatientData) -> str:
    """Medication adherence — MARS (newly extracted from ASP.csv)."""
    available = _available_scores(data.adherence_scores)
    if not available:
        return ""
    lines = ["Observance médicamenteuse :"]
    for interp in available:
        line = _score_line(interp)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_substance(data: ASPPatientData) -> str:
    """Substance use — Fagerström + PRISM + tobacco data (newly extracted)."""
    parts: list[str] = []

    # Substance use flags
    su = data.substance_use
    if su.tobacco_current:
        cpd_str = f", {su.tobacco_cpd:.0f} cigarettes/jour" if su.tobacco_cpd else ""
        parts.append(f"  - Tabagisme actif{cpd_str}")
    if su.alcohol_current:
        parts.append("  - Consommation d'alcool")
    if su.cannabis_current:
        parts.append("  - Consommation de cannabis")
    for s in su.other_substances:
        parts.append(f"  - {s}")

    # Instrument scores
    available = _available_scores(data.substance_scores)
    for interp in available:
        line = _score_line(interp)
        if line:
            parts.append(line)

    if not parts:
        return ""
    return "Usage de substances :\n" + "\n".join(parts)


def _build_limitations(data: ASPPatientData) -> str:
    """Document known data limitations for ASP cohort."""
    limitations = [
        "Aucun instrument de risque suicidaire (C-SSRS, ISF) disponible dans la cohorte ASP",
        "Antécédents psychiatriques familiaux non documentés dans la base de données",
        "Profil sensoriel détaillé et fonctionnement adaptatif (Vineland) non disponibles",
    ]
    # BDI item 9 partial mitigation
    if data.bdi_item9_suicidal_thoughts is not None:
        if data.bdi_item9_suicidal_thoughts > 0:
            limitations[0] += (
                f" — item 9 BDI-II = {data.bdi_item9_suicidal_thoughts}/3 "
                f"(pensées suicidaires rapportées)"
            )
        else:
            limitations[0] += " — item 9 BDI-II = 0/3 (absence de pensées suicidaires)"

    return "Limitations de l'évaluation :\n" + "\n".join(f"  - {l}" for l in limitations)


def _build_risk(data: ASPPatientData) -> str:
    """Risk factors — limited (no suicide instruments in ASP)."""
    lines = []

    # Metabolic risk via detect_metabolic_syndrome
    bio = data.biology
    bmi = bio.vitals.get("bmi")
    sex_code = "M" if not _is_female(data) else "F"
    waist = bio.vitals.get("waist")
    trig_val = next((lv.value for lv in bio.values if lv.name in ("Triglycerides", "Triglycérides")), None)
    hdl_val = next((lv.value for lv in bio.values if lv.name == "HDL"), None)
    gluc_val = next((lv.value for lv in bio.values if lv.name in ("Glucose", "Glycémie")), None)
    sbp = bio.vitals.get("sbp")
    dbp = bio.vitals.get("dbp")
    has_ms, ms_criteria = detect_metabolic_syndrome(waist, sex_code, trig_val, hdl_val, sbp, dbp, gluc_val)
    if has_ms:
        lines.append(f"  - Syndrome métabolique ({len(ms_criteria)} critères)")
    elif bmi is not None and bmi >= 30:
        bmi_cat = compute_bmi_category(bmi)
        label = bmi_cat[1] if bmi_cat else "Obésité"
        lines.append(f"  - {label} (risque métabolique)")

    # Trauma
    ctq = data.trauma_scores.get("CTQ")
    if ctq and ctq.score_available and ctq.severity_code in ("moderate_severe", "severe_extreme"):
        lines.append("  - Traumatismes de l'enfance significatifs")

    # Depression
    bdi_sev = _get_severity(data.depression_scores, "BDI-II")
    if bdi_sev == "severe":
        lines.append("  - Dépression sévère (BDI-II)")

    if not lines:
        return ""
    return "Facteurs de risque :\n" + "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN BUILDER
# ═════════════════════════════════════════════════════════════════════════════


def build_asp_profile(data: ASPPatientData) -> PatientProfile:
    """Build a comprehensive ASP patient profile.

    Section order follows standard psychiatric consultation structure,
    adapted for autism spectrum disorder. The PatientProfile field names
    are reused as generic slots — the vignette order is controlled by
    the ordered_sections list, not by field names.
    """
    # Build all sections
    all_sections = {
        "synthesis": _build_synthesis(data),
        "demographics": _build_demographics(data),
        "autism_diagnosis": _build_autism_diagnosis(data),
        "developmental": _build_developmental(data),
        "clinical_status": _build_clinical_status(data),
        "medical_history": _build_medical_history(data),
        "clinical_notes": _build_clinical_notes(data),
        "depression": _build_depression(data),
        "global": _build_global(data),
        "functioning": _build_functioning(data),
        "cognitive": _build_cognitive(data),
        "executive_function": _build_executive_function(data),
        "repetitive_behaviors": _build_repetitive_behaviors(data),
        "adhd": _build_adhd(data),
        "anxiety": _build_anxiety(data),
        "sleep": _build_sleep(data),
        "biology": _build_biology(data),
        "treatment": _build_treatment(data),
        "adherence": _build_adherence(data),
        "comorbidity": _build_comorbidity(data),
        "substance": _build_substance(data),
        "trauma": _build_trauma(data),
        "risk": _build_risk(data),
        "limitations": _build_limitations(data),
    }

    # Build vignette in correct order
    ordered_keys = [
        "synthesis", "demographics", "autism_diagnosis", "developmental",
        "clinical_status", "medical_history",
        "clinical_notes", "depression", "global", "functioning",
        "cognitive", "executive_function", "repetitive_behaviors",
        "adhd", "anxiety", "sleep", "biology", "treatment", "adherence",
        "comorbidity", "substance", "trauma", "risk", "limitations",
    ]
    vignette_parts = [all_sections[k] for k in ordered_keys if all_sections.get(k)]
    full_vignette = "\n\n".join(vignette_parts)

    # Map to PatientProfile fields (reused as generic slots)
    return PatientProfile(
        synthesis_section=all_sections.get("synthesis", ""),
        demographics_section=all_sections.get("demographics", ""),
        episode_criteria_section=all_sections.get("autism_diagnosis", ""),
        history_section=all_sections.get("developmental", ""),
        clinical_notes_section=all_sections.get("clinical_notes", ""),
        mood_section=all_sections.get("depression", ""),
        functional_section=all_sections.get("functioning", ""),
        cognitive_section=all_sections.get("cognitive", ""),
        sleep_section=all_sections.get("executive_function", ""),  # reuse slot
        anxiety_impulsivity_section=all_sections.get("repetitive_behaviors", ""),
        screening_section=all_sections.get("adhd", ""),  # reuse slot
        substance_section=all_sections.get("anxiety", ""),  # reuse slot
        biology_section=all_sections.get("biology", ""),
        treatment_section=all_sections.get("treatment", ""),
        comorbidity_section=all_sections.get("comorbidity", ""),
        trauma_section=all_sections.get("trauma", ""),
        risk_section=all_sections.get("risk", ""),
        full_vignette=full_vignette,
        data=data,
    )
