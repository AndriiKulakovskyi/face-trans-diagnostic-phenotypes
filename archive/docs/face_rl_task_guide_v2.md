# FACE Database — RL Task Creation Guide

**Purpose**: Operational guide for generating 1,000 verifiable training tasks from the FACE cohort for GRPO-based reinforcement learning of a precision psychiatry reasoning LLM.  
**Constraint**: Every task satisfies the oracle-first principle — ground truth is computed deterministically from patient data before the LLM generates reasoning.

---

## Task Budget Allocation

| # | Category | N Tasks | Cohorts | Rationale |
|---|----------|---------|---------|-----------|
| 1 | Metabolic & Somatic Risk Assessment | 120 | All | High clinical impact, fully deterministic oracle, code-heavy |
| 2 | Treatment Analysis | 160 | All | Core clinical skill, guideline-deterministic, multi-step reasoning |
| 3 | Diagnostic Reasoning | 130 | All | Highest reasoning complexity, strong training signal for integration steps |
| 4 | Suicide Risk Assessment | 80 | BD, SZ, DR | Safety-critical, asymmetric reward, essential for deployment |
| 5 | Cognitive Assessment | 100 | All | Code-required, well-normed instruments, clean verifiability |
| 6 | Longitudinal Trajectory Analysis | 120 | All | Exploits FACE's longitudinal strength, code-heavy |
| 7 | Rating Scale Interpretation | 80 | All | Format bootstrapping, curriculum anchor for early RL stages |
| 8 | Side Effect & Safety Monitoring | 80 | All | Underserved clinical need, high practical value |
| 9 | Transdiagnostic & Cross-Cohort Reasoning | 60 | Cross | Unique to FACE, highest difficulty, boundary reasoning |
| 10 | Data Quality & Clinical Reasoning Meta-Tasks | 70 | All | Novel task type, teaches meta-reasoning about evidence limits |
| **Total** | | **1,000** | | |

**Difficulty distribution per category**: 30% easy / 40% medium / 30% hard (unless otherwise specified).

**Cohort distribution**: Proportional to FACE composition with DR and ASD oversampled — approximately 35% BD, 25% SZ, 20% DR, 20% ASD across the full 1,000.

---

## Category 1 — Metabolic & Somatic Risk Assessment (120 tasks)

### Description

Psychiatric patients face drastically shortened life expectancy — primarily from cardiovascular disease driven by medication side effects and undertreated somatic comorbidities. These tasks train the model to interpret laboratory values, anthropometric data, and vital signs in the specific context of psychiatric pharmacotherapy. The clinical value is high: psychiatrists receive lab reports they were never trained to interpret in depth, and the reasoning model bridges this gap.

The oracle for every task is deterministic: compare values against reference ranges, apply diagnostic algorithms (IDF/ATP-III for metabolic syndrome, Framingham/SCORE2 for cardiovascular risk, CKD-EPI for renal function), and check medication-lab associations from a lookup table.

### Sub-types

| Sub-type | N | Code Modality | Description |
|----------|---|---------------|-------------|
| 1a. Metabolic syndrome diagnosis | 30 | code_required | Apply IDF or NCEP-ATP III criteria to anthropometric + lab data |
| 1b. Lab panel interpretation in psychiatric context | 35 | code_required | Flag abnormals, classify severity, attribute to medications |
| 1c. Cardiovascular risk estimation | 20 | code_required | Compute Framingham or SCORE2 10-year risk from clinical variables |
| 1d. Iatrogenic metabolic change detection | 20 | code_required | Detect clinically significant metabolic deterioration correlated with medication |
| 1e. Medication-specific monitoring protocol | 15 | code_preferred | Determine required lab monitoring and flag overdue tests |

### Example Task — Sub-type 1b (Medium difficulty)

**Vignette**:

> Patient: Jean-Marc D., 48 ans, suivi au Centre Expert pour schizophrénie depuis 2018.
>
> Traitement actuel: Clozapine 400 mg/j, Valproate de sodium 1000 mg/j
>
> Bilan biologique du 15/03/2025:
> - Glycémie à jeun: 6.4 mmol/L
> - HbA1c: 6.2%
> - Cholestérol total: 6.8 mmol/L
> - LDL-cholestérol: 4.2 mmol/L
> - HDL-cholestérol: 0.88 mmol/L
> - Triglycérides: 2.9 mmol/L
> - ASAT: 52 UI/L
> - ALAT: 68 UI/L
> - GGT: 95 UI/L
> - Créatinine: 92 µmol/L
> - DFG estimé (CKD-EPI): 82 mL/min
> - NFS: Leucocytes 4.8 G/L, PNN 2.1 G/L, Hb 14.2 g/dL, Plaquettes 185 G/L
> - Prolactine: 42 ng/mL
> - TSH: 3.2 mUI/L
> - IMC: 33.4 kg/m², Tour de taille: 112 cm
> - PA: 145/92 mmHg
>
> Question: Identifiez toutes les valeurs anormales, classifiez leur sévérité, évaluez le syndrome métabolique selon les critères IDF, et déterminez les anomalies potentiellement attribuables au traitement actuel.

**Oracle ground truth**:
```json
{
  "abnormal_values": {
    "fasting_glucose": {"value": 6.4, "status": "high", "severity": "moderate", "ref": "3.9-5.6 mmol/L"},
    "hba1c": {"value": 6.2, "status": "high", "severity": "mild", "ref": "<5.7%", "note": "pre-diabetes"},
    "total_cholesterol": {"value": 6.8, "status": "high", "severity": "moderate", "ref": "<5.2 mmol/L"},
    "ldl": {"value": 4.2, "status": "high", "severity": "moderate", "ref": "<3.4 mmol/L"},
    "hdl": {"value": 0.88, "status": "low", "severity": "moderate", "ref": ">1.0 mmol/L (male)"},
    "triglycerides": {"value": 2.9, "status": "high", "severity": "moderate", "ref": "<1.7 mmol/L"},
    "asat": {"value": 52, "status": "high", "severity": "mild", "ref": "10-40 UI/L"},
    "alat": {"value": 68, "status": "high", "severity": "mild", "ref": "10-40 UI/L"},
    "ggt": {"value": 95, "status": "high", "severity": "moderate", "ref": "10-50 UI/L"},
    "prolactin": {"value": 42, "status": "high", "severity": "mild", "ref": "4-15 ng/mL (male)"},
    "systolic_bp": {"value": 145, "status": "high", "severity": "moderate", "ref": "<130 mmHg"},
    "diastolic_bp": {"value": 92, "status": "high", "severity": "moderate", "ref": "<85 mmHg"}
  },
  "metabolic_syndrome_idf": {
    "central_obesity": true,
    "criteria_met": ["central_obesity", "elevated_triglycerides", "reduced_hdl", "elevated_bp", "elevated_fasting_glucose"],
    "n_criteria": 5,
    "diagnosis": true,
    "note": "All 5 IDF criteria met — severe metabolic syndrome"
  },
  "medication_attributions": {
    "clozapine": ["metabolic_syndrome", "weight_gain", "elevated_triglycerides", "hyperglycemia", "hyperprolactinemia"],
    "valproate": ["hepatic_enzyme_elevation", "weight_gain"]
  },
  "critical_alerts": [
    "Pre-diabetes (HbA1c 6.2%) requiring OGTT and diabetology referral",
    "Hepatic enzyme elevation on valproate — monitor trend, consider hepatotoxicity",
    "Neutrophil count 2.1 G/L — within safe range for clozapine but monitor (amber if <1.5)"
  ]
}
```

**Expected reasoning artifact**: The model should produce `extract` steps for each lab value, `compute` steps to compare against reference ranges and count IDF criteria, `lookup` steps to reference medication side effect profiles, a `safety_check` step for the clozapine neutrophil monitoring, and an `integrate` step attributing the metabolic pattern to the specific medications.

### Difficulty Calibration

| Level | Characteristics | Example |
|-------|----------------|---------|
| Easy | 5–8 lab values, obvious abnormals, single medication, straightforward MetS yes/no | Patient on olanzapine with clear MetS |
| Medium | 15–20 lab values, borderline values, 2+ medications, requires attribution reasoning | Example above |
| Hard | Full panel + longitudinal comparison, 3+ medications, subtle patterns (e.g., early lithium nephropathy), missing values requiring "cannot determine" reasoning | Two visits compared, medication changed between visits |

---

## Category 2 — Treatment Analysis (160 tasks)

### Description

Treatment decision-making is the central daily act of psychiatry. These tasks train the model on the full arc of pharmacological reasoning: Has a medication trial been adequate? Is the patient treatment-resistant? What does the guideline recommend next? Are there dangerous interactions? This category is the largest because it spans the most sub-types and has the most direct clinical impact.

The oracle draws from three sources: (1) published guideline algorithms (CANMAT, NICE, HAS, PORT), (2) pharmacological reference tables (therapeutic dose ranges, interaction databases), and (3) FACE medication history records.

### Sub-types

| Sub-type | N | Code Modality | Description |
|----------|---|---------------|-------------|
| 2a. Treatment adequacy assessment | 35 | code_preferred | Evaluate whether each trial in patient's history was adequate (dose, duration, adherence) |
| 2b. Treatment resistance staging | 30 | code_required | Apply Thase-Rush / Maudsley / Sachs / Kane staging algorithms |
| 2c. Guideline-concordant next step | 35 | code_preferred | Given current state + history, recommend next treatment per published guideline |
| 2d. Drug interaction detection | 30 | code_required | Identify clinically significant interactions in current polypharmacy |
| 2e. Dose optimization reasoning | 15 | code_required | Given serum levels and clinical response, recommend dose adjustment |
| 2f. ECT/rTMS candidacy | 15 | text_with_structured_differential | Assess eligibility for non-pharmacological interventions |

### Example Task — Sub-type 2b (Hard difficulty)

**Vignette**:

> Patiente: Sophie M., 39 ans, trouble bipolaire de type I, suivie depuis 2012.
>
> Historique des traitements thymorégulateurs:
> 1. Lithium 800 mg/j (2012–2014): lithiémie 0.6–0.8 mmol/L, durée 24 mois. Épisode dépressif sous traitement à M+8. Épisode maniaque à M+18. Arrêt pour insuffisance rénale débutante (DFG 62 mL/min).
> 2. Valproate 1500 mg/j (2014–2016): valproatémie 75 mg/L, durée 18 mois. Stabilité pendant 10 mois puis épisode mixte. Arrêt pour projet de grossesse.
> 3. Lamotrigine 200 mg/j (2016–2018): durée 24 mois. Bon contrôle des dépressions. Épisode maniaque sévère avec hospitalisation à M+20.
> 4. Quétiapine LP 600 mg/j (2018–2020): durée 22 mois. Épisode dépressif modéré à M+6. Prise de poids +14 kg. Épisode maniaque à M+18.
> 5. Aripiprazole 20 mg/j + Lamotrigine 200 mg/j (2020–present): en cours depuis 4 ans. Deux épisodes dépressifs légers. Pas d'épisode maniaque.
>
> Score ALDA (réponse au lithium): A=4, B=7 → score total = -3
>
> État actuel: euthymique, CGI-S = 2, MADRS = 6, YMRS = 2
> Dernière hospitalisation: 2018
>
> Question: Évaluez l'adéquation de chaque essai thérapeutique, classez le niveau de résistance au traitement selon les critères de Sachs, et déterminez si la patiente est actuellement stabilisée.

**Oracle ground truth**:
```json
{
  "trial_adequacy": [
    {
      "medication": "lithium",
      "dose_adequate": true,
      "serum_level_adequate": true,
      "duration_adequate": true,
      "outcome": "partial_response",
      "reason_stopped": "renal_toxicity",
      "adequate_trial": true
    },
    {
      "medication": "valproate",
      "dose_adequate": true,
      "serum_level_adequate": true,
      "duration_adequate": true,
      "outcome": "partial_response",
      "reason_stopped": "pregnancy_planning",
      "adequate_trial": true
    },
    {
      "medication": "lamotrigine",
      "dose_adequate": true,
      "serum_level_adequate": null,
      "duration_adequate": true,
      "outcome": "partial_response",
      "note": "effective for depression polarity but breakthrough mania",
      "adequate_trial": true
    },
    {
      "medication": "quetiapine_xr",
      "dose_adequate": true,
      "serum_level_adequate": null,
      "duration_adequate": true,
      "outcome": "non_response",
      "reason_stopped": "inefficacy_and_side_effects",
      "adequate_trial": true
    },
    {
      "medication": "aripiprazole_plus_lamotrigine",
      "dose_adequate": true,
      "duration_adequate": true,
      "outcome": "good_response",
      "note": "current regimen, no manic episodes in 4 years, residual mild depression only",
      "adequate_trial": true
    }
  ],
  "resistance_staging_sachs": {
    "n_adequate_trials_failed": 3,
    "polarity_analysis": "breakthrough_mania_on_3_agents_plus_breakthrough_depression",
    "resistance_stage": "stage_III",
    "lithium_response_alda": "non_responder",
    "note": "3 adequate mood stabilizer trials with breakthrough episodes; ALDA negative"
  },
  "current_stability": {
    "currently_stable": true,
    "duration_stability": "4_years",
    "residual_symptoms": "mild_depressive",
    "functional_status": "good",
    "recommendation": "maintain_current_regimen"
  }
}
```

**Expected reasoning artifact**: The model produces `extract` steps for each trial's parameters, `criterion` steps applying adequacy definitions (therapeutic dose range, minimum 6-month duration for mood stabilizers, serum level targets), `compute` steps calculating ALDA total score, `lookup` steps referencing Sachs staging criteria, and an `integrate` step synthesizing: "3 failed adequate trials = Stage III resistance, but current combination achieving stability — no indication to change."

---

## Category 3 — Diagnostic Reasoning (130 tasks)

### Description

These tasks train the core diagnostic skill: given a clinical presentation, reason systematically toward a diagnosis using DSM-5 criteria, consider differentials, and assign appropriate specifiers. This category produces the most complex reasoning artifacts because diagnosis integrates information across every domain — symptoms, course, family history, treatment response, comorbidities.

The oracle is the patient's actual FACE diagnosis. For differential diagnosis tasks, the oracle includes the correct diagnosis ranked within the top-3, plus verifiable supporting/opposing evidence for each candidate.

### Sub-types

| Sub-type | N | Code Modality | Description |
|----------|---|---------------|-------------|
| 3a. Primary diagnosis from presentation | 35 | text_with_structured_differential | Full diagnostic reasoning from clinical vignette |
| 3b. Differential diagnosis at diagnostic boundaries | 25 | text_with_structured_differential | BD-I with psychosis vs. schizoaffective; BD depression vs. TRD; ASD vs. SZ negative symptoms |
| 3c. DSM-5 specifier assignment | 25 | code_preferred | Assign specifiers (rapid cycling, mixed features, psychotic features, anxious distress, seasonal) |
| 3d. Comorbidity detection | 25 | code_preferred | Identify psychiatric (from MINI) and somatic comorbidities |
| 3e. Diagnostic revision triggers | 20 | text_with_structured_differential | Identify clinical signals suggesting the current diagnosis may need revision |

### Example Task — Sub-type 3b (Hard difficulty)

**Vignette**:

> Patient: Karim L., 27 ans, adressé au Centre Expert avec un diagnostic de schizophrénie posé il y a 3 ans.
>
> Antécédents: Premier épisode psychotique à 22 ans (idées de persécution, hallucinations auditives, retrait social progressif). Hospitalisé 3 semaines. Traitement par rispéridone 4 mg/j.
>
> Depuis le premier épisode: 2 rechutes psychotiques (à 24 ans et 26 ans), chacune précédée de 2–3 semaines d'insomnie, hyperactivité, dépenses excessives, et logorrhée. Après résolution des épisodes psychotiques, présence d'épisodes de ralentissement psychomoteur, hypersomnie, anhédonie durant 4–8 semaines.
>
> Évaluation actuelle:
> - PANSS: P=12, N=14, G=28, Total=54
> - MADRS: 8
> - YMRS: 3
> - Histoire familiale: mère — trouble bipolaire de type I; oncle maternel — schizophrénie
>
> MINI: Pas de trouble lié à l'usage de substances. Pas de trouble anxieux actuel.
>
> Fonctionnement cognitif: WAIS-IV QI total = 98. CVLT rappel différé = z-score -0.4 (normal). TMT-B = z-score -0.8 (normal).
>
> Fonctionnement: GAF = 58, travaille à mi-temps.
>
> Question: Évaluez si le diagnostic actuel de schizophrénie est approprié. Considérez les diagnostics alternatifs, en détaillant les arguments pour et contre chaque hypothèse.

**Oracle ground truth**:
```json
{
  "current_diagnosis_appropriate": false,
  "recommended_diagnosis": "schizoaffective_disorder_bipolar_type",
  "differential": [
    {
      "diagnosis": "schizoaffective_disorder_bipolar_type",
      "rank": 1,
      "supporting": [
        "Psychotic episodes co-occur with mood episodes (manic prodrome before each psychotic relapse)",
        "Mood episodes (manic + depressive) present for majority of illness duration",
        "Family history of BD-I in mother",
        "Preserved cognitive function (IQ 98, normal memory and executive function)",
        "Relatively preserved functioning (GAF 58, employed part-time)"
      ],
      "opposing": [
        "Negative symptoms present (PANSS-N=14) though mild",
        "Initial presentation was primarily psychotic"
      ]
    },
    {
      "diagnosis": "bipolar_I_with_psychotic_features",
      "rank": 2,
      "supporting": [
        "All psychotic episodes preceded by manic prodromes",
        "Mood-congruent timing of psychosis",
        "Family history supports bipolar spectrum",
        "Preserved cognition atypical for schizophrenia"
      ],
      "opposing": [
        "Requires confirmation that psychosis occurs ONLY during mood episodes — unclear from history whether psychotic symptoms persisted between mood episodes"
      ]
    },
    {
      "diagnosis": "schizophrenia",
      "rank": 3,
      "supporting": [
        "Current diagnosis — 3 psychotic episodes meet criterion A duration",
        "PANSS-N=14 suggests some negative symptoms"
      ],
      "opposing": [
        "Temporal pattern suggests mood episodes drive psychotic relapses, not independent",
        "Normal cognition atypical for schizophrenia of 5-year duration",
        "Depressive episodes between psychotic episodes not explained by schizophrenia",
        "Family history favors affective spectrum"
      ]
    }
  ],
  "key_discriminating_question": "Were psychotic symptoms ever present for ≥2 weeks WITHOUT concurrent mood symptoms? If no → BD-I with psychotic features. If yes → schizoaffective.",
  "specifiers_if_schizoaffective": ["bipolar_type"]
}
```

**Expected reasoning artifact**: The model uses `extract` steps to pull temporal relationships between psychotic and mood episodes, `criterion` steps applying DSM-5 criteria for each candidate diagnosis, a `differential` step with ranked candidates and evidence, and an `integrate` step identifying the key discriminating question that would resolve diagnostic uncertainty.

---

## Category 4 — Suicide Risk Assessment (80 tasks)

### Description

The highest-stakes task type. Training here uses asymmetric reward: underestimation is penalized 3× more than overestimation, and the safety gate zeros the entire reward for severe underestimation (predicting "low" when truth is "high" or "imminent"). These tasks require the model to integrate static risk factors (prior attempts, family history, childhood trauma), dynamic factors (current mood, hopelessness, substance use, social isolation), and protective factors into a structured risk stratification.

The oracle applies a structured risk algorithm combining weighted static and dynamic factors with protective factor adjustment.

### Sub-types

| Sub-type | N | Code Modality | Description |
|----------|---|---------------|-------------|
| 4a. Structured risk stratification | 35 | text_with_safety_check | Full risk assessment from clinical data |
| 4b. Risk change detection (longitudinal) | 25 | code_required | Compare risk profiles across consecutive visits, identify drivers of change |
| 4c. Safety plan element identification | 20 | text_with_safety_check | Given risk profile, identify appropriate safety planning components |

### Example Task — Sub-type 4a (Medium difficulty)

**Vignette**:

> Patiente: Amélie R., 34 ans, trouble bipolaire de type II, suivie depuis 2019.
>
> Antécédents suicidaires: Une tentative de suicide par intoxication médicamenteuse en 2017 (avant diagnostic), létalité modérée (hospitalisation en réanimation 48h). Pas de tentative depuis.
>
> Évaluation actuelle:
> - MADRS total: 28 (item 10 — pensées suicidaires: 3/6 — « pensées suicidaires fréquentes »)
> - C-SSRS: Idéation suicidaire active avec méthode (intoxication), sans intention immédiate, sans plan précis
> - BIS-11 total: 78 (impulsivité élevée)
> - CTQ: Abus émotionnel sévère (score 22/25), négligence physique modérée
>
> Facteurs contextuels:
> - Séparation conjugale il y a 2 mois
> - Arrêt de travail depuis 6 semaines
> - Consommation d'alcool augmentée (3–4 verres/jour, contre 0–1 habituellement)
> - Vit seule, un enfant de 5 ans en garde alternée
>
> Facteurs protecteurs:
> - Alliance thérapeutique décrite comme bonne par la psychiatre
> - Enfant identifié comme raison de vivre
> - Pas d'accès à des moyens létaux (armes, médicaments dangereux retirés après la TS de 2017)
>
> Traitement: Lamotrigine 200 mg/j, Quétiapine LP 300 mg/j
>
> Question: Stratifiez le risque suicidaire, identifiez les facteurs de risque aigus et chroniques, les facteurs protecteurs, et recommandez le niveau de surveillance approprié.

**Oracle ground truth**:
```json
{
  "risk_level": "high",
  "static_risk_factors": [
    "prior_suicide_attempt",
    "female_sex_for_attempt",
    "bipolar_disorder",
    "childhood_adversity_severe",
    "high_impulsivity"
  ],
  "dynamic_risk_factors": [
    "current_active_suicidal_ideation_with_method",
    "moderate_severe_depression_madrs_28",
    "recent_relationship_loss",
    "social_isolation_increased",
    "alcohol_use_increased",
    "occupational_loss"
  ],
  "protective_factors": [
    "child_as_reason_for_living",
    "therapeutic_alliance",
    "means_restriction_in_place",
    "no_immediate_intent"
  ],
  "risk_rationale": "Elevated risk due to convergence of prior attempt history, current active ideation with identified method, acute psychosocial stressors (separation, job loss), increased alcohol use, and severe depression. Protective factors (child, therapeutic alliance, means restriction) mitigate but do not eliminate risk. Absence of immediate intent or specific plan prevents classification as imminent.",
  "recommended_surveillance": "high_frequency_outpatient",
  "recommended_actions": [
    "increase_visit_frequency_to_weekly",
    "safety_plan_update",
    "alcohol_use_intervention",
    "consider_antidepressant_augmentation",
    "reassess_means_access",
    "emergency_contact_verification"
  ]
}
```

### Difficulty Calibration

| Level | Characteristics |
|-------|----------------|
| Easy | Clearly low risk (no history, no ideation, good support) or clearly imminent (active plan with intent) |
| Medium | Mixed factors as in the example — model must weigh competing signals |
| Hard | Near-boundary cases: moderate ideation + strong protective factors; patient minimizing symptoms but objective markers elevated; longitudinal risk that has changed subtly between visits |

---

## Category 5 — Cognitive Assessment (100 tasks)

### Description

Neuropsychological assessment interpretation requires transforming raw test scores into clinically meaningful cognitive profiles. These tasks are heavily code-dependent because they involve norm lookups, z-score conversions, and multi-domain profile construction. The FACE database contains neuropsychological batteries across all four cohorts, making this one of the richest data sources.

### Sub-types

| Sub-type | N | Code Modality | Description |
|----------|---|---------------|-------------|
| 5a. Neuropsych profile construction | 40 | code_required | Raw scores → z-scores → domain impairment classification |
| 5b. Cognitive trajectory analysis | 25 | code_required | Detect reliable change across visits using RCI methodology |
| 5c. Cognitive-functional discrepancy | 20 | code_required | Compare objective cognitive performance against functional outcomes |
| 5d. Pattern interpretation (diagnosis-specific) | 15 | code_preferred | Interpret whether cognitive profile is consistent with diagnosis |

### Example Task — Sub-type 5a (Medium difficulty)

**Vignette**:

> Patient: Thomas B., 42 ans, schizophrénie, diagnostic posé à 21 ans, 15 ans d'éducation.
>
> Bilan neuropsychologique:
> - WAIS-IV: Vocabulaire = 9, Similitudes = 8, Cubes = 7, Matrices = 7, Mémoire des chiffres = 6 (endroit=7, envers=5, séquençage=5), Code = 5, Symboles = 5
> - TMT: Partie A = 42s, Partie B = 118s
> - Fluences verbales: Phonémique (lettre P) = 11 mots/2min, Catégorielle (animaux) = 14 mots/2min
> - CVLT-II: Rappel immédiat total (essais 1–5) = 38, Rappel différé libre = 7, Reconnaissance = 14/16
> - TAP Alerte: Médiane TR = 285 ms
> - TAP Flexibilité: Médiane TR = 890 ms, Erreurs = 8
>
> Normes de référence: Homme, 42 ans, 15 ans d'éducation.
>
> Question: Convertissez les scores bruts en scores z (normes âge/éducation), identifiez les domaines cognitifs altérés (z < -1.5 = altération modérée, z < -2.0 = altération sévère), et produisez un profil cognitif synthétique.

**Oracle ground truth**:
```json
{
  "domain_scores": {
    "processing_speed": {
      "tests": ["WAIS_Coding", "WAIS_Symbol_Search", "TMT_A"],
      "z_scores": [-1.67, -1.67, -0.8],
      "composite_z": -1.38,
      "classification": "mild_to_moderate_impairment"
    },
    "working_memory": {
      "tests": ["WAIS_Digit_Span_total", "WAIS_Digit_Span_backward"],
      "z_scores": [-1.33, -1.67],
      "composite_z": -1.50,
      "classification": "moderate_impairment"
    },
    "executive_function": {
      "tests": ["TMT_B", "Phonemic_fluency", "TAP_Flexibility"],
      "z_scores": [-1.9, -1.5, -2.1],
      "composite_z": -1.83,
      "classification": "moderate_impairment"
    },
    "verbal_memory": {
      "tests": ["CVLT_total_learning", "CVLT_delayed_recall", "CVLT_recognition"],
      "z_scores": [-1.2, -1.5, -0.5],
      "composite_z": -1.07,
      "classification": "mild_impairment"
    },
    "verbal_intelligence": {
      "tests": ["WAIS_Vocabulary", "WAIS_Similarities"],
      "z_scores": [-0.33, -0.67],
      "composite_z": -0.50,
      "classification": "normal"
    }
  },
  "primary_impairments": ["executive_function", "working_memory", "processing_speed"],
  "preserved_domains": ["verbal_intelligence", "verbal_memory"],
  "profile_summary": "Fronto-subcortical profile with primary executive and processing speed deficits, relative preservation of verbal crystallized abilities, consistent with schizophrenia cognitive phenotype",
  "clinical_note": "Verbal IQ preservation (Vocabulary/Similarities normal) suggests adequate premorbid functioning with illness-related cognitive decline in executive and speed domains"
}
```

**Expected reasoning artifact**: Multiple `compute` steps converting each raw score to z-score (referencing norm tables), `criterion` steps classifying each domain, and an `integrate` step producing the clinical profile interpretation. The code blocks perform the actual z-score arithmetic.

---

## Category 6 — Longitudinal Trajectory Analysis (120 tasks)

### Description

The FACE database's greatest structural advantage is longitudinal follow-up — up to 23 visits per patient for bipolar disorder, 12 for schizophrenia. These tasks train the model to reason about temporal patterns: Is the patient improving, declining, cycling, or stable? Is a change real or measurement noise? Does a trajectory predict relapse?

Every task requires code execution because temporal analysis involves statistical computations (RCI, trend tests, rate-of-change).

### Sub-types

| Sub-type | N | Code Modality | Description |
|----------|---|---------------|-------------|
| 6a. Clinically significant change detection | 30 | code_required | Apply RCI to determine whether score change is real vs. noise |
| 6b. Treatment response classification | 30 | code_required | Classify as responder / partial / non-responder / remitter |
| 6c. Episode trajectory characterization | 25 | code_required | Classify illness pattern (predominantly depressive, rapid cycling, stable, progressive) |
| 6d. Relapse detection and prediction | 20 | code_required | Identify relapse events in longitudinal data, early warning signals |
| 6e. Functioning-symptom trajectory comparison | 15 | code_required | Detect functional lag or discordance with symptom trajectory |

### Example Task — Sub-type 6c (Hard difficulty)

**Vignette**:

> Patiente: Nadia K., 45 ans, trouble bipolaire de type I, suivie au Centre Expert depuis 2012.
>
> Scores longitudinaux:
>
> | Visite | Date | MADRS | YMRS | CGI-S | GAF |
> |--------|------|-------|------|-------|-----|
> | V1 | 03/2012 | 24 | 2 | 4 | 48 |
> | V2 | 09/2012 | 8 | 3 | 2 | 65 |
> | V3 | 03/2013 | 6 | 18 | 4 | 42 |
> | V4 | 09/2013 | 28 | 1 | 5 | 35 |
> | V5 | 03/2014 | 12 | 2 | 3 | 55 |
> | V6 | 09/2014 | 4 | 22 | 5 | 38 |
> | V7 | 03/2015 | 30 | 0 | 5 | 30 |
> | V8 | 09/2015 | 10 | 4 | 2 | 62 |
> | V9 | 03/2016 | 6 | 14 | 4 | 45 |
> | V10 | 09/2016 | 26 | 1 | 4 | 40 |
> | V11 | 03/2017 | 5 | 20 | 5 | 35 |
> | V12 | 09/2017 | 22 | 2 | 4 | 42 |
>
> Question: Caractérisez le pattern de trajectoire de cette patiente. Identifiez la présence de cycling rapide, la polarité prédominante, la tendance de fonctionnement à long terme, et les épisodes vérifiant les seuils de rechute (MADRS ≥ 20 pour dépression, YMRS ≥ 12 pour manie/hypomanie).

**Oracle ground truth**:
```json
{
  "episode_count": {
    "depressive_relapses": [1, 4, 7, 10, 12],
    "manic_hypomanic_relapses": [3, 6, 9, 11],
    "total_depressive": 5,
    "total_manic": 4
  },
  "rapid_cycling": {
    "qualifies": true,
    "criterion": ">=4 mood episodes per 12-month period",
    "evidence": "2012-2013: 4 episodes (dep V1, manic V3, dep V4, euthymia V5 then manic V6 starts next year). 2014-2015: 4 episodes. Pattern sustained across all years."
  },
  "predominant_polarity": {
    "polarity": "no_clear_predominance",
    "ratio": "5 depressive : 4 manic",
    "note": "Near-equal distribution, slight depressive predominance"
  },
  "functioning_trend": {
    "direction": "declining",
    "mean_gaf_first_half": 47.2,
    "mean_gaf_second_half": 40.3,
    "interpretation": "Gradual functional decline despite inter-episode recovery, suggesting incomplete functional restoration (functional scar)"
  },
  "inter_episode_residual": {
    "best_gaf_achieved": 65,
    "euthymic_visits": [2, 5, 8],
    "euthymic_gaf_trend": [65, 55, 62],
    "interpretation": "Euthymic GAF appears stable around 55-65, but time in euthymia is decreasing"
  },
  "trajectory_classification": "rapid_cycling_mixed_polarity_with_functional_decline"
}
```

---

## Category 7 — Rating Scale Interpretation (80 tasks)

### Description

These tasks serve a specific role in the training curriculum: they are the simplest complete instantiation of the structured reasoning schema and are used heavily in Stage 1 RL as format bootstrapping anchors. The easy variants are near-pure arithmetic; the training value comes from teaching the model the output format with minimal cognitive load. The medium and hard variants add cross-scale integration and clinical interpretation.

This category is intentionally smaller (80 tasks) because its primary function is curriculum support, not standalone clinical skill.

### Sub-types

| Sub-type | N | Code Modality | Description |
|----------|---|---------------|-------------|
| 7a. Single-scale scoring + severity | 25 | code_required | Compute total, subscales, classify severity |
| 7b. Multi-scale integration | 30 | code_required | Interpret 3–5 scales from same visit, produce integrated state summary |
| 7c. Self-report vs. clinician discrepancy | 25 | code_required | Compare self-report and clinician-rated measures, interpret discordance |

### Example Task — Sub-type 7c (Medium difficulty)

**Vignette**:

> Patient: Antoine P., 52 ans, trouble bipolaire de type I.
>
> Évaluation auto-rapportée:
> - QIDS-SR16 (auto-questionnaire dépression): score total = 18 (sévérité sévère)
> - Altman Self-Rating Mania Scale: score = 2 (non significatif)
> - COBRA (plaintes cognitives): score total = 22 (plaintes élevées)
>
> Évaluation clinique:
> - MADRS (cotée par le clinicien): score total = 11 (sévérité légère)
> - YMRS (cotée par le clinicien): score = 4 (non significatif)
> - CGI-S: 2 (malade à la limite)
>
> Bilan neuropsychologique: WAIS-IV QI = 105, CVLT rappel différé z = -0.2, TMT-B z = -0.5 (tout dans la norme)
>
> Question: Analysez la discordance entre les auto-évaluations et les évaluations cliniques. Quantifiez l'écart, proposez des hypothèses explicatives, et déterminez quel ensemble de mesures est plus fiable pour orienter la décision clinique.

**Oracle ground truth**:
```json
{
  "discrepancy_analysis": {
    "depression": {
      "self_report": {"instrument": "QIDS-SR16", "score": 18, "severity": "severe"},
      "clinician_rated": {"instrument": "MADRS", "score": 11, "severity": "mild"},
      "discordance": "patient_inflated",
      "magnitude": "2_severity_levels"
    },
    "cognition": {
      "subjective": {"instrument": "COBRA", "score": 22, "interpretation": "high_complaints"},
      "objective": {"instruments": ["WAIS-IV", "CVLT", "TMT-B"], "interpretation": "normal_performance"},
      "discordance": "subjective_greater_than_objective"
    }
  },
  "hypotheses": [
    "Cognitive distortion secondary to subsyndromal depression amplifying symptom perception",
    "Residual insight deficit following recent episode",
    "Somatization: physical symptoms scored on QIDS (items on appetite, sleep, energy) not captured equivalently by MADRS",
    "Metacognitive monitoring deficit: common in BD, subjective cognitive complaints uncorrelated with objective performance"
  ],
  "clinical_recommendation": {
    "more_reliable_for_severity": "clinician_rated",
    "more_reliable_for_subjective_burden": "self_report",
    "action": "Use MADRS for treatment decisions (mild depression, no medication change needed); address subjective distress through psychoeducation about cognitive complaints in BD"
  }
}
```

---

## Category 8 — Side Effect & Safety Monitoring (80 tasks)

### Description

Medication side effects are a primary driver of non-adherence and treatment discontinuation in psychiatry. These tasks train the model to attribute side effects to specific medications, assess severity using standardized scales, and determine whether side effects warrant medication adjustment. The FACE database records side effect data through PRISEM (Patient Rated Inventory of Side Effects), UKU Side Effect Rating Scale, AIMS (Abnormal Involuntary Movement Scale), and Barnes Akathisia Rating Scale.

### Sub-types

| Sub-type | N | Code Modality | Description |
|----------|---|---------------|-------------|
| 8a. Side effect attribution | 30 | code_preferred | Given medications + reported side effects, attribute to specific agent |
| 8b. Movement disorder assessment (SZ) | 20 | code_required | Interpret AIMS and Barnes scores, assess tardive dyskinesia and akathisia |
| 8c. Monitoring protocol compliance | 15 | code_required | Check if all required lab monitoring is current for each medication |
| 8d. Risk-benefit integration | 15 | text_with_structured_differential | Weigh therapeutic benefit against side effect burden to recommend action |

### Example Task — Sub-type 8a (Medium difficulty)

**Vignette**:

> Patiente: Claire S., 38 ans, trouble bipolaire de type I.
>
> Traitement actuel (depuis 8 mois):
> - Lithium 1000 mg/j (lithiémie: 0.72 mmol/L)
> - Quétiapine LP 400 mg/j (ajoutée il y a 3 mois pour épisode maniaque)
>
> Plaintes rapportées (PRISEM):
> - Tremblement des mains: modéré
> - Prise de poids: +8 kg en 3 mois
> - Somnolence diurne: sévère
> - Soif excessive / polyurie: modérée
> - Acné: léger
> - Constipation: modéré
>
> Bilan:
> - TSH: 5.8 mUI/L (limite haute, ref: 0.4–4.0)
> - Créatinine: 95 µmol/L (normal)
> - Glycémie à jeun: 5.9 mmol/L (limite haute)
>
> Question: Attribuez chaque effet secondaire au(x) médicament(s) le(s) plus probable(s), classez la sévérité globale de la charge en effets secondaires, et identifiez les alertes nécessitant une action immédiate.

**Oracle ground truth**:
```json
{
  "attributions": {
    "tremor": {"primary": "lithium", "secondary": null, "confidence": "high", "mechanism": "dose-dependent neurotoxicity"},
    "weight_gain": {"primary": "quetiapine", "secondary": "lithium", "confidence": "high", "mechanism": "H1 antagonism (quetiapine) + fluid retention/appetite (lithium)"},
    "daytime_somnolence": {"primary": "quetiapine", "secondary": null, "confidence": "high", "mechanism": "H1 antihistaminic effect"},
    "polydipsia_polyuria": {"primary": "lithium", "secondary": null, "confidence": "high", "mechanism": "nephrogenic diabetes insipidus"},
    "acne": {"primary": "lithium", "secondary": null, "confidence": "moderate", "mechanism": "lithium dermatopathy"},
    "constipation": {"primary": "quetiapine", "secondary": null, "confidence": "moderate", "mechanism": "anticholinergic effect"}
  },
  "global_side_effect_burden": "moderate_to_severe",
  "alerts": [
    {"finding": "TSH 5.8 — lithium-induced hypothyroidism", "urgency": "action_within_weeks", "recommendation": "Start levothyroxine, recheck TSH in 6 weeks"},
    {"finding": "Weight gain 8 kg in 3 months — clinically significant (>7%)", "urgency": "action_at_next_visit", "recommendation": "Dietary counseling, consider quetiapine dose reduction or switch"},
    {"finding": "Fasting glucose 5.9 — impaired fasting glucose, pre-diabetes risk", "urgency": "monitoring", "recommendation": "OGTT or HbA1c, repeat in 3 months"}
  ]
}
```

---

## Category 9 — Transdiagnostic & Cross-Cohort Reasoning (60 tasks)

### Description

These tasks exploit the unique structural advantage of the FACE database: four psychiatric cohorts assessed with partially overlapping instruments. No other dataset enables tasks requiring reasoning across diagnostic boundaries. These are the highest-difficulty tasks in the dataset and are reserved primarily for Stage 2–3 RL training.

### Sub-types

| Sub-type | N | Code Modality | Description |
|----------|---|---------------|-------------|
| 9a. Cross-cohort profile comparison | 20 | code_required | Compare two patients from different cohorts on shared assessment domains |
| 9b. Diagnostic boundary reasoning | 20 | text_with_structured_differential | Reason about presentations overlapping two diagnostic categories |
| 9c. Transdiagnostic factor identification | 20 | code_preferred | Distinguish transdiagnostic from diagnosis-specific risk factors in a patient |

### Example Task — Sub-type 9a (Hard difficulty)

**Vignette**:

> Deux patients évalués au Centre Expert le même mois:
>
> **Patient A** — Lucie T., 29 ans, trouble bipolaire de type I:
> - MADRS: 6, YMRS: 3 (euthymique)
> - CGI-S: 2, GAF: 62
> - Neuropsychologie: WAIS-IV QI=112, CVLT rappel différé z=-0.3, TMT-B z=-1.1
> - Métabolique: IMC 24.2, glycémie 4.8, triglycérides 1.1, HDL 1.4
> - CTQ: Abus émotionnel score 8 (minimal), total 32 (faible)
> - BIS-11: 72 (élevé)
> - Tentatives de suicide: 1
> - Traitements: Lithium 800 mg/j (seul thymorégulateur)
>
> **Patient B** — Marc V., 31 ans, schizophrénie:
> - PANSS: P=10, N=22, G=30, Total=62
> - CGI-S: 4, GAF: 45
> - Neuropsychologie: WAIS-IV QI=94, CVLT rappel différé z=-1.8, TMT-B z=-2.2
> - Métabolique: IMC 31.5, glycémie 6.1, triglycérides 2.4, HDL 0.9
> - CTQ: Abus émotionnel score 18 (sévère), total 68 (élevé)
> - BIS-11: 58 (normal)
> - Tentatives de suicide: 0
> - Traitements: Olanzapine 15 mg/j, Aripiprazole 10 mg/j
>
> Question: Comparez systématiquement ces deux patients sur chaque domaine d'évaluation partagé. Identifiez les domaines où la sévérité est comparable, ceux où elle diverge, et déterminez les facteurs transdiagnostiques vs. spécifiques au diagnostic.

**Oracle ground truth**:
```json
{
  "domain_comparison": {
    "symptom_severity": {"patient_a": "euthymic_cgi_2", "patient_b": "moderately_ill_cgi_4", "more_severe": "B"},
    "functioning": {"patient_a": "gaf_62_mild_impairment", "patient_b": "gaf_45_serious_impairment", "more_severe": "B"},
    "cognition_executive": {"patient_a": "tmtb_z_-1.1_mild", "patient_b": "tmtb_z_-2.2_severe", "more_severe": "B"},
    "cognition_memory": {"patient_a": "cvlt_z_-0.3_normal", "patient_b": "cvlt_z_-1.8_impaired", "more_severe": "B"},
    "cognition_iq": {"patient_a": "112_high_average", "patient_b": "94_average", "more_severe": "B"},
    "metabolic_risk": {"patient_a": "normal_bmi_24_no_mets", "patient_b": "obese_bmi_31_mets_criteria", "more_severe": "B"},
    "childhood_trauma": {"patient_a": "low_ctq_32", "patient_b": "high_ctq_68", "more_severe": "B"},
    "impulsivity": {"patient_a": "bis_72_elevated", "patient_b": "bis_58_normal", "more_severe": "A"},
    "suicide_risk": {"patient_a": "1_prior_attempt", "patient_b": "0_attempts", "more_severe": "A"}
  },
  "transdiagnostic_factors": [
    "Cognitive impairment (present in both, more severe in SZ — transdiagnostic but diagnosis-modulated)",
    "Childhood adversity (present in B, minimal in A — transdiagnostic risk factor)",
    "Metabolic risk (B has iatrogenic metabolic syndrome from olanzapine — transdiagnostic medication effect)"
  ],
  "diagnosis_specific_factors": [
    "Impulsivity elevated in BD (trait feature of bipolar), normal in SZ",
    "Negative symptoms dominant in SZ (PANSS-N=22), no equivalent dimension in BD",
    "Suicide attempt history in BD despite euthymia — reflects BD-specific risk trajectory",
    "Memory impairment in SZ (CVLT z=-1.8) disproportionate to BD (z=-0.3) — SZ-specific cognitive phenotype"
  ]
}
```

---

## Category 10 — Data Quality & Clinical Reasoning Meta-Tasks (70 tasks)

### Description

These tasks are unique in the medical LLM landscape: they train the model to reason about what it *cannot* conclude from incomplete, inconsistent, or implausible data. Real clinical data is messy — values are missing, self-report contradicts clinician assessment, data entry errors create implausible values, and instrument versions change over time. A model that cannot recognize these limitations will hallucinate confident conclusions from insufficient evidence.

### Sub-types

| Sub-type | N | Code Modality | Description |
|----------|---|---------------|-------------|
| 10a. Missing data impact assessment | 25 | code_preferred | Given a profile with specific missing values, reason about what can and cannot be concluded |
| 10b. Data plausibility detection | 20 | code_required | Identify likely data entry errors or implausible values |
| 10c. Clinical timeline reconstruction | 25 | code_required | From episode history, treatment dates, hospitalizations, reconstruct coherent timeline and flag inconsistencies |

### Example Task — Sub-type 10a (Medium difficulty)

**Vignette**:

> Patient: Youssef A., 36 ans, trouble bipolaire de type II, V3 (suivi annuel).
>
> Données disponibles à cette visite:
> - MADRS: 14
> - YMRS: non administré (mention: "patient pressé, questionnaire non complété")
> - CGI-S: 3
> - GAF: 55
> - Bilan biologique: non réalisé (mention: "patient à jeun impossible, report au mois prochain")
> - Neuropsychologie: non réalisée à cette visite
> - Traitement actuel: Lamotrigine 200 mg/j, Quétiapine 200 mg/j
> - FAST: score total 32 (altération modérée)
> - QIDS-SR16: 12 (dépression modérée)
>
> Questions cliniques posées:
> 1. Le patient est-il en épisode thymique actuel?
> 2. Son syndrome métabolique est-il contrôlé?
> 3. Peut-on évaluer la résistance au traitement?
> 4. Le fonctionnement est-il concordant avec la symptomatologie?
>
> Question: Pour chaque question clinique, déterminez si les données disponibles permettent d'y répondre. Si non, précisez quelles données manquantes sont nécessaires et quel est l'impact de leur absence.

**Oracle ground truth**:
```json
{
  "question_1_mood_episode": {
    "answerable": "partially",
    "available_evidence": "MADRS=14 (mild depression), QIDS=12 (moderate depression), CGI=3 (mildly ill)",
    "missing_critical": "YMRS not administered — cannot rule out mixed features or hypomanic symptoms",
    "conclusion": "Evidence suggests subsyndromal depressive symptoms but absence of YMRS prevents complete mood state characterization. In BD-II, hypomanic symptoms can co-occur with depression (mixed features) and would change management.",
    "required_data": ["YMRS"]
  },
  "question_2_metabolic": {
    "answerable": "no",
    "available_evidence": "None — no lab panel at this visit",
    "missing_critical": "Fasting glucose, lipid panel, waist circumference, blood pressure",
    "conclusion": "Cannot assess metabolic syndrome status. Patient is on quetiapine which carries metabolic risk. This monitoring gap should be flagged.",
    "required_data": ["fasting_glucose", "lipid_panel", "waist_circumference", "blood_pressure"]
  },
  "question_3_treatment_resistance": {
    "answerable": "partially",
    "available_evidence": "Current regimen known (lamotrigine + quetiapine), MADRS=14 suggests incomplete remission",
    "missing_critical": "Full treatment history at this visit not repeated — need prior visit data for adequate trial assessment",
    "conclusion": "Can note incomplete remission on current regimen but cannot stage resistance without reviewing complete pharmacological history from prior visits.",
    "required_data": ["complete_medication_history", "prior_trial_durations_and_doses"]
  },
  "question_4_functioning_concordance": {
    "answerable": "yes",
    "available_evidence": "MADRS=14 (mild), FAST=32 (moderate functional impairment), GAF=55 (moderate)",
    "conclusion": "Functional impairment (FAST 32, GAF 55) exceeds what mild depression (MADRS 14) would predict. This functional lag is common in BD-II and suggests residual functional deficit beyond current symptom level. Cognitive assessment would help determine whether cognitive impairment contributes to the discrepancy.",
    "concordance": "discordant_function_worse_than_symptoms"
  }
}
```

---

## Implementation Notes

### Patient Selection Strategy

For each task, select patients from the training pool (70% of FACE) using stratified sampling:

1. **Completeness filter**: Task requires variable set V; only select patients where ≥80% of V is non-missing (for easy/medium) or 60–80% (for hard tasks that deliberately include missing data).
2. **Difficulty assignment**: Easy = unambiguous presentation, complete data, single domain. Medium = comorbidity or borderline values. Hard = multi-morbidity, missing data, conflicting indicators.
3. **Cohort balance**: Maintain target proportions (35% BD, 25% SZ, 20% DR, 20% ASD) within each category where clinically applicable.

### Oracle Function Registry

Each task sub-type has a registered oracle function in the pipeline:

```
oracle_registry = {
    "1a_metabolic_syndrome": oracle_metabolic_syndrome_idf,
    "1b_lab_panel_interpretation": oracle_lab_panel_flags,
    "1c_cardiovascular_risk": oracle_framingham_score2,
    "2a_treatment_adequacy": oracle_adequate_trial_check,
    "2b_treatment_resistance": oracle_staging_algorithm,
    "2d_drug_interactions": oracle_interaction_database,
    "3a_primary_diagnosis": oracle_face_diagnosis,
    "4a_suicide_risk": oracle_structured_risk_algorithm,
    "5a_neuropsych_profile": oracle_zscore_computation,
    "6a_reliable_change": oracle_rci_computation,
    "6b_treatment_response": oracle_response_classification,
    "7a_scale_scoring": oracle_scale_computation,
    "10b_plausibility": oracle_range_checks,
    ...
}
```

### Vignette Language

All vignettes are generated in French with synthetic French demographics. Clinical instrument abbreviations remain in their standard form (MADRS, PANSS, CVLT, etc.) since these are internationally standardized. Reference tables and guideline criteria are cited in their published form.

### Quality Targets

| Metric | Target |
|--------|--------|
| Oracle verification pass rate (generation) | ≥ 95% of tasks have verifiable ground truth |
| LLM generation acceptance rate | ≥ 80% pass composite reward ≥ 0.8 after ≤ 3 retries |
| Difficulty distribution achieved | Within ±5% of target (30/40/30) per category |
| Cohort distribution achieved | Within ±5% of target per category |
| Unique patients used | ≥ 600 (avoid over-representing individual patients) |
