"""Pydantic v2 models for every YAML schema under config/glossary/.

These models validate YAML on load and provide a statically-typed view.
They do NOT replace the dataclasses in common_instruments.py or
common_extractors.py — the loader converts pydantic models into the
existing dataclasses so downstream code is untouched.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ─── Instruments & thresholds ────────────────────────────────────────────────


class SeverityBandConfig(_Base):
    min_score: float
    max_score: float
    code: str
    label_fr: str
    clinical_meaning_fr: str

    @model_validator(mode="after")
    def _check_range(self):
        if self.min_score > self.max_score:
            raise ValueError(
                f"SeverityBand {self.code}: min_score {self.min_score} > max_score {self.max_score}"
            )
        return self


class ThresholdBandsConfig(_Base):
    source: Optional[str] = None
    bands: list[SeverityBandConfig]


class InstrumentConfig(_Base):
    name: str
    full_name: str = ""
    full_name_fr: str = ""
    domain: str = ""
    total_column: str
    subscale_columns: dict[str, str] = Field(default_factory=dict)
    score_range: tuple[float, float] = (0, 100)
    higher_is_worse: bool = True
    evaluation_type: Literal["hetero", "auto"] = "hetero"
    unit: str = ""
    clinical_note_fr: str = ""

    severity_thresholds: list[SeverityBandConfig] = Field(default_factory=list)
    severity_thresholds_ref: Optional[str] = None
    screening_threshold: Optional[float] = None
    screening_positive_label_fr: str = "Dépistage positif"
    screening_negative_label_fr: str = "Dépistage négatif"

    @model_validator(mode="after")
    def _check_exclusive_mechanism(self):
        if self.severity_thresholds and self.severity_thresholds_ref:
            raise ValueError(
                f"{self.name}: cannot set both severity_thresholds and severity_thresholds_ref"
            )
        return self

    @field_validator("score_range")
    @classmethod
    def _check_range(cls, v):
        if v[0] > v[1]:
            raise ValueError(f"score_range {v}: lo > hi")
        return v


class InstrumentRegistryConfig(_Base):
    order: list[str]
    groups: dict[str, list[str]] = Field(default_factory=dict)


class InstrumentsFileConfig(_Base):
    schema_version: int = Field(alias="$schema_version")
    registry: Optional[InstrumentRegistryConfig] = Field(default=None, alias="$registry")
    instruments: dict[str, InstrumentConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _absorb_instruments(cls, values):
        if not isinstance(values, dict):
            return values
        known = {"$schema_version", "$registry", "instruments"}
        instruments = values.setdefault("instruments", {})
        for k in list(values.keys()):
            if k in known:
                continue
            v = values.pop(k)
            # Auto-inject the instrument key as name if missing
            if isinstance(v, dict) and "name" not in v:
                v["name"] = k
            instruments[k] = v
        return values


class ThresholdsFileConfig(_Base):
    schema_version: int = Field(alias="$schema_version")
    bands: dict[str, ThresholdBandsConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _absorb(cls, values):
        if not isinstance(values, dict):
            return values
        bands = values.setdefault("bands", {})
        for k in list(values.keys()):
            if k in {"$schema_version", "bands"}:
                continue
            bands[k] = values.pop(k)
        return values


# ─── Lab ranges ──────────────────────────────────────────────────────────────


class LabRangeConfig(_Base):
    csv_col: str
    name: str
    name_fr: str
    unit: str
    normal_range: Optional[tuple[float, float]] = None
    sex_specific: bool = False

    @field_validator("normal_range")
    @classmethod
    def _check_range(cls, v):
        if v is not None and v[0] > v[1]:
            raise ValueError(f"normal_range {v}: lo > hi")
        return v


class LabRangesFileConfig(_Base):
    schema_version: int = Field(alias="$schema_version")
    labs: list[LabRangeConfig]


# ─── Clinical constants ──────────────────────────────────────────────────────


class BMICategoryConfig(_Base):
    max: Optional[float]  # null == +infinity
    code: str
    label_fr: str


class MetSynSexThresholds(_Base):
    M: float
    F: float


class MetSynHypertension(_Base):
    sbp_mmhg: float
    dbp_mmhg: float


class MetabolicSyndromeConfig(_Base):
    source: str
    minimum_criteria_met: int
    abdominal_obesity_cm: MetSynSexThresholds
    hypertriglyceridemia_mmol_l: float
    hdl_low_mmol_l: MetSynSexThresholds
    hypertension: MetSynHypertension
    hyperglycemia_mmol_l: float


class FraminghamPointRow(_Base):
    min: Optional[float] = None
    max: Optional[float] = None
    pts: int


class FraminghamSbpRows(_Base):
    untreated: list[FraminghamPointRow]
    treated: list[FraminghamPointRow]


class FraminghamRiskRow(_Base):
    max_points: Optional[int]  # null == +infinity
    risk_pct: float


class FraminghamCategoryRow(_Base):
    max_pct: Optional[float]  # null == +infinity
    label_fr: str


class FraminghamConfig(_Base):
    source: str
    age_bounds: tuple[int, int]
    age_points: dict[str, list[FraminghamPointRow]]
    cholesterol_points: list[FraminghamPointRow]
    hdl_points: list[FraminghamPointRow]
    sbp_points: FraminghamSbpRows
    smoking_pts: int
    diabetes_pts: dict[str, int]
    risk_table: list[FraminghamRiskRow]
    categories: list[FraminghamCategoryRow]


class SexSpecificLabRange(_Base):
    aliases: list[str] = Field(default_factory=list)
    M: tuple[float, float]
    F: tuple[float, float]


class DrugInteractionConfig(_Base):
    drug1: str
    drug2: str
    severity: Literal["contre_indication", "major", "moderate"]
    alert_fr: str


class AbsoluteThresholdConfig(_Base):
    min: Optional[float] = None
    max: Optional[float] = None


class MedicationLabAlertRule(_Base):
    id: str
    treatment_flag: Optional[str] = None
    treatment_flags_any: list[str] = Field(default_factory=list)
    lab_names: list[str]
    direction: Optional[Literal["high", "low"]] = None
    absolute_threshold: Optional[AbsoluteThresholdConfig] = None
    alert_fr: str
    stop_after_first_match: bool = False


class CognitiveNormsTableRow(_Base):
    # We allow arbitrary test names under each decade.
    model_config = ConfigDict(extra="allow")


class CognitiveZScoreBand(_Base):
    max: Optional[float]  # null == +infinity
    label_fr: str


class CognitiveNormsConfig(_Base):
    tmt: dict[str, Any]
    stroop: dict[str, Any]
    z_score_bands: list[CognitiveZScoreBand]


class ClinicalConstantsConfig(_Base):
    schema_version: int = Field(alias="$schema_version")
    bmi_categories: list[BMICategoryConfig]
    metabolic_syndrome: MetabolicSyndromeConfig
    framingham: FraminghamConfig
    sex_specific_lab_ranges: dict[str, SexSpecificLabRange]
    drug_interactions: list[DrugInteractionConfig]
    severity_prefixes: dict[str, str]
    medication_lab_alerts: list[MedicationLabAlertRule]
    cognitive_norms: CognitiveNormsConfig


# ─── Categorical codes ───────────────────────────────────────────────────────


class CategoricalCodesFileConfig(_Base):
    schema_version: int = Field(alias="$schema_version")
    codes: dict[str, dict[str, str]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _absorb(cls, values):
        if not isinstance(values, dict):
            return values
        codes = values.setdefault("codes", {})
        for k in list(values.keys()):
            if k in {"$schema_version", "codes"}:
                continue
            codes[k] = values.pop(k)
        return values


# ─── Column map (CSV extraction layout per cohort) ───────────────────────────
#
# These models describe WHICH CSV COLUMNS hold WHICH structured field. They
# encode no clinical logic — only the mapping from semantic field names
# (used by the extractors) to raw CSV column names.
#
# Most sections are loosely typed with ``dict[str, Any]`` so each cohort can
# add its own custom sections (e.g. autism diagnosis, treatment resistance)
# without schema churn. The commonly-shared sections are validated strictly.


class DemographicsColumns(_Base):
    age: str
    sex: str
    site_id: str
    arm: str
    marital_status: Optional[str] = None
    education_level: str
    employment: Optional[str] = None
    employment_alt: Optional[str] = None  # ASP has two employment columns
    social_protection: Optional[str] = None


class VitalsColumns(_Base):
    # Keys are BiologicalPanel.vitals dict keys; values are CSV columns
    model_config = ConfigDict(extra="allow")


class ECGColumns(_Base):
    model_config = ConfigDict(extra="allow")


class SubstanceUseColumns(_Base):
    model_config = ConfigDict(extra="allow")

    tobacco: str
    alcohol: Optional[str] = None
    alcohol_type: Optional[str] = None
    cannabis: Optional[str] = None
    substance_use_disorder: Optional[str] = None
    other_substances: dict[str, str] = Field(default_factory=dict)
    cigarettes_per_day: Optional[str] = None


class SuicideIndicatorsColumns(_Base):
    madrs_item10: Optional[str] = None
    madrs_elevated_threshold: float = 3.0
    lifetime_attempts: Optional[str] = None
    lifetime_self_harm: Optional[str] = None


class SuicideHistoryColumns(_Base):
    # ISF — Suicidal ideation/intent lifetime items
    ever_felt_life_not_worth: Optional[str] = None
    ever_wished_dead: Optional[str] = None
    ever_thought_suicide: Optional[str] = None
    ever_planned_suicide: Optional[str] = None
    ever_attempted: Optional[str] = None
    n_attempts: Optional[str] = None
    has_violent_attempts: Optional[str] = None
    n_violent_attempts: Optional[str] = None
    has_serious_attempts: Optional[str] = None
    n_serious_attempts: Optional[str] = None
    # Lethality (Columbia TS)
    most_serious_method: Optional[str] = None
    most_violent_method: Optional[str] = None
    most_serious_trigger: Optional[str] = None


class FamilyHistoryColumns(_Base):
    maternal_psychiatric: Optional[str] = None
    paternal_psychiatric: Optional[str] = None
    maternal_substance: Optional[str] = None
    paternal_substance: Optional[str] = None
    maternal_suicide: Optional[str] = None
    paternal_suicide: Optional[str] = None
    bipolar_keyword: str = "bipolaire"
    # Extended pedigree: list of (key, label_fr) pairs to iterate over with
    # cohort-specific column suffixes.
    relatives: list[dict[str, str]] = Field(default_factory=list)
    relative_suffixes: dict[str, str] = Field(default_factory=dict)
    # First-order siblings with their own column prefix (e.g. frere1, soeur1)
    siblings: list[dict[str, str]] = Field(default_factory=list)
    # Counts of siblings/children from structure columns
    brothers_count_col: Optional[str] = None
    sisters_count_col: Optional[str] = None
    brothers_affected_col: Optional[str] = None
    sisters_affected_col: Optional[str] = None
    sons_count_col: Optional[str] = None
    daughters_count_col: Optional[str] = None
    sons_affected_col: Optional[str] = None
    daughters_affected_col: Optional[str] = None


class HospitalizationColumns(_Base):
    n_lifetime: Optional[str] = None
    n_last_year: Optional[str] = None
    duration_last: Optional[str] = None
    er_visits_flag: Optional[str] = None
    n_er_visits: Optional[str] = None
    work_absences_flag: Optional[str] = None
    n_work_absences: Optional[str] = None


class CurrentEpisodeCriteriaColumns(_Base):
    # Dataclass field name -> CSV column
    depressive_items: dict[str, str] = Field(default_factory=dict)
    manic_items: dict[str, str] = Field(default_factory=dict)
    depressive_total: Optional[str] = None
    manic_total: Optional[str] = None


class MostRecentEpisodeColumns(_Base):
    episode_type: Optional[str] = None
    severity: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    chronicity: Optional[str] = None
    postpartum: Optional[str] = None


class PsychiatricHistoryColumns(_Base):
    model_config = ConfigDict(extra="allow")


class TreatmentsColumns(_Base):
    model_config = ConfigDict(extra="allow")


class LifetimeMedicationsColumns(_Base):
    # class_name -> {ever: col, duration: col}
    classes: dict[str, dict[str, str]] = Field(default_factory=dict)


class NonPharmTreatmentColumns(_Base):
    model_config = ConfigDict(extra="allow")


class CognitiveProfileColumns(_Base):
    model_config = ConfigDict(extra="allow")


class AdditionalNeuropsychColumns(_Base):
    model_config = ConfigDict(extra="allow")


class DivaColumns(_Base):
    model_config = ConfigDict(extra="allow")


class CircadianColumns(_Base):
    model_config = ConfigDict(extra="allow")


class ComorbidityItem(_Base):
    # Either {col: ..., label_fr: ...} for direct column
    # or {key: ..., label_fr: ...} for key + suffix template
    col: Optional[str] = None
    key: Optional[str] = None
    label_fr: str


class ComorbiditiesColumns(_Base):
    somatic_suffix: str = "_mhoccur"
    somatic: list[ComorbidityItem] = Field(default_factory=list)
    psychiatric: list[ComorbidityItem] = Field(default_factory=list)
    general_anxiety_flag: Optional[str] = None
    substance_use_flag: Optional[str] = None
    general_anxiety_label_fr: str = "Trouble anxieux comorbide (non spécifié)"
    substance_use_label_fr: str = "Trouble lié à l'usage de substances"


class V1FollowUpColumns(_Base):
    # instrument_key -> _n1 column
    instruments: dict[str, str] = Field(default_factory=dict)


class CohortColumnMap(_Base):
    """Complete CSV extraction layout for a single cohort.

    Only the patient_id column and demographics are mandatory. All other
    sections are optional so each cohort can declare only the fields it
    actually extracts (ASP has no treatments.lithium, SZ has no DIVA, etc.).
    """

    model_config = ConfigDict(extra="allow")

    schema_version: int = Field(alias="$schema_version")
    patient_id_column: str
    demographics: DemographicsColumns
    vitals: Optional[VitalsColumns] = None
    ecg: Optional[ECGColumns] = None
    substance_use: Optional[SubstanceUseColumns] = None
    suicide_indicators: Optional[SuicideIndicatorsColumns] = None
    suicide_history: Optional[SuicideHistoryColumns] = None
    family_history: Optional[FamilyHistoryColumns] = None
    hospitalization: Optional[HospitalizationColumns] = None
    current_episode_criteria: Optional[CurrentEpisodeCriteriaColumns] = None
    most_recent_episode: Optional[MostRecentEpisodeColumns] = None
    psychiatric_history: Optional[PsychiatricHistoryColumns] = None
    treatments: Optional[TreatmentsColumns] = None
    lifetime_medications: Optional[LifetimeMedicationsColumns] = None
    non_pharm_treatments: Optional[NonPharmTreatmentColumns] = None
    cognitive_profile: Optional[CognitiveProfileColumns] = None
    additional_neuropsych: Optional[AdditionalNeuropsychColumns] = None
    diva_adhd: Optional[DivaColumns] = None
    circadian: Optional[CircadianColumns] = None
    comorbidities: Optional[ComorbiditiesColumns] = None
    v1_followup: Optional[V1FollowUpColumns] = None
    # Cohort-specific sections land in the extra dict automatically
