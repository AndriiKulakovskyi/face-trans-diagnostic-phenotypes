"""Patient profile extraction and clinical interpretation for FACE cohorts.

This module replaces raw questionnaire item extraction with score-based
extraction + clinically validated interpretations for building rich
patient vignettes.

Supports:
- BP (Bipolar Disorder) cohort
- SZ (Schizophrenia) cohort
- DR (Treatment-Resistant Depression) cohort
- ASP (Autism Spectrum Disorder / TSASDI) cohort
"""

from face_rlvr.profiles.common_instruments import (
    InstrumentDefinition,
    ScoreInterpretation,
    SeverityLevel,
    interpret_score,
)
from face_rlvr.profiles.common_extractors import (
    compute_bmi_category,
    detect_metabolic_syndrome,
    compute_framingham_risk,
    check_medication_lab_alerts,
    check_drug_interactions,
    compute_rci,
    detect_floor_ceiling_effects,
    compute_data_completeness,
    compute_cognitive_z_score,
)
from face_rlvr.profiles.bp_instruments import BP_INSTRUMENTS
from face_rlvr.profiles.bp_extractor import BPPatientData, extract_bp_patient
from face_rlvr.profiles.bp_profile_builder import PatientProfile, build_bp_profile

from face_rlvr.profiles.sz_instruments import SZ_INSTRUMENTS
from face_rlvr.profiles.sz_extractor import SZPatientData, extract_sz_patient
from face_rlvr.profiles.sz_profile_builder import build_sz_profile

from face_rlvr.profiles.dr_instruments import DR_INSTRUMENTS
from face_rlvr.profiles.dr_extractor import DRPatientData, extract_dr_patient
from face_rlvr.profiles.dr_profile_builder import build_dr_profile

from face_rlvr.profiles.asp_instruments import ASP_INSTRUMENTS
from face_rlvr.profiles.asp_extractor import ASPPatientData, extract_asp_patient
from face_rlvr.profiles.asp_profile_builder import build_asp_profile

__all__ = [
    # Common infrastructure
    "InstrumentDefinition",
    "PatientProfile",
    "ScoreInterpretation",
    "SeverityLevel",
    "interpret_score",
    # Clinical utilities
    "compute_bmi_category",
    "detect_metabolic_syndrome",
    "compute_framingham_risk",
    "check_medication_lab_alerts",
    "check_drug_interactions",
    "compute_rci",
    "detect_floor_ceiling_effects",
    "compute_data_completeness",
    "compute_cognitive_z_score",
    # BP
    "BP_INSTRUMENTS",
    "BPPatientData",
    "build_bp_profile",
    "extract_bp_patient",
    # SZ
    "SZ_INSTRUMENTS",
    "SZPatientData",
    "build_sz_profile",
    "extract_sz_patient",
    # DR
    "DR_INSTRUMENTS",
    "DRPatientData",
    "build_dr_profile",
    "extract_dr_patient",
    # ASP
    "ASP_INSTRUMENTS",
    "ASPPatientData",
    "build_asp_profile",
    "extract_asp_patient",
]
