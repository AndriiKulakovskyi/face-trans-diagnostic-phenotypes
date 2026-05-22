"""YAML glossary loader with caching and dataclass conversion.

The loader's job is to:
  1. Read YAML files from config/glossary/
  2. Validate them via Pydantic v2
  3. Cache results in module-level lru_caches
  4. Convert to the pre-existing dataclasses (SeverityLevel, InstrumentDefinition)
     so extractors/profile builders don't have to change their type hints.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from face_rlvr.profiles.common_instruments import InstrumentDefinition, SeverityLevel
from face_rlvr.profiles.glossary_schema import (
    CategoricalCodesFileConfig,
    ClinicalConstantsConfig,
    CohortColumnMap,
    InstrumentConfig,
    InstrumentsFileConfig,
    LabRangesFileConfig,
    SeverityBandConfig,
    ThresholdsFileConfig,
)


# ─── Path resolution ─────────────────────────────────────────────────────────


def _glossary_root() -> Path:
    """Resolve the config/glossary directory (source tree or installed wheel)."""
    # Try installed package data first (when wheel is installed)
    try:
        from importlib.resources import files
        pkg_path = Path(str(files("face_rlvr") / "_glossary_data"))
        if pkg_path.is_dir():
            return pkg_path
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass

    # Fallback: walk up from this file to find config/glossary at repo root
    here = Path(__file__).resolve()
    for ancestor in [here.parent, *here.parents]:
        candidate = ancestor / "config" / "glossary"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "config/glossary not found (neither package data nor source tree)"
    )


def _read_yaml(relative_path: str) -> Any:
    path = _glossary_root() / relative_path
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ─── Pydantic-level loaders (cached) ─────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_common_thresholds() -> ThresholdsFileConfig:
    return ThresholdsFileConfig.model_validate(_read_yaml("common/thresholds.yaml"))


@lru_cache(maxsize=1)
def _load_common_clinical_constants() -> ClinicalConstantsConfig:
    return ClinicalConstantsConfig.model_validate(_read_yaml("common/clinical_constants.yaml"))


@lru_cache(maxsize=1)
def _load_common_categorical_codes() -> CategoricalCodesFileConfig:
    return CategoricalCodesFileConfig.model_validate(_read_yaml("common/categorical_codes.yaml"))


@lru_cache(maxsize=1)
def _load_common_instruments_raw() -> InstrumentsFileConfig:
    return InstrumentsFileConfig.model_validate(_read_yaml("common/instruments.yaml"))


@lru_cache(maxsize=8)
def _load_cohort_instruments_raw(cohort: str) -> InstrumentsFileConfig:
    return InstrumentsFileConfig.model_validate(_read_yaml(f"{cohort}/instruments.yaml"))


@lru_cache(maxsize=8)
def _load_cohort_lab_ranges_raw(cohort: str) -> LabRangesFileConfig:
    return LabRangesFileConfig.model_validate(_read_yaml(f"{cohort}/lab_ranges.yaml"))


@lru_cache(maxsize=8)
def _load_cohort_categorical_codes(cohort: str) -> CategoricalCodesFileConfig | None:
    path = _glossary_root() / cohort / "categorical_codes.yaml"
    if not path.exists():
        return None
    return CategoricalCodesFileConfig.model_validate(_read_yaml(f"{cohort}/categorical_codes.yaml"))


@lru_cache(maxsize=8)
def _load_cohort_column_map(cohort: str) -> CohortColumnMap:
    """Load the CSV column map for a cohort (which CSV column holds which field)."""
    return CohortColumnMap.model_validate(_read_yaml(f"{cohort}/column_map.yaml"))


def get_cohort_column_map(cohort: str) -> CohortColumnMap:
    """Public accessor for a cohort's CSV column layout (from YAML)."""
    return _load_cohort_column_map(cohort)


def _invalidate_cache() -> None:
    """Clear all LRU caches. Used by tests that modify YAML at runtime."""
    for func in (
        _load_common_thresholds,
        _load_common_clinical_constants,
        _load_common_categorical_codes,
        _load_common_instruments_raw,
        _load_cohort_instruments_raw,
        _load_cohort_lab_ranges_raw,
        _load_cohort_categorical_codes,
        _load_cohort_column_map,
    ):
        func.cache_clear()


# ─── Conversion to dataclasses ───────────────────────────────────────────────


def _band_to_dataclass(b: SeverityBandConfig) -> SeverityLevel:
    return SeverityLevel(
        min_score=b.min_score,
        max_score=b.max_score,
        code=b.code,
        label_fr=b.label_fr,
        clinical_meaning_fr=b.clinical_meaning_fr,
    )


def _resolve_thresholds(cfg: InstrumentConfig) -> list[SeverityLevel]:
    if cfg.severity_thresholds:
        return [_band_to_dataclass(b) for b in cfg.severity_thresholds]
    if cfg.severity_thresholds_ref:
        shared = _load_common_thresholds().bands.get(cfg.severity_thresholds_ref)
        if shared is None:
            raise KeyError(
                f"{cfg.name}: severity_thresholds_ref '{cfg.severity_thresholds_ref}' "
                f"not found in common/thresholds.yaml"
            )
        return [_band_to_dataclass(b) for b in shared.bands]
    return []


def _cfg_to_instrument(cfg: InstrumentConfig) -> InstrumentDefinition:
    """Convert a pydantic InstrumentConfig into the existing InstrumentDefinition dataclass."""
    return InstrumentDefinition(
        name=cfg.name,
        full_name=cfg.full_name,
        full_name_fr=cfg.full_name_fr,
        domain=cfg.domain,
        total_column=cfg.total_column,
        subscale_columns=dict(cfg.subscale_columns),
        score_range=tuple(cfg.score_range),
        higher_is_worse=cfg.higher_is_worse,
        severity_thresholds=_resolve_thresholds(cfg),
        screening_threshold=cfg.screening_threshold,
        screening_positive_label_fr=cfg.screening_positive_label_fr,
        screening_negative_label_fr=cfg.screening_negative_label_fr,
        evaluation_type=cfg.evaluation_type,
        unit=cfg.unit,
        clinical_note_fr=cfg.clinical_note_fr,
    )


def _expand_codes(file_cfg: CategoricalCodesFileConfig) -> dict[str, dict[str, str]]:
    """Turn {"1": "foo"} into {"1": "foo", "1.0": "foo"} matching current Python behavior."""
    out: dict[str, dict[str, str]] = {}
    for dict_name, mapping in file_cfg.codes.items():
        expanded: dict[str, str] = {}
        for k, v in mapping.items():
            expanded[k] = v
            if "." not in k:
                expanded[f"{k}.0"] = v
        out[dict_name] = expanded
    return out


# ─── Public API ──────────────────────────────────────────────────────────────


def load_common_glossary() -> dict[str, Any]:
    """Return the common glossary (shared instruments + thresholds + codes + constants).

    Each component is loaded lazily: if its YAML file is missing, the key is
    absent from the returned dict instead of raising. This supports incremental
    migration where YAML files are added one at a time.
    """
    common_raw = _load_common_instruments_raw()
    result: dict[str, Any] = {
        "instruments": {k: _cfg_to_instrument(v) for k, v in common_raw.instruments.items()},
        "thresholds": _load_common_thresholds(),
    }
    # Optional components (may not exist yet during migration)
    cc_path = _glossary_root() / "common" / "clinical_constants.yaml"
    if cc_path.exists():
        result["clinical_constants"] = _load_common_clinical_constants()
    codes_path = _glossary_root() / "common" / "categorical_codes.yaml"
    if codes_path.exists():
        result["categorical_codes"] = _expand_codes(_load_common_categorical_codes())
    else:
        result["categorical_codes"] = {}
    return result


def load_cohort_glossary(cohort: str) -> dict[str, Any]:
    """Return merged glossary for a cohort: common instruments + cohort overrides + labs + codes.

    Lab ranges and categorical codes are loaded lazily; missing files are reported
    as empty/None to support incremental migration.
    """
    if cohort == "common":
        return load_common_glossary()

    cohort_raw = _load_cohort_instruments_raw(cohort)
    common_instruments = load_common_glossary()["instruments"]

    # Overlay cohort-specific overrides on top of common
    merged: dict[str, InstrumentDefinition] = dict(common_instruments)
    for key, cfg in cohort_raw.instruments.items():
        merged[key] = _cfg_to_instrument(cfg)

    # If the cohort file has a $registry block, restrict/reorder to its order
    registry = cohort_raw.registry
    registry_groups: dict[str, list[str]] = {}
    if registry is not None:
        merged = {k: merged[k] for k in registry.order if k in merged}
        registry_groups = registry.groups

    # Optional components (may not exist yet during migration)
    lab_path = _glossary_root() / cohort / "lab_ranges.yaml"
    lab_ranges = _load_cohort_lab_ranges_raw(cohort).labs if lab_path.exists() else []

    cat_codes = _load_cohort_categorical_codes(cohort)

    return {
        "instruments": merged,
        "registry_groups": registry_groups,
        "lab_ranges": lab_ranges,
        "categorical_codes": _expand_codes(cat_codes) if cat_codes else {},
    }


def get_cohort_instruments(cohort: str) -> dict[str, InstrumentDefinition]:
    return load_cohort_glossary(cohort)["instruments"]


def get_cohort_lab_ranges(cohort: str):
    return load_cohort_glossary(cohort)["lab_ranges"]


def get_cohort_instrument_groups(cohort: str) -> dict[str, list[str]]:
    return load_cohort_glossary(cohort)["registry_groups"]


def get_cohort_categorical_codes(cohort: str) -> dict[str, dict[str, str]]:
    return load_cohort_glossary(cohort)["categorical_codes"]


def get_clinical_constants() -> ClinicalConstantsConfig:
    return _load_common_clinical_constants()


def get_common_categorical_codes() -> dict[str, dict[str, str]]:
    return load_common_glossary()["categorical_codes"]
