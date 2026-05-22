"""SZ cohort instrument registry (data-driven from YAML).

Instrument definitions live in:
  - ``config/glossary/common/instruments.yaml`` (shared instruments)
  - ``config/glossary/sz/instruments.yaml``     (SZ-specific + overrides)

References for thresholds:
- PANSS: Kay et al. 1987
- Calgary: Addington et al. 1990
- PSP: Morosini et al. 2000
- AIMS: Guy 1976
- BARS: Barnes 1989
- S-QoL: Boyer et al. 2010
"""

from __future__ import annotations

from face_rlvr.profiles.common_instruments import InstrumentDefinition
from face_rlvr.profiles.glossary_loader import (
    get_cohort_instrument_groups,
    get_cohort_instruments,
)


# ═════════════════════════════════════════════════════════════════════════════
# REGISTRY (loaded from YAML)
# ═════════════════════════════════════════════════════════════════════════════

SZ_INSTRUMENTS: dict[str, InstrumentDefinition] = get_cohort_instruments("sz")

_groups = get_cohort_instrument_groups("sz")
SZ_PSYCHOSIS_INSTRUMENTS = _groups["SZ_PSYCHOSIS_INSTRUMENTS"]
SZ_DEPRESSION_INSTRUMENTS = _groups["SZ_DEPRESSION_INSTRUMENTS"]
SZ_GLOBAL_INSTRUMENTS = _groups["SZ_GLOBAL_INSTRUMENTS"]
SZ_FUNCTIONING_INSTRUMENTS = _groups["SZ_FUNCTIONING_INSTRUMENTS"]
SZ_MOVEMENT_INSTRUMENTS = _groups["SZ_MOVEMENT_INSTRUMENTS"]
SZ_MOOD_INSTRUMENTS = _groups["SZ_MOOD_INSTRUMENTS"]
SZ_SLEEP_INSTRUMENTS = _groups["SZ_SLEEP_INSTRUMENTS"]
SZ_ADHERENCE_INSTRUMENTS = _groups["SZ_ADHERENCE_INSTRUMENTS"]
SZ_TRAUMA_INSTRUMENTS = _groups["SZ_TRAUMA_INSTRUMENTS"]
SZ_SUBSTANCE_INSTRUMENTS = _groups["SZ_SUBSTANCE_INSTRUMENTS"]
SZ_SCREENING_INSTRUMENTS = _groups["SZ_SCREENING_INSTRUMENTS"]
