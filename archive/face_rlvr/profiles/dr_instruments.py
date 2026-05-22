"""DR cohort instrument registry (data-driven from YAML).

Instrument definitions live in:
  - ``config/glossary/common/instruments.yaml`` (shared instruments)
  - ``config/glossary/dr/instruments.yaml``     (DR-specific + overrides)

References for thresholds:
- MADRS: Montgomery & Asberg 1979
- QIDS-SR16: Rush et al. 2003
- ERD: Widlocher 1983
- SHAPS: Snaith et al. 1995
- BAS: Tyrer et al. 1984
- SPIN: Connor et al. 2000
- LEAPS: Lam et al. 2009
- EGF/GAF: APA DSM-IV-TR
- PCL-5: Weathers et al. 2013
- Sachs: Sachs 2004 (treatment resistance staging)
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

DR_INSTRUMENTS: dict[str, InstrumentDefinition] = get_cohort_instruments("dr")

_groups = get_cohort_instrument_groups("dr")
DR_DEPRESSION_INSTRUMENTS = _groups["DR_DEPRESSION_INSTRUMENTS"]
DR_MOOD_INSTRUMENTS = _groups["DR_MOOD_INSTRUMENTS"]
DR_GLOBAL_INSTRUMENTS = _groups["DR_GLOBAL_INSTRUMENTS"]
DR_FUNCTIONING_INSTRUMENTS = _groups["DR_FUNCTIONING_INSTRUMENTS"]
DR_ANXIETY_INSTRUMENTS = _groups["DR_ANXIETY_INSTRUMENTS"]
DR_SLEEP_INSTRUMENTS = _groups["DR_SLEEP_INSTRUMENTS"]
DR_ADHERENCE_INSTRUMENTS = _groups["DR_ADHERENCE_INSTRUMENTS"]
DR_TRAUMA_INSTRUMENTS = _groups["DR_TRAUMA_INSTRUMENTS"]
DR_SUBSTANCE_INSTRUMENTS = _groups["DR_SUBSTANCE_INSTRUMENTS"]
DR_SELF_ESTEEM_INSTRUMENTS = _groups["DR_SELF_ESTEEM_INSTRUMENTS"]
DR_PERSONALITY_INSTRUMENTS = _groups["DR_PERSONALITY_INSTRUMENTS"]
DR_SCREENING_INSTRUMENTS = _groups["DR_SCREENING_INSTRUMENTS"]
DR_IMPULSIVITY_INSTRUMENTS = _groups["DR_IMPULSIVITY_INSTRUMENTS"]
