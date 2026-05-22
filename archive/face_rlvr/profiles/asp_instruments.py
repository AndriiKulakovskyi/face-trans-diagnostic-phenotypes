"""ASP (TSASDI) cohort instrument registry (data-driven from YAML).

Instrument definitions live in:
  - ``config/glossary/common/instruments.yaml`` (shared instruments)
  - ``config/glossary/asp/instruments.yaml``    (ASP-specific + overrides)

References for thresholds:
- BDI-II: Beck et al. 1996
- RBS-R: Bodfish et al. 2000; Lam & Aman 2007
- WAIS-IV: Wechsler 2008
- BRIEF: Gioia et al. 2000
- AQ-24: Baron-Cohen et al. 2001 (short form)
- HAM-A: Hamilton 1959
- ADHD-RS: DuPaul et al. 1998
- LSAS: Liebowitz 1987
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

ASP_INSTRUMENTS: dict[str, InstrumentDefinition] = get_cohort_instruments("asp")

_groups = get_cohort_instrument_groups("asp")
ASP_DEPRESSION_INSTRUMENTS = _groups["ASP_DEPRESSION_INSTRUMENTS"]
ASP_GLOBAL_INSTRUMENTS = _groups["ASP_GLOBAL_INSTRUMENTS"]
ASP_FUNCTIONING_INSTRUMENTS = _groups["ASP_FUNCTIONING_INSTRUMENTS"]
ASP_REPETITIVE_INSTRUMENTS = _groups["ASP_REPETITIVE_INSTRUMENTS"]
ASP_COGNITIVE_INSTRUMENTS = _groups["ASP_COGNITIVE_INSTRUMENTS"]
ASP_EXECUTIVE_INSTRUMENTS = _groups["ASP_EXECUTIVE_INSTRUMENTS"]
ASP_AUTISM_SCREENING_INSTRUMENTS = _groups["ASP_AUTISM_SCREENING_INSTRUMENTS"]
ASP_ANXIETY_INSTRUMENTS = _groups["ASP_ANXIETY_INSTRUMENTS"]
ASP_ADHD_INSTRUMENTS = _groups["ASP_ADHD_INSTRUMENTS"]
ASP_TRAUMA_INSTRUMENTS = _groups["ASP_TRAUMA_INSTRUMENTS"]
ASP_SLEEP_INSTRUMENTS = _groups["ASP_SLEEP_INSTRUMENTS"]
ASP_ADHERENCE_INSTRUMENTS = _groups["ASP_ADHERENCE_INSTRUMENTS"]
ASP_SUBSTANCE_INSTRUMENTS = _groups["ASP_SUBSTANCE_INSTRUMENTS"]
