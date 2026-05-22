"""BP cohort instrument registry (data-driven from YAML).

Instrument definitions live in:
  - ``config/glossary/common/instruments.yaml`` (shared instruments)
  - ``config/glossary/bp/instruments.yaml``     (BP-specific + overrides)

To modify BP instruments (columns, thresholds, French labels), edit the YAML
files directly — no Python changes needed.

References for thresholds (bibliography preserved from pre-migration):
- MADRS: Montgomery & Åsberg 1979
- YMRS: Young et al. 1978
- CGI: Guy 1976 (ECDEU)
- FAST: Rosa et al. 2007
- BIS-10/11: Patton et al. 1995; Stanford et al. 2009
- STAI: Spielberger 1983; Gauthier & Bouchard 1993 (French norms)
- ASRM: Altman et al. 1997
- PSQI: Buysse et al. 1989
- ESS: Johns 1991
- COBRA: Rosa et al. 2013
- WURS: Ward et al. 1993
- AQ: Baron-Cohen et al. 2001
- MDQ: Hirschfeld et al. 2000
- EQ-5D: EuroQol Group 1990
- QIDS-SR16: Rush et al. 2003
- CTQ: Bernstein et al. 2003
- C-SSRS: Posner et al. 2011
- BDHI: Buss & Durkee 1957; Buss & Perry 1992
- ALDA: Alda 2002; Manchia et al. 2013
- ALS: Harvey et al. 1989
- AIM: Larsen & Diener 1987
- CSM: Smith et al. 1989
- Fagerström: Heatherton et al. 1991
- STOP-Bang: Chung et al. 2008
- SCIP: Purdon 2005
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

BP_INSTRUMENTS: dict[str, InstrumentDefinition] = get_cohort_instruments("bp")

_groups = get_cohort_instrument_groups("bp")
MOOD_INSTRUMENTS = _groups["MOOD_INSTRUMENTS"]
FUNCTIONAL_INSTRUMENTS = _groups["FUNCTIONAL_INSTRUMENTS"]
ANXIETY_IMPULSIVITY_INSTRUMENTS = _groups["ANXIETY_IMPULSIVITY_INSTRUMENTS"]
SLEEP_INSTRUMENTS = _groups["SLEEP_INSTRUMENTS"]
COGNITIVE_INSTRUMENTS = _groups["COGNITIVE_INSTRUMENTS"]
ADHERENCE_INSTRUMENTS = _groups["ADHERENCE_INSTRUMENTS"]
TRAUMA_INSTRUMENTS = _groups["TRAUMA_INSTRUMENTS"]
SUICIDE_INSTRUMENTS = _groups["SUICIDE_INSTRUMENTS"]
TREATMENT_RESPONSE_INSTRUMENTS = _groups["TREATMENT_RESPONSE_INSTRUMENTS"]
SUBSTANCE_INSTRUMENTS = _groups["SUBSTANCE_INSTRUMENTS"]
SCREENING_INSTRUMENTS = _groups["SCREENING_INSTRUMENTS"]
