from __future__ import annotations

import re
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

Transformer = Callable[[pd.Series, str], pd.Series]

RULES: dict[str, Transformer] = {}


def register(canonical_name: str):
    def decorator(fn: Transformer) -> Transformer:
        RULES[canonical_name] = fn
        return fn
    return decorator


_BINARY_MAP = {
    1: 1, 0: 0,
    "1": 1, "0": 0,
    1.0: 1, 0.0: 0,
    "Oui": 1, "oui": 1, "OUI": 1,
    "Non": 0, "non": 0, "NON": 0,
    "Yes": 1, "yes": 1, "YES": 1,
    "No": 0, "no": 0, "NO": 0,
    "Y": 1, "y": 1,
    "N": 0, "n": 0,
    "True": 1, "False": 0, True: 1, False: 0,
}

_VALUE_SET_RE = re.compile(r"\{([^}]+)\}")


def _parse_allowed_values(unit_or_value_set: str) -> set | None:
    if not unit_or_value_set:
        return None
    match = _VALUE_SET_RE.search(unit_or_value_set)
    if not match:
        return None
    tokens = [t.strip() for t in match.group(1).split(",")]
    allowed: set = set()
    for tok in tokens:
        first = tok.split("=", 1)[0].strip()
        if not first or first.upper() in {"NA", "NA=UNKNOWN", "UNKNOWN"}:
            continue
        try:
            allowed.add(int(first))
        except ValueError:
            allowed.add(first)
    return allowed or None


def _to_nullable_int(series: pd.Series, np_dtype: np.dtype, pd_dtype: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mask = numeric.isna().to_numpy()
    filled = numeric.fillna(0).round().to_numpy().astype(np_dtype)
    arr = pd.arrays.IntegerArray(filled, mask)
    return pd.Series(arr, index=series.index, name=series.name).astype(pd_dtype)


def _make_int_series(values, index, name, pd_dtype: str = "Int16") -> pd.Series:
    """Build a nullable-int Series from an iterable of int / NA values."""
    np_dtype = np.int8 if pd_dtype == "Int8" else np.int16
    out: list[int] = []
    mask: list[bool] = []
    for v in values:
        if v is pd.NA or v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(0); mask.append(True)
        else:
            out.append(int(v)); mask.append(False)
    arr = pd.arrays.IntegerArray(np.asarray(out, dtype=np_dtype), np.asarray(mask))
    return pd.Series(arr, index=index, name=name).astype(pd_dtype)


def _textual_recode(
    series: pd.Series,
    text_map: dict,
    pd_dtype: str = "Int16",
    extra_remap: dict | None = None,
) -> pd.Series:
    """Apply a text→int map. Values that are already numeric pass through
    `pd.to_numeric` (with optional per-cohort `extra_remap` applied after).
    Unmapped text becomes <NA>.
    """
    def convert(x):
        if x is pd.NA or x is None:
            return pd.NA
        if isinstance(x, float) and np.isnan(x):
            return pd.NA
        if isinstance(x, str):
            if x in text_map:
                m = text_map[x]
                return pd.NA if m is pd.NA else int(m)
            stripped = x.strip()
            if stripped in text_map:
                m = text_map[stripped]
                return pd.NA if m is pd.NA else int(m)
            try:
                v = int(float(stripped))
            except (TypeError, ValueError):
                return pd.NA
            if extra_remap and v in extra_remap:
                m = extra_remap[v]
                return pd.NA if m is pd.NA else int(m)
            return v
        try:
            v = int(x)
        except (TypeError, ValueError):
            try:
                v = int(float(x))
            except (TypeError, ValueError):
                return pd.NA
        if extra_remap and v in extra_remap:
            m = extra_remap[v]
            return pd.NA if m is pd.NA else int(m)
        return v
    return _make_int_series((convert(x) for x in series),
                             series.index, series.name, pd_dtype)


def identity_cast(
    series: pd.Series,
    cohort: str,
    dtype: str,
    canonical_name: str = "",
    unit_or_value_set: str = "",
) -> pd.Series:
    dtype_norm = (dtype or "").strip().lower()

    if dtype_norm == "int8 binary":
        mapped = series.map(_BINARY_MAP)
        out = pd.array(mapped, dtype="Int8")
        result = pd.Series(out, index=series.index, name=series.name)

    elif dtype_norm in {"int8 ordinal", "int8 categorical"}:
        result = _to_nullable_int(series, np.int16, "Int16")

    elif dtype_norm == "float":
        result = pd.to_numeric(series, errors="coerce").astype("float64")

    elif dtype_norm.startswith("date"):
        result = pd.to_datetime(series, errors="coerce")

    elif dtype_norm == "category":
        result = series.astype("string").astype("category")

    elif dtype_norm == "string":
        result = series.astype("string")

    else:
        result = series.astype("string")

    allowed = _parse_allowed_values(unit_or_value_set)
    if allowed is not None and dtype_norm in {"int8 binary", "int8 ordinal", "int8 categorical"}:
        non_null = result.dropna()
        unexpected = set(non_null.unique()) - allowed
        if unexpected:
            warnings.warn(
                f"[{cohort}] {canonical_name!r}: identity_cast produced values "
                f"outside {unit_or_value_set}: {sorted(unexpected)[:6]}",
                stacklevel=2,
            )
    return result


# ============================================================================
# Registered transformers
# ============================================================================
# Each entry below covers a variable where the cohorts disagree on encoding —
# usually BP/SZ store French text labels while DR stores integer codes. Maps
# are aligned to the Codage column (col F) and Rule column (col I) of the
# dictionary face-common-vars.xlsx.
# ----------------------------------------------------------------------------


@register("sex")
def _sex(series: pd.Series, cohort: str) -> pd.Series:
    # Rule: {0=Masculin, 1=Feminin}.
    text_map = {
        "Masculin": 0, "masculin": 0, "M": 0, "m": 0,
        "Feminin": 1, "feminin": 1, "Féminin": 1, "féminin": 1, "F": 1, "f": 1,
    }
    return _textual_recode(series, text_map, pd_dtype="Int8")


@register("ppartpremier_episode")
def _ppartpremier_episode(series: pd.Series, cohort: str) -> pd.Series:
    # R425: BP uses Oui/Non text; DR uses 1=Oui, 2=Non (invert), 3=NA.
    if cohort == "DR":
        return _textual_recode(series, text_map={}, pd_dtype="Int8",
                                extra_remap={1: 1, 2: 0, 3: pd.NA})
    return _textual_recode(series, text_map={"Oui": 1, "Non": 0,
                                              "Ne sais pas": pd.NA, "NSP": pd.NA,
                                              "Y": 1, "N": 0},
                            pd_dtype="Int8")


_FONDACODE_NETWORK_RE = re.compile(r"^(\d{1,2})(?=\d{8}$)")


def _network_site_code(fondacode: pd.Series) -> pd.Series:
    """Canonical FACE network site code = the 1–2 leading digits of fondacode.

    fondacode is a 9–10 digit FONDAMENTAL patient ID whose prefix is the network
    site number, consistent across BP/SZ/DR (verified: it agrees with the raw
    per-cohort SITEID 99.2 / 99.9 / 100 % of the time, and resolves the few
    mislabelled SITEID rows, e.g. SZ siteid=16 whose fondacode says 19=Toulouse).
    """
    text = fondacode.astype("string")
    extracted = text.str.extract(_FONDACODE_NETWORK_RE, expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def derive_siteid_city(
    siteid: pd.Series, fondacode: pd.Series, cohort: str
) -> pd.Series:
    """Canonical site code for the ``siteid_city`` feature.

    Prefer the fondacode-derived network code (uniform across cohorts); fall back
    to the raw per-cohort SITEID only where fondacode is missing/malformed. The
    per-cohort SITEID codebooks are partial, disjoint and occasionally
    mislabelled, so they are never the primary source. Output is a float site
    code; the integer→city name join (data/site_lookup.csv) is applied downstream
    where a human-readable label is needed (it is a pure relabelling of the code).
    """
    net = _network_site_code(fondacode)
    raw = pd.to_numeric(siteid, errors="coerce")
    return net.fillna(raw).astype("float64").rename(siteid.name)


@register("siteid_city")
def _siteid_city(series: pd.Series, cohort: str) -> pd.Series:
    # Fallback path: when only the SITEID series is available (no sibling
    # fondacode), the loader special-cases siteid_city via derive_siteid_city
    # before this rule is reached. This body covers the degenerate single-column
    # call so the transformer contract still holds.
    return pd.to_numeric(series, errors="coerce").astype("float64")


# ----- CONSTANTES ET ECG (intervals: milliseconds → seconds) ----------------
# qt / rr / qtc mix seconds and milliseconds within the same column across
# cohorts (BP ~3 % ms, DR ~9 %). Canonical output is SECONDS: any value > 5 is a
# millisecond entry and is divided by 1000 (a true ECG interval in seconds is
# well under 5; in milliseconds it is in the hundreds). Sanity bounds in the
# dictionary are then expressed in seconds and applied after this normalization.

def _ms_to_seconds(series: pd.Series, cohort: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    converted = numeric.where(~(numeric > 5), numeric / 1000.0)
    return converted.astype("float64")


@register("qt")
def _qt(series, cohort):
    return _ms_to_seconds(series, cohort)

@register("rr")
def _rr(series, cohort):
    return _ms_to_seconds(series, cohort)

@register("qtc")
def _qtc(series, cohort):
    return _ms_to_seconds(series, cohort)


# ----- SUICIDE (count fields with inequality / ceiling tokens) ---------------
# Count items such as ISF07 ("how many suicide attempts?") are occasionally
# recorded as a ceiling category, e.g. '>10' (SZ), or with a decimal comma. Plain
# pd.to_numeric drops those to NaN — losing real data in a field this preprocessing
# step then tries to recover (see skip_logic.py). We strip a leading inequality
# sign and normalise the decimal mark, so '>10' -> 10.0, '3,5' -> 3.5; genuine
# non-numeric junk still becomes NaN. Sanity bounds are applied afterwards.

def _count_with_inequalities(series: pd.Series, cohort: str) -> pd.Series:
    def convert(x):
        if x is None or x is pd.NA:
            return np.nan
        if isinstance(x, int | float):
            return float(x)
        s = str(x).strip().replace(",", ".").lstrip("><=≥≤ ").strip()
        try:
            return float(s)
        except ValueError:
            return np.nan
    return series.map(convert).astype("float64")


@register("isf07")
def _isf07(series, cohort):
    # "Combien de fois avez-vous tenté de vous suicider?" — count; '>10' in SZ.
    return _count_with_inequalities(series, cohort)


@register("isf09a")
def _isf09a(series, cohort):
    # "Nombre de tentatives de suicide graves" — count.
    return _count_with_inequalities(series, cohort)


# ----- BILAN BIOLOGIQUE (hematology: within-column unit mixing) --------------
# MCHC and HCT mix scales within the same column (data-entry / lab-system unit).
# MCHC: mostly g/dL (~30-37) but ~6 % BP / ~30 % DR recorded in g/L (~300-370 = x10)
#   -> any value > 100 is g/L and is divided by 10. Canonical output: g/dL.
# HCT: mostly % (~35-50) but ~4-20 % recorded in L/L (~0.35-0.50 = /100) -> any value
#   < 1 is L/L and is multiplied by 100. Canonical output: %. Gross sentinels (e.g.
#   3704) are left for the dictionary sanity bound to null. Bounds applied after this.

def _mchc_to_gdl(series, cohort):
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(~(numeric > 100), numeric / 10.0).astype("float64")

@register("mchc_lbstresc")
def _mchc(series, cohort):
    return _mchc_to_gdl(series, cohort)

def _hct_to_pct(series, cohort):
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(~(numeric < 1), numeric * 100.0).astype("float64")

@register("hct_lbstresc")
def _hct(series, cohort):
    return _hct_to_pct(series, cohort)


# ----- SOCIAL (education: SZ stores grade labels as text) --------------------
# BP/DR: clean ordinal (CP-CM2=1-5, collège=6-9, lycée=10-12, CAP=13, BEP=14,
# BAC+1..BAC+5=15-19, Doctorat=20). SZ: ~22 % stored as text tokens. Map the
# unambiguous tokens to the same ordinal; anything else → NA (logged-as-missing).
_EDULEVEL_TEXT_MAP = {
    "BAC": 12, "Bac": 12, "bac": 12,
    "CAP": 13, "BEP": 14,
    "Doctorat": 20, "Brevet": 9,
}


@register("edulevel")
def _edulevel(series, cohort):
    def convert(x):
        if x is pd.NA or x is None or (isinstance(x, float) and np.isnan(x)):
            return float("nan")
        if isinstance(x, str):
            s = x.strip()
            if s in _EDULEVEL_TEXT_MAP:
                return float(_EDULEVEL_TEXT_MAP[s])
            try:
                return float(s.replace(",", "."))
            except ValueError:
                return float("nan")
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")
    return pd.Series([convert(x) for x in series], index=series.index,
                     name=series.name, dtype="float64")


# ----- PERINATALITE ---------------------------------------------------------

@register("prembrth")
def _prembrth(series: pd.Series, cohort: str) -> pd.Series:
    # Codage: 1=prématurité, 2=à terme, 3=post-maturité, 0=Ne sais pas (→NA).
    text_map = {
        "Né à terme": 2, "Prématurité": 1, "Post-maturité": 3,
        "Ne sais pas": pd.NA,
    }
    return _textual_recode(series, text_map, pd_dtype="Int16",
                           extra_remap={0: pd.NA})


@register("naisstyp")
def _naisstyp(series: pd.Series, cohort: str) -> pd.Series:
    # Rule (col I): {0=Voie basse, 1=Césarienne, NA=Ne sais pas}.
    # DR raw codage uses 1=voie basse, 2=césarienne, 0=NSP — remap to 0/1.
    text_map = {
        "Voie basse": 0, "Césarienne": 1, "Ne sais pas": pd.NA,
    }
    return _textual_recode(series, text_map, pd_dtype="Int8",
                           extra_remap={0: pd.NA, 1: 0, 2: 1})


# ----- SUICIDE & SUICIDE-TIMING (ISF / LTSV / LTSG) -------------------------

_SUICIDE_TIMING_MAP = {
    "La semaine dernière": 0,
    "Il y a entre une semaine et la dernière visite": 0,
    "Il y a entre deux semaines et douze mois": 1,
    "Il y a plus d'un an": 2,
    "Il y a entre un et cinq ans": 2,
    "Il y a plus de cinq ans": 3,
}

for _canon in ("isf01a", "isf02a", "isf03a", "isf04a", "isf05a"):
    @register(_canon)
    def _isf_timing(series: pd.Series, cohort: str,
                    _m=dict(_SUICIDE_TIMING_MAP)) -> pd.Series:
        return _textual_recode(series, _m, pd_dtype="Int16")


# ----- ANTECEDENTS family-history -------------------------------------------

_PARENT_TROUBLE_MAP = {
    "Aucun": 0,
    "EDM ou Unipolaire": 1,
    "Bipolaire": 2,
    "Schizophrène": 3,
    "U": pd.NA, "Ne sais pas": pd.NA, "Inconnu": pd.NA,
}

_PARENT_SUICIDE_MAP = {
    "Aucun": 0,
    "Tentative de suicide": 1,
    "Suicide abouti": 2,
    "U": pd.NA, "Ne sais pas": pd.NA, "Inconnu": pd.NA,
}

@register("mere_trouble")
def _mere_trouble(series, cohort):
    return _textual_recode(series, _PARENT_TROUBLE_MAP, pd_dtype="Int16")

@register("pere_trouble")
def _pere_trouble(series, cohort):
    return _textual_recode(series, _PARENT_TROUBLE_MAP, pd_dtype="Int16")

@register("mere_suicide")
def _mere_suicide(series, cohort):
    return _textual_recode(series, _PARENT_SUICIDE_MAP, pd_dtype="Int16")

@register("pere_suicide")
def _pere_suicide(series, cohort):
    return _textual_recode(series, _PARENT_SUICIDE_MAP, pd_dtype="Int16")


# ----- ANTECEDENTS medical history flags ------------------------------------

@register("dysthyro_mhmodify")
def _dysthyro(series, cohort):
    # Subtype of thyroid disorder when present.
    return _textual_recode(series, {
        "Hypo-thyroïdie": 1, "Hyper-thyroïdie": 2, "Ne sais pas": pd.NA,
    }, pd_dtype="Int8")


@register("inflachro_mhmodify")
def _inflachro(series, cohort):
    return _textual_recode(series, {
        "Maladie de Crohn": 1, "Rectocolite hémorragique": 2,
        "Ne sais pas": pd.NA,
    }, pd_dtype="Int8")


@register("lupus_mhoccur")
def _lupus(series, cohort):
    # BP/DR use 0/1 numeric; SZ uses Y/N/Nc.
    return _textual_recode(series, {
        "Y": 1, "N": 0, "Nc": pd.NA, "U": pd.NA,
    }, pd_dtype="Int8")


def _antecedent_presence(series: pd.Series, cohort: str) -> pd.Series:
    """For variables where SZ stores the condition's label as a text presence
    flag (any non-null = condition present) while BP/DR store 0/1."""
    if cohort == "SZ":
        out: list[int] = []
        mask: list[bool] = []
        for x in series:
            if x is pd.NA or x is None or (isinstance(x, float) and np.isnan(x)):
                out.append(0); mask.append(True)
            else:
                out.append(1); mask.append(False)
        arr = pd.arrays.IntegerArray(np.asarray(out, dtype=np.int8), np.asarray(mask))
        return pd.Series(arr, index=series.index, name=series.name).astype("Int8")
    return _textual_recode(series, {}, pd_dtype="Int8")


@register("asthme_mhoccur")
def _asthme(series, cohort):
    return _antecedent_presence(series, cohort)

@register("allergie_mhoccur")
def _allergie(series, cohort):
    return _antecedent_presence(series, cohort)


# ----- ANTECEDENTS misc ----------------------------------------------------

@register("hvc_mhmodify")
def _hvc(series, cohort):
    # Dictionary dtype = 'category'. Pool verbatim text labels.
    return series.astype("string").astype("category")


# ----- AUTO-QUESTIONNAIRE (MDQ) --------------------------------------------

@register("mdq")
def _mdq(series, cohort):
    return _textual_recode(series, {
        "Positif": 1, "Négatif": 0, "Negatif": 0,
    }, pd_dtype="Int8")


# ----- NEUROPSYCHOLOGIE / CCLIN --------------------------------------------

@register("cclin04")
def _cclin04(series, cohort):
    return _textual_recode(series, {
        "Moins d'une semaine": 0, "Plus d'une semaine": 1,
    }, pd_dtype="Int8")


# ----- SUBSTANCES ----------------------------------------------------------

@register("suncf_cigarettes_lt")
def _suncf(series, cohort):
    # DR codage: 1=Non fumeur, 2=Ex-fumeur, 3=Fumeur actuel.
    return _textual_recode(series, {
        "Non fumeur": 1, "Ex-fumeur": 2, "Fumeur actuel": 3,
        "Statut inconnu": pd.NA, "Inconnu": pd.NA,
    }, pd_dtype="Int8")


# ----- SOCIAL --------------------------------------------------------------

@register("lvsbjind")
def _lvsbjind(series, cohort):
    # 1-8 ordinal per dictionary codage. SZ stores text labels.
    return _textual_recode(series, {
        "Seul": 1,
        "Chez ses parents": 2,
        "Dans son propre foyer familial": 3,
        "Chez ses enfants": 4,
        "Chez de la famille": 5,
        "Colocation": 6,
        "Collectivité": 7,
        "Autre(s)": 8, "Autres": 8, "Autre": 8,
    }, pd_dtype="Int8")


_JOBDUR_RANGE_RE = re.compile(r"entre\s+(\d+)\s+et\s+(\d+)\s+an", re.IGNORECASE)
_JOBDUR_NUM_RE = re.compile(r"(-?\d+(?:[.,]\d+)?)")


def _parse_jobdur_text(s: str) -> float:
    s = s.strip().lower()
    if not s:
        return float("nan")
    if "<1" in s or "-1" in s or "moins" in s:
        return 0.5
    m = _JOBDUR_RANGE_RE.search(s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (a + b) / 2
    m = _JOBDUR_NUM_RE.search(s)
    if m:
        return float(m.group(1).replace(",", "."))
    return float("nan")


@register("jobdur")
def _jobdur(series, cohort):
    # Output unit = years. BP/SZ: text ('5 ans', 'entre 10 et 20 ans', '<1 an').
    # DR: integer months → years.
    if cohort == "DR":
        return pd.to_numeric(series, errors="coerce").astype("float64") / 12.0
    parsed = []
    for x in series:
        if x is pd.NA or x is None:
            parsed.append(float("nan"))
        elif isinstance(x, str):
            parsed.append(_parse_jobdur_text(x))
        else:
            try:
                parsed.append(float(x))
            except (TypeError, ValueError):
                parsed.append(float("nan"))
    return pd.Series(parsed, index=series.index, name=series.name, dtype="float64")


# ----- EVALUATION MEDICALE -------------------------------------------------

_CETERM_MAP = {
    "Hypomanie": 1,
    "Manie sans caractéristiques psychotiques": 2,
    "Manie avec caractéristiques psychotiques": 3,
    "Episode Dépressif Majeur sans caractéristiques psychotiques": 4,
    "Episode Dépressif Majeur avec caractéristiques psychotiques": 5,
    "Mixte sans caractéristiques psychotiques": 6,
    "Mixte avec caractéristiques psychotiques": 7,
    "Inconnu": pd.NA, "Ne sais pas": pd.NA,
}

@register("cetermpremier_episode")
def _ceterm(series, cohort):
    return _textual_recode(series, _CETERM_MAP, pd_dtype="Int16")


_CESEV_MAP = {
    "Léger": 1,
    "Modéré": 2,
    "Sévère sans caractéristique psychotique": 3,
    "Sévère sans caractéristiques psychotiques": 3,
    "Sévère avec caractéristiques psychotiques congruentes": 4,
    "Sévère avec caractéristiques psychotiques non congruentes": 5,
    "Inconnu": pd.NA, "Ne sais pas": pd.NA,
}

@register("cesevtrouble_humeur_actuel")
def _cesev(series, cohort):
    return _textual_recode(series, _CESEV_MAP, pd_dtype="Int16")


# ----- SUICIDE attempt detail (LTSV / LTSG) --------------------------------
# These are very sparse (>97% NaN at yearly visits): only patients who attempted
# suicide get filled. Integer codes below preserve the categorical signal so
# the columns survive harmonization; per-cohort alignment of exact codes to
# DR's pre-numericized scheme is a follow-up clinical task.

_LTSV01_METHOD_MAP = {
    "Noyade": 2, "Pendaison": 4, "Saut": 3, "Arme à feu": 1,
    "Autre (précipitation sous métro/voiture/train, armes blanches, autre)": 5,
}

_LTSG01_OVERDOSE_MAP = {
    "Médicaments (drogues) avec effets sédatifs": 1,
    "Médicaments (drogues) sans effets sédatifs et pour toutes autres substances ingérées": 2,
    "Phlébotomie": 3,
}

_LTSG02_DRUG_MAP = {
    "Benzodiazépines / Hypnotique": 1,
    "Antipsychotique": 2,
    "Antidépresseur": 3,
    "Anticonvulsivant": 4,
    "Lithium": 5,
    "Autre": 6,
    "Inconnu": pd.NA,
}

_LTSG04_CONSCIOUSNESS_MAP = {
    "Complètement conscient et lucide": 1,
    "Conscient mais somnolent / engourdi": 2,
    "Endormi mais facilement réveillé": 3,
    "Comateux - évitement des stimuli douloureux; réflexes intacts; blessures suffisantes pour l'hospitalisation": 4,
    "Comateux - la plupart des réflexes absents, pas de défaillance respiratoire ou circulatoire, Soins Intensifs. et procédures médicales sérieuses": 5,
}

@register("ltsv01")
def _ltsv01(series, cohort):
    return _textual_recode(series, _LTSV01_METHOD_MAP, pd_dtype="Int16")

@register("ltsg01")
def _ltsg01(series, cohort):
    return _textual_recode(series, _LTSG01_OVERDOSE_MAP, pd_dtype="Int16")

@register("ltsg02")
def _ltsg02(series, cohort):
    return _textual_recode(series, _LTSG02_DRUG_MAP, pd_dtype="Int16")

@register("ltsg04")
def _ltsg04(series, cohort):
    return _textual_recode(series, _LTSG04_CONSCIOUSNESS_MAP, pd_dtype="Int16")


# --- Additional LTSV / LTSG attempt-detail items (BP text → code; DR already
#     numeric; SZ does not collect them). Like ltsv01/ltsg01-04 above these are
#     very sparse (a few dozen attempters at yearly visits). Lethality items
#     follow the Beck/Columbia medical-lethality ordinal (0 = no damage →
#     increasing severity → death); jump-height and drug-type are ordinal /
#     nominal. DR is already pre-numericized; exact BP↔DR code alignment for the
#     rare high-lethality categories remains a follow-up clinical task.

_LTSV02_JUMP_HEIGHT_MAP = {       # height of the jump (ordinal, low → high)
    "Hauteur d'homme": 1,
    "Hauteur modérée": 2,
    "D'un endroit élevé": 3,
}

_LTSV04_FIREARM_LETHALITY_MAP = {  # lethality if method = firearm
    "Pas de dommage": 0,
    "Balle logée dans une extrémité, légères hémorragies": 2,
    "Balle dans l'abdomen ou la poitrine, hémorragies importantes, signes vitaux instables": 5,
}

_LTSV05_IMMOLATION_LETHALITY_MAP = {  # lethality if method = self-immolation
    "Pas de dommage": 0,
    "Brûlures du premier degré": 1,
    "Brûlures du second degré": 2,
    "Brûlures du troisième degré sur 20% de la surface du corps": 4,
}

_LTSV06_DROWNING_LETHALITY_MAP = {  # lethality if method = drowning
    "Pas de dommage": 0,
    "Conscient - légère détresse respiratoire mais sans besoin de réanimation": 2,
    "Inconscient - efforts de réanimation très importants indispensables pour la survie": 4,
}

_LTSG03_DRUG_NONSEDATIVE_MAP = {   # non-sedative drug type used in overdose
    "Médicament en vente libre": 1,
    "Prescription autre que psychotrope": 2,
    "Autres substances ingérées": 3,
    "Lithium": 5,
    "Inconnu": pd.NA,
}

_LTSG05_OVERDOSE_LETHALITY_MAP = {  # lethality if method = medication overdose
    "Pas de conséquences médicales ou de traitement, ou minime": 0,
    "Quelques blessures (ex : bouche brûlée) et traitement en urgence ou en ambulatoire (ex : lavage gastrique)": 1,
    "Blessures suffisantes pour l'hospitalisation - signes vitaux et niveau de conscience peuvent être affectés": 3,
    "Effets majeures systématiques - tels que perforation gastro-intestinale, défaillance rénale, hémolyse, ou choc; signes vitaux instables": 4,
    "Mort": 6,
}

_LTSG06_PHLEBOTOMY_LETHALITY_MAP = {  # lethality if method = phlebotomy (cutting)
    "Surfaces griffées, pas ou peu de saignements, pas ou peu de soins requis pour les blessures": 1,
    "Saignement modéré avec caillot avant qu'il n'y ait perte de sang significative; simples soins requis": 2,
    "Saignement des veines majeures, danger de perte de sang considérable sans intervention chirurgicale - suture nécessaire mais pas de transfusion, zones vitales intactes et pas de changement dans les signes vitaux, soins en ambulatoire": 3,
    "Perte de sang importante, suture, transfusion nécessaire, réparation de tendons requise, blessures possibles sur la tête, le thorax, ou l'abdomen mais organes vitaux intacts et signes vitaux stables; rétablissement attendu après hospitalisation": 4,
}

for _canon, _map in (
    ("ltsv02", _LTSV02_JUMP_HEIGHT_MAP),
    ("ltsv04", _LTSV04_FIREARM_LETHALITY_MAP),
    ("ltsv05", _LTSV05_IMMOLATION_LETHALITY_MAP),
    ("ltsv06", _LTSV06_DROWNING_LETHALITY_MAP),
    ("ltsg03", _LTSG03_DRUG_NONSEDATIVE_MAP),
    ("ltsg05", _LTSG05_OVERDOSE_LETHALITY_MAP),
    ("ltsg06", _LTSG06_PHLEBOTOMY_LETHALITY_MAP),
):
    @register(_canon)
    def _ltsx(series, cohort, _m=_map):
        return _textual_recode(series, _m, pd_dtype="Int16")
