"""Skip-logic decoding: recover structural zeros from conditional (gated) items.

Many clinical instruments use *skip logic*: a gate question determines whether
downstream items are asked at all. In the ISF suicide module, ISF05 ("have you
ever attempted suicide?") gates ISF07 ("how many times?"), ISF08/ISF09
(violent/serious attempts) and their counts. When the gate is **No**, the
dependent items are *not asked* and arrive blank — but those blanks are **not
missing data**: they are *structural zeros* (no attempts ⇒ 0 attempts, 0 violent,
0 serious). Treating them as missing both discards the information that the
patient has zero and collapses the coverage of the count features (ISF07 sits at
~25–38 % observed when ~90 % of patients actually have a known value).

This module decodes that skip logic: where a gate has a value that logically
determines a dependent and the dependent is currently missing, the dependent is
set to its structural value. Most rules use a binary ``No == 0`` gate. Smoking
status is categorical instead: ``1 == never smoker`` determines lifetime
pack-years to be zero.

It is deliberately **not imputation** — we only fill cells the instrument's own
logic determines:

* gate = No,  dependent missing  → fill 0           (structural zero, recovered)
* gate = Yes, dependent missing  → left NaN         (genuinely unobserved)
* gate missing / unknown         → left NaN         (cannot infer)
* dependent already has a value   → never overwritten (rare gate=No-but-count>0
  data-entry inconsistencies are preserved exactly)

Rules are declarative and **ordered** so a gate filled by an earlier rule
cascades into a later one (ISF05=No → ISF08=0 → ISF08A=0). The ISF module is
present and structurally identical in all three FACE cohorts (the gate coding
differs — BP/SZ ``Oui/Non``, DR ``1/0`` — but ``rules.py`` harmonizes it to
``0/1`` before this step runs). The engine generalizes to any gated family.
Fagerstrom scores are deliberately not filled for never/former smokers: the
source instrument is administered to current smokers only, so those cells are
not applicable rather than observed zero nicotine-dependence scores.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

__all__ = [
    "DEFAULT_SKIP_RULES",
    "SMOKING_SKIP_RULES",
    "SUICIDE_SKIP_RULES",
    "SkipRule",
    "decode_skip_logic",
]


@dataclass(frozen=True)
class SkipRule:
    """One ``gate → dependents`` structural-zero rule.

    Where ``frame[gate]`` equals ``no_value`` (the harmonized "No") **and** a
    dependent cell is missing, that cell is set to ``fill``. Existing values are
    never overwritten; a gate that is missing / unknown triggers no fill.
    """

    gate: str
    dependents: tuple[str, ...]
    fill: float = 0.0
    no_value: float = 0.0
    rationale: str = ""


# ISF suicide module (3-cohort). ISF05 gates the whole attempt block; ISF08/ISF09
# gate their own counts. ORDER MATTERS: ISF05=No first sets ISF08/ISF09=0, which
# the later rules then cascade into ISF08A/ISF09A=0 (covers both "never attempted"
# and "attempted but none violent/serious").
SUICIDE_SKIP_RULES: tuple[SkipRule, ...] = (
    SkipRule(
        "isf05", ("isf07", "isf08", "isf09"),
        rationale="No lifetime suicide attempt -> 0 attempts; no violent/serious attempts.",
    ),
    SkipRule(
        "isf08", ("isf08a",),
        rationale="No violent attempts -> 0 violent attempts.",
    ),
    SkipRule(
        "isf09", ("isf09a",),
        rationale="No serious attempts -> 0 serious attempts.",
    ),
)


# Smoking status is harmonized to 1=never smoker, 2=former smoker,
# 3=current smoker in all cohorts. The source questionnaires leave pack-years
# blank for never smokers, but this is a known zero exposure, not missing data.
# Existing contradictory values are preserved for audit, as for all SkipRules.
SMOKING_SKIP_RULES: tuple[SkipRule, ...] = (
    SkipRule(
        "suncf_cigarettes_lt",
        ("sudose_cigarettes_lt",),
        no_value=1.0,
        rationale="Never smoker -> 0 lifetime cigarette pack-years.",
    ),
)


DEFAULT_SKIP_RULES: tuple[SkipRule, ...] = (
    *SUICIDE_SKIP_RULES,
    *SMOKING_SKIP_RULES,
)


def decode_skip_logic(
    frame: pd.DataFrame,
    rules: Iterable[SkipRule] = DEFAULT_SKIP_RULES,
    *,
    inplace: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fill structural zeros implied by instrument skip-logic.

    Parameters
    ----------
    frame:
        A patient × variable frame whose columns are canonical names. Works on
        either the wide V0 matrix or the long merged frame — the rule is applied
        row-wise. Gate columns must be numeric (harmonized so "No" == 0).
    rules:
        Ordered :class:`SkipRule` s. Order matters: a dependent filled by an
        earlier rule (e.g. ``isf08``) can serve as the gate of a later one.
    inplace:
        Mutate ``frame`` directly when ``True`` (default copies it).

    Returns
    -------
    (frame, report)
        ``report`` is a list of ``{"gate", "dependent", "n_filled"}`` dicts, for
        QA / audit (e.g. surfaced in the harmonization report).
    """
    out = frame if inplace else frame.copy()
    report: list[dict] = []
    for rule in rules:
        if rule.gate not in out.columns:
            continue
        # Compare on a numeric view so an Int8/object gate still matches; NaN
        # (missing / unknown) compares False and is therefore never decoded.
        gate_is_no = pd.to_numeric(out[rule.gate], errors="coerce") == rule.no_value
        for dep in rule.dependents:
            if dep not in out.columns:
                continue
            mask = gate_is_no & out[dep].isna()
            n = int(mask.sum())
            if n:
                out.loc[mask, dep] = rule.fill
            report.append({"gate": rule.gate, "dependent": dep, "n_filled": n})
    return out, report
