"""The reference (null) design — the diagnosis + severity + baseline-outcome bar every increment beats.

Builds the strictly-nested ladder per outcome (all rungs on the SAME complete-case modelling sample, so
their held-out ELPD is comparable):

  R0   nuisance      : age + sex            (+ a site random intercept, handled by the GLM)
  R1   + diagnosis   : R0 + DSM-5 arm       (the 7 subtypes — the thing to beat; implies cohort)
  R2   + severity    : R1 + Sev             (Sev = baseline CGI-S, or the G coordinate when the
                                             outcome IS CGI-S — `severity_col` is resolved by the caller)
  R3y  + baseline Y  : R2 + Y_V0            (the ANCOVA autoregression term — makes "incremental" honest)

Continuous predictors are z-scored on the modelling sample (Normal(0,1) priors, the project idiom);
`arm` is dummy-encoded (drop-first). The horizon outcome is z-scored for the gaussian family so ELPD is
on a common scale across the ladder. Stage 43 appends the durable-coordinate / strata blocks to R3y.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RUNGS = ("R0", "R1", "R2", "R3y")
_COVARS = ("age", "sex", "siteid_city", "arm")


def severity_column(spec, *, cgi_baseline_col: str) -> str:
    """The R2 severity term for an outcome: the G coordinate when the outcome itself is a severity scale
    (`severity_anchor == 'G'`), else the manifest baseline CGI-S column."""
    return "overall_severity__mean" if spec.severity_anchor == "G" else cgi_baseline_col


def modeling_frame(frame: pd.DataFrame, spec, *, horizon: str, severity_col: str,
                   covars=_COVARS) -> pd.DataFrame:
    """Complete-case rows for the ladder: baseline + horizon outcome, the severity term, and all
    covariates present (no imputation — incomplete rows are dropped, identically for every rung)."""
    need = [f"{spec.name}__V0", f"{spec.name}__{horizon}", severity_col, *covars]
    need = [c for c in need if c in frame.columns]
    return frame.dropna(subset=need).copy()


def outcome_vector(sub: pd.DataFrame, spec, *, horizon: str):
    """Return (y, family, n_cat). Gaussian outcomes are z-scored (common ELPD scale); bernoulli -> 0/1;
    ordinal -> 0..K-1 integer codes."""
    y = sub[f"{spec.name}__{horizon}"].to_numpy("float64")
    if spec.family == "gaussian":
        return (y - y.mean()) / y.std(), "gaussian", None
    if spec.family == "bernoulli":
        return y.astype(int), "bernoulli", None
    cats = np.unique(y)
    codes = np.searchsorted(cats, y)
    return codes.astype(int), "ordinal", int(len(cats))


def site_index(sub: pd.DataFrame, *, site_col: str = "siteid_city"):
    """Factorize site into a contiguous 0-based index + count, for the GLM's random intercept."""
    codes, uniq = pd.factorize(sub[site_col])
    return codes.astype("int64"), int(len(uniq))


def _z(sub: pd.DataFrame, col: str) -> np.ndarray:
    v = sub[col].to_numpy("float64")
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 0 else 1.0)


def design_for_rung(sub: pd.DataFrame, spec, rung: str, *, severity_col: str, horizon: str):
    """Build the [N, P] design (no intercept column) + column names for one rung. Strictly nested:
    R0 ⊂ R1 ⊂ R2 ⊂ R3y."""
    if rung not in RUNGS:
        raise ValueError(f"rung {rung!r} not in {RUNGS}")
    cols: dict[str, np.ndarray] = {"age": _z(sub, "age"), "sex": sub["sex"].to_numpy("float64")}
    if rung in ("R1", "R2", "R3y"):
        dummies = pd.get_dummies(sub["arm"], prefix="arm", drop_first=True)
        for c in dummies.columns:
            cols[c] = dummies[c].to_numpy("float64")
    if rung in ("R2", "R3y"):
        cols[f"sev::{severity_col}"] = _z(sub, severity_col)
    if rung == "R3y":
        cols[f"{spec.name}__V0"] = _z(sub, f"{spec.name}__V0")
    names = list(cols)
    X = np.column_stack([cols[n] for n in names]) if names else np.empty((len(sub), 0))
    return X, names
