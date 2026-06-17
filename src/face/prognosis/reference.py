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

from face.prognosis import CANON

RUNGS = ("R0", "R1", "R2", "R3y")
_COVARS = ("age", "sex", "siteid_city", "arm")
SPECIFICS = tuple(a for a in CANON if a != "overall_severity")   # the 8 ⊥G specific axes (the ceiling)
ARCH_COLS = tuple(f"arch_w{k}" for k in range(7))                 # 8 archetype weights, drop-one reference
TESS_COLS = tuple(f"tess_r{k}" for k in range(3))                 # 4 tessellation regions, drop-one reference


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


def coord_eiv_block(sub: pd.DataFrame, axes):
    """Errors-in-variables block for coordinate `axes`: standardized posterior mean (by the axis's
    population SD) + the per-patient measurement SD on the SAME scale, so beta_eiv is comparable to the
    z-scored fixed effects and wide-posterior patients self-down-weight. Returns (obs, sd, names)."""
    obs, sd = [], []
    for ax in axes:
        m = sub[f"{ax}__mean"].to_numpy("float64")
        s = sub[f"{ax}__sd"].to_numpy("float64")
        psd = m.std() or 1.0
        obs.append((m - m.mean()) / psd)
        sd.append(s / psd)
    return np.column_stack(obs), np.column_stack(sd), list(axes)


def fixed_block(sub: pd.DataFrame, cols):
    """Z-scored fixed predictor block (the archetype / tessellation membership representations). Returns
    ([N, len(cols)] design, names)."""
    mat = np.column_stack([_z(sub, c) for c in cols])
    return mat, list(cols)


def foundation_design(sub: pd.DataFrame, spec, *, severity_col: str, horizon: str):
    """The shared foundation for the head-to-head: age + sex + severity + baseline outcome — **no arm,
    no map** (site enters as the GLM random intercept). DSM-5 and the map are then added on top, so the
    four head-to-head models (D / +DSM-5 / +map / +both) are all nested on this same base."""
    cols = {"age": _z(sub, "age"), "sex": sub["sex"].to_numpy("float64"),
            f"sev::{severity_col}": _z(sub, severity_col),
            f"{spec.name}__V0": _z(sub, f"{spec.name}__V0")}
    names = list(cols)
    return np.column_stack([cols[n] for n in names]), names


def arm_block(sub: pd.DataFrame):
    """DSM-5 diagnosis as the 7-subtype `arm` dummies (drop-first reference). The competitor to beat."""
    d = pd.get_dummies(sub["arm"], prefix="arm", drop_first=True)
    return d.to_numpy("float64"), list(d.columns)


def armB_block(sub: pd.DataFrame, *, profiles_path):
    """Per-patient Arm-B (G-residualized) archetype weights: project each patient's 8-specific
    coordinate means onto the FIXED M2 Arm-B archetype profiles (reusing the M2 simplex projection),
    z-scored, drop-one-reference. The clean ⊥G strata representation. Returns ([N, A-1] design, names)."""
    from face.strata.archetypes import project_to_Z

    prof = pd.read_csv(profiles_path)
    ZB = prof[prof["arm"] == "B_specifics"][list(SPECIFICS)].to_numpy("float64")     # [A, 8]
    X = sub[[f"{ax}__mean" for ax in SPECIFICS]].to_numpy("float64")                  # [N, 8]
    W = project_to_Z(X, ZB)[:, :-1]                                                   # [N, A-1] drop last
    z = (W - W.mean(0)) / np.where(W.std(0) > 0, W.std(0), 1.0)
    return z, [f"archB_w{k}" for k in range(W.shape[1])]
