"""Phase-1 harness — paired representation comparison on the continuous-GAF backbone.

For each target (recovery / deterioration), horizon (V1 1-yr, V2 2-yr) and scope (pooled, BP+DR), every arm is
fit with the *same* XGBoost config on the *same* CV folds; the GAF@H regression gives a per-patient Gaussian
predictive, from which the binary probability is **derived by thresholding** (the pre-registered backbone).
We score CRPS (continuous), net benefit (decision-curve), Brier/log-loss/AUC/calibration (binary).

Two baselines bracket the autoregression question:
* **REF**  = DSM-5 + latent-G severity + baseline GAF  → the rigorous *incremental* bar (predict change beyond
  where the patient started).
* **REF0** = DSM-5 + latent-G severity, **GAF dropped** → the *unconditional* sensitivity (no autoregression
  control). REF − REF0 quantifies how much of the signal is "they were already (un)well".

Sufficiency (does raw add over the latent map?) is tested under both: ``REF+RAW+LAT`` vs ``REF+LAT-A`` and
``REF0+RAW`` vs ``REF0+LAT-A``, with a paired bootstrap CI against a ±1-SE-of-the-raw-arm band.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import SEED, cv, data, metrics, models

ARM_SPECS: dict[str, tuple[str, ...]] = {
    # incremental ladder, WITH baseline GAF (primary)
    "REF": ("ref",),
    "REF+RAW": ("ref", "raw"),
    "REF+LAT-mu": ("ref", "lat_mu"),
    "REF+LAT-sigma": ("ref", "lat_sigma"),
    "REF+LAT-A": ("ref", "lat_a"),
    "REF+RAW+LAT": ("ref", "raw", "lat_a"),
    # autoregression sensitivity, WITHOUT baseline GAF
    "REF0": ("ref0",),
    "REF0+RAW": ("ref0", "raw"),
    "REF0+LAT-A": ("ref0", "lat_a"),
}
# sufficiency pairs: (label, full_arm, latent_arm, raw_margin_arm)
SUFF_PAIRS = (
    ("with_gaf", "REF+RAW+LAT", "REF+LAT-A", "REF+RAW"),
    ("no_gaf", "REF0+RAW", "REF0+LAT-A", "REF0+RAW"),
)
NB_LO, NB_HI, NB_STEP = 0.05, 0.50, 0.01


# --------------------------------------------------------------------------- feature blocks
def _ref_matrix(E: pd.DataFrame, *, include_gaf: bool) -> np.ndarray:
    """Clinician bar: one-hot DSM-5 arm (NaN as its own level) + latent G severity (+ baseline GAF if asked)."""
    arm = E["arm"] if "arm" in E.columns else pd.Series("na", index=E.index)
    ohe = pd.get_dummies(arm.astype("category"), dummy_na=True).to_numpy("float64")
    cols = ["G_mean", "egf__V0"] if include_gaf else ["G_mean"]
    return np.hstack([ohe, E[cols].to_numpy("float64")])


def _blocks(E: pd.DataFrame, rawE: pd.DataFrame) -> dict[str, np.ndarray]:
    lat = data.latent_blocks()
    return {
        "ref": _ref_matrix(E, include_gaf=True),
        "ref0": _ref_matrix(E, include_gaf=False),
        "raw": rawE.to_numpy("float64"),                      # NaN preserved → XGBoost native missing
        "lat_mu": E[lat["LAT-mu"]].to_numpy("float64"),
        "lat_sigma": E[lat["LAT-sigma"]].to_numpy("float64"),
        "lat_a": E[lat["LAT-A"]].to_numpy("float64"),
    }


# --------------------------------------------------------------------------- derived probability
def _derive_p(target: str, mu: np.ndarray, sigma: float, gaf0: np.ndarray) -> np.ndarray:
    """Binary probability from the Gaussian GAF@H predictive N(mu, sigma)."""
    if target == "egf_recovery":                              # P(GAF@H >= 71)
        return norm.sf((71.0 - mu) / sigma)
    if target == "egf_deterioration":                         # P(GAF@H <= GAF0 - 10)
        return norm.cdf(((gaf0 - 10.0) - mu) / sigma)
    raise ValueError(target)


# --------------------------------------------------------------------------- bootstrap helpers
def _boot_diff(a: np.ndarray, b: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Paired bootstrap of mean(a) − mean(b) over patients (pointwise scores). Returns (delta, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    base = float(a.mean() - b.mean())
    d = np.array([a[idx].mean() - b[idx].mean() for idx in (rng.integers(0, n, n) for _ in range(n_boot))])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return base, float(lo), float(hi)


def _boot_se(a: np.ndarray, *, n_boot: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = len(a)
    return float(np.array([a[rng.integers(0, n, n)].mean() for _ in range(n_boot)]).std(ddof=1))


def _verdict(lo: float, hi: float, margin: float) -> str:
    if lo > margin:
        return "raw-adds"
    if hi < -margin:
        return "latent-better"
    if lo >= -margin and hi <= margin:
        return "sufficient"
    return "inconclusive"


# --------------------------------------------------------------------------- one target × horizon × scope
def run_cell(frame: pd.DataFrame, raw: pd.DataFrame, *, target: str, horizon: str, scope: str,
            seed: int = SEED, n_boot: int = 2000, n_splits: int = 5, n_repeats: int = 2) -> dict:
    el = data.eligible(frame, target, horizon)
    E = frame[el]
    rawE = raw.reindex(E.index)
    y_bin = E[f"ep_{target}__{horizon}"].to_numpy("int64")
    y_cont = E[f"egf__{horizon}"].to_numpy("float64")
    gaf0 = E["egf__V0"].to_numpy("float64")
    cohort = data.cohort_of(E)
    folds = cv.make_folds(y_bin, cohort, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    blocks = _blocks(E, rawE)
    tag = {"target": target, "horizon": horizon, "scope": scope}

    scalar_rows, nb_rows = [], []
    crps_pw, brier_pw, p_der = {}, {}, {}
    for arm, keys in ARM_SPECS.items():
        X = np.hstack([blocks[k] for k in keys])
        mu = models.oof_regress(X, y_cont, folds, seed=seed)
        sigma = float(max(np.std(y_cont - mu, ddof=1), 1e-6))
        p = _derive_p(target, mu, sigma, gaf0)
        cw = metrics.crps_gaussian_pointwise(y_cont, mu, sigma)
        crps_pw[arm], brier_pw[arm], p_der[arm] = cw, (p - y_bin) ** 2, p
        scalar_rows.append({**tag, "arm": arm, "n": int(len(y_bin)), "events": int(y_bin.sum()),
                            "crps": float(cw.mean()), "brier": metrics.brier(y_bin, p),
                            "logloss": metrics.log_loss(y_bin, p), "auc": metrics.auc(y_bin, p),
                            "cal_slope": metrics.calibration_slope(y_bin, p), "sigma": sigma})
        nb = metrics.net_benefit_band(y_bin, p, NB_LO, NB_HI, NB_STEP)
        for t, m, ta in zip(nb["thresholds"], nb["model"], nb["treat_all"], strict=True):
            nb_rows.append({**tag, "arm": arm, "threshold": float(t),
                            "net_benefit": float(m), "treat_all": float(ta)})

    suff_rows = []
    for label, full, lat, raw_arm in SUFF_PAIRS:
        for mname, pw in (("crps", crps_pw), ("brier", brier_pw)):
            delta, lo, hi = _boot_diff(pw[full], pw[lat], n_boot=n_boot, seed=seed)
            margin = _boot_se(pw[raw_arm], n_boot=n_boot, seed=seed + 1)
            # CRPS/Brier are lower-is-better → "full advantage" = −(full − lat); verdict on the advantage CI
            suff_rows.append({**tag, "contrast": label, "metric": mname, "delta_full_minus_lat": delta,
                              "ci_lo": lo, "ci_hi": hi, "raw_se_margin": margin,
                              "verdict": _verdict(-hi, -lo, margin)})
        d, lo, hi, _ = metrics.paired_auc_delta(y_bin, p_der[lat], p_der[full], n_boot=n_boot, seed=seed)
        suff_rows.append({**tag, "contrast": label, "metric": "auc", "delta_full_minus_lat": d,
                          "ci_lo": lo, "ci_hi": hi, "raw_se_margin": float("nan"),
                          "verdict": "raw-adds" if lo > 0 else ("latent-better" if hi < 0 else "tie")})

    return {"scalar": pd.DataFrame(scalar_rows), "net_benefit": pd.DataFrame(nb_rows),
            "sufficiency": pd.DataFrame(suff_rows)}


# --------------------------------------------------------------------------- full sweep
TARGETS = ("egf_recovery", "egf_deterioration")
HORIZONS = ("V1", "V2")
SCOPES = {"pooled": None, "bp_dr": ("bp", "dr")}


def run_all(*, seed: int = SEED, n_boot: int = 2000, n_splits: int = 5, n_repeats: int = 2,
            subsample: int | None = None) -> dict[str, pd.DataFrame]:
    """Run every target × horizon × scope; return concatenated scalar / net_benefit / sufficiency tables."""
    raw = data.load_raw()
    out: dict[str, list] = {"scalar": [], "net_benefit": [], "sufficiency": []}
    for scope, cohorts in SCOPES.items():
        frame = data.assemble(cohorts=cohorts)
        if subsample is not None and len(frame) > subsample:
            frame = frame.sample(subsample, random_state=seed)
        for target in TARGETS:
            for horizon in HORIZONS:
                r = run_cell(frame, raw, target=target, horizon=horizon, scope=scope, seed=seed,
                             n_boot=n_boot, n_splits=n_splits, n_repeats=n_repeats)
                for k in out:
                    out[k].append(r[k])
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}
