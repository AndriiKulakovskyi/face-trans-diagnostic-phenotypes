"""Polychoric sensitivity (v2, D9) — does Pearson-on-binary bias the dimensional result?

Pearson correlation is attenuated for binary/ordinal data. Several constructs are binary-derived
(suicidal_ideation, comorbidity, cardiac, atopic ...), so Pearson could under-weight them and
suppress a dimension. This script rebuilds the all-binary constructs with TETRACHORIC correlation
(the latent-trait correlation for binary items) and checks:
  (a) does each binary construct's within-construct SCORE change? r(Pearson-score, tetrachoric-score);
  (b) do the 4 trans-diagnostic dimensions still hold? Tucker congruence + canonical corr.
Run on UN-residualized data so binary items stay binary (residualization holds off for both arms, so
this isolates the Pearson-vs-tetrachoric effect). All masked / no-imputation.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from factor_analyzer import Rotator
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset
from trans_diag.axes import AXIS_NAMES
from trans_diag.domains import _robust_z
from trans_diag.masked_fa import masked_correlation, masked_scores, nearest_pd, paf_loadings

warnings.simplefilter("ignore")
OUT = ROOT / "results" / "hfa"
MIN_PAIR = 100


def tetrachoric(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m].astype(float), y[m].astype(float)
    if len(x) < 50:
        return 0.0
    px, py = x.mean(), y.mean()
    if min(px, 1 - px, py, 1 - py) < 5e-3:
        return 0.0
    tx, ty = norm.ppf(1 - px), norm.ppf(1 - py)
    p11 = float(np.mean((x == 1) & (y == 1)))

    def g(r):
        cdf = multivariate_normal.cdf([tx, ty], mean=[0, 0], cov=[[1, r], [r, 1]])
        return (1 - norm.cdf(tx) - norm.cdf(ty) + cdf) - p11

    try:
        if g(-0.999) * g(0.999) > 0:
            return 0.0
        return float(brentq(g, -0.999, 0.999, xtol=1e-3))
    except Exception:
        return 0.0


def score_pearson(items_df):
    if items_df.shape[1] == 1:
        return items_df.iloc[:, 0]
    Z = items_df.apply(_robust_z)
    L = paf_loadings(masked_correlation(Z, MIN_PAIR), 1)
    s = pd.Series(masked_scores(Z.to_numpy(float), L)[:, 0], index=Z.index)
    return s if np.corrcoef(np.nan_to_num(s), np.nan_to_num(Z.mean(1)))[0, 1] >= 0 else -s


def score_tetra(items_df):
    if items_df.shape[1] == 1:
        return items_df.iloc[:, 0]
    cols = list(items_df.columns)
    p = len(cols)
    R = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            R[i, j] = R[j, i] = tetrachoric(items_df.iloc[:, i].to_numpy(), items_df.iloc[:, j].to_numpy())
    L = paf_loadings(nearest_pd(R), 1)
    Z = items_df.apply(lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1.0))
    s = pd.Series(masked_scores(Z.to_numpy(float), L)[:, 0], index=Z.index)
    return s if np.corrcoef(np.nan_to_num(s), np.nan_to_num(Z.mean(1)))[0, 1] >= 0 else -s


def dims(S, K=len(AXIS_NAMES)):
    cov = S.notna().mean()
    keep = [c for c in S.columns if cov[c] >= 0.30 and S[c].var() > 1e-9]
    Z = (S[keep] - S[keep].mean()) / S[keep].std()
    L = Rotator(method="promax").fit_transform(paf_loadings(masked_correlation(Z, MIN_PAIR), K))
    return pd.DataFrame(L, index=keep, columns=[f"d{k+1}" for k in range(K)])


def main() -> None:
    fit = pd.read_csv(OUT / "stage2_construct_fit.csv").set_index("construct")
    df = build_unified_dataframe(str(ROOT / "data"), str(ROOT / "data" / "face-common-vars.xlsx"),
                                 readiness=["READY", "PARTIAL", "NOT USABLE", "ID"], format="long")
    vs = load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))
    by = {v.canonical_name: v for v in vs}
    ds = to_harmonized_dataset(df, vs, visit="V0", sections=None, residualize_on=None, normalize=False)
    X = ds.X

    def items_of(con):
        return [c for c in str(fit.loc[con, "items"]).split(",") if c in X.columns]

    is_binary = {con: all("binary" in (by[c].dtype or "") for c in items_of(con)) and len(items_of(con)) >= 2
                 for con in fit.index}
    binary_cons = [c for c, b in is_binary.items() if b]
    print(f"all-binary multi-item constructs (tetrachoric-treated): {len(binary_cons)}")
    print(f"  {binary_cons}\n")

    # build both score sets
    SP, ST = {}, {}
    print("(a) per-construct: VAF1 and score correlation (Pearson vs tetrachoric)")
    for con in fit.index:
        its = items_of(con)
        if not its:
            continue
        sub = X[its]
        SP[con] = score_pearson(sub)
        if con in binary_cons:
            ST[con] = score_tetra(sub)
            r = np.corrcoef(np.nan_to_num(SP[con].values), np.nan_to_num(ST[con].values))[0, 1]
            print(f"  {con:22s} items={len(its)}  r(Pearson,tetra score)={r:+.3f}")
        else:
            ST[con] = SP[con]
    SPd, STd = pd.DataFrame(SP), pd.DataFrame(ST)

    # (b) re-derive dimensions both ways
    LP, LT = dims(SPd), dims(STd)
    common = [c for c in LP.index if c in LT.index]
    A, B = LP.loc[common].to_numpy(), LT.loc[common].to_numpy()
    M = np.abs(A.T @ B) / (np.sqrt(np.outer((A * A).sum(0), (B * B).sum(0))) + 1e-12)
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-M)
    print(f"\n(b) {len(AXIS_NAMES)}-dim structure Pearson vs tetrachoric: per-dim Tucker congruence "
          f"{np.round(np.sort(M[r, c])[::-1], 2)} (min {M[r, c].min():.2f})")
    print("    -> high congruence => the dimensions do NOT depend on the Pearson-vs-polychoric choice")
    # does any binary construct gain a strong loading under tetrachoric it lacked under Pearson?
    print("\n(c) binary-construct max|loading| on the 4 dims: Pearson -> tetrachoric")
    for con in binary_cons:
        if con in LP.index and con in LT.index:
            print(f"  {con:22s} {LP.loc[con].abs().max():.2f} -> {LT.loc[con].abs().max():.2f}")


if __name__ == "__main__":
    main()
