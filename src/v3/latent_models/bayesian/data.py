"""V3 Bayesian engine — data layer (config-driven, no imputation).

Assembles the observed-cell model inputs from the harmonized V0 baseline, driven ENTIRELY
by the prior loading matrix (configs/prior_loading_matrix_v3.csv): which items are modeled,
their likelihood family / modeling block, and their burden orientation (item_sign).

Continuous block (gaussian / lognormal / student_t): oriented (higher = burden), lognormal
log-transformed, z-scored -> a [N x Jc] matrix with NaN = missing (never imputed), plus the
observed-pattern groups the marginalized likelihood sums over.

Explicit block (bernoulli / ordered_logistic / neg_binomial): native integer arrays kept on
their own scale (binary 0/1, ordinal 0..K-1, counts >=0), each with its own observed mask.

Cohort-balanced: the 500 most-complete patients per cohort (BP/SZ/DR equalized) so sample
size cannot drive structure. V0 only. Mean-fill is produced ONLY as a warm-start aid for
initialization (`mean_fill`), never used in any likelihood.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from v3.data import build_unified_dataframe, load_variables, to_harmonized_dataset  # noqa: E402

MATRIX = REPO / "configs" / "prior_loading_matrix_v3.csv"
SEED = 20260605
COHORTS = ["bp", "sz", "dr"]


@dataclass
class ModelData:
    # continuous block
    M: np.ndarray                      # [N, Jc] z-scored oriented; NaN = missing
    cont_items: list[str]
    cont_home: list[str]               # home factor per continuous item
    patterns: dict                     # {obs-tuple: [row idx]} with >= min_group members
    n_drop: int
    mean_fill: np.ndarray              # [N, Jc] mean-filled COPY — init/warm-start ONLY
    # explicit block (native scale)
    bin_items: list[str]
    Bin: np.ndarray                    # [N, Jb] 0/1, NaN = missing
    ord_items: list[str]
    Ord: np.ndarray                    # [N, Jo] 0..K-1, NaN = missing
    ord_K: list[int]                   # #categories per ordinal item
    cnt_items: list[str]
    Cnt: np.ndarray                    # [N, Jcnt] counts, NaN = missing
    expl_home: dict                    # {item: home factor} for every explicit item
    # bookkeeping
    cohort: np.ndarray
    index: pd.MultiIndex
    item_sign: dict = field(default_factory=dict)


def _item_meta(matrix_path: Path = MATRIX) -> pd.DataFrame:
    """One row per modeled item: home factor, family, block, sign."""
    m = pd.read_csv(matrix_path)
    home = (m[m.prior_type.isin(["primary", "g_anchor"])]
            .drop_duplicates("item").set_index("item")["factor"])
    meta = (m.drop_duplicates("item").set_index("item")
            [["likelihood_family", "modeling_block", "item_sign"]].copy())
    meta["home"] = home
    return meta


def load_model_data(n_per_cohort: int = 500, min_group: int = 10,
                    sleep: str = "objective", matrix_path: Path = MATRIX,
                    items_keep: list[str] | None = None) -> ModelData:
    """Build the observed-cell model inputs. `items_keep` optionally restricts the modeled
    item set (used by the staged build sequence)."""
    meta = _item_meta(matrix_path)
    variables = load_variables(str(REPO / "data" / "face-common-vars.xlsx"))
    df = build_unified_dataframe("data", str(REPO / "data" / "face-common-vars.xlsx"),
                                 readiness=["READY", "PARTIAL"], format="long")
    ds = to_harmonized_dataset(df, variables, visit="V0", normalize=False, apply_skip_logic=True)
    X = ds.X
    cohort = pd.Series(X.index.get_level_values("cohort"), index=X.index)

    items = [it for it in meta.index if it in X.columns]
    if items_keep is not None:
        items = [it for it in items if it in set(items_keep)]
    # canonical sleep: objective PSQI only (drop subjective items psqi/psqi13/15/17)
    if sleep == "objective":
        drop_subj = {"psqi", "psqi13", "psqi15", "psqi17"}
        items = [it for it in items if it not in drop_subj]

    cont_items, bin_items, ord_items, cnt_items = [], [], [], []
    for it in items:
        fam, blk = meta.loc[it, "likelihood_family"], meta.loc[it, "modeling_block"]
        if blk == "continuous":
            cont_items.append(it)
        elif fam == "bernoulli":
            bin_items.append(it)
        elif fam == "ordered_logistic":
            ord_items.append(it)
        elif fam == "neg_binomial":
            cnt_items.append(it)

    # ---- continuous matrix: orient (sign), log (lognormal), z-score ----
    Mdf = pd.DataFrame(index=X.index)
    for c in cont_items:
        sign = int(meta.loc[c, "item_sign"])
        fam = meta.loc[c, "likelihood_family"]
        v = pd.to_numeric(X[c], errors="coerce").astype(float)
        if fam == "lognormal":
            mn = np.nanmin(v.values)
            v = np.log1p(v - mn + 1e-6) if (mn is not None and mn <= 0) else np.log(v)
        v = sign * v
        sd = v.std()
        Mdf[c] = (v - v.mean()) / sd if sd and sd > 0 else np.nan

    # ---- explicit arrays (native scale) ----
    def _grab(cols):
        return pd.DataFrame({c: pd.to_numeric(X[c], errors="coerce") for c in cols}, index=X.index)

    Bdf, Odf, Cdf = _grab(bin_items), _grab(ord_items), _grab(cnt_items)

    # ---- cohort-balanced subsample: 500 most-complete (on continuous) per cohort ----
    keep = Mdf.notna().any(axis=1)
    Mdf, Bdf, Odf, Cdf, coh = Mdf[keep], Bdf[keep], Odf[keep], Cdf[keep], cohort[keep]
    obs_count = Mdf.notna().sum(axis=1).values
    idx = []
    for c in COHORTS:
        pool = np.where(coh.values == c)[0]
        if len(pool) == 0:
            continue
        order = pool[np.argsort(-obs_count[pool], kind="stable")]
        idx.extend(order[: min(n_per_cohort, len(pool))])
    idx = np.sort(np.array(idx))
    Mdf, Bdf, Odf, Cdf, coh = Mdf.iloc[idx], Bdf.iloc[idx], Odf.iloc[idx], Cdf.iloc[idx], coh.iloc[idx]

    Mv = Mdf.to_numpy()
    N, Jc = Mv.shape

    # ---- continuous observed-pattern groups (>= min_group) ----
    patterns: dict = {}
    for i in range(N):
        o = tuple(np.flatnonzero(~np.isnan(Mv[i])))
        if o:
            patterns.setdefault(o, []).append(i)
    patterns = {o: r for o, r in patterns.items() if len(r) >= min_group}
    n_drop = N - sum(len(r) for r in patterns.values())

    # mean-fill copy — INIT ONLY (never a likelihood input)
    col_mean = np.nanmean(Mv, axis=0)
    mean_fill = np.where(np.isnan(Mv), col_mean, Mv)

    # ordinal: recode each item to 0..K-1 on observed values
    Ov = Odf.to_numpy().astype(float)
    ord_K = []
    for k in range(Ov.shape[1]):
        col = Ov[:, k]
        obs = ~np.isnan(col)
        uniq = np.unique(col[obs])
        remap = {v: i for i, v in enumerate(uniq)}
        for v, i in remap.items():
            col[col == v] = i
        Ov[:, k] = col
        ord_K.append(max(2, len(uniq)))

    return ModelData(
        M=Mv, cont_items=cont_items, cont_home=[meta.loc[c, "home"] for c in cont_items],
        patterns=patterns, n_drop=n_drop, mean_fill=mean_fill,
        bin_items=bin_items, Bin=Bdf.to_numpy().astype(float),
        ord_items=ord_items, Ord=Ov, ord_K=ord_K,
        cnt_items=cnt_items, Cnt=Cdf.to_numpy().astype(float),
        expl_home={it: meta.loc[it, "home"] for it in bin_items + ord_items + cnt_items},
        cohort=coh.values, index=Mdf.index,
        item_sign={it: int(meta.loc[it, "item_sign"]) for it in items},
    )
