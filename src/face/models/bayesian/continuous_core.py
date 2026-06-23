"""Continuous-core measurement engine (Stages S1–S2) — marginalized bifactor/ESEM, full-N V0.

Driven by ``configs/prior_loading_matrix_v3.csv``; reads ``data/processed/baseline_v0.parquet``
(built by ``scripts/01_build_data.py``). Factor set: a general factor **G**
(``overall_severity``) + the continuous specific factors **cognition / metabolic / inflammatory /
sleep**. Continuous (gaussian / lognormal) indicators only — the binary/ordinal/count blocks are
added at later stages (S3+).

Two stages share one builder, gated by flags resolved from the prior matrix:

    S1  cross=False, windows=False, Phi = I        the certified "first stable fit"
    S2  cross=True,  windows=True,  Phi = Phi_full  ESEM cross-loadings + MADRS/QIDS/STAI windows
                                                    + inter-dimension correlations Phi

Model (the doc's §3.1):
    G_i  ~ Normal(0, 1)                          general factor (per patient)
    D_i  ~ Normal(0, Phi_spec)                   specific factors (Phi=I at S1; LKJ at S2)
    eta_ij = lambda_jG * G_i + sum_k lambda_jk * D_ik
    X_ij ~ Normal(eta_ij, sigma_j)               OBSERVED cells only (NaN -> no term; no imputation)

Loading cells come from the prior matrix x the stage flags:
  * ``pos``  (primary / G-anchor home cells) — ``TruncatedNormal(mu, sd, >0)``, sign-anchored on the
    burden-oriented data (this anchors the rotation; the specific factors keep their orientation).
  * ``signed`` (bifactor-G cells; at S2 also specific<->specific ``plausible_cross`` cells and the
    window cells) — ``Normal(mu, sd)``, shrunk toward 0. The ~980 ``unlikely`` cells stay hard-zero
    (the default ``plausible_only`` ESEM arm); the bifactor G cells keep sd 0.25, specific<->specific
    cross cells are tightened by ``cross_sd_scale`` so Phi carries the inter-factor association.

Bifactor identification: **G is held orthogonal to the specifics** (Phi_full has an identity G
row/col; only the specific block correlates via LKJ). The correlated-G sensitivity variant is an S5
addition, not done here.

The marginalized (Woodbury, low-rank) likelihood integrates the latents out and is fully vectorized
over patients with a 0/1 mask — no per-patient funnel, no observed-pattern grouping, no patient
dropped — which is what lets the full N = 9,013 certify on the Mac CPU. The Phi reparameterization
``Lam_tilde = Lam @ chol(Phi_full)`` makes ``Sigma = Lam Phi Lam' + Psi = Lam_tilde Lam_tilde' + Psi``,
so the exact S1 Woodbury kernel runs unchanged on ``Lam_tilde``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
# Processed-data dir is overridable via FACE_DATA_DIR so the engine can run on the synthetic
# FACE-like dataset (synthetic/generate_face_like.py) without the confidential cohort data (P7-03).
PROC = Path(os.environ.get("FACE_DATA_DIR", str(REPO / "data" / "processed")))
MATRIX = REPO / "configs" / "prior_loading_matrix_v3.csv"

S1_FACTORS = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep"]
# S3a (marginalized continuous) adds only DEVELOPMENTAL-RISK — it has strong continuous anchors
# (CTQ×6, age-of-onset, WURS, perinatal). SUICIDALITY is a binary-dominated factor (its only
# continuous indicator, isf07, is too thin to identify it — gives R-hat 1.55), so it is deferred to
# S3b's explicit mixed-likelihood block where the binary ISF ideation/attempt items anchor it.
S3A_FACTORS = S1_FACTORS + ["developmental_risk"]
# S3b's full factor set: suicidality + developmental enter the explicit block (binary/count/ordinal).
S3_FACTORS = S1_FACTORS + ["suicidality", "developmental_risk"]
# S5 full integration: + mania (continuous, marginalized) + substance (mixed: continuous Fagerström +
# count cigarettes + binary alcohol/cannabis SUD, so substance is an EXPLICIT factor).
S5_FACTORS = S3_FACTORS + ["mania_activation", "substance"]
# S4 tests the THIN BP/DR-only ANHEDONIA factor (one dedicated indicator, qids_anhedonia_interest;
# SZ has no QIDS) on top of the S3a continuous map. The methods-doc question: does a thin,
# cohort-specific factor identify at all, or does it merge into G / the depression windows? (Adjudication.)
S4_FACTORS = S3A_FACTORS + ["anhedonia"]
WINDOWS = ["madrs", "qidsr120", "staya"]   # cross-loading windows (no home factor; §2)
G_KEY = "overall_severity"


@dataclass
class CorePrep:
    M: np.ndarray                 # [N, J] z-scored, oriented continuous block; NaN = missing
    items: list[str]
    home: list[str]               # home factor per item ("" for the cross-loading windows)
    factor_cols: list[str]        # [G, *specifics], the loading-matrix column order
    spec_factors: list[str]       # specific factors (G excluded), in column order
    g_col: int                    # index of G in factor_cols (0)
    pos_cells: list               # [(j, c, mu, sd)] TruncatedNormal>0 (primary / G-anchor)
    sgn_cells: list               # [(j, c, mu, sd)] Normal (bifactor-G / cross / window)
    correlated: bool              # Phi = LKJ over specifics (else I); the S2 inter-dimension Phi
    cohort: np.ndarray
    index: pd.Index
    kind: dict = field(default_factory=dict)   # {(j, c): "g_anchor|primary|bifactor_G|cross|window"}
    g_correlated: bool = False    # S5 sensitivity: let G correlate with specifics (else G ⊥, bifactor)


def _balanced_idx(cohort: np.ndarray, n_total: int, rng) -> np.ndarray:
    """Indices for a cohort-balanced subsample: ~n_total/n_cohorts per cohort, capped at each
    cohort's size (so small cohorts like DR contribute all their patients). §3.6."""
    cohorts = list(dict.fromkeys(cohort))                  # preserve order, unique
    per = max(1, n_total // len(cohorts))
    out = []
    for c in cohorts:
        ix_c = np.flatnonzero(cohort == c)
        take = min(per, len(ix_c))
        out.append(rng.choice(ix_c, size=take, replace=False))
    return np.sort(np.concatenate(out))


# ----- covariate-adjusted sensitivity arm (issue P0-04) --------------------------------------------
# The published equation adjusts each item mean by β_jᵀ c_i (age, sex, education, site). For a Gaussian
# item this is Frisch–Waugh–Lovell-equivalent to residualizing the item on the covariate design BEFORE
# the factor model — so the marginalized Woodbury kernel is untouched (still zero-mean). Off by default
# (`covariate_adjust=False`): the primary z-scored encoding is byte-for-byte unchanged. The arm is
# scoped to age(spline)+sex+education(edulevel)+site (NOT cohort/diagnosis — those stay metadata /
# invariance grouping, preserving the transdiagnostic between-cohort signal).

def _age_spline(x: np.ndarray, df: int) -> np.ndarray:
    if df and df > 0:
        try:
            from sklearn.preprocessing import SplineTransformer
            return SplineTransformer(n_knots=df, degree=3, include_bias=False).fit_transform(x)
        except Exception:                                     # pragma: no cover (old sklearn fallback)
            return np.column_stack([x, x ** 2, x ** 3])
    return x


def _covariate_design(index: pd.Index, *, age_spline_df: int = 4,
                      extra_cols: tuple[str, ...] = ()) -> np.ndarray:
    """Confounder design aligned to `index`: intercept + ns(age) + age×sex + sex + edulevel + site dummies.
    Covariate NaNs are mean-imputed for the design only (the item's own NaNs are preserved upstream).
    `extra_cols` optionally appends further standardized confounders (e.g. ``on_antipsychotic``, ``bmi``)
    pulled from covariates_v0.parquet — the biology⊥G confound-sensitivity arm; empty by default (no change)."""
    cov_path, site_path = PROC / "covariates_v0.parquet", PROC / "site_v0.parquet"
    cov = pd.read_parquet(cov_path).reindex(index) if cov_path.exists() else pd.DataFrame(index=index)
    n = len(index)

    def _col(name):
        x = pd.to_numeric(cov[name], errors="coerce").to_numpy("float64") if name in cov.columns \
            else np.full(n, np.nan)
        m = float(np.nanmean(x)) if np.isfinite(np.nanmean(x)) else 0.0
        return np.nan_to_num(x, nan=m).reshape(-1, 1)

    age, sex, edu = _col("age"), _col("sex"), _col("edulevel")
    age_basis = _age_spline(age, age_spline_df)
    edu = (edu - edu.mean()) / (edu.std() or 1.0)
    blocks = [np.ones((n, 1)), age_basis, sex, edu, age_basis * sex]   # age×sex = sex-specific age curve
    if site_path.exists():
        site = pd.read_parquet(site_path)["siteid_city"].reindex(index).round().astype("Int64")
        d = pd.get_dummies(site.astype("object"), prefix="site", dummy_na=False, drop_first=True)
        if d.shape[1]:
            blocks.append(d.to_numpy("float64"))
    for name in extra_cols:                            # confound-sensitivity covariates (standardized)
        x = _col(name)
        blocks.append((x - x.mean()) / (x.std() or 1.0))
    return np.column_stack(blocks)


def _residualize_on_covariates(Vdf: pd.DataFrame, *, age_spline_df: int = 4,
                               extra_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """OLS-partial each (pre-z-score) item on the covariate design over its observed rows; NaN preserved."""
    A = _covariate_design(Vdf.index, age_spline_df=age_spline_df, extra_cols=extra_cols)
    out = Vdf.copy()
    min_obs = A.shape[1] + 2
    for c in out.columns:
        y = out[c].to_numpy("float64").copy()
        obs = np.isfinite(y)
        if int(obs.sum()) < min_obs:
            continue
        beta, *_ = np.linalg.lstsq(A[obs], y[obs], rcond=None)
        y[obs] = y[obs] - A[obs] @ beta
        out[c] = y
    return out


def prepare(factors: list[str] = S1_FACTORS, *, correlated: bool = False,
            windows: bool = False, specific_cross: bool = False, g_correlated: bool = False,
            cross_sd_scale: float = 0.25, window_sd_scale: float = 1.0, flat: bool = False,
            bifactor_g_sd: dict[str, float] | None = None,
            cohort_subset: list[str] | None = None, balanced: bool = False,
            keep_index: np.ndarray | None = None, force_factors_continuous: list[str] | None = None,
            n_subsample: int | None = None, emit_moments: bool = False, visit: str = "V0",
            covariate_adjust: bool = False, age_spline_df: int = 4,
            covariate_extra_cols: tuple[str, ...] = (), soft_unlikely: bool = False,
            seed: int = 20260605) -> CorePrep | tuple[CorePrep, dict]:
    """Load the persisted V0 baseline, encode the continuous block for `factors`, and resolve
    the per-cell loading priors from the matrix. Three orthogonal switches deform S1 -> S2:

      * `correlated`  — estimate Phi (LKJ) over the specifics instead of Phi = I. The S2
        inter-dimension correlations; G stays orthogonal (bifactor).
      * `windows`     — add the MADRS/QIDS/STAI window items (no home factor) as signed
        cross-loadings onto G / cognition / sleep. These are the well-identified ESEM
        cross-loadings (a window is not a dimension, so window->factor is a regression onto an
        anchored factor, not a rotation), kept at the full plausible sd (`window_sd_scale`=1).
      * `specific_cross` — free the specific<->specific (all metabolic<->inflammatory) cross-
        loadings. **Off by default at S2:** freeing them *both ways* alongside a free
        Phi_{metab,inflam} is rotationally aliased (it made full-N intractable — NUTS crawls the
        ridge), and they are *not separately identifiable* from Phi. With them off, **Phi carries
        the metabolic/inflammatory association** (the identified estimand). `cross_sd_scale` is the
        ridge guard used only if they are re-enabled (sensitivity arm).

    S1 = all switches off (Phi = I, simple structure). S2 = `correlated=windows=True,
    specific_cross=False`. `n_subsample` (random) is for smoke tests only.
    """
    m = pd.read_csv(MATRIX)
    meta = m.drop_duplicates("item").set_index("item")[
        ["likelihood_family", "modeling_block", "item_sign"]]
    home = (m[m.prior_type.isin(["primary", "g_anchor"])].drop_duplicates("item")
            .set_index("item")["factor"].to_dict())
    cell = {(r.item, r.factor): (r.prior_type, float(r.prior_mean), float(r.prior_sd))
            for r in m.itertuples()}
    spec_factors = [f for f in factors if f != G_KEY]
    factor_cols = [G_KEY] + spec_factors
    col = {f: i for i, f in enumerate(factor_cols)}

    fc = set(force_factors_continuous or [])    # treat these factors' non-Gaussian items as continuous
    items = sorted(it for it in home
                   if home[it] in factors
                   and (meta.loc[it, "modeling_block"] == "continuous" or home[it] in fc))
    use_windows = bool(windows)
    if use_windows:                        # window items: no home, plausible_cross onto our factors
        win = [w for w in WINDOWS if w in meta.index
               and meta.loc[w, "modeling_block"] == "continuous"
               and any(cell.get((w, f), ("", 0, 0))[0] == "plausible_cross" for f in factors)]
        items = sorted(set(items) | set(win))

    B = pd.read_parquet(PROC / f"baseline_{visit.lower()}.parquet")   # V0 default; V1/V2 for the §G1 refit
    items = [it for it in items if it in B.columns]
    if cohort_subset is not None:          # per-cohort fits (§8 invariance): filter rows, z-score WITHIN
        B = B[np.isin(np.asarray(B.index.get_level_values("cohort")), list(cohort_subset))]
    if keep_index is not None:             # explicit row resample (§8 site cluster-bootstrap; dups OK)
        B = B.iloc[keep_index]
    cohort = np.asarray(B.index.get_level_values("cohort"))

    moments = {} if emit_moments else None     # frozen V0 transform for follow-up scoring (M3 §3.1)
    raw, metarec = {}, {}                       # sign-oriented, log-transformed values BEFORE z-scoring
    for it in items:
        v = pd.to_numeric(B[it], errors="coerce").astype(float)
        fam = meta.loc[it, "likelihood_family"]
        logmin = None
        if fam == "lognormal":
            logmin = float(np.nanmin(v.values))
            v = np.log1p(v - logmin + 1e-6) if (np.isfinite(logmin) and logmin <= 0) else np.log(v)
        sgn = int(meta.loc[it, "item_sign"])
        raw[it] = sgn * v
        metarec[it] = (str(fam), sgn, logmin)
    Vdf = pd.DataFrame(raw, index=B.index)
    if covariate_adjust:                        # P0-04 arm: partial out age(spline)+sex+edu+site, then z-score
        Vdf = _residualize_on_covariates(Vdf, age_spline_df=age_spline_df,
                                         extra_cols=covariate_extra_cols)
    cols = {}
    for it in items:
        v = Vdf[it]
        mu, sd = v.mean(), v.std()
        cols[it] = (v - mu) / sd if sd and sd > 0 else v * 0.0
        if moments is not None:
            fam, sgn, logmin = metarec[it]
            moments[it] = (fam, sgn, logmin, float(mu), float(sd))
    Mdf = pd.DataFrame(cols, index=B.index)

    if n_subsample and n_subsample < len(Mdf):
        rng = np.random.default_rng(seed)
        ix = (_balanced_idx(cohort, n_subsample, rng) if balanced
              else np.sort(rng.choice(len(Mdf), size=n_subsample, replace=False)))
        Mdf, cohort = Mdf.iloc[ix], cohort[ix]

    homes = [home.get(it, "") for it in items]
    pos_cells, sgn_cells, kind = [], [], {}
    for j, it in enumerate(items):
        h = homes[j]
        for f in factor_cols:
            c = col[f]
            ptype, mu, sd = cell.get((it, f), ("unlikely_cross", 0.0, 0.05))
            if f == h:                                    # home cell -> positive anchor
                pos_cells.append((j, c, mu, sd))
                kind[(j, c)] = "g_anchor" if f == G_KEY else "primary"
            elif f == G_KEY:                              # bifactor-G cell (signed, sd unscaled)
                if h:                                     # homed specific item -> always (bifactor)
                    gx = cell.get((it, G_KEY), ("plausible_cross", 0.0, 0.25))
                    # bifactor_g_sd tightens the G-loading sd for chosen home factors. Used in the
                    # mixed model (§4.4 rung 3) to tighten the developmental/suicidality items' G-loadings:
                    # their home is an EXPLICIT factor, so a free G-loading makes them load on two
                    # explicit factors (G + home) → a ridge that stalls mixing. dev/suic are ≈⊥G, so
                    # tightening toward 0 removes the ridge without touching the biology⊥G estimand.
                    sd_g = bifactor_g_sd.get(h, gx[2]) if bifactor_g_sd else gx[2]
                    sgn_cells.append((j, c, gx[1], sd_g))
                    kind[(j, c)] = "bifactor_G"
                elif use_windows and ptype == "plausible_cross":   # window on G
                    sgn_cells.append((j, c, mu, sd))
                    kind[(j, c)] = "window"
            elif ptype == "plausible_cross":             # specific-column cross-loading
                if h == "":                              # window -> specific (kept free)
                    if use_windows:
                        sgn_cells.append((j, c, mu, sd * window_sd_scale))
                        kind[(j, c)] = "window"
                elif specific_cross:                     # specific <-> specific (ridge-guarded)
                    sgn_cells.append((j, c, mu, sd * cross_sd_scale))
                    kind[(j, c)] = "cross"
            elif soft_unlikely and ptype == "unlikely_cross":   # P0-05: shrink unlikely cells, not hard-zero
                # The methods doc says "unlikely" cells carry a soft Normal(0, 0.05); the default engine
                # hard-zeros them. This arm instantiates that soft prior on the ~unlikely specific cross
                # cells (NOT specific<->specific plausible_cross, which stay off — rotationally aliased
                # with Φ — and NOT G). For the soft-zero-vs-hard-zero sensitivity table.
                sgn_cells.append((j, c, 0.0, 0.05))
                kind[(j, c)] = "unlikely"

    if flat:                          # §5 confirmation: prior-free (identification-only) priors.
        # Drop the soft-prior informativeness, keep ONLY the identification constraints (home cells
        # stay TruncatedNormal>0 for sign-orientation; signed cells centered at 0). A flat-prior MAP
        # = MLE = FIML (§3.5), so Λ/Φ matching the soft-prior fit ⇒ not a Bayesian-prior artefact.
        pos_cells = [(j, c, 0.0, 5.0) for (j, c, _m, _s) in pos_cells]
        sgn_cells = [(j, c, 0.0, 5.0) for (j, c, _m, _s) in sgn_cells]

    prep = CorePrep(M=Mdf.to_numpy(), items=items, home=homes, factor_cols=factor_cols,
                    spec_factors=spec_factors, g_col=col[G_KEY],
                    pos_cells=pos_cells, sgn_cells=sgn_cells,
                    correlated=bool(correlated), cohort=cohort, index=Mdf.index, kind=kind,
                    g_correlated=bool(g_correlated))
    return (prep, moments) if moments is not None else prep


def _cell_arrays(cells):
    if not cells:
        z = np.zeros(0)
        return z.astype("int64"), z.astype("int64"), z, z
    r = np.array([c[0] for c in cells], "int64")
    cc = np.array([c[1] for c in cells], "int64")
    mu = np.array([c[2] for c in cells], "float64")
    sd = np.array([c[3] for c in cells], "float64")
    return r, cc, mu, sd


def _build_phi(pm, pt, prep, lkj_eta: float):
    """Phi_full [F, F]: G orthogonal (identity row/col); specific block ~ LKJ correlation.
    Returns (Phi, R) where R = chol(Phi) for the Woodbury reparameterization. Phi = I at S1."""
    F = len(prep.factor_cols)
    if not prep.correlated or F <= 2:
        return pt.eye(F), pt.eye(F)
    if getattr(prep, "g_correlated", False):
        # S5 sensitivity: correlate ALL factors incl G (no orthogonality) — reads G's correlation with
        # metabolic/inflammatory (the biology⊥G test). Parameterize the correlation directly as a
        # unit-row lower-triangular Cholesky: free strictly-lower entries, unit diagonal, then normalize
        # each row to unit L2 norm ⇒ L Lᵀ has unit diagonal (a valid correlation, guaranteed PD). This
        # avoids pm.LKJCorr (its n≥5 jitter-init is broken in this stack) AND LKJCholeskyCov's nuisance
        # sd_dist (whose near-zero draws funnel → divergences). Weakly-informative correlation prior.
        tl = np.tril_indices(F, -1)
        lower = pm.Normal("Phi_lower", 0.0, 1.0, shape=len(tl[0]))
        Lr = pt.set_subtensor(pt.eye(F)[tl], lower)
        L = Lr / pt.sqrt((Lr ** 2).sum(1, keepdims=True))
        Phi = L @ L.T
        return Phi, pt.linalg.cholesky(Phi + 1e-8 * pt.eye(F))
    spec_idx = [i for i in range(F) if i != prep.g_col]
    ns = len(spec_idx)
    # PyMC 6.0.1 LKJCorr returns the lower CHOLESKY FACTOR L (unit-norm rows), NOT the correlation
    # matrix — so the correlation is C = L Lᵀ (guaranteed PD, unit diagonal). (Symmetrizing L's
    # lower triangle instead, as a naive read suggests, yields an indefinite matrix → chol(Φ)=NaN.)
    Lcorr = pm.LKJCorr("Phi_spec", n=ns, eta=lkj_eta)
    Cs = Lcorr @ Lcorr.T
    E = np.zeros((F, ns))
    for k, i in enumerate(spec_idx):
        E[i, k] = 1.0
    eg = np.zeros((F, 1)); eg[prep.g_col, 0] = 1.0
    Phi = pt.as_tensor(E) @ Cs @ pt.as_tensor(E).T + pt.as_tensor(eg @ eg.T)
    return Phi, pt.linalg.cholesky(Phi)


def _build_loadings(pm, pt, prep, J, F):
    """Assemble Lam [J, F] from the pos (TruncatedNormal>0) and signed (Normal) cells."""
    pr, pc, pmu, psd = _cell_arrays(prep.pos_cells)
    sr, sc, smu, ssd = _cell_arrays(prep.sgn_cells)
    Lam = pt.zeros((J, F))
    if len(pr):
        vpos = pm.TruncatedNormal("lam_pos", mu=pmu, sigma=psd, lower=0.0, shape=len(pr))
        Lam = pt.set_subtensor(Lam[pr, pc], vpos)
    if len(sr):
        vsgn = pm.Normal("lam_cross", mu=smu, sigma=ssd, shape=len(sr))
        Lam = pt.set_subtensor(Lam[sr, sc], vsgn)
    return Lam


def _patterns(mask: np.ndarray):
    """Unique observed-cell patterns + per-patient index. The per-patient Woodbury matrix A_i
    depends on the row only through its observed pattern, so the (expensive) F×F Cholesky is done
    once per UNIQUE pattern (~half as many as patients) instead of once per patient."""
    pat, inv = np.unique(mask, axis=0, return_inverse=True)
    return pat.astype("float64"), inv.reshape(-1).astype("int64")


def _woodbury_potential(pt, r, mask, Lt, psi, pat_mask, pat_inv, kobs, F, log2pi):
    """Marginal Gaussian log-lik over observed cells, vectorized. Σ = Lt Ltᵀ + diag(psi); per
    patient -0.5[k log2π + logdet Σ_obs + rᵀ Σ_obs⁻¹ r] via the matrix-determinant lemma + Woodbury.

    Two speedups vs the naive per-patient form: (1) A = I + Σ_j mask_j (Lt_j Lt_jᵀ)/psi_j is one BLAS
    GEMM `pat_mask @ Q` (no [N,J,F] materialization); (2) A and its Cholesky are computed per UNIQUE
    pattern, then gathered to patients. `r` is the residual (x for the marginalized model, x−mean for
    the mixed model); mask/pat_mask/pat_inv/kobs are numpy constants; Lt/psi are pytensor."""
    J = pat_mask.shape[1]
    Qf = ((Lt[:, :, None] * Lt[:, None, :]) / psi[:, None, None]).reshape((J, F * F))   # [J, F²]
    A = (pt.eye(F).reshape((1, F * F)) + pt.as_tensor(pat_mask) @ Qf).reshape((pat_mask.shape[0], F, F))
    Lc = pt.linalg.cholesky(A)                                                            # [P, F, F]
    logdetA_p = 2.0 * pt.log(pt.diagonal(Lc, axis1=-2, axis2=-1)).sum(-1)                 # [P]
    logdetPsi_p = (pt.as_tensor(pat_mask) * pt.log(psi)[None, :]).sum(1)                  # [P]
    Wr = pt.as_tensor(mask) * r / psi[None, :]                                            # [N, J]
    b = Wr @ Lt                                                                           # [N, F]
    sol = pt.linalg.solve_triangular(Lc[pat_inv], b[:, :, None], lower=True)[:, :, 0]     # gather chol
    quadA = (sol ** 2).sum(-1)                                                            # [N]
    term1 = (Wr * r).sum(1)                                                               # [N]
    return -0.5 * (pt.as_tensor(kobs) * log2pi + logdetPsi_p[pat_inv] + logdetA_p[pat_inv]
                   + term1 - quadA)


def build_marginalized(prep: CorePrep, psi_floor: float = 0.05, lkj_eta: float = 2.0,
                       weights: np.ndarray | None = None):
    """Marginalized (Woodbury, low-rank) bifactor/ESEM — funnel-free, no per-patient latents.

    Integrates G, D out: each patient's observed cells ~ MVN(0, Lam Phi Lam' + diag(psi)). With
    Lam_tilde = Lam chol(Phi), Sigma = Lam_tilde Lam_tilde' + diag(psi) so the per-patient work is
    the S1 O(F^2) matrix-determinant-lemma + Woodbury kernel (F = 1 + #specifics), fully vectorized
    over patients via a 0/1 mask (no pattern grouping, no patient dropped). Run via
    `pm.sample(nuts_sampler="numpyro")` so JAX vmap-vectorizes the batched F×F linalg.

    `weights` (§3.6): per-patient likelihood weights for the 1/n_cohort-weighted sensitivity fit
    (equalizes each cohort's influence using all patients, instead of subsampling)."""
    import pymc as pm
    import pytensor.tensor as pt

    M = prep.M
    N, J = M.shape
    F = len(prep.factor_cols)
    mask = (~np.isnan(M)).astype("float64")
    x = np.nan_to_num(M, nan=0.0)
    kobs = mask.sum(1)
    log2pi = float(np.log(2.0 * np.pi))
    pat_mask, pat_inv = _patterns(mask)

    with pm.Model() as model:
        Lam = _build_loadings(pm, pt, prep, J, F)
        pm.Deterministic("Lam", Lam)
        Phi, R = _build_phi(pm, pt, prep, lkj_eta)
        pm.Deterministic("Phi", Phi)
        Lt = Lam @ R                                                   # [J, F] reparam loadings
        sigma = psi_floor + pm.HalfNormal("sigma", 1.0, shape=J)
        ll = _woodbury_potential(pt, pt.as_tensor(x), mask, Lt, sigma ** 2,
                                 pat_mask, pat_inv, kobs, F, log2pi)
        obs_ll = (pt.as_tensor(weights) * ll).sum() if weights is not None else ll.sum()
        pm.Potential("obs_ll", obs_ll)
    return model


def build_model(prep: CorePrep, lkj_eta: float = 2.0):
    """Explicit-latent bifactor/ESEM (triangulation arm). Returns a PyMC model with the latents
    instantiated: G ~ N(0,1); specifics non-centered with Cov = Phi_spec. Mathematically the same
    posterior as `build_marginalized`; slower at full N (kept for subsample triangulation)."""
    import pymc as pm
    import pytensor.tensor as pt

    M = prep.M
    N, J = M.shape
    F = len(prep.factor_cols)
    Ks = F - 1
    obs_r, obs_c = np.where(~np.isnan(M))
    y = M[obs_r, obs_c].astype("float64")
    obs_r = obs_r.astype("int64"); obs_c = obs_c.astype("int64")

    with pm.Model() as model:
        Lam = _build_loadings(pm, pt, prep, J, F)
        pm.Deterministic("Lam", Lam)
        Phi, R = _build_phi(pm, pt, prep, lkj_eta)
        pm.Deterministic("Phi", Phi)

        G = pm.Normal("G", 0.0, 1.0, shape=N)                         # G orthogonal to specifics
        z = pm.Normal("z", 0.0, 1.0, shape=(N, Ks))
        if prep.correlated and Ks >= 2:
            # specifics correlated: G is column 0, so R is block-diag [[1,0],[0,Rspec]] and
            # D = z @ Rspec' gives Cov(D rows) = Phi_spec. Phi=I at S1 -> Rspec=I -> D=z.
            D = pm.Deterministic("D", z @ R[1:, 1:].T)
        else:
            D = z

        lamG = Lam[:, prep.g_col]                                      # g_col == 0
        lamS = Lam[:, 1:]                                              # [J, Ks] specifics
        alpha = pm.Normal("alpha", 0.0, 1.5, shape=J)
        sigma = 0.05 + pm.HalfNormal("sigma", 1.0, shape=J)
        eta = alpha[None, :] + G[:, None] * lamG[None, :] + D @ lamS.T
        pm.Normal("y", mu=eta[obs_r, obs_c], sigma=sigma[obs_c], observed=y)
    return model


def warmstart_initvals(prep: CorePrep, from_stage: int = 1, from_items: list[str] | None = None,
                       reports_dir: Path | None = None) -> dict | None:
    """Continuation warm-start (§4.2): seed this stage's loadings/residuals from the previous
    certified stage's posterior, matched by NAME (item, factor) so it survives item reordering and
    added factors. Cells absent in the source (new items/factors) start at their prior mean (0
    for signed cells, the prior mean for positives); Phi is left to the sampler. This puts every
    chain in the previous stage's basin so the new structure deforms from a certified solution.

    `from_stage` selects the source (S2←S1, S3←S2, …); `from_items` is that stage's item order,
    used to map per-item residual scales from its idata."""
    import arviz as az
    rep = reports_dir or REPO / "reports"
    f = rep / f"04_stage{from_stage}_loadings.csv"
    if not f.exists():
        return None
    src = pd.read_csv(f)
    src_load = {(r.item, r.factor): float(r.loading) for r in src.itertuples()}

    lam_pos = np.array([max(0.02, src_load.get((prep.items[j], prep.factor_cols[c]), mu))
                        for (j, c, mu, sd) in prep.pos_cells], dtype="float64")
    lam_cross = np.array([src_load.get((prep.items[j], prep.factor_cols[c]), 0.0)
                          if prep.kind[(j, c)] == "bifactor_G" else 0.0
                          for (j, c, mu, sd) in prep.sgn_cells], dtype="float64")
    init = {"lam_pos": lam_pos, "lam_cross": lam_cross}

    nc = REPO / "results" / "face" / f"stage{from_stage}" / "idata.nc"
    if nc.exists() and from_items is not None:
        try:
            sig = az.from_netcdf(str(nc)).posterior["sigma"].mean(("chain", "draw")).values
            sig_map = {it: float(sig[k]) for k, it in enumerate(from_items) if k < len(sig)}
            # clamp to a sane residual-SD range: a near-zero warm-start sigma (an item the prior
            # stage fit to ~0 residual variance) blows the Woodbury precision to NaN under jitter.
            init["sigma"] = np.clip(np.array([sig_map.get(it, 0.8) for it in prep.items],
                                             dtype="float64"), 0.1, 1.2)
        except Exception:
            pass
    return init


def thomson_scores(prep: CorePrep, Lam: np.ndarray, Phi: np.ndarray,
                   psi: np.ndarray) -> np.ndarray:
    """Post-hoc regression (Thomson) factor scores from the marginalized fit, observed cells only.

    Sigma = Lam Phi Lam' + diag(psi); per observed-pattern B = Phi Lam_obs' Sigma_obs^{-1}; score row
    = (x_obs) B'. Provisional (a checkpoint read-out for stratification later, not a reported S2 claim)."""
    M = prep.M
    N, F = M.shape[0], Lam.shape[1]
    Sig = Lam @ Phi @ Lam.T + np.diag(psi)
    out = np.full((N, F), np.nan)
    pat: dict = {}
    for i in range(N):
        o = tuple(np.flatnonzero(~np.isnan(M[i])))
        if o:
            pat.setdefault(o, []).append(i)
    for o, rows in pat.items():
        oi = list(o)
        B = Phi @ Lam[oi].T @ np.linalg.pinv(Sig[np.ix_(oi, oi)])
        out[rows] = np.nan_to_num(M[np.ix_(rows, oi)]) @ B.T
    return out


# =========================== S3b — mixed-likelihood block ============================
# Adds the binary/count/ordinal suicidality + developmental indicators, which cannot be
# marginalized. Architecture (methods doc §3.2/§4.4): the factors touching non-Gaussian
# indicators — G + suicidality + developmental — get EXPLICIT non-centered latents f_e;
# the pure-continuous specifics (cognition/metabolic/inflammatory/sleep) stay MARGINALIZED
# as f_m, coupled to f_e through the shared Phi by the conditional decomposition
#     f_m | f_e ~ N(M f_e, S),  M = Phi_me Phi_ee^{-1},  S = Phi_mm - Phi_me Phi_ee^{-1} Phi_em.
# The continuous block is then a per-patient-mean Woodbury (mean B f_e, residual cov Λ_m S Λ_mᵀ+Ψ);
# the non-Gaussian indicators get Bernoulli / NegBin / ordered-logistic likelihoods on f_e.
EXPLICIT_FACTORS = ["overall_severity", "suicidality", "developmental_risk"]


@dataclass
class MixedPrep:
    base: CorePrep                 # continuous block (S3a prep): M, pos/sgn cells, factor_cols, Phi
    e_cols: list[int]              # explicit-factor columns in base.factor_cols  [G, suic, dev]
    m_cols: list[int]              # marginalized-factor columns                   [cog, met, inf, sleep]
    bin_items: list[str]
    Bin: np.ndarray                # [N, Jb] 0/1, NaN = missing
    ord_items: list[str]
    Ord: np.ndarray                # [N, Jo] 0..K-1, NaN = missing
    ord_K: list[int]
    cnt_items: list[str]
    Cnt: np.ndarray                # [N, Jcnt] counts, NaN = missing
    ng_home: dict                  # {item: explicit-col index of its home (1=suic, 2=dev)}
    ng_hp: dict                    # {item: (home_mu, home_sd)} primary prior (TruncatedNormal>0)
    ng_gp: dict                    # {item: (g_mu, g_sd)} bifactor-G prior (signed)


def prepare_mixed(factors: list[str] = S3_FACTORS, *, min_obs: int = 1500,
                  bifactor_g_sd: dict[str, float] | None = None, balanced: bool = False,
                  explicit_factors: list[str] | None = None, min_cohorts: int = 3,
                  cohort_subset: list[str] | None = None,
                  n_subsample: int | None = None, seed: int = 20260605) -> MixedPrep:
    """S3b inputs: the S3a continuous prep + the non-Gaussian (binary/ordinal/count) suicidality
    and developmental indicators, aligned to the same patients. Coverage filter: an indicator must
    be observed in all three cohorts with >= `min_obs` total (drops the sparse BP/DR-only C-SSRS/LTS
    and dr=0 items, which cannot identify a loading).

    `bifactor_g_sd` (§4.4 rung 3): tighten the G-loadings of items whose home is an EXPLICIT factor
    (suicidality, developmental_risk) — a free G-loading there makes them load on two explicit factors,
    a ridge that stalls mixing (the CTQ→G cells: ESS 30). Default tightens both toward 0 (they are ≈⊥G),
    which leaves the biology→G estimand untouched."""
    explicit_factors = explicit_factors or EXPLICIT_FACTORS
    if bifactor_g_sd is None:                                          # tighten every explicit specific →G
        bifactor_g_sd = {f: 0.05 for f in explicit_factors if f != G_KEY}
    base = prepare(factors, correlated=True, windows=True, bifactor_g_sd=bifactor_g_sd,
                   balanced=balanced, cohort_subset=cohort_subset, n_subsample=n_subsample, seed=seed)
    m = pd.read_csv(MATRIX)
    meta = m.drop_duplicates("item").set_index("item")[["modeling_block", "likelihood_family"]]
    home = (m[m.prior_type.isin(["primary", "g_anchor"])].drop_duplicates("item")
            .set_index("item")["factor"].to_dict())
    cell = {(r.item, r.factor): (float(r.prior_mean), float(r.prior_sd)) for r in m.itertuples()}
    e_idx = {f: i for i, f in enumerate(explicit_factors)}            # G=0, suic=1, dev=2, [substance=3]
    e_cols = [base.factor_cols.index(f) for f in explicit_factors]
    m_cols = [base.factor_cols.index(f) for f in base.factor_cols if f not in explicit_factors]

    B_full = pd.read_parquet(PROC / "baseline_v0.parquet")             # FULL data for eligibility
    coh_full = np.asarray(B_full.index.get_level_values("cohort"))
    ng = [f for f in explicit_factors if f != G_KEY]                  # non-Gaussian-bearing factors
    items = [it for it in home if home.get(it) in ng and it in meta.index
             and meta.loc[it, "modeling_block"] == "explicit" and it in B_full.columns]

    def covered(it: str) -> bool:                                      # full-N coverage (subsample-independent)
        # ≥ min_obs total AND observed in ≥ min_cohorts cohorts. min_cohorts=2 admits the BP/SZ-only
        # substance SUD items (DR=0) as legitimate 2-cohort indicators (observed-likelihood handles the
        # absent cohort); min_cohorts=3 (default) keeps the suicidality/developmental block as before.
        v = pd.to_numeric(B_full[it], errors="coerce")
        return v.notna().sum() >= min_obs and sum((v[coh_full == c].notna().sum()) > 0
                                                  for c in ("bp", "sz", "dr")) >= min_cohorts
    items = [it for it in items if covered(it)]
    B = B_full.loc[base.index]                                         # arrays on the (sub)sampled rows
    fam = {it: meta.loc[it, "likelihood_family"] for it in items}
    bin_items = sorted(it for it in items if fam[it] == "bernoulli")
    ord_items = sorted(it for it in items if fam[it] == "ordered_logistic")
    cnt_items = sorted(it for it in items if fam[it] == "neg_binomial")

    def grab(cols):
        return pd.DataFrame({c: pd.to_numeric(B[c], errors="coerce") for c in cols},
                            index=B.index).to_numpy().astype(float)

    Bin, Cnt, Ov = grab(bin_items), grab(cnt_items), grab(ord_items)
    ord_K = []
    for k in range(Ov.shape[1]):                                       # recode each ordinal to 0..K-1
        col = Ov[:, k]; obs = ~np.isnan(col); uniq = np.unique(col[obs])
        remap = {v: i for i, v in enumerate(uniq)}
        for v, i in remap.items():
            col[col == v] = i
        Ov[:, k] = col; ord_K.append(max(2, len(uniq)))

    ng_home, ng_hp, ng_gp = {}, {}, {}
    for it in bin_items + ord_items + cnt_items:
        ng_home[it] = e_idx[home[it]]
        ng_hp[it] = cell.get((it, home[it]), (0.7, 0.25))
        ng_gp[it] = cell.get((it, "overall_severity"), (0.0, 0.25))
    return MixedPrep(base=base, e_cols=e_cols, m_cols=m_cols,
                     bin_items=bin_items, Bin=Bin, ord_items=ord_items, Ord=Ov, ord_K=ord_K,
                     cnt_items=cnt_items, Cnt=Cnt, ng_home=ng_home, ng_hp=ng_hp, ng_gp=ng_gp)


def _sel(rows: list[int], F: int) -> np.ndarray:
    S = np.zeros((len(rows), F))
    for k, i in enumerate(rows):
        S[k, i] = 1.0
    return S


def _hurdle_nb_logp(pt, y, psi, mu, alpha):
    """Hurdle-NB log-likelihood from differentiable ops only — NO betainc, so it works under the
    numpyro/JAX NUTS sampler (``pm.HurdleNegativeBinomial`` uses NB.logcdf → betainc, whose gradient
    JAX does not support). ``psi`` = P(Y>0): zeros are the gate ``log(1-psi)``; positives are a
    zero-truncated NB. NB(0) = (α/(α+μ))^α; the truncation divides by 1-NB(0) (log1mexp)."""
    log_p0 = alpha * (pt.log(alpha) - pt.log(alpha + mu))                 # log NB(0)
    nb_lpmf = (pt.gammaln(y + alpha) - pt.gammaln(alpha) - pt.gammaln(y + 1.0)
               + alpha * (pt.log(alpha) - pt.log(alpha + mu))
               + y * (pt.log(mu) - pt.log(alpha + mu)))
    pos = pt.log(psi) + nb_lpmf - pt.log1mexp(log_p0)                     # zero-truncated NB, scaled by psi
    return pt.where(pt.eq(y, 0.0), pt.log1p(-psi), pos)


def build_mixed(mp: MixedPrep, psi_floor: float = 0.05, lkj_eta: float = 2.0,
                hurdle_counts: bool = False):
    """Hybrid explicit/marginalized mixed-likelihood model (S3b). f_e=(G,suic,dev) explicit; the
    continuous specifics marginalized and coupled via the conditional Phi decomposition."""
    import pymc as pm
    import pytensor.tensor as pt

    base = mp.base
    M = base.M
    N, Jc = M.shape
    F = len(base.factor_cols)
    Ke, Km = len(mp.e_cols), len(mp.m_cols)
    mask = (~np.isnan(M)).astype("float64")
    x = np.nan_to_num(M, nan=0.0)
    kobs = mask.sum(1)
    log2pi = float(np.log(2.0 * np.pi))
    pat_mask, pat_inv = _patterns(mask)
    Se, Sm = _sel(mp.e_cols, F), _sel(mp.m_cols, F)                    # selection matrices

    with pm.Model() as model:
        Lam = _build_loadings(pm, pt, base, Jc, F)                     # [Jc, F] continuous loadings
        pm.Deterministic("Lam", Lam)
        Phi, _ = _build_phi(pm, pt, base, lkj_eta)                     # [F, F] corrected, G orthogonal
        pm.Deterministic("Phi", Phi)

        Se_t, Sm_t = pt.as_tensor(Se), pt.as_tensor(Sm)
        Phi_ee = Se_t @ Phi @ Se_t.T                                   # [Ke, Ke]
        Phi_mm = Sm_t @ Phi @ Sm_t.T                                   # [Km, Km]
        Phi_me = Sm_t @ Phi @ Se_t.T                                   # [Km, Ke]
        Mmat = pt.linalg.solve(Phi_ee, Phi_me.T).T                     # [Km, Ke] = Phi_me Phi_ee^{-1}
        S = Phi_mm - Mmat @ Phi_me.T                                   # [Km, Km] residual cov
        C_S = pt.linalg.cholesky(S + 1e-8 * pt.eye(Km))
        L_ee = pt.linalg.cholesky(Phi_ee + 1e-8 * pt.eye(Ke))

        Lam_e = Lam @ Se_t.T                                           # [Jc, Ke]
        Lam_m = Lam @ Sm_t.T                                           # [Jc, Km]
        Bmat = Lam_e + Lam_m @ Mmat                                    # [Jc, Ke] mean loadings on f_e
        Lt = Lam_m @ C_S                                              # [Jc, Km] residual loadings

        z = pm.Normal("z_e", 0.0, 1.0, shape=(N, Ke))                  # explicit latents, non-centered
        f_e = pm.Deterministic("f_e", z @ L_ee.T)                      # Cov(rows) = Phi_ee

        sigma = psi_floor + pm.HalfNormal("sigma", 1.0, shape=Jc)
        # continuous block — per-patient-mean masked Woodbury on residual r = x - f_e Bᵀ
        r = pt.as_tensor(x) - f_e @ Bmat.T                            # [N, Jc]
        ll = _woodbury_potential(pt, r, mask, Lt, sigma ** 2, pat_mask, pat_inv, kobs, Km, log2pi)
        pm.Potential("cont_ll", ll.sum())

        # non-Gaussian indicators on f_e (home factor + bifactor-G), observed cells only
        for k, it in enumerate(mp.bin_items):
            y = mp.Bin[:, k]; obs = np.flatnonzero(~np.isnan(y))
            a = pm.Normal(f"a_{it}", 0.0, 1.5)
            lh = pm.TruncatedNormal(f"lh_{it}", mu=mp.ng_hp[it][0], sigma=mp.ng_hp[it][1], lower=0.0)
            lg = pm.Normal(f"lg_{it}", mp.ng_gp[it][0], mp.ng_gp[it][1])
            eta = a + lh * f_e[:, mp.ng_home[it]][obs] + lg * f_e[:, 0][obs]
            pm.Bernoulli(f"y_{it}", logit_p=eta, observed=y[obs].astype("int8"))
        for k, it in enumerate(mp.cnt_items):
            y = mp.Cnt[:, k]; obs = np.flatnonzero(~np.isnan(y))
            fh = f_e[:, mp.ng_home[it]][obs]
            a = pm.Normal(f"a_{it}", 0.0, 1.5)
            lh = pm.TruncatedNormal(f"lh_{it}", mu=mp.ng_hp[it][0], sigma=mp.ng_hp[it][1], lower=0.0)
            lg = pm.Normal(f"lg_{it}", mp.ng_gp[it][0], mp.ng_gp[it][1])
            alpha = pm.HalfNormal(f"alpha_{it}", 2.0)            # NB concentration (the reported fit's prior)
            eta = a + lh * fh + lg * f_e[:, 0][obs]
            if hurdle_counts:
                # OPT-IN SENSITIVITY (off by default; the reported map uses plain NB). A hurdle separates
                # the zero spike from the count process so the structural zeros stop being NB draws that
                # over-predict the high tail (isf09a item-level fix: plain NB predicted mean 13.4 vs obs
                # 0.14). psi = P(Y>0) is a FREE per-item probability — deliberately NOT latent-coupled (a
                # psi = sigmoid(a + λ·f_home) double-loads the count item on its factor → a ridge, R-hat
                # 1.56). Even decoupled, however, perturbing isf09a's likelihood destabilizes the fragile
                # suicidality↔developmental Φ cell for some seeds (re-fit seed-1 R-hat 1.55 vs the plain-NB
                # 1.01), so it is NOT adopted as primary — it trades a cosmetic item-level PPC fix for a
                # structural-correlation's convergence. The suicidality factor is carried by its 7 binary
                # ISF items (all reproduce in PPC); isf09a is a thin item-level contributor.
                apsi = pm.Normal(f"apsi_{it}", 0.0, 1.5)
                psi = pm.Deterministic(f"psi_{it}", pm.math.sigmoid(apsi))
                yv = pt.as_tensor(np.rint(y[obs]).astype("float64"))
                pm.Potential(f"y_{it}", _hurdle_nb_logp(pt, yv, psi, pt.exp(eta), alpha).sum())
            else:
                pm.NegativeBinomial(f"y_{it}", mu=pt.exp(eta), alpha=alpha,
                                    observed=np.rint(y[obs]).astype("int64"))
        for k, it in enumerate(mp.ord_items):
            y = mp.Ord[:, k]; obs = np.flatnonzero(~np.isnan(y)); K = int(mp.ord_K[k])
            cut = pm.Normal(f"c_{it}", mu=np.linspace(-1.5, 1.5, K - 1), sigma=2.0, shape=K - 1,
                            transform=pm.distributions.transforms.ordered)
            lh = pm.TruncatedNormal(f"lh_{it}", mu=mp.ng_hp[it][0], sigma=mp.ng_hp[it][1], lower=0.0)
            lg = pm.Normal(f"lg_{it}", mp.ng_gp[it][0], mp.ng_gp[it][1])
            eta = lh * f_e[:, mp.ng_home[it]][obs] + lg * f_e[:, 0][obs]
            pm.OrderedLogistic(f"y_{it}", eta=eta, cutpoints=cut,
                               observed=y[obs].astype("int32"), compute_p=False)
    return model
