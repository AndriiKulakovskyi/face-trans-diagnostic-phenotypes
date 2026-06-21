"""OOP QA distribution report for the FACE modeled indicators.

Recreates (and modernizes) the deleted ``scripts/qa_harmonization.py`` distribution view: a
self-contained HTML report with, per modeled indicator, a cross-cohort raw histogram AND the
rank-INT (empirical-CDF) Gaussianized marginal overlaid on N(0,1) — so the form of the *true*
data distribution and its Gaussianizability are both visible.

Each indicator is given a NAMED empirical form (binary / ordinal_K / count / count_zero_inflated /
continuous_symmetric / continuous_right_skewed / continuous_left_skewed / continuous_heavy_tailed /
semicontinuous / degenerate), reported alongside the declared likelihood family and the recommended
copula tier (gaussianize / keep_binary / keep_ordinal / keep_count) — the same tiering the
``gaussian_copula`` likelihood vertical of the measurement model uses.

Important framing: a Gaussianized *marginal* is necessary but NOT sufficient for the marginalized
(Woodbury) factor model — that needs residuals approximately Gaussian *given the latents* (a Gaussian
copula / joint-MVN assumption), which is validated post-fit by residual-normality / PPC, not here.

    from face.reporting.distribution_report import DistributionReport
    rep = DistributionReport()
    rep.run()   # -> results/reports/qa_distributions.html + reports/qa_distributions_summary.csv
"""
from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
PROC = REPO / "data" / "processed"
MATRIX = REPO / "configs" / "prior_loading_matrix_v3.csv"
RESULTS_HTML = REPO / "results" / "reports" / "qa_distributions.html"
SUMMARY_CSV = REPO / "reports" / "qa_distributions_summary.csv"

COHORT_COLOR = {"bp": "#3b6fb6", "sz": "#cf6679", "dr": "#1a7f37"}

# Tiering thresholds — kept in sync with MeasurementConfig.copula_min_distinct / copula_max_modal_frac
# so the report's "recommended tier" matches what the gaussian_copula vertical actually does.
COPULA_MIN_DISTINCT = 8
COPULA_MAX_MODAL_FRAC = 0.5
DISCRETE_PLOT_MAX = 12  # <= this many distinct values -> grouped-bar plot, else density histogram


def rank_int(x: np.ndarray) -> np.ndarray:
    """Rank-based inverse-normal (the nonparametric Gaussian-copula marginal transform):
    u = rank/(n+1), z = Phi^-1(u). Average ranks for ties. ``x`` is observed (no NaN)."""
    x = np.asarray(x, dtype="float64")
    n = x.size
    if n == 0:
        return x
    r = stats.rankdata(x)  # average ranks handle ties
    return stats.norm.ppf(r / (n + 1.0))


def _is_integer_valued(x: np.ndarray) -> bool:
    return bool(x.size) and bool(np.all(np.isfinite(x))) and bool(np.allclose(x, np.round(x)))


def _classify_form(x: np.ndarray, n_distinct: int, frac_zero: float, skew: float, kurt: float) -> str:
    """Name the empirical distribution form from the observed values."""
    if n_distinct <= 1:
        return "degenerate"
    if n_distinct == 2:
        return "binary"
    integer = _is_integer_valued(x)
    if integer and n_distinct <= DISCRETE_PLOT_MAX:
        if x.min() >= 0 and frac_zero >= 0.5:
            return "count_zero_inflated"
        return f"ordinal_{n_distinct}"
    if integer and x.min() >= 0 and frac_zero >= 0.30:
        return "count_zero_inflated"
    if integer and x.min() >= 0:
        return "count"
    if frac_zero >= 0.40:
        return "semicontinuous"  # spike at zero + continuous tail
    if kurt >= 3.0:
        return "continuous_heavy_tailed"
    if skew >= 0.75:
        return "continuous_right_skewed"
    if skew <= -0.75:
        return "continuous_left_skewed"
    return "continuous_symmetric"


def _recommend_tier(family: str, n_distinct: int, modal_frac: float) -> str:
    """The copula tier this indicator would get in the gaussian_copula vertical."""
    if family in ("gaussian", "lognormal", "student_t"):
        return "gaussianize"
    if n_distinct <= 2 or family == "bernoulli":
        return "keep_binary"
    if family in ("ordered_logistic", "neg_binomial"):
        if n_distinct >= COPULA_MIN_DISTINCT and modal_frac < COPULA_MAX_MODAL_FRAC:
            return "gaussianize"
        return "keep_ordinal" if family == "ordered_logistic" else "keep_count"
    return "keep_native"


def _normaltest_p(x: np.ndarray) -> float:
    """D'Agostino K^2 normality p-value (needs >= 8 obs); NaN if not computable."""
    x = x[np.isfinite(x)]
    if x.size < 8 or np.allclose(x, x[0]):
        return float("nan")
    try:
        return float(stats.normaltest(x).pvalue)
    except Exception:
        return float("nan")


@dataclass
class IndicatorDistribution:
    item: str
    home_factor: str
    family: str
    modeling_block: str
    item_sign: int
    n_obs: int
    per_cohort_obs: dict[str, int]
    frac_missing: float
    n_distinct: int
    modal_frac: float
    frac_zero: float
    minimum: float
    maximum: float
    mean: float
    sd: float
    median: float
    skew: float
    kurtosis: float
    normaltest_p_raw: float
    normaltest_p_yeojohnson: float
    empirical_form: str
    recommended_tier: str
    png_b64: str | None = field(default=None, repr=False)

    def summary_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("png_b64", "per_cohort_obs")}
        for c in ("bp", "sz", "dr"):
            d[f"n_{c}"] = self.per_cohort_obs.get(c, 0)
        return d


class DistributionReport:
    """Read modeled indicators, characterize each one's empirical distribution, and render a
    self-contained HTML report + a committable aggregate summary CSV."""

    def __init__(
        self,
        processed_dir: Path = PROC,
        prior_matrix: Path = MATRIX,
        html_path: Path = RESULTS_HTML,
        summary_csv: Path = SUMMARY_CSV,
    ):
        self.processed_dir = Path(processed_dir)
        self.prior_matrix = Path(prior_matrix)
        self.html_path = Path(html_path)
        self.summary_csv = Path(summary_csv)
        self.baseline: pd.DataFrame | None = None
        self.meta: pd.DataFrame | None = None
        self.cohort: np.ndarray | None = None
        self.bounds: dict[str, tuple[float | None, float | None]] = {}

    # ---- data ----
    def load(self) -> DistributionReport:
        self.baseline = pd.read_parquet(self.processed_dir / "baseline_v0.parquet")
        self.cohort = np.asarray(self.baseline.index.get_level_values("cohort"))
        meta_path = self.processed_dir / "indicator_metadata.parquet"
        if meta_path.exists():
            self.meta = pd.read_parquet(meta_path).set_index("item")
        else:  # derive from the prior matrix
            m = pd.read_csv(self.prior_matrix)
            home = (m[m.prior_type.isin(["primary", "g_anchor"])].drop_duplicates("item")
                    .set_index("item")["factor"])
            self.meta = (m.drop_duplicates("item").set_index("item")
                         [["likelihood_family", "modeling_block", "item_sign"]].copy())
            self.meta["home_factor"] = home
        self._load_bounds()
        return self

    def _load_bounds(self) -> None:
        """Optional sanity bounds (for plot reference lines); silently skip if the dictionary
        is unavailable."""
        try:
            from face.data import load_variables  # noqa: PLC0415
            for v in load_variables(str(REPO / "data" / "face-common-vars.xlsx")):
                name = getattr(v, "canonical_name", None)
                if name:
                    self.bounds[name] = (getattr(v, "sanity_min", None), getattr(v, "sanity_max", None))
        except Exception:
            self.bounds = {}

    # ---- per-indicator analysis ----
    def analyze(self, item: str) -> IndicatorDistribution:
        assert self.baseline is not None and self.meta is not None
        raw = pd.to_numeric(self.baseline[item], errors="coerce").astype("float64")
        sign = int(self.meta.loc[item, "item_sign"]) if "item_sign" in self.meta.columns else 1
        x = raw.to_numpy()
        obs = x[np.isfinite(x)]
        n = obs.size
        per_cohort = {c: int(np.isfinite(x[self.cohort == c]).sum()) for c in ("bp", "sz", "dr")}
        if n == 0:
            fam = str(self.meta.loc[item, "likelihood_family"])
            home = str(self.meta.get("home_factor", pd.Series()).get(item, ""))
            return IndicatorDistribution(item, home, fam, str(self.meta.loc[item, "modeling_block"]),
                                         sign, 0, per_cohort, 1.0, 0, 1.0, 0.0, *(float("nan"),) * 7,
                                         float("nan"), float("nan"), "degenerate", "keep_native")
        vals, counts = np.unique(obs, return_counts=True)
        n_distinct = int(vals.size)
        modal_frac = float(counts.max() / n)
        frac_zero = float(np.mean(obs == 0.0))
        sk = float(stats.skew(obs)) if n_distinct > 1 else 0.0
        ku = float(stats.kurtosis(obs)) if n_distinct > 1 else 0.0  # excess kurtosis
        fam = str(self.meta.loc[item, "likelihood_family"])
        home = str(self.meta.get("home_factor", pd.Series(dtype=object)).get(item, "") or "")
        block = str(self.meta.loc[item, "modeling_block"])
        # parametric Gaussianization quality (Yeo-Johnson handles zeros/negatives)
        p_yj = float("nan")
        if n_distinct > 2:
            try:
                p_yj = _normaltest_p(stats.yeojohnson(obs)[0])
            except Exception:
                p_yj = float("nan")
        return IndicatorDistribution(
            item=item, home_factor=home, family=fam, modeling_block=block, item_sign=sign,
            n_obs=n, per_cohort_obs=per_cohort, frac_missing=float(1 - n / len(x)),
            n_distinct=n_distinct, modal_frac=modal_frac, frac_zero=frac_zero,
            minimum=float(obs.min()), maximum=float(obs.max()), mean=float(obs.mean()),
            sd=float(obs.std()), median=float(np.median(obs)), skew=sk, kurtosis=ku,
            normaltest_p_raw=_normaltest_p(obs), normaltest_p_yeojohnson=p_yj,
            empirical_form=_classify_form(obs * sign, n_distinct, frac_zero, sk * sign, ku),
            recommended_tier=_recommend_tier(fam, n_distinct, modal_frac),
            png_b64=self._plot_png(item, raw, sign),
        )

    def _plot_png(self, item: str, raw: pd.Series, sign: int) -> str | None:
        x = raw.to_numpy()
        per = {c: x[(self.cohort == c) & np.isfinite(x)] for c in ("bp", "sz", "dr")}
        per = {c: v for c, v in per.items() if v.size}
        if not per:
            return None
        allvals = np.concatenate(list(per.values()))
        n_unique = int(np.unique(allvals).size)
        fig, (axr, axg) = plt.subplots(1, 2, figsize=(7.4, 2.7))
        # --- panel 1: raw, per cohort ---
        if n_unique <= DISCRETE_PLOT_MAX:
            cats = np.unique(allvals)
            w = 0.8 / max(len(per), 1)
            for i, (c, v) in enumerate(per.items()):
                props = pd.Series(v).value_counts(normalize=True).reindex(cats, fill_value=0)
                axr.bar(np.arange(len(cats)) + i * w, props.to_numpy(), w, alpha=0.85,
                        color=COHORT_COLOR[c], label=f"{c} (n={v.size})")
            axr.set_xticks(np.arange(len(cats)) + w * (len(per) - 1) / 2)
            axr.set_xticklabels([f"{v:g}" for v in cats], fontsize=6)
            axr.set_ylabel("proportion", fontsize=8)
        else:
            lo, hi = np.nanquantile(allvals, 0.005), np.nanquantile(allvals, 0.995)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(allvals.min()), float(allvals.max()) + 1e-9
            bins = np.linspace(lo, hi, 31)
            for c, v in per.items():
                axr.hist(np.clip(v, lo, hi), bins=bins, density=True, histtype="stepfilled",
                         alpha=0.45, color=COHORT_COLOR[c], label=f"{c} (n={v.size})")
            axr.set_ylabel("density", fontsize=8)
            for b in self.bounds.get(item, (None, None)):
                if b is not None:
                    axr.axvline(b, color="#555", ls="--", lw=0.8)
        axr.set_title(f"{item} — raw", fontsize=8)
        axr.legend(fontsize=6, frameon=False)
        axr.tick_params(labelsize=6)
        # --- panel 2: rank-INT Gaussianized marginal vs N(0,1) ---
        z = rank_int(sign * allvals)
        axg.hist(z, bins=31, density=True, histtype="stepfilled", alpha=0.5, color="#666")
        grid = np.linspace(-3.5, 3.5, 200)
        axg.plot(grid, stats.norm.pdf(grid), color="#cf222e", lw=1.2)
        axg.set_title("rank-INT (z) vs N(0,1)", fontsize=8)
        axg.tick_params(labelsize=6)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # ---- assembly ----
    def analyze_all(self) -> list[IndicatorDistribution]:
        assert self.baseline is not None and self.meta is not None
        items = [it for it in self.meta.index if it in self.baseline.columns]
        return [self.analyze(it) for it in items]

    def to_frame(self, dists: list[IndicatorDistribution] | None = None) -> pd.DataFrame:
        dists = dists or self.analyze_all()
        df = pd.DataFrame([d.summary_row() for d in dists])
        sort_cols = [c for c in ("home_factor", "family", "item") if c in df.columns]
        return df.sort_values(sort_cols).reset_index(drop=True)

    def render_html(self, dists: list[IndicatorDistribution], path: Path | None = None) -> Path:
        path = Path(path or self.html_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self.baseline) if self.baseline is not None else 0
        tiers = pd.Series([d.recommended_tier for d in dists]).value_counts().to_dict()
        out = ["<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
               "<title>FACE — indicator distributions (QA)</title>",
               f"<style>{_CSS}</style></head><body>",
               "<header><h1>FACE modeled-indicator distributions</h1>",
               f"<div class='muted'>N = {n:,} patients · {len(dists)} indicators · "
               f"recommended tiers: {html.escape(str(tiers))}</div>",
               "<div class='note'>Each card: <b>left</b> = raw per-cohort distribution (its named "
               "empirical form); <b>right</b> = rank-INT Gaussianized marginal vs N(0,1). "
               "A Gaussianized marginal is necessary but <i>not</i> sufficient for the marginalized "
               "factor model — conditional/residual Gaussianity is checked post-fit (PPC).</div>"
               "</header><div class='cards'>"]
        for d in sorted(dists, key=lambda x: (x.home_factor, x.family, x.item)):
            tier_cls = {"gaussianize": "ok", "keep_binary": "bad", "keep_ordinal": "warn",
                        "keep_count": "warn", "keep_native": "warn"}.get(d.recommended_tier, "warn")
            img = (f"<img alt='{html.escape(d.item)}' src='data:image/png;base64,{d.png_b64}'/>"
                   if d.png_b64 else "<div class='muted'>no observed values</div>")
            out.append(
                f"<div class='card'><h3>{html.escape(d.item)} "
                f"<span class='pill {tier_cls}'>{d.recommended_tier}</span></h3>{img}"
                f"<div class='meta'>form: <b>{d.empirical_form}</b> · declared: {d.family} "
                f"· home: {d.home_factor or '—'} · block: {d.modeling_block}</div>"
                f"<div class='meta'>n={d.n_obs:,} (miss {d.frac_missing:.0%}) · distinct={d.n_distinct} "
                f"· modal={d.modal_frac:.0%} · zero={d.frac_zero:.0%}</div>"
                f"<div class='meta'>skew={d.skew:.2f} · kurt={d.kurtosis:.2f} · "
                f"normal p(raw)={d.normaltest_p_raw:.1e} · p(YJ)={d.normaltest_p_yeojohnson:.1e}</div>"
                "</div>")
        out.append("</div></body></html>")
        path.write_text("\n".join(out))
        return path

    def save_csv(self, dists: list[IndicatorDistribution], path: Path | None = None) -> Path:
        path = Path(path or self.summary_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame(dists).to_csv(path, index=False)
        return path

    def run(self) -> dict[str, str]:
        """Load → analyze all → write HTML + summary CSV. Returns the output paths."""
        self.load()
        dists = self.analyze_all()
        html_p = self.render_html(dists)
        csv_p = self.save_csv(dists)
        return {"html": str(html_p), "summary_csv": str(csv_p), "n_indicators": str(len(dists))}


_CSS = """
*{box-sizing:border-box} body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;
color:#1a1a1a;background:#fafafa} header{background:#fff;border-bottom:1px solid #e5e5e5;
padding:16px 26px;position:sticky;top:0;z-index:5} h1{margin:0 0 4px;font-size:20px}
.muted{color:#777;font-size:12px} .note{font-size:12px;color:#444;margin-top:8px;max-width:1100px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px;padding:18px 26px}
.card{background:#fff;border:1px solid #e5e5e5;border-radius:8px;padding:12px}
.card h3{margin:0 0 6px;font-size:13px;font-family:ui-monospace,Menlo,monospace}
.card img{width:100%;height:auto;border:1px solid #f0f0f0;border-radius:4px}
.meta{font-size:11.5px;color:#333;margin-top:5px;font-family:ui-monospace,Menlo,monospace}
.pill{font-size:10px;font-weight:700;padding:1px 7px;border-radius:9px;color:#fff;vertical-align:middle}
.ok{background:#1a7f37} .bad{background:#cf222e} .warn{background:#9a6700}
"""
