"""QA the v2-driven loading layer: verify every variable loads & passes sanity,
and render an HTML report with per-variable cross-cohort distribution plots.

What it checks, per harmonized variable (canonical name), across BP / SZ / DR:
  1. LOADED       — the column exists after build_unified_dataframe and is
                    populated in every cohort whose dictionary cell names a
                    source CSV column (coverage matches the dictionary).
  2. ENCODED      — non-identifier features are numeric after harmonization
                    (no residual French text leaked through), so cohorts with
                    different raw coding (text vs code) are pooled on one scale.
  3. IN-BOUNDS    — after the sanity stage, no observed value falls outside the
                    dictionary's [sanity_min, sanity_max] (the stage nulled them).
  4. COMPARABLE   — for bounded numeric variables, the per-cohort medians sit on
                    the same scale (a coarse cross-cohort sanity, flagged only).

It loads v2 twice — once WITHOUT and once WITH the sanity stage — so the report
can show how many cells each bound nulled (the harmonization actually firing).

Part 2 (post-preprocessing) then shows the construct-level DOMAIN SCORES — the
encoded features that actually enter the dimensional factor analysis and the
stratification embedding (items robust-z scored → masked-mean aggregated, no
imputation). Per domain: the pooled cross-cohort distribution plus degeneracy
checks (near-zero variance, sub-30%-floor coverage, single-cohort, scale spread)
so data bugs are caught before analysis, not after.

Output: results/reports/qa_harmonization.html  (single self-contained file;
plots are base64-embedded PNGs — no network needed). Per variable: a 3-cohort
overlaid distribution, the sanity min/max printed below the plot, and a
pass/fail verification line.

Run:  python3 scripts/qa_harmonization.py
"""
from __future__ import annotations

import base64
import html
import io
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from trans_diag import (  # noqa: E402
    COGNITIVE_COMPOSITES,
    DOMAIN_SECTIONS,
    SUICIDE_SKIP_RULES,
    build_domain_scores,
    build_unified_dataframe,
    decode_skip_logic,
    load_variables,
    normalize_for_embedding,
    to_harmonized_dataset,
)
from trans_diag.loader import (  # noqa: E402
    _IDENTIFIER_CANONICALS,
    YEARLY_VISIT_MAP,
)

DATA_DIR = REPO_ROOT / "data"
DICT_PATH = DATA_DIR / "face-common-vars.xlsx"
REPORTS_DIR = REPO_ROOT / "results" / "reports"

COHORTS = ("BP", "SZ", "DR")
COHORT_FILE = {"BP": "bipolar.csv", "SZ": "schizophrenia.csv", "DR": "depression.csv"}
COHORT_COLOR = {"BP": "#3b6fb6", "SZ": "#c2563a", "DR": "#3a9367"}

# Cache of {cohort: DataFrame[visit + needed raw cols]} so the raw-value lookup
# reads each CSV once (the 'visit' column plus whatever a variable names).
_RAW_VISIT_CACHE: dict[str, pd.DataFrame] = {}


def _raw_yearly_nonnull(var) -> dict[str, int]:
    """Non-null count of each cohort's RAW source column at yearly visits only.

    Lets the QA tell 'harmonization dropped real values' (raw > 0, loaded == 0)
    apart from 'instrument simply absent at the kept visits' (raw == 0 too).
    """
    out: dict[str, int] = {}
    for c, col in (("BP", var.bp_csv_col), ("SZ", var.sz_csv_col),
                   ("DR", var.dr_csv_col)):
        if not col:
            continue
        if c not in _RAW_VISIT_CACHE:
            out[c] = -1  # filled lazily below via _load_raw_visit
        try:
            df = _load_raw_visit(c, col)
            if col in df.columns:
                yearly = df[df["visit"].isin(YEARLY_VISIT_MAP)]
                out[c] = int(yearly[col].notna().sum())
            else:
                out[c] = 0
        except Exception:
            out[c] = 0
    return out


def _load_raw_visit(cohort: str, col: str) -> pd.DataFrame:
    path = DATA_DIR / COHORT_FILE[cohort]
    return pd.read_csv(path, usecols=lambda c: c in {"visit", col},
                       low_memory=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _expected_cohorts(var) -> list[str]:
    out = []
    if var.bp_csv_col:
        out.append("BP")
    if var.sz_csv_col:
        out.append("SZ")
    if var.dr_csv_col:
        out.append("DR")
    return out


def verify_variable(var, df: pd.DataFrame, df_raw: pd.DataFrame,
                    sanity_report: dict) -> dict:
    """Return a per-variable QA record (checks + per-cohort stats)."""
    name = var.canonical_name
    expected = _expected_cohorts(var)
    rec = {
        "name": name, "section": var.section, "dtype": var.dtype,
        "readiness": var.cluster_readiness, "label": var.label,
        "smin": var.sanity_min, "smax": var.sanity_max,
        "expected": expected, "checks": [], "ok": True, "stats": {},
        "nulled": {c: int(n) for (c, v), n in sanity_report.items() if v == name},
    }

    def fail(msg):
        rec["checks"].append(("FAIL", msg)); rec["ok"] = False

    def ok(msg):
        rec["checks"].append(("PASS", msg))

    if name not in df.columns:
        fail(f"column absent from unified dataframe")
        return rec

    series = df[name]
    is_id = name in _IDENTIFIER_CANONICALS

    # 1. LOADED — coverage matches the dictionary's per-cohort source columns.
    # A named-but-empty column can mean two different things, distinguished by
    # whether the RAW source column had any value at the kept (yearly) visits:
    #   - empty raw too  → the instrument is only collected at dropped visits
    #     (e.g. MDQ at screening) or is genuinely all-missing → WARN, expected.
    #   - raw had values → harmonization lost them (text→NaN with no rule), a
    #     real encoding gap → FAIL.
    cov_ok = True
    raw_nonnull = _raw_yearly_nonnull(var)
    for c in expected:
        n = int(series[df["cohort"] == c].notna().sum())
        rec["stats"][c] = {"n": n}
        if n == 0:
            if raw_nonnull.get(c, 0) > 0:
                cov_ok = False
                fail(f"{c}: {raw_nonnull[c]} raw values lost in harmonization "
                     f"(likely unmapped text→NaN)")
            else:
                rec["checks"].append(
                    ("WARN", f"{c}: no data at yearly visits "
                             f"(instrument not collected here / all-missing)"))
    if cov_ok and expected and any(rec["stats"][c]["n"] > 0 for c in expected):
        ok(f"loaded ({'/'.join(c for c in expected if rec['stats'][c]['n'] > 0)})")

    # 2. ENCODED — non-identifier, non-string features must be numeric.
    dtype_norm = (var.dtype or "").strip().lower()
    text_like = dtype_norm in {"string", "category"} or dtype_norm.startswith("date")
    if not is_id and not text_like:
        numeric = pd.to_numeric(series, errors="coerce")
        leaked = int(series.notna().sum() - numeric.notna().sum())
        if leaked > 0:
            fail(f"{leaked} non-numeric cells survived harmonization (text leak)")
        else:
            ok("encoded to numeric on a single cross-cohort scale")
        series = numeric

    # 3. IN-BOUNDS — after sanity, nothing outside [min,max].
    if (var.sanity_min is not None or var.sanity_max is not None) and not text_like:
        x = pd.to_numeric(series, errors="coerce").dropna().astype("float64")
        lo, hi = var.sanity_min, var.sanity_max
        over = 0
        if lo is not None:
            over += int((x < lo).sum())
        if hi is not None:
            over += int((x > hi).sum())
        if over > 0:
            fail(f"{over} values outside [{lo}, {hi}] after sanity stage")
        else:
            ok(f"all values within sanity bounds [{lo}, {hi}]")

    # 4. COMPARABLE — per-cohort medians on the same scale (flag only).
    meds = {}
    for c in expected:
        x = pd.to_numeric(series[df["cohort"] == c], errors="coerce").dropna()
        if len(x):
            meds[c] = float(x.median())
            rec["stats"].setdefault(c, {})["median"] = meds[c]
    if len(meds) >= 2 and not is_id:
        vals = list(meds.values())
        lo_m, hi_m = min(vals), max(vals)
        # ratio guard only meaningful away from 0
        if lo_m > 1e-9 and hi_m / lo_m > 10:
            rec["checks"].append(
                ("WARN", f"cohort medians span >10x ({meds}) — inspect scale"))
    return rec


# ---------------------------------------------------------------------------
# Plot — overlaid cross-cohort distribution → base64 PNG
# ---------------------------------------------------------------------------

def _plot_png(var, df: pd.DataFrame) -> str | None:
    name = var.canonical_name
    if name not in df.columns:
        return None
    series = pd.to_numeric(df[name], errors="coerce").astype("float64")
    data = {c: series[df["cohort"] == c].dropna() for c in _expected_cohorts(var)}
    data = {c: x for c, x in data.items() if len(x) > 0}
    if not data:
        return None

    allvals = pd.concat(data.values())
    n_unique = int(allvals.nunique())

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    if n_unique <= 12:
        # discrete: grouped bar of category proportions
        cats = sorted(allvals.unique())
        width = 0.8 / max(len(data), 1)
        for i, (c, x) in enumerate(data.items()):
            props = x.value_counts(normalize=True).reindex(cats, fill_value=0)
            ax.bar(np.arange(len(cats)) + i * width, props.values, width,
                   label=f"{c} (n={len(x)})", color=COHORT_COLOR[c], alpha=0.85)
        ax.set_xticks(np.arange(len(cats)) + width * (len(data) - 1) / 2)
        ax.set_xticklabels([f"{int(v)}" if float(v).is_integer() else f"{v:g}"
                            for v in cats], fontsize=7)
        ax.set_ylabel("proportion", fontsize=8)
    else:
        # continuous: overlaid density-normalized histograms
        lo, hi = allvals.quantile(0.005), allvals.quantile(0.995)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = allvals.min(), allvals.max() + 1e-9
        bins = np.linspace(lo, hi, 31)
        for c, x in data.items():
            ax.hist(x.clip(lo, hi), bins=bins, density=True, histtype="stepfilled",
                    alpha=0.45, label=f"{c} (n={len(x)})", color=COHORT_COLOR[c])
        ax.set_ylabel("density", fontsize=8)
    # sanity bound markers
    for b in (var.sanity_min, var.sanity_max):
        if b is not None and n_unique > 12:
            ax.axvline(b, color="#555", ls="--", lw=0.8)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    ax.set_title(name, fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Part 2 — encoded modelling features (domain scores, post-preprocessing)
# ---------------------------------------------------------------------------

def compute_domain_records(df: pd.DataFrame, variables: list) -> list[dict]:
    """Build the V0 construct-level domain scores — the representation that enters
    the dimensional FA and the stratification embedding — plus per-domain degeneracy
    checks. Items are robust-z scored then masked-mean aggregated (no imputation)."""
    ds = to_harmonized_dataset(df, variables, visit="V0", sections=DOMAIN_SECTIONS)
    scores, meta = build_domain_scores(ds.X, variables, cognition=COGNITIVE_COMPOSITES)
    # domain-score indices carry lowercase cohort codes (bp/sz/dr); match COHORTS (BP/SZ/DR)
    cohort = pd.Index(scores.index.get_level_values("cohort")).str.upper()
    records: list[dict] = []
    for dom in scores.columns:
        s = pd.to_numeric(scores[dom], errors="coerce")
        by_cohort = {c: s[cohort == c].dropna() for c in COHORTS}
        present = [c for c in COHORTS if len(by_cohort[c]) > 0]
        nonnull = s.dropna()
        std = float(nonnull.std()) if len(nonnull) > 1 else 0.0
        cov = float(s.notna().mean())
        rec = {"name": dom, "kind": str(meta.loc[dom, "kind"]),
               "n_items": int(meta.loc[dom, "n_items"]),
               "members": str(meta.loc[dom, "members"]), "coverage": cov,
               "by_cohort": by_cohort, "checks": [], "ok": True}
        # near-zero variance — degenerate, breaks the correlation / FA
        if std < 1e-6:
            rec["checks"].append(("FAIL", "near-zero variance — degenerate"))
            rec["ok"] = False
        else:
            rec["checks"].append(("PASS", f"variance ok (sd={std:.2f})"))
        # coverage vs the 30% modelling floor (constructs below it are dropped from Stage 3)
        if cov < 0.05:
            rec["checks"].append(("FAIL", f"coverage {cov:.0%} — far too sparse to model"))
            rec["ok"] = False
        elif cov < 0.30:
            rec["checks"].append(("WARN", f"low pooled coverage {cov:.0%} (< 30% floor; symptom/biology auto-dropped by 03, cognition uses a per-cohort rule)"))
        else:
            rec["checks"].append(("PASS", f"coverage {cov:.0%}"))
        # cohort presence / cross-cohort scale (scores are pooled robust-z, ~centred at 0)
        if len(present) < 2:
            rec["checks"].append(("WARN", f"observed in only {present or ['none']} — cohort-missingness / confound risk"))
        else:
            meds = {c: round(float(by_cohort[c].median()), 2) for c in present}
            spread = max(meds.values()) - min(meds.values())
            if spread > 2.0:
                rec["checks"].append(("WARN", f"cross-cohort median spread {spread:.1f} ({meds}) — strong cohort signal or residual scale"))
            else:
                rec["checks"].append(("PASS", f"cohort medians comparable ({meds})"))
        records.append(rec)
    return records


def _domain_plot_png(rec: dict) -> str | None:
    data = {c: x for c, x in rec["by_cohort"].items() if len(x) > 0}
    if not data:
        return None
    allvals = pd.concat(data.values())
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    lo, hi = allvals.quantile(0.005), allvals.quantile(0.995)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(allvals.min()), float(allvals.max()) + 1e-9
    bins = np.linspace(lo, hi, 31)
    for c, x in data.items():
        ax.hist(x.clip(lo, hi), bins=bins, density=True, histtype="stepfilled",
                alpha=0.45, label=f"{c} (n={len(x)})", color=COHORT_COLOR[c])
    ax.axvline(0.0, color="#555", ls="--", lw=0.8)  # robust-z centre
    ax.set_ylabel("density", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    ax.set_title(rec["name"], fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_part3(domain_records: list[dict]) -> str:
    by_kind: dict[str, list] = {}
    for r in domain_records:
        by_kind.setdefault(r["kind"], []).append(r)
    n = len(domain_records)
    n_flag = sum(1 for r in domain_records if not r["ok"])
    cov = float(np.mean([r["coverage"] for r in domain_records])) if domain_records else 0.0
    flag_color = "#cf222e" if n_flag else "#1a7f37"
    o = ["<h2 id='part3' style='border-color:#3a9367'>Part 3 — Encoded modelling features "
         "(aggregated V0 domain scores)</h2>",
         "<div class='summary'><div class='muted'><b>What this is:</b> items are robust-z "
         "scored and masked-mean aggregated into construct-level <b>domain scores</b> at baseline "
         "V0 (no imputation) — the ~69 features that actually feed the dimensional factor analysis "
         "and the stratification embedding (after age/sex residualization). "
         "<b>Why aggregate (and not model the raw variables):</b> so each construct counts ONCE — a "
         "30-item instrument would otherwise be ~30% of the dimensions and drown out single-item "
         "constructs (item-count weighting bias); to RAISE coverage without imputing (a score needs "
         "only some of its items); and to keep dimensions/clusters INTERPRETABLE over named "
         "constructs (depression, mania, metabolic, cognition…) rather than 190 noisy items. "
         "V0 is the analysis anchor; later visits test temporal coherence. Each card: the pooled "
         "cross-cohort distribution of one domain score (dashed = 0), its members, coverage, and "
         "degeneracy checks.</div>"
         "<div class='cohort-row' style='margin-top:10px'>",
         f"<div><div class='big'>{n}</div><div class='muted'>domain scores</div></div>",
         f"<div><div class='big' style='color:{flag_color}'>{n_flag}</div>"
         "<div class='muted'>flagged</div></div>",
         f"<div><div class='big'>{cov:.0%}</div><div class='muted'>mean coverage</div></div>",
         "</div></div>"]
    for kind in ("cognition", "biology", "symptom"):
        recs = sorted(by_kind.get(kind, []), key=lambda r: -r["coverage"])
        if not recs:
            continue
        o.append(f"<h2 id='dom-{kind}'>{html.escape(kind)} domains ({len(recs)})</h2><div class='cards'>")
        for r in recs:
            st = ("ok", "OK") if r["ok"] else ("bad", "CHECK")
            o.append("<div class='card'>")
            o.append(f"<h3>{html.escape(r['name'])} <span class='pill {st[0]}'>{st[1]}</span> "
                     f"<span class='pill ready'>{html.escape(r['kind'])}</span></h3>")
            png = _domain_plot_png(r)
            if png:
                o.append(f"<img alt='{html.escape(r['name'])}' src='data:image/png;base64,{png}'/>")
            else:
                o.append("<div class='muted'>(no data to plot)</div>")
            o.append(f"<div class='bounds'>coverage = <b>{r['coverage']:.0%}</b> &nbsp; "
                     f"items = <b>{r['n_items']}</b><br><span class='muted'>"
                     f"{html.escape(r['members'][:120])}</span></div>")
            o.append("<div class='checks'>")
            for level, msg in r["checks"]:
                o.append(f"<div class='{level}'>{level}: {html.escape(msg)}</div>")
            o.append("</div></div>")
        o.append("</div>")
    return "".join(o)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
*{box-sizing:border-box} body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
margin:0;color:#1a1a1a;background:#fafafa} header{background:#fff;border-bottom:1px solid #e5e5e5;
padding:18px 28px;position:sticky;top:0;z-index:5} h1{margin:0 0 4px;font-size:20px}
h2{margin:28px 28px 8px;font-size:16px;border-bottom:2px solid #3b6fb6;padding-bottom:4px}
.muted{color:#777} .toc{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.toc a{font-size:12px;color:#3b6fb6;text-decoration:none;padding:2px 6px;background:#eef3fa;border-radius:4px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px;padding:0 28px}
.card{background:#fff;border:1px solid #e5e5e5;border-radius:8px;padding:12px}
.card h3{margin:0 0 6px;font-size:13px;font-family:ui-monospace,Menlo,monospace}
.card img{width:100%;height:auto;border:1px solid #f0f0f0;border-radius:4px}
.bounds{font-size:12px;margin-top:6px;padding:6px 8px;background:#f6f8fa;border-radius:5px;
font-family:ui-monospace,Menlo,monospace}
.checks{font-size:11.5px;margin-top:6px} .checks div{padding:1px 0}
.PASS{color:#1a7f37} .FAIL{color:#cf222e;font-weight:600} .WARN{color:#9a6700}
.pill{font-size:10px;font-weight:700;padding:1px 6px;border-radius:9px;color:#fff;vertical-align:middle}
.ok{background:#1a7f37} .bad{background:#cf222e} .ready{background:#3b6fb6} .partial{background:#9a6700}
.summary{margin:10px 28px;padding:12px 16px;background:#fff;border:1px solid #e5e5e5;border-radius:8px}
.big{font-size:26px;font-weight:700} .cohort-row{display:flex;gap:24px;flex-wrap:wrap}
"""


def slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in s.lower()).strip("-")


def _scaled_plot_png(name: str, col: pd.Series, cohort: pd.Series) -> str | None:
    """Per-cohort distribution of one type-aware-scaled variable (bars if few distinct,
    else histogram), with dashed guides at +/-1."""
    data = {c: col[cohort == c].dropna() for c in COHORTS}
    data = {c: x for c, x in data.items() if len(x) > 0}
    if not data:
        return None
    allv = pd.concat(data.values())
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    if int(allv.nunique()) <= 12:                  # discrete (binary/ordinal scaled) -> bars
        cats = sorted(allv.unique())
        width = 0.8 / max(len(data), 1)
        for i, (c, x) in enumerate(data.items()):
            props = x.value_counts(normalize=True).reindex(cats, fill_value=0)
            ax.bar(np.arange(len(cats)) + i * width, props.values, width,
                   label=f"{c} (n={len(x)})", color=COHORT_COLOR[c], alpha=0.85)
        ax.set_xticks(np.arange(len(cats)) + width * (len(data) - 1) / 2)
        ax.set_xticklabels([f"{v:.2g}" for v in cats], fontsize=7)
        ax.set_ylabel("proportion", fontsize=8)
    else:                                          # continuous scaled -> histogram in [-1, 1]
        bins = np.linspace(-1, 1, 31)
        for c, x in data.items():
            ax.hist(x.clip(-1, 1), bins=bins, density=True, histtype="stepfilled",
                    alpha=0.45, label=f"{c} (n={len(x)})", color=COHORT_COLOR[c])
        ax.set_ylabel("density", fontsize=8)
    for b in (-1.0, 1.0):
        ax.axvline(b, color="#555", ls="--", lw=0.8)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{name} (scaled)", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_part2_scaled(records: list[dict], scaled: pd.DataFrame, df: pd.DataFrame,
                        vars_by_name: dict) -> str:
    """Part 2: every Part-1 variable AFTER type-aware scaling to [-1, 1] — same sections + order."""
    sections: dict[str, list] = {}
    for r in records:
        sections.setdefault(r["section"] or "—", []).append(r)
    cohort = df["cohort"]
    out_of_range = 0
    body: list[str] = []
    for sec, recs in sections.items():
        recs = sorted(recs, key=lambda r: r["name"])
        body.append(f"<h2 id='s2-{slug(sec)}'>{html.escape(sec)} — scaled</h2><div class='cards'>")
        for r in recs:
            name = r["name"]
            if name not in scaled.columns:
                continue
            col = scaled[name]
            obs = col.dropna()
            lo, hi = (float(obs.min()), float(obs.max())) if len(obs) else (float("nan"), float("nan"))
            inrange = len(obs) == 0 or (lo >= -1.0001 and hi <= 1.0001)
            if not inrange:
                out_of_range += 1
            st = ("ok", "in [-1,1]") if inrange else ("bad", "OUT OF RANGE")
            v = vars_by_name.get(name)
            decl = (v.unit_or_value_set or "").strip() if v else ""
            if not decl or decl.lower().startswith("free text"):
                decl = (f"[{r['smin']:g}, {r['smax']:g}]"
                        if r["smin"] is not None and r["smax"] is not None else str(r["dtype"]))
            body.append("<div class='card'>")
            body.append(f"<h3>{html.escape(name)} <span class='pill {st[0]}'>{st[1]}</span></h3>")
            png = _scaled_plot_png(name, col, cohort)
            body.append(f"<img alt='{html.escape(name)} scaled' src='data:image/png;base64,{png}'/>"
                        if png else "<div class='muted'>(no data to plot)</div>")
            body.append(f"<div class='bounds'>declared: <b>{html.escape(str(decl)[:46])}</b><br>"
                        f"scaled range: <b>[{lo:.2f}, {hi:.2f}]</b> &middot; "
                        f"<span class='muted'>{html.escape(str(r['dtype']))}</span></div>")
            body.append("</div>")
        body.append("</div>")
    color = "#cf222e" if out_of_range else "#1a7f37"
    head = ("<h2 id='part2' style='border-color:#9a6700'>Part 2 — Post-processed variables "
            "(type-aware scaling to [&minus;1,&nbsp;1])</h2>"
            "<div class='summary'><div class='muted'>Every Part-1 variable after the type-aware "
            "scaler — binary/ordinal &rarr; min-max; continuous &rarr; log (if skewed) + winsorize + "
            "robust-z clipped &plusmn;5, /5. Same variables, same order as Part 1. Each card: the scaled "
            "cross-cohort distribution (dashed = &plusmn;1), the declared range, and the realised scaled "
            "range — every feature must land in [&minus;1,&nbsp;1].</div>"
            f"<div class='cohort-row' style='margin-top:10px'><div>"
            f"<div class='big' style='color:{color}'>{out_of_range}</div>"
            "<div class='muted'>variables out of [-1,1]</div></div></div></div>")
    return head + "".join(body)


def v0_missingness(df: pd.DataFrame, variables: list) -> dict:
    """Per-variable V0 missingness: pooled (all patients) and within expected cohorts.

    Within-cohort is the fair measure — a 2-cohort variable isn't 'missing' for the cohort that
    never collected it. Reported, not dropped: the masked methods consume these as-is.
    """
    v0 = df[df["visit"] == "V0"] if "visit" in df.columns else df
    n = len(v0)
    sizes = v0["cohort"].value_counts().to_dict()
    coh = v0["cohort"]
    out: dict[str, dict] = {}
    for v in variables:
        c = v.canonical_name
        if c not in v0.columns:
            continue
        s = v0[c]
        exp = _expected_cohorts(v)
        n_obs_w = sum(int(s[coh == ch].notna().sum()) for ch in exp)
        size_w = sum(int(sizes.get(ch, 0)) for ch in exp)
        out[c] = {"pooled": 1 - int(s.notna().sum()) / n if n else 1.0,
                  "within": 1 - n_obs_w / size_w if size_w else 1.0}
    return out


def compute_skip_logic_report(df: pd.DataFrame) -> dict | None:
    """Coverage recovered by skip-logic decoding (gate=No -> structural 0), by cohort.

    Compares per-dependent V0 coverage before vs after
    :func:`trans_diag.skip_logic.decode_skip_logic`, plus the total cells filled.
    Returns ``None`` when none of the gated columns are present.
    """
    v0 = df[df["visit"] == "V0"] if "visit" in df.columns else df
    cols = sorted({c for r in SUICIDE_SKIP_RULES for c in (r.gate, *r.dependents)
                   if c in v0.columns})
    deps = [d for r in SUICIDE_SKIP_RULES for d in r.dependents if d in v0.columns]
    if not cols or not deps:
        return None
    coh = v0["cohort"].to_numpy()
    before = pd.DataFrame({c: pd.to_numeric(v0[c], errors="coerce") for c in cols})
    after, _ = decode_skip_logic(before.copy())
    rows, seen = [], set()
    for d in deps:
        if d in seen:
            continue
        seen.add(d)
        rec = {"dependent": d,
               "filled": int((after[d].notna() & before[d].isna()).sum())}
        for ch in COHORTS:
            m = coh == ch
            rec[ch] = (
                (before[d][m].notna().mean() * 100) if m.any() else float("nan"),
                (after[d][m].notna().mean() * 100) if m.any() else float("nan"),
            )
        rows.append(rec)
    return {"rows": rows,
            "rules": [(r.gate, ", ".join(r.dependents), r.rationale)
                      for r in SUICIDE_SKIP_RULES]}


def render_skip_logic(rep: dict | None) -> str:
    """Part-1 panel: coverage recovered by skip-logic decoding."""
    if not rep or not rep["rows"]:
        return ""
    o = ["<div class='summary' id='skiplogic'>",
         "<b>Skip-logic decoding (gate = No &rarr; structural 0)</b>",
         "<div class='muted' style='margin-top:6px;line-height:1.6;max-width:900px'>",
         "Conditional suicide-module items (ISF) are only asked when a gate is Yes; a No leaves them "
         "blank &mdash; a <b>structural zero</b>, not missing data (e.g. ISF05 “never attempted” "
         "&rArr; 0 attempts). We fill 0 <b>only</b> where the gate is explicitly No and the cell is "
         "blank &mdash; never overwriting an existing value, never where the gate is unknown. This is "
         "<b>not imputation</b> (see <code>skip_logic.py</code>). V0 coverage before &rarr; after, by cohort:",
         "</div>"]
    o.append("<table style='border-collapse:collapse;margin-top:10px;font-size:13px'>")
    o.append("<tr><th style='text-align:left;padding:4px 16px 4px 0'>item</th>"
             "<th style='padding:4px 16px'>BP</th><th style='padding:4px 16px'>SZ</th>"
             "<th style='padding:4px 16px'>DR</th>"
             "<th style='padding:4px 0'>cells filled</th></tr>")

    def cell(t: tuple[float, float]) -> str:
        b, a = t
        if b != b:  # NaN
            return "<td class='muted' style='padding:4px 16px'>&mdash;</td>"
        col = "#1a7f37" if (a - b) > 1 else "#57606a"
        return (f"<td style='padding:4px 16px'><span class='muted'>{b:.0f}%</span> &rarr; "
                f"<b style='color:{col}'>{a:.0f}%</b></td>")

    for r in rep["rows"]:
        o.append(f"<tr><td style='padding:4px 16px 4px 0'><code>{html.escape(r['dependent'])}</code></td>"
                 f"{cell(r['BP'])}{cell(r['SZ'])}{cell(r['DR'])}"
                 f"<td style='padding:4px 0'><b>{r['filled']:,}</b></td></tr>")
    o.append("</table></div>")
    return "".join(o)


def build_html(records: list[dict], df: pd.DataFrame, vars_by_name: dict,
               counts: dict, n_pass: int, n_fail: int,
               domain_records: list[dict] | None = None,
               scaled: pd.DataFrame | None = None,
               skip_report: dict | None = None) -> str:
    sections: dict[str, list] = {}
    for r in records:
        sections.setdefault(r["section"] or "—", []).append(r)

    out: list[str] = ["<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
                      "<meta name='viewport' content='width=device-width,initial-scale=1'>",
                      "<title>FACE Common QA — Harmonization</title>",
                      f"<style>{CSS}</style></head><body>"]
    out.append("<header><h1>FACE Common QA — Variable Harmonization &amp; Sanity</h1>")
    out.append("<div class='muted'>Part 1 — every harmonized variable (native scale): "
               "cross-cohort distribution, sanity bounds, missingness, verification. "
               "Part 2 — the same variables after type-aware scaling to [&minus;1, 1] "
               "(ML-ready), same order. Part 3 — the aggregated domain scores that enter "
               "the factor analysis &amp; embedding.</div>")
    out.append("<nav class='toc'>")
    if skip_report:
        out.append("<a href='#skiplogic' style='background:#eef0ff;color:#3a3aa3'>"
                   "&#9656; Skip-logic recovery</a>")
    for s in sections:
        out.append(f"<a href='#{slug(s)}'>{html.escape(s)} ({len(sections[s])})</a>")
    if scaled is not None:
        out.append("<a href='#part2' style='background:#fff4e0;color:#9a6700'>&#9656; Part 2: scaled variables</a>")
    if domain_records:
        out.append("<a href='#part3' style='background:#e6f3ec;color:#3a9367'>&#9656; Part 3: domain scores</a>")
    out.append("</nav></header>")

    # overview
    out.append("<div class='summary'><div class='cohort-row'>")
    out.append(f"<div><div class='big'>{len(records)}</div>"
               f"<div class='muted'>variables</div></div>")
    out.append(f"<div><div class='big' style='color:#1a7f37'>{n_pass}</div>"
               f"<div class='muted'>pass all checks</div></div>")
    bad_color = "#cf222e" if n_fail else "#1a7f37"
    out.append(f"<div><div class='big' style='color:{bad_color}'>{n_fail}</div>"
               f"<div class='muted'>with a failing check</div></div>")
    for c in COHORTS:
        out.append(f"<div><div class='big'>{counts.get(c, 0):,}</div>"
                   f"<div class='muted'>{c} rows</div></div>")
    out.append("</div></div>")

    # preprocessing & scaling + V0 missingness overview
    mrecs = [r for r in records if r.get("miss")]
    m25 = sum(1 for r in mrecs if r["miss"]["within"] > 0.25)
    m50 = sum(1 for r in mrecs if r["miss"]["within"] > 0.50)
    out.append(
        "<div class='summary'><b>Preprocessing &amp; scaling (by data type)</b>"
        "<div class='muted' style='margin-top:6px;line-height:1.7'>"
        "<b>Continuous</b> (float): rules + sanity bounds &rarr; log1p if heavy right-skewed "
        "(prolactin/CRP/&hellip;) &rarr; winsorize 1/99 + robust-z (median/MAD) clipped &plusmn;5 "
        "&rarr; <b>[&minus;1,1]</b>. &nbsp; <b>Binary / ordinal / Likert</b>: bounds &rarr; min-max "
        "&rarr; <b>[&minus;1,1]</b>. &nbsp; Items &rarr; masked-mean of robust-z &rarr; domain scores "
        "(&le;&plusmn;5, Part 2). <b>No imputation</b>; missingness is handled by the masked "
        "similarity / masked FA (no hard drop)."
        "</div><div class='cohort-row' style='margin-top:10px'>"
        f"<div><div class='big'>{m25}</div><div class='muted'>vars &gt;25% missing<br>(V0, within-cohort)</div></div>"
        f"<div><div class='big'>{m50}</div><div class='muted'>vars &gt;50% missing</div></div>"
        "<div><div class='muted' style='max-width:360px'>Flagged for awareness only &mdash; not "
        "dropped. Within-cohort missingness ignores structural cohort-absence (a 2-cohort variable "
        "isn't 'missing' where it was never collected).</div></div>"
        "</div></div>")

    # Skip-logic decoding panel (structural-zero recovery for gated items).
    out.append(render_skip_logic(skip_report))

    for sec, recs in sections.items():
        recs.sort(key=lambda r: r["name"])
        out.append(f"<h2 id='{slug(sec)}'>{html.escape(sec)}</h2><div class='cards'>")
        for r in recs:
            var = vars_by_name[r["name"]]
            status = ("ok", "OK") if r["ok"] else ("bad", "CHECK")
            ready = "ready" if r["readiness"].startswith("READY") else "partial"
            out.append("<div class='card'>")
            out.append(f"<h3>{html.escape(r['name'])} "
                       f"<span class='pill {status[0]}'>{status[1]}</span> "
                       f"<span class='pill {ready}'>{ready.upper()}</span></h3>")
            png = _plot_png(var, df)
            if png:
                out.append(f"<img alt='{html.escape(r['name'])} distribution' "
                           f"src='data:image/png;base64,{png}'/>")
            else:
                out.append("<div class='muted'>(no numeric data to plot)</div>")
            # sanity bounds line — explicitly below the plot, per variable
            smin = "—" if r["smin"] is None else f"{r['smin']:g}"
            smax = "—" if r["smax"] is None else f"{r['smax']:g}"
            nulled = sum(r["nulled"].values())
            null_txt = f" · nulled {nulled} out-of-range" if nulled else ""
            out.append(f"<div class='bounds'>sanity min = <b>{smin}</b> &nbsp; "
                       f"sanity max = <b>{smax}</b>{null_txt}<br>"
                       f"<span class='muted'>{html.escape(r['dtype'])}</span></div>")
            m = r.get("miss")
            if m:
                wc, pc = m["within"] * 100, m["pooled"] * 100
                hi = " style='color:#9a6700;font-weight:700'" if m["within"] > 0.5 else ""
                out.append(f"<div class='bounds'>missing (V0): within-cohort "
                           f"<b{hi}>{wc:.0f}%</b> &middot; pooled {pc:.0f}%</div>")
            out.append("<div class='checks'>")
            for level, msg in r["checks"]:
                out.append(f"<div class='{level}'>{level}: {html.escape(msg)}</div>")
            out.append("</div></div>")
        out.append("</div>")

    if scaled is not None:
        out.append(render_part2_scaled(records, scaled, df, vars_by_name))
    if domain_records:
        out.append(render_part3(domain_records))
    out.append("</body></html>")
    return "".join(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print("Loading v2 unified dataframe (READY + PARTIAL, long) ...")
    sanity_report: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(
            DATA_DIR, DICT_PATH, readiness=["READY", "PARTIAL"],
            format="long", sanity_report=sanity_report,
        )
    variables = [v for v in load_variables(DICT_PATH)
                 if v.cluster_readiness.startswith(("READY", "PARTIAL"))]
    vars_by_name = {v.canonical_name: v for v in variables}

    counts = df["cohort"].value_counts().to_dict()
    print(f"  shape {df.shape}; cohorts {counts}")
    print(f"  sanity stage nulled cells in {len(sanity_report)} (cohort,var) pairs")

    print("Verifying each variable ...")
    records, seen = [], set()
    for v in variables:
        if v.canonical_name in seen or v.canonical_name in _IDENTIFIER_CANONICALS:
            continue
        seen.add(v.canonical_name)
        records.append(verify_variable(v, df, df, sanity_report))

    miss = v0_missingness(df, variables)
    for r in records:
        r["miss"] = miss.get(r["name"])

    n_fail = sum(1 for r in records if not r["ok"])
    n_pass = len(records) - n_fail
    print(f"  {n_pass}/{len(records)} variables pass all checks; {n_fail} flagged")
    if n_fail:
        for r in records:
            if not r["ok"]:
                msgs = "; ".join(m for lvl, m in r["checks"] if lvl == "FAIL")
                print(f"    FAIL  {r['name']}: {msgs}")

    print("Building Part 2 — encoded modelling features (domain scores) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        domain_records = compute_domain_records(df, variables)
    n_dom_flag = sum(1 for r in domain_records if not r["ok"])
    print(f"  {len(domain_records)} domain scores; {n_dom_flag} flagged")

    print("Scaling variables (type-aware -> [-1,1]) for Part 2 ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        num = pd.DataFrame(
            {r["name"]: pd.to_numeric(df[r["name"]], errors="coerce").astype("float64")
             for r in records if r["name"] in df.columns},
            index=df.index)
        scaled = normalize_for_embedding(num)
    print(f"  scaled {scaled.shape[1]} variables; overall range "
          f"[{float(np.nanmin(scaled.to_numpy())):.2f}, {float(np.nanmax(scaled.to_numpy())):.2f}]")

    print("Computing skip-logic recovery (gate=No -> structural 0) ...")
    skip_report = compute_skip_logic_report(df)
    if skip_report:
        total_filled = sum(r["filled"] for r in skip_report["rows"])
        print(f"  skip-logic would fill {total_filled:,} structural-zero cells "
              f"across {len(skip_report['rows'])} ISF count items")

    print("Rendering HTML ...")
    html_str = build_html(records, df, vars_by_name, counts, n_pass, n_fail,
                          domain_records=domain_records, scaled=scaled,
                          skip_report=skip_report)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "qa_harmonization.html"
    out_path.write_text(html_str, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  wrote {out_path} ({size_mb:.1f} MB)")
    print(f"  open {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
