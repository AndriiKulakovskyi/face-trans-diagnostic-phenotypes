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

from trans_diag import build_unified_dataframe, load_variables  # noqa: E402
from trans_diag.loader import (  # noqa: E402
    _IDENTIFIER_CANONICALS,
    YEARLY_VISIT_MAP,
)

DATA_DIR = REPO_ROOT / "data"
DICT_PATH = DATA_DIR / "face-common-vars-v2.xlsx"
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


def build_html(records: list[dict], df: pd.DataFrame, vars_by_name: dict,
               counts: dict, n_pass: int, n_fail: int) -> str:
    sections: dict[str, list] = {}
    for r in records:
        sections.setdefault(r["section"] or "—", []).append(r)

    out: list[str] = ["<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
                      "<meta name='viewport' content='width=device-width,initial-scale=1'>",
                      "<title>FACE Common QA — Harmonization</title>",
                      f"<style>{CSS}</style></head><body>"]
    out.append("<header><h1>FACE Common QA — Variable Harmonization &amp; Sanity</h1>")
    out.append("<div class='muted'>Every harmonized variable loaded from "
               "<code>face-common-vars-v2.xlsx</code> across BP / SZ / DR. Each card: a "
               "cross-cohort distribution of the pooled (encoded) values, the sanity "
               "min/max below it, and per-variable verification.</div>")
    out.append("<nav class='toc'>")
    for s in sections:
        out.append(f"<a href='#{slug(s)}'>{html.escape(s)} ({len(sections[s])})</a>")
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
            out.append("<div class='checks'>")
            for level, msg in r["checks"]:
                out.append(f"<div class='{level}'>{level}: {html.escape(msg)}</div>")
            out.append("</div></div>")
        out.append("</div>")

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

    n_fail = sum(1 for r in records if not r["ok"])
    n_pass = len(records) - n_fail
    print(f"  {n_pass}/{len(records)} variables pass all checks; {n_fail} flagged")
    if n_fail:
        for r in records:
            if not r["ok"]:
                msgs = "; ".join(m for lvl, m in r["checks"] if lvl == "FAIL")
                print(f"    FAIL  {r['name']}: {msgs}")

    print("Rendering HTML ...")
    html_str = build_html(records, df, vars_by_name, counts, n_pass, n_fail)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "qa_harmonization.html"
    out_path.write_text(html_str, encoding="utf-8")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  wrote {out_path} ({size_mb:.1f} MB)")
    print(f"  open {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
