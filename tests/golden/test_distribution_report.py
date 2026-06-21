"""Golden tests for the OOP QA distribution report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from face.reporting.distribution_report import (  # noqa: E402
    DistributionReport,
    _classify_form,
    _recommend_tier,
    rank_int,
)
from synthetic.generate_face_like import generate  # noqa: E402


def test_rank_int_is_standard_normal():
    rng = np.random.default_rng(0)
    x = rng.exponential(2.0, size=5000)  # strongly right-skewed
    z = rank_int(x)
    assert abs(z.mean()) < 0.05
    assert abs(z.std() - 1.0) < 0.05
    # monotone (rank-preserving): order is preserved
    assert np.all(np.argsort(x) == np.argsort(z))


def test_classify_form_names_the_distribution():
    assert _classify_form(np.array([0, 1, 0, 1, 1]), n_distinct=2, frac_zero=0.4, skew=0.0, kurt=0.0) == "binary"
    assert _classify_form(np.array([1, 2, 3, 4, 5] * 4), n_distinct=5, frac_zero=0.0, skew=0.0, kurt=0.0) == "ordinal_5"
    sym = _classify_form(np.linspace(-3, 3, 500), n_distinct=500, frac_zero=0.0, skew=0.0, kurt=0.0)
    assert sym == "continuous_symmetric"
    skewed = _classify_form(np.arange(500.0) + 0.5, n_distinct=500, frac_zero=0.0, skew=2.0, kurt=0.0)
    assert skewed == "continuous_right_skewed"
    assert _classify_form(np.zeros(10), n_distinct=1, frac_zero=1.0, skew=0.0, kurt=0.0) == "degenerate"


def test_recommend_tier_matches_copula_tiering():
    assert _recommend_tier("gaussian", n_distinct=100, modal_frac=0.1) == "gaussianize"
    assert _recommend_tier("lognormal", n_distinct=80, modal_frac=0.1) == "gaussianize"
    assert _recommend_tier("bernoulli", n_distinct=2, modal_frac=0.7) == "keep_binary"
    # low-cardinality ordinal -> stays ordinal
    assert _recommend_tier("ordered_logistic", n_distinct=5, modal_frac=0.4) == "keep_ordinal"
    # high-cardinality, non-degenerate count -> gaussianize
    assert _recommend_tier("neg_binomial", n_distinct=40, modal_frac=0.2) == "gaussianize"
    # zero-inflated count (modal mass high) -> stays count
    assert _recommend_tier("neg_binomial", n_distinct=18, modal_frac=0.75) == "keep_count"


def test_distribution_report_end_to_end_on_synthetic(tmp_path):
    outdir, _truth = generate(n=400, seed=1, out=tmp_path)
    report = DistributionReport(
        processed_dir=Path(outdir),
        html_path=tmp_path / "qa.html",
        summary_csv=tmp_path / "qa.csv",
    ).load()

    dists = report.analyze_all()
    assert dists, "no indicators analyzed"
    valid_tiers = {"gaussianize", "keep_binary", "keep_ordinal", "keep_count", "keep_native"}
    for d in dists:
        assert d.empirical_form  # every indicator gets a named form
        assert d.recommended_tier in valid_tiers
        assert d.n_obs >= 0

    frame = report.to_frame(dists)
    for col in ("item", "family", "empirical_form", "recommended_tier", "n_distinct", "modal_frac"):
        assert col in frame.columns

    html_path = report.render_html(dists)
    assert html_path.exists() and html_path.stat().st_size > 0
    text = html_path.read_text()
    assert "modeled-indicator distributions" in text
    assert "base64" in text  # embedded plots

    csv_path = report.save_csv(dists)
    assert csv_path.exists()


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "processed" / "baseline_v0.parquet").exists(),
    reason="needs processed FACE baseline",
)
def test_distribution_report_on_real_data_runs():
    report = DistributionReport(html_path=Path("/tmp/_qa_test.html"), summary_csv=Path("/tmp/_qa_test.csv"))
    out = report.run()
    assert int(out["n_indicators"]) > 100
    assert Path(out["html"]).exists()
