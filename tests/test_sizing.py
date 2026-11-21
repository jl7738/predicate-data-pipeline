from predx.config import Paths, load_config
from predx.sizing import BUILTIN_COUNTS, SCENARIOS, age_weights, corpus_size_at, coverage, funnel, render_markdown, run_sizing, summary_rate_by_year, write_report


def test_builtin_counts_agree_with_the_proposal_totals():
    summ = sum(v[0] for v in BUILTIN_COUNTS.values())
    allr = sum(sum(v) for v in BUILTIN_COUNTS.values())
    assert abs(summ - 94143) / 94143 < 0.005 and abs(allr - 176529) / 176529 < 0.005      # openFDA refreshes monthly
    split = BUILTIN_COUNTS["2015-2023"][0] + BUILTIN_COUNTS["2024-2024"][0] + BUILTIN_COUNTS["2025-2026"][0]
    assert split == 34782                                                                   # proposal: "about 34,600"


def test_age_model_matches_published_median_and_mean():
    import numpy as np

    ages, w, _ = age_weights()
    mean = float((ages * w).sum())
    cdf = np.cumsum(w)
    median = float(ages[np.searchsorted(cdf, 0.5)])
    assert abs(median - 53 / 12) < 0.15 and abs(mean - 74 / 12) < 0.35


def test_funnel_is_ordered_and_scenarios_bracket_each_other():
    rate = summary_rate_by_year(BUILTIN_COUNTS)
    assert rate[1980] < 0.2 and rate[2020] > 0.9
    res = {n: funnel(BUILTIN_COUNTS, p) for n, p in SCENARIOS.items()}
    for n, f in res.items():
        for sp, x in f.items():
            assert x["retrievable_queries"] <= x["subjects_with_predicate"] <= x["usable_subjects"] <= x["subjects_with_summary"]
            assert 0.5 < x["pair_coverage"] < 0.97
    assert res["low"]["train"]["retrievable_queries"] < res["mid"]["train"]["retrievable_queries"] < res["high"]["train"]["retrievable_queries"]
    assert res["mid"]["train"]["share_gold_decided_2024_plus"] == 0 and 0.05 < res["mid"]["test"]["share_gold_decided_2024_plus"] < 0.35


def test_later_corpus_start_never_raises_coverage_and_pool_grows_with_time():
    rate = summary_rate_by_year(BUILTIN_COUNTS)
    cov = [coverage("test", rate, y)["summary_coverage"] for y in (1976, 1996, 2006, 2011)]
    assert all(a >= b - 1e-9 for a, b in zip(cov, cov[1:]))
    assert corpus_size_at(2015.5, BUILTIN_COUNTS, 0.9) < corpus_size_at(2025.5, BUILTIN_COUNTS, 0.9)


def test_report_and_config_override(tmp_path):
    cfg = load_config([], ["sizing.scenarios.mid.pred_rate=0.5"])
    paths = Paths(cfg, root=tmp_path)
    rep = run_sizing(cfg, paths)
    assert rep["scenarios"]["mid"]["params"]["pred_rate"] == 0.5
    assert rep["scenarios"]["mid"]["funnel"]["train"]["subjects_with_predicate"] < 14000
    md = render_markdown(rep)
    assert "retrievable queries" in md and write_report(rep, paths).exists()
