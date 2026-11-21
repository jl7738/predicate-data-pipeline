"""CONCEPT TESTS on a synthetic 510(k) universe with PLANTED structure (see predx/synth.py).

Because we planted the truth, these tests check the pipeline recovers it exactly:
  * extracted predicates + roles == planted citations
  * every gold pair that is missing from the corpus is missing for the PLANTED reason
  * leakage planted in the text is detected and masked
  * temporal rules hold; the dataset passes schema validation
  * the audit numbers are internally consistent and the tiny-data case FAILS the size checks
"""
import json
import re
import shutil
from collections import Counter

import pytest

from conftest import make_cfg
from predx.audit import run_audit, write_report
from predx.config import Paths
from predx.dataset import build_dataset, load_dataset, validate_dataset
from predx.io import last_by_key, read_jsonl
from predx.parse_stage import run_parse
from predx.plan import in_any_window
from predx.synth import make_synthetic


def alnum(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def test_dataset_passes_schema_and_invariants(world):
    v = validate_dataset(world["paths"], world["cfg"])
    assert v["problems"] == {}, v["problems"]
    assert v["n_checked"]["corpus"] > 1000


def test_extraction_recovers_planted_citations_and_roles(world):
    truth = world["truth"]
    parsed = {r["k"]: r for r in read_jsonl(world["paths"].parsed)}
    bad, n = [], 0
    for k, t in truth.items():
        if k not in parsed or t["status"] not in ("ok", "no_desc"):
            continue
        n += 1
        got = {x["k"]: x["role"] for x in parsed[k]["predicates"] if x["role"] != "other"}
        want = {}
        if t["primary"]:
            want[t["primary"]] = "primary"
            want.update({a: "additional" for a in t["additional"]})
            want.update({r: "reference" for r in t["reference"]})
        if got != want:
            bad.append((k, got, want))
    assert n > 1000
    assert len(bad) <= 0.005 * n, bad[:5]


def test_sections_match_planted_text(world):
    truth = world["truth"]
    parsed = {r["k"]: r for r in read_jsonl(world["paths"].parsed)}
    bad_ifu = bad_desc = n = 0
    for k, t in truth.items():
        if t["status"] != "ok" or k not in parsed:
            continue
        n += 1
        p = parsed[k]
        bad_ifu += alnum(p["ifu"]) != alnum(t["ifu"])
        d = alnum(p["description"])
        bad_desc += not (d and d[:200] == alnum(t["description"])[:200] and "predicatedevice" not in d[:2000])
    assert bad_ifu <= 0.01 * n and bad_desc <= 0.01 * n, (bad_ifu, bad_desc, n)


def test_planted_failures_land_in_the_right_funnel_bucket(world):
    rows = {r["k"]: r for r in read_jsonl(world["paths"].dataset / "device_status.jsonl.gz")}
    expect = {"statement": "statement_only_meta", "nse": "decision_not_se", "pdf404": "no_pdf_404", "text_fail": "text_failed", "no_desc": "sections_missing", "ok": "usable"}
    for k, t in world["truth"].items():
        assert rows[k]["status"] == expect[t["status"]], (k, t["status"], rows[k]["status"])


def test_every_missing_gold_pair_has_the_planted_reason(world):
    truth = world["truth"]
    status_reason = {"statement": "statement_only", "nse": "decision_not_se", "pdf404": "no_pdf_404", "text_fail": "text_failed", "no_desc": "sections_missing"}
    _, queries, _, _ = load_dataset(world["paths"])
    dates = {r["k"]: r["decision_date"] for r in read_jsonl(world["paths"].meta)}
    seen = Counter()
    for qs in queries.values():
        for q in qs:
            types = truth[q["k"]]["types"]
            for g in q["gold"]:
                ty = types[g["k"]]
                if ty == "unknown":
                    want = "not_in_openfda"
                elif ty in ("pma", "den"):
                    want = "non_510k_identifier"
                elif ty == "future" or dates[g["k"]] >= q["date"]:
                    want = "temporal_violation"
                else:
                    want = status_reason.get(truth[g["k"]]["status"])      # None == in corpus
                assert g["reason"] == want, (q["k"], g, want)
                assert g["in_corpus"] == (want is None)
                seen[want] += 1
    assert {"not_in_openfda", "non_510k_identifier", "temporal_violation", "statement_only", "no_pdf_404", "text_failed", "sections_missing", None} <= set(seen)


def test_gold_is_never_a_reference_device_and_always_strictly_earlier(world):
    truth = world["truth"]
    corpus, queries, _, _ = load_dataset(world["paths"])
    date = {r["k"]: r["date"] for r in corpus}
    for qs in queries.values():
        for q in qs:
            refs = set(truth[q["k"]]["reference"])
            assert not refs & {g["k"] for g in q["gold"]}
            assert set(q["reference"]) == refs
            for k in q["gold_in_corpus"]:
                assert date[k] < q["date"]
            assert q["n_candidates"] == sum(1 for d in date.values() if d < q["date"])      # candidate rule, recomputed independently


def test_subjects_are_exactly_the_usable_windowed_devices_with_predicates(world):
    meta = {r["k"]: r for r in read_jsonl(world["paths"].meta)}
    _, queries, _, _ = load_dataset(world["paths"])
    want = {k for k, t in world["truth"].items() if t["status"] == "ok" and t["primary"] and in_any_window(meta[k]["decision_date"], world["cfg"])}
    got = {q["k"] for qs in queries.values() for q in qs}
    assert got == want
    for s, qs in queries.items():
        a, b = world["cfg"]["splits"][s]
        assert all(a <= q["date"] <= b for q in qs)


def test_planted_leakage_is_detected_then_masked(world):
    _, queries, _, _ = load_dataset(world["paths"])
    flagged = 0
    for qs in queries.values():
        for q in qs:
            leak = world["truth"][q["k"]]["leak"]
            if leak == "k":
                assert q["leak"]["raw_gold_k"] >= 1
                flagged += 1
            if leak == "name" and q["leak"]["raw_gold_name"] >= 1:
                flagged += 1
            assert q["leak"]["residual_k"] == 0 and q["leak"]["residual_gold_name"] == 0
            assert not re.search(r"\bK\d{6}\b", q["ifu"] + q["description"])
    assert flagged >= 10


def test_strict_masking_rebuild(world, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(world["root"], root)
    cfg, paths = make_cfg(root)
    build_dataset(cfg, paths, "strict")
    assert validate_dataset(paths, cfg)["problems"] == {}
    _, q_strict, _, card = load_dataset(paths)
    assert card["mask_level"] == "strict"
    assert all(q["leak"]["residual_k"] == 0 for qs in q_strict.values() for q in qs)
    _, q_std, _, _ = load_dataset(world["paths"])
    n_mask = lambda qs_: sum(q["description"].count("[DEVICE]") + q["ifu"].count("[DEVICE]") for qs in qs_.values() for q in qs)
    assert n_mask(q_strict) >= n_mask(q_std) > 0                       # strict masks at least as much as standard
    std = {q["k"]: q for qs in q_std.values() for q in qs}
    assert any(q["description"] != std[q["k"]]["description"] for qs in q_strict.values() for q in qs)


def test_audit_numbers_are_consistent(world, monkeypatch):
    # make the test independent of the machine: with internet (e.g. Colab) public tokenizers DO load and approx is False
    import transformers

    def _offline(*a, **k):
        raise OSError("offline (forced by test)")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _offline)
    rep = run_audit(world["cfg"], world["paths"], baselines=True, tokenizers=True)
    cv = rep["coverage"]
    assert sum(r["pairs"] for r in cv["by_reason"]) == cv["pairs_total"]
    assert cv["pairs_in_corpus"] == next(r["pairs"] for r in cv["by_reason"] if r["reason"] == "in_corpus")
    assert 0 < cv["pair_coverage_pct"] < 100
    assert rep["funnel_all"][0]["n"] == 1500 and rep["funnel_all"][-1]["n"] == rep["n_corpus"]
    ids = {c["id"]: c for c in rep["checks"]}
    assert ids["schema"]["status"] == "PASS" and ids["baseline_sanity"]["status"] == "PASS"
    assert ids["leak_residual_k"]["status"] == "PASS"
    assert ids["tokenizer_approx"]["status"] == "WARN"                        # offline -> loudly approximate
    for m in rep["tokens"]:
        assert m["approx"] and m["recommended_max_len"] in (128, 256, 512, 1024, None)
    # ceiling table: a later corpus start can only lower the ceiling
    ceil = [r["pair_ceiling_pct"] for r in cv["ceiling_by_corpus_start"]]
    assert ceil == sorted(ceil, reverse=True)
    # shortlist maths: max achievable Recall@k never decreases with k
    for s, v in rep["shortlist"].items():
        if v.get("n_retrievable"):
            mx = [r["mean_max_recall"] for r in v["by_k"]]
            assert mx == sorted(mx)
    write_report(rep, world["paths"])
    md = (world["paths"].reports / "audit.md").read_text()
    for h in ("## 1. Funnel", "## 3. Token lengths", "## 4. Predicate coverage", "## 5. Candidates", "## 6. Are there enough examples", "## 7. Leakage"):
        assert h in md
    assert (world["paths"].reports / "tables" / "coverage_by_reason.csv").exists()


def test_bm25_beats_random_and_popularity_is_computed(world):
    from predx.baselines import run_baselines

    b = run_baselines(world["cfg"], world["paths"], "test")
    assert b["bm25"]["hit@10"]["mean"] > 3 * b["random_expected"]["hit@10"]
    assert b["bm25"]["recall@10"]["lo"] <= b["bm25"]["recall@10"]["mean"] <= b["bm25"]["recall@10"]["hi"]
    assert "product_code_popularity" in b


def test_too_little_data_is_flagged(tmp_path):
    cfg, paths = make_cfg(tmp_path)
    make_synthetic(tmp_path, n=60, seed=3)
    run_parse(cfg, paths)
    build_dataset(cfg, paths)
    rep = run_audit(cfg, paths, baselines=False, tokenizers=False)
    assert rep["overall"] == "FAIL"
    ids = {c["id"]: c["status"] for c in rep["checks"]}
    assert ids["n_train"] == "FAIL" and ids["n_test"] == "FAIL"
    write_report(rep, paths)                                              # empty splits must not crash the report


def test_demo_cli_runs_offline_end_to_end(tmp_path):
    from predx.cli import main

    demo = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "config" / "demo.yaml")
    main(["--config", demo, "--set", f"paths.data_dir={tmp_path / 'd'}", "demo", "--n", "500", "--no-tokenizers"])
    assert (tmp_path / "d" / "reports" / "audit.md").exists()


def test_pdf_stage_inside_the_pipeline(tmp_path):
    """Real PDFs (digital + scanned) go through text -> parse and come out as the planted records."""
    from predx.text_stage import run_text

    cfg, paths = make_cfg(tmp_path, ["text.workers=2"])
    info = make_synthetic(tmp_path, n=300, seed=11, with_pdfs=6)
    assert len(info["pdfs"]) == 6
    stats = run_text(cfg, paths, info["pdfs"])
    assert stats["errors"] == 0 and stats["extracted"] == 6
    run_parse(cfg, paths)
    truth = json.loads((tmp_path / "truth.json").read_text())
    ts = last_by_key(paths.text_stats)
    parsed = {r["k"]: r for r in read_jsonl(paths.parsed)}
    kinds = Counter()
    for k in info["pdfs"]:
        kind = truth[k]["pdf"]
        kinds[kind] += 1
        if kind == "digital":
            assert ts[k]["ocr_pages"] == 0
            got = {x["k"]: x["role"] for x in parsed[k]["predicates"] if x["role"] != "other"}
            t = truth[k]
            want = {t["primary"]: "primary", **{a: "additional" for a in t["additional"]}, **{r: "reference" for r in t["reference"]}} if t["primary"] else {}
            assert got == want, (k, got, want)
            assert alnum(parsed[k]["ifu"]) == alnum(t["ifu"])
        else:
            assert ts[k]["ocr_pages"] > 0                                 # scanned pages were OCR'd
            assert parsed[k]["ifu"]                                       # indications still found in OCR text
    assert kinds["scanned"] >= 1 and kinds["digital"] >= 1
