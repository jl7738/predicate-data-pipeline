"""Audit of the built dataset. Answers the reviewer's data/eval questions with numbers and PASS/WARN/FAIL checks.

  1. Funnel / attrition                      -> "report how many devices each step removes"
  2. Extraction health by year, doc length   -> "are these docs really long?", OCR share, section/predicate rates
  3. Token lengths + truncation policy       -> "check token lengths of the two sections; define truncation/chunking"
  4. Predicate coverage = recall ceiling     -> "how many cited predicates have usable records in the corpus"
  5. Candidates & gold per query, k          -> "how many candidates / cited predicates per query; why ten?"
  6. Sufficiency + power                     -> "are there enough examples?"  (split sizes, CI width, slices, seen/unseen)
  7. Leakage                                 -> raw vs masked
  8. Sanity baselines (BM25, popularity)     -> proves text/labels/dates are wired correctly
  9. Projection to the full run (pilot)      -> subjects per split, disk, OCR CPU-hours, download hours
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import Counter, defaultdict

import numpy as np

from .baselines import run_baselines
from .dataset import MISSING_REASONS, STATUS_ORDER, load_dataset, validate_dataset
from .io import read_jsonl
from .metrics import wilson
from .tokens import load_tokenizer, overhead_tokens


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def qstats(x, ps=(50, 90, 95, 99)):
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "min": float(x.min()), **{f"p{p}": float(np.percentile(x, p)) for p in ps}, "max": float(x.max())}


def _status(value, thr, kind, ff):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "FAIL"
    if kind == "min":
        return "PASS" if value >= thr else ("WARN" if value >= thr * ff else "FAIL")
    return "PASS" if value <= thr else ("WARN" if value <= thr / ff else "FAIL")


# ---------------------------------------------------------------------------------------------------------------
def _funnel_table(rows):
    c = Counter(r["status"] for r in rows)
    total = len(rows)
    se = total - c["decision_not_se"]
    summ = se - c["statement_only_meta"]
    proc = summ - c["not_processed"]
    fetched = proc - c["no_pdf_404"] - c["fetch_error"]
    text_ok = fetched - c["text_failed"]
    parsed = text_ok - c["parse_failed"]
    non_stmt = parsed - c["statement_like"]
    steps = [
        ("openFDA 510(k) records", total),
        ("substantially-equivalent decisions", se),
        ("summary available (statement-only dropped)", summ),
        ("attempted (queued and processed)", proc),
        ("PDF downloaded", fetched),
        ("text extracted", text_ok),
        ("parsed without error", parsed),
        ("not a statement-like document", non_stmt),
        ("usable: indications (+description) found", c["usable"]),
    ]
    out, prev = [], None
    for name, n in steps:
        out.append({"step": name, "n": n, "dropped": (prev - n) if prev is not None else 0, "pct_of_prev": round(pct(n, prev), 1) if prev else 100.0})
        prev = n
    return out, {"fetched": fetched, "attempted": proc, "text_ok": text_ok, "non_stmt": non_stmt, "usable": c["usable"]}


def _extraction(rows):
    processed = [r for r in rows if r["status"] not in ("decision_not_se", "statement_only_meta", "not_processed")]
    by_year = []
    for y in sorted({r["year"] for r in processed if r["year"]}):
        rs = [r for r in processed if r["year"] == y]
        n = len(rs)
        fetched = sum(r["status"] not in ("no_pdf_404", "fetch_error") for r in rs)
        text_ok = sum(r["status"] not in ("no_pdf_404", "fetch_error", "text_failed") for r in rs)
        usable = sum(r["status"] == "usable" for r in rs)
        with_pred = sum(r["status"] == "usable" and (r["n_pred"] or 0) > 0 for r in rs)
        ocr = sum((r.get("ocr_pages") or 0) > 0 for r in rs)
        by_year.append({
            "year": y, "n": n, "pdf_rate": round(pct(fetched, n), 1), "text_rate": round(pct(text_ok, n), 1),
            "usable_rate": round(pct(usable, n), 1), "pred_rate_of_usable": round(pct(with_pred, usable), 1),
            "ocr_doc_pct": round(pct(ocr, n), 1), "median_pages": float(np.median([r["n_pages"] for r in rs if r["n_pages"]] or [float("nan")])),
        })
    usable_rows = [r for r in processed if r["status"] == "usable"]
    ocr_s = [r["seconds"] for r in processed if r.get("seconds") and (r.get("ocr_pages") or 0) > 0]
    dig_s = [r["seconds"] for r in processed if r.get("seconds") and not (r.get("ocr_pages") or 0)]
    overall = {
        "pages": qstats([r["n_pages"] for r in processed if r["n_pages"]]),
        "chars_read": qstats([r["n_chars"] for r in processed if r["n_chars"]]),
        "ocr_doc_pct": round(pct(sum((r.get("ocr_pages") or 0) > 0 for r in processed), len(processed)), 1),
        "ifu_source": dict(Counter(r["ifu_source"] for r in usable_rows)),
        "primary_by": dict(Counter(r["primary_by"] for r in usable_rows)),
        "ifu_capped": sum(bool(r.get("ifu_capped")) for r in usable_rows),
        "desc_capped": sum(bool(r.get("desc_capped")) for r in usable_rows),
        "ambiguous_pct": round(pct(sum(bool(r.get("ambiguous")) for r in usable_rows), len(usable_rows)), 1),
        "pred_section_found_pct": round(pct(sum(bool(r.get("pred_section_found")) for r in usable_rows), len(usable_rows)), 1),
        "primary_equals_first_other_pct": round(pct(sum(bool(r.get("first_other_matches_primary")) for r in usable_rows if r["n_pred"]), sum(1 for r in usable_rows if r["n_pred"])), 1),
        "se_language_in_inputs_pct": round(pct(sum(bool(r.get("se_language_ifu") or r.get("se_language_desc")) for r in usable_rows), len(usable_rows)), 1),
        "ocr_conf_mean": float(np.mean([r["ocr_conf_mean"] for r in processed if r.get("ocr_conf_mean") is not None] or [float("nan")])),
        "ocr_conf_lt70_pct": round(pct(sum((r.get("ocr_conf_mean") or 100) < 70 for r in processed if (r.get("ocr_pages") or 0) > 0),
                                       sum(1 for r in processed if (r.get("ocr_pages") or 0) > 0)), 1),
        "sec_per_doc_ocr": float(np.mean(ocr_s)) if ocr_s else None,
        "sec_per_doc_digital": float(np.mean(dig_s)) if dig_s else None,
        "pdf_mb_mean": float(np.mean([r["pdf_bytes"] for r in processed if r.get("pdf_bytes")] or [0]) / 1e6),
    }
    folders = Counter((r["year"], r["fetch_folder"]) for r in processed if r.get("fetch_folder"))
    patterns = [{"year": y, "folder": f, "n": n} for (y, f), n in sorted(folders.items(), key=lambda t: (t[0][0] or 0, t[0][1]))]
    return {"by_year": by_year, "overall": overall, "fetch_patterns": patterns}


# ---------------------------------------------------------------------------------------------------------------
def _alloc_vec(ifu, desc, avail, policy, budget):
    if policy == "head":
        ki = np.minimum(ifu, avail)
    else:
        ki = np.minimum(np.minimum(ifu, budget), avail)
    kd = np.minimum(desc, avail - ki)
    if policy == "budget":
        spare = np.maximum(avail - ki - kd, 0)
        ki = ki + np.where(ifu > ki, np.minimum(spare, ifu - ki), 0)
    return ki, kd


def _tokens(corpus, cfg):
    tcfg, tmpl = cfg["truncation"], cfg["input_template"]
    grid = tcfg["max_len_grid"]
    out = []
    ifu_texts = [r["ifu"] for r in corpus]
    desc_texts = [r["description"] for r in corpus]
    for m in cfg["models"]:
        tok = load_tokenizer(m, tcfg["allow_approx_tokenizer"])
        ifu = np.asarray(tok.count_many(ifu_texts))
        desc = np.asarray(tok.count_many(desc_texts))
        over = overhead_tokens(tok, m["doc_prefix"], tmpl)
        total = ifu + desc + over
        n = len(total)
        rows = []
        for L in sorted(set(grid + [m["max_len"]])):
            for policy in ("head", "budget"):
                ki, kd = _alloc_vec(ifu, desc, L - over, policy, tcfg["ifu_budget_tokens"])
                with np.errstate(divide="ignore", invalid="ignore"):
                    kept = np.where(desc > 0, kd / np.maximum(desc, 1), 1.0)
                rows.append({
                    "max_len": L, "policy": policy,
                    "pct_any_truncated": round(pct(int(((ki < ifu) | (kd < desc)).sum()), n), 1),
                    "pct_desc_truncated": round(pct(int((kd < desc).sum()), n), 1),
                    "pct_ifu_truncated": round(pct(int((ki < ifu).sum()), n), 1),
                    "mean_desc_kept_pct": round(float(kept.mean() * 100), 1) if n else None,
                })
        pol = tcfg["policy"]
        rec = next((r["max_len"] for r in sorted(rows, key=lambda r: r["max_len"]) if r["policy"] == pol and r["pct_desc_truncated"] <= tcfg["threshold_pct"]), None)
        cfgrow = next(r for r in rows if r["max_len"] == m["max_len"] and r["policy"] == pol)
        out.append({
            "model": m["name"], "hf_id": m["hf_id"], "approx": tok.approx, "approx_reason": tok.reason, "n_special": tok.n_special,
            "overhead_tokens": over, "configured_max_len": m["max_len"], "policy": pol,
            "ifu": qstats(ifu), "description": qstats(desc), "total_with_overhead": qstats(total),
            "pct_exceeding": {str(L): round(pct(int((total > L).sum()), n), 1) for L in grid},
            "truncation": rows, "configured_desc_truncated_pct": cfgrow["pct_desc_truncated"],
            "recommended_max_len": rec,
            "chunking_recommended": rec is None,
            "decision": (f"truncate at max_len={rec} ({pol} policy)" if rec else f"even at {max(grid)} tokens more than {tcfg['threshold_pct']}% of descriptions are cut: chunk the description (window {tcfg['chunk']['size']}/stride {tcfg['chunk']['stride']}, max-pool chunk scores)"),
        })
    return out


# ---------------------------------------------------------------------------------------------------------------
YEAR_BINS = [(0, 1999, "<2000"), (2000, 2004, "2000-04"), (2005, 2009, "2005-09"), (2010, 2014, "2010-14"), (2015, 2019, "2015-19"), (2020, 2024, "2020-24"), (2025, 3000, "2025+")]


def _bin(y):
    if y is None:
        return "unknown"
    for a, b, n in YEAR_BINS:
        if a <= y <= b:
            return n
    return "unknown"


def _coverage(queries, corpus):
    allq = [(s, q) for s, qs in queries.items() for q in qs]
    pairs = [(s, q, g) for s, q in allq for g in q["gold"]]
    total = len(pairs)
    inc = sum(g["in_corpus"] for _, _, g in pairs)
    reasons = Counter(g["reason"] or "in_corpus" for _, _, g in pairs)
    by_split = {}
    for s in queries:
        ps = [g for sp, _, g in pairs if sp == s]
        qs = queries[s]
        by_split[s] = {
            "n_subjects": len(qs), "n_pairs": len(ps),
            "pair_coverage_pct": round(pct(sum(g["in_corpus"] for g in ps), len(ps)), 1),
            "subjects_retrievable_pct": round(pct(sum(q["retrievable"] for q in qs), len(qs)), 1),
            "mean_recall_ceiling": round(float(np.mean([len(q["gold_in_corpus"]) / len(q["gold"]) for q in qs])), 3) if qs else None,
            "primary_pair_coverage_pct": round(pct(sum(g["in_corpus"] for g in ps if g["role"] == "primary"), sum(1 for g in ps if g["role"] == "primary")), 1),
        }
    ybin = defaultdict(lambda: [0, 0])
    for _, _, g in pairs:
        b = _bin(g["year"])
        ybin[b][0] += 1
        ybin[b][1] += int(g["in_corpus"])
    order = [n for _, _, n in YEAR_BINS] + ["unknown"]
    by_year = [{"predicate_decision_year": b, "pairs": ybin[b][0], "in_corpus": ybin[b][1], "coverage_pct": round(pct(ybin[b][1], ybin[b][0]), 1)} for b in order if b in ybin]
    start_tbl = []
    for Y in (None, 1996, 2000, 2005, 2010, 2015):
        ok = lambda g: g["in_corpus"] and (Y is None or (g["year"] or 0) >= Y)
        hit = [any(ok(g) for g in q["gold"]) for _, q in allq]
        start_tbl.append({
            "corpus_starts": "all" if Y is None else Y,
            "corpus_size": sum(1 for r in corpus if Y is None or r["year"] >= Y),
            "pair_ceiling_pct": round(pct(sum(ok(g) for _, _, g in pairs), total), 1),
            "subject_hit_ceiling_pct": round(pct(sum(hit), len(hit)), 1),
        })
    return {
        "pairs_total": total, "pairs_in_corpus": inc, "pair_coverage_pct": round(pct(inc, total), 1),
        "by_reason": [{"reason": r, "pairs": reasons[r], "pct": round(pct(reasons[r], total), 1)} for r in ["in_corpus"] + MISSING_REASONS if reasons.get(r)],
        "by_split": by_split, "by_predicate_year": by_year, "ceiling_by_corpus_start": start_tbl,
    }


def _shortlist(queries, cfg):
    ks = cfg["eval_spec"]["recall_k"]
    out = {}
    for s, qs in queries.items():
        rq = [q for q in qs if q["retrievable"]]
        if not rq:
            out[s] = {"n_retrievable": 0}
            continue
        g_all = [len(q["gold"]) for q in qs]
        g_inc = np.array([len(q["gold_in_corpus"]) for q in rq])
        ncand = np.array([q["n_candidates"] for q in rq], dtype=float)
        same = np.array([q["n_same_code_candidates"] for q in rq], dtype=float)
        sub = []
        for k in ks:
            sub.append({
                "k": k,
                "pct_queries_gold_gt_k": round(pct(int((g_inc > k).sum()), len(g_inc)), 1),
                "mean_max_recall": round(float(np.mean(np.minimum(g_inc, k) / g_inc)), 3),
                "random_expected_recall": round(float(np.mean(np.minimum(1.0, k / np.maximum(ncand, 1)))), 5),
                "pct_queries_k_covers_all_same_code": round(pct(int((same <= k).sum()), len(same)), 1),
            })
        same_code_gold = [np.mean([g["product_code"] == q["product_code"] for g in q["gold"] if g["in_corpus"]]) for q in rq]
        out[s] = {
            "n_subjects": len(qs), "n_retrievable": len(rq),
            "candidates": qstats(ncand, (25, 50, 75)), "same_code_candidates": qstats(same, (25, 50, 75)),
            "gold_extracted": qstats(g_all, (25, 50, 75)), "gold_in_corpus": qstats(g_inc, (25, 50, 75)),
            "gold_count_dist": {"1": int((g_inc == 1).sum()), "2": int((g_inc == 2).sum()), "3": int((g_inc == 3).sum()), "4+": int((g_inc >= 4).sum())},
            "pct_gold_same_product_code": round(float(np.mean(same_code_gold)) * 100, 1),
            "pct_subjects_with_gold_in_same_code": round(pct(sum(1 for q in rq if any(g["in_corpus"] and g["product_code"] == q["product_code"] for g in q["gold"])), len(rq)), 1),
            "by_k": sub,
        }
    return out


def _sufficiency(queries, pairs, cfg, bm25_r10=None):
    ck = cfg["checks"]
    train_gold = Counter(p["p"] for p in pairs)
    out = {"subjects": {}, "power": {}, "slices": {}, "seen_unseen": {}, "train": {}}
    for s, qs in queries.items():
        out["subjects"][s] = {"with_predicate": len(qs), "retrievable": sum(q["retrievable"] for q in qs)}
    p = bm25_r10 if bm25_r10 is not None else ck["assumed_recall10"]
    out["power"]["assumed_recall10"] = p
    out["power"]["p_source"] = "measured BM25 Recall@10 on test" if bm25_r10 is not None else "assumed (config checks.assumed_recall10)"
    for s in ("dev", "test"):
        n = out["subjects"][s]["retrievable"]
        hw = 1.96 * math.sqrt(p * (1 - p) / n) if n else float("nan")
        mde = 2.8 * math.sqrt(2 * p * (1 - p) / n) if n else float("nan")
        out["power"][s] = {"n": n, "ci_halfwidth": hw, "min_detectable_diff_unpaired": mde}
    rq = [q for q in queries["test"] if q["retrievable"]]
    for name, lo, hi in [("0 earlier same-code clearances", 0, 0), ("1-4", 1, 4), ("5-19", 5, 19), ("20+", 20, 10**9)]:
        n = sum(lo <= q["n_same_code_candidates"] <= hi for q in rq)
        out["slices"][name] = {"n": n, "ok": n >= ck["min_slice_n"]}
    for s in ("dev", "test"):
        gs = [(q, g) for q in queries[s] if q["retrievable"] for g in q["gold"] if g["in_corpus"]]
        seen = sum(g["k"] in train_gold for _, g in gs)
        per_q = [[g["k"] in train_gold for g in q["gold"] if g["in_corpus"]] for q in queries[s] if q["retrievable"]]
        out["seen_unseen"][s] = {
            "pairs": len(gs), "pct_pairs_seen_in_train": round(pct(seen, len(gs)), 1),
            "pct_queries_any_unseen_gold": round(pct(sum(not all(x) for x in per_q), len(per_q)), 1),
            "pct_queries_all_gold_unseen": round(pct(sum(not any(x) for x in per_q), len(per_q)), 1),
        }
    cnt = np.array(sorted(train_gold.values(), reverse=True))
    ntq = len(queries["train"])
    out["train"] = {
        "train_subjects": ntq, "positive_pairs": int(cnt.sum()) if len(cnt) else 0, "distinct_positives": int(len(cnt)),
        "mean_positives_per_subject": round(float(cnt.sum()) / ntq, 2) if ntq else None,
        "share_pairs_from_top1pct_predicates": round(float(cnt[: max(1, len(cnt) // 100)].sum() / cnt.sum()), 3) if len(cnt) else None,
        "pct_pairs_on_predicates_cited_10plus": round(pct(int(cnt[cnt >= 10].sum()), int(cnt.sum())), 1) if len(cnt) else None,
        "max_citations_of_one_predicate": int(cnt[0]) if len(cnt) else None,
    }
    return out


def _leakage(queries, card):
    qs = [q for s in queries.values() for q in s]
    n = len(qs)
    def rate(f):
        return round(pct(sum(1 for q in qs if f(q)), n), 2)
    return {
        "n_queries": n, "mask_level": card.get("mask_level"),
        "raw_gold_k_in_text_pct": rate(lambda q: q["leak"]["raw_gold_k"] > 0),
        "raw_any_other_k_in_text_pct": rate(lambda q: q["leak"]["raw_other_k"] > 0),
        "raw_predicate_language_pct": rate(lambda q: q["leak"]["pred_language"]),
        "raw_gold_name_in_text_pct": rate(lambda q: q["leak"]["raw_gold_name"] > 0),
        "residual_k_after_mask_pct": rate(lambda q: q["leak"]["residual_k"] > 0),
        "residual_gold_name_after_mask_pct": rate(lambda q: q["leak"]["residual_gold_name"] > 0),
    }


def _projection(rows, cfg, paths):
    win_all = [r for r in rows if r["split"] and r["status"] not in ("decision_not_se", "statement_only_meta")]
    remaining = sum(r["status"] == "not_processed" for r in win_all)
    if not remaining:
        return None
    sample = [r for r in win_all if r.get("in_subject_queue", True)]
    proc = [r for r in sample if r["status"] != "not_processed"]
    if len(proc) < 20:
        return {"note": f"only {len(proc)} processed subject-window devices in the sample; too few to project"}
    usable = [r for r in proc if r["status"] == "usable"]
    with_pred = [r for r in usable if r["n_pred"] > 0]
    usable_rate = len(usable) / len(proc)
    pred_rate = len(with_pred) / len(usable) if usable else 0
    out = {"processed_in_windows": len(proc), "remaining_in_windows": remaining, "usable_rate": usable_rate, "pred_rate": pred_rate}
    out["panels_in_sample"] = dict(Counter(r["panel"] for r in proc).most_common(5))
    cnt = Counter(r["split"] for r in win_all)
    out["summary_devices_by_split"] = dict(cnt)
    lo_u, hi_u = wilson(len(usable), len(proc))
    out["projected_usable_with_pred"] = {s: [round(n * lo_u * pred_rate), round(n * usable_rate * pred_rate), round(n * hi_u * pred_rate)] for s, n in cnt.items()}
    all_rem = sum(r["status"] == "not_processed" for r in rows if r["status"] not in ("decision_not_se", "statement_only_meta"))
    secs = [r["seconds"] for r in rows if r.get("seconds")]
    bts = [r["pdf_bytes"] for r in rows if r.get("pdf_bytes")]
    workers = cfg["text"].get("workers") or os.cpu_count() or 1
    out["remaining_all_summary_devices"] = all_rem
    out["mean_extract_seconds_per_doc"] = float(np.mean(secs)) if secs else None
    out["mean_pdf_mb"] = float(np.mean(bts)) / 1e6 if bts else None
    if secs:
        out["projected_extract_cpu_hours"] = all_rem * float(np.mean(secs)) / 3600
        out["projected_extract_wall_hours_at_workers"] = [workers, all_rem * float(np.mean(secs)) / 3600 / workers]
    if bts:
        out["projected_pdf_gb"] = all_rem * float(np.mean(bts)) / 1e9
    out["projected_download_hours"] = all_rem / max(cfg["fetch"]["requests_per_second"], 0.1) / 3600
    return out


# ---------------------------------------------------------------------------------------------------------------
def run_audit(cfg: dict, paths, baselines: bool = True, tokenizers: bool = True) -> dict:
    ck = cfg["checks"]
    ff = ck["fail_fraction"]
    rows = list(read_jsonl(paths.dataset / "device_status.jsonl.gz"))
    corpus, queries, pairs, card = load_dataset(paths)
    rep: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "n_meta": len(rows), "n_corpus": len(corpus)}

    all_tbl, all_cnt = _funnel_table(rows)
    win_rows = [r for r in rows if r["split"] and r.get("in_subject_queue", True)]
    win_tbl, win_cnt = _funnel_table(win_rows)
    rep["funnel_all"], rep["funnel_subject_windows"] = all_tbl, win_tbl
    rep["status_counts"] = {s: sum(1 for r in rows if r["status"] == s) for s in STATUS_ORDER}
    rep["extraction"] = _extraction(rows)
    rep["tokens"] = _tokens(corpus, cfg) if (tokenizers and corpus) else []
    rep["coverage"] = _coverage(queries, corpus)
    rep["shortlist"] = _shortlist(queries, cfg)
    rep["baselines"] = {}
    bm25_r10 = None
    if baselines and corpus and any(q["retrievable"] for q in queries["test"]):
        try:
            rep["baselines"] = run_baselines(cfg, paths, "test")
            bm25_r10 = rep["baselines"].get("bm25", {}).get("recall@10", {}).get("mean")
        except Exception as e:  # never let a baseline crash the audit
            rep["baselines"] = {"error": repr(e)}
    rep["sufficiency"] = _sufficiency(queries, pairs, cfg, bm25_r10)
    rep["leakage"] = _leakage(queries, card)
    rep["projection"] = _projection(rows, cfg, paths)
    rep["eval_spec"] = cfg["eval_spec"]
    rep["models"] = [{k: m[k] for k in ("name", "hf_id", "max_len", "zero_shot_baseline") if k in m} for m in cfg["models"]]
    val = validate_dataset(paths, cfg)
    rep["validation"] = val

    # ---- checks ------------------------------------------------------------------------------------------------
    checks = []

    def add(cid, desc, value, thr, kind="min", note=""):
        checks.append({"id": cid, "description": desc, "value": value, "threshold": thr, "kind": kind, "status": _status(value, thr, kind, ff), "note": note})

    nprob = sum(len(v) for v in val["problems"].values())
    checks.append({"id": "schema", "description": "dataset files well-formed; invariants hold (dates, gold subset of corpus, no self-citation, masked K-numbers)",
                   "value": nprob, "threshold": 0, "kind": "max", "status": "PASS" if nprob == 0 else "FAIL", "note": "; ".join(val["problems"].keys())})
    add("fetch_rate", "PDF download success (attempted devices)", pct(win_cnt["fetched"], win_cnt["attempted"]) / 100 if win_cnt["attempted"] else float("nan"), ck["min_fetch_rate"])
    add("text_rate", "text extraction success (downloaded PDFs)", win_cnt["text_ok"] / win_cnt["fetched"] if win_cnt["fetched"] else float("nan"), ck["min_text_rate"])
    add("sections_rate", "indications (+description) found (non-statement docs)", win_cnt["usable"] / win_cnt["non_stmt"] if win_cnt["non_stmt"] else float("nan"), ck["min_sections_rate"])
    usable_w = [r for r in win_rows if r["status"] == "usable"]
    add("pred_rate", "usable subject-window devices with >=1 extracted predicate", sum(r["n_pred"] > 0 for r in usable_w) / len(usable_w) if usable_w else float("nan"), ck["min_pred_rate"])
    add("pair_coverage", "(subject, predicate) pairs whose predicate is in the corpus = recall ceiling", rep["coverage"]["pair_coverage_pct"] / 100 if rep["coverage"]["pairs_total"] else float("nan"), ck["min_pair_coverage"])
    for s in ("train", "dev", "test"):
        add(f"n_{s}", f"retrievable {s} subjects", rep["sufficiency"]["subjects"][s]["retrievable"], ck["min_subjects"][s])
        tot = rep["sufficiency"]["subjects"][s]["with_predicate"]
        add(f"retrievable_{s}", f"share of {s} subjects with >=1 gold predicate in corpus", rep["sufficiency"]["subjects"][s]["retrievable"] / tot if tot else float("nan"), ck["min_retrievable_rate"])
    for t in rep["tokens"]:
        add(f"truncation_{t['model']}", f"% of descriptions truncated at max_len={t['configured_max_len']} ({t['policy']})" + (" [APPROX tokenizer]" if t["approx"] else ""),
            t["configured_desc_truncated_pct"], ck["max_truncated_pct"], "max")
    for s in ("test",):
        add("ci_halfwidth_test", "95% CI half-width of Recall@10 on test (" + rep["sufficiency"]["power"]["p_source"] + ")", rep["sufficiency"]["power"][s]["ci_halfwidth"], ck["max_ci_halfwidth"], "max")
    small = [k for k, v in rep["sufficiency"]["slices"].items() if not v["ok"]]
    checks.append({"id": "slices", "description": f"every prior-clearance slice on test has n >= {ck['min_slice_n']}", "value": len(small), "threshold": 0, "kind": "max",
                   "status": "PASS" if not small else "WARN", "note": ", ".join(small)})
    lk = rep["leakage"]
    checks.append({"id": "leak_residual_k", "description": "no K-numbers left in query text after masking", "value": lk["residual_k_after_mask_pct"], "threshold": 0, "kind": "max",
                   "status": "PASS" if (lk["residual_k_after_mask_pct"] == 0 or lk["mask_level"] == "none") else "FAIL", "note": f"mask level = {lk['mask_level']}"})
    add("leak_gold_name", "queries still naming a cited predicate after masking (fraction)", lk["residual_gold_name_after_mask_pct"] / 100, ck["max_gold_name_leak_rate"], "max")
    if cfg["expected"].get("enabled"):
        tol, ex = cfg["expected"]["tolerance"], cfg["expected"]
        n_sum = sum(1 for r in rows if r["sos"] == "SUMMARY")
        n_win = sum(1 for r in rows if r["sos"] == "SUMMARY" and r["split"])
        for cid, got, exp in (("expected_total", len(rows), ex["total_decisions"]), ("expected_summary", n_sum, ex["with_summary"]), ("expected_subjects", n_win, ex["subjects_2015_2026"])):
            dev = abs(got - exp) / exp if exp else float("nan")
            checks.append({"id": cid, "description": f"metadata count vs proposal ({exp:,})", "value": got, "threshold": exp, "kind": "approx",
                           "status": "PASS" if dev <= tol else "WARN", "note": f"{dev * 100:.1f}% off (tolerance {tol * 100:.0f}%)"})
    if rep["baselines"].get("bm25"):
        b = rep["baselines"]
        got, rnd = b["bm25"]["hit@10"]["mean"], b["random_expected"]["hit@10"]
        thr = min(3 * rnd, (1 + rnd) / 2)          # "3x random", capped so it stays reachable when the corpus is tiny
        st = "PASS" if got >= thr else "FAIL"
        note = f"random expected {rnd:.4f}"
        if b["n_queries"] < 30:
            st, note = ("PASS" if got >= thr else "WARN"), note + f"; only {b['n_queries']} queries, too few to judge"
        checks.append({"id": "baseline_sanity", "description": "BM25 Hit@10 is well above random (text/labels/dates wired correctly)", "value": got, "threshold": thr, "kind": "min", "status": st, "note": note})
    if any(t["approx"] for t in rep["tokens"]):
        checks.append({"id": "tokenizer_approx", "description": "token counts use an APPROXIMATE tokenizer: re-run on the VM with HF_TOKEN for real numbers", "value": None, "threshold": None,
                       "kind": "info", "status": "WARN", "note": "; ".join(f"{t['model']}: {t['approx_reason'][:80]}" for t in rep["tokens"] if t["approx"])})
    rep["checks"] = checks
    rep["overall"] = "FAIL" if any(c["status"] == "FAIL" for c in checks) else ("WARN" if any(c["status"] == "WARN" for c in checks) else "PASS")
    return rep


# ---------------------------------------------------------------------------------------------------------------
def _f(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        if abs(v) >= 100:
            return f"{v:,.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        return f"{v:.{nd}f}".rstrip("0").rstrip(".") if v != int(v) else f"{int(v)}"
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    return str(v)


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_f(c) for c in r) + " |")
    return "\n".join(out)


def _dict_table(rows, cols=None):
    if not rows:
        return "_(none)_"
    cols = cols or list(rows[0].keys())
    return _table(cols, [[r.get(c) for c in cols] for r in rows])


def render_markdown(rep: dict) -> str:
    L = []
    a = L.append
    a(f"# Data audit  —  overall: **{rep['overall']}**")
    a(f"_generated {rep['generated_at']} · {rep['n_meta']:,} metadata records · {rep['n_corpus']:,} corpus devices · mask level `{rep['leakage']['mask_level']}`_\n")
    a("Results are framed as **recovery of historical citations**: an uncited device is not necessarily an unsuitable predicate.\n")

    a("## 0. Checks")
    a(_table(["status", "check", "value", "threshold", "note"], [[c["status"], c["description"], c["value"], c["threshold"], c["note"]] for c in rep["checks"]]))

    a("\n## 1. Funnel (what each step removes)")
    a("**All openFDA records**\n")
    a(_dict_table(rep["funnel_all"]))
    a("\n**Subjects only (decision date inside the train/dev/test windows; in pilot mode: the pilot sample)**\n")
    a(_dict_table(rep["funnel_subject_windows"]))
    a("\nFinal status of every record:\n")
    a(_table(["status", "n"], list(rep["status_counts"].items())))

    ex = rep["extraction"]
    o = ex["overall"]
    a("\n## 2. Extraction health")
    a(f"- Document length (pages): {_f(o['pages'].get('p50'))} median, p90 {_f(o['pages'].get('p90'))}, max {_f(o['pages'].get('max'))}; characters read: median {_f(o['chars_read'].get('p50'))}, p90 {_f(o['chars_read'].get('p90'))}")
    a(f"- Documents needing OCR on at least one page: **{o['ocr_doc_pct']}%**; mean seconds/doc — OCR: {_f(o['sec_per_doc_ocr'], 1)}, digital: {_f(o['sec_per_doc_digital'], 1)}; mean PDF size {_f(o['pdf_mb_mean'], 2)} MB")
    if o.get("ocr_conf_mean") == o.get("ocr_conf_mean"):   # not NaN
        a(f"- OCR'd documents: mean tesseract word confidence {_f(o['ocr_conf_mean'], 1)}/100; {o['ocr_conf_lt70_pct']}% below 70. "
          "Low confidence = unreliable text, and **K-numbers read from OCR are the riskiest labels** (digit confusions such as 0/6/9 are NOT reflected in the confidence): "
          "make sure the hand-labelled sample includes OCR'd documents (`label-sample` flags them) and compare precision on them.")
    a(f"- Indications source: {o['ifu_source']}; sections hitting the section-length cap: indications {o['ifu_capped']}, description {o['desc_capped']}")
    a(f"- Predicate section located: {o['pred_section_found_pct']}% of usable docs; primary chosen by: {o['primary_by']}; ambiguous extractions: {o['ambiguous_pct']}%")
    a(f"- Rule-based primary equals the naive first-other-number in {o['primary_equals_first_other_pct']}% of docs (where they differ, the labelling sample is what tells you which is right)")
    a(f"- Model-input text containing predicate / substantial-equivalence language: {o['se_language_in_inputs_pct']}% (candidate leakage; see §7)\n")
    a(_dict_table(ex["by_year"]))
    a("\nWhich PDF folder spelling worked per year (verifies the URL pattern):\n")
    a(_dict_table(ex["fetch_patterns"]))

    a("\n## 3. Token lengths and truncation / chunking policy")
    if not rep["tokens"]:
        a("_(skipped)_")
    for t in rep["tokens"]:
        a(f"\n### {t['model']}  (`{t['hf_id']}`){'  — **APPROX tokenizer: ' + t['approx_reason'][:90] + '**' if t['approx'] else ''}")
        a(f"Prompt + headers + special tokens consume **{t['overhead_tokens']}** tokens before any text.\n")
        a(_table(["field", "mean", "p50", "p90", "p95", "p99", "max"], [[n, t[k].get("mean"), t[k].get("p50"), t[k].get("p90"), t[k].get("p95"), t[k].get("p99"), t[k].get("max")] for n, k in (("indications", "ifu"), ("description", "description"), ("total incl. overhead", "total_with_overhead"))]))
        a("\nShare of documents whose full input exceeds L tokens: " + ", ".join(f"L={k}: {v}%" for k, v in t["pct_exceeding"].items()))
        a("\n" + _dict_table(t["truncation"]))
        a(f"\n**Decision:** {t['decision']}.  Configured max_len={t['configured_max_len']} truncates {t['configured_desc_truncated_pct']}% of descriptions under `{t['policy']}`.")

    cv = rep["coverage"]
    a("\n## 4. Predicate coverage = the recall ceiling")
    a(f"{cv['pairs_in_corpus']:,} of {cv['pairs_total']:,} (subject, predicate) pairs have the predicate in the searchable corpus (**{cv['pair_coverage_pct']}%**). Recall@k can never exceed this on all-gold scoring; the rest are listed by cause.\n")
    a(_dict_table(cv["by_reason"]))
    a("\nPer split:\n")
    a(_table(["split", "subjects", "pairs", "pair coverage %", "primary coverage %", "subjects retrievable %", "mean Recall@inf ceiling"], [[s, v["n_subjects"], v["n_pairs"], v["pair_coverage_pct"], v["primary_pair_coverage_pct"], v["subjects_retrievable_pct"], v["mean_recall_ceiling"]] for s, v in cv["by_split"].items()]))
    a("\nCoverage by the predicate's own decision year (old predicates are the scanned, hard-to-extract ones):\n")
    a(_dict_table(cv["by_predicate_year"]))
    a("\nCeiling if the corpus only starts at year Y (shows what a smaller corpus costs):\n")
    a(_dict_table(cv["ceiling_by_corpus_start"]))

    a("\n## 5. Candidates and gold per query; what a top-k shortlist means")
    for s, v in rep["shortlist"].items():
        if not v.get("n_retrievable"):
            a(f"\n**{s}**: no retrievable subjects")
            continue
        a(f"\n**{s}** — {v['n_subjects']:,} subjects, {v['n_retrievable']:,} retrievable")
        a(f"- candidates per query: median {_f(v['candidates'].get('p50'))}, mean {_f(v['candidates'].get('mean'))}, min {_f(v['candidates'].get('min'))}, max {_f(v['candidates'].get('max'))}")
        a(f"- candidates sharing the subject's product code: median {_f(v['same_code_candidates'].get('p50'))}, p25 {_f(v['same_code_candidates'].get('p25'))}, p75 {_f(v['same_code_candidates'].get('p75'))}")
        a(f"- gold predicates in corpus per subject: mean {_f(v['gold_in_corpus'].get('mean'))}, median {_f(v['gold_in_corpus'].get('p50'))}, distribution {v['gold_count_dist']}; extracted (before corpus filter): mean {_f(v['gold_extracted'].get('mean'))}")
        a(f"- {v['pct_gold_same_product_code']}% of gold predicates share the subject's product code; {v['pct_subjects_with_gold_in_same_code']}% of subjects have at least one gold predicate in their own product code")
        a("\n" + _dict_table(v["by_k"]))

    su = rep["sufficiency"]
    a("\n## 6. Are there enough examples?")
    a(_table(["split", "subjects with predicate", "retrievable"], [[s, v["with_predicate"], v["retrievable"]] for s, v in su["subjects"].items()]))
    a(f"\nPower ({su['power']['p_source']}, p = {_f(su['power']['assumed_recall10'])}): " + "; ".join(f"{s}: n={su['power'][s]['n']:,}, 95% CI ±{_f(su['power'][s]['ci_halfwidth'])}, smallest reliably detectable Recall@10 gap ≈ {_f(su['power'][s]['min_detectable_diff_unpaired'])} (unpaired, conservative)" for s in ("dev", "test")))
    a("\nTest slices by number of earlier clearances with the same product code (hypothesis H3):\n")
    a(_table(["slice", "n", "enough?"], [[k, v["n"], "yes" if v["ok"] else "NO"] for k, v in su["slices"].items()]))
    a("\nSeen vs unseen predicates (is the model evaluated on predicates it never saw as a positive in training?):\n")
    a(_table(["split", "gold pairs", "% predicates seen in train", "% queries with any unseen gold", "% queries with ALL gold unseen"], [[s, v["pairs"], v["pct_pairs_seen_in_train"], v["pct_queries_any_unseen_gold"], v["pct_queries_all_gold_unseen"]] for s, v in su["seen_unseen"].items()]))
    tr = su["train"]
    a(f"\nTraining signal: {tr['positive_pairs']:,} positive pairs over {tr['train_subjects']:,} subjects ({tr['mean_positives_per_subject']} per subject), {tr['distinct_positives']:,} distinct predicates; top 1% of predicates take {tr['share_pairs_from_top1pct_predicates']} of pairs; {tr['pct_pairs_on_predicates_cited_10plus']}% of pairs are on predicates cited 10+ times (popularity skew); most-cited predicate: {tr['max_citations_of_one_predicate']}.")

    lk = rep["leakage"]
    a("\n## 7. Leakage")
    a(_table(["measure", "% of queries"], [["RAW text contains a gold predicate's K-number", lk["raw_gold_k_in_text_pct"]], ["RAW text contains any other K-number", lk["raw_any_other_k_in_text_pct"]], ["RAW text uses predicate / substantial-equivalence language", lk["raw_predicate_language_pct"]], ["RAW text names a gold predicate", lk["raw_gold_name_in_text_pct"]], [f"after `{lk['mask_level']}` masking: K-number remains", lk["residual_k_after_mask_pct"]], [f"after `{lk['mask_level']}` masking: gold name remains", lk["residual_gold_name_after_mask_pct"]]]))
    a("\nTo check masking is not doing the work: rebuild with `--mask-level strict`, re-run the baselines, confirm performance does not collapse.")

    b = rep.get("baselines") or {}
    a("\n## 8. Sanity baselines (wiring check, not the paper's numbers)")
    if b.get("bm25") or b.get("product_code_popularity"):
        ks = [f"recall@{k}" for k in rep["eval_spec"]["recall_k"]] + ["hit@1", "hit@10", "mrr", f"ndcg@{rep['eval_spec']['ndcg_k']}"]
        rows = []
        for name in ("bm25", "product_code_popularity"):
            if b.get(name):
                rows.append([name] + [f"{b[name][k]['mean']:.3f} [{b[name][k]['lo']:.3f}, {b[name][k]['hi']:.3f}]" if k in b[name] else "-" for k in ks])
        rows.append(["random (expected)"] + [f"{b['random_expected'][k]:.5f}" if k in b["random_expected"] else "-" for k in ks])
        a(f"{b['n_queries']:,} {b['split']} subjects (retrievable only), bootstrap 95% CI.\n")
        a(_table(["system"] + ks, rows))
    else:
        a(f"_(not run)_ {b.get('error', '')}")

    pj = rep.get("projection")
    a("\n## 9. Projection to the full run")
    if not pj:
        a("_All summary devices in the windows are processed; nothing to project._")
    elif "note" in pj:
        a(pj["note"])
    else:
        a(f"From {pj['processed_in_windows']:,} processed subject-window devices (panels: {pj['panels_in_sample']}), usable rate {pj['usable_rate']:.2f}, predicate-found rate {pj['pred_rate']:.2f}. **Panel-specific rates may differ for other specialties.**\n")
        a(_table(["split", "summary devices (metadata)", "projected subjects with predicate [low, point, high]"], [[s, n, str(pj["projected_usable_with_pred"][s])] for s, n in pj["summary_devices_by_split"].items()]))
        bits = []
        if pj.get("projected_pdf_gb") is not None:
            bits.append(f"PDF storage ≈ {pj['projected_pdf_gb']:.0f} GB (mean {pj['mean_pdf_mb']:.2f} MB)")
        if pj.get("projected_extract_cpu_hours") is not None:
            w, h = pj["projected_extract_wall_hours_at_workers"]
            bits.append(f"extraction ≈ {pj['projected_extract_cpu_hours']:.0f} CPU-hours ≈ {h:.1f} h wall-clock on {w} workers")
        bits.append(f"download ≈ {pj['projected_download_hours']:.1f} h at the configured request rate")
        a("\nRemaining work for all " + f"{pj['remaining_all_summary_devices']:,} unprocessed summary devices: " + "; ".join(bits) + ".")

    a("\n## 10. Run plan recorded for modelling (from config)")
    es = rep["eval_spec"]
    a(f"- Similarity: **{es['similarity']}** on L2-normalised embeddings (same for before/after fine-tuning); ablation: {es['similarity_ablation']}")
    a(f"- Metrics: Recall@{es['recall_k']}, Hit@{es['hit_k']}, MRR, nDCG@{es['ndcg_k']}; bootstrap 95% CI ({es['bootstrap']} resamples)")
    a(f"- Candidate rule: {es['candidate_rule']}")
    a("- Every model gets its own un-fine-tuned baseline: " + ", ".join(f"{m['name']} (zero-shot: {m.get('zero_shot_baseline', False)})" for m in rep["models"]))
    return "\n".join(L) + "\n"


def write_report(rep: dict, paths) -> None:
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "audit.json").write_text(json.dumps(rep, indent=2, default=str))
    (paths.reports / "audit.md").write_text(render_markdown(rep))
    tdir = paths.reports / "tables"
    tdir.mkdir(exist_ok=True)

    def dump(name, rows):
        if not rows:
            return
        with open(tdir / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    dump("funnel_all", rep["funnel_all"])
    dump("funnel_subject_windows", rep["funnel_subject_windows"])
    dump("extraction_by_year", rep["extraction"]["by_year"])
    dump("coverage_by_reason", rep["coverage"]["by_reason"])
    dump("coverage_by_predicate_year", rep["coverage"]["by_predicate_year"])
    dump("ceiling_by_corpus_start", rep["coverage"]["ceiling_by_corpus_start"])
    for t in rep["tokens"]:
        dump(f"truncation_{t['model']}", t["truncation"])
    for s, v in rep["shortlist"].items():
        dump(f"shortlist_by_k_{s}", v.get("by_k", []))
    dump("checks", [{k: c[k] for k in ("id", "status", "description", "value", "threshold", "note")} for c in rep["checks"]])
