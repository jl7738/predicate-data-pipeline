"""Stage `build`: parsed records -> evaluation-ready dataset, plus a per-device status table for the audit.

Outputs (data/dataset/):
  corpus.jsonl.gz          every usable device, sorted by (date, k). Candidate set of a query = rows [0, n_candidates)
  queries_{train,dev,test}.jsonl.gz   subjects with gold predicates (and WHY any gold predicate is missing from the corpus)
  train_pairs.jsonl.gz     (subject, predicate, role) positives for contrastive training
  device_status.jsonl.gz   one row per openFDA record: how far it got through the funnel and why it stopped
  dataset_card.json        counts + the config the downstream modelling code must honour

Definitions
  usable device   : SE decision, summary available, text extracted, sections found (policy: both | ifu_only)
  subject         : usable device decided inside a split window with >= 1 extracted primary/additional predicate
  gold            : primary + additional predicates (reference devices excluded, stored separately)
  retrievable     : >= 1 gold predicate is in the corpus with decision_date strictly before the subject's
  candidate set   : corpus devices with decision_date strictly before the subject's decision_date
"""
from __future__ import annotations

import json
import logging
import time
from bisect import bisect_left
from collections import Counter, defaultdict

from .io import last_by_key, read_jsonl, write_jsonl
from .knumbers import is_510k
from .leakage import leak_stats, mask_doc_text, mask_query_text, residual_leak
from .metadata import is_se, load_meta
from .plan import in_any_window

log = logging.getLogger("predx.dataset")

STATUS_ORDER = [
    "decision_not_se",
    "statement_only_meta",
    "not_processed",
    "no_pdf_404",
    "fetch_error",
    "text_failed",
    "parse_failed",
    "statement_like",
    "sections_missing",
    "usable",
]
MISSING_REASONS = [
    "non_510k_identifier",
    "not_in_openfda",
    "temporal_violation",
    "decision_not_se",
    "statement_only",
    "statement_only_detected",
    "before_corpus_start",
    "no_pdf_404",
    "fetch_error",
    "text_failed",
    "parse_failed",
    "sections_missing",
    "not_processed",
    "no_decision_date",
]


def is_usable(p: dict | None, cfg: dict) -> bool:
    if not p or p.get("parse_error"):
        return False
    mw = cfg["parse"]["min_words"]
    ok_ifu = p.get("ifu_words", 0) >= mw["ifu"]
    ok_desc = p.get("desc_words", 0) >= mw["description"]
    return ok_ifu if cfg["corpus"]["usable_policy"] == "ifu_only" else (ok_ifu and ok_desc)


def classify_device(m: dict, p, ts, fs, cfg: dict) -> str:
    if not is_se(m, cfg):
        return "decision_not_se"
    if (m.get("statement_or_summary") or "").upper() == "STATEMENT":
        return "statement_only_meta"
    if p is not None:
        if p.get("parse_error"):
            return "parse_failed"
        if p.get("statement_like"):
            return "statement_like"
        return "usable" if is_usable(p, cfg) else "sections_missing"
    if ts is not None and ts.get("error"):
        return "text_failed"
    if fs is not None:
        if fs["status"] == "not_found":
            return "no_pdf_404"
        if fs["status"] == "error":
            return "fetch_error"
    return "not_processed"


_STATUS_TO_REASON = {
    "decision_not_se": "decision_not_se",
    "statement_only_meta": "statement_only",
    "statement_like": "statement_only_detected",
    "no_pdf_404": "no_pdf_404",
    "fetch_error": "fetch_error",
    "text_failed": "text_failed",
    "parse_failed": "parse_failed",
    "sections_missing": "sections_missing",
    "not_processed": "not_processed",
}


def missing_reason(k: str, kind: str, meta: dict, status: dict, subject_date: str, min_date) -> str:
    if kind not in ("k", "bk"):
        return "non_510k_identifier"
    pm = meta.get(k)
    if pm is None:
        return "not_in_openfda"
    if not pm.get("decision_date"):
        return "no_decision_date"
    if pm["decision_date"] >= subject_date:
        return "temporal_violation"
    st = status.get(k)
    if st == "usable":
        return "before_corpus_start" if (min_date and pm["decision_date"] < min_date) else "not_processed"
    return _STATUS_TO_REASON.get(st, "not_processed")


def build_dataset(cfg: dict, paths, mask_level: str | None = None, subjects_queue: str | None = None) -> dict:
    """subjects_queue: restrict SUBJECTS (queries) to the K-numbers in this queue, e.g. the pilot sample. Devices
    fetched only because a subject cited them (predicate closure) still enter the corpus as candidates, but are not
    evaluated as subjects: otherwise the pilot sample would be biased toward frequently-cited devices."""
    from .plan import read_queue

    t0 = time.time()
    level = mask_level or cfg["leakage"]["mask_level"]
    subj_set = set(read_queue(paths, subjects_queue)) if subjects_queue else None
    meta = load_meta(paths)
    parsed = {r["k"]: r for r in read_jsonl(paths.parsed)}
    tstats = last_by_key(paths.text_stats)
    fstat = last_by_key(paths.fetch_status)
    min_date = cfg["corpus"].get("min_decision_date")
    out = paths.dataset
    out.mkdir(parents=True, exist_ok=True)

    status = {k: classify_device(m, parsed.get(k), tstats.get(k), fstat.get(k), cfg) for k, m in meta.items()}

    # ---- device_status (funnel table) ------------------------------------------------------------------------
    def status_rows():
        for k, m in meta.items():
            p, ts, fs = parsed.get(k) or {}, tstats.get(k) or {}, fstat.get(k) or {}
            preds = [x for x in p.get("predicates", []) if x["role"] in ("primary", "additional")]
            primary = next((x for x in p.get("predicates", []) if x["role"] == "primary"), None)
            yield {
                "k": k, "year": m.get("year"), "date": m.get("decision_date"), "product_code": m.get("product_code"),
                "panel": m.get("panel"), "sos": m.get("statement_or_summary"), "status": status[k],
                "split": in_any_window(m.get("decision_date"), cfg),
                "in_subject_queue": subj_set is None or k in subj_set,
                "n_pages": ts.get("n_pages"), "n_chars": ts.get("n_chars"), "ocr_pages": ts.get("ocr_pages"),
                "ocr_conf_mean": ts.get("ocr_conf_mean"), "seconds": ts.get("seconds"), "pdf_bytes": fs.get("bytes"), "fetch_folder": fs.get("folder"),
                "ifu_source": p.get("ifu_source"), "ifu_words": p.get("ifu_words"), "desc_words": p.get("desc_words"),
                "ifu_capped": p.get("ifu_capped"), "desc_capped": p.get("desc_capped"),
                "se_language_ifu": p.get("se_language_ifu"), "se_language_desc": p.get("se_language_desc"),
                "n_pred": len(preds), "primary_by": p.get("primary_by"), "ambiguous": p.get("ambiguous"),
                "pred_section_found": p.get("pred_section_found"),
                "first_other_matches_primary": (primary is not None and p.get("first_other") == primary["k"]) if p else None,
            }

    write_jsonl(out / "device_status.jsonl.gz", status_rows())

    # ---- corpus ------------------------------------------------------------------------------------------------
    corpus = []
    for k, m in meta.items():
        if status[k] != "usable" or not m.get("decision_date"):
            continue
        if min_date and m["decision_date"] < min_date:
            continue
        p = parsed[k]
        ifu, _ = mask_doc_text(p.get("ifu") or "", level)
        desc, _ = mask_doc_text(p.get("description") or "", level)
        corpus.append({
            "k": k, "date": m["decision_date"], "year": m["year"], "product_code": m["product_code"], "panel": m["panel"],
            "device_name": m["device_name"], "applicant": m["applicant"], "ifu": ifu, "description": desc,
            "n_pages": (tstats.get(k) or {}).get("n_pages"),
        })
    corpus.sort(key=lambda r: (r["date"], r["k"]))
    cidx = {r["k"]: i for i, r in enumerate(corpus)}
    dates = [r["date"] for r in corpus]
    code_dates: dict[str, list[str]] = defaultdict(list)
    for r in corpus:
        code_dates[r["product_code"]].append(r["date"])
    write_jsonl(out / "corpus.jsonl.gz", corpus)

    # ---- subjects / gold -----------------------------------------------------------------------------------------
    queries = {"train": [], "dev": [], "test": []}
    counts: Counter = Counter()
    pair_reasons: Counter = Counter()
    for r in corpus:
        split = in_any_window(r["date"], cfg)
        if not split or (subj_set is not None and r["k"] not in subj_set):
            continue
        counts[f"{split}.in_window"] += 1
        p = parsed[r["k"]]
        cands = [x for x in p.get("predicates", []) if x["role"] in ("primary", "additional")]
        if not cands:
            counts[f"{split}.no_predicate_extracted"] += 1
            continue
        refs = [x["k"] for x in p["predicates"] if x["role"] == "reference"]
        gold = []
        for x in cands:
            kk, pm = x["k"], meta.get(x["k"])
            if x["kind"] not in ("k", "bk") or pm is None or not pm.get("decision_date") or pm["decision_date"] >= r["date"]:
                reason, in_c = missing_reason(kk, x["kind"], meta, status, r["date"], min_date), False
            elif kk in cidx:
                reason, in_c = None, True
            else:
                reason, in_c = missing_reason(kk, x["kind"], meta, status, r["date"], min_date), False
            gold.append({
                "k": kk, "role": x["role"], "in_corpus": in_c, "reason": reason,
                "year": pm["year"] if pm else None, "product_code": pm["product_code"] if pm else None,
            })
            pair_reasons[(split, reason or "in_corpus")] += 1
        gold_names = [meta[g["k"]]["device_name"] for g in gold if g["k"] in meta]
        gold_ks = [g["k"] for g in gold]
        raw_ifu, raw_desc = p.get("ifu") or "", p.get("description") or ""
        leak = leak_stats(raw_ifu, raw_desc, gold_ks, gold_names, r["k"])
        ifu, _ = mask_query_text(raw_ifu, level, gold_names, r["device_name"], cfg)
        desc, _ = mask_query_text(raw_desc, level, gold_names, r["device_name"], cfg)
        leak.update(residual_leak(ifu + " " + desc, gold_names, r["device_name"]))
        inc = [g["k"] for g in gold if g["in_corpus"]]
        n_cand = bisect_left(dates, r["date"])
        q = {
            "k": r["k"], "split": split, "date": r["date"], "year": r["year"], "product_code": r["product_code"],
            "panel": r["panel"], "ifu": ifu, "description": desc, "gold": gold, "reference": refs,
            "gold_in_corpus": inc, "gold_primary_in_corpus": [g["k"] for g in gold if g["in_corpus"] and g["role"] == "primary"],
            "n_candidates": n_cand,
            "n_same_code_candidates": bisect_left(code_dates[r["product_code"]], r["date"]),
            "retrievable": bool(inc), "leak": leak,
        }
        queries[split].append(q)
        counts[f"{split}.with_predicate"] += 1
        counts[f"{split}.retrievable"] += int(bool(inc))

    for s, qs in queries.items():
        write_jsonl(out / f"queries_{s}.jsonl.gz", qs)
    pairs = []
    for q in queries["train"]:
        for g in q["gold"]:
            if g["in_corpus"]:
                pairs.append({"q": q["k"], "p": g["k"], "role": g["role"]})
    write_jsonl(out / "train_pairs.jsonl.gz", pairs)

    card = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
        "mask_level": level, "subjects_queue": subjects_queue,
        "n_meta": len(meta), "n_corpus": len(corpus),
        "status_counts": dict(Counter(status.values())),
        "counts": dict(counts),
        "pair_reasons": {f"{s}|{r}": n for (s, r), n in pair_reasons.items()},
        "config": {
            "splits": cfg["splits"], "corpus": cfg["corpus"], "parse": cfg["parse"], "leakage": cfg["leakage"],
            "models": cfg["models"], "input_template": cfg["input_template"], "truncation": cfg["truncation"],
            "eval_spec": cfg["eval_spec"],
        },
    }
    (out / "dataset_card.json").write_text(json.dumps(card, indent=2))
    log.info("build: corpus=%d, queries=%s, pairs=%d (%.1fs)", len(corpus), {s: len(v) for s, v in queries.items()}, len(pairs), card["seconds"])
    return card


# ---------------------------------------------------------------------------------------------------------------
def load_dataset(paths):
    d = paths.dataset
    corpus = list(read_jsonl(d / "corpus.jsonl.gz"))
    queries = {s: list(read_jsonl(d / f"queries_{s}.jsonl.gz")) for s in ("train", "dev", "test")}
    pairs = list(read_jsonl(d / "train_pairs.jsonl.gz"))
    card = json.loads((d / "dataset_card.json").read_text()) if (d / "dataset_card.json").exists() else {}
    return corpus, queries, pairs, card


def validate_dataset(paths, cfg: dict, max_report: int = 5) -> dict:
    """Schema + invariants. Returns {'problems': {name: [examples]}, 'n_checked': {...}}. Empty problems == format OK."""
    import re

    problems: dict[str, list] = defaultdict(list)

    def bad(name, ex):
        problems[name].append(ex)

    d = paths.dataset
    need = ["corpus.jsonl.gz", "queries_train.jsonl.gz", "queries_dev.jsonl.gz", "queries_test.jsonl.gz", "train_pairs.jsonl.gz", "dataset_card.json"]
    for f in need:
        if not (d / f).exists():
            bad("missing_file", f)
    if problems:
        return {"problems": {k: v[:max_report] + ([f"... +{len(v) - max_report} more"] if len(v) > max_report else []) for k, v in problems.items()}, "n_checked": {}}
    corpus, queries, pairs, card = load_dataset(paths)
    level = card.get("mask_level", "standard")
    kre, dre = re.compile(r"^(?:BK|K)\d{6}$"), re.compile(r"^\d{4}-\d{2}-\d{2}$")
    str_fields = ["k", "date", "product_code", "panel", "device_name", "ifu", "description"]
    seen = set()
    prev = ("", "")
    for r in corpus:
        for f in str_fields:
            if not isinstance(r.get(f), str):
                bad("corpus_field_type", (r.get("k"), f))
        if not isinstance(r.get("year"), int):
            bad("corpus_field_type", (r.get("k"), "year"))
        if not kre.match(r["k"]):
            bad("corpus_bad_k", r["k"])
        if not dre.match(r["date"]):
            bad("corpus_bad_date", (r["k"], r["date"]))
        if r["k"] in seen:
            bad("corpus_duplicate_k", r["k"])
        seen.add(r["k"])
        if (r["date"], r["k"]) < prev:
            bad("corpus_not_sorted", r["k"])
        prev = (r["date"], r["k"])
        if not r["ifu"].strip():
            bad("corpus_empty_ifu", r["k"])
        if cfg["corpus"]["usable_policy"] == "both" and not r["description"].strip():
            bad("corpus_empty_description", r["k"])
        if level != "none":
            from .knumbers import find_knumbers

            if find_knumbers(r["ifu"] + " " + r["description"], None, True):
                bad("corpus_k_number_not_masked", r["k"])
    cidx = {r["k"]: i for i, r in enumerate(corpus)}
    dates = [r["date"] for r in corpus]
    wins = {s: tuple(v) for s, v in cfg["splits"].items()}
    span = {}
    for s, qs in queries.items():
        for q in qs:
            if q["k"] not in cidx:
                bad("query_not_in_corpus", q["k"])
                continue
            if not (wins[s][0] <= q["date"] <= wins[s][1]):
                bad("query_outside_split_window", (q["k"], s, q["date"]))
            if q["n_candidates"] != bisect_left(dates, q["date"]):
                bad("query_n_candidates_mismatch", q["k"])
            if not q["ifu"].strip():
                bad("query_empty_ifu", q["k"])
            if level != "none" and q["leak"]["residual_k"] > 0:
                bad("query_k_number_not_masked", q["k"])
            ks = [g["k"] for g in q["gold"]]
            if len(ks) != len(set(ks)):
                bad("query_duplicate_gold", q["k"])
            if q["k"] in ks:
                bad("query_cites_itself", q["k"])
            inc = [g["k"] for g in q["gold"] if g["in_corpus"]]
            if inc != q["gold_in_corpus"]:
                bad("query_gold_in_corpus_inconsistent", q["k"])
            for g in q["gold"]:
                if g["role"] not in ("primary", "additional"):
                    bad("query_bad_gold_role", (q["k"], g["k"], g["role"]))
                if g["in_corpus"]:
                    if g["k"] not in cidx:
                        bad("gold_not_in_corpus", (q["k"], g["k"]))
                    elif corpus[cidx[g["k"]]]["date"] >= q["date"]:
                        bad("gold_not_before_subject", (q["k"], g["k"]))
                elif not g["reason"]:
                    bad("gold_missing_without_reason", (q["k"], g["k"]))
            if q["retrievable"] != bool(inc):
                bad("query_retrievable_flag", q["k"])
        if qs:
            span[s] = (min(q["date"] for q in qs), max(q["date"] for q in qs))
    order = [s for s in ("train", "dev", "test") if s in span]
    for a, b in zip(order, order[1:]):
        if span[a][1] >= span[b][0]:
            bad("splits_overlap_in_time", (a, b, span[a], span[b]))
    qset = {q["k"] for q in queries["train"]}
    for pr in pairs:
        if pr["q"] not in qset:
            bad("pair_query_not_train", pr)
        elif pr["p"] not in cidx:
            bad("pair_predicate_not_in_corpus", pr)
    return {
        "problems": {k: v[:max_report] + ([f"... +{len(v) - max_report} more"] if len(v) > max_report else []) for k, v in problems.items()},
        "n_checked": {"corpus": len(corpus), **{f"queries_{s}": len(v) for s, v in queries.items()}, "train_pairs": len(pairs)},
    }
