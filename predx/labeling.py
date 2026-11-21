"""Hand-labelling support for the proposal's "check extraction accuracy on 100 hand-labeled summaries".

  label-sample : writes labeling/label_sample.csv (+ one snippet file per subject) for you to fill in by opening the PDF
  label-eval   : re-reads the CSV and scores the CURRENT parsed.jsonl.gz against your labels
                 -> iterate: edit rules in sections.py/predicates.py, run `parse`, run `label-eval` again (no OCR re-run)

Fill in (K-numbers separated by ; or space):  hand_primary, hand_additional, hand_reference, and put y in `done`.
Leave hand_* empty and set done=y when the document truly cites no predicate.
"""
from __future__ import annotations

import csv
import logging
import math
import random
import re
from collections import defaultdict

from .fetch import run_fetch
from .io import read_jsonl, write_text
from .metadata import load_meta
from .metrics import wilson
from .plan import in_any_window

log = logging.getLogger("predx.label")
COLS = ["k", "year", "product_code", "pdf_path", "ocr_pages", "flagged", "extracted_primary", "extracted_additional",
        "extracted_reference", "extracted_other", "done", "hand_primary", "hand_additional", "hand_reference", "notes"]


def _ks(s: str) -> list[str]:
    return [x.upper() for x in re.findall(r"(?:BK|K|DEN|P)\d{6}", (s or "").upper())]


def make_label_sample(cfg: dict, paths, n: int | None = None, fetch_missing: bool = True) -> str:
    n = n or cfg["labeling"]["n"]
    frac = cfg["labeling"]["flagged_fraction"]
    meta = load_meta(paths)
    tstat = {r["k"]: r for r in read_jsonl(paths.text_stats)}
    pool = []
    for p in read_jsonl(paths.parsed):
        m = meta.get(p["k"])
        if not m or p.get("parse_error") or p.get("statement_like") or not p.get("ifu"):
            continue
        if not in_any_window(m.get("decision_date"), cfg):
            continue
        preds = p.get("predicates", [])
        flagged = bool(p.get("ambiguous")) or not any(x["role"] in ("primary", "additional") for x in preds) or (tstat.get(p["k"], {}).get("ocr_pages") or 0) > 0
        pool.append((p, m, flagged))
    rng = random.Random(cfg["seed"])
    flagged_pool = [x for x in pool if x[2]]
    other_pool = [x for x in pool if not x[2]]
    rng.shuffle(flagged_pool)
    # stratify the unflagged part by year
    by_year = defaultdict(list)
    for x in other_pool:
        by_year[x[1]["year"]].append(x)
    for v in by_year.values():
        rng.shuffle(v)
    n_flag = min(len(flagged_pool), round(n * frac))
    pick = flagged_pool[:n_flag]
    years = sorted(by_year)
    i = 0
    while len(pick) < n and any(by_year.values()):
        y = years[i % len(years)]
        if by_year[y]:
            pick.append(by_year[y].pop())
        i += 1
    pick = pick[:n]
    pick.sort(key=lambda x: x[0]["k"])
    paths.labeling.mkdir(parents=True, exist_ok=True)
    if fetch_missing:
        need = [p["k"] for p, _, _ in pick if not paths.pdf(p["k"]).exists()]
        if need:
            log.info("re-fetching %d PDFs for the labelling sample", len(need))
            cfg2 = {**cfg, "fetch": {**cfg["fetch"]}}
            run_fetch(cfg2, paths, need)
    out = paths.labeling / "label_sample.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for p, m, flagged in pick:
            by_role = defaultdict(list)
            for x in p["predicates"]:
                by_role[x["role"]].append(x["k"])
            w.writerow({
                "k": p["k"], "year": m["year"], "product_code": m["product_code"], "pdf_path": str(paths.pdf(p["k"])),
                "ocr_pages": (tstat.get(p["k"], {}).get("ocr_pages") or 0), "flagged": int(flagged),
                "extracted_primary": ";".join(by_role["primary"]), "extracted_additional": ";".join(by_role["additional"]),
                "extracted_reference": ";".join(by_role["reference"]), "extracted_other": ";".join(by_role["other"]),
                "done": "", "hand_primary": "", "hand_additional": "", "hand_reference": "", "notes": "",
            })
            lines = [f"{p['k']}  year={m['year']}  product_code={m['product_code']}  device={m['device_name']}", "-" * 70]
            for x in p["predicates"]:
                lines.append(f"[{x['role']:9s}] {x['k']}  x{x['n_mentions']}  p.{x['page'] + 1}   ...{x['snippet']}...")
            if not p["predicates"]:
                lines.append("(no K/DEN/P numbers found other than the subject's own)")
            write_text(paths.labeling / "snippets" / f"{p['k']}.txt", "\n".join(lines) + "\n")
    log.info("label sample: %d subjects -> %s", len(pick), out)
    return str(out)


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and (p + r) else float("nan")
    return p, r, f


def eval_labels(cfg: dict, paths, csv_path: str | None = None) -> dict:
    csv_path = csv_path or str(paths.labeling / "label_sample.csv")
    parsed = {p["k"]: p for p in read_jsonl(paths.parsed)}
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8")) if (r.get("done") or "").strip().lower() in ("y", "yes", "1", "x")]
    tp = fp = fn = 0
    rtp = rfp = rfn = 0
    prim_ok = fo_ok = exact = 0
    prim_den = 0
    errors = []
    for r in rows:
        p = parsed.get(r["k"])
        if p is None:
            continue
        hand_prim = _ks(r["hand_primary"])
        hand_gold = set(hand_prim) | set(_ks(r["hand_additional"]))
        hand_ref = set(_ks(r["hand_reference"]))
        pred_prim = next((x["k"] for x in p["predicates"] if x["role"] == "primary"), None)
        pred_gold = {x["k"] for x in p["predicates"] if x["role"] in ("primary", "additional")}
        pred_ref = {x["k"] for x in p["predicates"] if x["role"] == "reference"}
        tp += len(hand_gold & pred_gold)
        fp += len(pred_gold - hand_gold)
        fn += len(hand_gold - pred_gold)
        rtp += len(hand_ref & pred_ref)
        rfp += len(pred_ref - hand_ref)
        rfn += len(hand_ref - pred_ref)
        hp = hand_prim[0] if hand_prim else None
        prim_den += 1
        prim_ok += int(pred_prim == hp)
        fo_ok += int(p.get("first_other") == hp)
        ex = pred_gold == hand_gold
        exact += int(ex)
        if not ex:
            errors.append({"k": r["k"], "hand": sorted(hand_gold), "extracted": sorted(pred_gold), "missed": sorted(hand_gold - pred_gold), "spurious": sorted(pred_gold - hand_gold)})
    n = prim_den
    P, R, F = _prf(tp, fp, fn)
    rP, rR, rF = _prf(rtp, rfp, rfn)
    res = {
        "n_labelled": n,
        "gold_precision": P, "gold_recall": R, "gold_f1": F,
        "primary_accuracy": prim_ok / n if n else float("nan"), "primary_accuracy_ci95": wilson(prim_ok, n),
        "first_other_number_accuracy": fo_ok / n if n else float("nan"), "first_other_number_accuracy_ci95": wilson(fo_ok, n),
        "subject_exact_match": exact / n if n else float("nan"), "subject_exact_match_ci95": wilson(exact, n),
        "reference_precision": rP, "reference_recall": rR,
        "errors": errors[:30],
    }
    return res


def format_label_eval(res: dict) -> str:
    def f(x):
        return "n/a" if isinstance(x, float) and math.isnan(x) else f"{x:.3f}"
    L = [f"labelled subjects: {res['n_labelled']}",
         f"gold set (primary+additional): precision {f(res['gold_precision'])}  recall {f(res['gold_recall'])}  F1 {f(res['gold_f1'])}",
         f"primary accuracy: {f(res['primary_accuracy'])}  95% CI [{f(res['primary_accuracy_ci95'][0])}, {f(res['primary_accuracy_ci95'][1])}]   "
         f"(naive first-other-number: {f(res['first_other_number_accuracy'])})",
         f"exact set match per subject: {f(res['subject_exact_match'])}  95% CI [{f(res['subject_exact_match_ci95'][0])}, {f(res['subject_exact_match_ci95'][1])}]",
         f"reference devices: precision {f(res['reference_precision'])}  recall {f(res['reference_recall'])}"]
    for e in res["errors"][:10]:
        L.append(f"  miss {e['k']}: missed {e['missed']} spurious {e['spurious']}")
    return "\n".join(L)
