"""`predx ocr-stress`: how much of the signal survives increasingly bad scans?

Builds synthetic 510(k) PDFs with PLANTED predicates (same page structure as the pipeline expects: letter, FDA
indications form, summary), renders them as image-only PDFs at several degradation levels (skew, blur, lost resolution,
noise, uneven shading, speckle, JPEG artefacts), runs the real pdftotext/tesseract + parser, and compares with the planted
truth. It measures our rules + tesseract on SYNTHETIC scans: real scans are messier and the first real-PDF probe overrules
anything printed here. Use it to (a) see which stage degrades first (indications text? K-number digits? role labels?),
(b) decide whether OCR settings (dpi, psm) deserve tuning on the VM before the full run.
"""
from __future__ import annotations

import csv
import difflib
import json
import logging
import os
import random
import re
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .parse_stage import parse_document
from .pdftext import extract_pdf
from .synth import BRANDS, FAMILIES, IFU_HEADS, DESC_HEADS, PRED_HEADS, _desc, _ifu, _pred_block, _render_form, _render_letter, _render_summary
from .synth_pdf import render_pdf

log = logging.getLogger("predx.stress")


_KLIKE = re.compile(r"(?<![A-Za-z0-9])(?:BK|K)\d{6}(?!\d)", re.I)


def _kn(rng, lo=10, hi=24):
    return f"K{rng.randint(lo, hi):02d}{rng.randint(1000, 9999):04d}"


def make_doc(rng: random.Random, i: int) -> dict:
    pc = rng.choice(list(FAMILIES))
    fam = FAMILIES[pc]
    lineage = rng.sample(fam[2], 3)
    brand = rng.choice(BRANDS)
    name = f"{brand} {rng.choice(['X', 'Pro', 'Flex', 'Neo'])}{rng.randint(100, 999)} {fam[1].title()}"
    dev = {"k": f"K25{i + 1:04d}", "date": "2025-06-15", "applicant": f"{rng.choice(BRANDS)} Medical Inc.", "name": name, "pc": pc, "cat": fam[1]}
    n_pred = rng.choice([1, 1, 2, 3])
    preds = []
    while len(preds) < n_pred:
        k = _kn(rng)
        if k not in preds:
            preds.append(k)
    refs = [_kn(rng)] if rng.random() < 0.3 else []
    names = {k: f"{rng.choice(BRANDS)} {rng.choice(['A', 'B', 'S'])}{rng.randint(10, 99)} {fam[1].title()}" for k in preds + refs}
    ifu = _ifu(rng, name, fam, lineage)
    desc = _desc(rng, name, fam, lineage, rng.randint(70, 200))
    style = rng.choice("abc")
    block = _pred_block(rng, style, rng.choice(PRED_HEADS), preds[0], preds[1:], refs, names)
    pages = [_render_letter(dev)]
    form = rng.random() < 0.7
    if form:
        pages.append(_render_form(dev, ifu))
    order = rng.sample(["pred", "desc", "ifu"], 3)
    summ = _render_summary(rng, dev, ifu, desc, block, order, rng.choice(IFU_HEADS), rng.choice(DESC_HEADS), with_ifu_in_summary=not form)
    lines = summ.split("\n")
    for c in range(0, len(lines), 45):
        pages.append("\n".join(lines[c: c + 45]))
    return {"k": dev["k"], "pages": pages[:4], "ifu": ifu, "desc": desc, "primary": preds[0], "additional": preds[1:], "reference": refs, "style": style}


def _sim(a: str | None, b: str, n_words: int = 60) -> float:
    if not a:
        return 0.0
    x, y = a.lower().split()[:n_words], b.lower().split()[:n_words]
    return difflib.SequenceMatcher(None, x, y, autojunk=False).ratio()


def _work(args):
    path, k, tcfg_cfg = args
    cfg = tcfg_cfg
    t0 = time.time()
    try:
        ex = extract_pdf(path, cfg["text"])
        parsed = parse_document(k, ex["pages"], cfg, None)
        return {"k": k, "seconds": time.time() - t0, "parsed": parsed, "text": " ".join(ex["pages"]), "n_pages": ex["n_pages"], "ocr_pages": len(ex["ocr_pages"]), "error": None}
    except Exception as e:
        return {"k": k, "seconds": time.time() - t0, "parsed": {}, "text": "", "n_pages": 0, "ocr_pages": 0, "error": repr(e)[:200]}


def run_stress(cfg: dict, levels: list[int], n: int, seed: int, out: Path) -> list[dict]:
    out = Path(out)
    (out / "pdf").mkdir(parents=True, exist_ok=True)
    rows = []
    workers = cfg["text"].get("workers") or os.cpu_count() or 1
    for lv in levels:
        rng = random.Random(seed * 100 + lv)
        docs = [make_doc(random.Random(seed + i), i) for i in range(n)]            # same documents at every level
        tasks = []
        for d in docs:
            p = out / "pdf" / f"L{lv}_{d['k']}.pdf"
            render_pdf(d["pages"], p, scanned=True, degrade=lv, seed=rng.randrange(10**6))
            tasks.append((str(p), d["k"], cfg))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_work, tasks))
        m = {k: 0 for k in ("ifu_ok", "desc_ok", "primary_ok", "gold_exact", "ref_ok", "any_gold_found", "k_digits_exact", "errors")}
        n_gold = n_gold_found = n_cand = n_cand_true = 0
        n_ktok = n_ktok_ok = 0
        for d, r in zip(docs, res):
            if r["error"]:
                m["errors"] += 1
                continue
            p = r["parsed"]
            gold = [d["primary"], *d["additional"]]
            got = {x["k"]: x["role"] for x in p.get("predicates", [])}
            cand = {k for k, ro in got.items() if ro in ("primary", "additional")}
            m["ifu_ok"] += _sim(p.get("ifu"), d["ifu"]) >= 0.85
            m["desc_ok"] += _sim(p.get("description"), d["desc"]) >= 0.85
            m["primary_ok"] += next((x["k"] for x in p.get("predicates", []) if x["role"] == "primary"), None) == d["primary"]
            m["gold_exact"] += cand == set(gold)
            m["ref_ok"] += (set(k for k, ro in got.items() if ro == "reference") == set(d["reference"]))
            m["any_gold_found"] += bool(cand & set(gold))
            flat = r["text"].replace(" ", "")
            truth_ks = {d["k"], *gold, *d["reference"]}
            for tok in _KLIKE.findall(flat):                      # every K-number-looking token in the final text
                n_ktok += 1
                n_ktok_ok += tok.upper() in truth_ks
            m["k_digits_exact"] += all(k in flat for k in gold)
            n_gold += len(gold)
            n_gold_found += len(cand & set(gold))
            n_cand += len(cand)
            n_cand_true += len(cand & set(gold))
        ok = max(len(docs) - m["errors"], 1)
        row = {"level": lv, "docs": len(docs), "errors": m["errors"],
               "ifu_ok": m["ifu_ok"] / ok, "desc_ok": m["desc_ok"] / ok, "primary_ok": m["primary_ok"] / ok, "gold_set_exact": m["gold_exact"] / ok,
               "gold_recall": n_gold_found / max(n_gold, 1), "gold_precision": n_cand_true / max(n_cand, 1),
               "all_K_digits_survive_ocr": m["k_digits_exact"] / ok, "k_token_accuracy": n_ktok_ok / max(n_ktok, 1), "n_k_tokens": n_ktok,
               "reference_ok": m["ref_ok"] / ok,
               "sec_per_doc": sum(r["seconds"] for r in res) / max(len(res), 1)}
        rows.append(row)
        log.info("level %d done: %s", lv, {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()})
    cols = list(rows[0].keys())
    with open(out / "ocr_stress.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (out / "ocr_stress.json").write_text(json.dumps(rows, indent=1))
    print("\nOCR stress test (SYNTHETIC scans with planted truth; level 0 = clean render, 3 = very poor):")
    print(f"{'lvl':>3} {'docs':>4} {'IFU ok':>7} {'desc ok':>8} {'K digits':>9} {'K tok acc':>10} {'primary':>8} {'gold set':>9} {'recall':>7} {'prec':>6} {'ref ok':>7} {'s/doc':>6}")
    for r in rows:
        print(f"{r['level']:>3} {r['docs']:>4} {r['ifu_ok']:>7.0%} {r['desc_ok']:>8.0%} {r['all_K_digits_survive_ocr']:>9.0%} {r['k_token_accuracy']:>10.1%} {r['primary_ok']:>8.0%} "
              f"{r['gold_set_exact']:>9.0%} {r['gold_recall']:>7.0%} {r['gold_precision']:>6.0%} {r['reference_ok']:>7.0%} {r['sec_per_doc']:>6.1f}")
    print(f"\nPDFs kept in {out / 'pdf'} (open a level-3 one to see how bad it looks). Results: {out / 'ocr_stress.csv'}")
    return rows
