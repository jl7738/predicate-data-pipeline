"""`predx probe`: run the REAL pipeline (fetch -> pdftotext/OCR -> sections -> predicates) on a handful of PDFs and print
everything needed to see why a document did or did not parse. Meant for the first contact with real FDA PDFs:

    python -m predx.cli probe K243175 K191717            # downloads from FDA (needs network; use the VM)
    python -m predx.cli probe --pdf-dir ~/some/pdfs      # PDFs you already have (file name should contain the K-number)
    python -m predx.cli probe --ks-file config/probe_ks.txt

Writes <out>/probe_report.md (+ probe_summary.csv, text/<K>.txt with the full extracted text, page by page).
The report is deliberately long on evidence: heading candidates, section previews INCLUDING their last characters (to
spot a section running into the next one), predicate roles with snippets, and every line that mentions predicates, so
a failing rule can be fixed from the report alone.
"""
from __future__ import annotations

import csv
import logging
import re
import time
from pathlib import Path

from .config import Paths
from .knumbers import find_knumbers
from .parse_stage import parse_document
from .pdftext import extract_pdf, looks_garbled
from .sections import find_headings

log = logging.getLogger("predx.probe")

_K_IN_NAME = re.compile(r"(?<![A-Za-z0-9])((?:K|BK|DEN|P)\d{5,7}(?:/S\d+)?)", re.I)
_PAGE_TYPES = [
    ("ifu_form", re.compile(r"type\s+of\s+use|form\s+fda\s+3881", re.I)),
    ("510k_statement", re.compile(r"510\s*\(k\)\s+statement", re.I)),
    ("summary", re.compile(r"510\s*\(k\)\s+summary|summary\s+of\s+safety", re.I)),
    ("letter", re.compile(r"dear\s+(?:sir|madam|mr|ms|dr)|sincerely|silver\s+spring|we\s+have\s+reviewed\s+your", re.I)),
]
_EVIDENCE = re.compile(r"predicate|substantial(?:ly)?\s+equivalen|reference\s+device|legally\s+marketed|\b(?:K|BK|DEN|P)\s?\d{5,7}\b", re.I)


def page_type(text: str) -> str:
    for name, rx in _PAGE_TYPES:
        if rx.search(text):
            return name
    return "other" if len(text.strip()) > 80 else "blank/scan?"


def subject_from_name(p: Path) -> str:
    m = _K_IN_NAME.search(p.stem)
    return m.group(1).upper() if m else p.stem


def _first_line(t: str, n: int = 72) -> str:
    for line in t.splitlines():
        if line.strip():
            return re.sub(r"\s+", " ", line.strip())[:n]
    return ""


def _ends(s: str | None, n: int = 160) -> str:
    return "" if not s else re.sub(r"\s+", " ", s[-n:])


def _starts(s: str | None, n: int = 320) -> str:
    return "" if not s else re.sub(r"\s+", " ", s[:n])


def probe_pdf(path: Path, k: str, cfg: dict, known=None) -> dict:
    t0 = time.time()
    res = {"k": k, "file": str(path), "bytes": path.stat().st_size, "warnings": [], "error": None}
    try:
        ex = extract_pdf(path, cfg["text"])
    except Exception as e:
        res["error"] = repr(e)[:300]
        res["warnings"].append(f"EXTRACT_FAILED {res['error']}")
        return res
    pages = ex["pages"]
    res.update({"n_pages": ex["n_pages"], "n_read": ex["n_pages_read"], "ocr_pages": ex["ocr_pages"], "n_chars": ex["n_chars"],
                "extract_seconds": ex["seconds"], "pages": pages})
    res["page_info"] = [
        {"i": i + 1, "chars": len(p), "ocr": i in ex["ocr_pages"], "garbled": looks_garbled(p), "type": page_type(p), "first": _first_line(p)}
        for i, p in enumerate(pages)
    ]
    parsed = parse_document(k, pages, cfg, known)
    res["parsed"] = parsed

    text = "\n\f\n".join(pages)
    heads = find_headings(text)
    res["headings"] = [{"cat": h.cat, "page": text.count("\f", 0, h.line_start) + 1, "title": re.sub(r"\s+", " ", h.title)[:60]} for h in heads]
    res["all_knumbers"] = [(h.k, h.kind, h.repaired) for h in find_knumbers(text, known)]
    ev = []
    for ln, line in enumerate(text.splitlines(), 1):
        if _EVIDENCE.search(line):
            ev.append((ln, re.sub(r"\s+", " ", line.strip())[:150]))
    res["evidence"] = ev

    W = res["warnings"]
    npg = max(res["n_read"], 1)
    if res["n_chars"] < 400:
        W.append("NO_TEXT: <400 characters extracted (scan that OCR could not read, or an unusual PDF)")
    if len(ex["ocr_pages"]) / npg > 0.5:
        W.append(f"MOSTLY_OCR: {len(ex['ocr_pages'])}/{npg} pages needed OCR")
    if len(ex["ocr_pages"]) >= cfg["text"]["ocr"]["max_pages"]:
        W.append("OCR_PAGE_CAP_HIT: later pages were not OCR'd")
    if ex["n_pages"] > cfg["text"]["max_pages"]:
        W.append(f"PAGE_CAP: only the first {cfg['text']['max_pages']} of {ex['n_pages']} pages were read")
    if any(p["garbled"] for p in res["page_info"]):
        W.append("GARBLED_PAGES: " + ",".join(str(p["i"]) for p in res["page_info"] if p["garbled"]))
    if not any(re.search(re.escape(k), p.replace(" ", ""), re.I) for p in pages) and re.match(r"(?:K|BK|DEN|P)\d+", k, re.I):
        W.append(f"SUBJECT_K_NOT_IN_TEXT: {k} never appears (wrong file? OCR misread?)")
    mw = cfg["parse"]["min_words"]
    if parsed.get("statement_like"):
        W.append("STATEMENT_LIKE: looks like a 510(k) *statement* (no summary): not usable, expected to be dropped")
    if not parsed.get("ifu"):
        W.append("NO_IFU")
    elif parsed["ifu_words"] < mw["ifu"]:
        W.append(f"SHORT_IFU: {parsed['ifu_words']} words")
    if not parsed.get("description"):
        W.append("NO_DESC")
    elif parsed["desc_words"] < mw["description"]:
        W.append(f"SHORT_DESC: {parsed['desc_words']} words")
    if parsed.get("ifu_capped"):
        W.append("IFU_CAPPED: a runaway section hit max_section_chars (next heading not recognised?)")
    if parsed.get("desc_capped"):
        W.append("DESC_CAPPED: a runaway section hit max_section_chars (next heading not recognised?)")
    if parsed.get("se_language_ifu") or parsed.get("se_language_desc"):
        W.append("SE_LANGUAGE_IN_INPUT: predicate / equivalence wording inside a model-input section (leak risk)")
    preds = parsed.get("predicates", [])
    cand = [p for p in preds if p["role"] in ("primary", "additional")]
    if not preds and parsed.get("pred_section_found"):
        W.append("PREDICATE_SECTION_BUT_NO_KNUMBER: a predicate / comparison heading exists but no K-number was found (OCR digits? name-only citation?)")
    if not cand and any(p["role"] == "other" for p in preds):
        W.append("ONLY_UNCLASSIFIED_KNUMBERS: K-numbers found but none labelled as predicates")
    if any(p.get("repaired") for p in preds):
        W.append("OCR_REPAIRED_KNUMBER: " + ",".join(p["k"] for p in preds if p.get("repaired")))
    if parsed.get("ambiguous"):
        W.append("AMBIGUOUS_PRIMARY: " + str(parsed.get("primary_by")))
    res["seconds"] = round(time.time() - t0, 2)
    return res


def _md_doc(r: dict) -> list[str]:
    L = [f"## {r['k']}  —  `{Path(r['file']).name}`"]
    if r.get("error"):
        return L + [f"**EXTRACTION FAILED**: {r['error']}", ""]
    p = r["parsed"]
    L.append(f"{r['n_pages']} pages ({r['n_read']} read), {len(r['ocr_pages'])} OCR'd {r['ocr_pages'] or ''}, {r['n_chars']:,} chars, "
             f"{r['extract_seconds']}s extract, {r['bytes'] / 1e6:.2f} MB")
    L.append("")
    L.append("**Warnings:** " + ("; ".join(r["warnings"]) if r["warnings"] else "none"))
    L.append("")
    L.append("| page | type | chars | OCR | garbled | first line |\n|---|---|---|---|---|---|")
    for pi in r["page_info"][:20]:
        L.append(f"| {pi['i']} | {pi['type']} | {pi['chars']} | {'yes' if pi['ocr'] else ''} | {'YES' if pi['garbled'] else ''} | {pi['first'].replace('|', '/')} |")
    L.append("")
    L.append("**Heading candidates seen by the parser** (cat: ifu / desc / pred / other / field):")
    L.append(", ".join(f"`{h['cat']}`·p{h['page']}·“{h['title']}”" for h in r["headings"][:40]) or "_none_")
    L.append("")
    L.append(f"**Indications** (source={p.get('ifu_source')}, {p.get('ifu_words')} words, form page={p.get('form_page')}):")
    L.append(f"> START: {_starts(p.get('ifu'))}")
    L.append(f"> END:   …{_ends(p.get('ifu'))}")
    L.append("")
    L.append(f"**Description** ({p.get('desc_words')} words):")
    L.append(f"> START: {_starts(p.get('description'))}")
    L.append(f"> END:   …{_ends(p.get('description'))}")
    L.append("")
    L.append(f"**Predicates** (section found={p.get('pred_section_found')}, primary chosen by={p.get('primary_by')}, "
             f"first-other-number={p.get('first_other')}):")
    if p.get("predicates"):
        L.append("| K | role | kind | mentions | page | snippet |\n|---|---|---|---|---|---|")
        for x in p["predicates"]:
            L.append(f"| {x['k']}{' (OCR-repaired)' if x.get('repaired') else ''} | {x['role']} | {x['kind']} | {x['n_mentions']} | {x['page'] + 1} | {x['snippet'].replace('|', '/')} |")
    else:
        L.append("_none extracted_")
    L.append("")
    L.append("<details><summary>Lines mentioning predicate / equivalence / K-numbers (with line numbers in text/%s.txt)</summary>\n" % r["k"])
    for ln, line in r["evidence"][:40]:
        L.append(f"- L{ln}: {line}")
    L.append("\n</details>\n")
    return L


def run_probe(cfg: dict, ks: list[str], pdfs: list[Path], out: Path, fetch: bool = True, known=None) -> list[dict]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = Paths(cfg, root=out)
    items: list[tuple[str, Path]] = [(subject_from_name(p), p) for p in pdfs]
    st: dict = {}
    if ks and fetch:
        from .fetch import run_fetch

        res = run_fetch(cfg, paths, ks)
        log.info("probe fetch: %s", res)
        if paths.fetch_status.exists():
            from .io import last_by_key

            st = last_by_key(paths.fetch_status)
    for k in ks:
        if paths.pdf(k).exists():
            items.append((k, paths.pdf(k)))
        else:
            s = st.get(k, {}) if fetch else {}
            items.append((k, None))
            log.warning("probe: no PDF for %s (%s %s)", k, s.get("status", "not fetched"), s.get("error") or "")
    recs = []
    for k, p in items:
        if p is None:
            recs.append({"k": k, "file": "(not fetched)", "bytes": 0, "warnings": ["FETCH_FAILED"], "error": "no PDF", "page_info": []})
            continue
        log.info("probe: %s", p.name)
        r = probe_pdf(Path(p), k, cfg, known)
        recs.append(r)
        if r.get("pages"):
            (out / "text").mkdir(exist_ok=True)
            with open(out / "text" / f"{k}.txt", "w", encoding="utf-8") as fh:
                for i, pg in enumerate(r["pages"], 1):
                    fh.write(f"\n===== page {i}{' (OCR)' if (i - 1) in r['ocr_pages'] else ''} =====\n{pg}\n")
    _write_reports(recs, out)
    return recs


def _write_reports(recs: list[dict], out: Path) -> None:
    rows = []
    for r in recs:
        p = r.get("parsed") or {}
        cand = [x["k"] for x in p.get("predicates", []) if x["role"] in ("primary", "additional")]
        rows.append({
            "k": r["k"], "pages": r.get("n_pages"), "ocr_pages": len(r.get("ocr_pages") or []), "chars": r.get("n_chars"),
            "ifu_source": p.get("ifu_source"), "ifu_words": p.get("ifu_words"), "desc_words": p.get("desc_words"),
            "primary": next((x["k"] for x in p.get("predicates", []) if x["role"] == "primary"), ""),
            "predicates": ";".join(cand), "n_warnings": len(r["warnings"]), "warnings": " | ".join(w.split(":")[0] for w in r["warnings"]),
        })
    with open(out / "probe_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["k"])
        w.writeheader()
        w.writerows(rows)
    L = ["# Probe report", "", f"{len(recs)} documents", "",
         "| K | pages | OCR pp | IFU (src, words) | desc words | primary | other predicates | warnings |", "|---|---|---|---|---|---|---|---|"]
    for x in rows:
        others = [k for k in x["predicates"].split(";") if k and k != x["primary"]]
        L.append(f"| {x['k']} | {x['pages']} | {x['ocr_pages']} | {x['ifu_source']}, {x['ifu_words']} | {x['desc_words']} | {x['primary']} | {' '.join(others)} | {x['warnings']} |")
    L.append("")
    for r in recs:
        L += _md_doc(r)
    (out / "probe_report.md").write_text("\n".join(L), encoding="utf-8")


def print_summary(recs: list[dict]) -> None:
    print(f"\n{'K':<12}{'pg':>4}{'ocr':>5}{'ifu_w':>7}{'desc_w':>7}  {'primary':<10} warnings")
    for r in recs:
        p = r.get("parsed") or {}
        prim = next((x["k"] for x in p.get("predicates", []) if x["role"] == "primary"), "-")
        print(f"{r['k']:<12}{str(r.get('n_pages', '-')):>4}{len(r.get('ocr_pages') or []):>5}{str(p.get('ifu_words', '-')):>7}{str(p.get('desc_words', '-')):>7}  {prim:<10} "
              + (", ".join(w.split(":")[0] for w in r["warnings"]) or "ok"))
