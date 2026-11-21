"""Locate the two model-input sections (Indications for Use, Device Description) in a 510(k) summary PDF's text.

Real documents are: decision letter -> FDA "Indications for Use" form page -> 510(k) Summary.
Section layout and heading wording vary a lot, so extraction is rule-based and *measured* (see audit + labeling).

Design choices (all reversible, all recorded in the output record):
  * Indications: prefer the text of the FDA form page (the official, stamped statement); else the summary heading.
  * Description: first summary section titled like "Device Description" with enough words.
  * Sections stop at the next recognised heading (predicate / substantial-equivalence / performance / conclusion ...)
    so predicate discussions are NOT swallowed into the model input (label leakage).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def _w(p: str) -> str:
    # headings must stay on one line: \s -> [ \t]
    return p.replace(r"\s", r"[ \t]")


ENUM = r"(?:(?:[IVXLC]{1,6}|\d{1,2}(?:\.\d{1,2}){0,2}|[A-Za-z])[\.\):]?[ \t]+)?"

_HEAD_PATTERNS = {
    # Indications. Singular/plural, "Statement", combined "Intended Use / Indications for Use" variants.
    "ifu": _w(
        r"indications?\s+for\s+use\s*/\s*intended\s+use"
        r"|intended\s+use(?:\s*(?:/|and|&)\s*indications?\s+for\s+use)?"
        r"|indications?\s*(?:/|and|&)\s*intended\s+use"
        r"|statement\s+of\s+indications?\s+for\s+use"
        r"|indications?\s+for\s+use(?:\s+statement)?"
        r"|indications"
    ),
    "desc": _w(
        r"device\s+description|description\s+of\s+(?:the\s+)?(?:subject\s+)?device|product\s+description"
        r"|system\s+description|device\s+overview"
    ),
    # Predicate / comparison sections (their text is the answer key: never part of a model input).
    "pred": _w(
        r"(?:legally\s+marketed\s+)?predicate(?:\s+and\s+reference)?\s+devices?(?:\s*\(s\))?"
        r"|predicate\s*\(s\)|predicate\s+device\s+(?:information|comparison)"
        r"|summary\s+of\s+substantial\s+equivalence"
        r"|substantial\s+equivalence(?:\s+(?:discussion|comparison|determination|information|table|summary))?"
        r"|table\s+\d+\s*[-–:.]?\s*substantial\s+equivalence(?:\s+table)?"
        r"|comparison\s+(?:of|to|with)\s+(?:the\s+)?(?:technological\s+characteristics|predicate\s+devices?"
        r"|subject\s+device\s+(?:and|to|with)\s+(?:the\s+)?predicate\s+devices?)"
        r"|comparison\s+table"
        r"|technolog(?:y|ical)\s+(?:characteristics?|comparison)(?:\s+(?:comparison|table|summary))?"
        r"|reference\s+devices?|predicate"
    ),
    # Other major sections: they END a description / indications section.
    "other": _w(
        r"(?:summary\s+of\s+)?(?:non-?clinical|bench|clinical|biocompatibility|sterili[sz]ation|software|electrical|animal"
        r"|usability|human\s+factors|performance|shelf(?:\s|-)?life|packaging|electromagnetic|emc|cybersecurity|mechanical)"
        r"(?:\s+(?:and|&)\s+\w+)*(?:\s+(?:performance|safety|compatibility|characteristics|verification|validation))*"
        r"\s+(?:testing|tests?|data|studies|evaluation|information|summary)"
        r"|performance\s+(?:data|testing|summary|characteristics)"
        r"|conclusions?(?:\s+drawn)?|summary\s+of\s+(?:testing|studies|clinical|non-?clinical|performance)(?:\s+\w+)?"
        r"|general\s+information|review\s+summary|summary"
    ),
    # Field labels of the submitter / device block. They end a section but NOT a predicate section
    # (a predicate entry is usually followed by '510(k) number: K...' which must stay inside it).
    "field": _w(
        r"sponsor|submitter(?:'s)?(?:\s+information)?|contact(?:\s+person)?|date\s+(?:prepared|of\s+(?:preparation|summary))"
        r"|device(?:\s+trade)?\s+name|trade\s+name|proprietary(?:\s+or\s+trade)?\s+name|common(?:\s+or\s+usual)?\s+name"
        r"|classification(?:\s+name)?|regulation(?:\s+number)?|product\s+code|device\s+class(?:ification)?|device\s+information"
        r"|510\(k\)\s+(?:summary|number)|regulatory\s+(?:status|information)|type\s+of\s+use"
    ),
}


def _compile(pat: str):
    return re.compile(
        rf"^[ \t]*{ENUM}(?P<h>{pat})[ \t]*(?:\((?:describe|continued|cont\.?)\))?[ \t]*"
        rf"(?:(?P<sep>[:\-–—])[ \t]*(?P<rest>.*)|(?P<end>)[ \t]*$)",
        re.I | re.M,
    )


_HEAD_RX = {cat: _compile(p) for cat, p in _HEAD_PATTERNS.items()}


@dataclass
class Heading:
    cat: str
    line_start: int
    body_start: int
    title: str


def find_headings(text: str) -> list[Heading]:
    found: dict[int, Heading] = {}
    for cat, rx in _HEAD_RX.items():  # dict order = priority
        for m in rx.finditer(text):
            ls = m.start()
            if ls in found:
                continue
            body = m.start("rest") if m.group("sep") else m.end()
            found[ls] = Heading(cat, ls, body, m.group("h"))
    return [found[k] for k in sorted(found)]


# ---- FDA "Indications for Use" form page -----------------------------------------------------------------------
_IFU_HEAD_FORM = re.compile(
    r"indications?[ \t]+for[ \t]+use(?:[ \t]+statement)?(?:[ \t]*\([ \t]*describe[ \t]*\))?[ \t]*(?::|(?=[ \t]*(?:\n|$)))",
    re.I | re.M,
)
_FORM_STOP_PRIMARY = re.compile(
    r"type[ \t]+of[ \t]+use|concurrence[ \t]+of|please[ \t]+do[ \t]+not[ \t]+write|continue[ \t]+on[ \t]+a[ \t]+separate",
    re.I,
)
_FORM_STOP_SECONDARY = re.compile(r"^[ \t]*[\[\(☒☐xX_ ]*prescription[ \t]+use", re.I | re.M)


def extract_form_ifu(pages: list[str]):
    """Returns (raw_text | None, form_page_indices:set[int])."""
    form_pages: set[int] = set()
    best = None
    for pi, page in enumerate(pages):
        stop = _FORM_STOP_PRIMARY.search(page) or _FORM_STOP_SECONDARY.search(page)
        if not stop:
            continue
        heads = [m for m in _IFU_HEAD_FORM.finditer(page) if m.end() <= stop.start()]
        if not heads:
            continue
        form_pages.add(pi)
        if best is None:
            best = page[heads[-1].end(): stop.start()]
    return best, form_pages


# ---- cleaning --------------------------------------------------------------------------------------------------
_PAGE_LINE = re.compile(
    r"^[ \t]*(?:(?:K|BK)\d{6}[^\n]{0,60}?)?page[ \t]+\d+(?:[ \t]+of[ \t]+\d+)?[ \t]*$|^[ \t]*-?[ \t]*\d{1,3}[ \t]*-?[ \t]*$",
    re.I | re.M,
)


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\(cid:\d+\)", " ", s)
    s = _PAGE_LINE.sub(" ", s)
    s = re.sub(r"-\n[ \t]*(?=[a-z])", "", s)
    s = re.sub(r"[☒☐■□▪●◦\f]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _words(s: str | None) -> int:
    return len(s.split()) if s else 0


def _cap(s: str, max_chars: int):
    if len(s) <= max_chars:
        return s, False
    cut = s[:max_chars]
    m = list(re.finditer(r"[.!?]\s", cut))
    if m and m[-1].end() > max_chars * 0.6:
        cut = cut[: m[-1].end()]
    return cut.strip(), True


_SE_LANG = re.compile(r"predicate|substantial(?:ly)?\s+equivalen|\b(?:BK|K)\s?\d{2}\s?\d{4}\b", re.I)
_STATEMENT = re.compile(
    r"510\(k\)\s+statement|(?:safety\s+and\s+effectiveness|summary).{0,160}(?:will\s+be\s+made\s+available|available\s+upon\s+request|can\s+be\s+obtained)",
    re.I | re.S,
)


_SUMMARY_HEAD = re.compile(r"summary\s+of\s+safety\s+and\s+effectiveness|510\s*\(k\)\s+summary", re.I)
_LETTER_PAGE = re.compile(r"public\s+health\s+service|dear\s+(?:mr|ms|mrs|dr)|sincerely\s+yours|indications\s+for\s+use\s+statement|form\s+approved", re.I)


def _summary_body(pages: list[str], skip: set) -> str | None:
    """Old-format summaries have no 'Device Description' heading: use the 510(k) summary page itself
    (plus the next page if it is not an FDA letter / IFU form). Flagged desc_source=summary_body."""
    for i, pg in enumerate(pages):
        if i in skip or not _SUMMARY_HEAD.search(pg[:1500]) or _LETTER_PAGE.search(pg[:600]):
            continue
        body = pg
        if i + 1 < len(pages) and i + 1 not in skip and not _LETTER_PAGE.search(pages[i + 1][:600]):
            body += "\n" + pages[i + 1]
        return clean_text(body)
    return None


def parse_sections(pages: list[str], cfg: dict) -> dict:
    pcfg = cfg["parse"]
    min_ifu, min_desc = pcfg["min_words"]["ifu"], pcfg["min_words"]["description"]
    max_chars = pcfg["max_section_chars"]

    form_raw, form_pages = extract_form_ifu(pages)
    summary_text = "\n".join(p for i, p in enumerate(pages) if i not in form_pages)
    heads = find_headings(summary_text)

    bodies = []
    for i, h in enumerate(heads):
        end = heads[i + 1].line_start if i + 1 < len(heads) else len(summary_text)
        bodies.append((h, summary_text[h.body_start:end]))

    def candidates(cat: str) -> list[str]:
        out = []
        for i, (h, body) in enumerate(bodies):
            if h.cat != cat:
                continue
            j = i
            while _words(clean_text(body)) < 3 and j + 1 < len(bodies) and bodies[j + 1][0].cat == cat:
                j += 1
                body = body + " " + bodies[j][1]
            out.append(clean_text(body))
        return out

    ifu_summary = candidates("ifu")
    desc_c = candidates("desc")

    # --- indications
    ifu, ifu_source = None, None
    form_clean = clean_text(form_raw) if form_raw else None
    pools = {
        "form": [form_clean] if form_clean else [],
        "summary": ifu_summary,
    }
    for src in pcfg["ifu_preference"]:
        for c in pools.get(src, []):
            if _words(c) >= min_ifu:
                ifu, ifu_source = c, src
                break
        if ifu:
            break
    if not ifu:  # nothing long enough: keep the longest short one so audits can see it
        allc = [(c, s) for s, cs in pools.items() for c in cs]
        if allc:
            ifu, ifu_source = max(allc, key=lambda t: _words(t[0]))

    # --- description
    desc = None
    for c in desc_c:
        if _words(c) >= min_desc:
            desc = c
            break
    if desc is None and desc_c:
        desc = max(desc_c, key=_words)

    desc_source = "heading" if desc else None
    if (desc is None or _words(desc) < min_desc) and pcfg.get("desc_fallback", "summary_body") == "summary_body":
        fb = _summary_body(pages, form_pages)
        if fb and _words(fb) >= min_desc and _words(fb) > _words(desc):
            desc, desc_source = fb, "summary_body"

    ifu_capped = desc_capped = False
    if ifu and ifu_source != "form":
        ifu, ifu_capped = _cap(ifu, max_chars)
    if desc:
        desc, desc_capped = _cap(desc, max_chars)

    full = "\n".join(pages)
    statement_like = bool(_STATEMENT.search(full)) and not (ifu and desc) and len(full) < 12000
    return {
        "ifu": ifu or None,
        "description": desc or None,
        "ifu_source": ifu_source,
        "ifu_words": _words(ifu),
        "desc_words": _words(desc),
        "desc_source": desc_source,
        "ifu_capped": ifu_capped,
        "desc_capped": desc_capped,
        "form_page": min(form_pages) if form_pages else None,
        "n_headings": len(heads),
        "se_language_ifu": bool(ifu and _SE_LANG.search(ifu)),
        "se_language_desc": bool(desc and _SE_LANG.search(desc)),
        "statement_like": statement_like,
    }
