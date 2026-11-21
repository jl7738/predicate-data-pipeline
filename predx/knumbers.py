"""Finding and normalising FDA submission numbers in free text.

K / BK  : 510(k) (BK = CBER).        These can be in the openFDA 510(k) corpus.
DEN     : De Novo.                    Never in the 510(k) corpus.
P       : PMA.                        Never in the 510(k) corpus.

Clean matches are always returned. OCR-damaged matches (e.g. 'K21l140', 'KO12345') are returned only when
the repaired number is a real K-number (present in `known`), unless accept_unverified_ocr=True (masking).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_K = re.compile(r"(?<![A-Za-z0-9])(?P<pre>BK|K)(?:[ \t]?|-)(?P<yy>\d{2})[ \t]?(?P<seq>\d{4})(?![0-9])", re.I)
_DEN = re.compile(r"(?<![A-Za-z0-9])DEN(?P<num>\d{6})(?![0-9])", re.I)
_PMA = re.compile(r"(?<![A-Za-z0-9])P(?P<num>\d{6})(?![0-9])")
_LEN = re.compile(
    r"(?<![A-Za-z0-9])(?P<pre>BK|K)(?P<body>[0-9OoIlSsBZ][0-9OoIlSsBZ \t]{4,8}[0-9OoIlSsBZ])(?![A-Za-z0-9])", re.I
)
_REPAIR = str.maketrans("OoIlSsBZ", "00115582")


@dataclass(frozen=True)
class Hit:
    k: str
    kind: str  # k | bk | den | pma
    start: int
    end: int
    raw: str
    repaired: bool = False


def kind_of(k: str) -> str:
    k = k.upper()
    if k.startswith("BK"):
        return "bk"
    if k.startswith("DEN"):
        return "den"
    if k.startswith("K"):
        return "k"
    if k.startswith("P"):
        return "pma"
    return "other"


def is_510k(k: str) -> bool:
    return kind_of(k) in ("k", "bk")


def k_year(k: str) -> int | None:
    """Receipt year encoded in a K/BK number (510(k) numbering starts 1976)."""
    m = re.match(r"(?:BK|K)(\d{2})\d{4}$", k.upper())
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 76 else 2000 + yy


def find_knumbers(text: str, known=None, accept_unverified_ocr: bool = False) -> list[Hit]:
    hits: list[Hit] = []
    for m in _K.finditer(text):
        pre = m.group("pre").upper()
        hits.append(Hit(f"{pre}{m.group('yy')}{m.group('seq')}", "bk" if pre == "BK" else "k", m.start(), m.end(), m.group(0)))
    for m in _DEN.finditer(text):
        hits.append(Hit("DEN" + m.group("num"), "den", m.start(), m.end(), m.group(0)))
    for m in _PMA.finditer(text):
        hits.append(Hit("P" + m.group("num"), "pma", m.start(), m.end(), m.group(0)))

    spans = [(h.start, h.end) for h in hits]
    for m in _LEN.finditer(text):
        if any(m.start() < e and s < m.end() for s, e in spans):
            continue
        body = re.sub(r"[ \t]", "", m.group("body"))
        if len(body) != 6:
            continue
        rep = body.translate(_REPAIR)
        if not rep.isdigit() or rep == body:
            continue
        pre = m.group("pre").upper()
        k = pre + rep
        if accept_unverified_ocr or (known is not None and k in known):
            hits.append(Hit(k, "bk" if pre == "BK" else "k", m.start(), m.end(), m.group(0), True))
    hits.sort(key=lambda h: h.start)
    return hits
