"""Extract cited predicate submission numbers and assign roles (primary / additional / reference / other).

Rules (heuristic, measured by `label-sample` / `label-eval`):
  1. Find every K/BK/DEN/P number in the document; drop the subject's own number.
  2. Give each occurrence a hint from the nearest label *before* it in the same paragraph
     ("Reference device:", "Additional predicate:", "Primary predicate:", "Predicate device:" ...) or a trailing
     parenthetical label "(reference)". Occurrences inside a predicate / substantial-equivalence section with no
     label count as generic predicates. Everything else is "other" (e.g. an earlier clearance of the same product).
  3. primary    = labelled primary, else the first generic predicate in document order (first-other-number rule).
     additional = remaining labelled-additional / generic predicates.
     reference  = labelled reference devices (excluded from gold).
  4. If nothing qualifies and `first_other_fallback`, the first other-number becomes the primary (flagged).
"""
from __future__ import annotations

import re

from .knumbers import find_knumbers
from .sections import find_headings

_LABELS = [  # (name, regex, tie priority: lower wins)
    ("ref", re.compile(r"reference[ \t]+(?:devices?|products?|predicates?)|\(reference\)|\breference[ \t]*:", re.I), 0),
    ("add", re.compile(r"(?:additional|secondary|second|other|supplemental|supplementary)[ \t]+predicate(?:[ \t]+devices?)?", re.I), 1),
    ("prim", re.compile(r"(?:primary|main)[ \t]+predicate(?:[ \t]+devices?)?", re.I), 2),
    ("gen", re.compile(r"predicate(?:[ \t]*\(s\))?(?:[ \t]+devices?)?|substantially[ \t]+equivalent|legally[ \t]+marketed", re.I), 3),
    # weak cues only count INSIDE a predicate / substantial-equivalence section (a control or reference device is
    # often "compared to" in the testing section and must not become a predicate)
    ("weak", re.compile(r"previously[ \t]+cleared|cleared[ \t]+(?:under|device)|equivalent[ \t]+to|compared[ \t]+(?:to|with)|comparison[ \t]+(?:to|with)", re.I), 4),
]
_TRAIL = re.compile(r"^[ \t]*[\),;]?[ \t]*\((?P<lab>[^)\n]{1,40})\)")


def _label_of(s: str) -> str | None:
    best = None
    for name, rx, pr in _LABELS:
        m = rx.search(s)
        if m:
            key = (m.end(), -pr)
            if best is None or key > best[0]:
                best = (key, name)
    return best[1] if best else None


def _closest_label(window: str) -> str | None:
    best = None
    for name, rx, pr in _LABELS:
        for m in rx.finditer(window):
            if name == "ref" and re.search(r"(?:and|/|&)[ \t]*$", window[max(0, m.start() - 16): m.start()], re.I):
                continue  # "predicate and reference devices" heading: ambiguous, treat as generic
            key = (m.end(), -pr)
            if best is None or key > best[0]:
                best = (key, name)
    return best[1] if best else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract_predicates(pages: list[str], subject_k: str, known=None, window: int = 260, fallback: bool = True) -> dict:
    text = "\n\f\n".join(pages)
    all_hits = find_knumbers(text, known)
    hits = [h for h in all_hits if h.k != subject_k]

    heads = find_headings(text)
    spans = []
    for i, h in enumerate(heads):
        if h.cat != "pred":
            continue
        j = i + 1
        while j < len(heads) and heads[j].cat in ("pred", "field"):   # field labels ('510(k) number: K...') stay inside
            j += 1
        spans.append((h.body_start, heads[j].line_start if j < len(heads) else len(text)))

    def in_span(pos: int) -> bool:
        return any(a <= pos < b for a, b in spans)

    order: list[str] = []
    info: dict[str, dict] = {}
    for idx, h in enumerate(hits):
        lo = max(0, h.start - window)
        w = text[lo: h.start]
        cut = max(w.rfind("\n\n"), w.rfind("\f"))
        if cut >= 0:
            w = w[cut + 1:]
        trail = _TRAIL.match(text[h.end: h.end + 60])
        lab = (_label_of(trail.group("lab")) if trail else None) or _closest_label(w)
        if lab == "weak":
            hint = "gen" if in_span(h.start) else "other"
        else:
            hint = lab if lab else ("gen" if in_span(h.start) else "other")
        if h.k not in info:
            order.append(h.k)
            info[h.k] = {
                "labeled_idx": None,
                "kind": h.kind,
                "repaired": h.repaired,
                "hints": [],
                "n": 0,
                "page": text.count("\f", 0, h.start),
                "snippet": _norm(text[max(0, h.start - 90): h.end + 50]),
            }
        info[h.k]["hints"].append(hint)
        info[h.k]["n"] += 1
        if hint != "other" and info[h.k]["labeled_idx"] is None:
            info[h.k]["labeled_idx"] = idx

    final: dict[str, str] = {}
    for k in order:
        non_other = [x for x in info[k]["hints"] if x != "other"]
        final[k] = non_other[0] if non_other else "other"

    # document order of the first *labelled* occurrence: an earlier passing mention in the description must not
    # outrank the predicate section when applying the first-other-number rule
    lab_order = sorted((k for k in order if final[k] != "other"), key=lambda k: info[k]["labeled_idx"])
    prim = [k for k in lab_order if final[k] == "prim"]
    gen = [k for k in lab_order if final[k] == "gen"]
    add = [k for k in lab_order if final[k] == "add"]
    primary, primary_by = None, None
    if prim:
        primary, primary_by = prim[0], "label"
    elif gen:
        primary, primary_by = gen[0], "first_candidate"
    elif add:
        primary, primary_by = add[0], "first_candidate"
    fallback_used = False
    if primary is None and fallback:
        for k in order:
            if info[k]["kind"] in ("k", "bk"):
                primary, primary_by, fallback_used = k, "first_other_fallback", True
                break

    preds = []
    for k in order:
        h = final[k]
        if k == primary:
            role = "primary"
        elif h == "ref":
            role = "reference"
        elif h in ("prim", "add", "gen"):
            role = "additional"
        else:
            role = "other"
        preds.append(
            {
                "k": k,
                "role": role,
                "kind": info[k]["kind"],
                "repaired": info[k]["repaired"],
                "n_mentions": info[k]["n"],
                "page": info[k]["page"],
                "snippet": info[k]["snippet"],
            }
        )
    n_cand = sum(1 for p in preds if p["role"] in ("primary", "additional"))
    n_other = sum(1 for p in preds if p["role"] == "other")
    first_other = next((h.k for h in hits if h.kind in ("k", "bk")), None)
    ambiguous = fallback_used or (primary_by == "first_candidate" and n_cand > 1) or (n_cand == 0 and n_other > 0)
    return {
        "predicates": preds,
        "first_other": first_other,
        "pred_section_found": bool(spans),
        "primary_by": primary_by,
        "n_k_mentions": len(all_hits),
        "ambiguous": bool(ambiguous),
    }
