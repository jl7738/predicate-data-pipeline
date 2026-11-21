"""Runs on whatever real PDFs are in tests/real_pdfs/ (skipped when the folder is empty). See that folder's README."""
import json
from pathlib import Path

import pytest

from predx.config import load_config
from predx.probe import probe_pdf, subject_from_name

HERE = Path(__file__).parent / "real_pdfs"
PDFS = sorted(HERE.glob("*.pdf"))
EXPECTED = json.loads((HERE / "expected.json").read_text()) if (HERE / "expected.json").exists() else {}


@pytest.mark.skipif(not PDFS, reason="no real PDFs in tests/real_pdfs/ (see its README)")
@pytest.mark.parametrize("pdf", PDFS or [None], ids=lambda p: p.name if p else "none")
def test_real_pdf(pdf):
    k = subject_from_name(pdf)
    r = probe_pdf(pdf, k, load_config([]))
    print(f"\n{k}: pages={r.get('n_pages')} ocr={len(r.get('ocr_pages', []))} warnings={[w.split(':')[0] for w in r['warnings']]}")
    assert not r["error"], r["error"]
    assert r["n_chars"] > 400, "almost no text extracted"
    exp = EXPECTED.get(k)
    if exp:
        p = r["parsed"]
        got = {x["k"]: x["role"] for x in p["predicates"]}
        if "primary" in exp:
            assert got.get(exp["primary"]) == "primary", got
        if "additional" in exp:
            assert {x for x, ro in got.items() if ro == "additional"} == set(exp["additional"]), got
        if "reference" in exp:
            assert {x for x, ro in got.items() if ro == "reference"} == set(exp["reference"]), got
        if "ifu_contains" in exp:
            assert exp["ifu_contains"].lower() in (p.get("ifu") or "").lower()
