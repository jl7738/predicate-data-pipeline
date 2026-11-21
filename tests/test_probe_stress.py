"""Real-PDF path (pdftotext + tesseract) on synthetic PDFs with planted truth, plus the OCR degradation curve."""
import random
import shutil

import pytest

from predx.config import load_config
from predx.probe import probe_pdf, run_probe
from predx.stress import _sim, make_doc, run_stress
from predx.synth_pdf import render_pdf

CFG = load_config([])
OCR = pytest.mark.skipif(not (shutil.which("pdftotext") and shutil.which("pdftoppm") and shutil.which("tesseract")),
                         reason="poppler / tesseract not installed")


@OCR
def test_probe_recovers_planted_predicates_from_a_digital_pdf(tmp_path):
    d = make_doc(random.Random(11), 3)
    p = tmp_path / f"{d['k']}.pdf"
    render_pdf(d["pages"], p, scanned=False)
    r = probe_pdf(p, d["k"], CFG)
    got = {x["k"]: x["role"] for x in r["parsed"]["predicates"]}
    assert got.get(d["primary"]) == "primary"
    assert {k for k, ro in got.items() if ro in ("primary", "additional")} == {d["primary"], *d["additional"]}
    assert {k for k, ro in got.items() if ro == "reference"} == set(d["reference"])
    assert _sim(r["parsed"]["ifu"], d["ifu"]) > 0.95 and _sim(r["parsed"]["description"], d["desc"]) > 0.95
    assert r["ocr_pages"] == []


@OCR
def test_probe_writes_report_files_and_flags_problems(tmp_path):
    ds = [make_doc(random.Random(20 + i), i) for i in range(2)]
    pdfs = []
    for d in ds:
        p = tmp_path / "in" / f"{d['k']}.pdf"
        render_pdf(d["pages"], p, scanned=False)
        pdfs.append(p)
    blank = tmp_path / "in" / "K259999.pdf"
    render_pdf(["", ""], blank, scanned=False)                       # a PDF with no text layer and nothing to OCR
    recs = run_probe(CFG, [], pdfs + [blank], tmp_path / "out", fetch=False)
    assert (tmp_path / "out" / "probe_report.md").exists() and (tmp_path / "out" / "probe_summary.csv").exists()
    assert (tmp_path / "out" / "text" / f"{ds[0]['k']}.txt").exists()
    bad = [r for r in recs if r["k"] == "K259999"][0]
    assert any(w.startswith("NO_TEXT") for w in bad["warnings"]) and any(w.startswith("NO_IFU") for w in bad["warnings"])


@OCR
def test_ocr_degradation_curve_guard(tmp_path):
    """Clean scans must be perfect; mild degradation must stay usable. Guards the OCR pre-processing choice
    (binarising with Otsu would pass the prose checks but lose K-number digits: 0/6/9 confusions)."""
    rows = run_stress(CFG, [0, 1], 4, 5, tmp_path)
    clean, mild = rows
    assert clean["ifu_ok"] == 1 and clean["gold_recall"] == 1 and clean["gold_precision"] == 1
    assert mild["ifu_ok"] >= 0.75 and mild["gold_recall"] >= 0.75 and mild["all_K_digits_survive_ocr"] >= 0.75
