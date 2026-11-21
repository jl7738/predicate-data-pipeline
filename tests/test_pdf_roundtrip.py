"""CONCEPT TEST: real PDFs in, text out. Digital PDFs use the text layer; scanned (image-only) PDFs fall back to OCR."""
import shutil

import pytest

from predx.config import load_config
from predx.pdftext import extract_pdf
from predx.synth_pdf import render_pdf

CFG = load_config([])
PAGES = [
    "Food and Drug Administration\nRe: K211140\nDear Sir or Madam: we have reviewed your premarket notification.",
    "Indications for Use\n510(k) Number (if known)\nK211140\nIndications for Use (Describe)\nThe pulmonary nodule triage software assists radiologists reviewing chest CT studies.\nType of Use (Select one or both, as applicable)",
]


@pytest.mark.skipif(not shutil.which("pdftotext"), reason="poppler not installed")
def test_digital_pdf_uses_text_layer_without_ocr(tmp_path):
    p = tmp_path / "a.pdf"
    render_pdf(PAGES, p, scanned=False)
    r = extract_pdf(p, CFG["text"])
    assert r["n_pages"] == 2 and r["ocr_pages"] == []
    assert "pulmonary nodule triage software" in r["pages"][1]


@pytest.mark.skipif(not (shutil.which("pdftoppm") and shutil.which("tesseract")), reason="poppler/tesseract not installed")
def test_scanned_pdf_is_recovered_by_ocr(tmp_path):
    p = tmp_path / "s.pdf"
    render_pdf(PAGES, p, scanned=True)
    r = extract_pdf(p, CFG["text"])
    assert r["ocr_pages"] == [0, 1]                                  # no text layer -> both pages OCR'd
    blob = " ".join(r["pages"]).lower()
    for w in ("pulmonary", "radiologists", "indications"):
        assert w in blob, blob[:300]
    assert "K211140" in " ".join(r["pages"]).replace(" ", "")        # the submission number survives OCR


def test_corrupt_pdf_raises_so_the_stage_can_log_text_failed(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"%PDF-1.4 this is not a pdf")
    with pytest.raises(Exception):
        extract_pdf(p, CFG["text"])


def test_tsv_table_roundtrip_and_k_token_pattern():
    from predx.pdftext import _KTOK, _mean_conf, _tsv_words, _words_to_text

    head = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    rows = [
        "5\t1\t1\t1\t1\t1\t10\t10\t40\t12\t91.5\tPredicate",
        "5\t1\t1\t1\t1\t2\t60\t10\t90\t12\t88.0\tDevice:",
        "5\t1\t1\t1\t1\t3\t160\t10\t80\t12\t80.0\t(K152876)",
        "5\t1\t2\t1\t1\t1\t10\t40\t60\t12\t95.0\tDevice",
        "5\t1\t2\t1\t1\t2\t80\t40\t60\t12\t-1\t ",
    ]
    words = _tsv_words("\n".join([head, *rows]))
    assert [w["text"] for w in words] == ["Predicate", "Device:", "(K152876)", "Device"]
    assert _words_to_text(words) == "Predicate Device: (K152876)\n\nDevice\n"          # paragraph break between blocks
    assert abs(_mean_conf(words) - (91.5 + 88 + 80 + 95) / 4) < 1e-9
    assert _KTOK.match("(K152876)") and _KTOK.match("K1S2876,") and _KTOK.match("BK150001")
    assert not _KTOK.match("K12") and not _KTOK.match("Knowledge") and not _KTOK.match("K2500011234")
