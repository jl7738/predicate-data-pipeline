"""CONCEPT TEST: from a realistic summary PDF's text, do we get ONLY the indications + description (no predicate talk)?"""
import pytest

from predx.config import load_config
from predx.sections import clean_text, parse_sections

CFG = load_config([])

LETTER = """Food and Drug Administration
Re: K211140
Trade/Device Name: Reprocessed Pulse Oximeter
We have reviewed your Section 510(k) premarket notification and have determined the device is substantially equivalent
to legally marketed predicate devices.
K211140 - Acme Medical Page 2
"""
FORM = """Indications for Use
Form Approved: OMB No. 0910-0120
510(k) Number (if known)
K211140
Device Name
Reprocessed Pulse Oximeter
Indications for Use (Describe)
The reprocessed pulse oximeter sensor is indicated for continuous noninvasive monitoring of
arterial oxygen saturation in patients during motion and low perfusion.
Type of Use (Select one or both, as applicable)
Prescription Use (Part 21 CFR 801 Subpart D)
CONTINUE ON A SEPARATE PAGE IF NEEDED.
"""
SUMMARY = """510(k) SUMMARY
I. SUBMITTER
Acme Medical
II. PREDICATE DEVICE
Predicate Device: Masimo SET Sensor (K123456)
III. DEVICE DESCRIPTION
The sensor is a single-patient-use ven-
tilation accessory comprising an LED emitter and a photodiode detector. It connects to the monitor by a cable.
It is supplied non-sterile and is cleaned and function tested after each use before it is returned to the clinic.
IV. INDICATIONS FOR USE
The reprocessed pulse oximeter sensor is indicated for continuous noninvasive monitoring of arterial oxygen saturation.
V. COMPARISON OF TECHNOLOGICAL CHARACTERISTICS
The subject device has the same intended use as the predicate K123456.
VI. CONCLUSION
Substantially equivalent.
"""


def test_form_page_is_preferred_and_stops_at_type_of_use():
    r = parse_sections([LETTER, FORM, SUMMARY], CFG)
    assert r["ifu_source"] == "form"
    assert r["ifu"].startswith("The reprocessed pulse oximeter sensor is indicated")
    assert "Prescription" not in r["ifu"] and "Type of Use" not in r["ifu"]


def test_description_stops_before_comparison_and_has_no_predicate_text():
    r = parse_sections([LETTER, FORM, SUMMARY], CFG)
    d = r["description"]
    assert "LED emitter" in d and "ventilation accessory" in d       # hyphenated line break repaired
    assert "K123456" not in d and "predicate" not in d.lower()
    assert "Substantially equivalent" not in d
    assert not r["se_language_desc"] and not r["se_language_ifu"]


def test_falls_back_to_summary_heading_without_a_form_page():
    r = parse_sections([LETTER, SUMMARY], CFG)
    assert r["ifu_source"] == "summary"
    assert r["ifu"].startswith("The reprocessed pulse oximeter sensor is indicated for continuous")


@pytest.mark.parametrize("head", ["DEVICE DESCRIPTION", "Device Description:", "III. Device Description", "Description of the Device"])
def test_heading_variants(head):
    s = SUMMARY.replace("III. DEVICE DESCRIPTION", head)
    assert "LED emitter" in parse_sections([s], CFG)["description"]


def test_ifu_text_on_same_line_as_heading():
    s = "Intended Use / Indications for Use: The device is intended for adult patients in a hospital to measure pressure.\nCONCLUSION\nok"
    assert parse_sections([s], CFG)["ifu"].startswith("The device is intended for adult patients")


def test_missing_description_is_reported_not_invented():
    s = SUMMARY.replace("III. DEVICE DESCRIPTION", "III. OVERVIEW").replace("The sensor is a single", "x single")
    strict = load_config(overrides=["parse.desc_fallback=none"])
    r = parse_sections([LETTER, FORM, s], strict)
    assert r["description"] is None and r["desc_words"] == 0 and r["ifu_words"] >= 8
    r2 = parse_sections([LETTER, FORM, s], CFG)  # fallback on: never silent, always labelled
    assert r2["description"] is None or r2["desc_source"] == "summary_body"


def test_statement_only_document_is_detected():
    stmt = "510(k) STATEMENT\nAs required by 21 CFR 807.93, the safety and effectiveness information will be made available upon request.\n"
    r = parse_sections([LETTER, stmt], CFG)
    assert r["statement_like"] and r["ifu"] is None


def test_page_furniture_removed():
    assert clean_text("text\nK243175 - Boyle Wang Page 2\nmore\nPage 3 of 7\n  5  \nend") == "text more end"


def test_heading_regexes_compile_without_warnings_and_cover_hyphenated_variants():
    import subprocess, sys
    r = subprocess.run([sys.executable, "-W", "error", "-c", "import predx.sections"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]          # e.g. FutureWarning: possible nested set in a character class
    from predx.sections import find_headings
    cats = [(h.cat) for h in find_headings("Shelf-life Testing\nx\nShelf Life Data\ny\n")]
    assert cats == ["other", "other"]


def test_summary_body_fallback_for_old_format_summaries():
    """Pre-2000 summaries often have no 'Device Description' heading: the summary page becomes the description (flagged)."""
    from predx.config import load_config
    from predx.sections import parse_sections

    cfg = load_config()
    summary = (
        "SUMMARY OF SAFETY AND EFFECTIVENESS INFORMATION\nProprietary Name: Widget Syringe\n"
        "Predicate Device Information: A claim of substantial equivalence is made to Acme Syringe, K925039.\n"
        "Device Claims: For the injection of fluid into the body while reducing the risk of sharps injuries.\n"
        "Bench testing of the product confirms compliance with the applicable performance standards. "
        "Simulated use studies were completed by a wide variety of clinicians without failures in every case."
    )
    letter = "DEPARTMENT OF HEALTH & HUMAN SERVICES Public Health Service\nDear Mr. Smith: we have reviewed your notification."
    out = parse_sections([summary, letter], cfg)
    assert out["desc_source"] == "summary_body" and out["desc_words"] >= 15
    assert "Dear" not in out["description"] and "Public Health Service" not in out["description"]

    off = load_config(overrides=["parse.desc_fallback=none"])
    assert parse_sections([summary, letter], off)["description"] is None
