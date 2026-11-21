"""CONCEPT TEST: can we reliably spot FDA submission numbers (and not invent them)?"""
from predx.knumbers import find_knumbers, k_year, kind_of


def ks(text, **kw):
    return [(h.k, h.kind) for h in find_knumbers(text, **kw)]


def test_clean_numbers_and_kinds():
    t = "Predicate K211140; also BK170012, DEN200055 and PMA P160012."
    assert ks(t) == [("K211140", "k"), ("BK170012", "bk"), ("DEN200055", "den"), ("P160012", "pma")]


def test_tolerates_spacing_and_case():
    assert [k for k, _ in ks("see K 21 1140, k211141 and K-211142")] == ["K211140", "K211141", "K211142"]


def test_does_not_match_inside_other_tokens():
    assert ks("SK211140 K2111400 K21114 AK211140 part K-12") == []


def test_ocr_repair_only_when_number_is_real():
    text = "predicate K21l140 (OCR)"          # lowercase L instead of 1
    assert ks(text) == []                       # unknown -> never guessed
    assert ks(text, known={"K211140"}) == [("K211140", "k")]
    assert ks(text, accept_unverified_ocr=True) == [("K211140", "k")]   # used for masking only
    assert ks("KO12345", known={"K012345"}) == [("K012345", "k")]    # letter O for zero
    assert ks("KO12345", known={"K999999"}) == []


def test_year_decoding():
    assert k_year("K211140") == 2021
    assert k_year("K992001") == 1999
    assert k_year("K760001") == 1976
    assert k_year("DEN200055") is None
    assert kind_of("DEN1") == "den"
