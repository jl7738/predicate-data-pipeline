"""CONCEPT TEST: primary / additional / reference roles, and the naive first-other-number baseline."""
from predx.predicates import extract_predicates


def roles(pages, subject="K211140", **kw):
    r = extract_predicates(pages, subject, **kw)
    return {p["k"]: p["role"] for p in r["predicates"]}, r


def test_labelled_lines():
    t = """Re: K211140
PREDICATE DEVICE
Predicate Device: Alpha One (K111111)
Additional Predicate Device: Beta Two (K222222)
Reference Device: Gamma Three (K333333)
DEVICE DESCRIPTION
The device builds on earlier work."""
    got, r = roles([t])
    assert got == {"K111111": "primary", "K222222": "additional", "K333333": "reference"}
    assert r["primary_by"] == "label" or r["primary_by"] == "first_candidate"


def test_generic_list_uses_first_other_number_rule():
    got, r = roles(["Predicate Devices: K111111, K222222 and K444444\nDEVICE DESCRIPTION\nx"])
    assert got == {"K111111": "primary", "K222222": "additional", "K444444": "additional"}
    assert r["first_other"] == "K111111" and r["ambiguous"]


def test_trailing_parenthetical_labels():
    t = "PREDICATE DEVICES\nAlpha One, K111111 (primary predicate)\nBeta Two, K222222 (additional predicate)\nGamma, K333333 (reference device)\n"
    got, _ = roles([t])
    assert got == {"K111111": "primary", "K222222": "additional", "K333333": "reference"}


def test_subject_number_never_a_predicate():
    got, _ = roles(["Re: K211140\nPredicate Device: X (K111111)\nThis 510(k) K211140 is new"])
    assert "K211140" not in got


def test_predicate_and_reference_heading_is_not_read_as_reference():
    got, _ = roles(["Predicate and Reference Devices: K111111, K222222\n"])
    assert set(got.values()) <= {"primary", "additional"}
    assert got["K111111"] == "primary"


def test_non_510k_identifiers_are_kept_but_flagged_by_kind():
    r = extract_predicates(["Predicate Device: Legacy Valve (P160012)\nAdditional Predicate Device: Novel (DEN200055)"], "K211140")
    kinds = {p["k"]: p["kind"] for p in r["predicates"]}
    assert kinds == {"P160012": "pma", "DEN200055": "den"}


def test_numbers_outside_predicate_context_are_not_gold():
    t = "The company previously sold a valve.\nProduct history includes K555555 from 2010.\n\nPREDICATE DEVICE\nPredicate Device: A (K111111)\n"
    got, _ = roles([t], fallback=False)
    assert got["K111111"] == "primary" and got["K555555"] == "other"


def test_fallback_to_first_other_number_when_nothing_is_labelled():
    t = "Some free text mentioning K777777 only once."
    got, r = roles([t])
    assert got == {"K777777": "primary"} and r["primary_by"] == "first_other_fallback" and r["ambiguous"]
    got, _ = roles([t], fallback=False)
    assert got == {"K777777": "other"}


def test_ocr_damaged_number_repaired_only_when_known():
    t = "Predicate Device: Alpha (K11l111)"
    assert roles([t], known={"K111111"})[0] == {"K111111": "primary"}
    assert roles([t], known={"K999999"}, fallback=False)[0] == {}
