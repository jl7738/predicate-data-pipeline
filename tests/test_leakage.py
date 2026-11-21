"""CONCEPT TEST: queries must not name the answer."""
from predx.leakage import leak_stats, mask_doc_text, mask_names, mask_query_text, residual_leak

TEXT = "This design builds on K123456 and the Acme FlexPro Catheter. It is a catheter for vascular access. Similar to the predicate device."
CFG = {"leakage": {"name_min_chars": 8, "name_min_tokens": 2}}


def test_standard_masks_numbers_and_gold_names_only():
    out, st = mask_query_text(TEXT, "standard", ["Acme FlexPro Catheter"], "Zeta Catheter", CFG)
    assert "K123456" not in out and "Acme FlexPro" not in out
    assert "[K-NUMBER]" in out and "[DEVICE]" in out
    assert "vascular access" in out and "predicate" in out          # standard keeps the language
    assert st["k"] == 1 and st["names"] == 1


def test_name_that_is_part_of_the_subjects_own_name_is_kept():
    out, _ = mask_names("a pulse oximeter system", ["Pulse Oximeter System"], own_name="Masimo Pulse Oximeter System")
    assert "pulse oximeter system" in out.lower()


def test_short_or_generic_names_are_not_masked():
    out, n = mask_names("The catheter works.", ["Catheter"], "")
    assert out == "The catheter works." and n == 0


def test_strict_also_drops_predicate_sentences_and_brand_tokens():
    out, _ = mask_query_text(TEXT, "strict", ["Acme FlexPro Catheter"], "Zeta Catheter", CFG)
    assert "predicate" not in out.lower() and "FlexPro" not in out and "Acme" not in out
    assert "vascular access" in out


def test_leak_stats_and_residuals():
    s = leak_stats(TEXT, "", ["K123456"], ["Acme FlexPro Catheter"], "K999999")
    assert s["raw_gold_k"] == 1 and s["raw_gold_name"] == 1 and s["pred_language"]
    out, _ = mask_query_text(TEXT, "standard", ["Acme FlexPro Catheter"], "", CFG)
    assert residual_leak(out, ["Acme FlexPro Catheter"]) == {"residual_k": 0, "residual_gold_name": 0}
    assert residual_leak(TEXT, ["Acme FlexPro Catheter"])["residual_k"] == 1


def test_documents_get_number_masking_too():
    out, st = mask_doc_text("Cleared as K111111 and BK222222.", "standard")
    assert out == "Cleared as [K-NUMBER] and [K-NUMBER]." and st["k"] == 2
    assert mask_doc_text("K111111", "none")[0] == "K111111"
