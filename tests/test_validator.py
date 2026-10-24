"""Focused tests: deterministic validation rules."""

from validator import (
    check_caps_and_symbols,
    check_category,
    check_contact_details,
    check_description_length,
    check_exact_duplicate,
    check_keyword_stuffing,
    check_near_duplicate,
    check_price_format,
    check_required_fields,
    check_title_length,
    find_model_triggers,
    run_all_checks,
)

GOOD = {
    "title": "Aurora 20000mAh Power Bank, 22.5W Fast Charging, Black",
    "description": (
        "Portable power bank with two USB-C ports and one USB-A port. "
        "Charges a phone about four times. Includes a 30 cm USB-C cable."
    ),
    "category": "Electronics",
    "price": "49.99",
    "attributes": {"brand": "Aurora", "model": "PB-20", "condition": "New", "warranty": "12 months"},
    "seller": "Aurora Store",
    "tags": ["power bank", "charger"],
}


def test_good_listing_passes_all():
    failed = [c for c in run_all_checks(GOOD, []) if not c["passed"] and not c["skipped"]]
    assert failed == []


def test_required_fields_detects_missing():
    out = check_required_fields({**GOOD, "title": "   "})
    assert not out["passed"] and "title" in out["message"]


def test_price_format_rejects_symbols_and_decimals():
    assert not check_price_format({**GOOD, "price": "$49.99"})["passed"]
    assert not check_price_format({**GOOD, "price": "49.999"})["passed"]
    assert not check_price_format({**GOOD, "price": "0"})["passed"]
    assert check_price_format(GOOD)["passed"]


def test_title_and_description_length_bounds():
    assert not check_title_length({**GOOD, "title": "Short"})["passed"]
    assert not check_description_length({**GOOD, "description": "Too short"})["passed"]
    assert check_title_length(GOOD)["passed"]
    assert check_description_length(GOOD)["passed"]


def test_category_must_be_supported():
    assert not check_category({**GOOD, "category": "Gadgets"})["passed"]
    assert check_category(GOOD)["passed"]


def test_contact_details_detects_phone():
    bad = {**GOOD, "description": GOOD["description"] + " Call 9876543210 today."}
    out = check_contact_details(bad)
    assert not out["passed"] and "phone" in out["message"]


def test_duplicates_need_same_seller():
    other = {**GOOD}
    assert not check_exact_duplicate(GOOD, [other])["passed"]
    assert check_exact_duplicate(GOOD, [{**other, "seller": "Other Shop"}])["passed"]
    near = {**GOOD, "title": GOOD["title"].replace("Black", "White")}
    assert not check_near_duplicate(near, [GOOD])["passed"]


def test_keyword_stuffing_and_caps():
    assert not check_keyword_stuffing({**GOOD, "title": "Charger charger charger fast"})["passed"]
    assert not check_caps_and_symbols({**GOOD, "title": "BEST HEADPHONES!!!"})["passed"]
    assert check_caps_and_symbols(GOOD)["passed"]


def test_triggers_are_signals_not_failures():
    triggers = find_model_triggers({**GOOD, "description": "100% guaranteed cure, doctor recommended."})
    assert {t["phrase"] for t in triggers} >= {"100%", "guaranteed", "cure", "doctor"}
    
def test_public_api_exports():
    import validator
    for name in ["CATEGORY_ATTRIBUTES", "REQUIRED_FIELDS", "ALLOWED_CONDITIONS",
                 "run_all_checks", "find_model_triggers", "review_listing"]:
        assert hasattr(validator, name)
