"""validator/checks_pricing.py - price-related deterministic checks."""

from validator.helpers import PRICE_RE, clean_text, result


# ============================================================
# 4. PRICE FORMAT
# POL-3.3
# ============================================================

def check_price_format(listing):
    price = clean_text(listing.get("price"))

    if not PRICE_RE.fullmatch(price):
        return result(
            "price_format",
            False,
            ["POL-3.3"],
            (
                f"Price '{price}' is invalid. "
                "Use a positive number with at most two decimals, "
                "without currency symbols or commas."
            ),
        )

    # Format is valid, now ensure > 0.
    if float(price) <= 0:
        return result(
            "price_format",
            False,
            ["POL-3.3"],
            "Price must be greater than 0.",
        )

    return result(
        "price_format",
        True,
        ["POL-3.3"],
    )


# ============================================================
# 5. PRICE RANGE
# POL-3.4
# ============================================================

def check_price_range(listing):
    price = clean_text(listing.get("price"))

    # Don't report a second failure when the format itself
    # is already invalid.
    if not PRICE_RE.fullmatch(price):
        return result(
            "price_range",
            True,
            ["POL-3.4"],
            "Skipped because price format is invalid.",
            skipped=True,
        )

    numeric_price = float(price)

    passed = 0.01 <= numeric_price <= 100000.00

    return result(
        "price_range",
        passed,
        ["POL-3.4"],
        ""
        if passed
        else (
            f"Price {price} must be between "
            "0.01 and 100,000.00."
        ),
    )
