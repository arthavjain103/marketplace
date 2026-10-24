"""
validator/checks.py

All deterministic check functions for the Marketplace Listing Quality Reviewer.
No LLM is used for any of these checks.
"""
import re

from validator.checks_content import (  # noqa: F401
    check_caps_and_symbols,
    check_category,
    check_condition,
    check_contact_details,
    check_description_length,
    check_keyword_stuffing,
    check_required_attributes,
    check_required_fields,
    check_tags,
    check_title_length,
)
from validator.checks_pricing import check_price_format, check_price_range  # noqa: F401
from validator.constants import MODEL_TRIGGER_PHRASES
from validator.helpers import (
    _same_seller,
    clean_text,
    get_attributes,
    normalize_title,
    result,
    similarity,
)
# ============================================================
# 11. EXACT DUPLICATE
# POL-8.2
# ============================================================

def check_exact_duplicate(
    listing,
    other_listings,
):

    my_title = normalize_title(
        listing.get("title")
    )

    if not my_title:
        return result(
            "exact_duplicate",
            True,
            ["POL-8.2"],
            "Skipped because title is empty.",
            skipped=True,
        )

    for other in other_listings or []:

        if other is listing:
            continue

        if not _same_seller(listing, other):
            continue

        other_title = normalize_title(
            other.get("title")
        )

        if my_title == other_title:

            return result(
                "exact_duplicate",
                False,
                ["POL-8.2"],
                (
                    "Same seller already has this "
                    f"title: '{other.get('title')}'."
                ),
            )

    return result(
        "exact_duplicate",
        True,
        ["POL-8.2"],
    )


# ============================================================
# 12. NEAR DUPLICATE
# POL-8.3
# ============================================================

def check_near_duplicate(
    listing,
    other_listings,
):

    my_category = clean_text(
        listing.get("category")
    ).lower()

    my_title = normalize_title(
        listing.get("title")
    )

    my_description = clean_text(
        listing.get("description")
    )

    if not my_title and not my_description:
        return result(
            "near_duplicate",
            True,
            ["POL-8.3"],
            "Skipped because listing text is empty.",
            skipped=True,
        )

    for other in other_listings or []:

        if other is listing:
            continue

        # Same seller is required.
        if not _same_seller(listing, other):
            continue

        # Same category is required.
        other_category = clean_text(
            other.get("category")
        ).lower()

        if other_category != my_category:
            continue

        # Exact duplicate already handled separately.
        if (
            my_title
            and
            my_title == normalize_title(
                other.get("title")
            )
        ):
            continue

        title_similarity = similarity(
            listing.get("title"),
            other.get("title"),
        )

        description_similarity = similarity(
            listing.get("description"),
            other.get("description"),
        )

        if (
            title_similarity >= 0.90
            or
            description_similarity >= 0.90
        ):

            return result(
                "near_duplicate",
                False,
                ["POL-8.3"],
                (
                    f"Very similar to "
                    f"'{other.get('title')}' "
                    f"(title {title_similarity:.0%}, "
                    f"description {description_similarity:.0%})."
                ),
            )

    return result(
        "near_duplicate",
        True,
        ["POL-8.3"],
    )



# ============================================================
# MODEL REVIEW TRIGGERS
#
# NOT failures.
# These should be sent to RAG + LLM.
# ============================================================

def find_model_triggers(listing):

    attrs_text = " ".join(
        get_attributes(listing).values()
    )

    fields = {
        "title": str(
            listing.get("title") or ""
        ),
        "description": str(
            listing.get("description") or ""
        ),
        "attributes": attrs_text,
    }

    triggers = []

    for field, text in fields.items():

        lower_text = text.lower()

        for phrase in MODEL_TRIGGER_PHRASES:

            phrase_lower = phrase.lower()

            # "only X left"
            if phrase_lower == "only x left":

                match = re.search(
                    r"\bonly\s+\d+\s+(left|remaining)\b",
                    lower_text,
                )

            # Normal word/phrase.
            elif (
                phrase_lower[0].isalnum()
                and phrase_lower[-1].isalnum()
            ):

                match = re.search(
                    r"\b"
                    + re.escape(phrase_lower)
                    + r"\b",
                    lower_text,
                )

            # Things like #1 and 100%.
            else:

                match = (
                    phrase_lower
                    in lower_text
                )

            if match:

                triggers.append(
                    {
                        "field": field,
                        "phrase": phrase,
                    }
                )

    return triggers


# ============================================================
# RUN ALL DETERMINISTIC CHECKS
# ============================================================

def run_all_checks(
    listing,
    other_listings=None,
):

    other_listings = other_listings or []

    return [
        check_required_fields(listing),

        check_title_length(listing),

        check_description_length(listing),

        check_price_format(listing),

        check_price_range(listing),

        check_category(listing),

        check_required_attributes(listing),

        check_condition(listing),

        check_tags(listing),

        check_contact_details(listing),

        check_exact_duplicate(
            listing,
            other_listings,
        ),

        check_near_duplicate(
            listing,
            other_listings,
        ),

        check_keyword_stuffing(listing),

        check_caps_and_symbols(listing),
    ]


# ============================================================
# COMPLETE REVIEW
# ============================================================

def review_listing(
    listing,
    other_listings=None,
):
    """
    Returns:

        checks
        failed
        model_triggers
    """

    checks = run_all_checks(
        listing,
        other_listings,
    )

    failed = [
        check
        for check in checks
        if not check["passed"]
        and not check["skipped"]
    ]

    return {
        "checks": checks,
        "failed": failed,
        "model_triggers": find_model_triggers(
            listing
        ),
    }
