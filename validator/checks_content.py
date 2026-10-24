"""validator/checks_content.py - listing content deterministic checks."""

import re

from validator.constants import (
    ALLOWED_CONDITIONS,
    CATEGORY_ATTRIBUTES,
    REQUIRED_FIELDS,
)
from validator.helpers import (
    EMAIL_RE,
    EMOJI_RE,
    HANDLE_RE,
    REPEATED_PUNCT_RE,
    URL_RE,
    _has_phone,
    clean_text,
    get_attributes,
    is_missing,
    result,
)
# ============================================================
# 1. REQUIRED FIELDS
# POL-2.1, POL-2.3
# ============================================================

def check_required_fields(listing):
    missing = []

    for field in REQUIRED_FIELDS:

        if field == "attributes":
            value = get_attributes(listing)
        else:
            value = listing.get(field)

        if is_missing(value):
            missing.append(field)

    if missing:
        return result(
            "required_fields",
            False,
            ["POL-2.1", "POL-2.3"],
            "Missing or placeholder fields: " + ", ".join(missing),
        )

    return result(
        "required_fields",
        True,
        ["POL-2.1", "POL-2.3"],
    )


# ============================================================
# 2. TITLE LENGTH
# POL-3.1
# ============================================================

def check_title_length(listing):
    title = clean_text(listing.get("title"))
    length = len(title)

    passed = 15 <= length <= 80

    return result(
        "title_length",
        passed,
        ["POL-3.1"],
        ""
        if passed
        else f"Title has {length} characters; it must be 15 to 80.",
    )


# ============================================================
# 3. DESCRIPTION LENGTH
# POL-3.2
# ============================================================

def check_description_length(listing):
    description = clean_text(listing.get("description"))
    length = len(description)

    passed = 80 <= length <= 2000

    return result(
        "description_length",
        passed,
        ["POL-3.2"],
        ""
        if passed
        else (
            f"Description has {length} characters; "
            "it must be 80 to 2,000."
        ),
    )



# ============================================================
# 6. SUPPORTED CATEGORY
# POL-3.5, POL-4.1
# ============================================================

def check_category(listing):
    category = clean_text(
        listing.get("category")
    ).lower()

    passed = category in CATEGORY_ATTRIBUTES

    return result(
        "category_supported",
        passed,
        ["POL-3.5", "POL-4.1"],
        ""
        if passed
        else (
            f"Category '{listing.get('category')}' "
            "is not supported."
        ),
    )


# ============================================================
# 7. REQUIRED CATEGORY ATTRIBUTES
# POL-2.4, POL-4.1
# ============================================================

def check_required_attributes(listing):
    category = clean_text(
        listing.get("category")
    ).lower()

    # Don't create a second failure if category itself
    # is invalid.
    if category not in CATEGORY_ATTRIBUTES:
        return result(
            "required_attributes",
            True,
            ["POL-2.4", "POL-4.1"],
            "Skipped because category is not supported.",
            skipped=True,
        )

    attrs = get_attributes(listing)

    missing = []

    for options in CATEGORY_ATTRIBUTES[category]:

        found = any(
            name in attrs and not is_missing(attrs[name])
            for name in options
        )

        if not found:
            missing.append(
                " or ".join(options)
            )

    if missing:
        return result(
            "required_attributes",
            False,
            ["POL-2.4", "POL-4.1"],
            (
                "Missing required attributes: "
                + ", ".join(missing)
            ),
        )

    return result(
        "required_attributes",
        True,
        ["POL-2.4", "POL-4.1"],
    )


# ============================================================
# 8. CONDITION
# POL-4.2
# ============================================================

def check_condition(listing):
    attrs = get_attributes(listing)
    condition = attrs.get("condition")

    # Missing condition is handled by required_attributes
    # when the category requires it.
    if condition is None or not clean_text(condition):
        return result(
            "condition_value",
            True,
            ["POL-4.2"],
            "Skipped because no condition value was provided.",
            skipped=True,
        )

    normalized = clean_text(condition).lower()

    passed = normalized in ALLOWED_CONDITIONS

    return result(
        "condition_value",
        passed,
        ["POL-4.2"],
        ""
        if passed
        else (
            f"Condition '{condition}' is not one of "
            "the allowed values."
        ),
    )


# ============================================================
# 9. TAGS
# POL-3.6
# ============================================================

def check_tags(listing):
    tags = listing.get("tags") or []

    if isinstance(tags, str):
        tags = tags.split(",")

    tags = [
        clean_text(tag)
        for tag in tags
        if clean_text(tag)
    ]

    problems = []

    if len(tags) > 10:
        problems.append(
            f"{len(tags)} tags provided; maximum is 10"
        )

    for tag in tags:
        length = len(tag)

        if not 2 <= length <= 30:
            problems.append(
                f"tag '{tag}' has {length} characters; "
                "must be 2 to 30"
            )

    return result(
        "tags",
        not problems,
        ["POL-3.6"],
        "; ".join(problems),
    )


# ============================================================
# 10. CONTACT DETAILS
# POL-9.2
# ============================================================

def check_contact_details(listing):

    attrs_text = " ".join(
        f"{key} {value}"
        for key, value in get_attributes(listing).items()
    )

    tags = listing.get("tags") or []

    if isinstance(tags, str):
        tags_text = tags
    else:
        tags_text = " ".join(
            map(str, tags)
        )

    fields = {
        "title": str(
            listing.get("title") or ""
        ),
        "description": str(
            listing.get("description") or ""
        ),
        "attributes": attrs_text,
        "tags": tags_text,
        "seller": str(
            listing.get("seller") or ""
        ),
    }

    found = []

    for field, text in fields.items():

        if EMAIL_RE.search(text):
            found.append(
                f"email in {field}"
            )

        if URL_RE.search(text):
            found.append(
                f"web address in {field}"
            )

        if _has_phone(text):
            found.append(
                f"phone number in {field}"
            )

        if HANDLE_RE.search(text):
            found.append(
                f"social handle in {field}"
            )

    return result(
        "contact_details",
        not found,
        ["POL-9.2"],
        ""
        if not found
        else "Found: " + ", ".join(found),
    )
    

# ============================================================
# 13. KEYWORD STUFFING
# POL-8.6
# ============================================================

def check_keyword_stuffing(listing):

    title = clean_text(
        listing.get("title")
    ).lower()

    words = re.findall(
        r"[a-z0-9]+",
        title,
    )

    counts = {}

    for word in words:

        # The deterministic appendix specifically checks
        # repeated words 3+ times in title.
        counts[word] = counts.get(word, 0) + 1

    repeated = [
        word
        for word, count in counts.items()
        if count >= 3
    ]

    return result(
        "keyword_stuffing",
        not repeated,
        ["POL-8.6"],
        ""
        if not repeated
        else (
            "Word repeated 3+ times in title: "
            + ", ".join(repeated)
        ),
    )


# ============================================================
# 14. ALL CAPS / SYMBOLS
# BRD-2.2, BRD-2.3
# ============================================================

def check_caps_and_symbols(listing):

    title = clean_text(
        listing.get("title")
    )

    problems = []

    letters = [
        char
        for char in title
        if char.isalpha()
    ]

    # Entire title in uppercase.
    if (
        len(letters) >= 5
        and title == title.upper()
    ):
        problems.append(
            "title is entirely in capital letters"
        )

    # Emoji/decorative symbols.
    if EMOJI_RE.search(title):
        problems.append(
            "title contains emoji or decorative symbols"
        )

    # Repeated punctuation.
    if REPEATED_PUNCT_RE.search(title):
        problems.append(
            "title has repeated punctuation"
        )

    return result(
        "caps_and_symbols",
        not problems,
        ["BRD-2.2", "BRD-2.3"],
        "; ".join(problems),
    )


