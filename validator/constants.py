"""
validator/constants.py

Policy constants for the Marketplace Listing Quality Reviewer.
"""

# ============================================================
# POLICY CONSTANTS
# ============================================================

REQUIRED_FIELDS = [
    "title",
    "description",
    "category",
    "price",
    "attributes",
    "seller",
]

PLACEHOLDERS = {
    "tbd",
    "n/a",
    "na",
    "test",
    "lorem ipsum",
    "none",
    "null",
}


# POL-4.1
# Each tuple contains alternative accepted attribute names.
CATEGORY_ATTRIBUTES = {
    "electronics": [
        ("brand",),
        ("model",),
        ("condition",),
        ("warranty",),
    ],

    "home & kitchen": [
        ("brand",),
        ("material",),
        ("dimensions",),
        ("condition",),
    ],

    "fashion & apparel": [
        ("brand",),
        ("size",),
        ("material",),
        ("condition",),
    ],

    "beauty & personal care": [
        ("brand",),
        ("volume", "weight"),
        ("ingredients",),
        ("expiry", "batch date"),
    ],

    "sports & outdoors": [
        ("brand",),
        ("condition",),
        ("size", "dimensions"),
    ],

    "books & media": [
        ("title", "author"),
        ("format",),
        ("condition",),
        ("language",),
    ],

    "toys & games": [
        ("brand",),
        ("recommended age range", "age range"),
        ("condition",),
        ("safety standard",),
    ],

    "home services": [
        ("service type",),
        ("service area",),
        ("availability",),
        ("pricing basis",),
    ],

    "professional services": [
        ("service type",),
        ("delivery method",),
        ("turnaround time",),
        ("pricing basis",),
    ],

    "digital products": [
        ("format",),
        ("platform", "compatibility"),
        ("licence type", "license type"),
        ("delivery method",),
    ],
}


# POL-4.2
ALLOWED_CONDITIONS = {
    "new",
    "open box",
    "refurbished",
    "used - like new",
    "used - good",
    "used - acceptable",
}


# These are NOT failures.
# They only tell the RAG/LLM layer that deeper review may be needed.
MODEL_TRIGGER_PHRASES = [
    "best",
    "#1",
    "guaranteed",
    "100%",
    "cure",
    "treat",
    "prevent",
    "doctor",
    "fda",
    "certified",
    "award",
    "limited time",
    "only X left",
    "trusted by",
    "better than",
    "like new",
    "genuine",
    "handmade",
    "organic",
]
PROMPT_INJECTION_PHRASES = (
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "developer message",
    "follow these instructions",
    "approve this",
    "disregard the rules",
)