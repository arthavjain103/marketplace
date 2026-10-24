"""Focused tests: LLM JSON parsing and review grounding (no real Groq calls).

Mocks langchain_groq.ChatGroq.invoke so no real Groq API calls are made.
"""


import pytest

import llm.chat as chat


class _FakeMessage:
    def __init__(self, content):
        self.content = content


@pytest.fixture(autouse=True)
def groq(monkeypatch):
    """Patch ChatGroq.invoke to return canned responses (FIFO).

    Also binds a truthy GROQ_API_KEY so _get_llm does not raise, without
    ever hitting the network.
    """
    responses = []

    def fake_invoke(self, messages):
        return _FakeMessage(responses.pop(0) if responses else "{}")

    monkeypatch.setattr(chat.ChatGroq, "invoke", fake_invoke)
    monkeypatch.setattr(chat, "GROQ_API_KEY", "test-key")
    chat._llm_cache.clear()
    return responses


# ---------------------------------------------------------------------------
# _parse_json
# ---------------------------------------------------------------------------

def test_parse_json_plain_object():
    assert chat._parse_json('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_parse_json_strips_markdown_code_fences():
    text = "```json\n{\"a\": 1}\n```"
    assert chat._parse_json(text) == {"a": 1}


def test_parse_json_repairs_trailing_comma_before_brace():
    assert chat._parse_json('{"a": 1,}') == {"a": 1}


def test_parse_json_repairs_trailing_comma_before_bracket():
    assert chat._parse_json('{"a": [1, 2,],}') == {"a": [1, 2]}


def test_parse_json_extracts_first_balanced_block():
    text = "Here is the result:\n{\"findings\": []}\nAnd nothing else."
    assert chat._parse_json(text) == {"findings": []}


def test_parse_json_invalid_raises_value_error():
    with pytest.raises(ValueError):
        chat._parse_json("not json at all")


def test_parse_json_empty_string_raises_value_error():
    with pytest.raises(ValueError):
        chat._parse_json("")


# ---------------------------------------------------------------------------
# validate_review rule-id grounding
# ---------------------------------------------------------------------------

VALID = {"POL-6.2", "POL-1.3"}


def _finding(rule_ids, **overrides):
    base = {
        "field": "title",
        "quote": "guaranteed",
        "issue_type": "prohibited",
        "severity": "Major",
        "explanation": "claim",
        "rule_ids": rule_ids,
        "suggested_wording": "reword",
    }
    base.update(overrides)
    return base


def test_validate_review_keeps_grounded_findings():
    raw = {
        "findings": [_finding(["POL-6.2"])],
        "assumptions": [],
        "style_suggestions": [],
        "revised_listing": {},
    }
    out = chat.validate_review(raw, VALID)
    assert len(out["findings"]) == 1
    assert out["findings"][0]["rule_ids"] == ["POL-6.2"]
    assert out["style_suggestions"] == []


def test_validate_review_ungrounded_moved_to_style_suggestions():
    raw = {
        "findings": [_finding(["POL-999"])],  # not in valid set
        "assumptions": [],
        "style_suggestions": [],
        "revised_listing": {},
    }
    out = chat.validate_review(raw, VALID)
    assert out["findings"] == []
    assert any("title" in s and "claim" in s and "reword" in s
               for s in out["style_suggestions"])


def test_validate_review_mixed_grounded_and_ungrounded():
    raw = {
        "findings": [
            _finding(["POL-6.2"]),
            _finding(["POL-999"]),
        ],
        "assumptions": [],
        "style_suggestions": [],
        "revised_listing": {},
    }
    out = chat.validate_review(raw, VALID)
    assert len(out["findings"]) == 1
    assert len(out["style_suggestions"]) == 1


def test_validate_review_normalizes_enums():
    raw = {
        "findings": [_finding(
            ["POL-6.2"],
            field="UnknownField",
            issue_type="not-a-real-type",
            severity="critical",
        )],
        "assumptions": [],
        "style_suggestions": [],
        "revised_listing": {},
    }
    out = chat.validate_review(raw, VALID)
    f = out["findings"][0]
    assert f["field"] == "other"
    assert f["issue_type"] == "unclear"
    assert f["severity"] == "Critical"


def test_validate_review_sorts_by_severity():
    raw = {
        "findings": [
            _finding(["POL-6.2"], severity="Minor"),
            _finding(["POL-6.2"], severity="Critical"),
            _finding(["POL-6.2"], severity="Major"),
            _finding(["POL-6.2"], severity="Info"),
        ],
        "assumptions": [],
        "style_suggestions": [],
        "revised_listing": {},
    }
    out = chat.validate_review(raw, VALID)
    assert [f["severity"] for f in out["findings"]] == [
        "Critical", "Major", "Minor", "Info"]


def test_validate_review_revised_listing_restricted_to_allowed_fields():
    raw = {
        "findings": [],
        "assumptions": [],
        "style_suggestions": [],
        "revised_listing": {
            "title": "New",
            "description": "D",
            "tags": ["t"],
            "price": "9.99",   # not allowed
            "category": "X",   # not allowed
        },
    }
    out = chat.validate_review(raw, set())
    assert out["revised_listing"] == {"title": "New", "description": "D", "tags": ["t"]}


def test_validate_review_empty_input():
    out = chat.validate_review({}, VALID)
    assert out["findings"] == []
    assert out["style_suggestions"] == []
    assert out["revised_listing"] == {}


def test_validate_review_non_dict_raises():
    with pytest.raises(ValueError):
        chat.validate_review("not a dict", VALID)
