"""Evidence grounding: every evidence span must really occur in the review (anti-hallucination)."""

import unicodedata

import pytest

from review_insight.domain.enums import Aspect, Route, Sentiment, Urgency
from review_insight.domain.grounding import (
    UngroundedEvidenceError,
    normalize_text,
    validate_grounded,
)
from review_insight.domain.schemas import AnalysisResult, AspectSentiment

REVIEW = "Máy đẹp, pin trâu nhưng camera chụp đêm hơi tệ.\nShop giao chậm"


def _result(*spans: tuple[Aspect, str]) -> AnalysisResult:
    return AnalysisResult(
        aspects=[
            AspectSentiment(aspect=aspect, sentiment=Sentiment.NEGATIVE, evidence=evidence)
            for aspect, evidence in spans
        ],
        urgency=Urgency.MEDIUM,
        route_to=Route.PRODUCT,
    )


# --- normalize_text ---------------------------------------------------------


def test_normalize_text_composes_unicode_and_collapses_whitespace() -> None:
    decomposed = unicodedata.normalize("NFD", "  pin   trâu\n\tquá  ")

    assert normalize_text(decomposed) == "pin trâu quá"
    assert normalize_text(decomposed) == unicodedata.normalize("NFC", "pin trâu quá")


def test_normalize_text_is_idempotent() -> None:
    once = normalize_text(REVIEW)

    assert normalize_text(once) == once


# --- validate_grounded: accepted --------------------------------------------


def test_verbatim_spans_are_grounded() -> None:
    result = _result((Aspect.BATTERY, "pin trâu"), (Aspect.SER_ACC, "Shop giao chậm"))

    assert validate_grounded(result, source_text=REVIEW) is result


def test_no_aspects_is_trivially_grounded() -> None:
    result = AnalysisResult(aspects=[], urgency=Urgency.LOW, route_to=Route.NONE)

    assert validate_grounded(result, source_text=REVIEW) is result


@pytest.mark.parametrize(
    ("source", "evidence"),
    [
        # Same Vietnamese text, different Unicode encodings (NFD vs NFC): visually identical.
        (unicodedata.normalize("NFD", REVIEW), unicodedata.normalize("NFC", "pin trâu")),
        (unicodedata.normalize("NFC", REVIEW), unicodedata.normalize("NFD", "pin trâu")),
        # Span crosses a line break in the review; the model returns a single space.
        (REVIEW, "hơi tệ. Shop giao"),
        # Model changed only the casing.
        (REVIEW, "máy đẹp"),
    ],
    ids=["nfd-source", "nfd-evidence", "newline-vs-space", "casing"],
)
def test_harmless_formatting_differences_are_tolerated(source: str, evidence: str) -> None:
    result = _result((Aspect.BATTERY, evidence))

    assert validate_grounded(result, source_text=source) is result


# --- validate_grounded: rejected --------------------------------------------


def test_paraphrased_evidence_is_rejected_with_actionable_message() -> None:
    # "pin tốt" means the same thing, but it is NOT what the customer wrote.
    result = _result((Aspect.BATTERY, "pin trâu"), (Aspect.CAMERA, "camera chụp tối"))

    with pytest.raises(UngroundedEvidenceError) as exc_info:
        validate_grounded(result, source_text=REVIEW)

    error = exc_info.value
    assert [item.evidence for item in error.ungrounded] == ["camera chụp tối"]
    assert "CAMERA" in str(error)
    assert "camera chụp tối" in str(error)
    assert "verbatim" in str(error)  # tells the LLM how to fix it in the repair retry


def test_all_ungrounded_spans_are_reported_at_once() -> None:
    result = _result((Aspect.BATTERY, "pin tốt"), (Aspect.PRICE, "giá rẻ"))

    with pytest.raises(UngroundedEvidenceError) as exc_info:
        validate_grounded(result, source_text=REVIEW)

    assert {item.aspect for item in exc_info.value.ungrounded} == {Aspect.BATTERY, Aspect.PRICE}


def test_ungrounded_error_is_a_value_error() -> None:
    # Lets callers that only know about ValueError (e.g. Pydantic, retry logic) handle it.
    assert issubclass(UngroundedEvidenceError, ValueError)
