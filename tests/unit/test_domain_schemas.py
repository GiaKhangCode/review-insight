"""Domain vocabulary and the AnalysisResult contract (SPEC §4, §5)."""

import pytest
from pydantic import ValidationError

from review_insight.domain.enums import Aspect, Route, Sentiment, Urgency
from review_insight.domain.schemas import AnalysisResult, AspectSentiment


def test_aspect_vocabulary_matches_the_visfd_label_set() -> None:
    # Values must equal the dataset's labels exactly, so parsing gold labels is a lookup.
    assert {a.value for a in Aspect} == {
        "SCREEN", "CAMERA", "FEATURES", "BATTERY", "PERFORMANCE",
        "STORAGE", "DESIGN", "PRICE", "GENERAL", "SER&ACC",
    }  # fmt: skip


def test_label_vocabularies_are_closed_sets() -> None:
    assert {s.value for s in Sentiment} == {"positive", "negative", "neutral"}
    assert {u.value for u in Urgency} == {"low", "medium", "high"}
    assert {r.value for r in Route} == {"product", "customer_service", "sales_pricing", "none"}


def test_enums_compare_equal_to_their_string_values() -> None:
    # StrEnum: JSON from the LLM / DB rows ("negative") compare directly with members.
    assert Sentiment.NEGATIVE == "negative"
    assert Aspect("SER&ACC") is Aspect.SER_ACC


def test_valid_result_parses_from_llm_style_json() -> None:
    result = AnalysisResult.model_validate(
        {
            "aspects": [
                {"aspect": "BATTERY", "sentiment": "positive", "evidence": "pin trâu"},
                {"aspect": "SER&ACC", "sentiment": "negative", "evidence": "shop giao chậm"},
            ],
            "urgency": "medium",
            "route_to": "customer_service",
        }
    )

    assert result.aspects[0].aspect is Aspect.BATTERY
    assert result.route_to is Route.CUSTOMER_SERVICE


def test_no_aspect_mentioned_is_a_valid_empty_list() -> None:
    # e.g. a pure question ("có tai nghe không?"); in the dataset: a row labelled only {OTHERS}.
    result = AnalysisResult(aspects=[], urgency=Urgency.LOW, route_to=Route.NONE)

    assert result.aspects == []


@pytest.mark.parametrize(
    "bad_item",
    [
        {"aspect": "WIFI", "sentiment": "positive", "evidence": "wifi mạnh"},  # unknown aspect
        {"aspect": "CAMERA", "sentiment": "great", "evidence": "camera đẹp"},  # unknown sentiment
        {"aspect": "CAMERA", "sentiment": "positive", "evidence": "   "},  # blank evidence
    ],
)
def test_out_of_vocabulary_or_blank_values_are_rejected(bad_item: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        AspectSentiment.model_validate(bad_item)


def test_same_aspect_twice_is_rejected() -> None:
    # One sentiment per aspect per review, as in the dataset's annotation scheme.
    with pytest.raises(ValidationError, match="duplicate aspect"):
        AnalysisResult(
            aspects=[
                AspectSentiment(aspect=Aspect.CAMERA, sentiment=Sentiment.POSITIVE, evidence="a"),
                AspectSentiment(aspect=Aspect.CAMERA, sentiment=Sentiment.NEGATIVE, evidence="b"),
            ],
            urgency=Urgency.LOW,
            route_to=Route.NONE,
        )


def test_unknown_fields_are_rejected() -> None:
    # An LLM inventing extra keys is a contract violation, not something to silently drop.
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(
            {"aspects": [], "urgency": "low", "route_to": "none", "confidence": 0.9}
        )
