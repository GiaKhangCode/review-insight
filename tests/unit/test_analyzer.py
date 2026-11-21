"""LLM review analyzer (SPEC §6, §10): no network, the model call is faked."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from review_insight.domain.enums import Aspect, Route, Sentiment, Urgency
from review_insight.domain.schemas import AnalysisResult, AspectSentiment
from review_insight.llm.analyzer import (
    LangChainModelCall,
    RawCall,
    ReviewAnalyzer,
    build_gemini_call,
)
from review_insight.llm.pricing import PriceTable, TokenUsage, UnknownModelError
from review_insight.llm.prompts import load_prompt

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT = load_prompt(REPO_ROOT / "prompts/analyze_v1.yaml")
PRICES = PriceTable.model_validate(
    {
        "source": "https://example.com/pricing",
        "retrieved": "2026-11-21",
        "currency": "USD",
        "per_tokens": 1_000_000,
        "models": {"fake-model": {"input": 1.0, "cached_input": 0.1, "output": 10.0}},
    }
)
REVIEW = "pin trau xai ca ngay, nhung shop giao hang cham qua"
USAGE = TokenUsage(input_tokens=700, output_tokens=100)


def _result(*evidence: tuple[Aspect, Sentiment, str]) -> AnalysisResult:
    return AnalysisResult(
        aspects=[AspectSentiment(aspect=a, sentiment=s, evidence=e) for a, s, e in evidence],
        urgency=Urgency.MEDIUM,
        route_to=Route.CUSTOMER_SERVICE,
    )


GROUNDED = _result(
    (Aspect.BATTERY, Sentiment.POSITIVE, "pin trau xai ca ngay"),
    (Aspect.SER_ACC, Sentiment.NEGATIVE, "shop giao hang cham qua"),
)


class FakeCall:
    """Stands in for the model: returns a canned RawCall and records what it was sent."""

    def __init__(self, raw: RawCall) -> None:
        self.raw = raw
        self.sent: list[Sequence[BaseMessage]] = []

    async def __call__(self, messages: Sequence[BaseMessage]) -> RawCall:
        self.sent.append(messages)
        return self.raw


def _raw(output: AnalysisResult | None, *, error: str | None = None) -> RawCall:
    return RawCall(model="fake-model", output=output, error=error, usage=USAGE, latency_ms=850.0)


# --- ReviewAnalyzer ----------------------------------------------------------------------


async def test_a_valid_grounded_output_becomes_the_result() -> None:
    call = FakeCall(_raw(GROUNDED))
    analyzer = ReviewAnalyzer(call=call, prompt=PROMPT, prices=PRICES)

    analysis = await analyzer.analyze(REVIEW)

    assert analysis.result == GROUNDED
    assert analysis.failure is None
    assert (analysis.model, analysis.prompt_ref) == ("fake-model", "analyze_review@v1")
    assert analysis.latency_ms == 850.0
    assert analysis.cost_usd == pytest.approx((700 * 1.0 + 100 * 10.0) / 1e6)
    assert (analysis.evidence_total, analysis.evidence_ungrounded) == (2, 0)


async def test_the_review_is_sent_inside_the_rendered_prompt() -> None:
    call = FakeCall(_raw(GROUNDED))

    await ReviewAnalyzer(call=call, prompt=PROMPT, prices=PRICES).analyze(REVIEW)

    [messages] = call.sent
    assert [m.type for m in messages] == ["system", "human"]
    assert f"<review>\n{REVIEW}\n</review>" in str(messages[1].content)


async def test_a_schema_failure_is_reported_not_raised_and_still_costs_money() -> None:
    call = FakeCall(_raw(None, error="1 validation error: urgency 'urgent' is not valid"))

    analysis = await ReviewAnalyzer(call=call, prompt=PROMPT, prices=PRICES).analyze(REVIEW)

    assert analysis.result is None
    assert analysis.failure == "schema"
    assert analysis.error is not None
    assert "urgency" in analysis.error
    assert analysis.cost_usd > 0  # a failed call is still billed


async def test_ungrounded_evidence_rejects_the_result() -> None:
    paraphrased = _result(
        (Aspect.BATTERY, Sentiment.POSITIVE, "pin trau xai ca ngay"),
        (Aspect.SER_ACC, Sentiment.NEGATIVE, "giao hàng chậm"),  # "corrected", not copied
    )
    call = FakeCall(_raw(paraphrased))

    analysis = await ReviewAnalyzer(call=call, prompt=PROMPT, prices=PRICES).analyze(REVIEW)

    assert analysis.result is None
    assert analysis.failure == "ungrounded"
    assert analysis.error is not None
    assert "giao hàng chậm" in analysis.error
    assert (analysis.evidence_total, analysis.evidence_ungrounded) == (2, 1)


async def test_a_review_with_no_aspect_is_a_valid_result() -> None:
    empty = AnalysisResult(aspects=[], urgency=Urgency.LOW, route_to=Route.NONE)
    call = FakeCall(_raw(empty))

    analysis = await ReviewAnalyzer(call=call, prompt=PROMPT, prices=PRICES).analyze("có sạc ko?")

    assert analysis.result == empty
    assert analysis.evidence_total == 0


async def test_a_model_without_a_price_is_a_configuration_error() -> None:
    raw = RawCall(model="other-model", output=GROUNDED, error=None, usage=USAGE, latency_ms=1.0)

    with pytest.raises(UnknownModelError):
        await ReviewAnalyzer(call=FakeCall(raw), prompt=PROMPT, prices=PRICES).analyze(REVIEW)


# --- LangChainModelCall: reading the structured-output response --------------------------


def _structured(parsed: object, *, error: Exception | None = None, **ai: Any) -> RunnableLambda:
    """Mimics `llm.with_structured_output(AnalysisResult, include_raw=True)`."""
    message = AIMessage(content="", **ai)
    return RunnableLambda(lambda _: {"raw": message, "parsed": parsed, "parsing_error": error})


async def test_usage_and_served_model_are_read_from_the_raw_message() -> None:
    runnable = _structured(
        GROUNDED,
        usage_metadata={
            "input_tokens": 697,
            "output_tokens": 111,  # LangChain already adds thinking tokens here
            "total_tokens": 808,
            "input_token_details": {"cache_read": 512},
        },
        response_metadata={"model_name": "gemini-3.1-flash-lite-001"},
    )

    raw = await LangChainModelCall(runnable, model="gemini-3.1-flash-lite")(PROMPT.render(REVIEW))

    assert raw.output == GROUNDED
    assert raw.error is None
    assert raw.usage == TokenUsage(input_tokens=697, output_tokens=111, cached_input_tokens=512)
    assert raw.model == "gemini-3.1-flash-lite-001"  # what actually served the call
    assert raw.latency_ms >= 0


async def test_a_parsing_error_is_captured_as_text() -> None:
    runnable = _structured(
        None,
        error=ValueError("duplicate aspect(s): CAMERA"),
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )

    raw = await LangChainModelCall(runnable, model="m")(PROMPT.render(REVIEW))

    assert raw.output is None
    assert raw.error is not None
    assert "duplicate aspect" in raw.error
    assert raw.model == "m"  # no model_name in metadata: fall back to the configured one


async def test_an_empty_response_is_a_schema_failure() -> None:
    runnable = _structured(
        None, usage_metadata={"input_tokens": 10, "output_tokens": 0, "total_tokens": 10}
    )

    raw = await LangChainModelCall(runnable, model="m")(PROMPT.render(REVIEW))

    assert raw.output is None
    assert raw.error == "the model returned no structured output"


async def test_missing_usage_fails_loudly() -> None:
    # Without token counts the call cannot be costed; guessing 0 would under-report spend.
    runnable = _structured(GROUNDED)

    with pytest.raises(ValueError, match="usage"):
        await LangChainModelCall(runnable, model="m")(PROMPT.render(REVIEW))


# --- Factory -----------------------------------------------------------------------------


def test_build_gemini_call_needs_no_network() -> None:
    call = build_gemini_call(model="gemini-3.1-flash-lite", api_key=SecretStr("x"), timeout_s=15)

    assert isinstance(call, LangChainModelCall)
    assert call.model == "gemini-3.1-flash-lite"
