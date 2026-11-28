"""Langfuse tracing (SPEC §6): real SDK, spans captured in memory, nothing leaves the process."""

import json
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import BaseMessage
from langfuse import Langfuse
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from review_insight.config import Settings
from review_insight.domain.enums import Aspect, Route, Sentiment, Urgency
from review_insight.domain.schemas import AnalysisResult, AspectSentiment
from review_insight.llm.analyzer import RawCall, ReviewAnalyzer
from review_insight.llm.errors import LLMUnavailable
from review_insight.llm.pricing import PriceTable, TokenUsage
from review_insight.llm.prompts import load_prompt
from review_insight.observability.langfuse import TracedCall, analyze_traced, build_langfuse

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT = load_prompt(REPO_ROOT / "prompts/analyze_v2.yaml")
PRICES = PriceTable.model_validate(
    {
        "source": "https://example.com/pricing",
        "retrieved": "2026-11-28",
        "currency": "USD",
        "per_tokens": 1_000_000,
        "models": {"m": {"input": 1.0, "cached_input": 0.1, "output": 10.0}},
    }
)
REVIEW = "pin trâu dùng cả ngày"
USAGE = TokenUsage(input_tokens=700, output_tokens=100, cached_input_tokens=200)
GOOD = AnalysisResult(
    aspects=[
        AspectSentiment(aspect=Aspect.BATTERY, sentiment=Sentiment.POSITIVE, evidence="pin trâu")
    ],
    urgency=Urgency.LOW,
    route_to=Route.NONE,
)


def _raw(output: AnalysisResult | None) -> RawCall:
    error = None if output else "duplicate aspect(s): BATTERY"
    return RawCall(model="m", output=output, error=error, usage=USAGE, latency_ms=900.0)


class SequenceCall:
    def __init__(self, *answers: RawCall | Exception) -> None:
        self.answers = list(answers)

    async def __call__(self, messages: Sequence[BaseMessage]) -> RawCall:
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def langfuse(exporter: InMemorySpanExporter) -> Iterator[Langfuse]:
    # Real client; spans go to memory instead of the network (the host is unreachable).
    # The SDK shares one client per public key across the process, so each test uses its
    # own key: otherwise a client built by an earlier test (real exporter) is reused.
    client = Langfuse(
        public_key=f"pk-lf-test-{uuid.uuid4().hex}",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
    )
    yield client
    client.shutdown()


def _analyzer(call: SequenceCall, langfuse: Langfuse | None) -> ReviewAnalyzer:
    traced = call if langfuse is None else TracedCall(call, langfuse=langfuse, prices=PRICES)
    return ReviewAnalyzer(call=traced, prompt=PROMPT, prices=PRICES)


def _spans(langfuse: Langfuse, exporter: InMemorySpanExporter) -> dict[str, list[ReadableSpan]]:
    langfuse.flush()
    by_name: dict[str, list[ReadableSpan]] = {}
    for span in exporter.get_finished_spans():
        by_name.setdefault(span.name, []).append(span)
    return by_name


def _attr(span: ReadableSpan, key: str) -> Any:
    """A Langfuse attribute, decoded from the JSON the SDK stores it as."""
    assert span.attributes is not None
    value = span.attributes[f"langfuse.observation.{key}"]
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)  # metadata, input, output, usage and cost are JSON text
    except json.JSONDecodeError:
        return value  # plain strings: type, model name, level, status message


# --- Client factory ----------------------------------------------------------------------


def test_no_keys_means_no_client() -> None:
    assert build_langfuse(Settings(_env_file=None)) is None


def test_keys_build_a_client_without_contacting_the_server() -> None:
    settings = Settings(
        _env_file=None,
        langfuse_public_key=f"pk-lf-build-{uuid.uuid4().hex}",
        langfuse_secret_key="sk-lf-test",
        langfuse_host="http://127.0.0.1:9",
    )

    client = build_langfuse(settings)

    assert isinstance(client, Langfuse)
    client.shutdown()


# --- One trace per review, one generation per model call ---------------------------------


async def test_a_review_becomes_one_trace_with_one_generation(
    langfuse: Langfuse, exporter: InMemorySpanExporter
) -> None:
    analysis, trace_id = await analyze_traced(
        _analyzer(SequenceCall(_raw(GOOD)), langfuse), REVIEW, langfuse=langfuse
    )

    spans = _spans(langfuse, exporter)
    [root], [generation] = spans["analyze_review"], spans["llm_call"]
    assert generation.parent is not None
    assert generation.parent.span_id == root.context.span_id
    assert trace_id == format(root.context.trace_id, "032x")  # returned for X-Trace-ID (T11)

    assert _attr(root, "input") == {"review": REVIEW}
    assert _attr(root, "output") == GOOD.model_dump(mode="json")
    assert _attr(root, "metadata.prompt") == "analyze_review@v2"
    assert _attr(root, "metadata.attempts") == 1
    assert analysis.result == GOOD


async def test_the_generation_carries_model_usage_and_our_cost(
    langfuse: Langfuse, exporter: InMemorySpanExporter
) -> None:
    await analyze_traced(_analyzer(SequenceCall(_raw(GOOD)), langfuse), REVIEW, langfuse=langfuse)

    [generation] = _spans(langfuse, exporter)["llm_call"]
    assert _attr(generation, "type") == "generation"
    assert _attr(generation, "model.name") == "m"
    # Non-overlapping buckets, so Langfuse's total (their sum) equals the real token count.
    assert _attr(generation, "usage_details") == {
        "input": 500,
        "cache_read_input_tokens": 200,
        "output": 100,
    }
    # Our price table is the source of truth, not Langfuse's built-in model prices.
    assert _attr(generation, "cost_details") == {
        "total": pytest.approx((500 * 1.0 + 200 * 0.1 + 100 * 10.0) / 1e6)
    }


async def test_a_repair_shows_as_two_generations_under_one_trace(
    langfuse: Langfuse, exporter: InMemorySpanExporter
) -> None:
    call = SequenceCall(_raw(None), _raw(GOOD))

    await analyze_traced(_analyzer(call, langfuse), REVIEW, langfuse=langfuse)

    spans = _spans(langfuse, exporter)
    [root], generations = spans["analyze_review"], spans["llm_call"]
    assert len(generations) == 2
    assert _attr(generations[0], "level") == "WARNING"  # the rejected first answer
    assert _attr(root, "metadata.attempts") == 2
    assert _attr(root, "metadata.first_failure") == "schema"


async def test_a_rejected_review_is_flagged_on_the_trace(
    langfuse: Langfuse, exporter: InMemorySpanExporter
) -> None:
    call = SequenceCall(_raw(None), _raw(None))

    await analyze_traced(_analyzer(call, langfuse), REVIEW, langfuse=langfuse)

    [root] = _spans(langfuse, exporter)["analyze_review"]
    assert _attr(root, "level") == "WARNING"
    assert _attr(root, "metadata.failure") == "schema"


async def test_an_unavailable_model_is_recorded_as_an_error_and_still_raised(
    langfuse: Langfuse, exporter: InMemorySpanExporter
) -> None:
    call = SequenceCall(LLMUnavailable("503 after 3 attempts"))

    with pytest.raises(LLMUnavailable):
        await analyze_traced(_analyzer(call, langfuse), REVIEW, langfuse=langfuse)

    spans = _spans(langfuse, exporter)
    [root], [generation] = spans["analyze_review"], spans["llm_call"]
    assert _attr(generation, "level") == "ERROR"
    assert _attr(root, "level") == "ERROR"
    assert "503" in _attr(root, "status_message")


async def test_without_langfuse_analysis_works_and_nothing_is_traced(
    exporter: InMemorySpanExporter,
) -> None:
    analysis, trace_id = await analyze_traced(
        _analyzer(SequenceCall(_raw(GOOD)), None), REVIEW, langfuse=None
    )

    assert analysis.result == GOOD
    assert trace_id is None
    assert exporter.get_finished_spans() == ()
