"""Eval-only ModelCall wrappers: on-disk response cache and request throttling."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from evals.model_cache import DiskCachedCall, ThrottledCall
from review_insight.domain.enums import Aspect, Route, Sentiment, Urgency
from review_insight.domain.schemas import AnalysisResult, AspectSentiment
from review_insight.llm.analyzer import RawCall
from review_insight.llm.pricing import TokenUsage

RESULT = AnalysisResult(
    aspects=[AspectSentiment(aspect=Aspect.BATTERY, sentiment=Sentiment.POSITIVE, evidence="pin")],
    urgency=Urgency.LOW,
    route_to=Route.NONE,
)
OK = RawCall(
    model="m-001",
    output=RESULT,
    error=None,
    usage=TokenUsage(input_tokens=700, output_tokens=100, cached_input_tokens=50),
    latency_ms=2200.0,
)


def _messages(review: str, system: str = "You analyze reviews.") -> list[BaseMessage]:
    return [SystemMessage(content=system), HumanMessage(content=f"<review>\n{review}\n</review>")]


def _files(root: Path, pattern: str) -> list[Path]:
    # Sync helper: blocking filesystem calls do not belong inside async test bodies.
    return sorted(root.rglob(pattern))


class CountingCall:
    def __init__(self, raw: RawCall = OK) -> None:
        self.raw = raw
        self.calls = 0

    async def __call__(self, messages: Sequence[BaseMessage]) -> RawCall:
        self.calls += 1
        return self.raw


class FailingCall:
    async def __call__(self, messages: Sequence[BaseMessage]) -> RawCall:
        raise TimeoutError("request timed out")


# --- DiskCachedCall ----------------------------------------------------------------------


async def test_a_repeated_call_is_replayed_from_disk(tmp_path: Path) -> None:
    inner = CountingCall()
    cached = DiskCachedCall(inner, cache_dir=tmp_path, model="m")

    first = await cached(_messages("pin trâu"))
    second = await cached(_messages("pin trâu"))

    assert inner.calls == 1  # the API was hit once
    assert first.cached is False
    assert second.cached is True
    # Everything the eval reports is replayed, including the original latency and usage.
    assert (second.output, second.usage, second.latency_ms, second.model) == (
        OK.output,
        OK.usage,
        OK.latency_ms,
        OK.model,
    )


async def test_the_cache_survives_a_new_process(tmp_path: Path) -> None:
    await DiskCachedCall(CountingCall(), cache_dir=tmp_path, model="m")(_messages("pin trâu"))
    inner = CountingCall()

    replay = await DiskCachedCall(inner, cache_dir=tmp_path, model="m")(_messages("pin trâu"))

    assert inner.calls == 0
    assert replay.cached is True


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (("m", _messages("pin trâu")), ("m", _messages("pin yếu"))),  # another review
        (("m", _messages("pin trâu")), ("other", _messages("pin trâu"))),  # another model
        # Any prompt edit misses, even if nobody bumped the prompt version.
        (("m", _messages("pin trâu")), ("m", _messages("pin trâu", system="You rate reviews."))),
    ],
)
async def test_anything_that_changes_the_request_changes_the_key(
    tmp_path: Path,
    first: tuple[str, list[BaseMessage]],
    second: tuple[str, list[BaseMessage]],
) -> None:
    inner = CountingCall()

    for model, messages in (first, second):
        await DiskCachedCall(inner, cache_dir=tmp_path, model=model)(messages)

    assert inner.calls == 2


async def test_schema_failures_are_cached_too(tmp_path: Path) -> None:
    # An invalid response is still the model's answer to that request, and it was billed.
    failed = RawCall(
        model="m", output=None, error="duplicate aspect(s): PERFORMANCE", usage=OK.usage,
        latency_ms=1900.0,
    )  # fmt: skip
    inner = CountingCall(failed)
    cached = DiskCachedCall(inner, cache_dir=tmp_path, model="m")

    await cached(_messages("x"))
    replay = await cached(_messages("x"))

    assert inner.calls == 1
    assert (replay.output, replay.error) == (None, "duplicate aspect(s): PERFORMANCE")


async def test_exceptions_are_not_cached(tmp_path: Path) -> None:
    # A timeout says nothing about the model's answer: the next run must try again.
    with pytest.raises(TimeoutError):
        await DiskCachedCall(FailingCall(), cache_dir=tmp_path, model="m")(_messages("x"))

    assert _files(tmp_path, "*.json") == []


async def test_writes_leave_no_partial_files(tmp_path: Path) -> None:
    await DiskCachedCall(CountingCall(), cache_dir=tmp_path, model="m")(_messages("x"))

    assert len(_files(tmp_path, "*.json")) == 1
    assert _files(tmp_path, "*.part") == []


async def test_a_corrupted_entry_fails_loudly_instead_of_silently_recalling(
    tmp_path: Path,
) -> None:
    # Silently re-calling would spend quota; tell the user which file to delete.
    cached = DiskCachedCall(CountingCall(), cache_dir=tmp_path, model="m")
    await cached(_messages("x"))
    [entry] = _files(tmp_path, "*.json")
    entry.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match=entry.name):
        await cached(_messages("x"))


# --- ThrottledCall -----------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


async def test_calls_are_spaced_by_the_minimum_interval() -> None:
    clock = FakeClock()
    throttled = ThrottledCall(CountingCall(), min_interval_s=5.0, clock=clock, sleep=clock.sleep)

    await throttled(_messages("a"))  # first call: no wait
    clock.now += 1.0  # 1 s later
    await throttled(_messages("b"))  # must wait the remaining 4 s
    clock.now += 10.0  # long pause
    await throttled(_messages("c"))  # no wait

    assert clock.slept == [pytest.approx(4.0)]


async def test_spacing_is_start_to_start_so_slow_calls_do_not_add_waits() -> None:
    # 12 RPM means 12 request *starts* per minute, however long each request takes.
    clock = FakeClock()

    class SlowCall(CountingCall):
        async def __call__(self, messages: Sequence[BaseMessage]) -> RawCall:
            clock.now += 6.0  # the request itself takes 6 s
            return await super().__call__(messages)

    throttled = ThrottledCall(SlowCall(), min_interval_s=5.0, clock=clock, sleep=clock.sleep)

    await throttled(_messages("a"))
    await throttled(_messages("b"))

    assert clock.slept == []
