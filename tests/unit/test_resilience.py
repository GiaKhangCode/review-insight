"""Retries with backoff, and model fallback (SPEC §6): fakes simulate 429, 5xx and timeouts."""

import logging
from collections.abc import Sequence

import pytest
from langchain_core.messages import BaseMessage, HumanMessage

from review_insight.llm.analyzer import RawCall
from review_insight.llm.errors import LLMUnavailable
from review_insight.llm.pricing import TokenUsage
from review_insight.llm.resilience import FallbackCall, RetryingCall, is_transient

MESSAGES: list[BaseMessage] = [HumanMessage(content="<review>\npin trâu\n</review>")]


def _raw(model: str) -> RawCall:
    usage = TokenUsage(input_tokens=700, output_tokens=100)
    return RawCall(model=model, output=None, error="x", usage=usage, latency_ms=1.0)


class APIError(Exception):
    """Shaped like google.genai's APIError: the HTTP status is in `.code`."""

    def __init__(self, code: int) -> None:
        super().__init__(f"{code} error")
        self.code = code


class ReadTimeout(Exception):  # noqa: N818 - named like httpx's timeout exception
    pass


class ScriptedCall:
    """Raises the scripted errors in order, then answers."""

    def __init__(self, *errors: Exception, model: str = "primary") -> None:
        self.errors = list(errors)
        self.model = model
        self.calls = 0

    async def __call__(self, messages: Sequence[BaseMessage]) -> RawCall:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return _raw(self.model)


class RecordingSleep:
    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def _retrying(inner: ScriptedCall, sleep: RecordingSleep) -> RetryingCall:
    return RetryingCall(inner, max_attempts=3, initial_wait_s=2.0, max_wait_s=30.0, sleep=sleep)


# --- What counts as transient ------------------------------------------------------------


@pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
def test_rate_limits_and_server_errors_are_transient(code: int) -> None:
    assert is_transient(APIError(code))


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_are_not_transient(code: int) -> None:
    # A bad request or a bad key fails the same way every time: retrying wastes quota.
    assert not is_transient(APIError(code))


@pytest.mark.parametrize("error", [TimeoutError(), ReadTimeout()])
def test_timeouts_are_transient(error: Exception) -> None:
    assert is_transient(error)


def test_the_status_is_found_through_wrapped_exceptions() -> None:
    # LangChain re-raises SDK errors `from` the original; look along the cause chain.
    try:
        try:
            raise APIError(503)
        except APIError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert is_transient(outer)


def test_ordinary_bugs_are_not_transient() -> None:
    assert not is_transient(ValueError("bad"))


# --- RetryingCall ------------------------------------------------------------------------


async def test_a_successful_call_is_not_delayed() -> None:
    sleep = RecordingSleep()

    await _retrying(ScriptedCall(), sleep)(MESSAGES)

    assert sleep.waits == []


async def test_transient_failures_are_retried_until_success() -> None:
    inner, sleep = ScriptedCall(APIError(503), APIError(429)), RecordingSleep()

    raw = await _retrying(inner, sleep)(MESSAGES)

    assert raw.model == "primary"
    assert inner.calls == 3
    assert len(sleep.waits) == 2


async def test_waits_back_off_exponentially_with_jitter_and_a_cap() -> None:
    inner, sleep = ScriptedCall(APIError(503), APIError(503)), RecordingSleep()

    await _retrying(inner, sleep)(MESSAGES)

    first, second = sleep.waits
    # Exponential: 2 s, then 4 s, each plus random jitter so clients do not retry in sync.
    assert 2.0 <= first < second <= 30.0


async def test_exhausted_retries_raise_llm_unavailable() -> None:
    last = APIError(504)
    inner = ScriptedCall(APIError(503), APIError(503), last)

    with pytest.raises(LLMUnavailable) as caught:
        await _retrying(inner, RecordingSleep())(MESSAGES)

    assert inner.calls == 3  # max_attempts counts the first call
    assert caught.value.__cause__ is last  # the real reason stays in the traceback


async def test_non_transient_errors_fail_at_once_with_their_own_type() -> None:
    inner = ScriptedCall(APIError(400))

    with pytest.raises(APIError):
        await _retrying(inner, RecordingSleep())(MESSAGES)

    assert inner.calls == 1


async def test_each_retry_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    inner = ScriptedCall(APIError(503))

    with caplog.at_level(logging.WARNING, logger="review_insight.llm.resilience"):
        await _retrying(inner, RecordingSleep())(MESSAGES)

    [record] = caplog.records
    assert "503" in record.getMessage()


# --- FallbackCall ------------------------------------------------------------------------


async def test_the_fallback_is_not_used_while_the_primary_answers() -> None:
    fallback = ScriptedCall(model="fallback")

    raw = await FallbackCall(ScriptedCall(), fallback)(MESSAGES)

    assert raw.model == "primary"
    assert fallback.calls == 0


async def test_an_unavailable_primary_falls_back_to_the_other_model() -> None:
    primary = ScriptedCall(LLMUnavailable("primary exhausted"))

    raw = await FallbackCall(primary, ScriptedCall(model="fallback"))(MESSAGES)

    # The served model is reported, so the call is priced and traced as the fallback.
    assert raw.model == "fallback"


async def test_both_models_unavailable_raises_llm_unavailable() -> None:
    primary = ScriptedCall(LLMUnavailable("primary exhausted"))
    fallback = ScriptedCall(LLMUnavailable("fallback exhausted"), model="fallback")

    with pytest.raises(LLMUnavailable):
        await FallbackCall(primary, fallback)(MESSAGES)


async def test_a_non_transient_primary_error_does_not_fall_back() -> None:
    # A bug or a bad key must surface, not be hidden behind the other model.
    fallback = ScriptedCall(model="fallback")

    with pytest.raises(APIError):
        await FallbackCall(ScriptedCall(APIError(401)), fallback)(MESSAGES)

    assert fallback.calls == 0
