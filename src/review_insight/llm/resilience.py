"""Resilience wrappers around a `ModelCall`: retries with backoff, and model fallback.

Production composition (API, T11):  FallbackCall(RetryingCall(primary), RetryingCall(fallback))
Eval composition:                   DiskCachedCall(RetryingCall(ThrottledCall(model)))
Evals never fall back: a run measures one model, and answers from another would mix results.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence

from langchain_core.messages import BaseMessage
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from review_insight.llm.analyzer import ModelCall, RawCall
from review_insight.llm.errors import LLMUnavailable

logger = logging.getLogger(__name__)

# 429: rate limited. 5xx: the provider is overloaded or failing. Everything else (400, 401,
# 403, 404) fails the same way on every attempt, so retrying would only burn quota.
_TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})
_TIMEOUT_NAMES = ("Timeout", "TimeoutError", "TimeoutException")  # httpx, httpcore, ...


def is_transient(exc: BaseException) -> bool:
    """True if trying again later could succeed. Follows the `raise ... from` cause chain."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:  # `seen` guards against cycles
        seen.add(id(current))
        if isinstance(current, TimeoutError) or type(current).__name__.endswith(_TIMEOUT_NAMES):
            return True
        code = getattr(current, "code", None)  # google.genai APIError carries the HTTP status
        if isinstance(code, int) and code in _TRANSIENT_STATUS:
            return True
        current = current.__cause__ or current.__context__
    return False


class RetryingCall:
    """Retries transient failures with exponential backoff and jitter.

    Gives up after `max_attempts` calls in total and raises `LLMUnavailable` from the last
    error. Non-transient errors are raised at once, unchanged.
    """

    def __init__(
        self,
        inner: ModelCall,
        *,
        max_attempts: int = 3,
        initial_wait_s: float = 2.0,
        max_wait_s: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._inner = inner
        self._max_attempts = max_attempts
        self._initial_wait_s = initial_wait_s
        self._max_wait_s = max_wait_s
        self._sleep = sleep

    async def __call__(self, messages: Sequence[BaseMessage]) -> RawCall:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            # initial * 2^n, plus up to 1 s of random jitter, capped at max_wait_s.
            wait=wait_exponential_jitter(initial=self._initial_wait_s, max=self._max_wait_s),
            retry=retry_if_exception(is_transient),
            before_sleep=self._log_retry,
            sleep=self._sleep,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    return await self._inner(messages)
        except RetryError as exc:
            last = exc.last_attempt.exception()
            raise LLMUnavailable(
                f"no answer after {self._max_attempts} attempts; last error: {last}"
            ) from last
        raise AssertionError("unreachable: tenacity either returns or raises")

    def _log_retry(self, state: RetryCallState) -> None:
        error = state.outcome.exception() if state.outcome else None
        wait = state.next_action.sleep if state.next_action else 0.0
        logger.warning(
            "LLM call failed (%s); retry %d of %d in %.1f s",
            error,
            state.attempt_number,
            self._max_attempts - 1,
            wait,
        )


class FallbackCall:
    """Uses the fallback model only when the primary is unavailable (retries exhausted)."""

    def __init__(self, primary: ModelCall, fallback: ModelCall) -> None:
        self._primary = primary
        self._fallback = fallback

    async def __call__(self, messages: Sequence[BaseMessage]) -> RawCall:
        try:
            return await self._primary(messages)
        except LLMUnavailable as primary_error:
            # Only "unavailable" falls back. A bad key or a bug in the primary path raises
            # through unchanged, instead of hiding behind the (more expensive) fallback.
            logger.warning("primary model unavailable, using the fallback: %s", primary_error)
        try:
            return await self._fallback(messages)
        except LLMUnavailable as fallback_error:
            raise LLMUnavailable("both the primary and the fallback model are unavailable") from (
                fallback_error
            )
