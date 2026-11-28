"""Review analysis with an LLM (SPEC §6, §10).

Three layers, each replaceable on its own:
- `ReviewAnalyzer`: renders the prompt, prices the call, checks evidence grounding;
- `ModelCall`: any async callable from messages to a `RawCall` (the port);
- `LangChainModelCall`: the adapter over `with_structured_output(..., include_raw=True)`.
Caching, throttling, retries and fallback are `ModelCall` wrappers, so the analyzer never
changes. A rejected answer gets one repair retry: the same conversation plus the reason.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from review_insight.domain.grounding import UngroundedEvidenceError, validate_grounded
from review_insight.domain.schemas import AnalysisResult
from review_insight.llm.errors import InvalidLLMOutput
from review_insight.llm.pricing import PriceTable, TokenUsage
from review_insight.llm.prompts import PromptBundle, sanitize_review

Failure = Literal["schema", "ungrounded"]

_REPAIR_INSTRUCTION = (
    "Your previous answer was rejected: {error}\n"
    "Analyze the same review again and return a corrected answer that follows every "
    "instruction."
)


@dataclass(frozen=True, slots=True)
class RawCall:
    """What one model call returned, before any of our checks."""

    model: str  # the model that actually served the call
    output: AnalysisResult | None  # None when the response did not fit the schema
    error: str | None  # why output is None
    usage: TokenUsage
    latency_ms: float
    cached: bool = False  # True when replayed from a cache instead of calling the API


class ModelCall(Protocol):
    async def __call__(self, messages: Sequence[BaseMessage]) -> RawCall: ...


@dataclass(frozen=True, slots=True)
class Analysis:
    """Outcome of analyzing one review. Failures are data: a failed call is still billed."""

    result: AnalysisResult | None  # set only if schema-valid AND every evidence is grounded
    failure: Failure | None  # of the final attempt: what the caller gets
    first_failure: Failure | None  # of the first attempt: for first-try quality metrics
    attempts: int  # 1, or 2 when a repair was attempted
    error: str | None  # worded for the LLM, so it can be sent back as a repair instruction
    model: str  # model that served the final attempt
    prompt_ref: str
    usage: TokenUsage  # summed over attempts: every attempt is billed
    cost_usd: float
    latency_ms: float  # summed over attempts: the caller waited for all of them
    cached: bool  # True only if every attempt was replayed from a cache
    evidence_total: int  # evidence spans in the final schema-valid output
    evidence_ungrounded: int

    def unwrap(self) -> AnalysisResult:
        """The accepted result, or `InvalidLLMOutput` (for callers that want exceptions)."""
        if self.result is None:
            raise InvalidLLMOutput(self.failure or "schema", self.error or "no output")
        return self.result


@dataclass(frozen=True, slots=True)
class _Attempt:
    raw: RawCall
    cost_usd: float
    result: AnalysisResult | None
    failure: Failure | None
    error: str | None
    evidence_total: int
    evidence_ungrounded: int


class ReviewAnalyzer:
    def __init__(
        self, *, call: ModelCall, prompt: PromptBundle, prices: PriceTable, repair: bool = True
    ) -> None:
        self._call = call
        self._prompt = prompt
        self._prices = prices
        self._repair = repair

    async def analyze(self, text: str) -> Analysis:
        messages = self._prompt.render(text)
        first = await self._attempt(messages, text)
        if first.failure is None or not self._repair:
            return self._combine(first)

        # One repair: the same conversation plus the rejection reason. The reason echoes
        # model output (which may echo the review), so it is untrusted and sanitized too.
        reason = sanitize_review(first.error or "the answer was rejected")
        repair = HumanMessage(content=_REPAIR_INSTRUCTION.format(error=reason))
        second = await self._attempt([*messages, repair], text)
        return self._combine(first, second)

    async def _attempt(self, messages: Sequence[BaseMessage], text: str) -> _Attempt:
        raw = await self._call(messages)
        # Priced first: an unknown model is a configuration error and must stop the run.
        cost = self._prices.cost_usd(raw.model, raw.usage)

        if raw.output is None:
            error = raw.error or "the model returned no structured output"
            return _Attempt(raw, cost, None, "schema", error, 0, 0)

        total = len(raw.output.aspects)
        try:
            # Checked against the original text: evidence must be what the customer wrote.
            validate_grounded(raw.output, source_text=text)
        except UngroundedEvidenceError as exc:
            return _Attempt(raw, cost, None, "ungrounded", str(exc), total, len(exc.ungrounded))
        return _Attempt(raw, cost, raw.output, None, None, total, 0)

    def _combine(self, *attempts: _Attempt) -> Analysis:
        first, final = attempts[0], attempts[-1]
        return Analysis(
            result=final.result,
            failure=final.failure,
            first_failure=first.failure,
            attempts=len(attempts),
            error=final.error,
            model=final.raw.model,
            prompt_ref=self._prompt.ref,
            usage=TokenUsage(
                input_tokens=sum(a.raw.usage.input_tokens for a in attempts),
                output_tokens=sum(a.raw.usage.output_tokens for a in attempts),
                cached_input_tokens=sum(a.raw.usage.cached_input_tokens for a in attempts),
            ),
            cost_usd=sum(a.cost_usd for a in attempts),
            latency_ms=sum(a.raw.latency_ms for a in attempts),
            cached=all(a.raw.cached for a in attempts),
            evidence_total=final.evidence_total,
            evidence_ungrounded=final.evidence_ungrounded,
        )


class LangChainModelCall:
    """Adapter: a structured-output runnable built with `include_raw=True`."""

    def __init__(self, runnable: Runnable[Any, Any], *, model: str) -> None:
        self._runnable = runnable
        self.model = model

    async def __call__(self, messages: Sequence[BaseMessage]) -> RawCall:
        start = time.perf_counter()
        response = await self._runnable.ainvoke(list(messages))
        latency_ms = (time.perf_counter() - start) * 1000

        # LangChain types this as `dict | BaseModel`; only include_raw=True gives the dict.
        if not isinstance(response, dict):
            raise TypeError("the runnable must be built with include_raw=True")
        message = response["raw"]
        if not isinstance(message, AIMessage):
            raise TypeError(f"expected an AIMessage as raw output, got {type(message).__name__}")
        parsed = response.get("parsed")
        output = parsed if isinstance(parsed, AnalysisResult) else None
        error = None
        if output is None:
            parsing_error = response.get("parsing_error")
            error = (
                str(parsing_error) if parsing_error else "the model returned no structured output"
            )

        return RawCall(
            model=str(message.response_metadata.get("model_name") or self.model),
            output=output,
            error=error,
            usage=_usage_of(message),
            latency_ms=latency_ms,
        )


def _usage_of(message: AIMessage) -> TokenUsage:
    meta = message.usage_metadata
    if meta is None:
        raise ValueError("response has no usage metadata, so the call cannot be costed")
    cached = meta.get("input_token_details", {}).get("cache_read", 0)
    # output_tokens already includes thinking tokens (langchain-google-genai adds them).
    return TokenUsage(
        input_tokens=meta["input_tokens"],
        output_tokens=meta["output_tokens"],
        cached_input_tokens=cached,
    )


def build_gemini_call(*, model: str, api_key: SecretStr, timeout_s: float) -> LangChainModelCall:
    """A Gemini model call with structured output. Builds the client; makes no request."""
    llm = ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
        # Newer Gemini models reject custom sampling parameters, and LangChain would
        # otherwise send its own default (0.7). None = the model's server-side default, the
        # same rule for every model, so comparisons stay fair. Reproducibility comes from
        # the eval response cache, not from temperature 0.
        temperature=None,
        timeout=timeout_s,
        # Retries are ours to control (T10: backoff, fallback model). The client's default
        # of 6 silent retries would hide failures and burn the daily request quota.
        max_retries=0,
    )
    structured = llm.with_structured_output(AnalysisResult, include_raw=True)
    return LangChainModelCall(structured, model=model)
