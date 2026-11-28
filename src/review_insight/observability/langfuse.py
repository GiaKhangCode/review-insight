"""Langfuse tracing: one trace per analyzed review, one generation per model call.

    analyze_review   (span)        input review, output result, prompt, attempts, failure
     ├─ llm_call     (generation)  model, usage, cost (from config/pricing.yaml), level
     └─ llm_call     (generation)  only when a repair retry ran

Tracing observes and never changes behaviour: errors are recorded, then re-raised. With no
Langfuse keys configured, nothing is created and nothing is sent.
"""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage
from langfuse import Langfuse

from review_insight import __version__
from review_insight.config import Settings
from review_insight.llm.analyzer import Analysis, ModelCall, RawCall, ReviewAnalyzer
from review_insight.llm.pricing import PriceTable


def build_langfuse(settings: Settings) -> Langfuse | None:
    """A client when both keys are configured, else None. Makes no network call."""
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_host,
        environment=settings.app_env,  # local / ci / production, filterable in the UI
        release=__version__,
    )


class TracedCall:
    """`ModelCall` wrapper: records each model call as a Langfuse generation."""

    def __init__(self, inner: ModelCall, *, langfuse: Langfuse, prices: PriceTable) -> None:
        self._inner = inner
        self._langfuse = langfuse
        self._prices = prices

    async def __call__(self, messages: Sequence[BaseMessage]) -> RawCall:
        with self._langfuse.start_as_current_observation(
            name="llm_call",
            as_type="generation",
            input=[{"role": message.type, "content": message.content} for message in messages],
        ) as generation:
            try:
                raw = await self._inner(messages)
            except Exception as exc:
                generation.update(level="ERROR", status_message=str(exc))
                raise
            usage = raw.usage
            generation.update(
                model=raw.model,
                output=None if raw.output is None else raw.output.model_dump(mode="json"),
                # Non-overlapping buckets: Langfuse sums them, so the total stays correct.
                usage_details={
                    "input": usage.input_tokens - usage.cached_input_tokens,
                    "cache_read_input_tokens": usage.cached_input_tokens,
                    "output": usage.output_tokens,
                },
                # Our price table is the single source of truth for cost.
                cost_details={"total": self._prices.cost_usd(raw.model, usage)},
                metadata={"latency_ms": round(raw.latency_ms, 1), "cached": raw.cached},
                level="WARNING" if raw.output is None else "DEFAULT",
                status_message=raw.error,
            )
            return raw


async def analyze_traced(
    analyzer: ReviewAnalyzer, text: str, *, langfuse: Langfuse | None
) -> tuple[Analysis, str | None]:
    """Analyze one review inside a trace. Returns the analysis and the trace id (or None)."""
    if langfuse is None:
        return await analyzer.analyze(text), None

    with langfuse.start_as_current_observation(
        name="analyze_review", as_type="span", input={"review": text}
    ) as span:
        trace_id = langfuse.get_current_trace_id()
        try:
            analysis = await analyzer.analyze(text)
        except Exception as exc:
            span.update(level="ERROR", status_message=str(exc))
            raise
        span.update(
            output=None if analysis.result is None else analysis.result.model_dump(mode="json"),
            metadata={
                "prompt": analysis.prompt_ref,
                "model": analysis.model,
                "attempts": analysis.attempts,
                "failure": analysis.failure,
                "first_failure": analysis.first_failure,
                "cost_usd": analysis.cost_usd,
                "cached": analysis.cached,
            },
            level="WARNING" if analysis.failure else "DEFAULT",
            status_message=analysis.error,
        )
    return analysis, trace_id
