"""The analysis output contract (SPEC §5).

These models are used three ways, so field descriptions are written for all readers:
1. as the JSON schema given to the LLM for structured output (descriptions = instructions),
2. to validate whatever the LLM actually returned (never trust it blindly),
3. as the API response body and OpenAPI schema for the frontend.
"""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from review_insight.domain.enums import Aspect, Route, Sentiment, Urgency

NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Contract(BaseModel):
    # extra="forbid": invented keys are a contract violation, not something to drop silently.
    # frozen=True: a validated result cannot be mutated afterwards by mistake.
    model_config = ConfigDict(extra="forbid", frozen=True)


class AspectSentiment(_Contract):
    aspect: Aspect = Field(description="Product aspect the customer talks about.")
    sentiment: Sentiment = Field(description="Customer's sentiment toward this aspect.")
    evidence: NonBlankStr = Field(
        description="Exact span copied verbatim from the review that supports the sentiment."
    )


class AnalysisResult(_Contract):
    aspects: list[AspectSentiment] = Field(
        description="One entry per aspect mentioned in the review; empty if none is mentioned."
    )
    urgency: Urgency = Field(description="How fast a human must act on this review.")
    route_to: Route = Field(description="Team that should handle the review.")

    @model_validator(mode="after")
    def _one_sentiment_per_aspect(self) -> Self:
        names = [item.aspect for item in self.aspects]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            # Message is precise on purpose: it is fed back to the LLM in the repair retry (T8).
            raise ValueError(
                f"duplicate aspect(s): {', '.join(duplicates)}. "
                "Return each aspect at most once, with a single overall sentiment."
            )
        return self
