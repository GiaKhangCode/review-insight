"""Cost of an LLM call from its token usage and the price table in config/pricing.yaml."""

from datetime import date
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

DEFAULT_PRICING_PATH = Path("config/pricing.yaml")


class UnknownModelError(LookupError):
    """A model has no entry in the price table. Never priced at $0 by default."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TokenUsage(_Strict):
    """Token counts reported by the API for one call."""

    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt  # includes thinking tokens, which are billed as output
    cached_input_tokens: NonNegativeInt = 0  # the part of input_tokens served from cache

    @model_validator(mode="after")
    def _cached_is_part_of_input(self) -> Self:
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError(
                f"cached_input_tokens ({self.cached_input_tokens}) cannot exceed "
                f"input_tokens ({self.input_tokens})"
            )
        return self


class ModelPrice(_Strict):
    """USD per `PriceTable.per_tokens` tokens. All three are required, so none defaults to 0."""

    input: NonNegativeFloat
    cached_input: NonNegativeFloat
    output: NonNegativeFloat


class PriceTable(_Strict):
    source: str
    retrieved: date
    currency: Literal["USD"]
    per_tokens: PositiveInt
    models: dict[str, ModelPrice]

    def price_of(self, model: str) -> ModelPrice:
        # The API reports "models/gemini-3.1-flash-lite"; the table uses the bare id.
        model_id = model.removeprefix("models/")
        try:
            return self.models[model_id]
        except KeyError:
            raise UnknownModelError(
                f"no price for model {model_id!r} in config/pricing.yaml; "
                "add it from the official pricing page (see docs/GEMINI.md)"
            ) from None

    def cost_usd(self, model: str, usage: TokenUsage) -> float:
        price = self.price_of(model)
        uncached = usage.input_tokens - usage.cached_input_tokens
        total = (
            uncached * price.input
            + usage.cached_input_tokens * price.cached_input
            + usage.output_tokens * price.output
        )
        return total / self.per_tokens


def load_price_table(path: Path = DEFAULT_PRICING_PATH) -> PriceTable:
    return PriceTable.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
