"""Cost accounting from token usage and config/pricing.yaml (SPEC §3, §7.2)."""

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from review_insight.llm.pricing import PriceTable, TokenUsage, UnknownModelError, load_price_table

REPO_ROOT = Path(__file__).resolve().parents[2]

# Round numbers, independent of real prices: these tests check the formula, so they must
# not break when Google changes prices and config/pricing.yaml is updated.
TABLE = PriceTable.model_validate(
    {
        "source": "https://example.com/pricing",
        "retrieved": "2026-11-21",
        "currency": "USD",
        "per_tokens": 1_000_000,
        "models": {"m": {"input": 1.0, "cached_input": 0.1, "output": 10.0}},
    }
)


def test_cost_is_tokens_times_price_per_million() -> None:
    usage = TokenUsage(input_tokens=600, output_tokens=150)

    # (600 * $1 + 150 * $10) / 1M
    assert TABLE.cost_usd("m", usage) == pytest.approx(0.0021)


def test_cached_input_tokens_are_billed_at_the_cached_price() -> None:
    usage = TokenUsage(input_tokens=1000, output_tokens=0, cached_input_tokens=800)

    # 200 uncached * $1 + 800 cached * $0.1
    assert TABLE.cost_usd("m", usage) == pytest.approx((200 * 1.0 + 800 * 0.1) / 1e6)


def test_no_tokens_cost_nothing() -> None:
    assert TABLE.cost_usd("m", TokenUsage(input_tokens=0, output_tokens=0)) == 0.0


def test_api_style_model_names_are_accepted() -> None:
    # The API reports names as "models/<id>"; config uses the bare id.
    usage = TokenUsage(input_tokens=600, output_tokens=150)

    assert TABLE.cost_usd("models/m", usage) == TABLE.cost_usd("m", usage)


def test_a_model_without_a_price_fails_loudly() -> None:
    # Silently costing an unknown model at $0 would under-report spend.
    with pytest.raises(UnknownModelError, match=r"pricing\.yaml"):
        TABLE.cost_usd("gemini-9-ultra", TokenUsage(input_tokens=1, output_tokens=1))


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": -1, "output_tokens": 0},
        {"input_tokens": 10, "output_tokens": 0, "cached_input_tokens": 11},  # cached ⊆ input
    ],
)
def test_impossible_token_counts_are_rejected(usage: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        TokenUsage.model_validate(usage)


@pytest.mark.parametrize(
    "change",
    [
        {"models": {"m": {"input": -1.0, "cached_input": 0.1, "output": 10.0}}},
        {"models": {"m": {"input": 1.0, "output": 10.0}}},  # missing cached_input
        {"currency": "VND"},  # every cost in the project is USD
        {"discount": 0.5},  # unknown key
    ],
)
def test_invalid_price_tables_are_rejected(change: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PriceTable.model_validate(TABLE.model_dump() | change)


def test_the_committed_price_table_covers_the_spec_models() -> None:
    table = load_price_table(REPO_ROOT / "config/pricing.yaml")

    assert table.retrieved == date(2026, 11, 21)
    assert {"gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-embedding-2"} <= set(
        table.models
    )
