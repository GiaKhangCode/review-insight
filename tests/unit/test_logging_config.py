"""Structured logging: one JSON object per line outside local dev."""

import json
import logging
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from review_insight.config import Settings
from review_insight.observability.logging_config import JsonFormatter, configure_logging


def _record(exc_info: Any = None, **extra: Any) -> logging.LogRecord:
    record = logging.LogRecord(
        name="review_insight.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="analyzed %s reviews: %s",
        args=(3, "pin trâu, camera tệ"),
        exc_info=exc_info,
    )
    # Same effect as logger.info("...", extra={...}).
    for key, value in extra.items():
        setattr(record, key, value)
    return record


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    # configure_logging mutates global state; put it back so other tests are unaffected.
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def test_json_formatter_emits_single_line_json_with_core_fields() -> None:
    line = JsonFormatter().format(_record())

    payload = json.loads(line)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "review_insight.test"
    # %-args are interpolated and Vietnamese is kept readable (not \u-escaped).
    assert payload["message"] == "analyzed 3 reviews: pin trâu, camera tệ"
    assert "pin trâu" in line
    assert "ts" in payload
    assert "\n" not in line


def test_json_formatter_promotes_extra_fields_to_top_level() -> None:
    payload = json.loads(JsonFormatter().format(_record(request_id="req-123", latency_ms=42)))

    assert payload["request_id"] == "req-123"
    assert payload["latency_ms"] == 42


def test_json_formatter_drops_uvicorn_terminal_color_field() -> None:
    # Uvicorn attaches `color_message` (ANSI escape codes) for terminals; noise in JSON logs.
    payload = json.loads(JsonFormatter().format(_record(color_message="\x1b[36mhi\x1b[0m")))

    assert "color_message" not in payload


def test_json_formatter_keeps_traceback_inside_the_json_line() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        line = JsonFormatter().format(_record(exc_info=sys.exc_info()))

    assert "\n" not in line  # multi-line traceback is escaped, so log tools see ONE event
    assert "ValueError: boom" in json.loads(line)["exc_info"]


@pytest.mark.usefixtures("restore_root_logger")
def test_configure_logging_uses_json_outside_local() -> None:
    # A valid production config needs a key (T7); a fake one is enough to build Settings.
    configure_logging(
        Settings(app_env="production", log_level="WARNING", google_api_key="test-key")
    )

    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)


@pytest.mark.usefixtures("restore_root_logger")
def test_configure_logging_uses_plain_text_locally() -> None:
    configure_logging(Settings(app_env="local", log_level="DEBUG"))

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert not isinstance(root.handlers[0].formatter, JsonFormatter)
