"""Process-wide logging setup.

- `local`: human-readable lines for the developer's terminal.
- `ci` / `production`: one JSON object per line, so log tools (Azure Log Analytics)
  can filter by field, e.g. `request_id == "..."` or `latency_ms > 2000`.

Privacy rule for callers: log metadata (ids, lengths, latency), never full review text.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from review_insight.config import Settings

# Attributes every LogRecord has. Anything else on a record came from `extra={...}`.
_STANDARD_RECORD_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

# Extras added by libraries that are pure noise in JSON (uvicorn's ANSI-colored copy of msg).
_DROPPED_EXTRAS = frozenset({"color_message"})

_LOCAL_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

# Third-party loggers that install their own handlers; we route them through ours.
_THIRD_PARTY_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


class JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Promote `extra={...}` fields to top-level keys.
        for key, value in vars(record).items():
            if (
                key not in _STANDARD_RECORD_ATTRS
                and key not in _DROPPED_EXTRAS
                and not key.startswith("_")
            ):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        # ensure_ascii=False keeps Vietnamese readable; default=str never crashes on
        # non-JSON values (UUID, datetime, ...). json.dumps escapes newlines -> one line.
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(settings: Settings) -> None:
    """Install exactly one stdout handler on the root logger. Safe to call repeatedly."""
    handler = logging.StreamHandler(sys.stdout)
    if settings.app_env == "local":
        handler.setFormatter(logging.Formatter(_LOCAL_FORMAT))
    else:
        handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers[:] = [handler]  # replace, not append: avoids duplicated lines
    root.setLevel(settings.log_level)

    for name in _THIRD_PARTY_LOGGERS:
        third_party = logging.getLogger(name)
        third_party.handlers.clear()
        third_party.propagate = True  # bubble up to the root handler -> same format
