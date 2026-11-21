"""Eval-only `ModelCall` wrappers: an on-disk response cache and a request throttle.

Compose as `DiskCachedCall(ThrottledCall(real_call))`: cache hits return at once, and only
real API calls wait for the throttle. The cache lives in evals/.cache/ (gitignored).
"""

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

from review_insight.domain.schemas import AnalysisResult
from review_insight.llm.analyzer import ModelCall, RawCall
from review_insight.llm.pricing import TokenUsage

DEFAULT_CACHE_DIR = Path("evals/.cache/responses")

# Bump if the entry layout changes: old entries then simply stop matching.
_FORMAT = 1


def _cache_key(model: str, messages: Sequence[BaseMessage]) -> str:
    """Hash of everything that determines the answer: model + the exact messages sent."""
    payload = {
        "format": _FORMAT,
        "model": model,
        "messages": [[message.type, message.content] for message in messages],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _to_json(raw: RawCall) -> dict[str, Any]:
    return {
        "format": _FORMAT,
        "model": raw.model,
        "output": None if raw.output is None else raw.output.model_dump(mode="json"),
        "error": raw.error,
        "usage": raw.usage.model_dump(),
        "latency_ms": raw.latency_ms,
    }


def _read(path: Path) -> RawCall | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        output = data["output"]
        return RawCall(
            model=data["model"],
            output=None if output is None else AnalysisResult.model_validate(output),
            error=data["error"],
            usage=TokenUsage.model_validate(data["usage"]),
            latency_ms=float(data["latency_ms"]),  # the original call's latency, not the read's
            cached=True,
        )
    except (ValueError, KeyError, TypeError) as exc:  # JSON and validation errors included
        raise ValueError(
            f"corrupted cache entry {path}: {exc}. Delete the file to call the model again."
        ) from exc


def _write(path: Path, raw: RawCall) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps(_to_json(raw), ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)  # atomic: readers see the old state or the whole new file


class DiskCachedCall:
    """Replays responses already paid for. Exceptions (timeouts, 5xx) are never cached."""

    def __init__(self, inner: ModelCall, *, cache_dir: Path, model: str) -> None:
        self._inner = inner
        self._cache_dir = cache_dir
        self._model = model  # the requested model; part of the key

    async def __call__(self, messages: Sequence[BaseMessage]) -> RawCall:
        key = _cache_key(self._model, messages)
        path = self._cache_dir / key[:2] / f"{key}.json"  # 256 subfolders, not one huge dir

        # File I/O runs in a worker thread so it never blocks the event loop.
        hit = await asyncio.to_thread(_read, path)
        if hit is not None:
            return hit
        raw = await self._inner(messages)
        await asyncio.to_thread(_write, path, raw)
        return raw


class ThrottledCall:
    """Starts at most one request per `min_interval_s` (start-to-start), across all callers."""

    def __init__(
        self,
        inner: ModelCall,
        *,
        min_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._inner = inner
        self._min_interval_s = min_interval_s
        self._clock = clock  # monotonic: unaffected by system clock changes
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._next_start: float | None = None

    async def __call__(self, messages: Sequence[BaseMessage]) -> RawCall:
        async with self._lock:  # concurrent callers queue here, one start at a time
            now = self._clock()
            if self._next_start is not None and now < self._next_start:
                await self._sleep(self._next_start - now)
            self._next_start = self._clock() + self._min_interval_s
        # The request runs outside the lock: a slow call does not delay the next start.
        return await self._inner(messages)
