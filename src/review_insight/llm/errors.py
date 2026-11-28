"""Typed LLM errors. The API layer (T11) maps each one to an HTTP status.

- `LLMUnavailable`: no model produced an answer (transient errors outlasted retries and
  fallback). Retrying later may work. Maps to 503.
- `InvalidLLMOutput`: a model answered, but the answer was rejected (schema or grounding),
  even after the repair retry. Maps to 502.

Errors that are not here (a bad API key, a malformed request, a bug) propagate unchanged:
they are not LLM conditions to handle, but problems to fix.
"""

from typing import Literal


class LLMError(Exception):
    """Base class, so callers can catch every expected LLM failure at once."""


class LLMUnavailable(LLMError):  # noqa: N818 - name from SPEC; reads as a state, not an event
    """No model answered."""


class InvalidLLMOutput(LLMError):  # noqa: N818 - name from SPEC
    """A model answered, but the output was rejected."""

    def __init__(self, failure: Literal["schema", "ungrounded"], detail: str) -> None:
        super().__init__(f"{failure} failure: {detail}")
        self.failure = failure
        self.detail = detail
