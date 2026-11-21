"""Shared test fixtures."""

import pytest

from review_insight.config import Settings


@pytest.fixture(autouse=True)
def isolate_settings_from_the_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests see only what they set: no `.env` file, no secrets from the developer's shell.

    Without this, a test can pass locally because `.env` holds a real GOOGLE_API_KEY and
    fail in CI, where there is no `.env` (found when CI failed on T7).
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
