"""Settings (12-factor config): LLM settings added in T7."""

import pytest
from pydantic import SecretStr, ValidationError

from review_insight.config import Settings

LLM_VARS = (
    "GOOGLE_API_KEY",
    "PRIMARY_MODEL",
    "FALLBACK_MODEL",
    "LLM_TIMEOUT_S",
    "APP_ENV",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "PROMPT_LABEL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate from the developer's shell; `_env_file=None` below isolates from `.env`.
    for name in LLM_VARS:
        monkeypatch.delenv(name, raising=False)


def test_llm_defaults_match_the_spec() -> None:
    settings = Settings(_env_file=None)

    assert settings.primary_model == "gemini-3.1-flash-lite"
    assert settings.fallback_model == "gemini-3.5-flash-lite"
    assert settings.llm_timeout_s == 15
    assert settings.google_api_key is None  # not needed for /healthz, tests or CI


def test_llm_settings_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-value")
    monkeypatch.setenv("PRIMARY_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setenv("LLM_TIMEOUT_S", "7.5")

    settings = Settings(_env_file=None)

    assert settings.google_api_key == SecretStr("secret-value")
    assert settings.primary_model == "gemini-3.5-flash-lite"
    assert settings.llm_timeout_s == 7.5


def test_the_api_key_never_appears_in_repr_or_dumps(monkeypatch: pytest.MonkeyPatch) -> None:
    # Settings objects end up in logs and tracebacks; the secret must not.
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-value")

    settings = Settings(_env_file=None)

    assert "secret-value" not in repr(settings)
    assert "secret-value" not in str(settings.model_dump())
    assert settings.require_google_api_key().get_secret_value() == "secret-value"


def test_an_empty_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # `.env.example` ships `GOOGLE_API_KEY=` (empty); that must not pass as a key.
    monkeypatch.setenv("GOOGLE_API_KEY", "  ")

    assert Settings(_env_file=None).google_api_key is None


def test_requiring_a_missing_key_says_how_to_fix_it() -> None:
    with pytest.raises(ValueError, match=r"GOOGLE_API_KEY.*\.env"):
        Settings(_env_file=None).require_google_api_key()


def test_production_refuses_to_start_without_a_key() -> None:
    # Fail fast at startup, not on the first customer request.
    with pytest.raises(ValidationError, match="GOOGLE_API_KEY"):
        Settings(_env_file=None, app_env="production")


@pytest.mark.parametrize("timeout", [0, -1, 121])
def test_timeout_must_be_a_sane_number_of_seconds(timeout: float) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_timeout_s=timeout)


# --- Langfuse and prompt label (T9) ------------------------------------------------------


def test_langfuse_is_off_by_default() -> None:
    # No keys in tests and CI: tracing and remote prompts are disabled, nothing is sent.
    settings = Settings(_env_file=None)

    assert settings.langfuse_enabled is False
    assert settings.langfuse_host == "https://cloud.langfuse.com"
    assert settings.prompt_label == "production"


def test_langfuse_is_on_when_both_keys_are_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

    settings = Settings(_env_file=None)

    assert settings.langfuse_enabled is True
    assert "sk-lf-test" not in repr(settings)  # the secret key is a SecretStr


@pytest.mark.parametrize("only", ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"])
def test_half_a_langfuse_config_is_rejected(monkeypatch: pytest.MonkeyPatch, only: str) -> None:
    # One key without the other is a typo, not a choice: fail at startup instead of
    # silently running without traces.
    monkeypatch.setenv(only, "some-value")

    with pytest.raises(ValidationError, match="LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"):
        Settings(_env_file=None)


@pytest.mark.parametrize("label", ["Production", "prod label", ""])
def test_prompt_label_must_be_a_simple_name(label: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, prompt_label=label)
