"""Application settings, loaded from environment variables (12-factor config).

Every setting is typed and validated once at startup: a missing or malformed value
stops the app immediately (fail fast) instead of breaking a request hours later.
New settings are added here by the task that first needs them (LLM keys in T7,
DATABASE_URL in T11, ...).
"""

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AppEnv = Literal["local", "ci", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    """Typed view of the process environment.

    Precedence (highest first): constructor arguments > environment variables > `.env` file
    > defaults declared below. Env var names are case-insensitive (`APP_ENV` -> `app_env`).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env holds keys for later tasks; don't fail on them yet
    )

    app_env: AppEnv = "local"
    log_level: LogLevel = "INFO"

    # --- LLM (T7) ---
    # SecretStr: shown as '**********' in repr, logs and dumps; read with get_secret_value().
    # Optional so /healthz, unit tests and CI run without a key; required in production.
    google_api_key: SecretStr | None = None
    primary_model: str = "gemini-3.1-flash-lite"
    fallback_model: str = "gemini-3.5-flash-lite"
    llm_timeout_s: float = Field(default=15, gt=0, le=120)

    # --- Langfuse and prompts (T9) ---
    # Both keys set: tracing on, prompts fetched by label. Neither: off, local YAML prompts.
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    # Which Langfuse prompt label to serve; must match a label exactly (case-sensitive).
    prompt_label: str = Field(default="production", pattern=r"^[a-z0-9_-]+$")

    @field_validator("google_api_key", "langfuse_public_key", "langfuse_secret_key", mode="before")
    @classmethod
    def _blank_key_is_missing(cls, value: object) -> object:
        # `.env.example` ships `GOOGLE_API_KEY=`; an empty value is not a key.
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _production_needs_a_key(self) -> Self:
        if self.app_env == "production" and self.google_api_key is None:
            raise ValueError("GOOGLE_API_KEY must be set when APP_ENV=production")
        return self

    @model_validator(mode="after")
    def _langfuse_keys_come_in_pairs(self) -> Self:
        if (self.langfuse_public_key is None) != (self.langfuse_secret_key is None):
            raise ValueError(
                "set both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, or neither "
                "(one without the other would silently disable tracing)"
            )
        return self

    @property
    def langfuse_enabled(self) -> bool:
        return self.langfuse_public_key is not None and self.langfuse_secret_key is not None

    def require_google_api_key(self) -> SecretStr:
        """The key, for code paths that call Gemini; a clear error if it is not configured."""
        if self.google_api_key is None:
            raise ValueError("GOOGLE_API_KEY is not set: add it to .env (see .env.example)")
        return self.google_api_key


@lru_cache
def get_settings() -> Settings:
    """Build settings once per process and reuse them (used by the production entrypoint)."""
    return Settings()
