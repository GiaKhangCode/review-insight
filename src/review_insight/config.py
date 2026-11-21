"""Application settings, loaded from environment variables (12-factor config).

Every setting is typed and validated once at startup: a missing or malformed value
stops the app immediately (fail fast) instead of breaking a request hours later.
New settings are added here by the task that first needs them (LLM keys in T7,
DATABASE_URL in T11, ...).
"""

from functools import lru_cache
from typing import Literal

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


@lru_cache
def get_settings() -> Settings:
    """Build settings once per process and reuse them (used by the production entrypoint)."""
    return Settings()
