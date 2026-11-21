"""Behaviour of the app skeleton: liveness probe and API metadata."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from review_insight import __version__
from review_insight.config import Settings
from review_insight.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    # Settings are built explicitly, so tests never depend on a developer's .env file.
    settings = Settings(app_env="ci", log_level="WARNING")
    # Using TestClient as a context manager runs the app's startup/shutdown (lifespan).
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_healthz_returns_ok_with_version_and_env(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__, "env": "ci"}


def test_openapi_schema_exposes_title_and_version(client: TestClient) -> None:
    # The Next.js frontend will generate its TypeScript types from this document.
    response = client.get("/openapi.json")

    assert response.status_code == 200
    info = response.json()["info"]
    assert info["title"] == "ReviewInsight API"
    assert info["version"] == __version__
