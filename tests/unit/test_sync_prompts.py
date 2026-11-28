"""scripts/sync_prompts.py: push prompts/*.yaml to Langfuse, idempotently, and move a label."""

from pathlib import Path
from typing import Any

import pytest

from review_insight.llm.prompts import PromptSpec, fingerprint_of, load_prompt_spec
from scripts.sync_prompts import sync

REPO_ROOT = Path(__file__).resolve().parents[2]
V1 = load_prompt_spec(REPO_ROOT / "prompts/analyze_v1.yaml")
V2 = load_prompt_spec(REPO_ROOT / "prompts/analyze_v2.yaml")


class FakeStore:
    """In-memory Langfuse prompt store: versions numbered 1, 2, ... like the real one."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.labels: dict[str, dict[str, int]] = {}  # name -> label -> remote version

    def remote_fingerprints(self, name: str) -> dict[str, int]:
        return {
            c["config"]["fingerprint"]: i + 1
            for i, c in enumerate(self.created)
            if c["name"] == name
        }

    def create(self, spec: PromptSpec, *, messages: list[dict[str, str]], config: Any) -> int:
        self.created.append({"name": spec.name, "messages": messages, "config": config})
        return len([c for c in self.created if c["name"] == spec.name])

    def set_label(self, name: str, *, version: int, label: str) -> None:
        self.labels.setdefault(name, {})[label] = version


def test_the_first_sync_creates_every_version_in_order_and_labels_the_latest() -> None:
    store = FakeStore()

    actions = sync([V2, V1], store=store, label="staging")

    assert [c["config"]["yaml_version"] for c in store.created] == [1, 2]
    assert store.labels == {"analyze_review": {"staging": 2}}
    assert actions == [
        "analyze_review@v1: created as Langfuse version 1",
        "analyze_review@v2: created as Langfuse version 2",
        "analyze_review: label 'staging' -> v2 (Langfuse version 2)",
    ]


def test_what_is_pushed_is_a_chat_prompt_with_our_version_and_fingerprint() -> None:
    store = FakeStore()

    sync([V2], store=store, label="staging")

    [pushed] = store.created
    assert pushed["messages"] == V2.to_langfuse_messages()
    assert pushed["config"] == {
        "yaml_version": 2,
        "fingerprint": fingerprint_of(V2),
        "description": V2.description,
    }


def test_syncing_again_creates_nothing() -> None:
    # Idempotent: re-running (e.g. in CI on every merge) must not pile up duplicate versions.
    store = FakeStore()
    sync([V1, V2], store=store, label="staging")

    actions = sync([V1, V2], store=store, label="staging")

    assert len(store.created) == 2
    assert actions[:2] == [
        "analyze_review@v1: unchanged (Langfuse version 1)",
        "analyze_review@v2: unchanged (Langfuse version 2)",
    ]


def test_an_edited_prompt_is_a_new_remote_version_even_without_a_version_bump() -> None:
    store = FakeStore()
    sync([V2], store=store, label="staging")
    edited = V2.model_copy(update={"system": V2.system + "\nBe concise."})

    sync([edited], store=store, label="staging")

    assert len(store.created) == 2
    assert store.labels["analyze_review"]["staging"] == 2


def test_a_specific_version_can_be_labelled_for_a_rollback() -> None:
    store = FakeStore()
    sync([V1, V2], store=store, label="production")

    sync([V1, V2], store=store, label="production", version=1)

    assert store.labels["analyze_review"]["production"] == 1


def test_an_unknown_version_is_an_error() -> None:
    with pytest.raises(ValueError, match="v9"):
        sync([V1, V2], store=FakeStore(), label="production", version=9)


def test_a_dry_run_changes_nothing() -> None:
    store = FakeStore()

    actions = sync([V1, V2], store=store, label="production", dry_run=True)

    assert store.created == []
    assert store.labels == {}
    assert actions[0] == "analyze_review@v1: would create"
    assert actions[-1] == "analyze_review: would set label 'production' -> v2"
