"""Push prompts/*.yaml to Langfuse and point a label at one version (SPEC §6).

Usage:
    uv run python scripts/sync_prompts.py --label staging --dry-run   # show the plan only
    uv run python scripts/sync_prompts.py --label staging             # latest YAML version
    uv run python scripts/sync_prompts.py --label production --version 1   # rollback

Git is the source of truth. Each YAML version is created in Langfuse once (matched by the
text fingerprint, so re-running is safe); then the label is moved. The API serves whatever
version carries PROMPT_LABEL, so moving a label changes production without a deploy.
"""

import argparse
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from langfuse import Langfuse

from review_insight.config import Settings
from review_insight.llm.prompts import PromptSpec, fingerprint_of, load_prompt_spec
from review_insight.observability.langfuse import build_langfuse

PROMPTS_DIR = Path("prompts")


class PromptStore(Protocol):
    def remote_fingerprints(self, name: str) -> dict[str, int]:
        """Fingerprint -> Langfuse version, for every version of `name` already pushed."""
        ...

    def create(self, spec: PromptSpec, *, messages: list[dict[str, str]], config: Any) -> int:
        """Create a new Langfuse version; return its number."""
        ...

    def set_label(self, name: str, *, version: int, label: str) -> None: ...


def sync(
    specs: Sequence[PromptSpec],
    *,
    store: PromptStore,
    label: str,
    version: int | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Create missing versions (oldest first), then label `version` (default: the latest)."""
    by_name: defaultdict[str, list[PromptSpec]] = defaultdict(list)
    for spec in specs:
        by_name[spec.name].append(spec)

    actions: list[str] = []
    for name, versions in sorted(by_name.items()):
        versions.sort(key=lambda spec: spec.version)
        numbers = [spec.version for spec in versions]
        if len(set(numbers)) != len(numbers):
            raise ValueError(f"{name}: two YAML files share a version number: {numbers}")

        remote = store.remote_fingerprints(name)
        remote_version: dict[int, int | None] = {}  # YAML version -> Langfuse version
        for spec in versions:
            ref, fingerprint = f"{name}@v{spec.version}", fingerprint_of(spec)
            if fingerprint in remote:
                remote_version[spec.version] = remote[fingerprint]
                actions.append(f"{ref}: unchanged (Langfuse version {remote[fingerprint]})")
            elif dry_run:
                remote_version[spec.version] = None
                actions.append(f"{ref}: would create")
            else:
                config = {
                    "yaml_version": spec.version,
                    "fingerprint": fingerprint,
                    "description": spec.description,
                }
                created = store.create(spec, messages=spec.to_langfuse_messages(), config=config)
                remote[fingerprint] = remote_version[spec.version] = created
                actions.append(f"{ref}: created as Langfuse version {created}")

        target = version if version is not None else numbers[-1]
        if target not in remote_version:
            raise ValueError(f"{name}: no v{target} in {PROMPTS_DIR}/ (have {numbers})")
        target_remote = remote_version[target]
        if dry_run or target_remote is None:
            actions.append(f"{name}: would set label {label!r} -> v{target}")
        else:
            store.set_label(name, version=target_remote, label=label)
            actions.append(
                f"{name}: label {label!r} -> v{target} (Langfuse version {target_remote})"
            )
    return actions


class LangfusePromptStore:
    """`PromptStore` over the Langfuse client."""

    def __init__(self, client: Langfuse) -> None:
        self._client = client

    def remote_fingerprints(self, name: str) -> dict[str, int]:
        found: dict[str, int] = {}
        for meta in self._client.api.prompts.list(name=name, limit=100).data:
            if meta.name != name:
                continue
            for number in meta.versions:
                # cache_ttl_seconds=0: always read the server, never a stale local copy.
                prompt = self._client.get_prompt(
                    name, version=number, type="chat", cache_ttl_seconds=0
                )
                fingerprint = (prompt.config or {}).get("fingerprint")
                if isinstance(fingerprint, str):
                    found[fingerprint] = number
        return found

    def create(self, spec: PromptSpec, *, messages: list[dict[str, str]], config: Any) -> int:
        # The SDK types each message as a TypedDict union; ours are plain {role, content}
        # dicts of exactly that shape. list[Any] avoids list invariance without a cast per item.
        chat: list[Any] = [{"role": m["role"], "content": m["content"]} for m in messages]
        prompt = self._client.create_prompt(
            name=spec.name,
            prompt=chat,
            type="chat",
            labels=[],  # labels are moved explicitly by set_label
            config=config,
            commit_message=f"{spec.name}@v{spec.version} from {PROMPTS_DIR}/ (git)",
        )
        return int(prompt.version)

    def set_label(self, name: str, *, version: int, label: str) -> None:
        self._client.update_prompt(name=name, version=version, new_labels=[label])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--label", required=True, help="e.g. staging or production")
    parser.add_argument("--version", type=int, help="YAML version to label (default: latest)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    args = parser.parse_args(argv)

    client = build_langfuse(Settings())
    if client is None:
        print("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are not set (see .env.example)")
        return 2

    specs = [load_prompt_spec(path) for path in sorted(PROMPTS_DIR.glob("*.yaml"))]
    try:
        actions = sync(
            specs,
            store=LangfusePromptStore(client),
            label=args.label,
            version=args.version,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    finally:
        client.shutdown()
    print("\n".join(actions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
