"""Versioned prompt templates: loading, validation, safe rendering, and Langfuse round trips.

Git (prompts/*.yaml) is the source of truth; Langfuse serves a chosen version by label at
runtime. A prompt fetched from Langfuse passes the same validation as a YAML file, and any
problem (Langfuse down, a broken remote edit) falls back to the local YAML.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import yaml
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

logger = logging.getLogger(__name__)

TEMPLATE_VARIABLES = frozenset({"review"})
# LangChain templates write variables as {review}; Langfuse (mustache) as {{review}}.
_LANGCHAIN_VAR, _MUSTACHE_VAR = "{review}", "{{review}}"

# Anything that could pass for the <review> / </review> delimiter, in any case or spacing.
# Only its "<" is escaped: the rest of the customer's text stays exactly as written.
_DELIMITER_LOOKALIKE = re.compile(r"<(?=\s*/?\s*review\b)", re.IGNORECASE)


def sanitize_review(text: str) -> str:
    """Neutralize delimiter look-alikes so a review cannot close its own <review> block."""
    return _DELIMITER_LOOKALIKE.sub("&lt;", text)


def _build_template(system: str, user: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", system), ("human", user)])


class PromptSpec(BaseModel):
    """The YAML file, validated. A bad prompt fails at load time, not mid-eval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    version: PositiveInt
    description: str
    system: str
    user: str

    @model_validator(mode="after")
    def _variables_are_exactly_the_expected_ones(self) -> Self:
        found = set(_build_template(self.system, self.user).input_variables)
        if found != TEMPLATE_VARIABLES:
            raise ValueError(
                f"template variables must be {sorted(TEMPLATE_VARIABLES)}, found {sorted(found)} "
                "(literal braces must be doubled: {{ }})"
            )
        return self

    def to_langfuse_messages(self) -> list[dict[str, str]]:
        """Chat messages in Langfuse's template syntax."""
        if "{{" in self.system + self.user:
            # A literal "{{" would become ambiguous with mustache variables.
            raise ValueError("literal double braces cannot be converted to a Langfuse prompt")
        return [
            {"role": "system", "content": self.system.replace(_LANGCHAIN_VAR, _MUSTACHE_VAR)},
            {"role": "user", "content": self.user.replace(_LANGCHAIN_VAR, _MUSTACHE_VAR)},
        ]

    @classmethod
    def from_langfuse(cls, name: str, messages: list[dict[str, str]], config: Any) -> Self:
        """Rebuild (and re-validate) a spec from a Langfuse chat prompt and its config."""
        if [message.get("role") for message in messages] != ["system", "user"]:
            raise ValueError("expected exactly a system message and a user message")
        system, user = (m["content"].replace(_MUSTACHE_VAR, _LANGCHAIN_VAR) for m in messages)
        return cls(
            name=name,
            version=config["yaml_version"],  # ours, not Langfuse's auto-incremented number
            description=config.get("description", ""),
            system=system,
            user=user,
        )


@dataclass(frozen=True, slots=True)
class PromptBundle:
    name: str
    version: int  # the YAML version (ours)
    fingerprint: str  # hash of the prompt text: changes on any edit, even without a new version
    template: ChatPromptTemplate
    source: Literal["local", "langfuse"] = "local"
    langfuse_version: int | None = None  # Langfuse's own version number, when served from it
    langfuse_prompt: object | None = None  # passed to generations to link them to the prompt

    @property
    def ref(self) -> str:
        return f"{self.name}@v{self.version}"

    def render(self, review: str) -> list[BaseMessage]:
        # The review is passed as a variable value, so braces in it are never interpreted.
        return self.template.format_messages(review=sanitize_review(review))


def fingerprint_of(spec: PromptSpec) -> str:
    # "\0" separates the parts so that moving text between system and user changes the hash.
    return hashlib.sha256(f"{spec.system}\0{spec.user}".encode()).hexdigest()[:12]


def _bundle(spec: PromptSpec, **extra: Any) -> PromptBundle:
    return PromptBundle(
        name=spec.name,
        version=spec.version,
        fingerprint=fingerprint_of(spec),
        template=_build_template(spec.system, spec.user),
        **extra,
    )


def load_prompt_spec(path: Path) -> PromptSpec:
    return PromptSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_prompt(path: Path) -> PromptBundle:
    return _bundle(load_prompt_spec(path))


class PromptRegistry(Protocol):
    """The slice of the Langfuse client used here (keeps this module free of the SDK)."""

    def get_prompt(
        self,
        name: str,
        *,
        label: str | None = None,
        type: Literal["chat", "text"] = "text",  # shadows the builtin: Langfuse's name for it
        fetch_timeout_seconds: int | None = None,
        max_retries: int | None = None,
    ) -> Any: ...


def load_prompt_by_label(
    name: str, *, label: str, registry: PromptRegistry | None, fallback: Path
) -> PromptBundle:
    """The prompt Langfuse serves under `label`, or the local YAML if that is not possible."""
    if registry is None:
        return load_prompt(fallback)
    try:
        remote = registry.get_prompt(
            name, label=label, type="chat", fetch_timeout_seconds=5, max_retries=1
        )
        spec = PromptSpec.from_langfuse(name, remote.prompt, remote.config)
        return _bundle(
            spec, source="langfuse", langfuse_version=remote.version, langfuse_prompt=remote
        )
    except Exception as exc:  # any failure must degrade to the local prompt, never crash
        logger.warning(
            "could not use Langfuse prompt %r (label %r): %s; falling back to %s",
            name,
            label,
            exc,
            fallback,
        )
        return load_prompt(fallback)
