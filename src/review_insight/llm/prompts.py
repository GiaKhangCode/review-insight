"""Versioned prompt templates (prompts/*.yaml): loading, validation and safe rendering."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import yaml
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

TEMPLATE_VARIABLES = frozenset({"review"})

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


@dataclass(frozen=True, slots=True)
class PromptBundle:
    name: str
    version: int
    fingerprint: str  # hash of the prompt text: changes on any edit, even without a new version
    template: ChatPromptTemplate

    @property
    def ref(self) -> str:
        return f"{self.name}@v{self.version}"

    def render(self, review: str) -> list[BaseMessage]:
        # The review is passed as a variable value, so braces in it are never interpreted.
        return self.template.format_messages(review=sanitize_review(review))


def load_prompt(path: Path) -> PromptBundle:
    spec = PromptSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    # "\0" separates the parts so that moving text between system and user changes the hash.
    digest = hashlib.sha256(f"{spec.system}\0{spec.user}".encode()).hexdigest()
    return PromptBundle(
        name=spec.name,
        version=spec.version,
        fingerprint=digest[:12],
        template=_build_template(spec.system, spec.user),
    )
