"""Evidence grounding: an evidence span must be text the customer actually wrote.

Tolerated (formatting only): Unicode composition (NFC vs NFD), whitespace/line breaks, casing.
Rejected (wording): paraphrases, translations, invented quotes.
"""

import re
import unicodedata

from review_insight.domain.schemas import AnalysisResult, AspectSentiment

_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Canonical form of a text: NFC-composed, single spaces, trimmed.

    Also used for cache keys (T14), so it must stay deterministic and idempotent.
    """
    composed = unicodedata.normalize("NFC", text)
    return _WHITESPACE_RUN.sub(" ", composed).strip()


def _match_key(text: str) -> str:
    # casefold() is a stronger, Unicode-aware lower(): the right tool for caseless matching.
    return normalize_text(text).casefold()


class UngroundedEvidenceError(ValueError):
    """Raised when one or more evidence spans do not occur in the source review."""

    def __init__(self, ungrounded: list[AspectSentiment]) -> None:
        self.ungrounded = ungrounded
        details = "; ".join(f'{item.aspect.value}: "{item.evidence}"' for item in ungrounded)
        # Worded as an instruction: it is sent back to the LLM in the repair retry (T8).
        super().__init__(
            f"evidence not found in the review ({details}). "
            "Each evidence must be copied verbatim from the review text, without rewording."
        )


def validate_grounded(result: AnalysisResult, *, source_text: str) -> AnalysisResult:
    """Return `result` unchanged if every evidence span is grounded, else raise."""
    haystack = _match_key(source_text)
    ungrounded = [item for item in result.aspects if _match_key(item.evidence) not in haystack]
    if ungrounded:
        raise UngroundedEvidenceError(ungrounded)
    return result
