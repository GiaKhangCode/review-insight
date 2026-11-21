"""Closed label sets used by the LLM output, the gold data, the DB and the API.

`StrEnum` members are real strings: `Sentiment.NEGATIVE == "negative"`, they serialize
to plain JSON strings, and Pydantic rejects any value outside the set.
"""

from enum import StrEnum


class Aspect(StrEnum):
    """Product aspects, identical to the UIT-ViSFD label names (SPEC §4.1)."""

    SCREEN = "SCREEN"
    CAMERA = "CAMERA"
    FEATURES = "FEATURES"
    BATTERY = "BATTERY"
    PERFORMANCE = "PERFORMANCE"
    STORAGE = "STORAGE"
    DESIGN = "DESIGN"
    PRICE = "PRICE"
    GENERAL = "GENERAL"
    SER_ACC = "SER&ACC"  # service & accessories; "&" is not valid in a Python identifier


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Urgency(StrEnum):
    """How fast a human must act (rubric: SPEC §4.2)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Route(StrEnum):
    """Which team should handle the review (rubric: SPEC §4.2)."""

    PRODUCT = "product"
    CUSTOMER_SERVICE = "customer_service"
    SALES_PRICING = "sales_pricing"
    NONE = "none"
