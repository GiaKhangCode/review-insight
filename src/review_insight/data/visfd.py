"""Adapter for UIT-ViSFD (Phan et al., 2021): CSV files -> `GoldReview` domain objects.

Dataset quirks handled here and nowhere else (found by profiling the real files):
- label syntax `{ASPECT#Polarity};...`, polarity capitalized (`Positive`);
- `{OTHERS}` = "also talks about something outside the 10 aspects"; it co-occurs with real
  aspects in ~20% of rows, so it is ignored (a row with only OTHERS has no aspect labels);
- dates are day-first (`9/7/2020`); ~11% are relative (`2 tuần trước`) and become None;
- files start with a UTF-8 BOM.
"""

import csv
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal, NamedTuple

from review_insight.domain.enums import Aspect, Sentiment

Split = Literal["train", "dev", "test"]

SPLIT_FILES: dict[Split, str] = {"train": "Train.csv", "dev": "Dev.csv", "test": "Test.csv"}
SPLIT_SIZES: dict[Split, int] = {"train": 7786, "dev": 1112, "test": 2224}

_TOKEN = re.compile(r"\{([^{}]*)\}")
_OTHERS = "OTHERS"
_POLARITY = {
    "Positive": Sentiment.POSITIVE,
    "Negative": Sentiment.NEGATIVE,
    "Neutral": Sentiment.NEUTRAL,
}


class AspectLabel(NamedTuple):
    """One gold (aspect, sentiment) pair. Hashable, so labels compare as sets in metrics."""

    aspect: Aspect
    sentiment: Sentiment


@dataclass(frozen=True, slots=True)
class GoldReview:
    """A review with its human annotation. `text` is kept exactly as in the source file."""

    split: Split
    index: int
    text: str
    n_star: int
    posted_at: date | None
    labels: frozenset[AspectLabel]


def parse_label(raw: str) -> frozenset[AspectLabel]:
    """Parse `{BATTERY#Positive};{OTHERS};` into domain labels. Malformed input raises."""
    tokens = _TOKEN.findall(raw)
    if not tokens:
        raise ValueError(f"empty or unparseable label: {raw!r}")

    labels: dict[Aspect, Sentiment] = {}
    for token in tokens:
        if token == _OTHERS:
            continue
        aspect_name, _, polarity = token.partition("#")
        try:
            aspect, sentiment = Aspect(aspect_name), _POLARITY[polarity]
        except (ValueError, KeyError):
            raise ValueError(f"unknown label token {token!r} in {raw!r}") from None
        if aspect in labels:
            raise ValueError(f"duplicate aspect {aspect.value} in label {raw!r}")
        labels[aspect] = sentiment

    return frozenset(AspectLabel(aspect, sentiment) for aspect, sentiment in labels.items())


def parse_posted_date(raw: str) -> date | None:
    """Day-first absolute date, or None for relative/invalid values."""
    try:
        return datetime.strptime(raw.strip(), "%d/%m/%Y").date()
    except ValueError:
        return None


def load_split(split: Split, *, raw_dir: Path) -> list[GoldReview]:
    """Load one split. Any bad row fails loudly, naming the file and row index."""
    path = raw_dir / SPLIT_FILES[split]
    reviews: list[GoldReview] = []
    # utf-8-sig strips the BOM; otherwise the first column would be named "﻿index".
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                reviews.append(
                    GoldReview(
                        split=split,
                        index=int(row["index"]),
                        text=row["comment"],
                        n_star=int(row["n_star"]),
                        posted_at=parse_posted_date(row["date_time"]),
                        labels=parse_label(row["label"]),
                    )
                )
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{path.name}: index {row.get('index')}: {exc}") from exc
    return reviews


def stratified_sample(reviews: Sequence[GoldReview], *, n: int, seed: int) -> list[int]:
    """Pick `n` review indices, keeping the star-rating distribution of `reviews`.

    Proportional allocation per star rating, rounded with the largest-remainder method so
    the quotas sum exactly to `n`. Deterministic for a given seed. Returns sorted indices.
    """
    if not 0 < n <= len(reviews):
        raise ValueError(f"cannot sample n={n} from {len(reviews)} reviews")

    strata: dict[int, list[int]] = defaultdict(list)
    for review in reviews:
        strata[review.n_star].append(review.index)

    exact = {star: n * len(members) / len(reviews) for star, members in strata.items()}
    quota = {star: int(value) for star, value in exact.items()}
    leftover = n - sum(quota.values())
    by_remainder = sorted(exact, key=lambda star: (exact[star] - quota[star], star), reverse=True)
    for star in by_remainder[:leftover]:
        quota[star] += 1

    # Not for security: a seeded PRNG is exactly what makes the sample reproducible.
    rng = random.Random(seed)  # noqa: S311
    picked: list[int] = []
    for star in sorted(strata):
        picked.extend(rng.sample(sorted(strata[star]), quota[star]))
    return sorted(picked)
