"""Evaluation metrics (SPEC §7.2), written from the definitions and cross-checked against sklearn.

Conventions, identical to sklearn's so the numbers are comparable:
- a ratio with a zero denominator is 0.0 (``zero_division=0``);
- a macro average runs over the labels present in gold *or* predictions. A label that
  appears in neither cannot be scored, so it is left out rather than counted as F1 = 0.
"""

from __future__ import annotations

import statistics
from collections.abc import Collection, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from review_insight.domain.enums import Aspect, Sentiment

Pair = tuple[Aspect, Sentiment]
"""One (aspect, sentiment) decision. A review's labels are a set of pairs."""

H = TypeVar("H", bound=Hashable)
L = TypeVar("L", bound=str)


@dataclass(frozen=True, slots=True)
class PRF:
    """Confusion counts for one label (or summed over labels); P/R/F1 are derived from them."""

    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        # Equal to 2PR / (P + R), but without the intermediate rounding.
        return _ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    @property
    def support(self) -> int:
        """Number of gold positives."""
        return self.tp + self.fn

    def __add__(self, other: PRF) -> PRF:
        return PRF(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


_ZERO = PRF(0, 0, 0)


@dataclass(frozen=True, slots=True)
class AspectDetectionScores:
    per_aspect: dict[Aspect, PRF]  # all ten aspects, zero-support ones included
    micro: PRF
    macro_f1: float


@dataclass(frozen=True, slots=True)
class ClassificationScores(Generic[L]):
    labels: tuple[L, ...]
    accuracy: float
    macro_f1: float
    per_label: dict[L, PRF]
    confusion: list[list[int]]  # rows = gold, columns = predicted, both in `labels` order


# --- Multi-label (set) metrics -----------------------------------------------------------


def aspect_detection_scores(
    gold: Sequence[Collection[Pair]], pred: Sequence[Collection[Pair]]
) -> AspectDetectionScores:
    """Is each aspect mentioned? Sentiment is ignored."""
    _check_aligned(gold, pred)
    counts = _per_label_counts(
        [{aspect for aspect, _ in row} for row in gold],
        [{aspect for aspect, _ in row} for row in pred],
    )
    return AspectDetectionScores(
        per_aspect={aspect: counts.get(aspect, _ZERO) for aspect in Aspect},
        micro=sum(counts.values(), _ZERO),
        macro_f1=_macro_f1(counts.values()),
    )


def pair_scores(gold: Sequence[Collection[Pair]], pred: Sequence[Collection[Pair]]) -> PRF:
    """Micro P/R/F1 on exact (aspect, sentiment) matches. The primary metric.

    A right aspect with the wrong sentiment is one FP plus one FN.
    """
    _check_aligned(gold, pred)
    counts = _per_label_counts([set(row) for row in gold], [set(row) for row in pred])
    return sum(counts.values(), _ZERO)


def _per_label_counts(gold: Sequence[set[H]], pred: Sequence[set[H]]) -> dict[H, PRF]:
    """Per-label TP/FP/FN. Only labels seen in gold or pred get an entry."""
    counts: dict[H, PRF] = {}
    for gold_row, pred_row in zip(gold, pred, strict=True):
        for label in gold_row & pred_row:
            counts[label] = counts.get(label, _ZERO) + PRF(1, 0, 0)
        for label in pred_row - gold_row:
            counts[label] = counts.get(label, _ZERO) + PRF(0, 1, 0)
        for label in gold_row - pred_row:
            counts[label] = counts.get(label, _ZERO) + PRF(0, 0, 1)
    return counts


# --- Single-label classification (urgency, routing) --------------------------------------


def classification_scores(
    gold: Sequence[L], pred: Sequence[L], *, labels: Sequence[L]
) -> ClassificationScores[L]:
    _check_aligned(gold, pred)
    index = {label: i for i, label in enumerate(labels)}
    unknown = {value for value in (*gold, *pred) if value not in index}
    if unknown:
        raise ValueError(f"label(s) outside the label set {list(labels)}: {sorted(unknown)}")

    confusion = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred, strict=True):
        confusion[index[g]][index[p]] += 1

    per_label: dict[L, PRF] = {}
    for i, label in enumerate(labels):
        tp = confusion[i][i]
        predicted = sum(row[i] for row in confusion)  # column total
        actual = sum(confusion[i])  # row total
        per_label[label] = PRF(tp=tp, fp=predicted - tp, fn=actual - tp)

    correct = sum(confusion[i][i] for i in range(len(labels)))
    present = [prf for prf in per_label.values() if prf.tp + prf.fp + prf.fn > 0]
    return ClassificationScores(
        labels=tuple(labels),
        accuracy=correct / len(gold),
        macro_f1=_macro_f1(present),
        per_label=per_label,
        confusion=confusion,
    )


# --- Latency and cost --------------------------------------------------------------------


def latency_percentiles(latencies_ms: Sequence[float]) -> tuple[float, float]:
    """(p50, p95) with linear interpolation between ranks, the same as numpy's default."""
    if not latencies_ms:
        raise ValueError("cannot take percentiles of an empty latency list")
    if len(latencies_ms) == 1:
        return latencies_ms[0], latencies_ms[0]
    cuts = statistics.quantiles(latencies_ms, n=100, method="inclusive")  # cuts[k-1] = p_k
    return cuts[49], cuts[94]


def cost_per_1k(*, total_cost_usd: float, n_reviews: int) -> float:
    """Average cost of 1,000 reviews, from the cost of a run over `n_reviews`."""
    if n_reviews <= 0:
        raise ValueError(f"n_reviews must be positive, got {n_reviews}")
    return total_cost_usd / n_reviews * 1000


# --- Helpers -----------------------------------------------------------------------------


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _macro_f1(scores: Iterable[PRF]) -> float:
    f1s = [prf.f1 for prf in scores]
    return statistics.fmean(f1s) if f1s else 0.0


def _check_aligned(gold: Sequence[object], pred: Sequence[object]) -> None:
    if len(gold) != len(pred):
        raise ValueError(
            f"gold and pred must have the same length, got {len(gold)} and {len(pred)}"
        )
    if not gold:
        raise ValueError("cannot score an empty evaluation set")
