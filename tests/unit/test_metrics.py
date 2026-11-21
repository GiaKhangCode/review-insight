"""Evaluation metrics (SPEC §7.2): hand-computed cases, then cross-checks against sklearn."""

import random

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from evals.metrics import (
    PRF,
    Pair,
    aspect_detection_scores,
    classification_scores,
    cost_per_1k,
    latency_percentiles,
    pair_scores,
)
from review_insight.domain.enums import Aspect, Sentiment, Urgency

A, S = Aspect, Sentiment

# Three reviews, small enough to score by hand:
#   r1: CAMERA is found but with the wrong sentiment.
#   r2: PRICE is right; DESIGN is invented.
#   r3: the gold has no aspect; GENERAL is invented.
GOLD: list[set[Pair]] = [
    {(A.BATTERY, S.POSITIVE), (A.CAMERA, S.NEGATIVE)},
    {(A.PRICE, S.NEUTRAL)},
    set(),
]
PRED: list[set[Pair]] = [
    {(A.BATTERY, S.POSITIVE), (A.CAMERA, S.POSITIVE)},
    {(A.PRICE, S.NEUTRAL), (A.DESIGN, S.POSITIVE)},
    {(A.GENERAL, S.POSITIVE)},
]


# --- PRF ---------------------------------------------------------------------------------


def test_prf_is_derived_from_counts() -> None:
    prf = PRF(tp=2, fp=3, fn=1)

    assert prf.precision == pytest.approx(2 / 5)
    assert prf.recall == pytest.approx(2 / 3)
    assert prf.f1 == pytest.approx(0.5)  # 2tp / (2tp + fp + fn) = 4 / 8
    assert prf.support == 3  # gold positives = tp + fn


def test_prf_with_nothing_to_count_is_zero_not_a_crash() -> None:
    # Same convention as sklearn's zero_division=0.
    prf = PRF(tp=0, fp=0, fn=0)

    assert (prf.precision, prf.recall, prf.f1) == (0.0, 0.0, 0.0)


# --- Aspect detection (is the aspect mentioned? sentiment ignored) -----------------------


def test_aspect_detection_micro_counts_every_decision() -> None:
    scores = aspect_detection_scores(GOLD, PRED)

    # TP: BATTERY, CAMERA, PRICE · FP: DESIGN, GENERAL · FN: none
    assert scores.micro == PRF(tp=3, fp=2, fn=0)
    assert scores.micro.f1 == pytest.approx(0.75)


def test_aspect_detection_per_aspect_lists_all_ten_with_support() -> None:
    scores = aspect_detection_scores(GOLD, PRED)

    assert set(scores.per_aspect) == set(Aspect)
    assert scores.per_aspect[A.CAMERA] == PRF(tp=1, fp=0, fn=0)  # found; sentiment irrelevant
    assert scores.per_aspect[A.DESIGN] == PRF(tp=0, fp=1, fn=0)
    assert scores.per_aspect[A.STORAGE].support == 0


def test_aspect_detection_macro_skips_aspects_absent_from_gold_and_pred() -> None:
    scores = aspect_detection_scores(GOLD, PRED)

    # Present: BATTERY 1, CAMERA 1, PRICE 1, DESIGN 0, GENERAL 0 -> 3/5.
    # The other 5 aspects never appear, so they neither help nor hurt.
    assert scores.macro_f1 == pytest.approx(0.6)


# --- Aspect-sentiment pairs (primary metric) ---------------------------------------------


def test_pair_f1_counts_a_wrong_sentiment_as_one_fp_and_one_fn() -> None:
    scores = pair_scores(GOLD, PRED)

    # TP: (BATTERY,+), (PRICE,0) · FP: (CAMERA,+), (DESIGN,+), (GENERAL,+) · FN: (CAMERA,-)
    assert scores == PRF(tp=2, fp=3, fn=1)
    assert scores.f1 == pytest.approx(0.5)


def test_perfect_prediction_scores_one() -> None:
    assert pair_scores(GOLD[:2], GOLD[:2]).f1 == 1.0
    assert aspect_detection_scores(GOLD[:2], GOLD[:2]).macro_f1 == 1.0


@pytest.mark.parametrize(
    ("gold", "pred", "match"),
    [
        (GOLD, PRED[:2], "same length"),
        ([], [], "empty"),
    ],
)
def test_set_metrics_reject_misaligned_or_empty_input(
    gold: list[set[Pair]], pred: list[set[Pair]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        pair_scores(gold, pred)
    with pytest.raises(ValueError, match=match):
        aspect_detection_scores(gold, pred)


# --- Single-label classification (urgency, routing) --------------------------------------

U = Urgency
LABELS = [U.LOW, U.MEDIUM, U.HIGH]


def test_classification_accuracy_macro_f1_and_confusion_matrix() -> None:
    gold = [U.HIGH, U.LOW, U.LOW, U.MEDIUM]
    pred = [U.HIGH, U.LOW, U.MEDIUM, U.MEDIUM]

    scores = classification_scores(gold, pred, labels=LABELS)

    assert scores.accuracy == pytest.approx(3 / 4)
    # F1 per label: low 2/3, medium 2/3, high 1 -> macro 7/9
    assert scores.macro_f1 == pytest.approx(7 / 9)
    assert scores.per_label[U.LOW] == PRF(tp=1, fp=0, fn=1)
    # Rows = gold, columns = predicted, both in `labels` order.
    assert scores.confusion == [
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]


def test_classification_rejects_a_label_outside_the_label_set() -> None:
    with pytest.raises(ValueError, match="label"):
        classification_scores([U.LOW], ["critical"], labels=LABELS)


def test_classification_rejects_misaligned_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        classification_scores([U.LOW, U.HIGH], [U.LOW], labels=LABELS)


# --- Latency and cost --------------------------------------------------------------------


def test_latency_percentiles_interpolate_linearly() -> None:
    latencies = [100.0 * i for i in range(1, 11)]  # 100, 200, ..., 1000 ms

    p50, p95 = latency_percentiles(latencies)

    assert p50 == pytest.approx(550.0)
    assert p95 == pytest.approx(955.0)


def test_latency_of_a_single_request_is_that_request() -> None:
    assert latency_percentiles([42.0]) == (42.0, 42.0)


def test_latency_percentiles_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        latency_percentiles([])


def test_cost_per_1k_scales_the_run_cost() -> None:
    assert cost_per_1k(total_cost_usd=0.05, n_reviews=50) == pytest.approx(1.0)


def test_cost_per_1k_rejects_zero_reviews() -> None:
    with pytest.raises(ValueError, match="n_reviews"):
        cost_per_1k(total_cost_usd=0.0, n_reviews=0)


# --- Cross-checks: our formulas vs sklearn on random data --------------------------------


def _random_rows(rng: random.Random, n: int) -> list[set[Pair]]:
    return [
        {(aspect, rng.choice(list(Sentiment))) for aspect in Aspect if rng.random() < 0.3}
        for _ in range(n)
    ]


def _binarize(gold: list[set[str]], pred: list[set[str]]) -> tuple[np.ndarray, np.ndarray]:
    # Columns = labels present in gold or pred: the same label set our macro average uses.
    present = sorted(set().union(*gold, *pred))
    mlb = MultiLabelBinarizer(classes=present)
    return mlb.fit_transform(gold), mlb.transform(pred)


@pytest.mark.parametrize("seed", range(20))
def test_set_metrics_match_sklearn(seed: int) -> None:
    rng = random.Random(seed)  # noqa: S311 - test data, not cryptography
    gold, pred = _random_rows(rng, 40), _random_rows(rng, 40)

    aspects_gold = [{a.value for a, _ in row} for row in gold]
    aspects_pred = [{a.value for a, _ in row} for row in pred]
    y_true, y_pred = _binarize(aspects_gold, aspects_pred)
    detection = aspect_detection_scores(gold, pred)
    assert detection.micro.f1 == pytest.approx(
        f1_score(y_true, y_pred, average="micro", zero_division=0)
    )
    assert detection.macro_f1 == pytest.approx(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    pairs_gold = [{f"{a}#{s}" for a, s in row} for row in gold]
    pairs_pred = [{f"{a}#{s}" for a, s in row} for row in pred]
    y_true, y_pred = _binarize(pairs_gold, pairs_pred)
    assert pair_scores(gold, pred).f1 == pytest.approx(
        f1_score(y_true, y_pred, average="micro", zero_division=0)
    )


@pytest.mark.parametrize("seed", range(20))
def test_classification_matches_sklearn(seed: int) -> None:
    rng = random.Random(seed)  # noqa: S311 - test data, not cryptography
    gold = [rng.choice(LABELS) for _ in range(30)]
    pred = [rng.choice(LABELS) for _ in range(30)]

    scores = classification_scores(gold, pred, labels=LABELS)

    assert scores.accuracy == pytest.approx(accuracy_score(gold, pred))
    assert scores.macro_f1 == pytest.approx(f1_score(gold, pred, average="macro", zero_division=0))
    assert scores.confusion == confusion_matrix(gold, pred, labels=LABELS).tolist()


def test_latency_matches_numpy() -> None:
    rng = random.Random(0)  # noqa: S311 - test data, not cryptography
    latencies = [rng.uniform(200, 3000) for _ in range(137)]

    p50, p95 = latency_percentiles(latencies)

    assert p50 == pytest.approx(np.percentile(latencies, 50))
    assert p95 == pytest.approx(np.percentile(latencies, 95))
