"""TF-IDF + Logistic Regression baseline (SPEC §7.1, config B), on a tiny synthetic corpus."""

import unicodedata

import pytest

from evals.baseline_tfidf import TfidfBaseline, tune_c
from evals.metrics import Pair
from review_insight.domain.enums import Aspect, Sentiment

A, S = Aspect, Sentiment

# Each pattern repeated, so even a heavily regularized model can learn it.
_PATTERNS: list[tuple[str, set[Pair]]] = [
    ("pin trâu dùng cả ngày", {(A.BATTERY, S.POSITIVE)}),
    ("pin yếu sạc hoài", {(A.BATTERY, S.NEGATIVE)}),
    ("camera chụp đẹp", {(A.CAMERA, S.POSITIVE)}),
    ("shop giao hàng chậm", {(A.SER_ACC, S.NEGATIVE)}),
    ("pin trâu camera chụp đẹp", {(A.BATTERY, S.POSITIVE), (A.CAMERA, S.POSITIVE)}),
    ("máy ổn", set()),
]
TEXTS = [text for text, _ in _PATTERNS] * 5
LABELS = [labels for _, labels in _PATTERNS] * 5


@pytest.fixture(scope="module")
def model() -> TfidfBaseline:
    return TfidfBaseline(c=10.0, seed=42).fit(TEXTS, LABELS)


def test_learns_aspect_and_sentiment_from_ngrams(model: TfidfBaseline) -> None:
    pred = model.predict(["pin trâu lắm", "pin yếu quá", "giao hàng chậm ghê"])

    assert pred == [
        frozenset({(A.BATTERY, S.POSITIVE)}),
        frozenset({(A.BATTERY, S.NEGATIVE)}),
        frozenset({(A.SER_ACC, S.NEGATIVE)}),
    ]


def test_predicts_several_aspects_in_one_review(model: TfidfBaseline) -> None:
    [pred] = model.predict(["pin trâu camera chụp đẹp"])

    assert pred == {(A.BATTERY, S.POSITIVE), (A.CAMERA, S.POSITIVE)}


def test_predicts_at_most_one_sentiment_per_aspect(model: TfidfBaseline) -> None:
    # One 4-way classifier per aspect: absent / positive / negative / neutral.
    for pred in model.predict(TEXTS):
        aspects = [aspect for aspect, _ in pred]
        assert len(aspects) == len(set(aspects))


def test_aspect_never_seen_in_training_is_never_predicted(model: TfidfBaseline) -> None:
    # STORAGE has a single class ("absent") in training; LogisticRegression cannot
    # fit on one class, so that aspect falls back to a constant prediction.
    preds = model.predict(["bộ nhớ 128GB", *TEXTS])

    assert all(aspect is not A.STORAGE for pred in preds for aspect, _ in pred)


def test_text_is_normalized_before_vectorizing(model: TfidfBaseline) -> None:
    # Uppercase and NFD (decomposed accents, as some keyboards emit) must not matter.
    nfd = unicodedata.normalize("NFD", "PIN TRÂU LẮM")

    assert model.predict([nfd]) == model.predict(["pin trâu lắm"])


def test_training_is_deterministic() -> None:
    first = TfidfBaseline(c=1.0, seed=7).fit(TEXTS, LABELS).predict(TEXTS)
    second = TfidfBaseline(c=1.0, seed=7).fit(TEXTS, LABELS).predict(TEXTS)

    assert first == second


def test_predict_before_fit_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        TfidfBaseline().predict(["pin trâu"])


def test_fit_rejects_misaligned_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        TfidfBaseline().fit(TEXTS, LABELS[:-1])


# --- Hyperparameter tuning on dev --------------------------------------------------------


def test_tune_c_picks_the_c_with_the_best_dev_pair_f1() -> None:
    result = tune_c(
        train_texts=TEXTS,
        train_labels=LABELS,
        dev_texts=TEXTS[:6],
        dev_labels=LABELS[:6],
        c_grid=[0.01, 1.0, 10.0],
        seed=42,
    )

    assert set(result.dev_pair_f1) == {0.01, 1.0, 10.0}
    assert result.dev_pair_f1[result.best_c] == max(result.dev_pair_f1.values())
    # The returned model is the one fitted with best_c, ready to score the test split.
    assert result.model.c == result.best_c


def test_tune_c_breaks_ties_toward_the_smaller_c() -> None:
    # On this easy corpus several C values score 1.0; prefer the simplest model
    # (stronger regularization), which generalizes better.
    result = tune_c(
        train_texts=TEXTS,
        train_labels=LABELS,
        dev_texts=TEXTS[:6],
        dev_labels=LABELS[:6],
        c_grid=[100.0, 10.0],
        seed=42,
    )

    assert result.best_c == 10.0


def test_tune_c_rejects_an_empty_grid() -> None:
    with pytest.raises(ValueError, match="c_grid"):
        tune_c(
            train_texts=TEXTS,
            train_labels=LABELS,
            dev_texts=TEXTS,
            dev_labels=LABELS,
            c_grid=[],
            seed=42,
        )
