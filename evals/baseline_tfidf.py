"""Supervised baseline (SPEC §7.1, config B): TF-IDF n-grams + one classifier per aspect.

Each aspect gets its own 4-way Logistic Regression over {absent, positive, negative, neutral},
so a prediction can never hold two sentiments for one aspect. The LLM configs are judged
against this: it is cheap, fast and trained on the full train split.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Self

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from evals.metrics import Pair, pair_scores
from review_insight.domain.enums import Aspect, Sentiment
from review_insight.domain.grounding import normalize_text

_ABSENT = "absent"

# Default sklearn pattern needs 2+ characters per token; Vietnamese has meaningful
# one-letter syllables ("ổ", "ở"), so accept any run of word characters.
_SYLLABLE_PATTERN = r"(?u)\b\w+\b"


def _preprocess(text: str) -> str:
    """Same normalization as the rest of the system (NFC, collapsed whitespace), lowercased."""
    return normalize_text(text).lower()


class TfidfBaseline:
    def __init__(
        self, *, c: float = 1.0, ngram_max: int = 2, min_df: int = 1, seed: int = 42
    ) -> None:
        self.c = c
        self.ngram_max = ngram_max
        self.min_df = min_df
        self.seed = seed
        self._vectorizer: TfidfVectorizer | None = None
        # A fitted classifier, or a constant label when training data had a single class.
        self._classifiers: dict[Aspect, LogisticRegression | str] = {}

    def fit(self, texts: Sequence[str], labels: Sequence[Collection[Pair]]) -> Self:
        if len(texts) != len(labels):
            raise ValueError(
                f"texts and labels must have the same length, got {len(texts)} and {len(labels)}"
            )
        vectorizer = TfidfVectorizer(
            preprocessor=_preprocess,
            token_pattern=_SYLLABLE_PATTERN,
            ngram_range=(1, self.ngram_max),
            min_df=self.min_df,
            sublinear_tf=True,  # 1 + log(tf): the 5th "pin" adds less than the 1st
        )
        features = vectorizer.fit_transform(texts)

        sentiment_by_aspect = [dict(row) for row in labels]
        classifiers: dict[Aspect, LogisticRegression | str] = {}
        for aspect in Aspect:
            targets = [str(row.get(aspect, _ABSENT)) for row in sentiment_by_aspect]
            if len(set(targets)) == 1:
                # LogisticRegression cannot fit one class; always predict that class.
                classifiers[aspect] = targets[0]
            else:
                classifiers[aspect] = LogisticRegression(
                    C=self.c, max_iter=1000, random_state=self.seed
                ).fit(features, targets)

        self._vectorizer, self._classifiers = vectorizer, classifiers
        return self

    def predict(self, texts: Sequence[str]) -> list[frozenset[Pair]]:
        if self._vectorizer is None:
            raise RuntimeError("TfidfBaseline must be fit() before predict()")
        features = self._vectorizer.transform(texts)

        predicted: list[set[Pair]] = [set() for _ in texts]
        for aspect, classifier in self._classifiers.items():
            column = (
                [classifier] * len(texts)
                if isinstance(classifier, str)
                else classifier.predict(features)
            )
            for row, label in zip(predicted, column, strict=True):
                if label != _ABSENT:
                    row.add((aspect, Sentiment(label)))
        return [frozenset(row) for row in predicted]


# --- Hyperparameter tuning on dev --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TuningResult:
    best_c: float
    dev_pair_f1: dict[float, float]  # every C tried -> its dev pair F1
    model: TfidfBaseline  # fitted with best_c


def tune_c(
    *,
    train_texts: Sequence[str],
    train_labels: Sequence[Collection[Pair]],
    dev_texts: Sequence[str],
    dev_labels: Sequence[Collection[Pair]],
    c_grid: Sequence[float],
    seed: int,
    ngram_max: int = 2,
    min_df: int = 1,
) -> TuningResult:
    """Fit on train once per C; keep the C with the best dev pair F1. Test is never seen here."""
    if not c_grid:
        raise ValueError("c_grid must contain at least one value")

    models: dict[float, TfidfBaseline] = {}
    scores: dict[float, float] = {}
    for c in sorted(c_grid):
        models[c] = TfidfBaseline(c=c, ngram_max=ngram_max, min_df=min_df, seed=seed)
        models[c].fit(train_texts, train_labels)
        scores[c] = pair_scores(dev_labels, models[c].predict(dev_texts)).f1

    # Highest F1 wins; on a tie, the smaller C (stronger regularization, simpler model).
    best_c = max(scores, key=lambda c: (scores[c], -c))
    return TuningResult(best_c=best_c, dev_pair_f1=scores, model=models[best_c])
