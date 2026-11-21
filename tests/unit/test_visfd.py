"""UIT-ViSFD loader: label parsing, CSV loading and fixed evaluation samples."""

import csv
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from review_insight.data.visfd import (
    SPLIT_SIZES,
    AspectLabel,
    GoldReview,
    load_split,
    parse_label,
    parse_posted_date,
    stratified_sample,
)
from review_insight.domain.enums import Aspect, Sentiment

# --- parse_label --------------------------------------------------------------


def test_parse_label_maps_dataset_tokens_to_domain_enums() -> None:
    labels = parse_label("{BATTERY#Positive};{SER&ACC#Negative};")

    assert labels == frozenset(
        {
            AspectLabel(Aspect.BATTERY, Sentiment.POSITIVE),
            AspectLabel(Aspect.SER_ACC, Sentiment.NEGATIVE),
        }
    )


def test_others_token_is_ignored_even_when_mixed_with_aspects() -> None:
    # In the real data OTHERS co-occurs with aspects (~20% of rows):
    # it means "also talks about something outside the 10 aspects", not "no aspect".
    assert parse_label("{BATTERY#Negative};{OTHERS};") == frozenset(
        {AspectLabel(Aspect.BATTERY, Sentiment.NEGATIVE)}
    )
    assert parse_label("{OTHERS};") == frozenset()


@pytest.mark.parametrize(
    "raw",
    [
        "",  # every row must carry at least one token
        "{WIFI#Positive};",  # unknown aspect
        "{CAMERA#Great};",  # unknown polarity
        "{CAMERA};",  # missing polarity
        "{CAMERA#Positive};{CAMERA#Negative};",  # one sentiment per aspect
    ],
)
def test_malformed_labels_raise_instead_of_being_skipped(raw: str) -> None:
    with pytest.raises(ValueError, match="label"):
        parse_label(raw)


# --- parse_posted_date --------------------------------------------------------


def test_absolute_dates_are_day_first() -> None:
    assert parse_posted_date("9/7/2020") == date(2020, 7, 9)


@pytest.mark.parametrize("raw", ["2 tuần trước", "6 ngày trước", "", "31/02/2020"])
def test_relative_or_invalid_dates_become_none(raw: str) -> None:
    # ~11% of rows have relative dates ("2 weeks ago") with no anchor: unusable.
    assert parse_posted_date(raw) is None


# --- load_split ---------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # the real files carry a BOM
        writer = csv.DictWriter(f, fieldnames=["index", "comment", "n_star", "date_time", "label"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_split_reads_rows_into_gold_reviews(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "Dev.csv",
        [
            {
                "index": "0",
                "comment": "Pin trâu\nmáy đẹp",
                "n_star": "5",
                "date_time": "14/12/2019",
                "label": "{BATTERY#Positive};{DESIGN#Positive};",
            },
            {
                "index": "1",
                "comment": "Có tai nghe không shop?",
                "n_star": "3",
                "date_time": "2 tuần trước",
                "label": "{OTHERS};",
            },
        ],
    )

    reviews = load_split("dev", raw_dir=tmp_path)

    assert reviews[0] == GoldReview(
        split="dev",
        index=0,
        text="Pin trâu\nmáy đẹp",  # original text kept as-is; normalization happens downstream
        n_star=5,
        posted_at=date(2019, 12, 14),
        labels=frozenset(
            {
                AspectLabel(Aspect.BATTERY, Sentiment.POSITIVE),
                AspectLabel(Aspect.DESIGN, Sentiment.POSITIVE),
            }
        ),
    )
    assert reviews[1].labels == frozenset()
    assert reviews[1].posted_at is None


def test_load_split_reports_the_offending_row(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "Test.csv",
        [
            {
                "index": "7",
                "comment": "x",
                "n_star": "1",
                "date_time": "",
                "label": "{WIFI#Positive};",
            }
        ],
    )

    with pytest.raises(ValueError, match=r"Test\.csv.*index 7"):
        load_split("test", raw_dir=tmp_path)


# --- stratified_sample --------------------------------------------------------


def _reviews_with_stars(stars: list[int]) -> list[GoldReview]:
    return [
        GoldReview(
            split="test", index=i, text=f"r{i}", n_star=s, posted_at=None, labels=frozenset()
        )
        for i, s in enumerate(stars)
    ]


def test_stratified_sample_preserves_star_distribution() -> None:
    reviews = _reviews_with_stars([5] * 600 + [1] * 200 + [3] * 200)  # 60% / 20% / 20%

    picked = stratified_sample(reviews, n=100, seed=42)

    by_index = {r.index: r for r in reviews}
    assert len(picked) == 100
    assert Counter(by_index[i].n_star for i in picked) == {5: 60, 1: 20, 3: 20}


def test_stratified_sample_is_deterministic_sorted_and_unique() -> None:
    reviews = _reviews_with_stars([1, 2, 3, 4, 5] * 40)

    first = stratified_sample(reviews, n=50, seed=42)

    assert first == stratified_sample(reviews, n=50, seed=42)  # same seed -> same sample
    assert first != stratified_sample(reviews, n=50, seed=7)  # seed actually matters
    assert first == sorted(set(first))


def test_stratified_sample_rejects_n_larger_than_population() -> None:
    with pytest.raises(ValueError, match="n=11"):
        stratified_sample(_reviews_with_stars([5] * 10), n=11, seed=42)


# --- real data (skipped until scripts/download_data.py has been run) ----------

REAL_RAW_DIR = Path("data/raw/UIT-ViSFD")


@pytest.mark.skipif(not REAL_RAW_DIR.exists(), reason="dataset not downloaded")
@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_real_splits_load_completely(split: str) -> None:
    reviews = load_split(split, raw_dir=REAL_RAW_DIR)  # type: ignore[arg-type]

    assert len(reviews) == SPLIT_SIZES[split]  # type: ignore[index]
