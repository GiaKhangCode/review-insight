"""Eval runner (SPEC §7.4): config loading, eval-set selection, leakage guard, results file."""

import csv
import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from evals.run import RunConfig, load_config, run, score, select_reviews

REPO_ROOT = Path(__file__).resolve().parents[2]

# (comment, label) in the dataset's own syntax, so the real loader is exercised.
_ROWS = [
    ("pin trâu dùng cả ngày", "{BATTERY#Positive};"),
    ("pin yếu sạc hoài", "{BATTERY#Negative};"),
    ("camera chụp đẹp", "{CAMERA#Positive};"),
    ("shop giao hàng chậm", "{SER&ACC#Negative};"),
    ("máy ổn", "{OTHERS};"),
]


def _write_split(path: Path, rows: list[tuple[str, str]], *, repeat: int) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # BOM, like the real files
        writer = csv.writer(f)
        writer.writerow(["index", "comment", "n_star", "date_time", "label"])
        for i, (comment, label) in enumerate(rows * repeat):
            writer.writerow([i, comment, 5, "9/7/2020", label])


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "raw"
    directory.mkdir()
    _write_split(directory / "Train.csv", _ROWS, repeat=6)  # 30 rows
    _write_split(directory / "Dev.csv", _ROWS, repeat=2)  # 10 rows, index 0..9
    _write_split(directory / "Test.csv", _ROWS, repeat=1)  # 5 rows, index 0..4
    return directory


@pytest.fixture
def samples_path(tmp_path: Path) -> Path:
    path = tmp_path / "samples.json"
    samples = {
        "dev_small": {"split": "dev", "indices": [5, 6, 7, 8, 9]},
        "test_small": {"split": "test", "indices": [1, 3]},
    }
    path.write_text(json.dumps({"seed": 42, "samples": samples}), encoding="utf-8")
    return path


@pytest.fixture
def config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "name": "baseline_tfidf",
            "analyzer": "tfidf",
            "seed": 42,
            "tfidf": {"ngram_max": 2, "min_df": 1, "c_grid": [1.0, 10.0]},
            "tune": {"split": "dev", "exclude_sample": "dev_small"},
        }
    )


# --- Config ------------------------------------------------------------------------------


def test_the_committed_baseline_config_is_valid() -> None:
    config = load_config(REPO_ROOT / "evals/configs/baseline.yaml")

    assert config.name == "baseline_tfidf"
    assert config.tune is not None
    assert config.tune.exclude_sample == "dev_100"


@pytest.mark.parametrize(
    "change",
    [
        {"temperature": 0.2},  # unknown key: probably a typo, never silently ignored
        {"name": "Baseline TF-IDF"},  # becomes part of a file name: [a-z0-9_] only
        {"tfidf": {"ngram_max": 2, "min_df": 1, "c_grid": []}},  # nothing to tune
        {"tfidf": None},  # the tfidf analyzer needs its parameters
    ],
)
def test_invalid_configs_are_rejected(config: RunConfig, change: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate(config.model_dump() | change)


def test_load_config_parses_yaml(tmp_path: Path, config: RunConfig) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "name: baseline_tfidf\nanalyzer: tfidf\nseed: 42\n"
        "tfidf: {ngram_max: 2, min_df: 1, c_grid: [1.0, 10.0]}\n"
        "tune: {split: dev, exclude_sample: dev_small}\n",
        encoding="utf-8",
    )

    assert load_config(path) == config


# --- Selecting the eval set --------------------------------------------------------------


def test_select_a_full_split(raw_dir: Path, samples_path: Path) -> None:
    reviews = select_reviews("test", raw_dir=raw_dir, samples_path=samples_path)

    assert [r.index for r in reviews] == [0, 1, 2, 3, 4]


def test_select_a_named_sample_by_its_committed_indices(raw_dir: Path, samples_path: Path) -> None:
    reviews = select_reviews("test_small", raw_dir=raw_dir, samples_path=samples_path)

    assert [(r.split, r.index) for r in reviews] == [("test", 1), ("test", 3)]


def test_limit_takes_the_first_n(raw_dir: Path, samples_path: Path) -> None:
    reviews = select_reviews("dev_small", raw_dir=raw_dir, samples_path=samples_path, limit=2)

    assert [r.index for r in reviews] == [5, 6]


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("train", "never scored"),  # SPEC §4.1: train is for fitting and few-shot only
        ("test_999", "unknown eval set"),
    ],
)
def test_select_rejects_train_and_unknown_names(
    raw_dir: Path, samples_path: Path, name: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        select_reviews(name, raw_dir=raw_dir, samples_path=samples_path)


# --- Scoring -----------------------------------------------------------------------------


def test_score_reports_pair_detection_latency_and_cost(raw_dir: Path, samples_path: Path) -> None:
    gold = select_reviews("test", raw_dir=raw_dir, samples_path=samples_path)
    perfect = [review.labels for review in gold]

    metrics = score(gold, perfect, latencies_ms=[1.0, 2.0, 3.0, 4.0, 5.0], total_cost_usd=0.0)

    assert metrics["pair"]["f1"] == 1.0
    assert metrics["aspect_detection"]["per_aspect"]["BATTERY"]["support"] == 2
    assert metrics["aspect_detection"]["per_aspect"]["STORAGE"]["support"] == 0
    assert metrics["latency_ms"] == {"p50": 3.0, "p95": pytest.approx(4.8)}
    assert metrics["cost_per_1k_usd"] == 0.0
    json.dumps(metrics)  # must be JSON-serializable as is


# --- End to end --------------------------------------------------------------------------


def test_run_writes_a_traceable_results_file(
    tmp_path: Path, raw_dir: Path, samples_path: Path, config: RunConfig
) -> None:
    results_dir = tmp_path / "results"

    path = run(
        config,
        split="test",
        limit=None,
        raw_dir=raw_dir,
        samples_path=samples_path,
        results_dir=results_dir,
        git_sha="abc1234",
        today=date(2026, 11, 21),
    )

    assert path == results_dir / "2026-11-21_baseline_tfidf_test.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    # Everything needed to reproduce the number travels with the number.
    assert result["run"] == {
        "split": "test",
        "limit": None,
        "n": 5,
        "git_sha": "abc1234",
        "date": "2026-11-21",
        "seed": 42,
    }
    assert result["config"] == config.model_dump(mode="json")
    assert result["analyzer"]["best_c"] in (1.0, 10.0)
    assert result["analyzer"]["tune_n"] == 5  # dev (10) minus dev_small (5)
    assert 0.0 <= result["metrics"]["pair"]["f1"] <= 1.0


def test_run_with_limit_puts_n_in_the_file_name(
    tmp_path: Path, raw_dir: Path, samples_path: Path, config: RunConfig
) -> None:
    path = run(
        config,
        split="test",
        limit=3,
        raw_dir=raw_dir,
        samples_path=samples_path,
        results_dir=tmp_path,
        git_sha="abc1234",
        today=date(2026, 11, 21),
    )

    assert path.name == "2026-11-21_baseline_tfidf_test_n3.json"


def test_run_refuses_to_score_reviews_used_for_tuning(
    tmp_path: Path, raw_dir: Path, samples_path: Path, config: RunConfig
) -> None:
    # C is tuned on dev minus dev_small, so scoring on full dev would be scoring on
    # data the model was selected with: an optimistic, leaked number.
    with pytest.raises(ValueError, match="used for tuning"):
        run(
            config,
            split="dev",
            limit=None,
            raw_dir=raw_dir,
            samples_path=samples_path,
            results_dir=tmp_path,
            git_sha="abc1234",
            today=date(2026, 11, 21),
        )


def test_scoring_the_held_out_dev_sample_is_allowed(
    tmp_path: Path, raw_dir: Path, samples_path: Path, config: RunConfig
) -> None:
    path = run(
        config,
        split="dev_small",
        limit=None,
        raw_dir=raw_dir,
        samples_path=samples_path,
        results_dir=tmp_path,
        git_sha="abc1234",
        today=date(2026, 11, 21),
    )

    assert path.exists()
