"""Eval runner (SPEC §7.4): config loading, eval-set selection, leakage guard, results file."""

import csv
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import BaseMessage
from pydantic import ValidationError

from evals.run import RunConfig, load_config, run, score, select_reviews
from review_insight.domain.enums import Aspect, Route, Sentiment, Urgency
from review_insight.domain.schemas import AnalysisResult, AspectSentiment
from review_insight.llm.analyzer import RawCall
from review_insight.llm.pricing import TokenUsage

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


# --- LLM analyzer (gemini) ---------------------------------------------------------------

_USAGE = TokenUsage(input_tokens=700, output_tokens=100)


def _answer(*aspects: tuple[Aspect, Sentiment, str]) -> AnalysisResult:
    return AnalysisResult(
        aspects=[AspectSentiment(aspect=a, sentiment=s, evidence=e) for a, s, e in aspects],
        urgency=Urgency.LOW,
        route_to=Route.NONE,
    )


# One canned model answer per test review: right, schema failure, ungrounded, empty.
_ANSWERS: dict[str, AnalysisResult | None] = {
    "pin trâu dùng cả ngày": _answer((Aspect.BATTERY, Sentiment.POSITIVE, "pin trâu")),
    "pin yếu sạc hoài": _answer((Aspect.BATTERY, Sentiment.NEGATIVE, "pin yếu")),
    "camera chụp đẹp": _answer((Aspect.CAMERA, Sentiment.POSITIVE, "ảnh sắc nét")),  # invented
    "shop giao hàng chậm": None,  # schema failure
    "máy ổn": _answer(),
}


# What a repair retry gets back: the two rejected reviews, now answered correctly.
_REPAIRED: dict[str, AnalysisResult] = {
    "camera chụp đẹp": _answer((Aspect.CAMERA, Sentiment.POSITIVE, "camera chụp đẹp")),
    "shop giao hàng chậm": _answer((Aspect.SER_ACC, Sentiment.NEGATIVE, "giao hàng chậm")),
}


class ScriptedCall:
    """Fake ModelCall: answers by the review found in the prompt; repairs succeed."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, messages: Sequence[BaseMessage]) -> RawCall:
        self.calls += 1
        prompt = "\n".join(str(m.content) for m in messages)
        review = next(text for text in _ANSWERS if f"\n{text}\n" in prompt)
        is_repair = "previous answer was rejected" in str(messages[-1].content)
        output = _REPAIRED[review] if is_repair else _ANSWERS[review]
        return RawCall(
            model="gemini-3.1-flash-lite",
            output=output,
            error=None if output is not None else "duplicate aspect(s): SER&ACC",
            usage=_USAGE,
            latency_ms=2000.0,
        )


@pytest.fixture
def gemini_config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "name": "zeroshot_31",
            "analyzer": "gemini",
            "seed": 42,
            "gemini": {
                "model": "gemini-3.1-flash-lite",
                "prompt": str(REPO_ROOT / "prompts/analyze_v1.yaml"),
                "repair": False,  # first-try behaviour; repair has its own test below
            },
        }
    )


def _run_gemini(
    config: RunConfig, tmp_path: Path, raw_dir: Path, samples_path: Path, split: str = "test"
) -> dict[str, object]:
    path = run(
        config,
        split=split,
        limit=None,
        raw_dir=raw_dir,
        samples_path=samples_path,
        results_dir=tmp_path / "results",
        git_sha="abc1234",
        today=date(2026, 11, 21),
        model_call=ScriptedCall(),
        pricing_path=REPO_ROOT / "config/pricing.yaml",
    )
    result: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return result


@pytest.mark.parametrize("name", ["zeroshot_31", "zeroshot_35"])
def test_the_committed_zero_shot_configs_are_valid(name: str) -> None:
    config = load_config(REPO_ROOT / f"evals/configs/{name}.yaml")

    assert config.analyzer == "gemini"
    assert config.gemini is not None
    assert (REPO_ROOT / config.gemini.prompt).exists()


def test_a_gemini_config_needs_its_section(gemini_config: RunConfig) -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate(gemini_config.model_dump() | {"gemini": None})


def test_llm_run_reports_reliability_cost_and_provenance(
    tmp_path: Path, raw_dir: Path, samples_path: Path, gemini_config: RunConfig
) -> None:
    result = _run_gemini(gemini_config, tmp_path, raw_dir, samples_path)

    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    first_try = {
        "schema_valid_rate": 0.8,  # 4 of 5 parsed
        "accepted_rate": 0.6,  # 3 of 5 schema-valid AND grounded
    }
    assert metrics["reliability"] == {
        "first_try": first_try,
        "after_repair": first_try,  # repair is off in this config
        "repaired": 0,
        "repair_calls": 0,
        "evidence_grounding_rate": 0.6667,  # 2 of 3 evidence spans found in the review
        "failures": {"schema": 1, "ungrounded": 1},
        "cache_hits": 0,
    }
    # 5 calls x (700 x $0.25 + 100 x $1.50) / 1M, scaled to 1,000 reviews; failures included.
    assert metrics["cost_per_1k_usd"] == pytest.approx(0.325)
    assert metrics["latency_ms"] == {"p50": 2000.0, "p95": 2000.0}
    assert result["analyzer"] == {
        "model": "gemini-3.1-flash-lite",
        "prompt": "analyze_review@v1",
        "prompt_fingerprint": result["analyzer"]["prompt_fingerprint"],
        "pricing_retrieved": "2026-11-21",
    }


def test_rejected_outputs_count_as_predicting_nothing(
    tmp_path: Path, raw_dir: Path, samples_path: Path, gemini_config: RunConfig
) -> None:
    # Production would not serve a rejected output, so the eval must not score it either.
    result = _run_gemini(gemini_config, tmp_path, raw_dir, samples_path)

    pair = result["metrics"]["pair"]
    # TP: the two BATTERY reviews. FN: CAMERA (ungrounded) and SER&ACC (schema failure).
    assert (pair["tp"], pair["fp"], pair["fn"]) == (2, 0, 2)


def test_an_llm_config_may_be_scored_on_full_dev(
    tmp_path: Path, raw_dir: Path, samples_path: Path, gemini_config: RunConfig
) -> None:
    # Nothing is tuned by the runner for LLM configs, so the leakage guard has nothing to block.
    result = _run_gemini(gemini_config, tmp_path, raw_dir, samples_path, split="dev")

    assert result["run"]["n"] == 10


def test_repair_is_reported_separately_from_first_try_quality(
    tmp_path: Path, raw_dir: Path, samples_path: Path, gemini_config: RunConfig
) -> None:
    assert gemini_config.gemini is not None
    gemini = gemini_config.gemini.model_copy(update={"repair": True})
    config = gemini_config.model_copy(update={"gemini": gemini})

    result = _run_gemini(config, tmp_path, raw_dir, samples_path)

    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    reliability = metrics["reliability"]
    # Same first answers as without repair: the repair cannot hide first-try quality.
    assert reliability["first_try"] == {"schema_valid_rate": 0.8, "accepted_rate": 0.6}
    assert reliability["after_repair"] == {"schema_valid_rate": 1.0, "accepted_rate": 1.0}
    assert (reliability["repaired"], reliability["repair_calls"]) == (2, 2)
    pair = metrics["pair"]
    assert (pair["tp"], pair["fp"], pair["fn"]) == (4, 0, 0)
    # 7 calls (5 first + 2 repairs) for 5 reviews: repairs are part of the cost.
    assert metrics["cost_per_1k_usd"] == pytest.approx(7 * 0.000325 / 5 * 1000)
