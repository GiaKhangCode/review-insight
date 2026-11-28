"""Logging an eval run to Langfuse as a dataset run with item- and run-level scores."""

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest

from evals.langfuse_runs import (
    dataset_items,
    dataset_name,
    item_pair_f1,
    log_dataset_run,
)
from review_insight.data.visfd import AspectLabel, GoldReview
from review_insight.domain.enums import Aspect, Sentiment

A, S = Aspect, Sentiment


def _review(index: int, text: str, *labels: tuple[Aspect, Sentiment]) -> GoldReview:
    return GoldReview(
        split="dev",
        index=index,
        text=text,
        n_star=5,
        posted_at=date(2020, 7, 9),
        labels=frozenset(AspectLabel(a, s) for a, s in labels),
    )


GOLD = [
    _review(3, "pin trâu", (A.BATTERY, S.POSITIVE)),
    _review(8, "camera đẹp, giao chậm", (A.CAMERA, S.POSITIVE), (A.SER_ACC, S.NEGATIVE)),
]
METRICS: dict[str, Any] = {
    "pair": {"f1": 0.8},
    "aspect_detection": {"micro": {"f1": 0.9}, "macro_f1": 0.85},
    "latency_ms": {"p50": 900.0, "p95": 1700.0},
    "cost_per_1k_usd": 0.67,
    "reliability": {"after_repair": {"accepted_rate": 1.0}},
}


# --- Building the dataset ----------------------------------------------------------------


def test_one_dataset_per_eval_set() -> None:
    assert dataset_name("dev_100") == "visfd-dev_100"


def test_items_carry_the_review_its_gold_labels_and_a_stable_id() -> None:
    items = dataset_items("dev_100", GOLD)

    assert items[1] == {
        # Stable id: re-running upserts the same item instead of duplicating it.
        "id": "visfd-dev_100-8",
        "input": {"review": "camera đẹp, giao chậm"},
        "expected_output": {
            "aspects": [
                {"aspect": "CAMERA", "sentiment": "positive"},
                {"aspect": "SER&ACC", "sentiment": "negative"},
            ]
        },
        "metadata": {"split": "dev", "index": 8, "n_star": 5},
    }


# --- Item-level score --------------------------------------------------------------------


def _pairs(*pairs: tuple[Aspect, Sentiment]) -> list[dict[str, str]]:
    return [{"aspect": a.value, "sentiment": s.value} for a, s in pairs]


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        (_pairs((A.BATTERY, S.POSITIVE)), _pairs((A.BATTERY, S.POSITIVE)), 1.0),
        ([], [], 1.0),  # correctly found nothing: a perfect answer, not 0/0
        (_pairs((A.CAMERA, S.NEGATIVE)), _pairs((A.CAMERA, S.POSITIVE)), 0.0),
        # 1 TP, 0 FP, 1 FN -> 2 / (2 + 0 + 1)
        (
            _pairs((A.CAMERA, S.POSITIVE), (A.PRICE, S.NEUTRAL)),
            _pairs((A.CAMERA, S.POSITIVE)),
            2 / 3,
        ),
    ],
)
def test_item_pair_f1(
    gold: list[dict[str, str]], pred: list[dict[str, str]], expected: float
) -> None:
    assert item_pair_f1(gold, pred) == pytest.approx(expected)


# --- Logging the run (fake client, no network) -------------------------------------------


@dataclass
class FakeItem:
    id: str
    input: Any
    expected_output: Any
    metadata: Any


@dataclass
class FakeDataset:
    items: list[FakeItem]
    experiments: list[dict[str, Any]] = field(default_factory=list)

    def run_experiment(self, **kwargs: Any) -> Any:
        # Mimics the SDK: run the task per item, then item and run evaluators. Like the SDK,
        # an evaluator may return one Evaluation or a list of them; both are flattened.
        def flat(returned: Any) -> list[Any]:
            return list(returned) if isinstance(returned, list) else [returned]

        results = []
        for item in self.items:
            output = kwargs["task"](item=item)
            scores = []
            for evaluator in kwargs["evaluators"]:
                scores += flat(
                    evaluator(
                        input=item.input,
                        output=output,
                        expected_output=item.expected_output,
                        metadata=item.metadata,
                    )
                )
            results.append({"output": output, "scores": scores})
        run_scores = [s for e in kwargs["run_evaluators"] for s in flat(e(item_results=results))]
        self.experiments.append({**kwargs, "results": results, "run_scores": run_scores})
        return type("Result", (), {"dataset_run_url": "https://lf.example/run/1"})()


class FakeClient:
    def __init__(self) -> None:
        self.datasets: dict[str, FakeDataset] = {}

    def create_dataset(self, *, name: str, **kwargs: Any) -> None:
        self.datasets.setdefault(name, FakeDataset(items=[]))

    def create_dataset_item(self, *, dataset_name: str, id: str, **kwargs: Any) -> None:
        items = self.datasets[dataset_name].items
        items[:] = [item for item in items if item.id != id]  # upsert by id
        items.append(
            FakeItem(
                id=id,
                input=kwargs["input"],
                expected_output=kwargs["expected_output"],
                metadata=kwargs["metadata"],
            )
        )

    def get_dataset(self, name: str) -> FakeDataset:
        return self.datasets[name]


OUTPUTS = {
    3: {"aspects": _pairs((A.BATTERY, S.POSITIVE)), "failure": None},
    8: {"aspects": [], "failure": "schema"},  # rejected answer: predicts nothing
}


def _log(client: FakeClient, outputs: dict[int, dict[str, Any]] = OUTPUTS) -> str | None:
    return log_dataset_run(
        client,
        split="dev_100",
        gold=GOLD,
        outputs=outputs,
        metrics=METRICS,
        run_name="2026-11-28_zeroshot_35_dev_100",
        metadata={"git_sha": "abc1234", "prompt": "analyze_review@v2"},
    )


def test_a_run_is_logged_with_per_item_scores_and_no_model_calls() -> None:
    client = FakeClient()

    url = _log(client)

    dataset = client.datasets["visfd-dev_100"]
    [experiment] = dataset.experiments
    assert url == "https://lf.example/run/1"
    assert experiment["run_name"] == "2026-11-28_zeroshot_35_dev_100"
    assert experiment["metadata"] == {"git_sha": "abc1234", "prompt": "analyze_review@v2"}
    # The task replays our predictions, so the dataset run cannot differ from the results file.
    assert [r["output"] for r in experiment["results"]] == [OUTPUTS[3], OUTPUTS[8]]
    first, second = (r["scores"] for r in experiment["results"])
    assert [(s.name, s.value) for s in first] == [("pair_f1", 1.0), ("accepted", 1)]
    assert [(s.name, s.value) for s in second] == [("pair_f1", 0.0), ("accepted", 0)]


def test_run_level_scores_come_from_the_results_file() -> None:
    client = FakeClient()

    _log(client)

    [experiment] = client.datasets["visfd-dev_100"].experiments
    assert {s.name: s.value for s in experiment["run_scores"]} == {
        "pair_f1": 0.8,
        "aspect_f1_micro": 0.9,
        "aspect_f1_macro": 0.85,
        "latency_p95_ms": 1700.0,
        "cost_per_1k_usd": 0.67,
        "accepted_rate": 1.0,
    }


def test_the_baseline_has_no_acceptance_score() -> None:
    # TF-IDF outputs are never rejected; an "accepted" score would be meaningless noise.
    client = FakeClient()
    baseline_outputs = {i: {"aspects": o["aspects"]} for i, o in OUTPUTS.items()}

    _log(client, baseline_outputs)

    [experiment] = client.datasets["visfd-dev_100"].experiments
    names = {s.name for r in experiment["results"] for s in r["scores"]}
    assert names == {"pair_f1"}


def test_logging_twice_upserts_items_instead_of_duplicating_them() -> None:
    client = FakeClient()

    _log(client)
    _log(client)

    assert len(client.datasets["visfd-dev_100"].items) == 2
