"""Log a finished eval run to Langfuse as a dataset run, with item- and run-level scores.

Langfuse v5 creates dataset runs only through `run_experiment`, which calls a task per item.
Here the task replays predictions the runner already made (from the results of this run),
so logging makes no model call, costs nothing, and cannot disagree with the results file.

Dataset items hold review text from UIT-ViSFD. They live in a private Langfuse project, like
the prompts sent to Gemini; never make these datasets or their traces public.
"""

from collections.abc import Collection, Mapping, Sequence
from typing import Any, Protocol

from langfuse import Evaluation

from evals.metrics import PRF, Pair
from review_insight.data.visfd import GoldReview


class DatasetsClient(Protocol):
    """The slice of the Langfuse client used here."""

    def create_dataset(
        self, *, name: str, description: str | None = None, metadata: Any = None
    ) -> Any: ...

    def create_dataset_item(
        self,
        *,
        dataset_name: str,
        input: Any = None,
        expected_output: Any = None,
        metadata: Any = None,
        id: str | None = None,
    ) -> Any: ...

    def get_dataset(self, name: str) -> Any: ...


def dataset_name(split: str) -> str:
    return f"visfd-{split}"


def pairs_json(pairs: Collection[Pair]) -> list[dict[str, str]]:
    """Labels as sorted JSON objects, so equal label sets always serialize the same way."""
    return [
        {"aspect": aspect.value, "sentiment": sentiment.value}
        for aspect, sentiment in sorted(pairs, key=lambda p: (p[0].value, p[1].value))
    ]


def dataset_items(split: str, gold: Sequence[GoldReview]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{dataset_name(split)}-{review.index}",  # stable: re-logging upserts
            "input": {"review": review.text},
            "expected_output": {"aspects": pairs_json(review.labels)},
            "metadata": {"split": review.split, "index": review.index, "n_star": review.n_star},
        }
        for review in gold
    ]


def item_pair_f1(gold: list[dict[str, str]], pred: list[dict[str, str]]) -> float:
    """Pair F1 for a single review. Nothing expected and nothing predicted is a perfect 1.0."""
    expected = {(p["aspect"], p["sentiment"]) for p in gold}
    predicted = {(p["aspect"], p["sentiment"]) for p in pred}
    if not expected and not predicted:
        return 1.0
    prf = PRF(
        tp=len(expected & predicted),
        fp=len(predicted - expected),
        fn=len(expected - predicted),
    )
    return prf.f1


def _run_scores(metrics: Mapping[str, Any]) -> list[Evaluation]:
    detection = metrics["aspect_detection"]
    scores = {
        "pair_f1": metrics["pair"]["f1"],
        "aspect_f1_micro": detection["micro"]["f1"],
        "aspect_f1_macro": detection["macro_f1"],
        "latency_p95_ms": metrics["latency_ms"]["p95"],
        "cost_per_1k_usd": metrics["cost_per_1k_usd"],
    }
    if "reliability" in metrics:  # LLM runs only
        scores["accepted_rate"] = metrics["reliability"]["after_repair"]["accepted_rate"]
    return [Evaluation(name=name, value=value) for name, value in scores.items()]


def log_dataset_run(
    client: DatasetsClient,
    *,
    split: str,
    gold: Sequence[GoldReview],
    outputs: Mapping[int, Mapping[str, Any]],
    metrics: Mapping[str, Any],
    run_name: str,
    metadata: Mapping[str, str],
) -> str | None:
    """Upsert the eval set as a dataset, then record this run on it. Returns the run URL.

    `outputs` maps a review index to {"aspects": [...]} plus, for LLM runs, "failure".
    """
    name = dataset_name(split)
    client.create_dataset(
        name=name,
        description=f"UIT-ViSFD {split}: fixed eval set (indices in evals/data/samples.json)",
        metadata={"source": "UIT-ViSFD", "terms": "research use only; keep private"},
    )
    for item in dataset_items(split, gold):
        client.create_dataset_item(dataset_name=name, **item)

    def task(*, item: Any, **_: Any) -> Mapping[str, Any]:
        return outputs[item.metadata["index"]]

    def score_item(*, output: Any, expected_output: Any, **_: Any) -> list[Evaluation]:
        f1 = item_pair_f1(expected_output["aspects"], output["aspects"])
        scores = [Evaluation(name="pair_f1", value=f1)]
        if "failure" in output:  # only LLM outputs can be rejected
            scores.append(Evaluation(name="accepted", value=int(output["failure"] is None)))
        return scores

    def score_run(**_: Any) -> list[Evaluation]:
        return _run_scores(metrics)

    result = client.get_dataset(name).run_experiment(
        name=run_name,
        run_name=run_name,
        description=f"{run_name} (replayed from evals/results; no model calls)",
        task=task,
        evaluators=[score_item],
        run_evaluators=[score_run],
        max_concurrency=1,
        metadata=dict(metadata),
    )
    url: str | None = getattr(result, "dataset_run_url", None)
    return url
