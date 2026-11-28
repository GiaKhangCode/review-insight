"""Eval runner (SPEC §7.4): fit/tune an analyzer, score it on an eval set, write a results file.

Usage:
    uv run python -m evals.run --config evals/configs/baseline.yaml --split test_500
    uv run python -m evals.run --config evals/configs/zeroshot_31.yaml --split dev_100

`--split` is a full split (dev, test) or a committed sample (dev_100, test_500, smoke_50).
Results go to evals/results/<date>_<config>_<split>[_n<limit>].json, with the config,
git sha and seed, so every number can be traced back to what produced it.

LLM runs replay paid responses from evals/.cache/, so an interrupted run (timeout, quota)
resumes for free: run the same command again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evals.baseline_tfidf import TfidfBaseline, tune_c
from evals.metrics import (
    PRF,
    Pair,
    aspect_detection_scores,
    cost_per_1k,
    latency_percentiles,
    pair_scores,
)
from evals.model_cache import DEFAULT_CACHE_DIR, DiskCachedCall, ThrottledCall
from review_insight.config import Settings
from review_insight.data.visfd import GoldReview, Split, load_split
from review_insight.llm.analyzer import ModelCall, ReviewAnalyzer, build_gemini_call
from review_insight.llm.pricing import DEFAULT_PRICING_PATH, load_price_table
from review_insight.llm.prompts import load_prompt
from review_insight.llm.resilience import RetryingCall

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/UIT-ViSFD")
SAMPLES_PATH = Path("evals/data/samples.json")
RESULTS_DIR = Path("evals/results")

# Splits that may be scored. `train` is deliberately absent (SPEC §4.1).
_SCORABLE: dict[str, Split] = {"dev": "dev", "test": "test"}


# --- Config ------------------------------------------------------------------------------


class _Strict(BaseModel):
    # A typo in a config key must fail loudly, not silently fall back to a default.
    model_config = ConfigDict(extra="forbid", frozen=True)


class TfidfParams(_Strict):
    ngram_max: int = Field(ge=1, le=3)
    min_df: int = Field(ge=1)
    c_grid: list[Annotated[float, Field(gt=0)]] = Field(min_length=1)


class TuneParams(_Strict):
    split: Literal["dev"]  # hyperparameters are never chosen on test
    exclude_sample: str | None = None


class GeminiParams(_Strict):
    model: str
    prompt: str  # path to a prompts/*.yaml file, relative to the repo root
    # 5 s between request starts = 12 RPM, under the free tier's 15 RPM (docs/GEMINI.md).
    min_interval_s: float = Field(default=5.0, gt=0)
    # One repair retry for rejected answers. First-try quality is reported either way.
    repair: bool = True


class RunConfig(_Strict):
    name: str = Field(pattern=r"^[a-z0-9_]+$")  # used in the results file name
    analyzer: Literal["tfidf", "gemini"]
    seed: int
    tfidf: TfidfParams | None = None
    tune: TuneParams | None = None
    gemini: GeminiParams | None = None

    @model_validator(mode="after")
    def _analyzer_has_its_sections(self) -> Self:
        if self.analyzer == "tfidf" and (self.tfidf is None or self.tune is None):
            raise ValueError("analyzer 'tfidf' needs both a 'tfidf' and a 'tune' section")
        if self.analyzer == "gemini" and self.gemini is None:
            raise ValueError("analyzer 'gemini' needs a 'gemini' section")
        return self


def load_config(path: Path) -> RunConfig:
    # safe_load builds only plain types (dict, list, str, ...), never arbitrary objects.
    return RunConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


# --- Eval sets ---------------------------------------------------------------------------


class _Sample(BaseModel):
    split: Split
    indices: list[int]


class _SamplesFile(BaseModel):
    samples: dict[str, _Sample]  # other top-level keys (dataset, seed, ...) are metadata


def _load_samples(path: Path) -> dict[str, _Sample]:
    return _SamplesFile.model_validate_json(path.read_text(encoding="utf-8")).samples


def _pick(reviews: Sequence[GoldReview], indices: Sequence[int]) -> list[GoldReview]:
    by_index = {review.index: review for review in reviews}
    missing = [i for i in indices if i not in by_index]
    if missing:
        raise ValueError(f"sample indices not found in the split: {missing[:5]}")
    return [by_index[i] for i in indices]


def select_reviews(
    name: str, *, raw_dir: Path, samples_path: Path, limit: int | None = None
) -> list[GoldReview]:
    """Resolve an eval-set name to reviews: a full split or a committed sample."""
    split = _SCORABLE.get(name)
    if split is not None:
        reviews = load_split(split, raw_dir=raw_dir)
    elif name == "train":
        raise ValueError("train is never scored: it is for fitting and few-shot examples only")
    else:
        samples = _load_samples(samples_path)
        if name not in samples:
            known = [*_SCORABLE, *sorted(samples)]
            raise ValueError(f"unknown eval set {name!r}; expected one of {known}")
        sample = samples[name]
        if sample.split not in _SCORABLE:
            raise ValueError(f"sample {name!r} is drawn from {sample.split}, which is never scored")
        reviews = _pick(load_split(sample.split, raw_dir=raw_dir), sample.indices)
    return reviews if limit is None else reviews[:limit]


# --- Scoring -----------------------------------------------------------------------------


def _prf_json(prf: PRF) -> dict[str, float | int]:
    return {
        "precision": round(prf.precision, 4),
        "recall": round(prf.recall, 4),
        "f1": round(prf.f1, 4),
        "support": prf.support,
        "tp": prf.tp,
        "fp": prf.fp,
        "fn": prf.fn,
    }


def score(
    gold: Sequence[GoldReview],
    pred: Sequence[frozenset[Pair]],
    *,
    latencies_ms: Sequence[float],
    total_cost_usd: float,
) -> dict[str, Any]:
    """All SPEC §7.2 metrics that need only ViSFD labels, as a JSON-ready dict."""
    labels = [review.labels for review in gold]
    pairs = pair_scores(labels, pred)
    detection = aspect_detection_scores(labels, pred)
    p50, p95 = latency_percentiles(latencies_ms)
    return {
        "pair": _prf_json(pairs),
        "aspect_detection": {
            "micro": _prf_json(detection.micro),
            "macro_f1": round(detection.macro_f1, 4),
            "per_aspect": {a.value: _prf_json(prf) for a, prf in detection.per_aspect.items()},
        },
        "latency_ms": {"p50": round(p50, 1), "p95": round(p95, 1)},
        "cost_per_1k_usd": cost_per_1k(total_cost_usd=total_cost_usd, n_reviews=len(gold)),
    }


# --- Analyzers ---------------------------------------------------------------------------


def _tuning_set(config: RunConfig, *, raw_dir: Path, samples_path: Path) -> list[GoldReview]:
    """Reviews used to choose hyperparameters; they must never be scored."""
    if config.tune is None:
        return []
    reviews = load_split(config.tune.split, raw_dir=raw_dir)
    if config.tune.exclude_sample is None:
        return reviews
    held_out = _load_samples(samples_path)[config.tune.exclude_sample]
    if held_out.split != config.tune.split:
        raise ValueError(f"exclude_sample is from {held_out.split}, not {config.tune.split}")
    excluded = set(held_out.indices)
    return [review for review in reviews if review.index not in excluded]


def _fit_tfidf(
    config: RunConfig, tuning: Sequence[GoldReview], *, raw_dir: Path
) -> tuple[TfidfBaseline, dict[str, Any]]:
    params = config.tfidf
    if params is None:  # already guaranteed by RunConfig; this narrows the type for mypy
        raise ValueError("analyzer 'tfidf' needs a 'tfidf' section")
    train = load_split("train", raw_dir=raw_dir)
    result = tune_c(
        train_texts=[r.text for r in train],
        train_labels=[r.labels for r in train],
        dev_texts=[r.text for r in tuning],
        dev_labels=[r.labels for r in tuning],
        c_grid=params.c_grid,
        seed=config.seed,
        ngram_max=params.ngram_max,
        min_df=params.min_df,
    )
    info = {
        "best_c": result.best_c,
        "tune_pair_f1_by_c": {str(c): round(f1, 4) for c, f1 in result.dev_pair_f1.items()},
        "train_n": len(train),
        "tune_n": len(tuning),
    }
    return result.model, info


@dataclass(frozen=True, slots=True)
class _Prediction:
    """One scored review, whatever produced it."""

    pairs: frozenset[Pair]  # empty when the output was rejected: production would not serve it
    latency_ms: float
    cost_usd: float = 0.0
    failure: str | None = None  # after the repair retry, if any
    first_failure: str | None = None  # on the first attempt
    attempts: int = 1
    cached: bool = False
    evidence_total: int = 0
    evidence_ungrounded: int = 0


def _predict_tfidf(model: TfidfBaseline, gold: Sequence[GoldReview]) -> list[_Prediction]:
    predictions = []
    for review in gold:  # one review per call, like an API request, so latency is comparable
        start = time.perf_counter()
        [pairs] = model.predict([review.text])
        predictions.append(_Prediction(pairs, (time.perf_counter() - start) * 1000))
    return predictions


async def _predict_llm(analyzer: ReviewAnalyzer, gold: Sequence[GoldReview]) -> list[_Prediction]:
    predictions = []
    # Sequential on purpose: the free tier's RPM is the bottleneck, not our concurrency.
    for i, review in enumerate(gold, start=1):
        analysis = await analyzer.analyze(review.text)
        pairs = (
            frozenset((item.aspect, item.sentiment) for item in analysis.result.aspects)
            if analysis.result is not None
            else frozenset()
        )
        predictions.append(
            _Prediction(
                pairs=pairs,
                latency_ms=analysis.latency_ms,
                cost_usd=analysis.cost_usd,
                failure=analysis.failure,
                first_failure=analysis.first_failure,
                attempts=analysis.attempts,
                cached=analysis.cached,
                evidence_total=analysis.evidence_total,
                evidence_ungrounded=analysis.evidence_ungrounded,
            )
        )
        if i % 10 == 0 or i == len(gold):
            hits = sum(p.cached for p in predictions)
            failed = sum(p.failure is not None for p in predictions)
            logger.info("%d/%d reviews (cache hits %d, rejected %d)", i, len(gold), hits, failed)
    return predictions


def _rates(failures: Sequence[str | None]) -> dict[str, float]:
    n = len(failures)
    return {
        "schema_valid_rate": round(sum(f != "schema" for f in failures) / n, 4),
        "accepted_rate": round(sum(f is None for f in failures) / n, 4),
    }


def _reliability(predictions: Sequence[_Prediction]) -> dict[str, Any]:
    """SPEC §7.2: schema validity on the first try and after repair, evidence grounding."""
    evidence = sum(p.evidence_total for p in predictions)
    ungrounded = sum(p.evidence_ungrounded for p in predictions)
    return {
        "first_try": _rates([p.first_failure for p in predictions]),
        "after_repair": _rates([p.failure for p in predictions]),
        "repaired": sum(p.first_failure is not None and p.failure is None for p in predictions),
        "repair_calls": sum(p.attempts > 1 for p in predictions),
        "evidence_grounding_rate": round(1 - ungrounded / evidence, 4) if evidence else None,
        "failures": {
            "schema": sum(p.failure == "schema" for p in predictions),
            "ungrounded": sum(p.failure == "ungrounded" for p in predictions),
        },
        "cache_hits": sum(p.cached for p in predictions),
    }


def _real_model_call(params: GeminiParams, *, cache_dir: Path) -> ModelCall:
    """Cache( Retry( Throttle( Gemini ) ) ).

    Cache outermost: hits never wait or spend quota. Retry outside the throttle: every
    retry waits its turn, so retries cannot burst past the RPM limit. No fallback model:
    an eval run measures one model.
    """
    settings = Settings()
    gemini = build_gemini_call(
        model=params.model,
        api_key=settings.require_google_api_key(),
        timeout_s=settings.llm_timeout_s,
    )
    throttled = ThrottledCall(gemini, min_interval_s=params.min_interval_s)
    return DiskCachedCall(RetryingCall(throttled), cache_dir=cache_dir, model=params.model)


# --- Run ---------------------------------------------------------------------------------


def run(
    config: RunConfig,
    *,
    split: str,
    limit: int | None,
    raw_dir: Path,
    samples_path: Path,
    results_dir: Path,
    git_sha: str,
    today: date,
    model_call: ModelCall | None = None,  # injected in tests; None = real Gemini + cache
    pricing_path: Path = DEFAULT_PRICING_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    gold = select_reviews(split, raw_dir=raw_dir, samples_path=samples_path, limit=limit)

    # Leakage guard, checked before any training: scored reviews must be unseen in tuning.
    tuning = _tuning_set(config, raw_dir=raw_dir, samples_path=samples_path)
    leaked = {(r.split, r.index) for r in gold} & {(r.split, r.index) for r in tuning}
    if leaked:
        raise ValueError(
            f"{len(leaked)} of {len(gold)} reviews in {split!r} were used for tuning; "
            "score on a held-out sample or on test"
        )

    info: dict[str, Any]
    if config.gemini is not None:
        prompt = load_prompt(Path(config.gemini.prompt))
        prices = load_price_table(pricing_path)
        call = model_call or _real_model_call(config.gemini, cache_dir=cache_dir)
        analyzer = ReviewAnalyzer(
            call=call, prompt=prompt, prices=prices, repair=config.gemini.repair
        )
        predictions = asyncio.run(_predict_llm(analyzer, gold))
        info = {
            "model": config.gemini.model,
            "prompt": prompt.ref,
            "prompt_fingerprint": prompt.fingerprint,
            "pricing_retrieved": prices.retrieved.isoformat(),
        }
    else:
        model, info = _fit_tfidf(config, tuning, raw_dir=raw_dir)
        predictions = _predict_tfidf(model, gold)  # local CPU inference: no per-call cost

    metrics = score(
        gold,
        [p.pairs for p in predictions],
        latencies_ms=[p.latency_ms for p in predictions],
        total_cost_usd=sum(p.cost_usd for p in predictions),  # failed calls are billed too
    )
    if config.gemini is not None:
        metrics["reliability"] = _reliability(predictions)

    result = {
        "run": {
            "split": split,
            "limit": limit,
            "n": len(gold),
            "git_sha": git_sha,
            "date": today.isoformat(),
            "seed": config.seed,
        },
        "config": config.model_dump(mode="json"),
        "analyzer": info,
        "metrics": metrics,
    }

    suffix = "" if limit is None else f"_n{limit}"
    path = results_dir / f"{today.isoformat()}_{config.name}_{split}{suffix}.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return path


def current_git_sha() -> str:
    """Short HEAD sha, suffixed `-dirty` if tracked files have uncommitted changes."""
    try:
        # Fixed argument lists, no shell, no user input. S607 flags "git" resolved via PATH,
        # which is the intent here: use whatever git the developer or CI runner has.
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if status else sha


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", required=True, help="dev, test, or a sample name")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    path = run(
        load_config(args.config),
        split=args.split,
        limit=args.limit,
        raw_dir=RAW_DIR,
        samples_path=SAMPLES_PATH,
        results_dir=RESULTS_DIR,
        git_sha=current_git_sha(),
        today=date.today(),
    )

    metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    detection = metrics["aspect_detection"]
    print(f"pair F1            {metrics['pair']['f1']:.4f}")
    print(f"aspect F1 micro    {detection['micro']['f1']:.4f}")
    print(f"aspect F1 macro    {detection['macro_f1']:.4f}")
    print(f"latency p50 / p95  {metrics['latency_ms']['p50']} / {metrics['latency_ms']['p95']} ms")
    print(f"cost per 1k        ${metrics['cost_per_1k_usd']:.4f}")
    if "reliability" in metrics:
        reliability = metrics["reliability"]
        first, final = reliability["first_try"], reliability["after_repair"]
        for key, label in (("schema_valid_rate", "schema valid"), ("accepted_rate", "accepted")):
            print(f"{label:18} {first[key]:.1%} -> {final[key]:.1%}  (first try -> after repair)")
        print(f"repaired           {reliability['repaired']} of {reliability['repair_calls']}")
        print(f"evidence grounded  {reliability['evidence_grounding_rate']}")
        print(f"cache hits         {reliability['cache_hits']}")
    print(f"written to         {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
