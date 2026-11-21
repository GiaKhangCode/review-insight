"""Create the fixed evaluation samples (SPEC §4.1) and print a distribution check.

Usage:
    uv run python scripts/make_samples.py

Writes evals/data/samples.json (committed). Every eval run selects reviews by these
indices, so all configurations are compared on exactly the same data. Re-running with the
same seed and dataset must produce a byte-identical file.
"""

import json
from collections import Counter
from pathlib import Path

from review_insight.data.visfd import GoldReview, load_split, stratified_sample
from review_insight.domain.enums import Aspect

RAW_DIR = Path("data/raw/UIT-ViSFD")
OUTPUT = Path("evals/data/samples.json")
SEED = 42


def aspect_share(reviews: list[GoldReview]) -> dict[Aspect, float]:
    """Fraction of reviews that mention each aspect (any sentiment)."""
    counts = Counter(label.aspect for review in reviews for label in review.labels)
    return {aspect: counts[aspect] / len(reviews) for aspect in Aspect}


def print_distribution_check(
    name: str, population: list[GoldReview], sample: list[GoldReview]
) -> None:
    stars_pop = Counter(r.n_star for r in population)
    stars_smp = Counter(r.n_star for r in sample)
    print(f"\n{name}: {len(sample)} of {len(population)}")
    print("  stars   population  sample")
    for star in sorted(stars_pop):
        print(
            f"  {star}       {stars_pop[star] / len(population):9.1%}"
            f"  {stars_smp[star] / len(sample):6.1%}"
        )
    pop, smp = aspect_share(population), aspect_share(sample)
    print("  aspect        population  sample  (share of reviews mentioning it)")
    for aspect in Aspect:
        print(f"  {aspect.value:<12}  {pop[aspect]:9.1%}  {smp[aspect]:6.1%}")


def main() -> None:
    test = load_split("test", raw_dir=RAW_DIR)
    dev = load_split("dev", raw_dir=RAW_DIR)

    test_500 = stratified_sample(test, n=500, seed=SEED)
    # smoke_50 is drawn FROM test_500, so the CI gate measures a subset of the main sample.
    test_500_reviews = [r for r in test if r.index in set(test_500)]
    smoke_50 = stratified_sample(test_500_reviews, n=50, seed=SEED)
    dev_100 = stratified_sample(dev, n=100, seed=SEED)

    payload = {
        "dataset": "UIT-ViSFD",
        "method": "stratified by n_star, largest-remainder allocation",
        "seed": SEED,
        "samples": {
            "test_500": {"split": "test", "indices": test_500},
            "smoke_50": {"split": "test", "indices": smoke_50},
            "dev_100": {"split": "dev", "indices": dev_100},
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    new_content = json.dumps(payload, indent=2) + "\n"
    changed = not OUTPUT.exists() or OUTPUT.read_bytes() != new_content.encode("utf-8")
    # newline="\n": byte-identical output on Windows and Linux (no CRLF translation).
    OUTPUT.write_text(new_content, encoding="utf-8", newline="\n")

    by_index = {r.index: r for r in test}
    print_distribution_check("test_500", test, [by_index[i] for i in test_500])
    print_distribution_check("smoke_50", test, [by_index[i] for i in smoke_50])
    print(f"\n{'wrote' if changed else 'unchanged'}: {OUTPUT}")


if __name__ == "__main__":
    main()
