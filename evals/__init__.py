"""Offline evaluation tooling: metrics, baselines and the eval runner (SPEC §7).

Lives outside `src/review_insight` on purpose: it is never shipped in the production image.
It may import from `review_insight`; `review_insight` must never import from here.
"""
