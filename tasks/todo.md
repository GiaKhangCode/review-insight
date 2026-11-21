# Tasks: ReviewInsight v1.0

Legend: size XS/S/M · 💡 = AI Engineering concept practiced · 🧑 = must be done by you (human judgment)
Default verification for every task: `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest` passes, and CI is green on the PR.

---

## Phase 0: Bootstrap

- [x] **T1: Repo and tooling scaffold** (S)
  - Acceptance: `git init` on `main` · `pyproject.toml` (uv, ruff, mypy strict, pytest config) · `.gitignore`, `.gitattributes` (eol=lf), `.env.example`, `LICENSE` (MIT), `README.md` stub · `.pre-commit-config.yaml` with ruff, ruff-format and gitleaks
  - Verify: `uv sync` works · `uv run pre-commit run --all-files` passes · a deliberately planted fake key is blocked by gitleaks
  - Files: `pyproject.toml`, `.pre-commit-config.yaml`, `.gitignore`, `.gitattributes`, `.env.example`
  - 💡 reproducible environments, secret hygiene

- [x] **T2: App skeleton and CI** (S) · deps: T1
  - Acceptance: `create_app()` factory · `config.py` with pydantic-settings (fails fast on missing env) · `GET /healthz` → 200 · JSON logging · `.github/workflows/ci.yml` running lint, mypy and pytest on PRs
  - Verify: `uv run uvicorn review_insight.main:app` and `curl localhost:8000/healthz` · first PR shows CI green
  - Files: `src/review_insight/{main,config}.py`, `observability/logging_config.py`, `tests/unit/test_{health,logging_config}.py`, `.github/workflows/ci.yml`
  - 💡 12-factor config

## Phase 1: Measure first

- [x] **T3: Domain schemas** (S) · deps: T2
  - Acceptance: enums `Aspect` (10), `Sentiment`, `Urgency`, `Route` · `AspectSentiment(aspect, sentiment, evidence)` · `AnalysisResult` · `validate_grounded()` rejects evidence that is not a substring (after NFC normalization)
  - Verify: unit tests for valid, invalid and Unicode-normalization edge cases
  - Files: `src/review_insight/domain/{schemas,enums,grounding}.py`, `tests/unit/test_grounding.py`
  - 💡 output contracts, grounding

- [x] **T4: Dataset download and loader** (S) · deps: T3
  - Acceptance: `scripts/download_data.py` fetches the zip to `data/raw/` (gitignored) and verifies SHA-256 · `parse_label("{BATTERY#Positive};{OTHERS};")` → domain objects · stratified fixed samples saved by index: `test_500`, `smoke_50`, `dev_100` (seed 42) · `docs/DATA_CARD.md` (source, citation, terms, label stats, imbalance)
  - Verify: parser unit tests, including `OTHERS` and malformed labels · loader returns 7,786 / 1,112 / 2,224 rows
  - Files: `scripts/download_data.py`, `src/review_insight/data/visfd.py`, `evals/data/samples.json`, `docs/DATA_CARD.md`, `tests/unit/test_visfd.py`
  - 💡 data provenance, leakage-free splits

- [x] **T5: Metrics module** (S) · deps: T3
  - Acceptance: aspect-detection F1 (micro and macro), pair F1 (micro), per-aspect P/R/F1 with support, urgency/routing accuracy, macro F1 and confusion matrix, latency p50/p95, cost per 1k
  - Verify: unit tests against small cases computed by hand, plus cross-checks against `sklearn.metrics`
  - Files: `evals/metrics.py`, `tests/unit/test_metrics.py`
  - 💡 multi-label evaluation, micro vs macro averaging

- [x] **T6: TF-IDF baseline and eval runner** (M) · deps: T4, T5
  - Acceptance: `evals/baseline_tfidf.py` trains on train, tunes on dev, scores `test_500` · `evals/run.py --config X --split Y --limit N` writes `evals/results/<date>_<config>_<split>.json` with metrics and run metadata (git sha, config, seed)
  - Verify: baseline numbers are produced and are in a plausible range vs the paper (aspect F1 ≈ 0.8)
  - Files: `evals/{baseline_tfidf,run}.py`, `evals/configs/baseline.yaml`, `tests/unit/test_{baseline_tfidf,run}.py`
  - Result: `test_500` pair F1 ≈ 0.74, aspect F1 micro ≈ 0.85, p95 ≈ 6 ms, $0 (C tuned on dev minus `dev_100`)
  - 💡 baselines, experiment tracking

- [x] **T7: Zero-shot Gemini analyzer** (M) · deps: T6 · ⚠️ highest-risk task
  - Acceptance: list available Gemini models and record their rate limits in `docs/` · `prompts/analyze_v1.yaml` (delimited untrusted input) · `ReviewAnalyzer` via LangChain `with_structured_output` · token usage and cost from `config/pricing.yaml` (filled from the official pricing page) · on-disk response cache for eval runs (re-running costs $0)
  - Verify: `evals/run.py --config evals/configs/zeroshot_31.yaml --split dev_100` for both models · unit tests use a fake model call
  - Files: `src/review_insight/llm/{analyzer,prompts,pricing}.py`, `src/review_insight/config.py`, `prompts/analyze_v1.yaml`, `config/pricing.yaml`, `evals/{model_cache,run}.py`, `evals/configs/zeroshot_*.yaml`, `docs/GEMINI.md`
  - 💡 structured output, prompt design, cost accounting

### ✅ Checkpoint A: go/no-go (tag `v0.1.0`)
- [x] Baseline vs zero-shot (2 models) table on `dev_100`, with pair F1, cost per 1k and p95

  | Config | Pair F1 | Aspect F1 micro / macro | Accepted | p50 / p95 | Cost per 1k |
  |---|---|---|---|---|---|
  | TF-IDF baseline | **0.752** | 0.868 / 0.755 | 100% | 5 ms / 8 ms | $0 |
  | zero-shot 3.1 | 0.686 | 0.750 / 0.657 | 78% | 1.5 s / 5.8 s | $0.42 |
  | zero-shot 3.5 | 0.729 | 0.807 / 0.773 | 88% | 1.5 s / 8.5 s | $0.59 |

  - Rejected outputs score as empty predictions. On accepted outputs only, pair F1 is 0.806 (3.1) and 0.795 (3.5).
  - Every schema failure (24 of 24) is a duplicated aspect: the model lists one entry per evidence span. Most ungrounded evidence (3.1) strips diacritics or joins spans with "...".
  - p95 is above the 3 s target. It was measured on the free tier during 503 "high demand" periods.
  - Transient 503/504 errors every ~10–20 requests; runs were resumed from the cache (retries are T10).
- [x] Rate limits known; eval budget per day computed (`docs/GEMINI.md`)
- [ ] **Review with you:** proceed as-is, adjust the prompt, or change model?

## Phase 2: Harden the LLM layer

- [ ] **T8: Retries, timeout, fallback and repair** (S) · deps: T7
  - Acceptance: tenacity with exponential backoff and jitter on 429/5xx (max 3) · 15 s timeout · fallback model after exhaustion · one repair retry when grounding or schema fails · typed errors `LLMUnavailable`, `InvalidLLMOutput`
  - Verify: unit tests with fake models that simulate 429, timeout and invalid JSON
  - Files: `src/review_insight/llm/{resilience,analyzer,errors}.py`, `tests/unit/test_resilience.py`
  - 💡 failure modes of LLM APIs

- [ ] **T9: Langfuse tracing and prompt management** (M) · deps: T7
  - Acceptance: every analyzer call is traced (model, prompt name and version, tokens, cost, latency) · `scripts/sync_prompts.py --label staging|production` pushes `prompts/*.yaml` · runtime loads the prompt by `PROMPT_LABEL`, falling back to the local YAML if Langfuse is down · eval runs logged as Langfuse dataset runs with scores
  - Verify: traces visible in Langfuse UI · unit test for fallback when the Langfuse client raises
  - Files: `src/review_insight/observability/langfuse.py`, `llm/prompts.py`, `scripts/sync_prompts.py`, `evals/run.py`
  - 💡 LLM observability, prompt versioning

- [ ] **T10: Few-shot strategies** (M) · deps: T7
  - Acceptance: `StaticFewShot(k=5)` (hand-picked, covers all aspects) · `DynamicFewShot(k=5)` with train embeddings from `gemini-embedding-2` precomputed once to `data/embeddings/train.npz` and cosine top-k · strategy chosen by config · test examples are never in the pool
  - Verify: unit tests with fake embeddings (deterministic top-k) · dev_100 run for each strategy
  - Files: `src/review_insight/llm/fewshot.py`, `scripts/embed_train.py`, `tests/unit/test_fewshot.py`, `evals/configs/*`
  - 💡 in-context learning, embeddings, semantic similarity

### ✅ Checkpoint B
- [ ] All 7 configs run on `dev_100` without crashing; traces in Langfuse
- [ ] Review with you: pick the default config for API development

## Phase 3: Serve

- [ ] **T11: Database layer** (M) · deps: T2
  - Acceptance: SQLAlchemy async models for the 4 tables in SPEC §4.3 · Alembic initial migration · `docker-compose.yml` with a local `postgres` service · repository functions · works against Supabase (session pooler, or `statement_cache_size=0`)
  - Verify: `uv run alembic upgrade head` on local and Supabase · integration test inserts and reads back
  - Files: `src/review_insight/db/{models,session,repositories}.py`, `migrations/`, `docker-compose.yml`, `tests/integration/test_repositories.py`
  - 💡 schema design for LLM outputs (jsonb + flattened table)

- [ ] **T12: API foundation** (S) · deps: T11
  - Acceptance: `X-API-Key` auth dependency · request-id middleware (`X-Request-ID`) · error envelope and exception handlers mapping to SPEC §5 codes · `/readyz` checks the DB
  - Verify: tests for 401, 422 envelope, request-id echo and readyz when the DB is down
  - Files: `src/review_insight/api/{deps,errors,middleware,health}.py`, `tests/unit/test_api_errors.py`

- [ ] **T13: `POST /v1/analyze` end to end** (M) · deps: T8, T9, T12
  - Acceptance: validates input (1–2,000 chars) → analyzer → persists review, analysis and aspects → response per SPEC §5, with `meta` and `X-Trace-ID`
  - Verify: integration test with a fake analyzer and real Postgres · manual `curl` against real Gemini
  - Files: `src/review_insight/api/analyze.py`, `services/analysis_service.py`, `tests/integration/test_analyze.py`

- [ ] **T14: Exact-match cache** (S) · deps: T13
  - Acceptance: key `(sha256(normalized text), model, prompt_version)` · a hit returns the stored result with `cached: true` and no LLM call
  - Verify: test asserts the fake analyzer is called once for two identical requests · server-side hit latency < 100 ms
  - Files: `services/analysis_service.py`, `db/repositories.py`, `tests/integration/test_cache.py`
  - 💡 caching and idempotency for LLM calls

- [ ] **T15: Batch jobs** (M) · deps: T14
  - Acceptance: `POST /v1/batch` (≤1,000 items) → 202 + `job_id` · background processing bounded by `BATCH_CONCURRENCY` · per-item failure does not fail the job · `GET /v1/batch/{id}` shows status and counts · on startup, `running` jobs are marked `failed`
  - Verify: integration test with 20 items, 2 of them forced to fail · startup-recovery test
  - Files: `src/review_insight/api/batch.py`, `services/batch_service.py`, `tests/integration/test_batch.py`
  - 💡 concurrency control under rate limits

- [ ] **T16: `GET /v1/insights`** (S) · deps: T13
  - Acceptance: counts by urgency, top negative aspects and daily negative share per aspect, filtered by date and `product_id`, computed in SQL
  - Verify: integration test on seeded data with known expected aggregates
  - Files: `src/review_insight/api/insights.py`, `db/queries.py`, `tests/integration/test_insights.py`

- [ ] **T17: CORS, rate limiting and OpenAPI contract** (S) · deps: T13
  - Acceptance: `CORS_ORIGINS` allowlist · per-API-key rate limit → 429 envelope · `tests/contract/openapi.snapshot.json`, where any change requires a deliberate snapshot update
  - Verify: tests for preflight, 429 and snapshot diff
  - Files: `src/review_insight/main.py`, `api/ratelimit.py`, `tests/contract/test_openapi.py`

### ✅ Checkpoint C (tag `v0.2.0`)
- [ ] `docker compose up` plus `alembic upgrade head` → analyze, batch and insights all work locally against real Gemini
- [ ] Coverage ≥ 80%; review the API with you

## Phase 4: Evaluate for real

- [ ] **T18: Hand-labelled urgency/routing set and adversarial set** (S) · 🧑 you label · can start after T4
  - Acceptance: export 130 reviews from `test_500` to a CSV, label them in a spreadsheet using the SPEC §4.2 rubric, then convert to `evals/data/urgency_routing_gold.jsonl` · plus 20 adversarial reviews (prompt injection, teencode, emoji-only, mixed languages) · a note on ambiguous cases and how you decided
  - Verify: schema-validated by a test · distribution of labels reported
  - Files: `scripts/export_for_labeling.py`, `evals/data/*.jsonl`, `evals/LABELING.md`
  - 💡 annotation guidelines, label quality

- [ ] **T19: Full experiment matrix** (M) · deps: T10, T18
  - Acceptance: all 7 configs on `test_500` plus the gold sets · selection rule from SPEC §7.3 applied in code (not by eye) · winning config written to `evals/baseline.json` · final run of the winner on the full test split
  - Verify: re-running uses the response cache (0 new API calls) · results JSON committed
  - Files: `evals/{run,select}.py`, `evals/results/*.json`, `evals/baseline.json`
  - 💡 controlled experiments, pre-registered decision rules

- [ ] **T20: Evaluation report** (S) · deps: T19 · 🧑 you write the error analysis
  - Acceptance: `evals/report.py` generates the tables in `evals/REPORT.md` (comparison, per-aspect, confusion matrices, cost/latency) · you write 10+ annotated error examples and the conclusion · comparison with the paper's Bi-LSTM numbers
  - Verify: the report regenerates deterministically from the results files
  - Files: `evals/report.py`, `evals/REPORT.md`
  - 💡 error analysis

### ✅ Checkpoint D (tag `v0.3.0`)
- [ ] Production config chosen by rule; `.env` defaults updated · review the report with you

## Phase 5: Ship

- [ ] **T21: Production Dockerfile** (S) · deps: Checkpoint C
  - Acceptance: multi-stage build with uv, non-root user, `HEALTHCHECK`, slim image · compose runs the api image together with postgres
  - Verify: `docker build` passes · container serves `/healthz` · image size noted in the README
  - Files: `Dockerfile`, `.dockerignore`, `docker-compose.yml`

- [ ] **T22: CI eval gate** (S) · deps: T19, T21
  - Acceptance: a CI job, path-filtered to `prompts/`, `src/review_insight/llm/` and `config/`, runs `smoke_50` with the production config and fails per SPEC §7.5 · `GOOGLE_API_KEY` and Langfuse keys stored as GH secrets · Docker build job on every PR
  - Verify: a demo PR with a deliberately broken prompt is blocked; reverting it passes
  - Files: `.github/workflows/ci.yml`, `evals/gate.py`
  - 💡 evals as CI, regression testing for prompts

- [ ] **T23: Azure deployment and CD** (M) · deps: T21 · ⚠️ ask first: spends Azure credit
  - Acceptance: `docs/deploy.md` with the exact `az` commands (resource group, ACA environment, app with `min-replicas 0`, secrets) · GitHub OIDC federated credential (no stored Azure password) · `cd.yml`: on tag `v*`, build → push to GHCR → `az containerapp update` · migrations run as a step before rollout
  - Verify: tagging `v0.4.0` deploys automatically · public `/healthz` and `/v1/analyze` work · warm and cold latency measured
  - Files: `.github/workflows/cd.yml`, `docs/deploy.md`
  - 💡 serving, deployment, environment parity

- [ ] **T24: Docs and v1.0.0** (S) · deps: T20, T23
  - Acceptance: README with problem, architecture (mermaid), results table, trade-offs, limitations, quickstart, citation and a Vietnamese summary · ADRs 0001–0004 · `docs/architecture.md` · `CHANGELOG.md` · tag `v1.0.0`
  - Verify: a fresh clone following only the README reaches a working local stack
  - Files: `README.md`, `docs/adr/*`, `docs/architecture.md`, `CHANGELOG.md`

### ✅ Checkpoint E: Done
- [ ] Every item in SPEC §14 checked off, with evidence links in the release notes
