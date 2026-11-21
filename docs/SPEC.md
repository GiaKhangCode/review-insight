# Spec: ReviewInsight — LLM-powered Vietnamese Customer Feedback Analysis

> Status: **DRAFT v0.1** — awaiting review
> Owner: Khang · Last updated: 2026-11-21
> Project 1 of 3 in the AI Engineering portfolio (simple → complex).

---

## 1. Objective

### 1.1 Problem

A Vietnamese online phone retailer receives thousands of product reviews and customer messages every week across its e-commerce storefronts. Today:

- Nobody reads them all, so recurring product defects (battery, overheating, screen) are noticed weeks late.
- Urgent complaints (defective unit, refund demand, threat of public escalation) sit in the same queue as "great phone, 5 stars" and get answered late.
- Management has no weekly view of *what* customers are unhappy about, only an average star rating.

### 1.2 Solution

An HTTP service that turns raw Vietnamese review text into **structured, validated JSON**:

- **Aspect-based sentiment**: which product aspects are mentioned (battery, camera, price…) and the sentiment of each, with the supporting text span.
- **Urgency** (`low | medium | high`) and **routing** (which team should handle it).
- **Aggregated insights** over a date range (top negative aspects, urgent-ticket count, trend).

Single requests are synchronous. Bulk imports run as an async batch job.

### 1.3 Users

| User | Need |
|---|---|
| Customer-service lead | Sees urgent tickets first, routed to the right team |
| Product manager | Weekly report of negative aspects per product |
| Developer integrating the service | Stable, documented REST API with predictable errors |

### 1.4 Why an LLM (and proof that it is worth it)

A supervised TF-IDF + Logistic Regression baseline is trained on the same data. The project must show, with numbers, where the LLM wins (zero training data needed for new fields such as urgency and routing, evidence spans, robustness to slang and teencode) and where it loses (cost, latency, possibly raw F1 on fixed labels). **If the baseline beats the LLM on aspect F1, the README reports that honestly.** That finding is a valid result, not a failure.

### 1.5 AI Engineering concepts practiced

Prompt engineering (zero-shot, few-shot, dynamic few-shot) · structured output and schema validation · embeddings for example selection · offline evaluation against ground truth · hand annotation with a rubric · cost/latency/quality trade-off analysis · tracing and prompt versioning (Langfuse) · caching · retries, timeouts, fallback model · treating user text as untrusted input · API serving · containerization · CI with an eval gate · cloud deployment.

---

## 2. Scope

### Must have (v1.0)
1. `POST /v1/analyze`: analyze one review synchronously.
2. `POST /v1/batch` and `GET /v1/batch/{job_id}`: async batch of up to 1,000 reviews.
3. `GET /v1/insights`: aggregates over a date range.
4. Persistence of reviews, analyses and jobs in Supabase Postgres.
5. Exact-match result cache keyed by `(text_hash, model, prompt_version)`.
6. Langfuse tracing on every LLM call, plus prompts managed and versioned in Langfuse.
7. Offline evaluation harness and a results report comparing at least 7 configurations (see §7).
8. Docker image, `docker compose` local stack, deployment to Azure Container Apps.
9. GitHub Actions CI: lint, type check, tests, eval smoke gate. CD on version tag.

### Should have
- CORS allowlist (`CORS_ORIGINS`) and a stable OpenAPI schema, so the future Next.js frontend can generate typed clients (`openapi-typescript`).
- Per-API-key rate limiting.
- Adversarial set (prompt injection inside review text) included in the eval.

### Won't have (this project)
- Frontend UI in v1.0: a Next.js app will be built *after* the API is done, as a separate milestone with its own spec.
- RAG, agents, multi-turn chat (covered by Projects 2 and 3).
- User accounts or OAuth: a static API key per environment is enough.
- PII redaction (covered in Project 3).
- Fine-tuning.

---

## 3. Tech Stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11 | |
| Package manager | `uv` | Lockfile committed |
| API | FastAPI + Pydantic v2 + Uvicorn | |
| LLM orchestration | LangChain (`langchain-google-genai`) | `with_structured_output(PydanticModel)` |
| Models | `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite` | Configured by env, never hard-coded |
| Embeddings | `gemini-embedding-2` | Only for dynamic few-shot example selection, in-memory cosine (no vector DB) |
| Observability | Langfuse Cloud (Hobby tier), confirmed | Tracing, prompt management, datasets, experiment scores |
| Database | Supabase (managed Postgres) | Accessed with SQLAlchemy 2 (async) + asyncpg; migrations with Alembic |
| Retry | `tenacity` | |
| Baseline ML | scikit-learn | TF-IDF + one-vs-rest Logistic Regression |
| Frontend (later) | Next.js | Separate milestone after v1.0 API; consumes OpenAPI-generated types |
| Container | Docker (multi-stage, non-root) | |
| Registry | GitHub Container Registry (GHCR) | Free; saves Azure credit |
| Hosting | Azure Container Apps (Azure for Students) | `min-replicas=0` to scale to zero |
| CI/CD | GitHub Actions | Azure login via OIDC (no long-lived secrets) |
| Quality | ruff, ruff-format, mypy (strict on `src/`), pytest (+ pytest-asyncio, pytest-cov), pre-commit, gitleaks | |
| Test HTTP client | `httpx2` (dev only) | Required by Starlette's `TestClient`; replaces `httpx`, which Starlette deprecated (approved 2026-11-21) |

Exact library versions are pinned in `uv.lock`. Model pricing lives in `config/pricing.yaml`, filled from the official Gemini pricing page at implementation time. **No prices are hard-coded from memory.**

---

## 4. Data

### 4.1 Labeled dataset: UIT-ViSFD
Vietnamese Smartphone Feedback Dataset from UIT (Vietnamese e-commerce smartphone reviews, about 11k comments, 10 aspects × 3 sentiments, official train/dev/test split).

Aspects: `SCREEN, CAMERA, FEATURES, BATTERY, PERFORMANCE, STORAGE, DESIGN, PRICE, GENERAL, SER&ACC`. Sentiments: `positive, negative, neutral`.

- **train** → baseline training and the pool of few-shot examples (never used for scoring).
- **dev** → prompt iteration.
- **test** → final reported numbers only. A fixed, stratified 500-review sample (`seed=42`) is used for routine runs because of API rate limits. The full test split is used once for the final report.

**Source (verified 2026-11-21):** `https://github.com/LuongPhan/UIT-ViSFD/raw/main/UIT-ViSFD.zip` (≈940 KB). It contains `Train.csv` (7,786), `Dev.csv` (1,112) and `Test.csv` (2,224), with columns `index, comment, n_star, date_time, label`.

- Label format: `{BATTERY#Positive};{CAMERA#Negative};`. The special label `{OTHERS};` means no aspect is mentioned, and maps to an empty `aspects` list.
- **Known imbalance** (train): `STORAGE` has only 91 labels in total and `SCREEN#Neutral` only 56. Macro-F1 is therefore noisy, which is why micro pair F1 is the primary metric and per-aspect support is always reported.
- **Terms:** free for research use, and the paper must be cited (SA2SL, KSEM 2021). Commercial use requires contacting the authors. This is a non-commercial portfolio project, so the README cites the paper, and the raw data is **not** committed. `scripts/download_data.py` fetches it and checks its SHA-256.

### 4.2 Hand-labeled set: urgency and routing
ViSFD has no urgency or routing labels, so we create **150 reviews** (sampled from ViSFD test plus 20 adversarial examples), labeled by hand against the rubric below. They are stored in `evals/data/urgency_routing_gold.jsonl`.

**Urgency rubric**
| Level | Definition | Example signal |
|---|---|---|
| `high` | Defect or safety issue, refund/return demand, or threat to escalate publicly | "pin phồng", "máy nóng ran", "đòi hoàn tiền", "sẽ báo lên…" |
| `medium` | Clear negative experience without a demand for action | "camera chụp tối", "giao hàng chậm" |
| `low` | Positive, neutral or mixed with no complaint requiring follow-up | "máy đẹp, pin trâu" |

**Routing**: `product` (hardware or software aspects) · `customer_service` (service, delivery, accessories, warranty) · `sales_pricing` (price, promotion) · `none` (no action needed). When several apply, route by the highest-urgency issue.

### 4.3 Database schema (Postgres)

```
reviews          (id uuid pk, text text, text_hash char(64) unique, source text, product_id text null, created_at timestamptz)
analyses         (id uuid pk, review_id fk, model text, prompt_name text, prompt_version int,
                  result jsonb, urgency text, route_to text,
                  input_tokens int, output_tokens int, cost_usd numeric, latency_ms int,
                  langfuse_trace_id text, created_at timestamptz,
                  unique(review_id, model, prompt_version))
analysis_aspects (analysis_id fk, aspect text, sentiment text)       -- flattened for aggregation
batch_jobs       (id uuid pk, status text, total int, succeeded int, failed int,
                  created_at timestamptz, finished_at timestamptz null)
```

---

## 5. API Contract

All endpoints are under `/v1` and require the `X-API-Key` header, except `/healthz` and `/readyz`. Every response carries an `X-Request-ID` header, and LLM responses also carry `X-Trace-ID` (the Langfuse trace id).

### `POST /v1/analyze`
Request:
```json
{ "text": "Máy đẹp, pin trâu nhưng camera chụp đêm hơi tệ, shop giao chậm", "source": "shopee", "product_id": "ip15-128" }
```
Response `200`:
```json
{
  "review_id": "6f1c…",
  "aspects": [
    { "aspect": "DESIGN",  "sentiment": "positive", "evidence": "Máy đẹp" },
    { "aspect": "BATTERY", "sentiment": "positive", "evidence": "pin trâu" },
    { "aspect": "CAMERA",  "sentiment": "negative", "evidence": "camera chụp đêm hơi tệ" },
    { "aspect": "SER&ACC", "sentiment": "negative", "evidence": "shop giao chậm" }
  ],
  "urgency": "medium",
  "route_to": "customer_service",
  "meta": { "model": "gemini-3.1-flash-lite", "prompt_version": 3, "cached": false, "latency_ms": 840 }
}
```
Validation: `text` is 1–2,000 characters after trimming; `evidence` must be a substring of the input (checked server-side; failures trigger a repair retry).

### `POST /v1/batch` → `202 { "job_id": "…" }`; `GET /v1/batch/{job_id}` → status, counts, and a results URL when done
Concurrency is bounded by an `asyncio.Semaphore` (`BATCH_CONCURRENCY`, default 5). Jobs run in-process. **Known limitation:** jobs in progress are lost on restart and marked `failed` at startup. This is documented, not hidden.

### `GET /v1/insights?from=2026-11-01&to=2026-11-21&product_id=…`
Returns count by urgency, the top negative aspects with counts, and daily negative-share per aspect.

### Errors
JSON body `{ "error": { "code": "...", "message": "...", "request_id": "..." } }`.
| HTTP | code | When |
|---|---|---|
| 401 | `unauthorized` | Missing or invalid API key |
| 422 | `invalid_input` | Validation failed |
| 429 | `rate_limited` | Client exceeded its quota |
| 502 | `llm_invalid_output` | Schema still invalid after the repair retry |
| 503 | `llm_unavailable` | Primary and fallback models both failed or timed out |

---

## 6. LLM Pipeline Design

```
text ─► normalize (NFC, trim, collapse whitespace) ─► hash ─► cache lookup ──hit──► return
                                                         │miss
                                                         ▼
                 select few-shot examples (static | dynamic via embeddings)
                                                         ▼
           prompt (from Langfuse, label=production) ─► Gemini (structured output)
                     │  timeout 15s, retry ×3 on 429/5xx (exp. backoff + jitter)
                     │  on final failure ─► fallback model
                                                         ▼
             validate (Pydantic + evidence ⊂ text) ──fail──► 1 repair retry ──fail──► 502
                                                         ▼
                           persist + Langfuse scores ─► return
```

- **Prompt-injection posture**: review text is untrusted data. It is wrapped in explicit delimiters, the system prompt states that instructions inside the review must be ignored, and the output is constrained by schema. The adversarial eval set measures whether this holds.
- **Prompt versioning**: prompts live in Langfuse and are mirrored in `prompts/*.yaml` in git (the source of truth for review). `scripts/sync_prompts.py` pushes a new version. The served version is pinned via env `PROMPT_LABEL`.
- **Configuration** (env, validated with `pydantic-settings`): `PRIMARY_MODEL`, `FALLBACK_MODEL`, `FEWSHOT_STRATEGY`, `FEWSHOT_K`, `PROMPT_LABEL`, `LLM_TIMEOUT_S`, `BATCH_CONCURRENCY`, `DATABASE_URL`, `GOOGLE_API_KEY`, `LANGFUSE_*`, `API_KEYS`, `CORS_ORIGINS`.

---

## 7. Evaluation

### 7.1 Experiment matrix (same fixed 500-review test sample for all)
| # | Config |
|---|---|
| B | TF-IDF + LR baseline (trained on train split) |
| 1–2 | Zero-shot × {3.1-flash-lite, 3.5-flash-lite} |
| 3–4 | Static few-shot (k=5) × both models |
| 5–6 | Dynamic few-shot (k=5 nearest train examples via `gemini-embedding-2`) × both models |

### 7.2 Metrics
| Metric | Definition |
|---|---|
| Aspect detection F1 (micro and macro) | Is the aspect mentioned? (multi-label) |
| Aspect-sentiment pair F1 (micro) | Strict match on the (aspect, sentiment) pair. **Primary metric.** |
| Per-aspect F1 | Used for error analysis |
| Urgency accuracy, macro F1, and confusion matrix | On the 150 hand-labeled reviews |
| Routing accuracy | On the 150 hand-labeled reviews |
| Schema validity rate | Valid on first try / after the repair retry |
| Evidence grounding rate | Share of evidence spans that are exact substrings |
| Injection resistance | Share of adversarial reviews with schema-valid, on-task output |
| Latency p50/p95 | Per request, measured client-side |
| Cost per 1,000 reviews (USD) | From token counts × `pricing.yaml` |

### 7.3 Production-config selection rule (decided **before** seeing results)
Choose the config with the highest pair F1 among those with p95 latency ≤ 3 s and cost ≤ 2× the cheapest LLM config. Ties go to the cheaper config.

### 7.4 Artifacts
- `evals/results/<date>_<config>.json`: raw metrics per run. Each run is also logged as a Langfuse dataset run.
- `evals/REPORT.md`: comparison table, per-aspect breakdown, 10+ annotated error examples, and the conclusion.
- `evals/baseline.json`: the metrics of the current production config, used by the CI gate.

### 7.5 CI eval gate
On PRs touching `prompts/`, `src/review_insight/llm/` or `config/`, CI runs a **50-review smoke eval**. It fails if pair F1 drops by more than 0.03 or schema validity falls below 98% relative to `evals/baseline.json`.

---

## 8. Commands

```bash
uv sync                                                       # install deps
uv run alembic upgrade head                                   # apply DB migrations
uv run uvicorn review_insight.main:app --reload               # dev server on :8000
uv run pytest --cov=review_insight --cov-fail-under=80        # tests + coverage
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run python scripts/download_data.py                        # fetch UIT-ViSFD
uv run python -m evals.baseline_tfidf                         # train and score the baseline
uv run python -m evals.run --config evals/configs/fewshot_dynamic_31.yaml --split test --limit 500
uv run python -m evals.report                                 # regenerate evals/REPORT.md
uv run python scripts/sync_prompts.py --label staging         # push prompts to Langfuse
docker compose up --build                                     # api + postgres locally
```

---

## 9. Project Structure

```
review-insight/
├── src/review_insight/
│   ├── main.py              # FastAPI app factory, middleware, routers
│   ├── config.py            # pydantic-settings
│   ├── api/                 # routers: analyze, batch, insights, health; error handlers; auth
│   ├── domain/              # Pydantic schemas (Aspect, AnalysisResult…), enums, business rules
│   ├── llm/                 # analyzer, prompt loading, few-shot selectors, retry/fallback
│   ├── db/                  # SQLAlchemy models, repositories, session
│   └── observability/       # Langfuse setup, logging (JSON), request-id middleware
├── prompts/                 # versioned prompt templates (YAML)
├── config/pricing.yaml
├── migrations/              # Alembic
├── evals/
│   ├── configs/  data/  results/
│   ├── run.py  metrics.py  baseline_tfidf.py  report.py
│   └── REPORT.md
├── scripts/
├── tests/unit/  tests/integration/
├── docs/SPEC.md  docs/adr/  docs/architecture.md
├── .github/workflows/ci.yml  cd.yml
├── Dockerfile  docker-compose.yml  .env.example  .pre-commit-config.yaml
├── pyproject.toml  uv.lock  CHANGELOG.md  README.md  LICENSE
```

---

## 10. Code Style

- ruff (lint + format, line length 100), mypy strict on `src/`, type hints everywhere.
- Code, comments, commits and docs are in English. A short Vietnamese summary goes in the README.
- Layered dependencies: `api → domain ← llm, db`. `domain` imports nothing from FastAPI, LangChain or SQLAlchemy.
- The LLM is behind a protocol so tests never hit the network.

```python
class ReviewAnalyzer(Protocol):
    async def analyze(self, text: str) -> AnalysisResult: ...


class GeminiReviewAnalyzer:
    """Analyzes one review with structured output, retries, and model fallback."""

    def __init__(
        self, primary: BaseChatModel, fallback: BaseChatModel, prompt: PromptBundle
    ) -> None:
        self._chain = prompt.template | primary.with_structured_output(LLMAnalysis)
        self._fallback_chain = prompt.template | fallback.with_structured_output(LLMAnalysis)

    async def analyze(self, text: str) -> AnalysisResult:
        raw = await self._invoke_with_fallback({"review": text, "examples": self._examples(text)})
        return validate_grounded(raw, source_text=text)  # raises InvalidLLMOutput
```

---

## 11. Testing Strategy

| Level | What | Tooling |
|---|---|---|
| Unit | Schemas, evidence grounding, metrics math, cache keys, few-shot selection, error mapping | pytest; fake analyzer and fake embeddings |
| Integration | API ↔ DB with a real Postgres container; migrations apply cleanly | pytest + `docker compose` / GH Actions service container; LLM faked |
| Contract | OpenAPI schema snapshot, so breaking changes fail the test | pytest |
| Eval (not unit tests) | Real LLM calls, scored on labeled data | `evals/`, run manually and as the CI smoke gate |

Coverage must be at least 80% on `src/`. Every bug fix comes with a regression test.

---

## 12. Git and Delivery Workflow

- Separate public GitHub repo `review-insight`. `main` is protected and receives changes only via PRs (squash merge).
- Branches: `feat/…`, `fix/…`, `chore/…`, `docs/…`, `eval/…`.
- [Conventional Commits](https://www.conventionalcommits.org/) (`feat(api): add batch endpoint`). Commits are authored solely by the repo owner, with no AI co-author trailers.
- Semantic versioning tags (`v0.1.0` … `v1.0.0`), each with a `CHANGELOG.md` entry. A tag triggers CD: build → push to GHCR → deploy to ACA.
- Architecture decisions are recorded as ADRs in `docs/adr/` (e.g. *0001-supabase-as-plain-postgres*, *0002-in-process-batch-jobs*).

---

## 13. Boundaries

- **Always:** run lint, types and tests before committing · keep secrets in `.env` (gitignored) and ACA secrets · update this spec when a decision changes · use the fixed eval sample and seed so runs are comparable · log the model and prompt version with every analysis.
- **Ask first:** adding a dependency outside §3 · changing the DB schema or API contract · changing eval metrics, gate thresholds or the selection rule · anything that spends Azure credit beyond ACA scale-to-zero.
- **Never:** commit secrets, raw datasets or `.env` · tune prompts on the test split · edit `evals/baseline.json` to make the gate pass · delete or skip failing tests to get CI green · add AI attribution to commits.

---

## 14. Success Criteria (definition of done for v1.0.0)

1. A public ACA URL serves `/healthz` and `/v1/analyze`. Warm p95 latency is ≤ 3 s for a single review (cold start measured and reported separately).
2. Schema validity is ≥ 99% after the repair retry on the 500-review test sample.
3. `evals/REPORT.md` compares all 7 configs on identical data, with pair F1, urgency/routing accuracy, cost per 1,000 reviews and p50/p95 latency. The production config is chosen by the §7.3 rule.
4. 150 hand-labeled urgency/routing examples are committed together with the rubric.
5. Every LLM call appears in Langfuse with model, prompt version, tokens and cost. The trace id is returned in `X-Trace-ID`.
6. A cache hit returns in < 100 ms (server-side) and makes no LLM call.
7. From a clean clone, `cp .env.example .env` (filled in) plus `docker compose up` runs the full stack.
8. CI is green on `main`: lint, mypy, tests at ≥ 80% coverage, and the eval smoke gate.
9. The README contains the problem statement, an architecture diagram, the results table, trade-offs, known limitations and run instructions.
10. The history shows PR-based Conventional Commits, an ADR for each key decision, and tag `v1.0.0`.

---

## 15. Open Questions

1. **Dataset source.** Crawling Lazada was considered and is not recommended (ToS restrictions on automated access, reviewer personal data, no labels). UIT-ViSFD is confirmed as the labeled core (see §4.1). Still pending: whether to add UIT-ViOCD (complaint detection) as extra signal for urgency.
2. **Gemini free-tier limits** for the chosen models (RPM/RPD) determine whether the full test split fits in one day, and whether the Gemini Batch API is needed for eval runs.

Resolved: git email is verified on GitHub · no Streamlit dashboard, Next.js frontend after the API · Langfuse Cloud.
