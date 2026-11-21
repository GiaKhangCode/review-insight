# Gemini models, pricing and limits

Facts the LLM configs depend on, recorded on **2026-11-21**. Check them again before any
cost claim in the README, because Google changes prices and free-tier quotas without notice.

## 1. Models available to the project key

From `client.models.list()` (google-genai SDK):

| Model | Role (SPEC §3) | Input limit | Output limit |
|---|---|---|---|
| `gemini-3.1-flash-lite` | `PRIMARY_MODEL` | 1,048,576 | 65,536 |
| `gemini-3.5-flash-lite` | `FALLBACK_MODEL` and comparison model | 1,048,576 | 65,536 |
| `gemini-embedding-2` | dynamic few-shot retrieval (T8) | 8,192 | n/a |

A review is roughly 50–200 tokens, so context size is not a constraint.

## 2. Pricing (Standard paid tier, USD per 1M tokens)

Source: <https://ai.google.dev/gemini-api/docs/pricing> (redirects to
`aistudio.google.com/docs/pricing`). Machine-readable copy: `config/pricing.yaml`.

| Model | Input (text) | Output, including thinking tokens |
|---|---|---|
| `gemini-3.1-flash-lite` | $0.25 | $1.50 |
| `gemini-3.5-flash-lite` | $0.30 | $2.50 |
| `gemini-embedding-2` | $0.20 | n/a |

- Output dominates the cost: 3.5 costs about 1.7× as much as 3.1 per output token.
- **Thinking tokens are billed as output.** Their count is recorded per call, so it can be
  limited if it turns out to drive cost.
- The embedding price came from a part of the page that could only be read as a summary.
  Verify it in T8 before using it.
- **Reported costs use paid-tier prices even though the project runs on the free tier.**
  The goal is the cost of running this in production, not the $0 we happen to pay.

## 3. Rate limits (this project, Free tier, from AI Studio)

The docs state that limits "depend on a variety of factors (such as your usage tier) and can
be viewed in Google AI Studio". They apply **per project** (not per key), each model has its
own quota, and RPD resets at **midnight Pacific time**.

| Model | RPM | TPM | RPD |
|---|---|---|---|
| `gemini-3.1-flash-lite` | 15 | 250,000 | 500 |
| `gemini-3.5-flash-lite` | 15 | 250,000 | 500 |
| `gemini-embedding-2` | not checked yet (T8) | | |

### What binds
- **RPD is the real constraint.** At 15 RPM a full day's quota takes about 34 minutes.
- **RPM:** one request every 4 s. The runner throttles itself to about 12 RPM (one request
  every 5 s) instead of relying on HTTP 429 retries. Concurrency above 1 does not help.
- **TPM** does not bind. Even a 2,000-token few-shot prompt × 15 RPM is 30K TPM.
- Forum reports describe 429 errors below the displayed limits, so retries with backoff
  (T10) are still needed.

## 4. Eval budget

Every (model, prompt version, review) is called **once**: responses are cached on disk, so
re-scoring, changing a metric or regenerating the report costs no quota. A schema-repair retry
is an extra request.

| Run | Requests per model | Days at 500 RPD |
|---|---|---|
| Checkpoint A: zero-shot on `dev_100` | 100 | 0.2 |
| One prompt iteration on `dev_100` | 100 | 0.2 (so at most ~4 iterations a day) |
| 3 configs (zero-shot, static, dynamic) on `test_500` | 1,500 | 3 |
| Urgency/routing set (150 reviews) × 3 configs | 450 | ~1 |
| CI smoke eval (`smoke_50`) | 50 per PR touching prompts | 0.1 |

- The two models have separate quotas, so they run in parallel. The full comparison on
  `test_500` takes about **3 days** of quota, plus a margin for retries.
- The full test split (2,224 reviews) is run once, for the production config only: about
  **4.5 days** on one model. It is scheduled last.
- Local development and the deployed API **share the same quota**. Leave headroom on days
  when an eval run is planned.

## 5. Data use on the free tier

On the free tier, Google may use prompts and responses to improve its products. That is
acceptable here: the inputs are reviews from a public research dataset. A deployment that
handles real customer reviews would need the paid tier.
