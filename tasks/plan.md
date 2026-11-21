# Implementation Plan: ReviewInsight v1.0

> Source of truth for *what* to build: [docs/SPEC.md](../docs/SPEC.md). This file covers *in what order* and *why*.
> Task details (acceptance criteria, verification, files) live in [tasks/todo.md](todo.md).

## Overview

We build the service in 6 phases. The riskiest unknown is whether Gemini Flash-Lite does Vietnamese aspect-based sentiment well enough and cheaply enough within free-tier limits. We answer that in **Phase 1, before writing any API code**, by building the data loader, metrics, the TF-IDF baseline and a zero-shot LLM analyzer, then comparing them on the dev split. Everything after that (reliability, serving, deployment) is well-understood engineering.

## Architecture Decisions

| # | Decision | Rationale | ADR |
|---|---|---|---|
| 1 | Eval harness before API | Fail fast on the main risk; the API wraps an analyzer whose quality is already measured | — |
| 2 | `ReviewAnalyzer` protocol, LLM injected | Unit tests never call the network; eval runner and API share the same analyzer | — |
| 3 | Supabase used as plain Postgres via SQLAlchemy + Alembic | Portable, migration history in git, same code runs on a local Postgres container in CI | 0001 |
| 4 | Batch jobs in-process (`asyncio.Semaphore`) | No Redis/Celery for v1. The restart limitation is documented | 0002 |
| 5 | Prompts in git (`prompts/*.yaml`), synced to Langfuse | Git is reviewable via PRs. Langfuse provides runtime versioning and labels | 0003 |
| 6 | Dynamic few-shot uses precomputed train embeddings in a `.npz` file, with in-memory cosine | About 7.8k vectors fit in memory; a vector DB would be overkill (and comes in Project 2) | — |
| 7 | GHCR + Azure Container Apps with scale-to-zero | Free registry; idle cost close to $0 on student credit | 0004 |

## Dependency Graph

```
T1 tooling ─► T2 app skeleton + CI
                 │
     T3 domain schemas ─┬─► T4 data loader ─► T6 baseline + eval runner ─► T7 zero-shot LLM ══► CHECKPOINT A (go/no-go)
                        └─► T5 metrics ─────┘                                      │
                                                     T8 reliability ◄──────────────┤
                                                     T9 Langfuse    ◄──────────────┤
                                                     T10 few-shot   ◄──────────────┘
                                                            ║ CHECKPOINT B
     T11 DB + migrations ─► T12 API foundation ─► T13 /analyze ─► T14 cache ─► T15 batch ─► T16 insights ─► T17 CORS/ratelimit/contract
                                                            ║ CHECKPOINT C
     T18 hand labels (you) ─► T19 full experiment matrix ─► T20 report
                                                            ║ CHECKPOINT D
     T21 Docker ─► T22 CI eval gate ─► T23 Azure CD ─► T24 docs + v1.0.0
                                                            ║ CHECKPOINT E (= spec §14)
```

Phases 3 and 4 are mostly independent after Checkpoint B. T18 (hand labelling) is human work and can run **in parallel** with Phase 3.

## Phases at a glance

| Phase | Tasks | Outcome | Rough effort |
|---|---|---|---|
| 0. Bootstrap | T1–T2 | Repo, tooling, `/healthz`, CI green | 0.5–1 day |
| 1. Measure first | T3–T7 | Baseline vs zero-shot numbers on dev → go/no-go | 2–3 days |
| 2. Harden the LLM layer | T8–T10 | Retries, fallback, Langfuse tracing and prompts, few-shot | 2 days |
| 3. Serve | T11–T17 | Full API on local docker compose | 3–4 days |
| 4. Evaluate for real | T18–T20 | 7-config report, production config chosen | 2–3 days |
| 5. Ship | T21–T24 | Deployed on Azure, CD on tag, v1.0.0 | 2 days |

Total: roughly 2.5–3 weeks part-time.

## Git workflow per task

1. `git switch -c feat/<task-slug>` from an up-to-date `main`.
2. Small Conventional Commits (`feat(llm): add zero-shot analyzer`), authored by you only.
3. Push, open a PR whose description links the task (`Implements T7, SPEC §6`), wait for CI, then squash-merge.
4. Tags: `v0.1.0` after Checkpoint A, `v0.2.0` after C, `v0.3.0` after D, `v1.0.0` at E.

## Prerequisites (before T1)

| Item | Status |
|---|---|
| `uv` installed (`powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"`) | ❌ not installed |
| GitHub CLI `gh` (optional, for PRs from the terminal) | ❌ not installed |
| Azure CLI `az` (needed by T23) | ❌ not installed |
| Docker Desktop | ✅ 29.8 |
| Git identity verified on GitHub | ✅ |
| Google AI Studio API key | ☐ |
| Langfuse Cloud project (public + secret key) | ☐ |
| Supabase project (DB connection string) | ☐ needed by T11 |
| Azure for Students subscription active | ☐ needed by T23 |

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Gemini free-tier RPM/RPD too low for eval runs | High | Measure in T7. Use the fixed 500-sample set, cache all eval responses on disk, and use the Gemini Batch API if needed |
| Model ids in the spec are unavailable or renamed | Med | T7 lists available models via the API first. Model ids are env-configured |
| LLM loses to the TF-IDF baseline on pair F1 | Low (for the project) | Report it honestly. The LLM still adds urgency/routing and evidence without labels. That trade-off becomes the story |
| asyncpg + Supabase transaction pooler breaks prepared statements | Med | Use the session pooler or direct connection, or set `statement_cache_size=0`. Covered in T11 |
| Langfuse/LangChain SDK APIs changed recently | Med | Check the current official docs while implementing T9 (source-driven) |
| Windows dev vs Linux CI differences (line endings, paths) | Low | `.gitattributes` with `eol=lf`, and `pathlib` everywhere |
| Leaking API keys in a public repo | High | gitleaks pre-commit hook and CI step; `.env` gitignored; secrets only in GH/ACA secrets |
| ACA cold start inflates p95 | Low | Report warm and cold latency separately (spec §14.1) |

## Open Questions

1. **Who writes the code?** For the CV and for interviews, you have to be able to explain every line. Suggested mode: for each task, I explain the concept and the design, you write (or co-write) the code, and I review it. The alternative is that I implement and you review each PR with explanations. This decision changes how we run every task.
