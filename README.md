# ReviewInsight

> 🚧 **Work in progress.** Building in public, task by task. See [tasks/todo.md](tasks/todo.md) for status.

LLM-powered service that turns raw **Vietnamese e-commerce reviews** into structured, validated insight:
aspect-based sentiment with evidence spans, urgency, and routing to the right team. Every design
choice is backed by an offline evaluation against a supervised baseline.

## Why

Online retailers receive thousands of reviews a week. Nobody reads them all, so recurring defects
surface late and urgent complaints wait in the same queue as five-star praise.
Full problem statement and requirements: [docs/SPEC.md](docs/SPEC.md).

## Planned stack

FastAPI · LangChain · Gemini (Flash-Lite, `gemini-embedding-2`) · Langfuse · Supabase Postgres ·
Docker · GitHub Actions · Azure Container Apps

## Quickstart (development)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync                      # create .venv and install dev tools
uv run pre-commit install    # enable git hooks (lint, format, secret scan)
uv run pytest
```

## Data

Evaluation uses the **UIT-ViSFD** dataset (Phan et al., 2021). It is downloaded by a script and is
not redistributed in this repository. Citation and terms will be added in `docs/DATA_CARD.md`.

## License

Code: [MIT](LICENSE). The dataset is subject to its authors' terms.
