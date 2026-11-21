"""Domain layer: the problem's vocabulary and output contract.

Pure Python + Pydantic only. Must not import FastAPI, LangChain or SQLAlchemy, so the
API, LLM, DB and eval layers can all depend on it without depending on each other.
"""
