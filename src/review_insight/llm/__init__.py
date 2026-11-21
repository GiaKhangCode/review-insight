"""LLM adapter layer: prompts, the Gemini-backed analyzer, and cost accounting.

The only package allowed to import LangChain or the Gemini SDK. The rest of the app talks to
it through the domain contract (`AnalysisResult`), so the model provider can be swapped or
faked in tests without touching the API or the domain.
"""
