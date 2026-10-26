"""Explicit local model policy and reproducible indexing defaults."""

CHAT_MODELS = ("gemma3:12b", "gemma3:4b")
EMBED_MODEL = "embeddinggemma:300m"
OLLAMA_HOST = "http://localhost:11434"
CATALOG_URL = "https://clef-staging.pages.dev/"
CONTEXT_TOKENS = 8192


class RagError(Exception):
    """An actionable error suitable for display without a traceback."""
