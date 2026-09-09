"""Local, approved Ollama models through LangChain integrations."""

from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langsmith import tracing_context
from ollama import Client

from clef_rag.config import (
    CHAT_MODELS,
    CONTEXT_TOKENS,
    EMBED_MODEL,
    OLLAMA_HOST,
    RagError,
)

# A conservative byte budget keeps input within EmbeddingGemma's 2K context.
EMBED_INPUT_BYTES = 1900


class RetrievalEmbeddings(Embeddings):
    """Apply Google's retrieval prompts; LangChain performs model inference."""

    def __init__(self, underlying):
        self.underlying = underlying

    def embed_documents(self, texts):
        if any(len(text.encode("utf-8")) > EMBED_INPUT_BYTES for text in texts):
            raise RagError(
                "Embedding input is too long. Re-ingest with the configured text splitter."
            )
        return self.underlying.embed_documents(texts)

    def embed_query(self, text):
        query = "task: search result | query: " + text
        if len(query.encode("utf-8")) > EMBED_INPUT_BYTES:
            raise RagError(
                "Please shorten the question to fit the embedding model's context."
            )
        return self.underlying.embed_query(query)


class LocalModels:
    def __init__(self, chat_model=CHAT_MODELS[0]):
        if chat_model not in CHAT_MODELS:
            raise RagError(
                f"Model not allowed: {chat_model}. Choose from {CHAT_MODELS}."
            )
        self.chat_model = chat_model
        # The SDK is used only to verify installed model names and digests.
        self.client = Client(host=OLLAMA_HOST, timeout=600, trust_env=False)
        self._digests = {}

    def digest(self, model):
        if model not in (*CHAT_MODELS, EMBED_MODEL):
            raise RagError(f"Model not allowed: {model}")
        if model not in self._digests:
            try:
                available = {
                    item.model: item.digest for item in self.client.list().models
                }
            except Exception as exc:
                raise RagError(
                    "Cannot reach local Ollama. Start it with: OLLAMA_NO_CLOUD=1 ollama serve"
                ) from exc
            if model not in available:
                raise RagError(
                    f"Required local model is missing. Run: ollama pull {model}"
                )
            self._digests[model] = available[model]
        return self._digests[model]

    def embeddings(self):
        self.digest(EMBED_MODEL)
        return RetrievalEmbeddings(
            OllamaEmbeddings(
                model=EMBED_MODEL,
                base_url=OLLAMA_HOST,
                keep_alive=1800,
                client_kwargs={"timeout": 600, "trust_env": False},
            )
        )

    def chat(self, system, question, schema=None, max_tokens=900):
        self.digest(self.chat_model)
        prompt = ChatPromptTemplate.from_messages(
            [("system", "{system}"), ("human", "{question}")]
        )
        llm = ChatOllama(
            model=self.chat_model,
            base_url=OLLAMA_HOST,
            temperature=0.1,
            num_ctx=CONTEXT_TOKENS,
            num_predict=max_tokens,
            seed=42,
            keep_alive=1800,
            disable_streaming=True,
            client_kwargs={"timeout": 600, "trust_env": False},
        )
        runnable = (
            llm.with_structured_output(schema, method="json_schema", include_raw=True)
            if schema
            else llm
        )
        try:
            # Disable external tracing even when the surrounding shell enables it.
            with tracing_context(enabled=False):
                result = (prompt | runnable).invoke(
                    {"system": system, "question": question}
                )
        except Exception as exc:
            raise RagError(f"Local LangChain answer request failed: {exc}") from exc
        message = result["raw"] if schema else result
        if message.response_metadata.get("done_reason") == "length":
            raise RagError("The answer hit its output limit. Try a narrower question.")
        if schema:
            if result["parsing_error"] or result["parsed"] is None:
                raise RagError(
                    "The local model returned invalid structured output. Try rephrasing the question."
                )
            return result["parsed"]
        return message.content
