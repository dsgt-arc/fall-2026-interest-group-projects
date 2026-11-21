"""LangChain hybrid retrieval over Chroma and a BM25 retriever."""

import re
from dataclasses import asdict, dataclass

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langsmith import tracing_context

from clef_rag.catalog import list_documents
from clef_rag.config import RagError


@dataclass(frozen=True)
class RetrievalSettings:
    mode: str = "hybrid"
    dense_weight: float = 0.5
    rrf_constant: int = 60
    candidate_depth: int = 20
    top_k: int = 6

    def __post_init__(self):
        if self.mode not in ("hybrid", "dense", "bm25"):
            raise ValueError("Retrieval mode must be hybrid, dense, or bm25.")
        if not 0 <= self.dense_weight <= 1:
            raise ValueError("Dense weight must be between 0 and 1.")
        if self.rrf_constant < 0 or self.candidate_depth < 1 or self.top_k < 1:
            raise ValueError("RRF constant must be nonnegative and depths must be positive.")

    def to_dict(self):
        return asdict(self)


def lexical_tokens(text):
    return re.findall(r"\w+", text.casefold())


def source_record(document):
    return {
        **document.metadata,
        "id": document.metadata["chunk_id"],
        "text": document.page_content.split("\n", 1)[1],
    }


def overlaps(left, right):
    if left["id"] == right["id"]:
        return True
    if left["document_id"] != right["document_id"] or left["page"] != right["page"]:
        return False
    intersection = max(
        0,
        min(left["end_char"], right["end_char"])
        - max(left["start_char"], right["start_char"]),
    )
    smaller = min(
        left["end_char"] - left["start_char"], right["end_char"] - right["start_char"]
    )
    return intersection > 0.5 * smaller


class Retriever:
    def __init__(self, index, models, settings=None):
        self.index, self.models = index, models
        self.settings = settings or RetrievalSettings()

    def search(self, question, *, track=None, author=None, paper_ids=None, top_k=None):
        top_k = self.settings.top_k if top_k is None else top_k
        store = self.index.vectorstore(self.models) if self.settings.mode != "bm25" else None
        catalog = list_documents(
            self.index.db, track=track, author=author, paper_ids=paper_ids
        )
        if not catalog:
            raise RagError("No documents match the selected filters.")
        if paper_ids:
            unavailable = [
                d["id"] for d in catalog if d["status"] not in ("indexed", "partial")
            ]
            if unavailable:
                raise RagError(
                    "Full text is unavailable for "
                    + ", ".join(unavailable)
                    + ". Catalog metadata remains available."
                )
            if len(paper_ids) > top_k:
                raise RagError(f"Compare at most {top_k} papers at once.")
        chunks = self.index.documents([d["id"] for d in catalog])
        if not chunks:
            return []
        if paper_ids and len(paper_ids) > 1:
            groups = [
                self._rank(
                    question,
                    store,
                    [d for d in chunks if d.metadata["document_id"] == paper_id],
                )
                for paper_id in paper_ids
            ]
            ranked = [
                group[i] for i in range(self.settings.candidate_depth)
                for group in groups if len(group) > i
            ]
        else:
            ranked = self._rank(question, store, chunks)
        leading_ids = paper_ids or (
            [ranked[0].metadata["document_id"]] if ranked else []
        )
        introductions = []
        for document_id in leading_ids:
            candidates = [d for d in chunks if d.metadata["document_id"] == document_id]
            if candidates:
                introductions.append(
                    min(
                        candidates,
                        key=lambda d: (d.metadata["page"], d.metadata["start_char"]),
                    )
                )
        selected = []
        for document in introductions + ranked:
            item = source_record(document)
            if not any(overlaps(item, previous) for previous in selected):
                item["source_id"] = f"S{len(selected) + 1}"
                item["rank"] = len(selected) + 1
                selected.append(item)
            if len(selected) == top_k:
                break
        return selected

    def _rank(self, question, store, documents):
        if not documents:
            return []
        ids = sorted({d.metadata["document_id"] for d in documents})
        depth = min(self.settings.candidate_depth, len(documents))
        semantic = None
        lexical = None
        if self.settings.mode != "bm25":
            semantic = store.as_retriever(
                search_kwargs={"k": depth, "filter": {"document_id": {"$in": ids}}}
            )
        if self.settings.mode != "dense":
            lexical = BM25Retriever.from_documents(
                documents, k=depth, preprocess_func=lexical_tokens
            )
        if self.settings.mode == "hybrid":
            retriever = EnsembleRetriever(
                retrievers=[semantic, lexical],
                weights=[self.settings.dense_weight, 1 - self.settings.dense_weight],
                c=self.settings.rrf_constant, id_key="chunk_id",
            )
        else:
            retriever = semantic if self.settings.mode == "dense" else lexical
        try:
            with tracing_context(enabled=False):
                return retriever.invoke(question)
        except Exception as exc:
            raise RagError(f"LangChain retrieval failed: {exc}") from exc
