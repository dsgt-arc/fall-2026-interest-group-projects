"""LangChain hybrid retrieval over Chroma and a BM25 retriever."""

import re

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langsmith import tracing_context

from clef_rag.catalog import list_documents
from clef_rag.config import RagError


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
    def __init__(self, index, models):
        self.index, self.models = index, models

    def search(self, question, *, track=None, author=None, paper_ids=None, top_k=6):
        store = self.index.vectorstore(self.models)
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
            ranked = [group[i] for i in range(20) for group in groups if len(group) > i]
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
        semantic = store.as_retriever(
            search_kwargs={
                "k": min(20, len(documents)),
                "filter": {"document_id": {"$in": ids}},
            }
        )
        lexical = BM25Retriever.from_documents(
            documents, k=min(20, len(documents)), preprocess_func=lexical_tokens
        )
        ensemble = EnsembleRetriever(
            retrievers=[semantic, lexical], weights=[0.5, 0.5], c=60, id_key="chunk_id"
        )
        try:
            with tracing_context(enabled=False):
                return ensemble.invoke(question)
        except Exception as exc:
            raise RagError(f"LangChain retrieval failed: {exc}") from exc
