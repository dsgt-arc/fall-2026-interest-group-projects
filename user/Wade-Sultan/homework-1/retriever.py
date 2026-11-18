"""The starter hybrid retriever with its hardcoded knobs exposed as settings.

With default Settings this reproduces clef_rag.retrieve.Retriever exactly; only
the constants are parameterized, so the starter code under homework-1/ stays intact.
The embedding-mismatch strategies in strategies.py are further settings, all off
by default.
"""

from dataclasses import dataclass

from langchain_community.retrievers import BM25Retriever
from langsmith import tracing_context

from clef_rag.catalog import list_documents
from clef_rag.config import RagError
from clef_rag.retrieve import Retriever, lexical_tokens, overlaps, source_record

from strategies import (
    Doc2Query,
    expand_query,
    fuse,
    hyde_vector,
    hypothetical_passage,
    load_glossary,
    rerank,
    rewrites,
)

QUERY_MODES = ("plain", "multi_query", "hyde")


@dataclass(frozen=True)
class Settings:
    # Candidates each retriever (dense and BM25) returns before fusion.
    candidates: int = 20
    # Reciprocal-rank-fusion weights and smoothing constant.
    semantic_weight: float = 0.5
    lexical_weight: float = 0.5
    rrf_c: int = 60
    # Passages kept after fusion and overlap removal; also the metric ranking depth.
    top_k: int = 6
    # Force the first passage of the leading (or each named) paper to the front.
    prepend_intro: bool = True
    # Embedding-mismatch strategies (strategies.py). The defaults turn them all off.
    # Append titles for "paperN" mentions and expand acronyms before searching.
    expand_query: bool = False
    # "plain"; "multi_query" searches the question plus model rewordings; "hyde" runs
    # the dense search on a model-written hypothetical answer instead of the question.
    query_mode: str = "plain"
    # Rewordings generated in multi_query mode.
    rewrites: int = 3
    # RRF weight for matches against doc2query's generated questions; 0 is off.
    doc2query_weight: float = 0.0
    # gemma3 reorders this many top fused candidates; 0 is off.
    rerank_depth: int = 0

    def __post_init__(self):
        if self.query_mode not in QUERY_MODES:
            raise ValueError(f"query_mode must be one of {QUERY_MODES}, not {self.query_mode!r}")


class ConfigurableRetriever(Retriever):
    def __init__(self, index, models, settings=Settings()):
        super().__init__(index, models)
        self.settings = settings
        self.glossary = load_glossary(index) if settings.expand_query else None
        self.doc2query = Doc2Query(index, models) if settings.doc2query_weight else None
        self.embeddings = models.embeddings() if settings.query_mode == "hyde" else None

    def search(self, question, *, track=None, author=None, paper_ids=None, top_k=None):
        s = self.settings
        top_k = top_k or s.top_k
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
            ranked = [
                group[i]
                for i in range(s.candidates)
                for group in groups
                if len(group) > i
            ]
        else:
            ranked = self._rank(question, store, chunks)
        introductions = []
        if s.prepend_intro:
            leading_ids = paper_ids or (
                [ranked[0].metadata["document_id"]] if ranked else []
            )
            for document_id in leading_ids:
                candidates = [
                    d for d in chunks if d.metadata["document_id"] == document_id
                ]
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
        s = self.settings
        k = min(s.candidates, len(documents))
        ids = sorted({d.metadata["document_id"] for d in documents})
        where = {"document_id": {"$in": ids}}
        query = (
            expand_query(question, self.glossary, self.index.db) if s.expand_query else question
        )
        semantic = store.as_retriever(search_kwargs={"k": k, "filter": where})
        lexical = BM25Retriever.from_documents(
            documents, k=k, preprocess_func=lexical_tokens
        )
        # Ranked lists in the order EnsembleRetriever used: dense, then BM25, per query.
        lists, weights = [], []
        try:
            with tracing_context(enabled=False):
                if s.query_mode == "hyde":
                    passage = hypothetical_passage(self.models, query)
                    vector = hyde_vector(self.models, self.embeddings, passage)
                    lists += [
                        store.similarity_search_by_vector(vector, k=k, filter=where),
                        lexical.invoke(query),
                    ]
                    weights += [s.semantic_weight, s.lexical_weight]
                else:
                    queries = (
                        rewrites(self.models, query, s.rewrites)
                        if s.query_mode == "multi_query"
                        else [query]
                    )
                    for text in queries:
                        lists += [semantic.invoke(text), lexical.invoke(text)]
                        weights += [s.semantic_weight, s.lexical_weight]
                if self.doc2query:
                    lists.append(self.doc2query.search(query, documents, k))
                    weights.append(s.doc2query_weight)
        except RagError:
            raise
        except Exception as exc:
            raise RagError(f"LangChain retrieval failed: {exc}") from exc
        ranked = fuse(lists, weights, s.rrf_c)
        if s.rerank_depth:
            ranked = rerank(self.models, question, ranked, s.rerank_depth)
        return ranked
