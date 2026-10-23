"""In-memory replica of the starter retriever, with switches for each experiment.

The starter (clef_rag.retrieve.Retriever) does: dense top-20 from Chroma (cosine) + BM25 top-20,
weighted reciprocal-rank fusion (weights 0.5/0.5, c=60), then prepends the opening chunk of the
top-ranked paper, drops overlapping chunks, and keeps 6. This module reproduces that from the
stored vectors so dozens of variants run in seconds; `parity_check` in experiments.py confirms it
matches the real starter retriever on the same questions.
"""

import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np
from rank_bm25 import BM25Okapi

from clef_rag.retrieve import lexical_tokens, overlaps, source_record
from common import RESULTS


@dataclass(frozen=True)
class Config:
    name: str = "baseline"
    dense_k: int = 20          # dense candidates
    bm25_k: int = 20           # BM25 candidates
    w_dense: float = 0.5       # RRF weights
    w_bm25: float = 0.5
    c: int = 60                # RRF constant
    intro: str = "always"      # "always" (starter: opening chunk first) | "named" (only when papers are named)
                               # | "last" (named papers first; otherwise the top paper's opening chunk in the last slot)
    dense_search: str = "hnsw" # "hnsw" = the starter's Chroma ANN query | "exact" = brute-force cosine
    dims: int = 768            # Matryoshka truncation of EmbeddingGemma vectors (exact search only)
    rerank: int = 0            # >0: gemma3:12b listwise rerank of that many fused candidates
    top_k: int = 6

    def but(self, **kw):
        return replace(self, **kw)


class Engine:
    def __init__(self, index, models):
        self.index, self.models = index, models
        data = index.collection.get(include=["documents", "metadatas", "embeddings"])
        self.ids = data["ids"]
        self.texts = data["documents"]
        self.metas = data["metadatas"]
        emb = np.asarray(data["embeddings"], dtype=np.float32)
        self.emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.doc_of = np.array([m["document_id"] for m in self.metas])
        self.track_of = np.array([m.get("track") or "" for m in self.metas])
        self._pos = {cid: i for i, cid in enumerate(self.ids)}
        self.tokens = [lexical_tokens(t) for t in self.texts]
        self.bm25_all = BM25Okapi(self.tokens)  # same defaults as LangChain's BM25Retriever
        self._qcache_path = RESULTS / "query_embeddings.json"
        self._qcache = json.loads(self._qcache_path.read_text()) if self._qcache_path.exists() else {}
        self._embedder = models.embeddings()
        self._rcache_path = RESULTS / "rerank_cache.json"
        self._rcache = json.loads(self._rcache_path.read_text()) if self._rcache_path.exists() else {}

    # ---------- primitives ----------
    def query_vec(self, q):
        if q not in self._qcache:
            self._qcache[q] = self._embedder.embed_query(q)  # adds "task: search result | query: "
            self._qcache_path.parent.mkdir(parents=True, exist_ok=True)
            self._qcache_path.write_text(json.dumps(self._qcache))
        v = np.asarray(self._qcache[q], dtype=np.float32)
        return v / np.linalg.norm(v)

    def dense_hnsw(self, q, mask, k):
        """Exactly what the starter's Chroma retriever runs: HNSW search with a document_id filter."""
        papers = sorted(set(self.doc_of[mask]))
        res = self.index.collection.query(
            query_embeddings=[self._qcache[q] if q in self._qcache else self.query_vec(q).tolist()],
            n_results=k, where={"document_id": {"$in": papers}}, include=[],
        )
        pos = self._pos
        return [pos[c] for c in res["ids"][0]]

    def dense(self, q, mask, k, dims):
        v = self.query_vec(q)[:dims]
        v = v / np.linalg.norm(v)
        X = self.emb[:, :dims]
        if dims < self.emb.shape[1]:
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
        idx = np.flatnonzero(mask)
        sims = X[idx] @ v
        order = np.argsort(-sims, kind="stable")[:k]
        return [int(idx[i]) for i in order]

    def bm25(self, q, mask, k):
        idx = np.flatnonzero(mask)
        if len(idx) == len(self.ids):
            scores = self.bm25_all.get_scores(lexical_tokens(q))
        else:  # the starter rebuilds BM25 over the filtered documents, so IDF changes too
            scores = BM25Okapi([self.tokens[i] for i in idx]).get_scores(lexical_tokens(q))
        order = np.argsort(scores)[::-1][:k]  # rank_bm25.get_top_n ordering
        return [int(idx[i]) for i in order]

    @staticmethod
    def rrf(lists, weights, c):
        scores, first = {}, []
        for lst, w in zip(lists, weights):
            for rank, i in enumerate(lst, 1):
                if i not in scores:
                    first.append(i)
                    scores[i] = 0.0
                scores[i] += w / (rank + c)
        return sorted(first, key=lambda i: -scores[i])  # stable: ties keep first appearance

    def rank(self, q, mask, cfg):
        lists, weights = [], []
        if cfg.w_dense > 0:
            k = min(cfg.dense_k, int(mask.sum()))
            lists.append(self.dense_hnsw(q, mask, k) if cfg.dense_search == "hnsw" else self.dense(q, mask, k, cfg.dims))
            weights.append(cfg.w_dense)
        if cfg.w_bm25 > 0:
            lists.append(self.bm25(q, mask, min(cfg.bm25_k, int(mask.sum()))))
            weights.append(cfg.w_bm25)
        fused = self.rrf(lists, weights, cfg.c)
        if cfg.rerank:
            fused = self.llm_rerank(q, fused[: cfg.rerank]) + fused[cfg.rerank :]
        return fused

    # ---------- gemma3:12b listwise reranker ----------
    def llm_rerank(self, q, cand):
        from pydantic import BaseModel

        class Ranking(BaseModel):
            ranking: list[int]

        key = hashlib.sha256(json.dumps([q, [self.ids[i] for i in cand]]).encode()).hexdigest()
        if key not in self._rcache:
            blocks = []
            for n, i in enumerate(cand, 1):
                m = self.metas[i]
                body = self.texts[i].split("\n", 1)[-1][:650]
                blocks.append(f"[{n}] ({m['title'][:90]}, p.{m['page']})\n{body}")
            prompt = (
                f"Question: {q}\n\nCandidate passages:\n\n" + "\n\n".join(blocks)
                + "\n\nRank the passages by how directly they answer the question, most useful first. "
                "Return the passage numbers only."
            )
            try:
                out = self.models.chat(
                    "You are a precise search-result reranker for scientific papers.",
                    prompt, schema=Ranking, max_tokens=200,
                ).ranking
            except Exception:  # noqa: BLE001 - fall back to the fused order
                out = []
            self._rcache[key] = out
            self._rcache_path.write_text(json.dumps(self._rcache))
        seen, order = set(), []
        for n in self._rcache[key]:
            if isinstance(n, int) and 1 <= n <= len(cand) and n not in seen:
                seen.add(n)
                order.append(cand[n - 1])
        order += [i for i in cand if i not in set(order)]  # anything the model dropped keeps its place
        return order

    # ---------- the starter's search() with switches ----------
    def search(self, q, cfg, *, paper_ids=None, track=None, restrict=None):
        mask = np.ones(len(self.ids), dtype=bool)
        if restrict is not None:  # e.g. the papers matching an --author filter
            mask &= np.isin(self.doc_of, list(restrict))
        if paper_ids:
            mask &= np.isin(self.doc_of, paper_ids)
        if track:  # the starter matches tracks by case-insensitive substring (catalog.list_documents)
            mask &= np.char.find(np.char.lower(self.track_of), track.lower()) >= 0
        if not mask.any():
            return []
        if paper_ids and len(paper_ids) > 1:
            groups = [self.rank(q, mask & (self.doc_of == p), cfg) for p in paper_ids]
            ranked = [g[i] for i in range(20) for g in groups if len(g) > i]
        else:
            ranked = self.rank(q, mask, cfg)
        use_intro = cfg.intro == "always" or (cfg.intro in ("named", "last") and paper_ids)
        leading = (paper_ids or ([self.doc_of[ranked[0]]] if ranked else [])) if use_intro else []
        intros = []
        for d in leading:
            cand = [i for i in np.flatnonzero(self.doc_of == d)]
            if cand:
                intros.append(min(cand, key=lambda i: (self.metas[i]["page"], self.metas[i]["start_char"])))
        tail = []
        if cfg.intro == "last" and not paper_ids and ranked:
            d = self.doc_of[ranked[0]]
            tail = [min(np.flatnonzero(self.doc_of == d), key=lambda i: (self.metas[i]["page"], self.metas[i]["start_char"]))]
        selected = []
        for i in intros + ranked:
            if tail and len(selected) == cfg.top_k - 1:
                break  # reserve the last slot for the top paper's opening chunk
            from langchain_core.documents import Document

            item = source_record(Document(id=self.ids[i], page_content=self.texts[i], metadata=self.metas[i]))
            if not any(overlaps(item, prev) for prev in selected):
                item["rank"] = len(selected) + 1
                selected.append(item)
            if len(selected) == cfg.top_k:
                break
        for i in tail + ranked:  # fill the reserved slot (or any gap) without duplicates
            if len(selected) == cfg.top_k:
                break
            from langchain_core.documents import Document

            item = source_record(Document(id=self.ids[i], page_content=self.texts[i], metadata=self.metas[i]))
            if not any(overlaps(item, prev) for prev in selected):
                item["rank"] = len(selected) + 1
                selected.append(item)
        return selected
