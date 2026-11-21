"""BM25 weights, text encoding, and the shared retrieval metrics."""

import math

import numpy as np
import torch


def bm25_weights(counts, *, k1: float = 1.2, b: float = 0.75):
    """Return a sparse term-weight matrix using positive-IDF BM25.

    Each nonzero count f becomes IDF * f*(k1+1)/(f+k1*(1-b+b*length/avg_length)).
    k1 controls frequency saturation; b controls document-length normalization.
    IDF = log(1 + (N - df + 0.5)/(df + 0.5)). See:
    https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables
    """
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    average_length = lengths.mean()
    if average_length == 0:
        raise ValueError("Cannot index an empty-token corpus.")
    document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log1p(
        (counts.shape[0] - document_frequency + 0.5) / (document_frequency + 0.5)
    )
    weights = counts.astype(np.float64).tocoo()
    frequency = weights.data.copy()
    normalization = k1 * (1 - b + b * lengths[weights.row] / average_length)
    weights.data = idf[weights.col] * frequency * (k1 + 1) / (frequency + normalization)
    return weights.tocsr()


def ranking_metrics(ranked_ids: list[str], positives: set[str]) -> dict[str, float]:
    """Binary-relevance metrics at ten for one query, including zero-hit queries."""
    if not positives:
        raise ValueError("Each evaluated query needs at least one positive document.")
    hits = [did in positives for did in ranked_ids[:10]]
    dcg = sum(1 / math.log2(rank + 1) for rank, hit in enumerate(hits, 1) if hit)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(positives)) + 1))
    return {
        "Recall@10": sum(hits) / len(positives),
        "MRR@10": next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0),
        "nDCG@10": dcg / ideal,
    }


def encode_texts(model, tokenizer, texts, device):
    """Encode one batch, preserving gradients when called during training."""
    batch = tokenizer(
        texts, padding=True, truncation=True, max_length=model.max_length,
        return_tensors="pt", return_token_type_ids=False,
    )
    return model(batch["input_ids"].to(device), batch["attention_mask"].to(device).bool())


@torch.inference_mode()
def evaluate_holdout(model, tokenizer, documents, records, device, batch_size=8):
    """Rank the full corpus for each validation query, then average its metrics."""
    model.eval()

    def encode_batches(texts):
        vectors = torch.cat([
            encode_texts(model, tokenizer, texts[start : start + batch_size], device).cpu()
            for start in range(0, len(texts), batch_size)
        ])
        if not torch.isfinite(vectors).all():
            raise RuntimeError("Validation embeddings are not finite.")
        return vectors

    document_ids, query_ids = sorted(documents), sorted(records)
    document_vectors = encode_batches([documents[did] for did in document_ids])
    query_vectors = encode_batches([records[qid]["text"] for qid in query_ids])
    totals = {"Recall@10": 0.0, "MRR@10": 0.0, "nDCG@10": 0.0}
    for row, qid in enumerate(query_ids):
        scores = query_vectors[row] @ document_vectors.T
        # Stable ties follow sorted document IDs, matching the lexical baselines.
        indices = torch.argsort(scores, descending=True, stable=True)[:10].tolist()
        metrics = ranking_metrics(
            [document_ids[i] for i in indices], set(records[qid]["positive_document_ids"])
        )
        for name, value in metrics.items():
            totals[name] += value
    return {name: value / len(query_ids) for name, value in totals.items()}
