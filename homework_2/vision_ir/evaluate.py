"""Cosine-similarity search and the two retrieval metrics used in this lesson."""

import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so a dot product equals cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("Expected a 2-D array with one embedding per row.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain NaN or infinite values.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def standardize(vectors: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Z-score every dimension with the mean and spread of `reference` (no labels used)."""
    reference = np.asarray(reference, dtype=np.float64)
    mean, spread = reference.mean(axis=0), reference.std(axis=0)
    return ((np.asarray(vectors, dtype=np.float64) - mean) / np.maximum(spread, 1e-6)).astype(np.float32)


def mean_pair_cosine(vectors: np.ndarray) -> float:
    """Average cosine similarity over all pairs of different images."""
    unit = normalize(vectors).astype(np.float64)
    total = unit.sum(axis=0)
    count = len(unit)
    return float((total @ total - count) / (count * (count - 1)))


def cosine_scores(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Rows are queries and columns are gallery images; higher means more similar."""
    return normalize(queries) @ normalize(gallery).T


def average_precision(ranked_hits: np.ndarray) -> float:
    """Mean of precision values at the rank of every relevant item.

    `ranked_hits` is a boolean vector for one query's full ranking. With relevant
    items at ranks 1 and 3, AP = (1/1 + 2/3) / 2 = 0.833.
    """
    ranked_hits = np.asarray(ranked_hits, dtype=bool)
    total = int(ranked_hits.sum())
    if total == 0:
        raise ValueError("Each evaluated query needs at least one relevant item.")
    ranks = np.flatnonzero(ranked_hits) + 1
    return float((np.arange(1, total + 1) / ranks).mean())


def retrieval_metrics(scores: np.ndarray, relevant: np.ndarray, ignore=None) -> dict[str, float]:
    """Average mAP and P@1 over queries, ranking the whole gallery for each one.

    scores:   (queries, gallery) similarities.
    relevant: (queries, gallery) booleans; True marks a correct match.
    ignore:   optional (queries, gallery) booleans removed before ranking, e.g. the
              query image itself when queries are drawn from the gallery.
    Ties keep gallery order (stable sort), so results are deterministic.
    """
    scores = np.asarray(scores, dtype=np.float64)
    relevant = np.asarray(relevant, dtype=bool)
    if scores.shape != relevant.shape:
        raise ValueError("scores and relevant must have the same (queries, gallery) shape.")
    ignore = np.zeros_like(relevant) if ignore is None else np.asarray(ignore, dtype=bool)
    if ignore.shape != relevant.shape:
        raise ValueError("ignore must match the (queries, gallery) shape.")
    per_query = per_query_metrics(scores, relevant, ignore)
    return {name: float(np.mean(values)) for name, values in per_query.items()}


def per_query_metrics(scores, relevant, ignore) -> dict[str, np.ndarray]:
    """Return one AP and one P@1 value per query (useful for finding failures)."""
    if not np.isfinite(scores).all():
        raise ValueError("Similarity scores contain NaN or infinite values.")
    aps, top1 = [], []
    for row_scores, row_relevant, row_ignore in zip(scores, relevant, ignore, strict=True):
        keep = ~row_ignore
        order = np.argsort(-row_scores[keep], kind="stable")
        ranked_hits = row_relevant[keep][order]
        aps.append(average_precision(ranked_hits))
        top1.append(float(ranked_hits[0]))
    return {"mAP": np.array(aps), "P@1": np.array(top1)}


def top_k(scores: np.ndarray, k: int, ignore=None) -> np.ndarray:
    """Indices of the k best gallery items per query, excluding ignored pairs."""
    scores = np.asarray(scores, dtype=np.float64).copy()
    if ignore is not None:
        scores[np.asarray(ignore, dtype=bool)] = -np.inf
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]
