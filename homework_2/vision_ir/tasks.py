"""Score one model's embeddings on the two retrieval tasks."""

import numpy as np

from .data import DOMAINS, CopyQuery, Photo
from .evaluate import cosine_scores, per_query_metrics, top_k


def same_group_task(photos: list[Photo]) -> tuple[np.ndarray, np.ndarray]:
    """Task 1: each image searches all others for the 9 images of its own group."""
    groups = np.array([photo.group for photo in photos])
    ignore = np.eye(len(photos), dtype=bool)  # never return the query itself
    relevant = (groups[:, None] == groups[None, :]) & ~ignore
    return relevant, ignore


def same_group_per_query(gallery: np.ndarray, photos: list[Photo]) -> dict[str, np.ndarray]:
    relevant, ignore = same_group_task(photos)
    return per_query_metrics(cosine_scores(gallery, gallery), relevant, ignore)


def same_group_scores(gallery: np.ndarray, photos: list[Photo]) -> dict[str, float]:
    """Overall mAP and P@1, plus mAP for queries from each domain."""
    per_query = same_group_per_query(gallery, photos)
    domains = np.array([photo.domain for photo in photos])
    scores = {"mAP": float(per_query["mAP"].mean()), "P@1": float(per_query["P@1"].mean())}
    for domain in DOMAINS:
        scores[domain.name] = float(per_query["mAP"][domains == domain.name].mean())
    return scores


def copy_scores(queries: np.ndarray, gallery: np.ndarray, photos: list[Photo],
                copies: list[CopyQuery]) -> dict[str, float]:
    """Task 2: P@1 for finding the unedited source image, per edit and overall.

    Also reports, among the misses, how often the top result was one of the
    source's 9 group-mates: the model recognized the object but not this photo.
    """
    gallery_ids = np.array([photo.image_id for photo in photos])
    groups = np.array([photo.group for photo in photos])
    relevant = np.array([gallery_ids == copy.source_id for copy in copies])
    if not (relevant.sum(axis=1) == 1).all():
        raise ValueError("Every edited copy needs exactly one source image in the gallery.")
    scores = cosine_scores(queries, gallery)
    per_query = per_query_metrics(scores, relevant, np.zeros_like(relevant))
    edits = np.array([copy.edit for copy in copies])
    results = {edit: float(per_query["P@1"][edits == edit].mean()) for edit in dict.fromkeys(edits)}
    results["mean"] = float(per_query["P@1"].mean())
    top = top_k(scores, 1)[:, 0]
    missed = per_query["P@1"] == 0
    source_groups = groups[relevant.argmax(axis=1)]
    group_mate = groups[top] == source_groups
    results["misses that were group-mates"] = (
        float(group_mate[missed].mean()) if missed.any() else float("nan")
    )
    return results
