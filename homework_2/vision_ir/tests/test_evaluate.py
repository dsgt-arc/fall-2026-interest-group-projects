import numpy as np
import pytest

from vision_ir.evaluate import (
    average_precision,
    cosine_scores,
    retrieval_metrics,
    top_k,
)


def test_average_precision_worked_example():
    # Relevant items at ranks 1 and 3: (1/1 + 2/3) / 2.
    assert average_precision(np.array([True, False, True, False])) == pytest.approx(5 / 6)


def test_average_precision_single_relevant_is_reciprocal_rank():
    assert average_precision(np.array([False, False, True])) == pytest.approx(1 / 3)


def test_average_precision_requires_a_relevant_item():
    with pytest.raises(ValueError):
        average_precision(np.array([False, False]))


def test_average_precision_matches_sklearn():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    scores = rng.normal(size=50)
    labels = rng.random(50) < 0.3
    order = np.argsort(-scores, kind="stable")
    assert average_precision(labels[order]) == pytest.approx(
        average_precision_score(labels, scores)
    )


def test_retrieval_metrics_ignores_the_query_itself():
    # Query 0 is gallery item 0; without ignoring it, it would trivially rank first.
    scores = np.array([[1.0, 0.2, 0.9]])
    relevant = np.array([[True, False, True]])
    ignore = np.array([[True, False, False]])
    metrics = retrieval_metrics(scores, relevant, ignore)
    assert metrics == {"mAP": pytest.approx(1.0), "P@1": pytest.approx(1.0)}


def test_retrieval_metrics_averages_over_queries():
    scores = np.array([[0.9, 0.1], [0.9, 0.1]])
    relevant = np.array([[True, False], [False, True]])
    metrics = retrieval_metrics(scores, relevant)
    assert metrics["P@1"] == pytest.approx(0.5)
    assert metrics["mAP"] == pytest.approx((1.0 + 0.5) / 2)


def test_retrieval_metrics_ties_follow_gallery_order():
    scores = np.zeros((1, 3))
    relevant = np.array([[False, True, False]])
    assert retrieval_metrics(scores, relevant)["P@1"] == 0.0


def test_retrieval_metrics_rejects_shape_mismatch_and_nan():
    with pytest.raises(ValueError):
        retrieval_metrics(np.zeros((2, 3)), np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError):
        retrieval_metrics(np.array([[np.nan, 1.0]]), np.array([[True, False]]))


def test_cosine_scores_are_scale_invariant():
    queries = np.array([[1.0, 0.0], [3.0, 3.0]])
    gallery = np.array([[10.0, 0.0], [0.0, 0.5]])
    scores = cosine_scores(queries, gallery)
    np.testing.assert_allclose(scores, [[1.0, 0.0], [2**-0.5, 2**-0.5]], atol=1e-6)


def test_top_k_skips_ignored_items():
    scores = np.array([[0.9, 0.8, 0.7]])
    ignore = np.array([[True, False, False]])
    assert top_k(scores, 2, ignore).tolist() == [[1, 2]]
