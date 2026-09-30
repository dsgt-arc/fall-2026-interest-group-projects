import json
import math
from types import SimpleNamespace

import pytest

from clef_rag.cli import parser
from clef_rag.evaluate import evaluate
from clef_rag.metrics import aggregate_retrieval_metrics, retrieval_metrics


def source(paper, page):
    return {"document_id": paper, "page": page}


def test_ranked_metrics_match_hand_calculated_values():
    # Deduplicated ranking: irrelevant, relevant, irrelevant, relevant.
    # A third relevant page is not retrieved. Repeated golds must not affect recall.
    gold = [
        {"paper": "a", "pages": [1, 2, 1]},
        {"paper": "b", "pages": [1]},
        {"paper": "a", "pages": [1]},
    ]
    sources = [
        source("wrong", 1),
        source("a", 1),
        source("a", 1),
        source("a", 9),
        source("b", 1),
    ]
    result = retrieval_metrics(sources, gold, (4, 1, 3, 3))
    assert list(result) == ["1", "3", "4"]
    assert all(value == 0 for value in result["1"].values())
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert result["3"] == pytest.approx(
        {
            "precision": 1 / 3,
            "recall": 1 / 3,
            "hit": 1,
            "reciprocal_rank": 1 / 2,
            "average_precision": (1 / 2) / 3,
            "ndcg": (1 / math.log2(3)) / ideal,
        }
    )
    assert result["4"] == pytest.approx(
        {
            "precision": 2 / 4,
            "recall": 2 / 3,
            "hit": 1,
            "reciprocal_rank": 1 / 2,
            "average_precision": (1 / 2 + 2 / 4) / 3,
            "ndcg": (1 / math.log2(3) + 1 / math.log2(5)) / ideal,
        }
    )


def test_perfect_short_ranking_and_missing_results():
    gold = [{"paper": "a", "pages": [1]}, {"paper": "b", "pages": [1]}]
    perfect = retrieval_metrics([source("a", 1), source("b", 1)], gold, (1, 2, 6))
    assert all(value == 1 for value in perfect["2"].values())
    assert perfect["1"]["recall"] == perfect["1"]["average_precision"] == 0.5
    assert perfect["1"]["ndcg"] == 1
    assert perfect["6"]["precision"] == pytest.approx(2 / 6)
    assert perfect["6"]["recall"] == perfect["6"]["average_precision"] == 1
    assert all(
        value == 0
        for scores in retrieval_metrics([], gold).values()
        for value in scores.values()
    )


@pytest.mark.parametrize("cutoffs", [(), (0,), (-1,), (True,), (1.5,), ("3",)])
def test_invalid_cutoffs_are_rejected(cutoffs):
    with pytest.raises(ValueError, match="positive integers"):
        retrieval_metrics([], [{"paper": "a", "pages": [1]}], cutoffs)


@pytest.mark.parametrize(
    "gold",
    [
        [],
        [{"paper": "a", "pages": []}],
        [{"paper": "a", "pages": [0]}],
        [{"paper": "", "pages": [1]}],
    ],
)
def test_invalid_gold_is_not_silently_scored_as_zero(gold):
    with pytest.raises(ValueError):
        retrieval_metrics([], gold)


def test_macro_averages_include_empty_runs_but_exclude_unjudged_queries():
    gold = [{"paper": "a", "pages": [1]}]
    rows = [
        {"retrieval_metrics": retrieval_metrics([source("a", 1)], gold, (1,))},
        {"retrieval_metrics": retrieval_metrics([], gold, (1,))},
        {"category": "catalog"},
        {"error": "Retrieval failed."},
    ]
    aggregate = aggregate_retrieval_metrics(rows)
    assert aggregate["1"] == {
        name: 0.5 for name in ("precision", "recall", "hit_rate", "mrr", "map", "ndcg")
    }
    assert aggregate_retrieval_metrics(rows[2:]) is None


@pytest.mark.parametrize("retrieval_only", [False, True])
def test_evaluator_persists_metrics_and_exposes_unscored_cases(
    tmp_path, monkeypatch, retrieval_only
):
    gold = [{"paper": "a", "pages": [1, 2]}]
    cases = [
        {"id": name, "category": "paper", "question": name, "gold_sources": gold}
        for name in ("hit", "empty", "failed")
    ] + [
        {
            "id": "catalog",
            "category": "catalog",
            "question": "catalog",
            "expected_counts": {"listed_papers": 2},
        },
        {"id": "unjudged", "category": "paper", "question": "unjudged"},
    ]
    questions, output = tmp_path / "questions.json", tmp_path / "report.json"
    questions.write_text(json.dumps(cases))

    def search(_self, question, **_kwargs):
        if question == "failed":
            raise RuntimeError("Simulated retrieval failure")
        return (
            [source("a", 3), source("a", 2), source("a", 2)]
            if question == "hit"
            else []
        )

    def answer(index, models, question, **_kwargs):
        return {
            "retrieved_sources": search(None, question),
            "sources": [source("a", 2)],  # Metrics must not use the answer's citations.
            "data": {"listed_papers": 2},
            "supported": True,
        }

    monkeypatch.setattr("clef_rag.evaluate.Retriever.search", search)
    monkeypatch.setattr("clef_rag.evaluate.ask", answer)
    report = evaluate(
        SimpleNamespace(manifest={}, db=None),
        SimpleNamespace(chat_model="gemma3:4b"),
        questions,
        output,
        retrieval_only,
        cutoffs=(1, 3),
    )
    assert json.loads(output.read_text()) == report
    assert report["retrieval_evaluation"]["cutoffs"] == [1, 3]
    hit = report["results"][0]
    assert hit["retrieved_pages"] == [["a", 3], ["a", 2]]
    assert hit["gold_pages"] == [["a", 1], ["a", 2]]
    # The original pass check accepts any annotated page; recall counts every gold page.
    assert hit["retrieval_pass"] is True
    assert hit["retrieval_metrics"]["3"]["recall"] == 0.5
    assert hit["retrieval_metrics"]["1"]["hit"] == 0
    summary = report["summary"]
    assert summary["cases_run"] == (4 if retrieval_only else 5)
    assert summary["retrieval_eligible"] == 3
    assert summary["retrieval_checked"] == 2
    assert summary["retrieval_unscored"] == summary["errors"] == 1
    assert summary["retrieval_metrics"]["3"]["mrr"] == 0.25
    assert summary["catalog_passed"] == (0 if retrieval_only else 1)


def test_evaluation_without_judgments_has_no_metric_average(tmp_path):
    questions, output = tmp_path / "questions.json", tmp_path / "nested/report.json"
    questions.write_text("[]")
    result = evaluate(
        SimpleNamespace(manifest={}),
        SimpleNamespace(chat_model="gemma3:4b"),
        questions,
        output,
    )
    assert result["summary"]["retrieval_metrics"] is None
    assert (
        result["summary"]["retrieval_checked"]
        == result["summary"]["retrieval_unscored"]
        == 0
    )
    assert output.is_file()


def test_cli_metric_cutoffs():
    assert parser().parse_args(["evaluate"]).k == [1, 3, 6]
    assert parser().parse_args(["evaluate", "--k", "1", "4"]).k == [1, 4]
    with pytest.raises(SystemExit):
        parser().parse_args(["evaluate", "--k", "0"])
