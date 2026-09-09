"""Record retrieval checks and generated answers for manual citation review."""

import json
import time

from clef_rag.answer import ask
from clef_rag.catalog import resolve_papers
from clef_rag.metrics import (
    DEFAULT_CUTOFFS,
    aggregate_retrieval_metrics,
    normalize_cutoffs,
    page_qrels,
    page_run,
    retrieval_metrics,
)
from clef_rag.retrieve import Retriever


def evaluate(
    index,
    models,
    questions_path,
    output_path,
    retrieval_only=False,
    *,
    cutoffs=DEFAULT_CUTOFFS,
):
    cutoffs = normalize_cutoffs(cutoffs)
    cases = json.loads(questions_path.read_text())
    output = {
        "index": index.manifest,
        "model": models.chat_model,
        "retrieval_evaluation": {
            "implementation": "ir-measures",
            "cutoffs": list(cutoffs),
            "unit": "unique paper/PDF-page pair, ordered by first retrieved occurrence",
            "relevance": "binary; every listed gold page is relevant; unlisted pages score zero",
            "ranking": "final retrieved passages, before answer context selection; no extra retrieval for larger cutoffs",
            "aggregation": "macro mean over successfully scored questions with gold pages; errors excluded and counted",
        },
        "results": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        if retrieval_only and case["category"] in ("catalog", "unsupported"):
            continue
        started = time.perf_counter()
        row = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
        }
        try:
            if case.get("gold_sources"):
                # Validate judgments before making model requests and retain them for review.
                row["gold_pages"] = [
                    json.loads(key) for key in page_qrels(case["gold_sources"])
                ]
            if retrieval_only:
                ids = resolve_papers(index.db, case.get("papers", [])) or None
                sources = Retriever(index, models).search(
                    case["question"], paper_ids=ids, track=case.get("track")
                )
                row["retrieved_sources"] = sources
            else:
                result = ask(
                    index,
                    models,
                    case["question"],
                    paper_refs=case.get("papers"),
                    track=case.get("track"),
                )
                row.update(result)
                sources = result.get("retrieved_sources", [])
                if case["category"] == "catalog":
                    row["catalog_pass"] = all(
                        result.get("data", {}).get(k) == v
                        for k, v in case["expected_counts"].items()
                    )
                if case["category"] == "unsupported":
                    row["abstention_pass"] = (
                        result.get("supported") is False
                        or result.get("kind") == "unsupported"
                    )
                row["manual_review"] = (
                    "Pending: verify factual claims and page support against expected_answer."
                )
                row["expected_answer"] = case.get("expected_answer")
            if case.get("gold_sources"):
                row["retrieved_pages"] = [json.loads(key) for key in page_run(sources)]
                row["retrieval_metrics"] = retrieval_metrics(
                    sources, case["gold_sources"], cutoffs
                )
                row["retrieval_pass"] = all(
                    any(
                        s["document_id"] == gold["paper"] and s["page"] in gold["pages"]
                        for s in sources
                    )
                    for gold in case["gold_sources"]
                )
        except Exception as exc:  # noqa: BLE001 - retain failures and continue the evaluation
            row["error"] = str(exc)
        row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        output["results"].append(row)
        # Save after every case so a slow or interrupted run is still reviewable.
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    rows = output["results"]
    output["summary"] = {
        "cases_run": len(rows),
        "errors": sum("error" in row for row in rows),
        "retrieval_passed": sum(row.get("retrieval_pass", False) for row in rows),
        "retrieval_eligible": sum(
            bool(c.get("gold_sources"))
            for c in cases
            if not retrieval_only or c["category"] not in ("catalog", "unsupported")
        ),
        "retrieval_checked": sum("retrieval_metrics" in row for row in rows),
        "retrieval_metrics": aggregate_retrieval_metrics(rows),
        "catalog_passed": sum(row.get("catalog_pass", False) for row in rows),
        "abstention_passed": sum(row.get("abstention_pass", False) for row in rows),
        "answerable_questions_supported": sum(
            row.get("supported") is True
            for row in rows
            if row["category"] in ("paper", "synthesis")
        ),
        "manual_claim_review": "Required; automatic checks do not establish factual correctness.",
    }
    output["summary"]["retrieval_unscored"] = (
        output["summary"]["retrieval_eligible"] - output["summary"]["retrieval_checked"]
    )
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    return output
