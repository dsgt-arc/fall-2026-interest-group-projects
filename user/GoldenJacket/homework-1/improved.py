"""The improved retriever inside the starter's own answer pipeline, plus a side-by-side comparison.

Only retrieval changes: routing, the answer prompt, citation repair, and the models are the
starter's code. The chosen settings live in results/best_config.json (written after the dev sweep).

    uv run --project ../../../homework-1/clef-rag python improved.py ask "question" [--track BioASQ]
    uv run --project ../../../homework-1/clef-rag python improved.py compare questions/my_questions.json
"""

import argparse
import json
import time

import clef_rag.answer as starter_answer
from clef_rag.catalog import list_documents
from clef_rag.retrieve import Retriever as StarterRetriever
from common import HERE, RESULTS, load_json, open_index, save_json
from engine import Config, Engine

_ENGINE = None


def best_config():
    path = RESULTS / "best_config.json"
    return Config(**load_json(path)) if path.exists() else Config()


class ImprovedRetriever:
    """Same interface as clef_rag.retrieve.Retriever."""

    def __init__(self, index, models):
        global _ENGINE
        if _ENGINE is None:
            _ENGINE = Engine(index, models)
        self.index, self.engine, self.cfg = index, _ENGINE, best_config()

    def search(self, question, *, track=None, author=None, paper_ids=None, top_k=6):
        restrict = None
        if author:
            restrict = {d["id"] for d in list_documents(self.index.db, author=author)}
        sources = self.engine.search(
            question, self.cfg.but(top_k=top_k), paper_ids=paper_ids, track=track, restrict=restrict
        )
        for n, s in enumerate(sources, 1):
            s["source_id"] = f"S{n}"
        return sources


def ask_with(retriever_cls, index, models, question, **kw):
    starter_answer.Retriever = retriever_cls  # swap retrieval only
    try:
        started = time.perf_counter()
        result = starter_answer.ask(index, models, question, **kw)
        result["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        return result
    finally:
        starter_answer.Retriever = StarterRetriever


def brief(result):
    used = {u for u in result.get("used_sources", result.get("citations", [])) or []}
    return {
        "answer": result.get("answer"),
        "supported": result.get("supported"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "retrieved": [
            {"source": s.get("source_id"), "paper": s["document_id"], "page": s["page"], "title": s.get("title")}
            for s in result.get("retrieved_sources", [])
        ],
        "used_sources": sorted(used) if used else None,
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--track")
    c = sub.add_parser("compare")
    c.add_argument("questions", help="JSON list of {question, track?}")
    args = ap.parse_args()
    index, models = open_index()
    if args.cmd == "ask":
        print(json.dumps(brief(ask_with(ImprovedRetriever, index, models, args.question, track=args.track)), indent=2))
        return
    rows = []
    for q in load_json(HERE / args.questions if not args.questions.startswith("/") else args.questions):
        row = {"question": q["question"], "track": q.get("track")}
        for name, cls in (("baseline", StarterRetriever), ("improved", ImprovedRetriever)):
            try:
                row[name] = brief(ask_with(cls, index, models, q["question"], track=q.get("track")))
            except Exception as exc:  # noqa: BLE001
                row[name] = {"error": str(exc)}
        rows.append(row)
        print(f"done: {q['question'][:70]}", flush=True)
        save_json(RESULTS / "my_questions_answers.json", rows)


if __name__ == "__main__":
    main()
