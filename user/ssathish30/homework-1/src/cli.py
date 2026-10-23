"""Command-line interface for the local CLEF corpus assistant."""

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

from clef_rag.answer import ask, catalog_answer
from clef_rag.catalog import resolve_papers, stats
from clef_rag.config import CHAT_MODELS, RagError
from clef_rag.index import Index
from clef_rag.ingest import ingest
from clef_rag.models import LocalModels
from clef_rag.retrieve import Retriever


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parser():
    root = argparse.ArgumentParser(
        description="Ask questions about CLEF 2026 using local Google models through Ollama."
    )
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("ingest", "Extract PDFs and build or update the local index"),
        ("stats", "Show exact corpus coverage and extraction issues"),
        ("papers", "List catalog papers by track, author, title or ID"),
        ("search", "Inspect passages found by hybrid retrieval"),
        ("ask", "Answer a question with paper/page citations"),
        ("evaluate", "Run the checked question set and save results"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--index-dir", type=Path, default=Path("index"))
        command.add_argument(
            "--json", action="store_true", help="Print machine-readable output"
        )
        if name == "ingest":
            command.add_argument("--data-dir", type=Path, default=Path("data"))
            command.add_argument(
                "--catalog-html",
                type=Path,
                help="Use a saved CEUR HTML page (offline ingestion)",
            )
            command.add_argument(
                "--refresh-catalog",
                action="store_true",
                help="Fetch a new catalog snapshot",
            )
            command.add_argument("--batch-size", type=positive, default=16)
        if name in ("stats", "papers", "search", "ask"):
            command.add_argument("--track", help="Track name or substring")
            command.add_argument("--author", help="Author name or substring")
        if name in ("papers", "search", "ask"):
            command.add_argument(
                "--paper",
                action="append",
                default=[],
                help="Paper ID or title; repeat to compare papers",
            )
        if name == "papers":
            command.add_argument("--title", help="Title substring")
        if name in ("search", "ask"):
            command.add_argument("question")
        if name in ("search", "evaluate"):
            command.add_argument(
                "--top-k",
                type=positive,
                default=6,
                help="Passages returned per question; raise it to measure recall at depth",
            )
        if name in ("search", "evaluate"):
            command.add_argument(
                "--weights",
                type=float,
                nargs=2,
                default=[0.5, 0.5],
                metavar=("SEMANTIC", "LEXICAL"),
                help="Rank-fusion weights; use 1 0 or 0 1 to isolate one retriever",
            )
            command.add_argument(
                "--rrf-c",
                type=positive,
                default=60,
                help="Reciprocal-rank-fusion constant (default: 60)",
            )
            command.add_argument(
                "--candidates",
                type=positive,
                default=20,
                help="Candidates each retriever contributes before fusion",
            )
            command.add_argument(
                "--no-introductions",
                action="store_true",
                help="Skip the opening-context passage so metrics score the fused ranking",
            )
            command.add_argument(
                "--rerank-depth",
                type=int,
                default=0,
                help="Retrieve this many passages then rerank with the chat model down to --top-k",
            )
            command.add_argument(
                "--shuffle-seed",
                type=int,
                help="Randomize the fused order; negative control for evaluation validity",
            )
        if name in ("ask", "evaluate"):
            command.add_argument("--model", choices=CHAT_MODELS, default=CHAT_MODELS[0])
        if name == "evaluate":
            command.add_argument(
                "--k",
                type=positive,
                nargs="+",
                default=[1, 3, 6],
                help="Metric cutoffs over unique retrieved pages (default: 1 3 6); does not change retrieval depth",
            )
            command.add_argument(
                "--questions", type=Path, default=Path("eval/questions.json")
            )
            command.add_argument(
                "--output", type=Path, default=Path("index/evaluation.json")
            )
            command.add_argument(
                "--retrieval-only", action="store_true", help="Skip generated answers"
            )
            command.add_argument(
                "--ignore-question-filters",
                action="store_true",
                help="Search the whole corpus instead of each question's papers/track",
            )
    return root


def retrieval_options(args):
    """Collect the ablation settings shared by search and evaluate."""

    return {
        "weights": tuple(args.weights),
        "rrf_c": args.rrf_c,
        "candidates": args.candidates,
        "introductions": not args.no_introductions,
        "shuffle_seed": args.shuffle_seed,
        "rerank_depth": args.rerank_depth,
    }


def render_sources(sources):
    for source in sources:
        print(
            f"\n[{source['source_id']}] {source['title']}\n"
            f"  {source['filename']}, PDF page {source['page']}\n"
            f"  {source['url']}#page={source['page']}"
        )


def main(argv=None):
    args = parser().parse_args(argv)
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    started = time.perf_counter()
    try:
        if args.command == "ingest":
            result = ingest(
                args.data_dir,
                args.index_dir,
                LocalModels(),
                html_file=args.catalog_html,
                refresh=args.refresh_catalog,
                batch_size=args.batch_size,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            print(
                json.dumps(result, indent=2)
                if args.json
                else f"Indexed {result['documents']} catalog entries and {result['chunks']} passages in "
                f"{result['ingestion_seconds']:.1f}s. Use clef-rag stats to inspect coverage."
            )
            return 0
        with Index(args.index_dir) as index:
            if args.command == "stats":
                result = stats(index.db, args.track, args.author)
                result["index"] = index.manifest
                if not args.json:
                    print(
                        catalog_answer(index, "stats", args.track, args.author)[
                            "answer"
                        ]
                    )
                    print(f"\nPassages: {index.manifest['chunks']}")
                    print(
                        "Missing papers: "
                        + (", ".join(result["missing_papers"]) or "none")
                    )
                    print(
                        f"Documents with extraction issues: {len(result['extraction_issues'])}"
                    )
                    for issue in result["extraction_issues"]:
                        print(
                            f"- {issue['id']}: {issue['status']}; {issue['error'] or '; '.join(issue['warnings'])}"
                        )
            elif args.command == "papers":
                ids = resolve_papers(index.db, args.paper) if args.paper else None
                result = catalog_answer(
                    index, "papers", args.track, args.author, args.title, ids
                )
                if not args.json:
                    print(result["answer"])
            elif args.command == "search":
                ids = resolve_papers(index.db, args.paper) if args.paper else None
                sources = Retriever(
                    index, LocalModels(), **retrieval_options(args)
                ).search(
                    args.question,
                    track=args.track,
                    author=args.author,
                    paper_ids=ids,
                    top_k=args.top_k,
                )
                result = {"question": args.question, "sources": sources}
                if not args.json:
                    print(
                        f"{len(sources)} matching passages (search examples, not an exhaustive paper list):"
                    )
                    for source in sources:
                        render_sources([source])
                        print(f"  Rank: {source['rank']}\n{source['text']}")
            elif args.command == "ask":
                result = ask(
                    index,
                    LocalModels(args.model),
                    args.question,
                    track=args.track,
                    author=args.author,
                    paper_refs=args.paper,
                )
                if not args.json:
                    print(result["answer"])
                    render_sources(result["sources"])
            else:
                from clef_rag.evaluate import evaluate

                result = evaluate(
                    index,
                    LocalModels(args.model),
                    args.questions,
                    args.output,
                    args.retrieval_only,
                    cutoffs=args.k,
                    retrieval=retrieval_options(args),
                    ignore_filters=args.ignore_question_filters,
                    top_k=args.top_k,
                )
                if not args.json:
                    print(json.dumps(result["summary"], indent=2))
                    print(f"Detailed results: {args.output}")
            result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (RagError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Completed ingestion batches are cached; rerun to resume.",
            file=sys.stderr,
        )
        return 130
