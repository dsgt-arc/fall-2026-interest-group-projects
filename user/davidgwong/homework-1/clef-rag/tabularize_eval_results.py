#!/usr/bin/env python3
"""
tabularize_eval_results.py

Builds a comparison table across RAG-evaluation JSON files that differ by
retrieval parameters:
    - retrieval_eval["retriever weights [semantic, lexical]"]
    - retrieval_eval["max candidate to retrieve:"]
    - retrieval_eval["search top k (retrieval depth)"]

One row per question per file. No config/metadata noise (embedding model,
chunk size, doc counts, etc.) - just the three swept parameters, the
question/expected/generated answer, the gold vs. retrieved source pages,
and the per-question metrics.

Output: a single CSV, questions.csv, with columns:
    Candidate Amount | Retrieval Depth | Weight Pair | Run |
    Question | Expected Answer | Generated Answer |
    Gold Source(s) | Retrieved Source(s) | Gold Rank(s) | Gold Covered |
    Category | Kind | Supported | Retrieval Pass | Error | Elapsed (s) |
    Precision@1 | Recall@1 | Hit@1 | MRR@1 | AP@1 | nDCG@1 |
    Precision@3 | Recall@3 | Hit@3 | MRR@3 | AP@3 | nDCG@3 |
    Precision@6 | Recall@6 | Hit@6 | MRR@6 | AP@6 | nDCG@6
(cutoffs other than 1/3/6 are picked up automatically if present)

Also writes by_params.csv: the same metrics averaged across all questions,
grouped by (Candidate Amount, Retrieval Depth, Weight Pair) - the fastest
way to see which parameter combination wins.

Usage:
    python tabularize_eval_results.py /path/to/json_dir --out ./tables
    python tabularize_eval_results.py "runs/*.json" --out ./tables
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

# Exact keys as they appear in the "retrieval_evaluation" block of each file.
KEY_WEIGHTS = "retriever weights [semantic, lexical]"
KEY_CANDIDATES = "max candidate to retrieve:"
KEY_DEPTH = "search top k (retrieval depth)"

METRIC_LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "hit": "Hit",
    "reciprocal_rank": "MRR",
    "average_precision": "AP",
    "ndcg": "nDCG",
}


def format_pages(pages, with_rank=False):
    """[['paper408', 1], ['paper408', 5]] -> 'paper408:p1; paper408:p5'
    With with_rank=True -> 'rank1 paper408:p1; rank2 paper408:p5'
    (retrieved_pages is already ordered by first-retrieved rank)."""
    if not pages:
        return None
    if with_rank:
        return "; ".join(f"rank{i} {doc}:p{page}" for i, (doc, page) in enumerate(pages, start=1))
    return "; ".join(f"{doc}:p{page}" for doc, page in pages)


def gold_ranks(gold_pages, retrieved_pages):
    """For each gold (doc, page), report its 1-based rank within
    retrieved_pages, or 'not retrieved' if it never shows up.
    e.g. 'paper1:p1 -> rank 1; paper1:p9 -> not retrieved'"""
    if not gold_pages:
        return None
    retrieved_pages = retrieved_pages or []
    rank_lookup = {tuple(p): i for i, p in enumerate(retrieved_pages, start=1)}
    parts = []
    for doc, page in gold_pages:
        rank = rank_lookup.get((doc, page))
        parts.append(f"{doc}:p{page} -> rank {rank}" if rank else f"{doc}:p{page} -> not retrieved")
    return "; ".join(parts)


def gold_covered(gold_pages, retrieved_pages):
    """True if every gold (doc, page) pair appears among the retrieved
    pages, False if any is missing, None if there's no gold to check."""
    if not gold_pages:
        return None
    retrieved_set = {tuple(p) for p in (retrieved_pages or [])}
    return all(tuple(g) in retrieved_set for g in gold_pages)


def load_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rev = data.get("retrieval_evaluation", {})
    candidate_amount = rev.get(KEY_CANDIDATES)
    retrieval_depth = rev.get(KEY_DEPTH)
    weights = rev.get(KEY_WEIGHTS)
    weight_pair = "/".join(str(w) for w in weights) if isinstance(weights, list) else weights

    rows = []
    for r in data.get("results", []):
        gold_pages = r.get("gold_pages")
        retrieved_pages = r.get("retrieved_pages")
        row = {
            "Candidate Amount": candidate_amount,
            "Retrieval Depth": retrieval_depth,
            "Weight Pair (semantic/lexical)": weight_pair,
            "Run": path.name,
            "Question": r.get("question"),
            "Expected Answer": r.get("expected_answer"),
            "Generated Answer": r.get("answer"),
            "Gold Source(s)": format_pages(gold_pages),
            "Retrieved Source(s)": format_pages(retrieved_pages, with_rank=True),
            "Gold Rank(s)": gold_ranks(gold_pages, retrieved_pages),
            "Gold Covered": gold_covered(gold_pages, retrieved_pages),
            "Category": r.get("category"),
            "Kind": r.get("kind"),
            "Supported": r.get("supported"),
            "Retrieval Pass": r.get("retrieval_pass"),
            "Error": r.get("error"),
            "Elapsed (s)": r.get("elapsed_seconds"),
        }
        for cutoff, metrics in (r.get("retrieval_metrics") or {}).items():
            for raw_key, value in metrics.items():
                label = METRIC_LABELS.get(raw_key, raw_key)
                row[f"{label}@{cutoff}"] = value
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Directory of .json files, or a glob pattern")
    ap.add_argument("--out", default="./eval_tables", help="Output directory for CSVs")
    args = ap.parse_args()

    in_path = Path(args.input)
    if in_path.is_dir():
        files = sorted(in_path.rglob("*.json"))
    else:
        files = sorted(Path(p) for p in glob.glob(args.input, recursive=True))

    if not files:
        print(f"No JSON files found for input: {args.input}", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    for path in files:
        try:
            all_rows.extend(load_file(path))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Skipping {path} ({e})", file=sys.stderr)

    df = pd.DataFrame(all_rows)

    # Put metric columns in a sensible cutoff-then-metric order at the end.
    param_cols = ["Candidate Amount", "Retrieval Depth", "Weight Pair (semantic/lexical)", "Run"]
    text_cols = ["Question", "Expected Answer", "Generated Answer",
                 "Gold Source(s)", "Retrieved Source(s)", "Gold Rank(s)", "Gold Covered"]
    other_cols = ["Category", "Kind", "Supported", "Retrieval Pass", "Error", "Elapsed (s)"]
    metric_cols = sorted(
        [c for c in df.columns if c not in param_cols + text_cols + other_cols],
        key=lambda c: (int(c.split("@")[1]), c.split("@")[0])
    )
    df = df[param_cols + text_cols + other_cols + metric_cols]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "questions.csv", index=False)
    print(f"Questions: {len(df)} rows -> {out_dir/'questions.csv'}")

    # Aggregate view: mean of every numeric metric per parameter combination,
    # plus the gold-coverage rate (share of questions whose gold source(s)
    # were among the retrieved pages).
    numeric_metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])]
    agg_cols = numeric_metric_cols.copy()
    if "Gold Covered" in df.columns:
        df["Gold Coverage Rate"] = df["Gold Covered"].astype("boolean").astype("Float64")
        agg_cols.append("Gold Coverage Rate")

    if agg_cols:
        grouped = (
            df.groupby(["Candidate Amount", "Retrieval Depth", "Weight Pair (semantic/lexical)"])[agg_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        grouped.to_csv(out_dir / "by_params.csv", index=False)
        print(f"By-parameter averages -> {out_dir/'by_params.csv'}")


if __name__ == "__main__":
    main()