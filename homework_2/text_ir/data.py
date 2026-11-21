"""Download SciFact and load its fixed training/validation file lists."""

import csv
import io
import json
import math
import os
import random
import zipfile
from pathlib import Path
from urllib.request import urlopen


def prepare_scifact(data_dir: Path) -> None:
    """Download only the corpus, queries, and official training relevance labels."""
    filenames = ("corpus.jsonl", "queries.jsonl", "qrels/train.tsv")
    if all((data_dir / name).is_file() for name in filenames):
        return
    print("Downloading BEIR SciFact...")
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    with urlopen(url, timeout=60) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for name in filenames:
            destination = data_dir / name
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(f"scifact/{name}"))


def prepare_splits(data_dir: Path, splits_path: Path, *, seed: int = 42) -> None:
    """Reuse existing lists, or reserve 20% of training claims for validation.

    Identical normalized query texts stay together to prevent leakage. Query files
    can be regenerated after a fresh download without changing split membership.
    """
    with (data_dir / "queries.jsonl").open() as stream:
        queries = {row["_id"]: row["text"] for row in map(json.loads, stream)}
    positives = {}
    with (data_dir / "qrels/train.tsv").open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if float(row["score"]) > 0:
                positives.setdefault(row["query-id"], set()).add(row["corpus-id"])
    # queries.jsonl also contains test claims; only train-qrel claims enter this split.
    queries = {qid: queries[qid] for qid in positives}
    if splits_path.exists():
        paths = json.loads(splits_path.read_text())
    else:
        groups = {}
        for qid in sorted(queries):
            normalized = " ".join(queries[qid].casefold().split())
            groups.setdefault(normalized, []).append(qid)
        groups = list(groups.values())
        if len(groups) < 2:
            raise ValueError("Need at least two distinct claims for a train/validation split.")
        random.Random(seed).shuffle(groups)
        holdout_ids = set()
        for group in groups[:-1]:  # Leave at least one complete group for training.
            holdout_ids.update(group)
            if len(holdout_ids) >= math.ceil(len(queries) * 0.2):
                break
        paths = {"train": [], "holdout": []}
        for qid in sorted(queries):
            if not qid.isdecimal():
                raise ValueError("SciFact query IDs must be numeric filenames.")
            path = data_dir / "split_records" / f"{qid}.json"
            relative = Path(os.path.relpath(path, splits_path.parent)).as_posix()
            paths["holdout" if qid in holdout_ids else "train"].append(relative)
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        splits_path.write_text(json.dumps(paths, indent=2) + "\n")

    for name in ("train", "holdout"):
        for relative in paths[name]:
            path = splits_path.parent / relative
            if not path.is_file():
                qid = path.stem
                record = {
                    "query_id": qid,
                    "text": queries[qid],
                    "positive_document_ids": sorted(positives[qid]),
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record, indent=2) + "\n")


def read_dataset(data_dir: Path, splits_path: Path):
    """Create missing split lists, then read and validate the shared dataset."""
    if not splits_path.exists():
        prepare_splits(data_dir, splits_path)
    with (data_dir / "corpus.jsonl").open() as stream:
        documents = {
            row["_id"]: f"{row.get('title', '')} {row['text']}".strip()
            for row in map(json.loads, stream)
        }
    paths = json.loads(splits_path.read_text())
    splits, seen_ids = {}, set()
    for name in ("train", "holdout"):
        if not paths[name]:
            raise ValueError("Training and validation each need at least one query.")
        records = {}
        for relative in paths[name]:
            if Path(relative).is_absolute():
                raise ValueError("Split paths must be relative to splits.json.")
            record = json.loads((splits_path.parent / relative).read_text())
            qid = record["query_id"]
            if qid in seen_ids:
                raise ValueError("A query appears more than once in the split lists.")
            positives = set(record["positive_document_ids"])
            if not positives or not positives <= documents.keys():
                raise ValueError("Each query needs positive documents that exist in the corpus.")
            seen_ids.add(qid)
            records[qid] = record
        splits[name] = records
    texts = {
        name: {" ".join(row["text"].casefold().split()) for row in records.values()}
        for name, records in splits.items()
    }
    if texts["train"] & texts["holdout"]:
        raise ValueError("Identical query text crosses the train/validation boundary.")
    return documents, splits
