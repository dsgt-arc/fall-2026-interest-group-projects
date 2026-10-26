"""Retrieval experiments: one factor at a time on dev, combine the winners, report once on test.

    uv run --project ../../../homework-1/clef-rag python experiments.py parity
    uv run --project ../../../homework-1/clef-rag python experiments.py dev
    uv run --project ../../../homework-1/clef-rag python experiments.py final --best '<json overrides>'

Scoring reuses the starter's ir-measures code (clef_rag.metrics) unchanged: unique paper/PDF-page
pairs in first-retrieved order, binary relevance, cutoffs 1/3/6.
"""

import argparse
import json
import random
import time
from statistics import fmean

from clef_rag.catalog import resolve_papers
from clef_rag.metrics import aggregate_retrieval_metrics, retrieval_metrics
from clef_rag.retrieve import Retriever
from common import PILOT_QUESTIONS, RESULTS, TESTSET, load_json, open_index, save_json
from engine import Config, Engine

BASE = Config()


def cases(split):
    if split == "pilot":
        return [c for c in load_json(PILOT_QUESTIONS) if c.get("gold_sources")]
    return [c for c in load_json(TESTSET) if c["split"] == split]


def run(engine, cfg, rows, index):
    out = []
    for c in rows:
        ids = resolve_papers(index.db, c.get("papers", [])) or None
        src = engine.search(c["question"], cfg, paper_ids=ids, track=c.get("track"))
        m = retrieval_metrics(src, c["gold_sources"])
        gold_papers = {g["paper"] for g in c["gold_sources"]}
        out.append({
            "id": c["id"],
            "retrieval_metrics": m,
            "paper_hit@6": float(any(s["document_id"] in gold_papers for s in src)),
            "pages": [[s["document_id"], s["page"]] for s in src],
        })
    agg = aggregate_retrieval_metrics(out)
    agg["paper_hit@6"] = fmean(r["paper_hit@6"] for r in out)
    return agg, out


def per_query(rows, metric="ndcg", k="6"):
    return [r["retrieval_metrics"][k][metric] for r in rows]


def paired_test(a, b, n=20000, seed=0):
    """Two-sided paired randomization (sign-flip) test on per-question differences."""
    d = [y - x for x, y in zip(a, b)]
    obs = abs(fmean(d))
    rng = random.Random(seed)
    hits = sum(abs(fmean(x if rng.random() < 0.5 else -x for x in d)) >= obs - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def line(name, agg):
    m6, m1 = agg["6"], agg["1"]
    return (f"| {name} | {m1['hit_rate']:.3f} | {m6['hit_rate']:.3f} | {m6['mrr']:.3f} | "
            f"{m6['ndcg']:.3f} | {m6['recall']:.3f} | {agg['paper_hit@6']:.3f} |")


HEADER = ("| Config | Hit@1 | Hit@6 | MRR@6 | nDCG@6 | Recall@6 | Paper hit@6 |\n"
          "| --- | --- | --- | --- | --- | --- | --- |")


def parity(engine, index, models):
    """Does the in-memory replica return the same pages as the real starter retriever?"""
    starter = Retriever(index, models)
    rows = cases("pilot") + cases("dev")[:40]
    same, overlap = 0, []
    for c in rows:
        ids = resolve_papers(index.db, c.get("papers", [])) or None
        a = [(s["document_id"], s["page"]) for s in starter.search(c["question"], paper_ids=ids, track=c.get("track"))]
        b = [(s["document_id"], s["page"]) for s in engine.search(c["question"], BASE, paper_ids=ids, track=c.get("track"))]
        same += a == b
        overlap.append(len(set(a) & set(b)) / max(1, len(set(a) | set(b))))
    res = {"questions": len(rows), "identical_page_lists": same, "mean_page_jaccard": fmean(overlap)}
    save_json(RESULTS / "parity.json", res)
    print(res)


def ann_recall(engine):
    """How many of the true (exact cosine) top-20 neighbors does the starter's HNSW query return?"""
    import numpy as np

    mask = np.ones(len(engine.ids), dtype=bool)
    qs = [c["question"] for c in load_json(TESTSET)]
    overlap = [len(set(engine.dense_hnsw(q, mask, 20)) & set(engine.dense(q, mask, 20, 768))) / 20 for q in qs]
    res = {
        "questions": len(qs),
        "mean_recall_at_20": fmean(overlap),
        "questions_below_0.5": sum(o < 0.5 for o in overlap),
        "questions_perfect": sum(o == 1 for o in overlap),
        "hnsw_config": engine.index.collection.configuration_json.get("hnsw") if hasattr(engine.index.collection, "configuration_json") else None,
    }
    save_json(RESULTS / "ann_recall.json", res)
    print(res)


def dev_sweep(engine, index):
    rows = cases("dev")
    grid = [
        BASE,
        BASE.but(name="exact dense search (no HNSW)", dense_search="exact"),
        BASE.but(name="intro: only when papers are named", intro="named"),
        BASE.but(name="dense only", w_bm25=0),
        BASE.but(name="BM25 only", w_dense=0),
        BASE.but(name="RRF weights 0.7 dense / 0.3 BM25", w_dense=0.7, w_bm25=0.3),
        BASE.but(name="RRF weights 0.3 dense / 0.7 BM25", w_dense=0.3, w_bm25=0.7),
        BASE.but(name="50 candidates each", dense_k=50, bm25_k=50),
        BASE.but(name="100 candidates each", dense_k=100, bm25_k=100),
        BASE.but(name="RRF c=20", c=20),
        BASE.but(name="exact search, embedding dims 512", dense_search="exact", dims=512),
        BASE.but(name="exact search, embedding dims 256", dense_search="exact", dims=256),
        BASE.but(name="exact search, embedding dims 128", dense_search="exact", dims=128),
    ]
    report = {}
    print(HEADER)
    for cfg in grid:
        t = time.perf_counter()
        agg, _ = run(engine, cfg, rows, index)
        report[cfg.name] = {"config": cfg.__dict__, "metrics": agg, "seconds": round(time.perf_counter() - t, 1)}
        print(line(cfg.name, agg), flush=True)
    save_json(RESULTS / "dev_sweep.json", report)


def evaluate_split(engine, index, split, configs):
    rows = cases(split)
    report, per = {}, {}
    print(f"\n## {split} ({len(rows)} questions)\n" + HEADER)
    for cfg in configs:
        agg, out = run(engine, cfg, rows, index)
        report[cfg.name] = {"config": cfg.__dict__, "metrics": agg}
        per[cfg.name] = out
        print(line(cfg.name, agg), flush=True)
    base = per[configs[0].name]
    for cfg in configs[1:]:
        report[cfg.name]["p_vs_baseline"] = {
            m: paired_test(per_query(base, m), per_query(per[cfg.name], m))
            for m in ("ndcg", "reciprocal_rank", "hit")
        }
    save_json(RESULTS / f"{split}_results.json", {"summary": report, "per_question": per})
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["parity", "ann", "dev", "final"])
    ap.add_argument("--best", default="{}", help="JSON overrides of Config for the chosen system")
    ap.add_argument("--extra", default="[]", help="JSON list of extra configs to report on test")
    args = ap.parse_args()
    index, models = open_index()
    engine = Engine(index, models)
    if args.stage == "parity":
        parity(engine, index, models)
    elif args.stage == "ann":
        ann_recall(engine)
    elif args.stage == "dev":
        dev_sweep(engine, index)
    else:
        best = BASE.but(**json.loads(args.best))
        extra = [BASE.but(**e) for e in json.loads(args.extra)]
        configs = [BASE, *extra, best]
        for split in ("dev", "test", "pilot"):
            evaluate_split(engine, index, split, configs)


if __name__ == "__main__":
    main()
