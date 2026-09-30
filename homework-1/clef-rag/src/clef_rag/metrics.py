"""Standard IR metrics over ranked, deduplicated paper/PDF-page pairs."""

import json
from statistics import fmean

import ir_measures as ir

DEFAULT_CUTOFFS = (1, 3, 6)
MEASURES = {
    "precision": ir.P,
    "recall": ir.R,
    "hit": ir.Success,
    "reciprocal_rank": ir.RR,
    "average_precision": ir.AP,
    "ndcg": ir.nDCG,
}
AGGREGATE_NAMES = {
    "hit": "hit_rate",
    "reciprocal_rank": "mrr",
    "average_precision": "map",
}


def normalize_cutoffs(cutoffs):
    cutoffs = tuple(cutoffs)
    if not cutoffs or any(type(k) is not int or k < 1 for k in cutoffs):
        raise ValueError("Retrieval metric cutoffs must be positive integers.")
    return tuple(sorted(set(cutoffs)))


def page_key(paper, page):
    if not isinstance(paper, str) or not paper.strip():
        raise ValueError("Retrieval judgments require a nonempty paper ID.")
    if type(page) is not int or page < 1:
        raise ValueError("Retrieval judgments require positive, one-based PDF pages.")
    # JSON encoding avoids collisions between paper IDs and page delimiters.
    return json.dumps([paper, page], ensure_ascii=False)


def page_run(sources):
    """Use first occurrence order, independent of display rank or Chroma scores."""
    pages = dict.fromkeys(page_key(s["document_id"], s["page"]) for s in sources)
    return {page: float(-rank) for rank, page in enumerate(pages, 1)}


def page_qrels(gold_sources):
    pages = {}
    for source in gold_sources:
        if not isinstance(source.get("pages"), list) or not source["pages"]:
            raise ValueError("Each gold source must contain at least one PDF page.")
        for page in source["pages"]:
            pages[page_key(source["paper"], page)] = 1
    if not pages:
        raise ValueError("Retrieval metrics require at least one gold page.")
    return pages


def retrieval_metrics(sources, gold_sources, cutoffs=DEFAULT_CUTOFFS):
    """Unlisted pages have binary relevance zero; short rankings are not padded with hits."""
    cutoffs = normalize_cutoffs(cutoffs)
    qrels, run = page_qrels(gold_sources), page_run(sources)
    measures = [measure @ k for k in cutoffs for measure in MEASURES.values()]
    scores = ir.calc_aggregate(measures, {"question": qrels}, {"question": run})
    return {
        str(k): {name: scores[measure @ k] for name, measure in MEASURES.items()}
        for k in cutoffs
    }


def aggregate_retrieval_metrics(rows):
    """Macro-average scored queries only; no judgments means undefined, not zero."""
    scored = [row["retrieval_metrics"] for row in rows if "retrieval_metrics" in row]
    if not scored:
        return None
    return {
        k: {
            AGGREGATE_NAMES.get(name, name): fmean(scores[k][name] for scores in scored)
            for name in MEASURES
        }
        for k in scored[0]
    }
