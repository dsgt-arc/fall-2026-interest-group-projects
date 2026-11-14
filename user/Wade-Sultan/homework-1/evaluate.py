"""Diagnose the configs in configs.toml against five common RAG failure modes.

Each check is one section of results/diagnostics.md:
  chunking       index-wide chunk shape, and whether each gold_quote survives as one chunk
  embedding      dense-only vs BM25-only retrieval, question terms absent from the corpus,
                 and robustness to optional paraphrases
  noise          off-target passages in the final ranking; with --generate, whether hard
                 distractors from other papers change answers or get cited
  context        what answer.select_evidence would drop at the configured top_k; with
                 --generate, whether answers depend on where the gold passages sit
  hallucination  with --generate: numbers and words in answers that their citations do
                 not contain, recall of expected_answer, and abstention on unsupported questions

Retrieval checks take seconds. --generate calls the local chat model about 15 times
per question, so it is slow; use --only to try a few questions first.

Optional question fields, beyond the starter's questions.json format:
  gold_quotes  exact passages that contain the answer, as they appear in extracted text
  paraphrases  rewordings of the question that should retrieve the same gold pages
"""

import argparse
import json
import random
import re
import statistics
import zlib
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from clef_rag.answer import generate_answer, route_question, select_evidence
from clef_rag.catalog import resolve_papers
from clef_rag.config import CONTEXT_TOKENS, RagError
from clef_rag.index import Index
from clef_rag.metrics import normalize_cutoffs, retrieval_metrics
from clef_rag.models import LocalModels
from clef_rag.retrieve import lexical_tokens, source_record

from experiments import RETRIEVER_KEYS, load_configs
from retriever import ConfigurableRetriever, Settings

HERE = Path(__file__).resolve().parent
STARTER = HERE / "../../../homework-1/clef-rag"
# A subdirectory, so experiments.py (which globs results/*.json) never reads these files.
RESULTS = HERE / "results" / "diagnostics"
SUMMARY = HERE / "results" / "diagnostics.md"

# Chunks smaller than this are mostly page tails with too little context to embed well.
FRAGMENT_BYTES = 400
# Mirrors the evidence budget in clef_rag.answer.select_evidence.
EVIDENCE_BUDGET = 12000
# Size of generate_answer's system prompt, and a rough bytes-per-token rate for English
# PDF text. Together they estimate prompt size; Ollama does not report it through the starter.
SYSTEM_PROMPT_BYTES = 1321
BYTES_PER_TOKEN = 3.5
# A sentence end followed by a numbered heading such as "4.3 Results". Approximate.
HEADING = re.compile(r"[.!?] \d{1,2}(?:\.\d{1,2}){0,3}\.? [A-Z][a-z]")
CITATION = re.compile(r"\[S\d+\]")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
STOPWORDS = set(
    """a an and are as at be by can did do does for from has have how in is it its of on
    or that the their these this to was were what when where which who why will with
    paper papers""".split()
)


def norm(text):
    return re.sub(r"\s+", " ", text).casefold().strip()


def terms(text):
    return [t for t in lexical_tokens(text) if t not in STOPWORDS]


def stems(text):
    """Tokens with common English suffixes removed, so "uses" matches "use"."""
    # Dropping a final "e" as well lets "summarizes" and "summarize" meet at "summariz".
    return {
        re.sub(r"(?<=\w{3})e$", "", re.sub(r"(?<=\w{3})(ing|ed|es|(?<!s)s)$", "", t))
        for t in lexical_tokens(text)
    }


def stem(term):
    return next(iter(stems(term)))


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def rate(flags):
    return mean(float(f) for f in flags if f is not None)


def page_of(source):
    return (source["document_id"], source["page"])


def first_gold_rank(sources, gold):
    return next((i for i, s in enumerate(sources, 1) if page_of(s) in gold), None)


def renumber(sources):
    return [{**s, "source_id": f"S{i}", "rank": i} for i, s in enumerate(sources, 1)]


def brief(source):
    return {k: source[k] for k in ("source_id", "document_id", "page", "id")}


# ---------------------------------------------------------------- chunking


def chunk_shape(chunks):
    """Index-wide statistics over every indexed passage (as source records)."""
    sizes = sorted(len(c["text"].encode("utf-8")) for c in chunks)
    continued = [c for c in chunks if c["start_char"] > 0]
    page_starts = [c for c in chunks if c["start_char"] == 0 and c["page"] > 1]
    return {
        "chunks": len(chunks),
        "median_bytes": statistics.median(sizes),
        "fragment_rate": rate(s < FRAGMENT_BYTES for s in sizes),
        # Within a page, a lowercase start means the splitter cut mid-sentence.
        "mid_sentence_rate": rate(c["text"][:1].islower() for c in continued),
        # At the top of a page, it means a sentence was cut by the page break.
        "page_break_cut_rate": rate(c["text"][:1].islower() for c in page_starts),
        # A heading after the first few characters means the chunk spans two sections.
        "section_span_rate": rate(bool(HEADING.search(c["text"], 3)) for c in chunks),
    }


def longest_prefix(quote, texts):
    """Length of the longest prefix of quote that occurs inside a single text."""
    low, high = 0, len(quote)
    while low < high:
        mid = (low + high + 1) // 2
        if any(quote[:mid] in t for t in texts):
            low = mid
        else:
            high = mid - 1
    return low


def quote_status(quote, retrieved, paper_texts):
    q = norm(quote)
    if any(q in norm(s["text"]) for s in retrieved):
        return "retrieved"
    if any(q in t for t in paper_texts):
        return "not_retrieved"
    # Split: the quote's start and end live in different chunks that together cover it.
    head = longest_prefix(q, paper_texts)
    tail = longest_prefix(q[::-1], [t[::-1] for t in paper_texts])
    return "split" if head and tail and head + tail >= len(q) else "missing"


# ---------------------------------------------------------------- hallucination


def grounding(answer, cited, question):
    """Numbers and content words in the answer that neither its citations nor the question contain."""
    text = CITATION.sub("", answer)
    support = " ".join(s["text"] for s in cited) + " " + question
    vocab = stems(support)
    # Numbers are checked separately, so leave them out of the word check.
    words = [t for t in terms(text) if not t.isdigit()]
    return {
        "ungrounded_numbers": sorted(set(NUMBER.findall(text)) - set(NUMBER.findall(support))),
        "ungrounded_terms": sorted({w for w in words if stem(w) not in vocab}),
        "ungrounded_term_rate": rate(stem(w) not in vocab for w in words),
    }


def expected_recall(expected, answer):
    wanted = {stem(t) for t in terms(expected)}
    if not wanted:
        return None
    return len(wanted & stems(CITATION.sub("", answer))) / len(wanted)


def run_answer(case, sources, models, gold):
    try:
        result = generate_answer(case["question"], sources, models)
    except RagError as exc:
        # The starter withholds answers with invalid citations; that is a refusal, not a claim.
        return {"supported": False, "error": str(exc)}
    cited = result.get("sources", [])
    row = {
        "supported": result["supported"],
        "answer": result["answer"],
        "cited": [brief(s) for s in cited],
        "cited_gold": any(page_of(s) in gold for s in cited),
    }
    if result["supported"]:
        row.update(grounding(result["answer"], cited, case["question"]))
    if case.get("expected_answer"):
        row["expected_recall"] = (
            expected_recall(case["expected_answer"], result["answer"])
            if result["supported"]
            else 0.0
        )
    return row


# ---------------------------------------------------------------- per question


class Diagnoser:
    def __init__(self, config, index, models, chunks, vocab, generate, distractors):
        settings = Settings(**{k: config[k] for k in RETRIEVER_KEYS if k in config})
        self.fused = ConfigurableRetriever(index, models, settings)
        # Single-retriever variants skip the page-1 prepend, doc2query and reranking so the
        # ranking is that retriever's own; query strategies (expansion, HyDE...) still apply.
        alone = dict(prepend_intro=False, doc2query_weight=0.0, rerank_depth=0)
        self.dense = ConfigurableRetriever(
            index, models, replace(settings, semantic_weight=1.0, lexical_weight=0.0, **alone)
        )
        self.lexical = ConfigurableRetriever(
            index, models, replace(settings, semantic_weight=0.0, lexical_weight=1.0, **alone)
        )
        self.config, self.index, self.models = config, index, models
        self.cutoffs = normalize_cutoffs(config["cutoffs"])
        self.vocab, self.generate, self.distractors = vocab, generate, distractors
        self.by_paper = {}
        for chunk in chunks:
            self.by_paper.setdefault(chunk["document_id"], []).append(chunk)

    def search(self, retriever, question, case, **kwargs):
        papers = case.get("papers", []) if self.config["paper_filter"] else []
        return retriever.search(
            question,
            paper_ids=resolve_papers(self.index.db, papers) or None,
            track=case.get("track") if self.config["track_filter"] else None,
            **kwargs,
        )

    def run(self, case):
        gold = {(g["paper"], p) for g in case.get("gold_sources", []) for p in g["pages"]}
        gold_papers = {paper for paper, _ in gold}
        sources = self.search(self.fused, case["question"], case)
        row = {"retrieved": [brief(s) for s in sources]}
        if gold:
            row["chunking"] = self.chunking(case, sources, gold_papers)
            row["embedding"] = self.embedding(case, sources, gold, gold_papers)
            row["noise"] = self.noise(case, sources, gold, gold_papers)
            row["context"] = self.context(case, sources, gold)
        if self.generate:
            row["generation"] = self.generation(case, sources, gold, gold_papers)
        return row

    def chunking(self, case, sources, gold_papers):
        texts = [norm(c["text"]) for p in gold_papers for c in self.by_paper.get(p, [])]
        return {
            "retrieved_fragments": sum(
                len(s["text"].encode("utf-8")) < FRAGMENT_BYTES for s in sources
            ),
            "quotes": [
                {"quote": q, "status": quote_status(q, sources, texts)}
                for q in case.get("gold_quotes", [])
            ],
        }

    def embedding(self, case, sources, gold, gold_papers):
        question = case["question"]
        asked = set(terms(question))
        gold_vocab = {
            t
            for p in gold_papers
            for c in self.by_paper.get(p, [])
            if (p, c["page"]) in gold
            for t in lexical_tokens(c["text"])
        }
        return {
            "first_gold_rank": {
                "fused": first_gold_rank(sources, gold),
                "dense": first_gold_rank(self.search(self.dense, question, case), gold),
                "lexical": first_gold_rank(self.search(self.lexical, question, case), gold),
            },
            # Terms no passage contains: BM25 cannot match them, and they are often IDs like "paper6".
            "unmatched_terms": sorted(asked - self.vocab),
            "gold_overlap": len(asked & gold_vocab) / len(asked) if asked else None,
            "paraphrases": [
                {
                    "question": p,
                    "fused": first_gold_rank(self.search(self.fused, p, case), gold),
                    "dense": first_gold_rank(self.search(self.dense, p, case), gold),
                }
                for p in case.get("paraphrases", [])
            ],
        }

    def noise(self, case, sources, gold, gold_papers):
        k = self.cutoffs[-1]
        metrics = retrieval_metrics(sources, case["gold_sources"], self.cutoffs)[str(k)]
        return {
            "precision": metrics["precision"],
            "off_target": sum(s["document_id"] not in gold_papers for s in sources),
            "passages": len(sources),
            "distinct_papers": len({s["document_id"] for s in sources}),
            "rank1_gold": bool(sources) and page_of(sources[0]) in gold,
        }

    def context(self, case, sources, gold):
        question = case["question"]
        kept = select_evidence(question, sources)
        kept_ids = {s["id"] for s in kept}
        # select_evidence charges each whole serialized record, metadata included.
        used = sum(len(json.dumps(s, ensure_ascii=False).encode("utf-8")) for s in kept)
        evidence = [
            {
                "source_id": s["source_id"],
                "paper": s["document_id"],
                "title": s["title"],
                "pdf_page": s["page"],
                "text": s["text"],
            }
            for s in kept
        ]
        prompt = "Question: " + question + "\nEvidence:\n" + json.dumps(evidence, ensure_ascii=False)
        gold_retrieved = [s for s in sources if page_of(s) in gold]
        return {
            "dropped": len(sources) - len(kept),
            "gold_dropped": sum(s["id"] not in kept_ids for s in gold_retrieved),
            "gold_lost": bool(gold_retrieved) and not any(s["id"] in kept_ids for s in gold_retrieved),
            "budget_used": used / (EVIDENCE_BUDGET - len(question.encode("utf-8"))),
            "est_prompt_tokens": round(
                (SYSTEM_PROMPT_BYTES + len(prompt.encode("utf-8"))) / BYTES_PER_TOKEN
            ),
        }

    def generation(self, case, sources, gold, gold_papers):
        if case["category"] == "unsupported":
            route = route_question(case["question"], self.index.db, self.models)["operation"]
            if route == "unsupported_count":
                return {"route": route, "abstained": True}
            if route != "content":
                return {"route": route}  # a catalog answer; nothing to score
            base = run_answer(case, sources, self.models, gold)
            return {"route": route, "abstained": not base["supported"], "base": base}

        out = {"base": run_answer(case, sources, self.models, gold)}

        # Noise: hard distractors are what the configured retriever ranks highest across
        # the whole corpus once gold papers are excluded, inserted at seeded positions.
        held = {s["id"] for s in sources}
        pool = [
            s
            for s in self.fused.search(case["question"], top_k=self.distractors + len(sources) + 6)
            if s["document_id"] not in gold_papers and s["id"] not in held
        ][: self.distractors]
        rng = random.Random(zlib.crc32(case["id"].encode()))
        noisy = list(sources)
        for distractor in pool:
            noisy.insert(rng.randint(0, len(noisy)), distractor)
        out["noisy"] = run_answer(case, renumber(noisy), self.models, gold)
        out["noisy"]["distractors"] = len(pool)
        pool_ids = {s["id"] for s in pool}
        out["noisy"]["cited_distractor"] = any(c["id"] in pool_ids for c in out["noisy"].get("cited", []))

        # Context position: move every gold passage, as one block, to the front, middle or end.
        kept = select_evidence(case["question"], sources)
        golden = [s for s in kept if page_of(s) in gold]
        rest = [s for s in kept if page_of(s) not in gold]
        if golden and len(rest) >= 2:
            mid = len(rest) // 2
            orders = {
                "first": golden + rest,
                "middle": rest[:mid] + golden + rest[mid:],
                "last": rest + golden,
            }
            out["position"] = {
                name: run_answer(case, renumber(order), self.models, gold)
                for name, order in orders.items()
            }
        return out


# ---------------------------------------------------------------- summary


def summarize(rows, shape, k):
    scored = [r for r in rows if "embedding" in r]

    def ranks(name):
        return [r["embedding"]["first_gold_rank"][name] for r in scored]

    def hit(values):
        return rate(r is not None and r <= k for r in values)

    def mrr(values):
        return mean(1 / r if r is not None and r <= k else 0.0 for r in values)

    paraphrases = [p for r in scored for p in r["embedding"]["paraphrases"]]
    quotes = Counter(q["status"] for r in scored for q in r["chunking"]["quotes"])
    answerable = [r["generation"] for r in rows if "generation" in r and r["category"] != "unsupported"]
    unsupported = [r["generation"] for r in rows if "generation" in r and r["category"] == "unsupported"]
    bases = [g["base"] for g in answerable]
    supported = [b for b in bases if b["supported"]]
    noisy = [g for g in answerable if "noisy" in g]
    positioned = [g["position"] for g in answerable if "position" in g]
    contexts = [r["context"] for r in scored]
    return {
        "chunking": {
            **shape,
            "retrieved_fragments": mean(r["chunking"]["retrieved_fragments"] for r in scored),
            "quotes": {s: quotes[s] for s in ("retrieved", "not_retrieved", "split", "missing")},
        },
        "embedding": {
            **{f"hit_{n}": hit(ranks(n)) for n in ("fused", "dense", "lexical")},
            **{f"mrr_{n}": mrr(ranks(n)) for n in ("fused", "dense", "lexical")},
            "unmatched_questions": sum(bool(r["embedding"]["unmatched_terms"]) for r in scored),
            "gold_overlap": mean(r["embedding"]["gold_overlap"] for r in scored),
            "paraphrases": len(paraphrases),
            "paraphrase_hit_fused": hit(p["fused"] for p in paraphrases) if paraphrases else None,
            "paraphrase_hit_dense": hit(p["dense"] for p in paraphrases) if paraphrases else None,
        },
        "noise": {
            "precision": mean(r["noise"]["precision"] for r in scored),
            "off_target_rate": mean(
                r["noise"]["off_target"] / r["noise"]["passages"]
                for r in scored
                if r["noise"]["passages"]
            ),
            "distinct_papers": mean(r["noise"]["distinct_papers"] for r in scored),
            "rank1_gold": rate(r["noise"]["rank1_gold"] for r in scored),
            "supported_clean": rate(g["base"]["supported"] for g in noisy),
            "supported_noisy": rate(g["noisy"]["supported"] for g in noisy),
            "recall_clean": mean(g["base"].get("expected_recall") for g in noisy),
            "recall_noisy": mean(g["noisy"].get("expected_recall") for g in noisy),
            "cited_distractor": rate(g["noisy"]["cited_distractor"] for g in noisy),
        },
        "context": {
            "mean_dropped": mean(c["dropped"] for c in contexts),
            "questions_dropping": sum(c["dropped"] > 0 for c in contexts),
            "gold_lost": sum(c["gold_lost"] for c in contexts),
            "max_budget_used": max((c["budget_used"] for c in contexts), default=None),
            "max_est_prompt_tokens": max((c["est_prompt_tokens"] for c in contexts), default=None),
            "context_tokens": CONTEXT_TOKENS,
            **{
                f"recall_{name}": mean(p[name].get("expected_recall") for p in positioned)
                for name in ("first", "middle", "last")
            },
            **{
                f"cited_gold_{name}": rate(p[name]["cited_gold"] for p in positioned)
                for name in ("first", "middle", "last")
            },
        },
        "hallucination": {
            "answered": rate(b["supported"] for b in bases),
            "expected_recall": mean(b.get("expected_recall") for b in bases),
            "ungrounded_number_answers": sum(bool(b["ungrounded_numbers"]) for b in supported)
            if supported
            else None,
            "ungrounded_term_rate": mean(b["ungrounded_term_rate"] for b in supported),
            "abstained": rate(g["abstained"] for g in unsupported if "abstained" in g),
            "errors": sum("error" in b for b in bases) if bases else None,
        },
    }


def fmt(value, kind):
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{value:.0%}"
    if kind == "num":
        return f"{value:.3f}"
    if kind == "quotes":
        return "/".join(str(value[s]) for s in ("retrieved", "not_retrieved", "split", "missing"))
    return str(round(value) if isinstance(value, float) else value)


# (section, note, [(column, key, format)]); --generate columns read n/a otherwise.
TABLES = [
    (
        "chunking",
        "Index-wide shape, then per question. Quotes are retrieved/not retrieved/split/missing:"
        " split means no single chunk holds the quote; missing usually means the quote text"
        " does not match extraction.",
        [
            ("Chunks", "chunks", "int"),
            ("Median bytes", "median_bytes", "int"),
            ("Fragments", "fragment_rate", "pct"),
            ("Mid-sentence starts", "mid_sentence_rate", "pct"),
            ("Page-break cuts", "page_break_cut_rate", "pct"),
            ("Span a heading (approx)", "section_span_rate", "pct"),
            ("Fragments retrieved / q", "retrieved_fragments", "num"),
            ("Quotes", "quotes", "quotes"),
        ],
    ),
    (
        "embedding",
        "First-gold hit rate and MRR within top_k. Dense and BM25 run alone without the page-1"
        " prepend. A large dense vs BM25 gap, or low paraphrase hits, points to vocabulary mismatch.",
        [
            ("Hit fused", "hit_fused", "pct"),
            ("Hit dense", "hit_dense", "pct"),
            ("Hit BM25", "hit_lexical", "pct"),
            ("MRR fused", "mrr_fused", "num"),
            ("MRR dense", "mrr_dense", "num"),
            ("MRR BM25", "mrr_lexical", "num"),
            ("Qs with unmatched terms", "unmatched_questions", "int"),
            ("Query/gold term overlap", "gold_overlap", "pct"),
            ("Paraphrases", "paraphrases", "int"),
            ("Para hit fused", "paraphrase_hit_fused", "pct"),
            ("Para hit dense", "paraphrase_hit_dense", "pct"),
        ],
    ),
    (
        "noise",
        "Off-target means passages from papers with no gold page. The --generate columns compare"
        " answers before and after adding hard distractors from other papers.",
        [
            ("P@k", "precision", "num"),
            ("Off-target", "off_target_rate", "pct"),
            ("Papers / ranking", "distinct_papers", "num"),
            ("Rank 1 gold", "rank1_gold", "pct"),
            ("Answered clean", "supported_clean", "pct"),
            ("Answered noisy", "supported_noisy", "pct"),
            ("Recall clean", "recall_clean", "num"),
            ("Recall noisy", "recall_noisy", "num"),
            ("Cited a distractor", "cited_distractor", "pct"),
        ],
    ),
    (
        "context",
        "Drops are passages select_evidence silently discards over its byte budget; gold lost"
        " means every retrieved gold passage was discarded. Prompt tokens are an estimate."
        " The --generate columns move the gold passages to the front, middle or end.",
        [
            ("Dropped / q", "mean_dropped", "num"),
            ("Qs with drops", "questions_dropping", "int"),
            ("Gold lost", "gold_lost", "int"),
            ("Max budget used", "max_budget_used", "pct"),
            ("Max est. tokens", "max_est_prompt_tokens", "int"),
            ("Context tokens", "context_tokens", "int"),
            ("Recall first", "recall_first", "num"),
            ("Recall middle", "recall_middle", "num"),
            ("Recall last", "recall_last", "num"),
            ("Cites gold first", "cited_gold_first", "pct"),
            ("Cites gold middle", "cited_gold_middle", "pct"),
            ("Cites gold last", "cited_gold_last", "pct"),
        ],
    ),
    (
        "hallucination",
        "Requires --generate. Ungrounded numbers or words appear in an answer but not in its cited"
        " passages or the question; review those answers by hand in the JSON. Abstained is the"
        " share of unsupported questions the system declined.",
        [
            ("Answered", "answered", "pct"),
            ("Expected recall", "expected_recall", "num"),
            ("Answers w/ ungrounded numbers", "ungrounded_number_answers", "int"),
            ("Ungrounded word rate", "ungrounded_term_rate", "pct"),
            ("Abstained (unsupported)", "abstained", "pct"),
            ("Errors", "errors", "int"),
        ],
    ),
]


def write_summary():
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    runs.sort(key=lambda r: (r["config"] != "baseline", r["config"], r["generate"]))
    lines = [
        "# RAG diagnostics",
        "",
        "Generated by `evaluate.py`; do not edit by hand. Rows ending in `+gen` also ran"
        " the answer-generation checks.",
        "",
        "| Run | Changes | Questions | Errors | Run at |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        changes = ", ".join(f"`{k}={v}`" for k, v in run["changed_from_defaults"].items())
        lines.append(
            f"| {run['run']} | {changes or 'none'} | {run['questions']} | {run['errors']}"
            f" | {run['run_at']} |"
        )
    for section, note, columns in TABLES:
        lines += [
            "",
            f"## {section.capitalize()}",
            "",
            note,
            "",
            "| Run | " + " | ".join(label for label, _, _ in columns) + " |",
            "| --- |" + " --- |" * len(columns),
        ]
        for run in runs:
            values = run["summary"][section]
            cells = [fmt(values.get(key), kind) for _, key, kind in columns]
            lines.append(f"| {run['run']} | " + " | ".join(cells) + " |")
    SUMMARY.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- main


def diagnose_config(name, config, index, models, chunks, vocab, shape, args):
    questions = json.loads((HERE / config["questions"]).resolve().read_text())
    # Catalog questions do not touch retrieval; unsupported ones only matter for abstention.
    cases = [
        c
        for c in questions
        if (c.get("gold_sources") or (args.generate and c["category"] == "unsupported"))
        and (not args.only or c["id"] in args.only)
    ]
    diagnoser = Diagnoser(config, index, models, chunks, vocab, args.generate, args.distractors)
    rows = []
    for number, case in enumerate(cases, 1):
        row = {"id": case["id"], "category": case["category"], "question": case["question"]}
        try:
            row.update(diagnoser.run(case))
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - record the failure and keep going
            row["error"] = str(exc)
            status = "error: " + str(exc)
        rows.append(row)
        print(f"  [{number}/{len(cases)}] {case['id']}: {status}", flush=True)
    run = name + ("+gen" if args.generate else "")
    return {
        "config": name,
        "run": run,
        "generate": args.generate,
        "description": config["description"],
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {k: v for k, v in config.items() if k not in ("description", "changed")},
        "changed_from_defaults": config["changed"],
        "index": {
            k: index.manifest.get(k)
            for k in ("generation", "created_at", "embedding_model", "chunk_bytes", "overlap_bytes")
        },
        "questions": len(rows),
        "errors": sum("error" in r for r in rows),
        "summary": summarize(rows, shape, normalize_cutoffs(config["cutoffs"])[-1]),
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("configs", nargs="*", help="Config names (default: all)")
    parser.add_argument(
        "--generate", action="store_true", help="Also run answer-generation checks (slow)"
    )
    parser.add_argument(
        "--distractors", type=int, default=3, help="Hard distractors added in the noise check"
    )
    parser.add_argument(
        "--only",
        type=lambda value: set(value.split(",")),
        help="Comma-separated question IDs to run, e.g. paper-1,synthesis-2",
    )
    parser.add_argument("--config-file", type=Path, default=HERE / "configs.toml")
    parser.add_argument("--index-dir", type=Path, default=STARTER / "index")
    args = parser.parse_args()

    configs = load_configs(args.config_file)
    missing = set(args.configs) - set(configs)
    if missing:
        raise SystemExit(f"Unknown configs: {sorted(missing)}. Have: {sorted(configs)}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    models = LocalModels()
    with Index(args.index_dir.resolve()) as index:
        chunks = [source_record(d) for d in index.documents()]
        vocab = {t for c in chunks for t in lexical_tokens(c["text"])}
        shape = chunk_shape(chunks)
        for name in args.configs or configs:
            print(f"{name}: {configs[name]['description']}")
            result = diagnose_config(name, configs[name], index, models, chunks, vocab, shape, args)
            out = RESULTS / f"{result['run']}.json"
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    write_summary()
    print(f"Wrote {SUMMARY}")


if __name__ == "__main__":
    main()
