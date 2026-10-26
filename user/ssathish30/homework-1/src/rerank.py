"""Second-stage reranking of retrieved passages with the local chat model.

Recall is far higher at depth 50 than at depth 6 (97.4% against 81.6% on
questions-hard), so the gold passage is usually retrieved and then ranked too
low to survive the cut. This stage reorders a deep candidate pool before it is
truncated, which is where that gap can be recovered.
"""

import json

from clef_rag.config import RagError

SYSTEM = (
    "You judge whether a passage from a research paper answers a question. "
    "You reply only with the requested structured fields."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 3},
    },
    "required": ["score"],
}

PROMPT = """Question: {question}

Passage (from "{title}", PDF page {page}):
{text}

Rate how well this passage answers the question:
3 = contains the answer outright
2 = directly about the question's subject, likely supporting
1 = same paper or topic, does not answer it
0 = unrelated

Reply with the score only."""


def rerank(models, question, sources, *, keep, batch_log=None):
    """Score each passage against the question and return the best ``keep``.

    Ties keep the first-stage order, so reranking can only reorder what
    retrieval already found. Scoring failures fall back to the original rank.
    """

    if keep < 1:
        raise ValueError("keep must be at least 1")
    if len(sources) <= keep:
        return sources

    scored = []
    for position, source in enumerate(sources):
        try:
            verdict = models.chat(
                SYSTEM,
                PROMPT.format(
                    question=question,
                    title=source.get("title", "")[:150],
                    page=source.get("page"),
                    text=source.get("text", "")[:1200],
                ),
                schema=SCHEMA,
                max_tokens=50,
            )
            score = int(verdict.get("score", 0))
        except Exception:
            # A passage the model refuses to score keeps its retrieval position
            # rather than being dropped.
            score = -1
        scored.append((score, -position, source))
        if batch_log:
            batch_log(position, score)

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    kept = []
    for rank, (_, _, source) in enumerate(scored[:keep], start=1):
        item = dict(source)
        item["source_id"] = f"S{rank}"
        item["rank"] = rank
        kept.append(item)
    return kept
