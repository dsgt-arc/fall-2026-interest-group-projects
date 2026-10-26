"""Build a harder retrieval test set from the corpus itself.

Why: the starter's 20-question pilot names its papers and labels mostly first pages, while the
retriever always adds the top paper's opening chunk. Every reasonable system scores ~1.0 on it,
so it cannot tell a better retriever from a worse one.

How: sample passages from page 2 onward across every track, ask the allowed gemma3:12b model to
write one question each passage answers without naming the paper, and keep the source page as the
gold judgment (a known-item search). Questions are split by hash into dev (for tuning) and test
(reported once), so improvements are not tuned on the numbers we report.

    uv run --project ../../../homework-1/clef-rag python make_testset.py --n 240
"""

import argparse
import hashlib
import random
import re
from collections import defaultdict

from pydantic import BaseModel

from common import TESTSET, open_index, save_json


class Generated(BaseModel):
    usable: bool
    question: str
    answer: str


SYSTEM = "You write evaluation questions for a search engine over CLEF 2026 working-notes papers."
PROMPT = """Passage from page {page} of a CLEF 2026 working-notes paper (CLEF lab: {track}):
<<<
{text}
>>>
Write ONE specific factual question that this passage answers and that a researcher could plausibly search for.
Rules:
- Do not mention the paper's title, team name, system name, author names, "this paper", "the passage", or "the authors".
- The passage alone must answer it. Prefer the passage's distinctive facts (methods, numbers, datasets, findings).
- Paraphrase; do not copy long phrases from the passage. Naming the CLEF lab or task is fine.
- If the passage is mostly references, acknowledgments, a bare table of numbers, or boilerplate, set usable to false.
Also give the short answer."""

REFERENCE_LIKE = re.compile(r"\[\d+\]|et al\.|doi|https?://|arXiv|Proceedings", re.I)


def body(text):
    return text.split("\n", 1)[1] if "\n" in text else text


def candidate(meta, text):
    b = body(text)
    digits = sum(c.isdigit() for c in b) / max(1, len(b))
    return (
        meta["page"] >= 2
        and len(b) >= 700
        and len(REFERENCE_LIKE.findall(b)) < 4
        and digits < 0.25
    )


def leaks(question, title):
    q = question.casefold()
    if re.search(r"\b(this|the) (paper|passage|study|work)\b|\bthe authors\b", q):
        return True
    words = re.findall(r"\w+", title.casefold())
    team = title.split(" at ")[0].casefold() if " at " in title else None
    if team and len(team) >= 3 and team in q:
        return True
    grams = {" ".join(words[i : i + 4]) for i in range(len(words) - 3)}
    return any(g in q for g in grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240, help="passages to try")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    index, models = open_index()
    data = index.collection.get(include=["documents", "metadatas"])
    by_track = defaultdict(list)
    for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"], strict=True):
        if meta.get("document_id", "").startswith("paper") and candidate(meta, text):
            by_track[meta.get("track") or "unknown"].append((cid, text, meta))

    # Stratify across tracks (round robin, one passage per paper at most) so no lab dominates.
    rng = random.Random(args.seed)
    pools = {t: rng.sample(v, len(v)) for t, v in sorted(by_track.items())}
    picked, used_papers = [], set()
    while len(picked) < args.n and any(pools.values()):
        for track in list(pools):
            while pools[track]:
                cid, text, meta = pools[track].pop()
                if meta["document_id"] not in used_papers:
                    used_papers.add(meta["document_id"])
                    picked.append((cid, text, meta))
                    break
            if len(picked) >= args.n:
                break
        if all(not p for p in pools.values()):
            break

    cases, skipped = [], defaultdict(int)
    for i, (cid, text, meta) in enumerate(picked, 1):
        try:
            g = models.chat(
                SYSTEM,
                PROMPT.format(page=meta["page"], track=meta["track"], text=body(text)),
                schema=Generated,
                max_tokens=300,
            )
        except Exception:  # noqa: BLE001 - a failed generation just skips the passage
            skipped["error"] += 1
            continue
        q = " ".join(g.question.split())
        if not g.usable:
            skipped["unusable"] += 1
        elif not 6 <= len(q.split()) <= 45:
            skipped["length"] += 1
        elif leaks(q, meta["title"]):
            skipped["names the paper"] += 1
        else:
            split = "dev" if int(hashlib.sha256(cid.encode()).hexdigest(), 16) % 2 == 0 else "test"
            cases.append({
                "id": f"syn-{len(cases) + 1:03d}",
                "split": split,
                "category": "paper",
                "question": q,
                "expected_answer": g.answer,
                "track_of_source": meta["track"],
                "source_chunk": cid,
                "gold_sources": [{"paper": meta["document_id"], "pages": [meta["page"]]}],
            })
        print(f"[{i}/{len(picked)}] kept {len(cases)}", flush=True)

    save_json(TESTSET, cases)
    tracks = defaultdict(int)
    for c in cases:
        tracks[c["track_of_source"]] += 1
    print(f"saved {len(cases)} questions to {TESTSET}")
    print("dev/test:", sum(c["split"] == "dev" for c in cases), "/", sum(c["split"] == "test" for c in cases))
    print("skipped:", dict(skipped))
    print("by track:", dict(sorted(tracks.items())))


if __name__ == "__main__":
    main()
