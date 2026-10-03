# Candidate questions — verification queue

`candidates-2026-09-23.json` holds 64 draft questions for the replacement
evaluation set replacing the saturated pilot. They are
drafts. None is part of an evaluation set until verified against the PDF.

## How they were produced

`gemma3:12b` read one passage from each of 64 papers and wrote a question whose
answer lies in that passage. Passage selection was mechanical, not model-driven:

- Four papers per track, all 16 tracks (the pilot covers 2).
- Page 2 or later, sampled from the middle 50% of each paper — methods and
  results, not abstracts. **No gold page is page 1**, the page the retriever
  prepends unconditionally.
- The nine papers used by `questions.json` are excluded, so the new set and the
  regression suite stay independent.
- No `papers` field, so retrieval must find the paper among all 508.

Gold pages come from where the passage sits in the PDF, not from what the
retriever returned, so labels do not inherit the system's bias.

## Status of each entry

| `_verify.screen` | Count | Meaning |
| --- | --- | --- |
| `UNSCREENED` | 58 | passed the word-check; needs your verification |
| `rewritten` | 2 | auto-rewritten and re-checked; verify as normal |
| `NEEDS_MANUAL_REWRITE` | 4 | broken, automated repair failed; rewrite by hand |

## Verifying one

Open `_verify.url` — it deep-links to the exact PDF page — and check:

1. The passage in `_verify.source_passage` really appears on that page.
2. The question is answerable from it, and `expected_answer` is correct.
3. **No other paper could answer it.** This is the one a reader cannot check for
   you, and the most common defect: a question anchored on a public tool
   (`bge-small`, `BERT`, `Binoculars`) rather than something specific to the work.

Then set `_verify.status` to `VERIFIED`, or delete the entry. Strip `_verify`
before the file becomes an evaluation set.

## What the automated screening could and could not do

A mechanical word-check for context-dependent phrasing ("the authors",
"described", "the text") flagged 5 of 64 and is reliable.

An LLM judge was tried and **abandoned**. It rejected 64 of 64, and its rewrites
degraded good questions — one clear question about CHASTE became "According to
Table 15 to 18, ..." and four rewrites introduced the phrase "the passage",
which is what the screen exists to remove. Those rewrites were reverted.

A narrow rewrite of only the word-check failures repaired 2 of 6. The other 4
avoided the banned words while staying just as vague ("the workflow",
"contributors" for "the authors") and are marked `NEEDS_MANUAL_REWRITE`.

The lesson for the write-up: `gemma3:12b` drafts usable questions from a
passage, but cannot reliably judge whether a question is unambiguous across a
508-paper corpus. Verification is not optional.

## Still missing

These 64 are all `category: paper` — single-fact questions. The replacement set
also needs `synthesis` (multi-paper), and should carry forward `catalog` and
`unsupported` from `questions.json`, which test counting and abstention and do
not saturate.

---

# Synthesis candidates

`candidates-synthesis-2026-09-24.json` holds 16 multi-paper questions, one per
track, each requiring evidence from two papers. `gold_sources` lists both, so a
retriever must surface a page from each to score full recall.

Papers are drawn from the same track, excluding the pilot's nine and the 64 used
by the single-paper candidates, so all three sets stay independent.

The word-check flagged 1 of 16; it was repaired by hand without changing the
question's content.

## Known bias in this subset

These questions tend to quote paper titles verbatim, for example
`'DS@GT ARC at LongEval'`. That makes them unambiguous — the point of the
exercise — but it also converts part of the task into title matching, which
favours BM25 and is easier than a content question with no title in it.

This subset is therefore **not** a clean test of semantic retrieval. Treat
synthesis and paper questions as separate strata when reading results, and do
not average them into one headline number without saying so. Rewriting a few to
describe the systems without naming their titles would give a harder contrast,
and is worth doing by hand during verification.

## Combined coverage

| Set | Questions | Papers | Tracks | Gold pages |
| --- | --- | --- | --- | --- |
| `questions.json` (pilot) | 12 scored | 9 | 2 | all page 1 |
| `candidates-2026-09-23.json` | 64 | 64 | 16 | pages 3-17 |
| `candidates-synthesis-2026-09-24.json` | 16 | 32 | 16 | pages 2-9 |

Verified candidates, plus `catalog` and `unsupported` carried over from the
pilot, become `questions-hard.json` with a dev/test split.
