# Homework 1 — CLEF RAG

**Sarvesh Sathish (ssathish30)**

The headline result is not a tuning gain. It is that **the evaluation shipped with
the project could not measure retrieval at all**, and that the mechanism which made
it score a perfect 1.0000 is also the largest measured drag on real retrieval
quality.

---

## 1. The provided evaluation was saturated

`homework-1/clef-rag/eval/questions.json` reports Recall, Hit Rate, MRR, MAP and
nDCG of exactly `1.0000` at k=3 and k=6. A perfect score and a broken measurement
produce the same number, so I tested which one it was by **destroying the ranking**
— replacing the fused order with a random shuffle — and re-running.

| Question set | Hybrid nDCG@6 | Randomly shuffled | Gap |
| --- | --- | --- | --- |
| `questions.json` (12 scored) | 1.0000 | **0.9167** | 0.08 |

Destroying retrieval entirely cost 8 points. Three structural causes, all
verifiable from the repository:

1. **The test told the retriever which paper to open.** `evaluate.py` passes each
   question's `papers` field as a hard filter. 9 of 12 questions name their own
   paper, 2 name a track; **1 of 12 searches all 508 papers**.
2. **Every gold page is page 1.** All 14 annotations, without exception.
3. **The code force-prepends page 1.** `retrieve.py` inserts the opening passage of
   the leading paper at rank 1, *outside* the ensemble, before anything the ranker
   chose.

Filter to one paper → prepend its page 1 → answer key says page 1 → perfect score.
The ranker's output starts at rank 2 and is never graded.

Decomposing where the 1.0000 comes from:

| Condition | nDCG@6 |
| --- | --- |
| As published | 1.0000 |
| Remove the prepend | 0.2494 |
| Remove the question filters | 0.6250 |
| **Remove both** | **0.2040** |

Honest open-corpus retrieval on those questions is **0.2040**. The rest was
scaffolding.

This is not a criticism of the authors — `eval/README.md` states plainly that high
scores "do not establish general search quality" and calls the set a development
set, not a benchmark. The defect is only exposed when it is used as a comparison
instrument, which is what this assignment requires.

---

## 2. A replacement evaluation set

`eval/questions-hard.json` — 84 questions, 76 scored.

| | pilot | this set |
| --- | --- | --- |
| Scored questions | 12 | 76 |
| Tracks covered | 2 of 16 | **16 of 16** |
| Distinct papers | 9 | **96** |
| Questions naming their own paper | 9 | **0** |
| Gold pages on page 1 | 14 of 14 | **0** |

Questions were drafted by `gemma3:12b` from passages chosen **mechanically**: four
papers per track, page 2 or later, sampled from the middle half of each paper, so
gold pages land in methods and results rather than abstracts. Gold pages come from
where the passage sits in the PDF, never from retrieval output, so labels do not
inherit the bias of the system under test. The pilot's `catalog` and `unsupported`
questions are carried over unchanged — they test counting and abstention, which do
not saturate.

**Validation.** The same shuffled-ranker control, re-run against the new set:

| Question set | Hybrid | Shuffled control | Gap |
| --- | --- | --- | --- |
| `questions.json` | 1.0000 | 0.9167 | 0.08 |
| **`questions-hard.json`** | 0.4344 | **0.1091** | **0.33** |

The instrument can now detect a difference in retrieval quality. That gap is the
evidence every later result rests on.

---

## 3. Diagnosis before optimisation

Rather than sweep parameters blind, I measured **where** quality is lost.

**Recall at increasing depth** (all 76 questions, prepend disabled):

| Depth | Gold page found |
| --- | --- |
| 6 | 81.6% |
| 20 | 92.1% |
| **50** | **97.4%** |

The correct page is almost always retrieved and then ranked too low to survive the
cut to six. **This is a ranking problem, not a finding problem** — which is why I
did *not* pursue embedding-dimension tuning: shrinking vectors tests capacity to
find, and finding is not the bottleneck.

**Failure analysis** (14 questions scoring zero):

| Cause | Count |
| --- | --- |
| Right paper, wrong page | 8 |
| Vague question (my own label defect) | 4 |
| Wrong paper entirely | 2 |

Only 2 of 76 are genuine retrieval misses. Paper-level accuracy is **92.1%**.

---

## 4. Results

Dev (49 scored) selected each challenger; the held-out test split (27 scored) was
evaluated once. Primary metric **nDCG@6**, fixed before any test-set run.
Significance is a two-sided permutation test over per-question deltas
(`eval/tools/paired_test.py`), validated on two controls: a run against itself
returns p=1.0 with zero differences, and hybrid vs the shuffled control returns
+0.3253 at p<0.0001.

| Change | Δ nDCG@6 | W/L/T | p | Verdict |
| --- | --- | --- | --- | --- |
| **Disable the page-1 prepend** | **+0.2160** | **23/0/4** | **<0.0001** | **confirmed** |
| Lexical-weighted fusion | +0.0239 | 9/5/13 | 0.4559 | not established |
| LLM reranking (depth 20) | +0.0149 | 6/5/16 | 0.7803 | not established |
| More candidates (10→200) | — | — | — | **actively harmful** |

### What is established

**Disabling the opening-context prepend improves retrieval by +0.216 nDCG@6**,
winning 23 of 27 held-out questions and losing none. `P@1` is `0.0000` in every
prepend-active arm: rank 1 always holds a page-1 passage, and no gold page in this
set is page 1, so the slot is always wasted.

The irony is the finding: **the prepend is what made the pilot score 1.0000, and is
simultaneously the largest drag on real retrieval.** It read as a strength only
because the answer key was built around its behaviour.

It should not simply be deleted. `eval/RESULTS.md` records that it was added after
an answer misattributed a related-work method to the paper's own system — page 1
carries the title and abstract that prevent that. The correct fix is to give the
generator that context **without** charging it a graded retrieval slot.

### What is not established

**Lexical weighting.** The dev sweep is monotonic across five settings — more
lexical weight scores higher at every step — and the direction is positive on test,
but +0.0239 at p=0.4559 does not clear significance at n=27.

I attempted to rescue this with a targeted hypothesis: keyword matching should help
specifically on terms coined in these 2026 papers (`GutBrainIE`, `MedCascade`,
`SYZYGY`) that the embedding model was trained too early to represent. Reading 16
examples produced a confident mechanism. **Measuring it over all 76 refuted it** —
the keyword advantage was *largest* for questions whose rarest token is common, the
opposite of the prediction. The effect is broad and modest, not concentrated.

**Reranking.** Scoring a 20-passage pool with `gemma3:12b` and keeping the best 6
gained +0.077 on dev and +0.015 on test. The test split's baseline is higher (0.710
vs 0.614) — it drew easier questions — leaving less headroom. Cost is ~18s/question
against ~1s, a 20× latency increase.

**More candidates hurts.** Recall@6 falls from 0.73 at 10 candidates to 0.66 at 200.
Rank fusion scores by position: a passage ranked 40th by *both* retrievers
(`0.5/100 + 0.5/100 = 0.0100`) outranks one ranked 1st by a single retriever
(`0.5/61 = 0.0082`). A deeper pool lets mediocre-but-agreed-upon passages
accumulate score through mere presence. Depth without judgement is harmful; that is
precisely what reranking was meant to fix.

---

## 5. How to run

Code changes are in `src/` (four `clef_rag` modules) and `changes.patch` applies
them to the shared starter code:

```bash
cd homework-1/clef-rag
git apply ../../user/ssathish30/homework-1/changes.patch
uv sync --locked && python download_papers.py && uv run clef-rag ingest
```

New flags added to `search` and `evaluate`:

```
--weights SEM LEX      rank-fusion weights (was hardcoded 0.5 0.5)
--rrf-c C              fusion constant (was hardcoded 60)
--candidates N         candidates per retriever (was hardcoded 20)
--top-k N              passages returned (evaluate had this hardcoded at 6)
--no-introductions     skip the page-1 prepend
--shuffle-seed N       randomise the ranking — the negative control
--rerank-depth N       retrieve N, rerank with the chat model down to --top-k
--ignore-question-filters   search all 508 papers, ignoring each question's hints
```

Reproduce the headline result:

```bash
# instrument validation — the shuffled control must collapse
uv run clef-rag evaluate --retrieval-only --questions eval/questions-hard.json \
  --no-introductions --output /tmp/hybrid.json
uv run clef-rag evaluate --retrieval-only --questions eval/questions-hard.json \
  --no-introductions --shuffle-seed 42 --output /tmp/shuffled.json

# the confirmed improvement, on held-out questions
uv run clef-rag evaluate --retrieval-only --questions eval/questions-hard-test.json \
  --candidates 20 --output /tmp/with-prepend.json
uv run clef-rag evaluate --retrieval-only --questions eval/questions-hard-test.json \
  --candidates 20 --no-introductions --output /tmp/without-prepend.json
uv run python eval/tools/paired_test.py /tmp/with-prepend.json /tmp/without-prepend.json
```

---

## 6. Limitations

- **Questions are machine-drafted and not content-verified.** A mechanical check
  re-extracted each cited PDF page and confirmed the stored passage appears there —
  **195 of 198 passed** — but nobody has confirmed every answer is correct or that
  no other paper could answer each question. Four failures above are traceable to
  this. Label noise affects all arms equally, so it largely cancels in paired
  comparisons; it would matter far more for absolute scores.
- **n=27 held out is underpowered.** Detecting effects near +0.03 needs roughly
  150–200 questions. Both non-results above are consistent with real small effects
  that this sample cannot resolve. `eval/candidates-batch2-2026-09-25.json` holds
  95 further screened candidates toward that.
- **The test split was used twice** (prepend, then reranking). A third question
  needs fresh held-out questions.
- **Answer quality is unmeasured.** Everything here evaluates retrieval. The
  generation half — whether answers are factually supported by their citations —
  has no metric, as the evaluation report itself states.
- **Synthesis questions quote paper titles**, which makes them unambiguous but
  converts part of the task into title matching and favours BM25. They are a
  separate stratum (mean nDCG@6 0.2365, vs 0.4872 for single-paper questions) and
  should not be averaged into one headline number.

---

## 7. Files

```
README.md          this document
changes.patch      source changes, applies to homework-1/clef-rag
src/               the four modified clef_rag modules
eval/              question sets, candidate drafts, the paired-test tool
results/           every run, with provenance; EXPERIMENTS.md narrates them
fashion-fm/        a secondary result, see fashion-fm/EXPERIMENTS.md
```

`results/EXPERIMENTS.md` is the full experiment log: every run carries the git SHA,
model digests, passage count and exact command, so two runs can be checked for
comparability before their numbers are compared.

`fashion-fm/` covers a separate finding: the committed training config
(`epochs: 1`) reproduced no released checkpoint — the shipped model came from epoch
5, step 295. Training to 60 epochs raised condition accuracy from 0.9490 to 0.9880
over 1000 samples per arm, scored by an identical judge classifier, with the
hardest class (shirt) improving 0.670 → 0.920.
