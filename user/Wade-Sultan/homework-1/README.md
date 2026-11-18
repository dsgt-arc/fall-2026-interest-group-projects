# Homework 1 Wade Sultan

## Results

### The starter questions cannot show an improvement

On the starter's 12 annotated questions, the unchanged system already scores Hit@1 = 1.00,
MRR = 1.00 and Recall@3 = 1.00 ([results/summary.md](results/summary.md)). Since there is no
room to improve on that set, I wrote 20 harder questions of my own
([my-questions.md](my-questions.md)): 8 about details past page 1, 5 worded differently from
the paper, 4 comparisons across papers and 3 the papers cannot answer. The 17 with gold pages
are scored below; the 3 unanswerable ones have no gold pages, so retrieval metrics skip them.

On my questions the baseline finds a gold page in its top 6 passages for only 8 of 17 questions.

### Retrieval on my questions

Macro averages over the 17 scored questions, from
[results/my-questions/summary.md](results/my-questions/summary.md). Hit is shown as a count of
questions because each question moves it by 0.059. Bold marks the best value in each column.

| Config | Hit@3 | Recall@3 | nDCG@3 | Hit@6 | Recall@6 | MRR@6 | MAP@6 | nDCG@6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 5/17 | 0.191 | 0.153 | **8/17** | 0.353 | 0.234 | 0.143 | 0.218 |
| no-intro | 5/17 | 0.191 | 0.172 | **8/17** | 0.353 | 0.240 | **0.184** | **0.244** |
| expand | 6/17 | 0.250 | 0.190 | **8/17** | 0.338 | **0.252** | 0.155 | 0.224 |
| multi-query | 6/17 | 0.245 | 0.190 | 7/17 | 0.275 | 0.240 | 0.140 | 0.199 |
| hyde | 3/17 | 0.088 | 0.100 | 7/17 | 0.324 | 0.192 | 0.112 | 0.182 |
| doc2query | 5/17 | 0.191 | 0.153 | 7/17 | 0.309 | 0.218 | 0.128 | 0.192 |
| rerank | 6/17 | 0.250 | 0.184 | **8/17** | **0.368** | 0.220 | 0.159 | 0.227 |
| expand-multi-query | 6/17 | 0.230 | 0.176 | 7/17 | 0.260 | 0.240 | 0.130 | 0.187 |
| expand-hyde | 4/17 | 0.132 | 0.124 | 7/17 | 0.309 | 0.210 | 0.120 | 0.185 |
| expand-doc2query | 6/17 | 0.250 | 0.190 | 7/17 | 0.309 | 0.237 | 0.147 | 0.208 |
| expand-doc2query-rerank | **7/17** | **0.324** | **0.233** | 7/17 | 0.324 | 0.245 | 0.165 | 0.226 |

Questions with a gold page in the top 6, by question type:

| Config | Detail (8) | Wording (5) | Comparison (4) | Total (17) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 3 | 1 | 4 | 8 |
| no-intro | 3 | 1 | 4 | 8 |
| expand | 3 | 1 | 4 | 8 |
| multi-query | 2 | 1 | 4 | 7 |
| hyde | 4 | 0 | 3 | 7 |
| doc2query | 3 | 1 | 3 | 7 |
| rerank | 3 | 2 | 3 | 8 |
| expand-multi-query | 2 | 1 | 4 | 7 |
| expand-hyde | 4 | 0 | 3 | 7 |
| expand-doc2query | 3 | 1 | 3 | 7 |
| expand-doc2query-rerank | 3 | 1 | 3 | 7 |

What the numbers show:

- **No change finds more questions in the top 6.** Hit@6 never rises above the baseline's 8/17.
  The gains are in ordering: some strategies move a gold page that was already in the top 6
  into the top 3. Because the answer step reads all six passages, Hit@6 and Recall@6 are the
  closest measure of whether the model is given the answer at all.
- **Best top 3: expand-doc2query-rerank.** It has a gold page in the top 3 for 7/17 questions
  (baseline 5/17) and raises Recall@3 from 0.191 to 0.324. It pays for this at depth 6, where it
  loses one question (comparison-4) that the baseline found.
- **expand is the only strategy with no per-question loss.** Compared with the baseline, it
  changes one question (detail-2 moves into the top 3) and leaves every other hit unchanged.
  It has the best MRR@6 (0.252).
- **Turning off the page-1 prepend helps MAP and nDCG.** no-intro finds the same 8 questions but
  ranks them better (MAP@6 0.184 vs 0.143). The prepend forces the leading paper's first passage
  to rank 1, but only 3 of my 17 questions have a gold page on page 1. This also explains why
  Hit@1 is 2/17 for every config with the prepend on: rank 1 is always a page-1 passage, and it
  is gold only for comparison-2 and comparison-3. (rerank drops this to 1/17.)
- **HyDE hurts the most.** Hit@3 falls to 3/17, and neither HyDE config finds any of the five
  differently worded questions. It is the only strategy that finds detail-6, but loses
  comparison-4 and wording-5.
- **doc2query adds nothing on its own.** It changes no hit except to lose comparison-4. Its
  generated questions come from each paper's opening page, while 14 of the 17 questions have
  their gold pages only past page 1.
- **multi-query and rerank trade questions rather than add them.** multi-query is the only
  strategy that finds wording-4, but loses detail-4 and wording-5. rerank is the only one that
  finds detail-5 and wording-1, but loses detail-4 and comparison-4.
- **Five questions are never found** by any config: detail-1, detail-7, detail-8, wording-2 and
  wording-3. Across all eleven configs, 12 of 17 questions are found by at least one, so the
  strategies are complementary, but none of them dominates the baseline.

Caveats: with 17 questions, most differences in the table are one or two questions, which is
too few to call any of them a reliable improvement. The gold pages are my own annotations,
usually one page per paper, so a passage that answers the question from a different page
counts as a miss. Mean time per query ranged from 2.56 s (expand) to 2.97 s (rerank), but the
gemma3 outputs were mostly cached from an earlier run, so these timings understate the cost of
the LLM strategies on a cold run.

### Answers on my questions: baseline vs expand

To check whether the retrieval differences reach the answers, I ran the answer-generation
checks on all 20 of my questions for the baseline and for `expand`, my chosen config
(`./evaluate.sh baseline expand --questions my-questions.json --generate`; full output in
[results/diagnostics/my-questions/summary.md](results/diagnostics/my-questions/summary.md)).

| Measure | baseline | expand |
| --- | ---: | ---: |
| Answerable questions answered (of 17) | 10 | 11 |
| Mean recall of my expected answer | 0.345 | 0.408 |
| Answers citing a gold page | 4 | 5 |
| Unsupported questions declined (of 3) | 3 | 3 |
| Answers with a number not in the cited passages | 0 | 0 |
| Ungrounded word rate | 8% | 9% |
| Expected-answer recall with 3 distractors added | 0.301 | 0.322 |
| Answers citing a distractor | 24% | 18% |

The two configs give word-for-word identical answers on 15 of the 17 answerable questions.
Almost the whole gain comes from one question, detail-2. The other difference is wording-2,
where `expand` adds one sentence but both answers miss the point (see below). wording-3 hit the
model's output limit in both runs and produced no answer, so the recall average covers 16
questions.

Three examples, with the baseline and `expand` answers side by side:

**detail-2: "How many text samples are in the Kaggle Academic out-of-distribution dataset?"**
(expected: 6,069, paper416 page 3)

- baseline: "I don't have enough evidence in the available corpus to answer this question."
- expand: "The Kaggle Academic dataset contains 6,069 text samples. [S2]" (cites paper416, page 3)

This is the one question where expansion changed retrieval enough to matter. The baseline had
the gold passage at rank 5 and still declined; with expansion, paper416 becomes the leading
paper, its gold page moves to rank 2, and the model answers correctly.

**comparison-3: "How do the DS@GT and zhengqiaozeng systems for Voight-Kampff AI detection
differ?"** (expected: DS@GT LoRA-fine-tunes Qwen2.5-1.5B; zhengqiaozeng fine-tunes DeBERTa-v3-large)

- baseline: "I don't have enough evidence in the available corpus to answer this question."
- expand: the same refusal.

Retrieval scores this question as a hit at rank 1, because a gold page of paper434
(zhengqiaozeng) comes first. But five of the six passages are from paper434; the baseline has
only page 1 of paper397 (DS@GT) at rank 6, and `expand` has nothing from paper397. With one side
of the comparison missing, the model declines. comparison-2 fails the same way (all six passages
come from paper477, none from paper476). Retrieval metrics count both as successes, which they
are not for a comparison question.

**unsupported-1: "How much did the gfreitas team on ImageCLEF spend in AI API costs during
evaluation?"** (expected: not stated)

- baseline: "I don't have enough evidence in the available corpus to answer this question."
- expand: the same refusal.

Both configs declined all three unanswerable questions, so expansion did not make the system
more willing to answer from loosely related passages.

Other things the answers show that retrieval metrics miss:

- **Correct answers from pages I did not mark.** wording-4 is answered correctly ("EXIST")
  from paper157, not my gold paper152, so it counts as a retrieval miss.
- **Confident wrong answers.** detail-1 answers "twenty-two generators" from a different PAN
  paper (paper422) instead of the 23 in paper397. Each claim is grounded in its citation, so the
  hallucination check does not flag it.
- **wording-2 is wrong in both configs.** Both answers say merging three queries beat a single
  query; the gold page reports mixed results. `expand` repeats the claim in an extra sentence.
- **Comparison questions are the weakest spot.** Besides detail-2, both configs decline the
  same 5 answerable questions. Two are details whose gold page is never retrieved (detail-5,
  detail-7). The other three are comparison-2, -3 and -4: in each, all or nearly all six passages
  come from one paper (comparison-4 retrieves only paper390, never paper492), so the model never
  sees the second paper it is asked to compare.

### Diagnostics on the starter questions (baseline only)

[results/diagnostics.md](results/diagnostics.md) runs the baseline through five failure-mode
checks on the starter questions (12 for retrieval, 16 with the answer-generation checks).

- **Embedding:** the fused ranking finds a gold page for 100% of questions, but dense retrieval
  alone finds 42% and BM25 alone 67%. On average only 60% of a question's terms appear on its
  gold page, and 5 of 12 questions contain a term no passage in the corpus contains (often a
  paper ID like `paper6`, which BM25 cannot match). This vocabulary gap is what the strategies
  above target.
- **Chunking:** 26% of chunks are cut at a page break and about 30% span a section heading;
  5% are short fragments.
- **Context:** no passages are dropped by the byte budget (the largest prompt used 99% of it).
  Moving the gold passages around shows a middle-position dip: answer recall is 0.718 with gold
  first, 0.577 in the middle and 0.762 last.
- **Noise:** adding hard distractors from other papers did not lower answer recall (0.741 clean,
  0.750 noisy), but 25% of noisy answers cited a distractor.
- **Hallucination:** no answer contained a number missing from its cited passages, 4% of answer
  words were ungrounded, and the system declined every unsupported question.

## Evaluation

The question this section answers: how do I know whether a change made the system better?

### What I measured, and why

An answer can only be as good as the passages it is given, so I judged changes mainly by
retrieval, scored with ir-measures on unique paper/PDF-page pairs against gold pages I
annotated. Each metric answers a different question:

- **Hit@6 and Recall@6: is the answer in front of the model at all?** The answer step reads
  the six retrieved passages, so a gold page outside the top 6 can never be used. This is the
  metric I weighted most.
- **Hit@3, MRR and nDCG: is the answer near the top?** Order matters even inside the top 6.
  When the diagnostics move the gold passages within the context, answer recall changes a lot:
  on the starter questions it is worst with gold in the middle (0.577 vs 0.718 first), and on my
  questions it is worst with gold last (0.406 vs 0.662 first, baseline). detail-2 shows the same
  thing on a single question: the baseline declined with the gold page at rank 5, and answered
  correctly once `expand` moved it to rank 2.
- **MAP: are all the gold pages ranked well?** This matters for the comparison questions,
  which have several gold pages across papers.
- **Precision is not informative here.** Most questions have one or two gold pages, so P@6 has
  a ceiling of 1/6 or 2/6 even for a perfect ranking.

Averages hide trades on a set this small, so I also compared configs question by question
(which questions each config gains and loses against the baseline). I count a config as better
only if it finds at least as many questions in the top 6 and does not lose questions the
baseline found.

### Why I wrote my own questions

A test set only shows an improvement if the baseline can fail on it. The starter's 12
questions are already at the ceiling (Hit@1 = 1.00, Recall@3 = 1.00), so they can only show
whether a change makes things worse. Each section of [my-questions.md](my-questions.md) targets
a specific weakness:

| Category | Count | What it tests |
| --- | ---: | --- |
| Detail | 8 | Facts past page 1, with the paper not named, so the page-1 prepend cannot help |
| Wording | 5 | Synonyms and plain language in place of the paper's terms (the vocabulary gap the strategies target) |
| Comparison | 4 | Answers that need pages from more than one paper |
| Unsupported | 3 | Questions the papers cannot answer; the system should decline |

The baseline finds 8 of the 17 scored questions in its top 6, so this set leaves room to show
a gain.

### Failure-mode diagnostics

Retrieval metrics say whether a ranking got worse, not why. [evaluate.py](evaluate.py) checks
the five common RAG failure modes (chunking, embedding mismatch, retrieval noise, context
overflow, hallucination), and with `--generate` it also tests the answers themselves:
recall of the expected answer, numbers and words not found in the cited passages, whether hard
distractors get cited, and whether the system declines unsupported questions. On the starter
questions, the dense-vs-BM25 gap pointed to vocabulary mismatch, which is why the strategies I
built target the query side.

I also ran the answer checks on my own questions for the baseline and my chosen config, and read
the answers side by side. This caught things retrieval metrics cannot: correct answers from
pages I did not mark, confident answers from the wrong paper, and comparison questions that
count as retrieval hits but fail because only one of the papers was retrieved.

### Conclusion

By the criteria above, none of the changes is clearly better than the starter system. No
config finds more of my questions in the top 6, and every difference in the Results tables is
one or two questions out of 17.

If I had to pick one change, it would be `expand` (title and acronym expansion): it is the
only strategy that loses no question the baseline found, it moves one gold page into the top
3, and it has the best MRR@6. It also costs no extra LLM calls. `expand-doc2query-rerank` is
better if only the top 3 matters (7/17 vs 5/17), but it loses one question at depth 6.

The answer checks agree with that choice but do not strengthen it much. `expand` answers 11 of
17 answerable questions against the baseline's 10, raises expected-answer recall from 0.345 to
0.408, cites distractors less often (18% vs 24%), and still declines all three unanswerable
questions. But its answers are identical to the baseline's on 15 of 17 questions, and nearly
all of the gain is one question (detail-2).

The clearest problem the evaluation found is one none of my strategies addresses: comparison
questions retrieve passages from only one of the papers they compare, so the model declines.
The starter already adds opening context for each paper a question names by ID, but my
questions name teams, not paper IDs. Resolving team names to papers and retrieving a few
passages from each would be the next thing to try.

### Limits of this evaluation

- **Small set.** 17 scored questions means one question moves Hit by 0.059. A real claim of
  improvement would need many more questions.
- **One annotator.** I chose the gold pages, usually one page per paper. A passage that answers
  the question from a page I did not mark counts as a miss.
- **Retrieval is not answer quality, and the answer checks are crude.** Expected-answer recall
  is word overlap with my expected answers, which misses correct paraphrases and rewards
  partial matches. For example, detail-6's answer restates the question without naming the
  metric (MRR@5) and still scores 0.5. I read the answers by hand to catch cases like this,
  but only for two configs.
- **Model-dependent.** The LLM strategies depend on gemma3:12b's outputs. They are cached, so
  reruns are repeatable, but a different sample could rank differently.

## How to run

Requires the starter setup in `homework-1/clef-rag` (models pulled, `clef-rag ingest` done).

```sh
./run.sh              # every config in configs.toml
./run.sh baseline     # just the named configs
```

Experiments are defined in `configs.toml`: `[defaults]` is the starter system, and
each `[configs.<name>]` table overrides only the settings it changes. Each run writes
`results/<name>.json` and regenerates the comparison tables in `results/summary.md`.

To check the same configs against five common RAG failure modes (chunking, embedding
mismatch, retrieval noise, context overflow, hallucination):

```sh
./evaluate.sh                               # retrieval checks, every config (about a minute each)
./evaluate.sh baseline --generate           # also answer-generation checks (slow, uses the chat model)
./evaluate.sh baseline --generate --only paper-4,unsupported-2
./evaluate.sh baseline expand --questions my-questions.json --generate   # my questions
```

With `--questions`, results go to `results/diagnostics/<file stem>/` (with its own
`summary.md`) so they do not mix with the starter-question runs.

Results go to `results/diagnostics/` and `results/diagnostics.md`. Questions can carry two
optional fields: `gold_quotes` (exact answer passages, to test chunking) and `paraphrases`
(rewordings, to test embedding robustness).

### Embedding-mismatch strategies

`strategies.py` adds five strategies as retriever settings, so each is one config in
`configs.toml` and both scripts above measure it:

| Setting | Strategy |
| --- | --- |
| `expand_query = true` | Append titles for `paperN` mentions; expand acronyms with a glossary mined from the corpus |
| `query_mode = "multi_query"` | Search the question plus `rewrites` gemma3 rewordings, fused with RRF |
| `query_mode = "hyde"` | Dense search on a gemma3-written hypothetical answer (HyDE) |
| `doc2query_weight > 0` | Also match questions gemma3 generated from each paper's opening page |
| `rerank_depth > 0` | gemma3 reorders the top fused candidates |

doc2query needs a one-time build (one gemma3 call per paper, about 30 minutes on an RTX 3090;
it resumes if interrupted):

```sh
./doc2query.sh            # all papers; ./doc2query.sh --limit 5 for a quick test
```

gemma3 outputs are cached under `cache/` (not committed), so reruns on the same machine are fast
and return the same results.
With every strategy off, the retriever reproduces the starter's rankings exactly.

## Part 2: fashion-fm

I trained the starter's fashion-fm model (text-conditioned flow matching on FashionMNIST)
without code changes. [fashion-fm/train.yml](fashion-fm/train.yml) is the starter config with
two changes: `epochs` goes from 1 to 20, and `data_dir` points at the starter's downloaded
dataset.

On an RTX 3090, 20 epochs took 4.4 minutes (about 13 seconds each).

| Epoch | Loss | Gradient norm |
| ---: | ---: | ---: |
| 1 | 0.529 | 0.79 |
| 2 | 0.311 | 0.41 |
| 5 | 0.254 | 0.40 |
| 10 | 0.237 | 0.36 |
| 20 | 0.226 | 0.26 |

Most of the loss drop happens in the first two epochs, and the curve is nearly flat after
epoch 10, so more epochs would likely add little. The gradient norm falls steadily with no
spikes.

![Training metrics](fashion-fm/outputs/fashion-fm/training_metrics.png)

Four images for each of the prompts "a plain t-shirt", "a sneaker", "small handbag" and
"ankle boot", one row per prompt. All four prompts produce recognizable items:

![Generated images](fashion-fm/outputs/generated.png)

`fashion-fm/outputs/fashion-fm/` also has a ten-class sample grid for every epoch
(`samples-0001.png` to `samples-0020.png`) and the per-epoch numbers in `train_metrics.csv`.
Model checkpoints are not committed.

To reproduce, set up the starter's environment (`uv sync` in `homework-1/fashion-fm`), then
from `fashion-fm/` in this folder:

```sh
uv run --project ../../../../homework-1/fashion-fm fashion-fm train --config train.yml
uv run --project ../../../../homework-1/fashion-fm fashion-fm generate \
  --prompt "a plain t-shirt" --prompt "a sneaker" --prompt "small handbag" --prompt "ankle boot"
```

Training resumes from `outputs/fashion-fm/latest.pt` if it exists, so delete `outputs/` to
start over.
