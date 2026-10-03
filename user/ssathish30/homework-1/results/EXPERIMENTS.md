# CLEF RAG Experiments

## Method: validating the evaluation before trusting it

Before any comparison is reported, the question set is checked for whether it can
detect a difference at all. Four variants run against it:

| Variant | Setting | Expectation if the set is valid |
| --- | --- | --- |
| Dense only | `--weights 1 0` | below hybrid |
| Lexical only | `--weights 0 1` | below hybrid |
| Hybrid | `--weights 0.5 0.5` | best |
| **Shuffled control** | `--shuffle-seed N` | near floor |

The shuffled control is decisive: a system whose ranking has been destroyed. If it
scores level with hybrid, the question set cannot measure retrieval and no result
from it means anything. On the original 12-question pilot it scored 0.9167 against
1.0000 — an 8-point penalty for having no ranking at all. On `questions-hard.json`
it scores 0.1091 against 0.4344.

Report this table alongside any tuning result, as evidence the instrument can see.


Append one row per evaluation run. Each row's raw report is stored beside this
file as `<name>-<date>.json` with a `provenance` block recording the git SHA,
model digests, and corpus counts. Numbers are only comparable when the
embedding model and passage count match; record both.

Reproduce a row with:

```sh
uv run clef-rag evaluate --retrieval-only --k 1 3 6 --output index/retrieval-metrics.json
```

## Retrieval metrics

Macro averages over the annotated questions in `eval/questions.json`.

| Run | Date | SHA | Embed model | Passages | k | P | R | Hit | MRR | MAP | nDCG |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 2026-09-20 | 8cab67e | embeddinggemma:300m | 21046 | 1 | 1.0000 | 0.9167 | 1.0000 | 1.0000 | 0.9167 | 1.0000 |
| baseline | 2026-09-20 | 8cab67e | embeddinggemma:300m | 21046 | 3 | 0.3889 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| baseline | 2026-09-20 | 8cab67e | embeddinggemma:300m | 21046 | 6 | 0.1944 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Notes

### baseline (2026-09-20)

Reproduces the reference run in `eval/RESULTS.md` exactly: 21,046 passages, and
model digests `f4031aab637d` / `85462619ee72` matching that document. Corpus is
508 of 511 listed papers plus the preface; `paper236`, `paper318`, and
`paper345` return non-PDF responses upstream and are unavailable.

**This row cannot be improved on.** Recall, Hit Rate, MRR, MAP, and nDCG are all
1.0000 at k=3 and k=6, so any change can at best tie it. The precision column
falls with k only because unannotated pages count as nonrelevant against the
one or two gold pages per question; it is not a quality signal.

Before measuring any retrieval change, replace the question set with one that
has headroom: more questions, weighted toward multi-paper and cross-track
queries where six passages are genuinely insufficient, annotated beyond the
first page of each paper. Record that as a new `baseline-hard` row and treat it,
not this row, as the comparison point.

## Evaluation-validity ablation (2026-09-23)

Raw report: `ablation-2026-09-23.json`. Protocol: see Method above.

**Question asked:** can `eval/questions.json` tell working retrieval from
destroyed retrieval? **Answer: no.**

### A. Baseline configuration (page-1 prepend active, question filters applied)

| Variant | nDCG@6 | MRR@6 | Recall@6 | MAP@6 | P@1 |
| --- | --- | --- | --- | --- | --- |
| Hybrid 0.5/0.5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Dense only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Lexical only | 0.7917 | 0.7778 | 0.8333 | 0.7778 | 0.7500 |
| **Shuffled control** | **0.9167** | **0.9167** | **0.9167** | **0.9167** | **0.9167** |

A randomly shuffled ranking loses only 8 points against the real system, and
hybrid is indistinguishable from dense-only. The set cannot measure ranking.

### B. Page-1 prepend disabled (metrics score the fused ranking)

| Variant | nDCG@6 | MRR@6 | Recall@6 | MAP@6 | P@1 |
| --- | --- | --- | --- | --- | --- |
| **Lexical only** | **0.4588** | 0.4306 | 0.6250 | 0.3889 | 0.2500 |
| Hybrid 0.5/0.5 | 0.2494 | 0.1944 | 0.4167 | 0.1944 | 0.0833 |
| Dense only | 0.2436 | 0.1875 | 0.4167 | 0.1875 | 0.0833 |
| Shuffled control | 0.2092 | 0.1653 | 0.3750 | 0.1514 | 0.0833 |

### C. Question filters dropped (retrieval searches all 508 papers)

| Variant | nDCG@6 | MRR@6 | Recall@6 | MAP@6 | P@1 |
| --- | --- | --- | --- | --- | --- |
| Hybrid, prepend on | 0.6250 | 0.6111 | 0.6667 | 0.6111 | 0.5833 |
| Shuffled, prepend on | 0.4489 | 0.4333 | 0.5000 | 0.4333 | 0.4167 |
| Hybrid, prepend off | 0.2040 | 0.1625 | 0.3333 | 0.1625 | 0.0833 |

### Where the published 1.0000 comes from

| Condition | nDCG@6 |
| --- | --- |
| Baseline as published | 1.0000 |
| Remove page-1 prepend | 0.2494 |
| Remove question filters | 0.6250 |
| Remove both | 0.2040 |

The prepend supplies roughly three quarters of the score. Honest open-corpus
retrieval on these questions is 0.2040, not 1.0000.

### Secondary finding: the 50/50 default is not a good setting

Once the prepend stops fixing position 1, lexical-only (0.4588) is nearly twice
hybrid (0.2494), and hybrid is barely above the shuffled control (0.2092). The
balanced fusion dilutes a strong BM25 signal with a weak dense one. This is
consistent with the corpus: CLEF 2026 coins terms (`GutBrainIE`,
`MultiClinSum-2`, `ELCardioCC`) that `embeddinggemma:300m` has never seen and
BM25 matches exactly.

The page-1 annotations should if anything favour dense retrieval, since
abstracts are prose; BM25 winning regardless strengthens the reading.

**This is a hypothesis, not a result.** n=12, all gold pages are page 1, and 11
of 12 questions pre-filter the corpus. Confirming it requires the question set
in `eval/CANDIDATES.md` and a paired significance test.

## Instrument validation on questions-hard.json (2026-09-24)

Raw report: `ablation-hard-2026-09-24.json`. Protocol: see Method above.
Question set: `eval/questions-hard.json`, 84 questions, 76 scored. No question carries a
`papers` filter and no gold page is page 1, so retrieval searches all 508 papers and the
opening-context prepend is never automatically correct.

### The control now separates

| Question set | Hybrid nDCG@6 | Shuffled control | Gap |
| --- | --- | --- | --- |
| `questions.json` (pilot, 12 scored) | 1.0000 | 0.9167 | 0.08 |
| `questions-hard.json` (76 scored) | 0.4344 | **0.1091** | **0.33** |

Destroying the ranking cost 8 points on the pilot and 33 points here. The set can
detect a difference in retrieval quality; the pilot could not.

### Prepend active (product behaviour)

| Variant | nDCG@6 | MRR@6 | Recall@6 | MAP@6 | Hit@6 |
| --- | --- | --- | --- | --- | --- |
| Lexical only | **0.4804** | 0.3811 | 0.8092 | 0.3573 | 0.8684 |
| Hybrid 0.5/0.5 | 0.4344 | 0.3421 | 0.7434 | 0.3189 | 0.8026 |
| Dense only | 0.3385 | 0.2768 | 0.5592 | 0.2521 | 0.6184 |
| Shuffled control | 0.1091 | 0.0781 | 0.2105 | 0.0723 | 0.2368 |

### Prepend disabled (scores the fused ranking)

| Variant | nDCG@6 | MRR@6 | Recall@6 | MAP@6 | Hit@6 |
| --- | --- | --- | --- | --- | --- |
| Lexical only | **0.7104** | 0.7026 | 0.8224 | 0.6575 | 0.8947 |
| Hybrid 0.5/0.5 | 0.6382 | 0.6305 | 0.7500 | 0.5850 | 0.8158 |
| Shuffled control | 0.1480 | 0.1265 | 0.2237 | 0.1194 | 0.2500 |

### Two candidate improvements, not yet established

1. **Lexical-weighted fusion.** Lexical-only beats hybrid 0.5/0.5 in every configuration
   (0.4804 vs 0.4344; 0.7104 vs 0.6382), and dense-only is worst throughout. On the pilot
   hybrid and dense-only were indistinguishable at 1.0000, so this was invisible.
2. **The prepend now costs points.** P@1 is 0.0000 in every prepend-active arm: rank 1
   always holds a page-1 passage, and no gold page is page 1. Disabling it raises hybrid
   from 0.4344 to 0.6382.

Both are single-run differences with no significance test. Confirming them requires a
dev/test split and a paired test over per-question scores, which this report stores.

### Strata differ sharply

Mean nDCG@6 for hybrid: `paper` 0.4872 (n=60), `synthesis` 0.2365 (n=16). Multi-paper
questions are much harder even though they quote paper titles. Report the strata
separately rather than averaging them.

### Caveat

Candidates are UNVERIFIED apart from the mechanical page-location check, which 79 of 80
passed. Label noise affects all arms alike, so it largely cancels in the paired
comparisons above; it would matter more for absolute scores than for ranking systems.

## Phase 5: weight and prepend sweep with paired testing (2026-09-25)

Raw report: `phase5-2026-09-25.json`. Test script: `eval/tools/paired_test.py`.

`questions-hard.json` was split by category into dev (55 questions, 49 scored) and
test (29, 27 scored). Dev chose the challenger; test was evaluated once. The primary
metric, nDCG@6, and the single primary comparison were fixed before any test-set run.

Significance is a two-sided permutation test over per-question deltas, exact for
n<=20 non-zero differences and sampled at 100,000 permutations otherwise. It was
validated on two controls: a run against itself returns p=1.0 with zero differences,
and hybrid against the shuffled control returns +0.3253 at p<0.0001.

### Dev sweep, nDCG@6 (49 scored)

| weights (semantic / lexical) | prepend ON | prepend OFF |
| --- | --- | --- |
| 1.0 / 0.0 | 0.3414 | 0.5087 |
| 0.7 / 0.3 | 0.3428 | 0.5400 |
| 0.5 / 0.5 (default) | 0.4194 | 0.6165 |
| 0.3 / 0.7 | 0.4542 | 0.6633 |
| 0.0 / 1.0 | **0.4776** | **0.7054** |

Monotonic in both columns: more lexical weight scores higher at every step, and the
prepend costs points at every weight setting. Dev selected `0.0 / 1.0` with the
prepend disabled.

### Held-out test, nDCG@6 (27 scored)

| Comparison | Baseline | Challenger | Delta | W/L/T | p |
| --- | --- | --- | --- | --- | --- |
| **Primary** — combined vs default | 0.4616 | **0.7194** | +0.2579 | 19/4/4 | **0.0001** |
| Secondary A — weights only | 0.4616 | 0.4854 | +0.0239 | 9/5/13 | 0.4559 |
| Secondary B — prepend only | 0.4616 | **0.6776** | +0.2160 | **23/0/4** | **<0.0001** |

Holm across the two secondaries: B survives, A does not.

### What is established

**Disabling the opening-context prepend improves retrieval by +0.216 nDCG@6**, winning
23 of 27 questions and losing none. The prepend places page 1 of the leading paper at
rank 1, outside the ensemble; since no gold page in this set is page 1, that slot is
always wasted. P@1 is 0.0000 in every prepend-active arm.

### What is not established

**Lexical-weighted fusion.** The direction is positive on both dev and test and the dev
trend is monotonic across five settings, but the held-out effect is +0.0239 at p=0.4559.
Unconfirmed, not disproven. Two reasons to keep it open: n=27 is underpowered for an
effect this small, and Secondary A varied weights with the prepend still enabled, which
wastes rank 1 regardless and compresses the difference. Dev suggests a larger gap with
the prepend off (0.6165 to 0.7054). Testing that would need a fresh held-out split,
since this one is spent.

### Note for the write-up

The prepend is what made the pilot score 1.0000 — every gold page there was page 1, the
page it injects. The same mechanism is the largest measured drag on retrieval quality
here. It read as a strength only because the answer key was built around it.
