# RAG Retrieval Improvements — Homework 1

Two experiments on the hybrid retriever in `retrieve.py`, both run on 2026-09-20 with the `clef-rag evaluate` CLI: (1) the semantic/lexical weighting of the `EnsembleRetriever`, and (2) retrieval depth (`top_k`).

## Summary

| Change | Result | Kept? |
|---|---|---|
| Ensemble weights (dense / BM25) | The tested non-zero dense weights give identical pilot scores; BM25-only scores lower | No change (0.5 / 0.5) |
| `top_k` 6 → 3 | ~33% lower mean elapsed query time, unchanged pilot scores, **but drops papers from multi-paper comparison answers** | Yes, as a tradeoff (see Conclusions) |

RRF fusion (`c=60`), index, chunking, and embedding model were unchanged throughout.

## How I evaluated

- **Pilot set (20 questions):** 12 content questions (scored for retrieval), 4 catalog questions, and 4 unsupported questions that should be abstained on.
- **Retrieval-only:** `uv run clef-rag evaluate --retrieval-only --k 1 3 6` reports precision, recall, hit rate, MRR, MAP, and nDCG.
- **Full pipeline:** `uv run clef-rag evaluate` adds LLM generation, catalog checks, abstention checks, automatic citation/support checks, and elapsed query time. Manual factual review remains separate.
- **Qualitative check:** three hand-written ImageCLEF questions outside the pilot set, run at both `top_k` values.

## Experiment 1: Retriever weighting

Four weightings (dense / lexical) were compared with retrieval-only evaluation on the 12 scored questions (0 errors).

| Config | P@1 | R@1 | R@3 | MRR@3 | nDCG@3 |
|---|---|---|---|---|---|
| 0.5/0.5 (baseline), 0.7/0.3, 1.0/0.0 — **identical** | 1.000 | 0.917 | 1.000 | 1.000 | 1.000 |
| 0.0/1.0 (lexical-only) | 0.750 | 0.667 | 0.833 | 0.778 | 0.792 |

Results at k=6 match k=3 except for precision (0.194 vs 0.167 for lexical-only). Full per-k tables are in `results/`.

**Finding (corrected during maintainer integration):** the three tested configurations with non-zero dense weight have the same pilot metric scores, but different retrieved page lists. Compared with baseline, the dense-heavy lists differ on 10 of 12 scored questions and the dense-only lists differ on all 12. For `paper-1`, baseline pages are `[1, 3, 2, 14]`, dense-heavy pages are `[1, 3, 2, 14, 11]`, and dense-only pages are `[1, 10, 12, 3, 2]`. All 14 gold references are page 1, so these scores do not establish that dense retrieval dominates RRF or that the lexical component adds no useful evidence. BM25-only scores lower on this pilot (P@1 and MRR@1 0.75); the baseline weighting is retained without a broader superiority claim.

## Experiment 2: Retrieval depth (`top_k`)

`top_k` was lowered from 6 to 3 in the experiment's `Retriever.search()` default, used by `ask` and `evaluate`. The starter's `search` command supplies its own `--top-k` value; the integration wrapper below makes its default consistent with the selected experiment. Full evaluation ran on all 20 pilot questions (0 errors); each content question received exactly 3 sources instead of 6.

**Pilot scores unchanged, lower elapsed time.** Retrieval precision/recall @1 and @3, catalog (4/4), abstention (4/4), and automatic support checks (12/12) were identical at both depths. These checks do not establish equal answer quality.

| | top_k=6 | top_k=3 | Change |
|---|---|---|---|
| Mean elapsed query time (12 scored content Qs) | 57.9s | 38.5s | -33% |
| Median elapsed query time | 54.1s | 38.9s | -28% |
| Total time, 20 questions | 791.5s | 509.5s | -36% |

**Qualitative check on questions outside the pilot set:**

| Question type | top_k=6 | top_k=3 | Verdict |
|---|---|---|---|
| **Single-paper:** "What does the ImageCLEF MultimodalReasoning task evaluate, and how is it different from standard visual question answering?" | Cited `paper235.pdf` p.1, p.2, and p.2 again (`[S2]` and `[S4]` are the same page, likely from overlapping chunks) | Two distinct sources, no duplicate | Better: nothing lost, redundancy removed |
| **Multi-paper comparison:** "Which vision-language models did different teams use for the ImageCLEF Multimodal Reasoning task, and how did their approaches differ?" | Three papers: task overview (`paper235`), SRH-ReliableAI (`paper280`), UNED-Martinez (`paper273`); compared three teams' methods | Only `paper235`; both team papers dropped; no cross-team comparison | **Regression** |
| **Unanswerable:** "What was the exact winning accuracy score in the Deepfake Detection task?" | Abstained | Abstained | No change |

The comparison regression contradicts the project's own design note that six passages are selected so comparisons can draw on multiple papers. The abstention result shows less context did not make the model more likely to fabricate an answer.

## Conclusions and next steps

- **Weighting:** this pilot does not distinguish the metric scores of the three tested configurations with non-zero dense weight; keep 0.5/0.5 without claiming identical rankings.
- **`top_k=3` is a tradeoff, not a strict improvement.** The recorded run is faster and the qualitative single-paper example contains less duplicate context, while the multi-paper example loses whole papers. Catalog queries bypass passage retrieval, so this experiment does not establish a catalog-quality improvement.
- **Recommended next step:** choose `top_k` by question type, for example scaling it with the number of named or compared papers and keeping ~6 for comparison and synthesis questions. The current `retrieve.py` still uses a fixed default of 3.

## Caveats

- The pilot set is small (12 scored retrieval questions, 20 total). Its synthesis questions compare at most two papers, and its metrics score retrieval and citation grounding, not whether an answer covers every compared paper. It therefore **cannot detect** the comparison regression above.
- Only one comparison question was tried by hand, so how often `top_k=3` fails on multi-paper questions is unknown.
- Latency comes from one run per config on one machine; treat the percentages as indicative.
- Only `top_k` 3 and 6 were tested, not 4 or 5.
- Experiment 1 compared retrieval metrics only; answer quality was not re-run per weighting. A more lexically heavy question set (exact author names, version strings) might show BM25 contributing more.
- Citation checks confirm claims are grounded in cited passages, not that they are factually correct; answers were not re-verified against the PDFs.

## Reproducing and raw results

Weights were varied via `weights=[...]` in `Retriever._rank()`; `top_k` via the default in `Retriever.search()`. Raw outputs are in [`results/`](results/):

| File | Config |
|---|---|
| `retrieval-metrics.json` | Baseline (0.5/0.5, top_k=6) |
| `retrieval-metrics-v2.json` | Dense-heavy (0.7/0.3) |
| `retrieval-metrics-dense.json` | Dense-only (1.0/0.0) |
| `retrieval-metrics-lexical.json` | Lexical-only (0.0/1.0) |
| `retrieval-metrics-topk3.json` | top_k=3, retrieval-only |
| `evaluation.json` | Baseline, full pipeline |
| `evaluation-topk3.json` | top_k=3, full pipeline |

## Maintainer integration — 2026-10-26

The original result JSON files and submitted `retrieve.py` are retained. The
corrections above distinguish scores from ranked pages, automatic checks from
manual review, and elapsed query time from model-only generation time. They are
checks of committed evidence, not new model runs. The three qualitative examples
remain the student's reported observations; they were not independently rerun.

Run the experiment from the repository root with the current shared starter at
`homework_1/clef-rag`. The wrapper creates and removes a temporary source copy,
changes only the dense/BM25 weights and retrieval default there, and leaves
shared source untouched. It preserves the student's existing intro-prepend and
other starter behavior. Data and index defaults point to the shared starter's
local `data/` and `index/`; the corpus, index, environment, and archive are
never copied into the temporary project.

Install `uv` and Ollama, download the models, and prepare the corpus/index using
the shared starter README before running model experiments. Model calls,
dependency installation, and data ingestion occur only when the corresponding
commands below are actually run. The setup-only check uses Python's standard
library and performs none of those operations:

```bash
python3 user/tnguyen3011/homework-1/clef-rag/run_experiment.py --dry-run -- evaluate --retrieval-only
```

To run the original settings, choose `--dense-weight` from `0.5`, `0.7`,
`1.0`, and `0.0`; lexical weight is `1 - dense_weight`. Use retrieval depth
six for the weighting comparison and choose three for the depth experiment:

```bash
# Baseline: 0.5 dense / 0.5 lexical, six passages.
python3 user/tnguyen3011/homework-1/clef-rag/run_experiment.py --dense-weight 0.5 --top-k 6 -- evaluate --retrieval-only --k 1 3 6 --output user/tnguyen3011/homework-1/clef-rag/runs/baseline-new.json

# Dense-heavy comparison; repeat with 1.0 or 0.0 for the other weightings.
python3 user/tnguyen3011/homework-1/clef-rag/run_experiment.py --dense-weight 0.7 --top-k 6 -- evaluate --retrieval-only --k 1 3 6 --output user/tnguyen3011/homework-1/clef-rag/runs/dense-heavy-new.json

# Depth experiment: three passages and metric cutoffs no larger than three.
python3 user/tnguyen3011/homework-1/clef-rag/run_experiment.py --dense-weight 0.5 --top-k 3 -- evaluate --retrieval-only --k 1 3 --output user/tnguyen3011/homework-1/clef-rag/runs/topk3-new.json

# Full pipeline; repeat with --top-k 6 for its baseline.
python3 user/tnguyen3011/homework-1/clef-rag/run_experiment.py --top-k 3 -- evaluate --output user/tnguyen3011/homework-1/clef-rag/runs/full-topk3-new.json
```

Arguments after `--` are forwarded to `clef-rag`; path arguments are resolved
from the caller's working directory. For example, add `--index-dir /path/to/index`
after `evaluate` to reuse another local index. `search` receives the selected
depth unless an explicit `--top-k` is supplied after `search`. The `--k`
evaluation option changes metric cutoffs, not retrieval depth. New outputs go
under ignored `runs/` paths so they cannot overwrite the submitted evidence.
The wrapper prints its effective settings and invocation to stderr for run
provenance. Its source-structure checks fail if the starter no longer exposes
the expected retriever configuration. Integration was checked against starter
revision `ecd1ad678e2011475050ec5d20e4fe70e14dce38`; matching historical scores is
not guaranteed across regenerated corpora, indexes, model versions, or hardware.
