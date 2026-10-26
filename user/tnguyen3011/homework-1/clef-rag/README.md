# RAG Retrieval Improvements — Homework 1

Two experiments on the hybrid retriever in `retrieve.py`, both run on 2026-09-20 with the `clef-rag evaluate` CLI: (1) the semantic/lexical weighting of the `EnsembleRetriever`, and (2) retrieval depth (`top_k`).

## Summary

| Change | Result | Kept? |
|---|---|---|
| Ensemble weights (dense / BM25) | Any non-zero dense weight gives identical rankings; BM25-only is clearly worse | No change (0.5 / 0.5) |
| `top_k` 6 → 3 | ~33% faster generation, no loss on pilot metrics, **but drops papers from multi-paper comparison answers** | Yes, as a tradeoff (see Conclusions) |

RRF fusion (`c=60`), index, chunking, and embedding model were unchanged throughout.

## How I evaluated

- **Pilot set (20 questions):** 12 content questions (scored for retrieval), 4 catalog questions, and 4 unsupported questions that should be abstained on.
- **Retrieval-only:** `uv run clef-rag evaluate --retrieval-only --k 1 3 6` reports precision, recall, hit rate, MRR, MAP, and nDCG.
- **Full pipeline:** `uv run clef-rag evaluate` adds LLM generation, catalog checks, abstention checks, citation review, and latency.
- **Qualitative check:** three hand-written ImageCLEF questions outside the pilot set, run at both `top_k` values.

## Experiment 1: Retriever weighting

Four weightings (dense / lexical) were compared with retrieval-only evaluation on the 12 scored questions (0 errors).

| Config | P@1 | R@1 | R@3 | MRR@3 | nDCG@3 |
|---|---|---|---|---|---|
| 0.5/0.5 (baseline), 0.7/0.3, 1.0/0.0 — **identical** | 1.000 | 0.917 | 1.000 | 1.000 | 1.000 |
| 0.0/1.0 (lexical-only) | 0.750 | 0.667 | 0.833 | 0.778 | 0.792 |

Results at k=6 match k=3 except for precision (0.194 vs 0.167 for lexical-only). Full per-k tables are in `results/`.

**Finding:** every config with a non-zero dense weight produced the same retrieved pages, confirmed by diffing per-question page lists. Under RRF the dense ranking dominates whenever it contributes. Dropping it (BM25 only) cut precision@1 and MRR from 1.00 to 0.75, which fits the pilot questions being mostly conceptual ("which tasks focus on VLM models") rather than exact-phrase lookups. The lexical component added no measurable benefit, so I kept the baseline.

## Experiment 2: Retrieval depth (`top_k`)

`top_k` was lowered from 6 to 3 in `Retriever.search()`, the single call site used by `ask`, `evaluate`, and `search`. Full evaluation ran on all 20 pilot questions (0 errors); each content question received exactly 3 sources instead of 6.

**Pilot set: no quality change, big speedup.** Retrieval precision/recall @1 and @3, catalog (4/4), abstention (4/4), and citation review (12/12) were identical at both depths.

| | top_k=6 | top_k=3 | Change |
|---|---|---|---|
| Mean generation latency (12 content Qs) | 57.9s | 38.5s | -33% |
| Median generation latency | 54.1s | 38.9s | -28% |
| Total time, 20 questions | 791.5s | 509.5s | -36% |

**Qualitative check on questions outside the pilot set:**

| Question type | top_k=6 | top_k=3 | Verdict |
|---|---|---|---|
| **Single-paper:** "What does the ImageCLEF MultimodalReasoning task evaluate, and how is it different from standard visual question answering?" | Cited `paper235.pdf` p.1, p.2, and p.2 again (`[S2]` and `[S4]` are the same page, likely from overlapping chunks) | Two distinct sources, no duplicate | Better: nothing lost, redundancy removed |
| **Multi-paper comparison:** "Which vision-language models did different teams use for the ImageCLEF Multimodal Reasoning task, and how did their approaches differ?" | Three papers: task overview (`paper235`), SRH-ReliableAI (`paper280`), UNED-Martinez (`paper273`); compared three teams' methods | Only `paper235`; both team papers dropped; no cross-team comparison | **Regression** |
| **Unanswerable:** "What was the exact winning accuracy score in the Deepfake Detection task?" | Abstained | Abstained | No change |

The comparison regression contradicts the project's own design note that six passages are selected so comparisons can draw on multiple papers. The abstention result shows less context did not make the model more likely to fabricate an answer.

## Conclusions and next steps

- **Weighting:** the embedding model does essentially all the retrieval work on this set; keep 0.5/0.5.
- **`top_k=3` is a tradeoff, not a strict improvement.** It is better for single-paper, definitional, and catalog questions (faster, less duplicate context) and worse for multi-paper comparisons, where it silently drops whole papers.
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
