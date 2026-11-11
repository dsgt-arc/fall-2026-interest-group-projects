# Homework 1: CLEF RAG improvements (syang723)

## Summary

- **The provided pilot set cannot measure improvement.** The baseline already scores the maximum possible
  value on every retrieval metric for the 12 scored questions. I built a 47-question set across all 16
  tracks with exhaustively checked gold pages: [eval/questions-hard.json](eval/questions-hard.json).
- **One clear improvement: where the opening chunk goes.** The retriever always puts page 1 of the top
  paper at rank 1. Moving it to rank 2 (`RAG_INTRO=second`) raises hard-set MRR@6 from 0.434 to 0.764.
  Per question: 31 better, 0 worse (sign test p ≈ 1e-9). Recall@6 does not change on either set.
- **No effect or worse:** BM25/dense fusion weights near 0.5, candidate depth (10/20/50), and smaller
  chunks (1000 and 700 bytes). Dense-only and BM25-only retrieval were both clearly worse than hybrid.
- **Bug fixed:** with smaller chunks the original system failed on every query. Chroma loaded all chunks
  in one SQL query and hit SQLite's 32,766-variable limit once the index passed that many chunks.

## How I judged "better"

The generator reads all six retrieved passages, so the two retrieval-metric families answer different
questions:

- **Recall@6 / Hit@6:** does the evidence reach the LLM at all? This is the main RAG quality signal, because
  a page outside the six cannot support the answer.
- **MRR / nDCG / P@1:** is the evidence near the top? This matters when the answer's context budget trims
  passages, and for citation order. It is secondary.

A change was accepted only if:

1. it improved the hard set,
2. it did not lower Recall@6 on either set, and
3. a per-question paired comparison favoured it. With 47 questions, mean differences alone can be noise,
   so I report wins/losses and an exact two-sided sign test (`experiments/paired.py`).

I used two question sets because each is biased in a different direction:

- **Original pilot set:** gold is almost always page 1 (abstracts). It rewards the page-1 injection.
- **Hard set:** gold is mostly body pages. It penalises the page-1 injection.

Reporting both stops a change from looking good just because of which set it was measured on.

## Hard question set

See [eval/README.md](eval/README.md) for details.

- 2 sampled papers per track. No paper IDs or titles appear in the questions, so retrieval runs over the
  full corpus.
- Answers are specific facts from methods and results pages.
- Gold pages: every page in all 509 PDFs that matches a question's fact anchors was reviewed. Pages that
  only matched by coincidence were dropped.
- 6 questions have gold pages in several papers, e.g. a dataset size quoted by many teams.

## Code changes

The diff is in [changes.patch](changes.patch). It applies to `homework_1/clef-rag/src`. Defaults reproduce
the original system exactly: after the changes, the baseline produced identical rankings on 47/47 hard and
12/12 pilot questions.

| File | Change |
|---|---|
| `retrieve.py` | `RAG_INTRO` = `first` (original) / `second` / `off`: position of the leading paper's opening chunk when no paper is named. `RAG_CANDIDATES`: per-retriever candidate depth (default 20). `RAG_W_DENSE`: dense weight in the ensemble; BM25 gets the remainder (default 0.5). |
| `ingest.py` | `RAG_CHUNK_BYTES` / `RAG_OVERLAP_BYTES` (defaults 1400 / 250). Re-ingest into a separate `--index-dir`. |
| `index.py` | `documents()` reads Chroma in pages of 10,000 instead of one unbounded `get()`. This fixes the "too many SQL variables" failure for indexes above about 32.7k chunks. |

## Results (retrieval-only, `ir-measures`, macro averages)

### Hard set (47 questions)

| Config | P@1 | MRR@6 | R@3 | R@6 | Hit@6 | nDCG@6 | MAP@6 |
|---|---|---|---|---|---|---|---|
| baseline (intro first, w=0.5, cand 20, chunk 1400) | 0.043 | 0.434 | 0.679 | 0.823 | 0.915 | 0.537 | 0.403 |
| **intro = second** | **0.702** | 0.764 | 0.679 | **0.823** | **0.915** | 0.736 | 0.656 |
| intro = off | 0.702 | 0.793 | 0.740 | 0.827 | 0.936 | 0.766 | 0.695 |
| w_dense = 0.0 (BM25 only) | 0.064 | 0.413 | 0.644 | 0.778 | 0.894 | 0.503 | 0.375 |
| w_dense = 0.3 | 0.064 | 0.441 | 0.702 | 0.831 | 0.915 | 0.543 | 0.411 |
| w_dense = 0.7 | 0.064 | 0.444 | 0.695 | 0.793 | 0.894 | 0.535 | 0.413 |
| w_dense = 1.0 (dense only) | 0.043 | 0.392 | 0.632 | 0.726 | 0.851 | 0.469 | 0.347 |
| candidates = 10 | 0.043 | 0.432 | 0.679 | 0.801 | 0.915 | 0.524 | 0.393 |
| candidates = 50 | 0.043 | 0.431 | 0.679 | 0.802 | 0.894 | 0.529 | 0.399 |
| intro second + w 0.3 | 0.660 | 0.739 | 0.702 | 0.831 | 0.915 | 0.719 | 0.633 |
| chunk 1000 + intro second | 0.638 | 0.718 | 0.684 | 0.778 | 0.894 | 0.689 | 0.613 |
| chunk 700 + intro second | 0.596 | 0.686 | 0.647 | 0.779 | 0.894 | 0.660 | 0.571 |

### Original pilot set (12 scored questions)

| Config | P@1 | MRR@6 | R@6 | Hit@6 | nDCG@6 |
|---|---|---|---|---|---|
| baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **intro = second** | 0.833 | 0.917 | **1.000** | **1.000** | 0.938 |
| intro = off | 0.833 | 0.833 | 0.833 | 0.833 | 0.833 |
| w_dense = 0.0 | 0.750 | 0.778 | 0.833 | 0.833 | 0.792 |
| w_dense 0.3 / 0.7 / 1.0, candidates 10 / 50 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| chunk 1000 + intro second | 0.750 | 0.875 | 1.000 | 1.000 | 0.908 |
| chunk 700 + intro second | 0.833 | 0.917 | 1.000 | 1.000 | 0.938 |

### Paired comparisons (hard set)

| Comparison | Metric | Better | Worse | Tied | Sign test p |
|---|---|---|---|---|---|
| baseline → intro second | RR@6 | 31 | 0 | 16 | 9.3e-10 |
| baseline → intro off | RR@6 | 41 | 0 | 6 | 9.1e-13 |
| baseline → w_dense 0.3 | RR@6 | 3 | 4 | 40 | 1.0 |
| intro second: chunk 1400 → 1000 | Recall@6 | 1 | 7 | 39 | 0.07 |
| intro second: chunk 1400 → 700 | Recall@6 | 3 | 6 | 38 | 0.51 |

## Interpretation

1. **The page-1 injection costs rank 1 on body-page questions.** On the hard set, rank 1 was page 1 for
   47/47 questions, so the best-matching passage always sat at rank 2 or lower. Moving the opening chunk
   to rank 2 fixes this without losing any page from the top six.
2. **Removing the injection (`off`) scores best on the hard set but is rejected.** It drops the abstract
   page for two pilot questions (`paper-2`, `synthesis-4`), which then leave the top six entirely. The
   injection is useful; only its position was wrong.
3. **Fusion weight and candidate depth are flat near the default.** Both extremes lose recall: dense-only
   loses 0.10 Recall@6 on the hard set, and BM25-only misses two pilot questions. That confirms hybrid
   retrieval is worth keeping, but it does not justify a new weight.
4. **Smaller chunks hurt.** The LLM context is fixed at six passages, so smaller chunks mean less text per
   query (about 8.4 KB at 1400 bytes vs 4.2 KB at 700). Facts also get split from their context. Losses
   fall mostly on multi-page gold questions. 1400 bytes is close to the embedding input cap (1,900 bytes
   minus the title prefix), so there was little room to test larger chunks.
5. **What this does not show.** `intro = second` reorders the same six passages; Recall@6 is unchanged.
   Because the generator reads all six, its effect on answer quality is probably much smaller than the
   rank-metric gain suggests. I did not finish an answer-level evaluation (see below). None of the changes
   I tried improved Recall@6 by a meaningful amount.

## Limitations and not done

- **Answer-level evaluation:** generation runs with `gemma3:12b` were started and stopped because they
  made the machine unusable. A planned key-fact check (expected numbers and terms found in the answer)
  is the natural next step.
- **Chunk 1600 bytes:** started and stopped for the same reason.
- **Hard set construction:** one person wrote the questions after reading the gold pages, so wording may
  share terms with those pages and favour BM25. Gold pages come from pypdf text, so facts that exist only
  in figures are missed.
- **Statistical power:** with 47 questions, small differences (e.g. weight 0.3 vs 0.5) are
  indistinguishable from noise.

## Reproduce

```sh
cd homework_1/clef-rag
git apply ../../user/syang723/homework-1/changes.patch   # adds the env-var knobs and the paging fix
uv sync --locked && uv run clef-rag ingest

# retrieval sweep (each config runs on both the hard and pilot sets)
bash ../../user/syang723/homework-1/experiments/sweep.sh index \
  base= intro_second="RAG_INTRO=second" w03="RAG_W_DENSE=0.3" cand50="RAG_CANDIDATES=50"

# chunk-size experiment
RAG_CHUNK_BYTES=1000 RAG_OVERLAP_BYTES=180 uv run clef-rag ingest --index-dir index-chunk1000
bash ../../user/syang723/homework-1/experiments/sweep.sh index-chunk1000 c1000_second="RAG_INTRO=second"

# tables and paired tests
cd ../../user/syang723/homework-1
python3 experiments/summarize.py results/hard-*.json
python3 experiments/paired.py results/hard-base.json results/hard-intro_second.json reciprocal_rank 6
```

## Files

| Path | Contents |
|---|---|
| `changes.patch` | Source changes to `homework_1/clef-rag/src` |
| `eval/questions-hard.json` | 47-question hard set (same schema as the pilot set) |
| `eval/README.md`, `eval/tools/` | Set design and scripts that rebuild the gold pages |
| `experiments/sweep.sh` | Runs retrieval-only evaluation for env-var configs on both sets |
| `experiments/summarize.py`, `experiments/paired.py` | Metric tables; per-question paired comparison with sign test |
| `results/{hard,orig}-<config>.json` | Summary, per-question metrics and ranked pages for each config (passage text removed to keep files small) |
