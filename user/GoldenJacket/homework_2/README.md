# Homework 2: Zachary Wallace (GoldenJacket)

## Part 1

I ran every cell of the starter's `teaching.ipynb` top to bottom with the default settings (`TRAIN_MODELS = True`, 50 epochs for the custom model, 20 for LoRA) on my Mac (M4 Pro, MPS). The executed notebook is `teaching.ipynb` in this folder, and I left the starter code under `homework_2/` untouched.

### Results (162 validation claims, full 5,183 abstract corpus)

| Method | nDCG@10 | Recall@10 | MRR@10 |
|---|---|---|---|
| BM25 | 0.731 | 0.849 | 0.696 |
| TF-IDF | 0.631 | 0.824 | 0.574 |
| Custom model | 0.574 | 0.629 | 0.567 |
| MiniLM | 0.642 | 0.751 | 0.609 |
| MiniLM + LoRA | 0.664 | 0.810 | 0.624 |
| EmbeddingGemma | **0.905** | **0.988** | **0.880** |

The custom model peaked at epoch 36 of 50 and LoRA peaked at epoch 10 of 20; both kept those checkpoints. `comparison.csv` and `comparison.png` hold the same numbers, and `splits.json` is the train/validation split the notebook generated (seed 42), so the run can be repeated on the same queries.

### Responses

**1. Which method would I choose, does LoRA help, and what would change my choice?**

*For finding one good result,* I would pick EmbeddingGemma. Its MRR@10 of 0.880 means the first relevant abstract is usually at rank 1, and it beats the next best method (BM25, 0.696) by 0.18. If I could not run a neural model at all, I would pick BM25, not TF-IDF or MiniLM; plain word matching outranked every model we trained or adapted here.

*For recovering all known evidence,* EmbeddingGemma again. Its Recall@10 is 0.988, so it found nearly every judged abstract in the top 10. BM25 is second at 0.849. This matches what I saw in Homework 1, where keyword search did most of the work on the CLEF papers. Scientific claims are full of exact gene, drug, and protein names, and BM25 rewards exactly that kind of rare term. A hybrid of BM25 and Gemma is what I would test next.

*Does LoRA improve all three metrics?* Yes, over frozen MiniLM it improves every one: nDCG@10 by +0.022, Recall@10 by +0.060, and MRR@10 by +0.015. Most of the gain is in recall; it barely moves where the first hit lands. Two caveats. First, that MRR gain is small for 162 queries and one training run, and I did not run a significance test, so I would not claim it is real yet. Second, even with LoRA, MiniLM still trails BM25 on all three metrics. LoRA helped the model it was attached to, but it did not make that model the right choice.

*What extra information would change my choice?* What I measured on this run:

| | Trained parameters | Saved weights | Vector size | Time on my Mac |
|---|---|---|---|---|
| BM25 / TF-IDF | none | sparse index | vocabulary size | seconds |
| Custom model | all (from random) | 17 MB | 128 | about 3 min for 50 epochs |
| MiniLM + LoRA | 73,728 (0.32% of 22.8M) | 303 KB adapter + 87 MB backbone | 384 | about 45 min for 20 epochs* |
| EmbeddingGemma | none (frozen, 300M) | 1.2 GB download | 768 | about 4 min to download and encode everything |

\*Each LoRA epoch also re-encodes the whole corpus for validation, and my laptop was in use during the run, so this number is rough.

Gemma needs no training, but it is roughly 13 times the size of MiniLM, and every query has to pass through that 300M model at search time. BM25 needs no GPU and answers in milliseconds. So my choice would flip toward BM25, or toward BM25 retrieving candidates that Gemma reranks, if any of these held:

- **Query latency** is strict, or queries must run on a CPU-only server. Gemma's per-query encoding time is the number I would measure first.
- **Memory** is tight. Here the document vectors are small (about 16 MB for Gemma), but at millions of documents, 768 dimensions cost twice what MiniLM's 384 do, before any index overhead.
- **The corpus changes often.** Every new document must be re-encoded, which costs nothing for BM25 and the most for Gemma.
- **Training budget.** The LoRA run took the longest of anything here and was still beaten by a model I never trained. Fine-tuning only makes sense if it beats the best frozen option.

**2. Validation selection, single run, and possible pretraining overlap.**

I agree these numbers are not a final benchmark, and the two warnings affect the methods differently:

- **Checkpoint selection favors the trained models.** The custom model and LoRA each got to keep their best of 50 and 20 epochs on these same 162 claims, which flatters their scores. BM25, TF-IDF, frozen MiniLM, and Gemma had no such selection. So the real gap between LoRA and Gemma is probably at least as large as the table shows.
- **One run is one draw.** LoRA's +0.015 MRR could change sign with a different seed. Before trusting it, I would rerun LoRA with a few seeds and run a paired test across queries, as I did for Homework 1.
- **Pretraining overlap mostly worries me for Gemma.** SciFact is a public BEIR dataset, and if Gemma saw these claims and abstracts during pretraining, part of its 0.905 could be memory rather than retrieval skill. That is the one result I would most want to check against a corpus it could not have seen.

The fix is the one the notebook names: freeze everything now (the method, LoRA rank 8, epoch 10, the prompts), and only then score the official SciFact test claims once. Those numbers would be the ones to report.

## How to run

From `homework_2/` in the repo root:

```bash
uv sync --locked
uv run hf auth login   # accept the EmbeddingGemma terms on Hugging Face first
uv run jupyter lab text_ir/teaching.ipynb
```

Then run all cells. To run it the way I did, without opening Jupyter: `cd text_ir && uv run jupyter nbconvert --to notebook --execute teaching.ipynb --ExecutePreprocessor.timeout=-1`.
