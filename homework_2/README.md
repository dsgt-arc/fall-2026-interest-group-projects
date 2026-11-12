# Homework 2

## Part 1

Run all the cells in teaching.ipynb and discuss the questions at the bottom of the notebook.

### Discussion

1. Which method would you choose for finding one good result? For recovering all known evidence? Does LoRA improve all three metrics? What extra information about latency, memory, and training cost would change your choice?

1. These checkpoints were selected using this validation set, and results come from one training run. Public pretraining overlap with SciFact cannot be ruled out. Freeze choices before using the official test set for a final evaluation.

### Responses

**Add your responses to the questions above in this file and submit a PR.**

Validation results (162 claims, full 5,183-abstract corpus):

| Method | nDCG@10 | Recall@10 | MRR@10 |
|---|---|---|---|
| BM25 | 0.731 | 0.849 | 0.696 |
| TF-IDF | 0.631 | 0.824 | 0.574 |
| Custom model | 0.574 | 0.629 | 0.567 |
| MiniLM | 0.642 | 0.751 | 0.609 |
| MiniLM + LoRA | 0.664 | 0.810 | 0.624 |
| EmbeddingGemma | **0.905** | **0.988** | **0.880** |

1. **Method choice.**
   - *One good result (MRR@10):* EmbeddingGemma (0.880). Among cheap options, BM25 (0.696) beats every small neural model.
   - *All known evidence (Recall@10):* EmbeddingGemma (0.988), then BM25 (0.849).
   - *LoRA:* improves all three metrics over frozen MiniLM, mostly recall (+0.06). nDCG and MRR gain only about 0.02, and it still trails BM25.
   - *Cost:* BM25 needs no training or GPU, has a tiny index, and queries in milliseconds. EmbeddingGemma (300M parameters) needs far more memory and minutes to embed the corpus, though document vectors are computed once. The custom model and LoRA add training cost (50 and 20 epochs) for weaker results. Tight latency or memory budgets favor BM25. A BM25 + EmbeddingGemma hybrid, or BM25 retrieval followed by a re-ranker, trades a little cost for most of Gemma's quality.

2. **Caveats.** These numbers are optimistic. Epochs were chosen on this validation set, results come from one seed. EmbeddingGemma's lead may be partly memorization if it saw SciFact during pretraining. A fair final comparison would fix all choices (models, epochs, hyperparameters) first, then evaluate once on the official test set, ideally over several seeds.