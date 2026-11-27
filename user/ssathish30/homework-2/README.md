# Homework 2

## Text retrieval on SciFact

**BM25 was the best of the five retrievers measured for both tasks.** On 162 validation claims, it put a relevant abstract first 101 times. Its mean reciprocal rank of 0.6965 makes it my choice for a reader who needs one good result. For a reviewer collecting evidence, it also had the highest recall. The top ten recovered 84.88% of the labeled evidence per claim, on average.

| Retriever | MRR@10 | Recall@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| BM25 | 0.6965 | 0.8488 | 0.7313 |
| TF-IDF | 0.5743 | 0.8241 | 0.6313 |
| Custom Transformer | 0.5552 | 0.6317 | 0.5652 |
| Frozen MiniLM | 0.6094 | 0.7505 | 0.6420 |
| MiniLM with LoRA | 0.6343 | 0.8194 | 0.6730 |

**LoRA improved all three metrics.** Enabling the adapter on the same MiniLM backbone raised recall by 6.89 percentage points, MRR by 0.0249, and nDCG by 0.0309. It brought MiniLM within 2.93 percentage points of BM25 on recall. BM25 also ranked evidence earlier than TF-IDF. The two methods placed a relevant abstract first for 101 and 73 claims respectively, but found at least one in the top ten for 140 and 136.

The 20-epoch LoRA training cell took 3 h 58 min on the M1 CPU, including validation after every epoch, with the vision run sharing the machine. Before deploying either retriever, I would measure median and 95th-percentile time for one query against a prebuilt index, peak memory with the model and index loaded, and the cost of rebuilding the index after the corpus changes. Query latency affects the reader directly. A frequently updated corpus also makes index rebuilds part of the operating cost.

All methods searched the same 5,183 abstracts. The notebook used 647 claims for training and the same 162 validation claims for checkpoint selection and scoring. The best LoRA checkpoint came from epoch 15. Epoch 20 had a lower nDCG. I would freeze the checkpoint and retrieval settings before opening the official test labels. This run used one seed, and its scores cannot rule out overlap between SciFact and a pretrained model's training data.

EmbeddingGemma remains unmeasured. The pinned model requires Hugging Face access to its weights. The other five rows are recorded at full precision in [text-comparison.csv](text-comparison.csv).

**Work performed.** The course staff supplied the notebooks and model implementations. Codex executed the experiments, checked the outputs, and drafted the responses for Sarvesh's review.
