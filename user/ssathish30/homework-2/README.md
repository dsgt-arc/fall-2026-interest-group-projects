# Homework 2

## Text retrieval on SciFact

The course notebook ranked 5,183 abstracts for each of 162 validation claims. It reserved 647 other claims for training. All scores below use the top ten results per claim.

| Method | MRR@10 | Recall@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| BM25 | 0.6965 | 0.8488 | 0.7313 |
| TF-IDF | 0.5743 | 0.8241 | 0.6313 |

BM25 gained 0.1222 in MRR over TF-IDF and 0.0247 in recall. Most of its advantage was in how early the first relevant abstract appeared. To choose a system for a reader who needs one paper, I would compare MRR. For a reviewer collecting evidence, I would compare recall. Even BM25 left about 15% of the labeled evidence outside the top ten on average.

These are validation scores. The training code uses this same split to select its best epoch, so a comparison on it can favor the selected checkpoint. A final comparison should fix the model and settings before using the official test labels. One training seed gives no estimate of run-to-run variation. The experiment also cannot establish whether a pretrained encoder saw SciFact during pretraining.

The course staff supplied the models and notebook. Codex ran the experiment and drafted this write-up for my review.
