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

## Image retrieval on GPR1200

The six models each embedded 1,250 original images and 1,000 edited copies. The experiment used the instructor’s November 9 vision notebook. [Notebook](https://github.com/dsgt-arc/fall-2026-interest-group-projects/blob/322c07e/homework_2/vision_ir/teaching.ipynb) · [Run summary](summary.csv)

1. **Domains and mistakes.** Landmarks and Products were the two highest-scoring domains for every model. The widest spread was on Sketches (0.146–0.898 mAP), followed by Nature (0.172–0.912). Objects also separated ConvNeXt (0.775) from DiT (0.203). In the Nature example, the query shows a dragonfly on a branch against blue sky. DINOv2 returned five dragonflies. ViT’s first hit was a bird perched against blue sky. In the landmark example, I-JEPA’s first four hits showed a different statue. ViT retrieved five views of the building behind the query statue.

2. **Sketches and pretraining.** DINOv2 led on Sketches (0.898 mAP), followed by ConvNeXt (0.856) and ViT (0.682). DiT received ImageNet labels as inputs during denoising and scored 0.146. The classifiers beat both JEPA models (0.292 each). The overall leader trained without class labels. The sketch classes overlap with ImageNet, and DINOv2’s pretraining data included landmark imagery. These overlaps could favor familiar subjects across models and prevent attributing the differences to labels alone.

3. **Edits and misses.** As its grayscale training augmentation suggested, DINOv2 found all 125 grayscale originals. I-JEPA found 120 of 125 blurred originals despite having no explicit blur augmentation. When the first result was wrong, it still belonged to the correct group in 67% of ViT misses, 61% of ConvNeXt misses and 75% of DINOv2 misses. The shares were 29% for I-JEPA, 27% for V-JEPA 2 and 24% for DiT.

4. **Depth and pooling.** Among the sampled depths, ViT and DINOv2 peaked at block 12/12, ConvNeXt at stage 4/4, I-JEPA at 28/32, V-JEPA 2 at 21/24 and DiT at 14/28. One explanation is that later layers specialize in the training target. A detail useful for reconstructing a noisy patch or predicting a hidden region can vary between two photos of the same object. Averaging final patch tokens gave ViT 0.691 mAP and DINOv2 0.718. Their final CLS vectors scored 0.797 and 0.877 respectively, so pooling accounted for a substantial gap even at the same depth.

5. **Vector crowding and cost.** DiT’s average cosine between different images was 0.995. Standardization lifted its mAP from 0.378 to 0.471. It ranked sixth both ways, and every other model also kept its rank. Rescaling helped DiT, but its standardized mAP still trailed the next model by 0.068. DINOv2 embedded the full set in 429 seconds and scored 0.877 mAP. I-JEPA took 4,852 seconds for 0.532, and DiT took 16,348 seconds for 0.378. Their larger models also scored lower on exact-copy retrieval (0.763 and 0.681 versus 0.895). These wall times include competing work on the same CPU.

**Work performed.** The course staff supplied the notebooks and model implementations. Codex executed the experiments, checked the outputs, and drafted the responses for Sarvesh's review.
