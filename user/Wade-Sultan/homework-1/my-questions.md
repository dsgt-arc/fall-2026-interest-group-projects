# My CLEF questions

One question per line, in the form:

    - Question? | paper: paperN, page P | answer: what the page says

Only the question is required. The `paper:` and `answer:` notes are what make a question
scoreable, so fill them in when you can: the paper ID (as `clef-rag` shows it, e.g.
`paper12`) and the PDF page number where the answer appears. If the answer spans papers,
list each one: `paper: paper3, page 4; paper9, page 2`.

The questions below are placeholders showing the format. Replace each one (and the bits in
`[brackets]`) with a real question, and delete any you don't use.

## Details deep inside one paper (8-10)

Don't name the paper in the question, and pick facts that live past page 1.

- How many unique generator models were used to generate the machine-generated portion of the PAN dataset? | paper: paper397, page 3 | answer: 23
- How many text samples are in the Kaggle Academic out-of-distribution dataset? | paper: paper416, page 3 | answer: 6,069
- What GPU did the HULAT-HAI authors run their models on? | paper: paper468, page 4 | answer: NVIDIA L4 GPU
- What low-resource language did the main dataset for the ELOQUENT 2026 task use as the anchor? | paper: paper386, page 3 | answer: Tibetan
- What token limit did the DS@GT team on the PAN 2026 task set? | paper: paper397, page 5 | answer: 512
- What is the name of the metric that assigns a score equal to the reciprocal rank of an item by relevance, considering only the top 5 documents? | paper: paper66, page 3 | answer: MRR@5
- How many independent fine tuning runs did the GetGut@AAU team perform on each dataset? | paper: paper17, page 7 | answer: 5
- What context retrieval method did the gfreitas team say helped most? | paper: paper255, page 7 | answer: visual context retrieval 


## Different wording from the paper (4-5)

Ask about something a paper covers, but use synonyms, a spelled-out acronym, or plain
language in place of the paper's own terms.

- How did the clinical summarization team that used group-relative reinforcement learning do compared with the baselines? | paper: paper2, page 14 | answer: ixa-sum's GRPO system with dynamic one-shot prompting beat both the zero-shot and fine-tuned MCS-mT5-large baselines and led BERTScore in English (0.876), Spanish (0.882), Catalan (0.882) and French (0.873)
- For the BioASQ team that had an LLM refine its own keyword searches, did merging the rankings of three queries beat a single query? | paper: paper7, page 6 | answer: Mixed. Phase A+ Batch 3: list F-measure up (0.2717 vs 0.2128) but factoid MRR down (0.3431 vs 0.4216). Phase B Batch 3: factoid MRR up (0.5000 vs 0.4608) but list F-measure down (0.2629 vs 0.2989)
- How much did evolving the prompt automatically improve English results over the stock model for the HIPE team? | paper: paper221, page 8 | answer: macro recall rose from 0.591 to 0.654 with a prompt optimized on GPT-OSS-20B, and to 0.660 after also fine-tuning on distilled reasoning chains
- Which task asks for teams to improve the detection of sexism via multimodal contextual, visual, and social cues? | paper: paper152, page 1 | answer: EXIST
- Which run achieve the best Average Difference between Overall Depression Levels? | paper: paper147, page 5 | answer: 3


## Comparisons across papers or a task (3-4)

- Which PlantCLEF teams tried retrieval-augmented classification, and did it help? | paper: paper320, page 8; paper320, page 10; paper360, page 9 | answer: several teams blended kNN or prototype retrieval scores with model logits; one team found it (and LoRA) did not beat their tuned tiling baseline under limited compute
- How do the JANZZ.technology and SBERT contrastive fine-tuning systems for TalentCLEF Task B differ? | paper: paper476, page 1; paper476, page 3; paper477, page 1 | answer: paper476 uses two stages, a gte-modernbert bi-encoder then a fine-tuned Qwen3-Reranker cross-encoder; paper477 fine-tunes Sentence-BERT with contrastive learning on alias-enriched ESCO skills, raising MAP from 0.1430 to 0.2564
- How do the DS@GT and zhengqiaozeng systems for Voight-Kampff AI detection differ? | paper: paper397, page 1; paper397, page 5; paper434, page 1; paper434, page 4 | answer: DS@GT LoRA-fine-tunes the decoder model Qwen2.5-1.5B; zhengqiaozeng fine-tunes the encoder DeBERTa-v3-large with an R-Drop plus contrastive loss. Each team's extra technique (homoglyph/synonym augmentation, extra losses) gave no real gain
- What do the PAN Reasoning Trajectory Detection and Touché labs have in common in how they evaluate submissions? | paper: paper390, page 5; paper492, page 5 | answer: both use macro-F1 as the primary metric so every class counts equally, including rare ones

## Not answerable from the papers (2-3)

- How much did the gfreitas team on ImageCLEF spend in AI API costs during evaluation? | answer: not stated
- How much time did the TalentCLEF organizers spend on annotation? | answer: not stated
- Which team will win TalentCLEF next year? | answer: not answerable
