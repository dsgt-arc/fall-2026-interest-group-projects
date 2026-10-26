# CLEF RAG

In this assignment, we explore building a retrieval-augmented generation (RAG) system for the CLEF 2026 working 
notes papers.

## Installation

The following software is required to run CLEF RAG:

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/download)

Ask questions about the CLEF 2026 working notes using LangChain, persistent
ChromaDB, and local Ollama models.
Generation uses Google's `gemma3:12b` (optionally `gemma3:4b`); embeddings use
Google's `embeddinggemma:300m`. These are the only allowed models. The application
does not use Chinese-origin models or automatically substitute another model.

```sh
# setup the python environment
uv sync --locked

# pull the required Ollama models
ollama pull gemma3:12b
ollama pull embeddinggemma:300m

# download the CLEF 2026 working notes papers
python download_papers.py

# ingest the papers into the RAG system
uv run clef-rag ingest

# ask a question
uv run clef-rag ask "What is the BioASQ track about?"
```
## The corpus

The initial corpus contains 508 available papers and one preface. Use `stats` to see
current listed, downloaded, indexed, and supporting-document counts separately.

```sh
uv run clef-rag stats
uv run clef-rag papers --track BioASQ
uv run clef-rag papers --author "Udo Kruschwitz"
uv run clef-rag papers --title "MedCascade"
uv run clef-rag search "biomedical question answering"
uv run clef-rag ask "Which approach does paper30 use?"
uv run clef-rag ask "Compare the retrieval approaches" --paper paper6 --paper paper8
uv run clef-rag ask "How many papers are in BioASQ?"
uv run clef-rag ask "What methods are described?" --track CheckThat!
```

`--paper` accepts an ID, filename, or unambiguous title substring and can be
repeated. Paper IDs, quoted matching titles, and complete titles in questions are
also recognized. `--track` and `--author` filter retrieval before ranking. Use
`--model gemma3:4b` on `ask` after explicitly downloading that smaller model when
memory is limited. The 12B default was selected after inspecting answer quality
on this machine's GPUs. Add `--json`
to any command for structured output; answers include model digests, retrieved
passages, used sources, and elapsed time. Every command supports `--index-dir`.

## Ingestion and Indexing

See chapters
- [Document Processing Pipeline - RAG Indexing Pipeline](https://www.youtube.com/watch?v=mHxLXzYjQRE&t=1707s)
- [Hands-on - Create a Vector DB Using Chroma](https://www.youtube.com/watch?v=mHxLXzYjQRE&t=3665s)

LangChain's `PyPDFLoader` extracts PDFs page by page; `RecursiveCharacterTextSplitter`
creates passages of up to 1,400 UTF-8 bytes with up to 250 bytes of overlap. Title/track headers count
against a conservative 1,900-byte embedding input limit. 

The embeddings are stored locally in ChromaDB. ChromaDB persists passage text, metadata, and vectors.
LangChain's Chroma retriever and `BM25Retriever` each retrieve up to 20 candidates after source filtering;
`EnsembleRetriever` combines their rankings. The application selects six evidence
passages, including opening context for the best matching paper or each named
paper in a comparison. The answer's context budget may reduce the final set.
SQLite stores the paper catalog so exact counts and unavailable papers remain
queryable. It is not used for vector storage or keyword ranking.


```sh
# Update the catalog after downloading newly available PDFs.
python download_papers.py
uv run clef-rag ingest --refresh-catalog

# Use custom locations or a previously saved catalog without accessing the web.
uv run clef-rag ingest --data-dir data --index-dir index --catalog-html saved-catalog.html
uv run clef-rag ingest --batch-size 8

# Automated tests and the 20-question pilot evaluation.
uv run pytest -q
uv run clef-rag evaluate --retrieval-only --output index/retrieval-evaluation.json
uv run clef-rag evaluate --retrieval-only --k 1 3 6 --output index/retrieval-metrics.json
uv run clef-rag evaluate --output index/evaluation.json
```

## Evaluation

Evaluation reports Precision, Recall, Hit Rate, MRR, MAP, and nDCG at the requested
cutoffs using [ir-measures](https://github.com/terrierteam/ir_measures), with per-question results and macro averages. Scoring uses unique paper/PDF-page pairs and the existing gold-page annotations. `--k`
changes metric cutoffs, not the six-passage retrieval depth. See
[eval/README.md](eval/README.md) for definitions and annotation limitations.

The downloader itself needs only Python's standard library. It saves PDFs under
`data/` next to the script, skips completed PDFs, retries failures up to three
attempts, and reports failed links with a nonzero exit code. Use
`python download_papers.py --output-dir data --workers 2` to adjust its settings.

See [RAG_PLAN.md](RAG_PLAN.md) for the architecture and [eval/README.md](eval/README.md)
for evaluation criteria and manual citation review.

## TODO: Improving the Systems

Your task is to explore ways to improve the retrieval-augmented generation (RAG) system, including optimizing embeddings, retrievers, and evaluation metrics. Lastly, you need to come up with some questions
on your own regarding CLEF. You should query our initial system, and your improved system and compare the results.

The following chapters of the video can help you provide context.

[Embedding Dimensions - Deep Dive](https://www.youtube.com/watch?v=mHxLXzYjQRE&t=2892s)
[Debugging RAG Systems](https://www.youtube.com/watch?v=mHxLXzYjQRE&t=5596s)

Here are some suggestions for improving the system: 

- Experiment with different embedding dimensions and models.
- Tune the retriever parameters, such as the number of candidates retrieved.
- Explore alternative retrieval strategies, such as hybrid or dense retrieval methods.
- Consider incorporating more data into the corpus, such as the CLEF 2025 dataset.

The most important question you should be prepared to answer about your work:

- How do you know your RAG system is better? What metrics or evaluation methods did you use to determine improvement and why?