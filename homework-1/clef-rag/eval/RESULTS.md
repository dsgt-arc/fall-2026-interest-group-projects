Validation completed on 2026-09-09 UTC after migrating to LangChain and ChromaDB.
The retrieval-metrics update was subsequently checked on the same corpus.

| Check | Result |
| --- | --- |
| Automated Python tests | 32 passed after the retrieval-metrics update |
| Ruff checks | Passed |
| Dependency lockfile | Validated with `uv lock --check` |
| Catalog coverage | 511 listed papers; 508 downloaded and indexed; one indexed preface |
| Chroma passages | 21,046 |
| Catalog evaluation | 4/4 exact-count checks passed |
| Content retrieval | Expected document/pages retrieved for 12/12 questions |
| Content answers | 12/12 returned answers accepted by the local citation-review chain |
| Unsupported questions | 4/4 abstained or explained that exhaustive counting is unsupported |
| Evaluation errors | 0 across 20 cases |
| External-network guard | A live question completed with socket connections restricted to loopback, even with tracing enabled in the environment |
| Initial LangChain ingestion | 493.89 seconds; 21,044 unique embeddings generated |
| Unchanged ingestion | 39.93 seconds; 509 extractions and 21,046 passage embeddings reused; no new embeddings |
| Content answer latency | Median 7.74 seconds; range 5.13–14.79 seconds in the final run |

The new retrieval-only evaluation scored all 12 annotated questions, with zero
errors or unscored cases, using `ir-measures` 0.4.3. The following are macro
averages over unique paper/PDF-page pairs; details are saved in
`index/retrieval-metrics.json`.

| k | Precision | Recall | Hit Rate | MRR | MAP | nDCG |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.0000 | 0.9167 | 1.0000 | 1.0000 | 0.9167 | 1.0000 |
| 3 | 0.3889 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 6 | 0.1944 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The pilot labels one or two first pages per question, and the retriever includes
opening context for named papers. These scores reflect that setup. Precision
counts unannotated pages as nonrelevant, even though they may contain useful
evidence. They are not estimates of exhaustive relevance or general search quality.
Metric tests cover independently calculated rankings, duplicate pages, missing
results, partial recall, cutoff validation, macro averaging, report output, and
error accounting in both full and retrieval-only evaluation modes.

The answer model was Google's `gemma3:12b`, digest
`f4031aab637d1ffa37b42570452ae0e4fad0314754d17ded67322e4b95836f8a`.
The embedding model was Google's `embeddinggemma:300m`, digest
`85462619ee721b466c5927d109d4cb765861907d5417b9109caebc4e614679f1`.
The optional `gemma3:4b` model was not evaluated in this migration run.

Tests exercise real temporary Chroma databases with deterministic test embeddings,
PDF loading and splitting, cache reuse, changed and deleted papers, failed-build
recovery, filtered hybrid retrieval, model restrictions, citation validation and
repair, legacy-index handling, and the absence of relative imports. The live
corpus evaluation uses local Ollama models. LangChain Community currently emits a
package deprecation warning during tests; its PDF and BM25 integrations pass the
checks with the locked dependencies.

The question set and expected answers are in `eval/questions.json`. Detailed
answers, citations, retrieved passages, and timings are saved locally in
`index/evaluation-langchain.json`. The loopback-only query is recorded in
`index/offline-langchain-check.json`. Generated reports and indexes are ignored
by Git. Use the commands in `eval/README.md` to produce a new report.

Source inspection during migration caught an answer incorrectly attributing
methods from a related-work passage to the paper's own system. The answer chain
now reviews citation support, attempts one correction with rejected evidence
removed, and withholds answers that still fail review. The final paper6/paper8
comparison was checked against the cited abstracts and system description.
These local model checks are fallible: acceptance by the reviewer is not a measure
of factual accuracy, and the evaluation report retains manual-review reminders.

The pilot concentrates on BioASQ and CheckThat! and is a development set, not a
held-out benchmark. It does not establish accuracy on every paper or task. Four
documents have low-text page warnings. Complex tables, scanned text, and unseen
questions still warrant inspecting the source PDFs. Papers 236, 318, and 345
remain unavailable and cannot supply full-text evidence.
