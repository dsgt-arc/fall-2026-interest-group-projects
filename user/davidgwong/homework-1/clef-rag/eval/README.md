The 20 pilot questions cover catalog counts, eight individual papers, four
comparisons or task summaries, and four unsupported questions. Expected answers
and page references were checked against the saved catalog and extracted PDF
first pages during implementation. This initial set emphasizes BioASQ plus
CheckThat!; expand to other tracks before drawing corpus-wide quality conclusions.

```sh
uv run clef-rag evaluate --retrieval-only --output index/retrieval-evaluation.json
uv run clef-rag evaluate --retrieval-only --k 1 3 6 --output index/retrieval-metrics.json
uv run clef-rag evaluate --output index/evaluation.json
```

Both evaluation modes report standard retrieval metrics computed by
[`ir-measures`](https://ir-measur.es/en/latest/measures.html). Cutoffs default to
1, 3, and 6; `--k` accepts one or more positive integers. Each question records
`retrieval_metrics`, and `summary.retrieval_metrics` contains the arithmetic mean
across scored questions, keyed by cutoff.

| Summary metric | Meaning at cutoff k |
| --- | --- |
| `precision` | Relevant pages in the first k positions, divided by k |
| `recall` | Relevant pages retrieved by k, divided by all annotated relevant pages |
| `hit_rate` | Fraction of questions with at least one relevant page by k |
| `mrr` | Mean reciprocal rank of the first relevant page; zero if absent by k |
| `map` | Mean average precision: sum of precision at each relevant rank through k, divided by all annotated relevant pages |
| `ndcg` | Discounted relevance through k, normalized by the ideal ranking, with binary relevance and log2(rank + 1) discounts |

Per-question fields use `hit`, `reciprocal_rank`, and `average_precision` for the
corresponding unaveraged values. Metrics are in [0, 1], with higher values better.
MAP uses the total gold-page count as its denominator even when it exceeds k.
Precision always divides by k, including when fewer results are returned.

The evaluated ranking is the final retrieved evidence list, before the answer's
context budget or citation selection. Repeated `(paper ID, PDF page)` pairs are
collapsed in first-occurrence order before applying cutoffs. Different pages in
the same paper remain distinct. `retrieved_pages` and `gold_pages` make this
conversion reviewable in each result. Every unique page in `gold_sources` receives
relevance 1; all other pages receive 0. Duplicate annotations do not add weight.
For example, `{"paper": "paper6", "pages": [1, 2]}` defines two relevant pages.

`--k` controls scoring only. Retrieval still returns at most six passages, which
may collapse to fewer unique pages. Asking for k=10 does not retrieve more results;
missing ranks earn no credit. An empty result set with gold pages scores zero.
Questions without gold pages are excluded from metric averages. Failed cases are
also excluded and exposed by `retrieval_unscored` and `errors`; use these counts
alongside the scores. `retrieval_eligible` counts cases with judgments, while
`retrieval_checked` counts those actually scored. No scored queries produces
`retrieval_metrics: null` rather than an invented zero average.

The existing `retrieval_pass` check is retained: the final six passages must
include at least one acceptable page for every gold-source entry. With multiple
pages per entry, this check can pass while page recall is below 1.
Catalog checks compare exact counts;
unsupported checks require abstention. A stale count after a catalog refresh
requires updating the corresponding expectation, not changing correct application
behavior.

These are coarse metrics against incomplete page annotations. An unlisted page
can still be useful, and a passage on a gold page can omit the supporting sentence.
Most pilot judgments identify first pages; the retriever deliberately prepends
opening context for named papers. High scores on this set therefore do not establish
general search quality. Add judgments beyond abstracts and more unfiltered questions
before comparing retrieval strategies.

The generated report includes answers, actual citations, retrieved passages,
latency, and errors. Review every factual claim against its source page and the
expected answer. Citation-ID validation and the local model's citation review
do not independently establish factual accuracy.

See [RESULTS.md](RESULTS.md) for the completed LangChain/Chroma migration evaluation.
