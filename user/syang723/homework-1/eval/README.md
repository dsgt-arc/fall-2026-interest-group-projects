# Hard retrieval question set

`questions-hard.json` has 47 questions in the same format as `homework_1/clef-rag/eval/questions.json`, so
`clef-rag evaluate --questions` can run it directly.

## Why a new set

The 12 scored questions in the original pilot set reach the maximum possible score on every metric for the
baseline. Precision@3 = 14/36 and Precision@6 = 14/72 are the ceilings given 14 gold pages. A set with no
headroom cannot show that a change helps.

## Design

- **All 16 tracks:** 2 papers sampled per track (seed 2026), excluding papers used in the pilot set.
- **No paper IDs or titles in questions:** retrieval runs over the whole corpus, with no paper filter and no
  title shortcut.
- **Body-page facts:** answers are specific numbers, hyperparameters or design choices from methods and
  results pages, not abstracts.
- **Exhaustive gold pages:** each question has regex anchors (`tools/hard_spec.py`). Every page in the full
  corpus that matches all of them was reviewed. Coincidental matches were dropped. Pages that genuinely
  state the same fact (e.g. a dataset size quoted by several teams in one track) were kept, so 6 questions
  have gold pages in several papers.

## Rebuild

```sh
cd homework_1/clef-rag
uv run python ../../user/syang723/homework-1/eval/tools/extract_pages.py data /tmp/pages.json
python3 ../../user/syang723/homework-1/eval/tools/build.py /tmp/pages.json ../../user/syang723/homework-1/eval/tools \
  ../../user/syang723/homework-1/eval/questions-hard.json index/<build>/catalog.sqlite
```

## Limitations

- Gold pages come from pypdf text. Facts that exist only in figures or images are not covered.
- The questions were written by one person from pages that person read. Phrasing may favour lexical overlap
  with those pages.
