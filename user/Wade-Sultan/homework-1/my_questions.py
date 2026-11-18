"""Convert my-questions.md into a question file experiments.py can score.

Each `- Question? | paper: paperN, page P; ... | answer: ...` line becomes one case in
the starter's questions.json format. Lines still holding `[placeholders]` are skipped.
Cases carry no `papers` or `track` field, so retrieval searches the whole corpus.
"""

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Section heading prefix -> category. Only "unsupported" changes how the scripts score a case.
CATEGORIES = {
    "Details": "detail",
    "Different wording": "wording",
    "Comparisons": "comparison",
    "Not answerable": "unsupported",
}
SOURCE = re.compile(r"(paper\d+)\s*,\s*pages?\s+([\d,\s]+)")


def gold_sources(spec):
    pages = {}
    for paper, numbers in SOURCE.findall(spec):
        for number in re.findall(r"\d+", numbers):
            pages.setdefault(paper, [])
            if int(number) not in pages[paper]:
                pages[paper].append(int(number))
    return [{"paper": paper, "pages": sorted(p)} for paper, p in pages.items()]


def parse(text):
    cases, counts, category = [], {}, None
    for line in text.splitlines():
        if line.startswith("## "):
            heading = line[3:]
            category = next((c for h, c in CATEGORIES.items() if heading.startswith(h)), None)
            continue
        if not (category and line.startswith("- ")) or "[" in line:
            continue
        question, *fields = [part.strip() for part in line[2:].split(" | ")]
        notes = dict(f.split(":", 1) for f in fields if ":" in f)
        counts[category] = counts.get(category, 0) + 1
        case = {"id": f"{category}-{counts[category]}", "category": category, "question": question}
        if category != "unsupported" and "paper" in notes:
            case["gold_sources"] = gold_sources(notes["paper"])
        if "answer" in notes:
            case["expected_answer"] = notes["answer"].strip()
        cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=Path, nargs="?", default=HERE / "my-questions.md")
    parser.add_argument("--output", type=Path, default=HERE / "my-questions.json")
    args = parser.parse_args()
    cases = parse(args.source.read_text())
    args.output.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
    for category in dict.fromkeys(c["category"] for c in cases):
        print(f"  {category}: {sum(c['category'] == category for c in cases)}")
    print(f"Wrote {len(cases)} questions to {args.output}")


if __name__ == "__main__":
    main()
