"""Shared paths and loaders. Everything runs inside the starter's environment:

    uv run --project ../../../homework-1/clef-rag python <script>.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STARTER = HERE.parents[2] / "homework-1" / "clef-rag"
INDEX_DIR = STARTER / "index"
PILOT_QUESTIONS = STARTER / "eval" / "questions.json"
TESTSET = HERE / "eval" / "testset.json"
RESULTS = HERE / "results"

sys.path.insert(0, str(STARTER / "src"))  # import the starter package unchanged

from clef_rag.index import Index  # noqa: E402
from clef_rag.models import LocalModels  # noqa: E402


def open_index():
    return Index(INDEX_DIR), LocalModels()


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
