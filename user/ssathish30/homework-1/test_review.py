"""CPU-only integration checks; no packages, datasets, or model calls required.

Run with: python3 user/ssathish30/homework-1/test_review.py
"""

import ast
import json
from pathlib import Path
import tempfile
import time
import unittest

import setup_review

HERE = Path(__file__).resolve().parent


def load_without_optional_dependencies(name, **namespace):
    """Execute the submitted functions with their model/library imports stubbed."""
    tree = ast.parse((HERE / "src" / name).read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, name, "exec"), namespace)
    return namespace


class MemoryPath:
    def __init__(self, text=""):
        self.text = text
        self.parent = self

    def read_text(self):
        return self.text

    def write_text(self, text):
        self.text = text

    def mkdir(self, **kwargs):
        pass


class ReviewChecks(unittest.TestCase):
    def test_full_evaluation_rejects_overrides_before_running(self):
        calls = []
        namespace = load_without_optional_dependencies(
            "evaluate.py", json=json, time=time, DEFAULT_CUTOFFS=(1, 3, 6),
            normalize_cutoffs=tuple, aggregate_retrieval_metrics=lambda rows: {},
            ask=lambda *args, **kwargs: calls.append(kwargs),
        )
        for options in (
            {"retrieval": {"introductions": False}},
            {"retrieval": {"weights": [1, 0]}},
            {"retrieval": {"rerank_depth": 20}},
            {"ignore_filters": True}, {"top_k": 1},
        ):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "require --retrieval-only"):
                    namespace["evaluate"](None, None, None, None, **options)
        self.assertEqual(calls, [])

    def test_full_evaluation_accepts_cli_default_settings(self):
        calls = []

        def ask(*args, **kwargs):
            calls.append(kwargs)
            return {"kind": "content", "sources": [], "retrieved_sources": []}

        namespace = load_without_optional_dependencies(
            "evaluate.py", json=json, time=time, DEFAULT_CUTOFFS=(1, 3, 6),
            normalize_cutoffs=tuple, aggregate_retrieval_metrics=lambda rows: {}, ask=ask,
        )
        questions = MemoryPath(json.dumps([
            {"id": "q1", "category": "paper", "question": "q", "papers": ["p1"]}
        ]))
        report = namespace["evaluate"](
            type("Index", (), {"manifest": {}})(),
            type("Models", (), {"chat_model": "stub"})(), questions, MemoryPath(),
            retrieval={"weights": [0.5, 0.5], "rrf_c": 60, "candidates": 20,
                       "introductions": True, "shuffle_seed": None, "rerank_depth": 0},
        )
        self.assertEqual(report["summary"]["errors"], 0)
        self.assertEqual(len(calls), 1)

    def test_reranker_failure_preserves_original_top_passages(self):
        namespace = load_without_optional_dependencies("rerank.py")

        class Model:
            def chat(self, *args, **kwargs):
                raise RuntimeError("simulated scoring failure")

        sources = [{"id": "first", "text": "a"}, {"id": "second", "text": "b"}]
        result = namespace["rerank"](Model(), "q", sources, keep=1)
        self.assertEqual([r["id"] for r in result], ["first"])
        self.assertEqual(result[0]["source_id"], "S1")
        self.assertNotIn("source_id", sources[0])

    def test_reranker_success_still_uses_scores_and_stable_ties(self):
        namespace = load_without_optional_dependencies("rerank.py")

        class Model:
            def __init__(self):
                self.scores = iter([0, 3, 3])

            def chat(self, *args, **kwargs):
                return {"score": next(self.scores)}

        sources = [{"id": str(i), "text": "passage"} for i in range(3)]
        result = namespace["rerank"](Model(), "q", sources, keep=2)
        self.assertEqual([r["id"] for r in result], ["1", "2"])

    def test_setup_applies_patch_and_supplies_evaluation_tools(self):
        with tempfile.TemporaryDirectory(prefix="arc-pr6-check-") as temporary:
            destination = Path(temporary) / "review"
            setup_review.prepare(destination)
            clef = destination / "homework_1/clef-rag"
            for relative in ("eval/questions-hard.json", "eval/questions-hard-test.json",
                             "eval/tools/paired_test.py", "src/clef_rag/rerank.py"):
                self.assertTrue((clef / relative).is_file(), relative)
            self.assertFalse((clef / "data.zip").exists())
            self.assertFalse((clef / "data").exists())
            config = destination / "homework_1/fashion-fm/configs/train.yml"
            self.assertIn("epochs: 60", config.read_text())
            with self.assertRaises(FileExistsError):
                setup_review.prepare(destination)
            self.assertTrue(clef.is_dir())


if __name__ == "__main__":
    unittest.main()
