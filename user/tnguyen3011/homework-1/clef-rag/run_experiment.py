#!/usr/bin/env python3
"""Run the submitted RAG settings in a temporary copy of the shared starter."""

import argparse
import ast
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def configure_retriever(source, dense_weight, top_k):
    """Change only the experiment's two parameters in a parsed source copy."""
    tree = ast.parse(source)
    defaults_changed = weights_changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Retriever":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "search":
                    for i, argument in enumerate(method.args.kwonlyargs):
                        if argument.arg == "top_k":
                            method.args.kw_defaults[i] = ast.Constant(top_k)
                            defaults_changed += 1
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "EnsembleRetriever":
                for keyword in node.keywords:
                    if keyword.arg == "weights":
                        keyword.value = ast.List(
                            elts=[ast.Constant(dense_weight), ast.Constant(1.0 - dense_weight)],
                            ctx=ast.Load(),
                        )
                        weights_changed += 1
    if (defaults_changed, weights_changed) != (1, 1):
        raise ValueError("Starter layout changed; expected one retrieval default and one ensemble.")
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def command_arguments(command, starter, output_dir, top_k):
    """Resolve caller-supplied paths before switching into the temporary project."""
    command = list(command)
    path_flags = {"--index-dir", "--data-dir", "--catalog-html", "--questions", "--output"}
    present = set()
    i = 1
    while i < len(command):
        flag, separator, value = command[i].partition("=")
        if flag in path_flags:
            present.add(flag)
            if separator:
                command[i] = flag + "=" + str(Path(value).expanduser().resolve())
            else:
                if i + 1 == len(command):
                    raise ValueError(f"{flag} requires a path")
                i += 1
                command[i] = str(Path(command[i]).expanduser().resolve())
        i += 1
    defaults = {"--index-dir": starter / "index"}
    if command[0] == "ingest":
        defaults["--data-dir"] = starter / "data"
    if command[0] == "evaluate":
        defaults["--questions"] = starter / "eval" / "questions.json"
        defaults["--output"] = output_dir / "evaluation-new.json"
    for flag, value in defaults.items():
        if flag not in present:
            command += [flag, str(value)]
    if command[0] == "search" and not any(
        value == "--top-k" or value.startswith("--top-k=") for value in command[1:]
    ):
        command += ["--top-k", str(top_k)]
    return command


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--starter", type=Path, default=here.parents[3] / "homework_1" / "clef-rag")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check setup and print the plan without uv, models, or data.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or command[0] not in {"ingest", "stats", "papers", "search", "ask", "evaluate"}:
        parser.error("Pass a clef-rag command after --, for example: -- evaluate --retrieval-only")
    if not math.isfinite(args.dense_weight) or not 0 <= args.dense_weight <= 1:
        parser.error("--dense-weight must be between 0 and 1")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    starter = args.starter.expanduser().resolve()
    required = ("pyproject.toml", "uv.lock", "README.md", "src", "eval")
    if any(not (starter / item).exists() for item in required):
        parser.error(f"Expected a complete shared RAG starter at {starter}")
    try:
        forwarded = command_arguments(command, starter, here / "runs", args.top_k)
        uv = shutil.which("uv")
        if not args.dry_run and uv is None:
            parser.error("Install uv before running an experiment.")
        with tempfile.TemporaryDirectory(prefix="arc-pr5-rag-") as temporary:
            project = Path(temporary) / "clef-rag"
            project.mkdir()
            # Explicit allowlist: never copy the corpus archive, index, or environment.
            for item in required:
                source = starter / item
                destination = project / item
                if source.is_dir():
                    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__"))
                else:
                    shutil.copy2(source, destination)
            retriever = project / "src" / "clef_rag" / "retrieve.py"
            retriever.write_text(configure_retriever(
                retriever.read_text(), args.dense_weight, args.top_k
            ))
            invocation = [uv or "uv", "run", "--locked", "--project", str(project),
                          "clef-rag", *forwarded]
            plan = {
                "dense_weight": args.dense_weight,
                "lexical_weight": 1.0 - args.dense_weight,
                "retrieval_top_k": args.top_k,
                "starter": str(starter),
                "temporary_project": str(project),
                "command": invocation,
                "cwd": str(project),
                "dry_run": args.dry_run,
            }
            print(json.dumps(plan, indent=2), file=sys.stderr)
            if args.dry_run:
                return 0
            # Default outputs persist in the student's ignored runs directory.
            (here / "runs").mkdir(exist_ok=True)
            return subprocess.run(invocation, cwd=project, check=False).returncode
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
