"""Prepare an isolated copy of the current starters with this submission applied.

Uses existing local starter files; does not install packages, fetch LFS, download data,
call models, or change the repository's shared starter directories.
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def prepare(destination=None):
    submission = Path(__file__).resolve().parent
    repository = submission.parents[2]
    starter = repository / "homework_1"
    if not starter.is_dir():
        raise ValueError("Run from a checkout containing the current homework_1 starter.")
    if destination is None:
        workspace = Path(tempfile.mkdtemp(prefix="arc-pr6-review-"))
    else:
        workspace = Path(destination).expanduser().resolve()
        # Refuse to reuse an existing directory or overwrite another review.
        workspace.mkdir(parents=True, exist_ok=False)

    omitted = shutil.ignore_patterns(
        ".git", ".venv", "__pycache__", ".pytest_cache", ".DS_Store",
        "data", "data.zip", "index", "outputs", "logs",
    )
    try:
        shutil.copytree(starter, workspace / "homework_1", ignore=omitted)
        patch = submission / "changes.patch"
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=workspace, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=workspace, check=True)
        clef = workspace / "homework_1/clef-rag"
        # The experiments and comparison tool are student files, not starter files.
        shutil.copytree(submission / "eval", clef / "eval", dirs_exist_ok=True)
    except Exception:
        # Only this newly-created workspace is removed on failure.
        shutil.rmtree(workspace)
        raise
    return workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, help="New directory; must not exist")
    args = parser.parse_args()
    try:
        workspace = prepare(args.destination)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Setup failed: {exc}\n")
    print(f"Prepared isolated review: {workspace}")
    print(f"CLEF RAG: {workspace / 'homework_1/clef-rag'}")
    print(f"Fashion FM: {workspace / 'homework_1/fashion-fm'}")
    print("Downloaded data, environments, indexes, and training outputs were omitted.")
    print("No downloads or model runs were started; any local starter checkpoint was copied.")


if __name__ == "__main__":
    main()
