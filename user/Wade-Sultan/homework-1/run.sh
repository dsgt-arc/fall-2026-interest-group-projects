#!/usr/bin/env bash
# Run retrieval experiments with the starter project's environment.
# Usage: ./run.sh [config ...]   (no names = every config in configs.toml)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
exec uv run --project "$here/../../../homework-1/clef-rag" python "$here/experiments.py" "$@"
