#!/usr/bin/env bash
# Build the doc2query index with the starter project's environment.
# Usage: ./doc2query.sh [--limit N]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
exec uv run --project "$here/../../../homework-1/clef-rag" python "$here/doc2query.py" "$@"
