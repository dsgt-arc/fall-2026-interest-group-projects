#!/usr/bin/env bash
# Run the five-issue diagnostics with the starter project's environment.
# Usage: ./evaluate.sh [config ...] [--generate] [--only id,id]   (no names = every config)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
exec uv run --project "$here/../../../homework-1/clef-rag" python "$here/evaluate.py" "$@"
