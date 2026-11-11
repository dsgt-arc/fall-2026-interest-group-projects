#!/usr/bin/env bash
# Retrieval-only sweep over env-var knobs. Run from homework_1/clef-rag.
# Usage: bash sweep.sh <index-dir> <name>=<ENV ASSIGNMENTS> ...
#   e.g. bash sweep.sh index base= w07="RAG_W_DENSE=0.7" combo="RAG_INTRO=second RAG_W_DENSE=0.7"
set -euo pipefail
OUT=../../user/syang723/homework-1/results
IDX=$1; shift
for spec in "$@"; do
  name=${spec%%=*}; envs=${spec#*=}
  for set in hard:../../user/syang723/homework-1/eval/questions-hard.json orig:eval/questions.json; do
    env $envs uv run clef-rag evaluate --retrieval-only --index-dir "$IDX" --k 1 3 6 \
      --questions "${set#*:}" --output "$OUT/${set%%:*}-$name.json" > /dev/null
  done
  echo "done $name"
done
