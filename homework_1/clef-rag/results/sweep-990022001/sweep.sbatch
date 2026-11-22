#!/bin/bash
#SBATCH --job-name=clef-sweep
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=clef-sweep-%j.out

set -euo pipefail
cd "$HOME/scratch/fall-2026-interest-group-projects/homework_1/clef-rag"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$HOME/scratch/.uv-cache"
export OLLAMA_MODELS="$HOME/scratch/ollama/models"
export OLLAMA_NO_CLOUD=1

resume_id="${CLEF_RESUME_JOB_ID:-}"
if [[ -n "$resume_id" && ! "$resume_id" =~ ^[0-9]+$ ]]; then
    echo "CLEF_RESUME_JOB_ID must be the numeric original job ID." >&2
    exit 1
fi
smoke="${CLEF_SWEEP_SMOKE:-0}"
if [[ "$smoke" != 0 && "$smoke" != 1 ]]; then
    echo "CLEF_SWEEP_SMOKE must be 0 or 1." >&2
    exit 1
fi
generation=$(.venv/bin/python -c 'import json; print(json.load(open("index/current.json"))["generation"])')
if [[ ! "$generation" =~ ^build-[0-9a-f]{32}$ ]]; then
    echo "Invalid index generation: $generation" >&2
    exit 1
fi
results="results/sweep-${resume_id:-$SLURM_JOB_ID}"
mkdir -p results
if [[ -n "$resume_id" ]]; then
    if [[ ! -d "$results" ]]; then
        echo "Sweep does not exist: $results" >&2
        exit 1
    fi
else
    mkdir "$results"
fi
exec 9>"$results/.lock"
if ! flock -n 9; then
    echo "Another job is using $results." >&2
    exit 1
fi

input_hashes() {
    sha256sum src/clef_rag/*.py eval/questions.json pyproject.toml uv.lock \
        .python-version sweep.sbatch index/current.json "index/$generation/manifest.json"
}
if [[ -n "$resume_id" ]]; then
    if [[ ! -f "$results/input-hashes.txt" || ! -f "$results/sweep-mode.txt" ]] ||
        [[ "$(cat "$results/sweep-mode.txt")" != "$smoke" ]] ||
        ! input_hashes | cmp -s "$results/input-hashes.txt" -; then
        echo "Cannot resume: sweep mode or recorded inputs have changed or are missing." >&2
        exit 1
    fi
    echo "Resuming $results"
else
    mkdir "$results/source"
    cp src/clef_rag/*.py "$results/source/"
    cp eval/questions.json "$results/questions.json"
    cp sweep.sbatch "$results/sweep.sbatch"
    input_hashes > "$results/input-hashes.txt"
    printf '%s\n' "$smoke" > "$results/sweep-mode.txt"
fi

# The application uses localhost:11434; never attach to another user's server.
.venv/bin/python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 11434)); s.close()'
"$HOME/scratch/ollama/bin/ollama" serve > "$results/ollama-${SLURM_JOB_ID}.log" 2>&1 &
ollama_pid=$!
trap 'kill "$ollama_pid" 2>/dev/null || true' EXIT
for ((i=0; i<90; i++)); do
    if ! kill -0 "$ollama_pid" 2>/dev/null; then
        echo "Ollama exited during startup; see $results/ollama-${SLURM_JOB_ID}.log" >&2
        exit 1
    fi
    if curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        break
    fi
    sleep 2
done
curl -fsS --max-time 5 http://127.0.0.1:11434/api/tags > /dev/null
curl -fsS --max-time 300 http://127.0.0.1:11434/api/embed \
    -d '{"model":"embeddinggemma:300m","input":"warmup"}' > /dev/null
curl -fsS --max-time 10 http://127.0.0.1:11434/api/ps |
    .venv/bin/python -c 'import json,sys; models=json.load(sys.stdin)["models"]; sys.exit(0 if any(m["name"] == "embeddinggemma:300m" and m.get("size_vram", 0) > 0 for m in models) else 1)'

valid_report() {
    .venv/bin/python - "$@" <<'PY'
import json
import sys

path, mode, weight, constant, depth, top, generation = sys.argv[1:]
try:
    with open(path) as stream:
        report = json.load(stream)
    with open("eval/questions.json") as stream:
        cases = json.load(stream)
    expected = [c["id"] for c in cases if c["category"] not in ("catalog", "unsupported")]
    summary = report["summary"]
    valid = (
        report["retrieval_settings"] == {
            "mode": mode, "dense_weight": float(weight), "rrf_constant": int(constant),
            "candidate_depth": int(depth), "top_k": int(top),
        }
        and report["index"]["generation"] == generation
        and report["retrieval_evaluation"]["cutoffs"] == [1, 3, 6, 10, 15, 20]
        and [row["id"] for row in report["results"]] == expected
        and summary["cases_run"] == len(expected)
        and summary["errors"] == 0
        and summary["retrieval_unscored"] == 0
    )
except (OSError, ValueError, KeyError, TypeError) as error:
    print(f"Invalid report {path}: {error}", file=sys.stderr)
    sys.exit(1)
if not valid:
    print(f"Incomplete or mismatched report: {path}", file=sys.stderr)
    sys.exit(1)
PY
}

run_eval() {
    local mode=$1 weight=$2 constant=$3 depth=$4 top=$5
    local output="$results/${mode}-w${weight}-c${constant}-d${depth}-top${top}.json"
    local partial="${output%.json}.partial-${SLURM_JOB_ID}-$$.json"
    if [[ -e "$output" ]]; then
        if ! valid_report "$output" "$mode" "$weight" "$constant" "$depth" "$top" "$generation"; then
            echo "Refusing to overwrite an invalid completed report." >&2
            return 1
        fi
        echo "Skipping completed $output"
        return
    fi
    if ! input_hashes | cmp -s "$results/input-hashes.txt" -; then
        echo "Inputs changed during the sweep; stopping." >&2
        return 1
    fi
    echo "Running $output"
    uv run --no-sync clef-rag evaluate --retrieval-only \
        --retrieval-mode "$mode" --dense-weight "$weight" --rrf-constant "$constant" \
        --candidate-depth "$depth" --top-k "$top" --k 1 3 6 10 15 20 \
        --questions eval/questions.json --index-dir index --output "$partial"
    valid_report "$partial" "$mode" "$weight" "$constant" "$depth" "$top" "$generation"
    mv "$partial" "$output"
    echo "Completed $output"
}

if [[ "$smoke" == 1 ]]; then
   run_eval hybrid 0.5 60 20 6
   run_eval bm25 0.5 60 40 10
else
   for depth in 20 40 60 80 100; do
       for top in 6 10 15 20; do
           for weight in 0.25 0.5 0.75; do
               for constant in 10 60 100; do
                   run_eval hybrid "$weight" "$constant" "$depth" "$top"
               done
            done
           run_eval dense 0.5 60 "$depth" "$top"
           run_eval bm25 0.5 60 "$depth" "$top"
        done
    done
fi

exit 0
