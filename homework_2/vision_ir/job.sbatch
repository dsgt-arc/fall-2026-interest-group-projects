#!/bin/bash
#SBATCH -J dsgt-vision-ir                    # Job name
#SBATCH -N 1                                 # Number of nodes
#SBATCH -n 1                                 # Number of tasks
#SBATCH -c 4                                 # CPU cores (image decoding)
#SBATCH --mem=32G                            # Headroom for the largest model and the images
#SBATCH --gres=gpu:1                         # Any GPU; do not add -C (see README)
#SBATCH -t 02:00:00                          # First run also installs PyTorch and downloads 7.6 GB
#SBATCH -o vision-ir-%j.out                  # Log file in the directory you submit from
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)

# Submit from homework_2/:   sbatch vision_ir/job.sbatch
# Results: artifacts/vision/teaching.executed.ipynb plus the saved figures and CSVs.
# If the job runs out of time, resubmit: finished downloads and embeddings are reused.

set -euo pipefail # bash strict mode

# Keep the environment, uv cache, and ~7.6 GB of model weights on scratch,
# away from the 30 GB home quota.
export XDG_CACHE_HOME="$HOME/scratch/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PROJECT_ENVIRONMENT="$HOME/scratch/venvs/homework-2"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f vision_ir/teaching.ipynb ]; then
    echo "Submit this job from the homework_2/ directory: sbatch vision_ir/job.sbatch" >&2
    exit 1
fi
command -v uv >/dev/null || {
    echo "uv not found: install it first (see the README section 'Run it on PACE ICE')." >&2
    exit 1
}

uv sync --locked
uv run --no-sync python -m vision_ir.prefetch   # downloads only what is missing

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
uv run --no-sync python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else \
'PyTorch cannot use this GPU (it needs a CUDA 13 driver). Remove any -C option and resubmit.')"

# Everything is cached now; offline mode avoids Hugging Face rate limits mid-run.
export HF_HUB_OFFLINE=1

# Run every notebook cell top to bottom and save a copy with all outputs.
exec uv run --no-sync jupyter nbconvert --to notebook --execute vision_ir/teaching.ipynb \
    --output teaching.executed.ipynb --output-dir artifacts/vision \
    --ExecutePreprocessor.timeout=-1
