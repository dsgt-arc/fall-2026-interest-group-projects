#!/bin/bash
#SBATCH -J dsgt-fashion-fm                   # Job name
#SBATCH -A paceship-clef2026_img_gans        # Account name
#SBATCH -N 1                                 # Number of nodes
#SBATCH -n 1                                 # Number of tasks
#SBATCH -c 4                                 # Number of CPU cores per task
#SBATCH --gres=gpu:1                         # Number of GPUs
#SBATCH -C RTX6000                           # GPU constraint
#SBATCH -t 16:00:00                          # Time limit
#SBATCH -q inferno                           # QoS
#SBATCH -o logs/train-dsgt-fashion-fm-%j.out # Standard output log file
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=eric.regina@gatech.edu    # E-mail address to receive notifications

set -euo pipefail # bash strict mode

# consider putting these in you ~/.bashrc or ~/.zshrc
# if you plan on running the code interactively
export XDG_CACHE_HOME="$HOME/scratch/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PROJECT_ENVIRONMENT="$HOME/scratch/venvs/ImageCLEFmedGANs-2026"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

uv sync --locked

# Make sure to use exec for the job to automatically stop!
exec uv run fashion-fm train --config configs/train.yml

# to run with profiling, using the following below

# srun nsys profile \
#     --trace=cuda,cudnn,cublas,osrt \
#     --sample=cpu \
#     --output=logs/profile-dsgt-fashion-fm-%j \
#     --stats=true \
#     uv run fashion-fm train --config configs/train.yml

# exit 0 # you MUST have an exit signal for the job to automatically stop!