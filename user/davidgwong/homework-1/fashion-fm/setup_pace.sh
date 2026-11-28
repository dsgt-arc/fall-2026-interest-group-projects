#!/bin/bash
set -euo pipefail

# Create required local directories
mkdir -p logs outputs/fashion-fm data

ln -sfn pyproject.cuda.toml pyproject.toml

export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="$HOME/scratch/cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PROJECT_ENVIRONMENT="$HOME/scratch/venvs/ImageCLEFmedGANs-2026"

uv sync --locked
uv run python -c "from torchvision.datasets import FashionMNIST; FashionMNIST('data', train=True, download=True); FashionMNIST('data', train=False, download=True)"

echo "PACE setup and data download complete."