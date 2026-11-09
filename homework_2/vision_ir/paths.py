"""Where Part 2 keeps data, model weights, and results."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # homework_2/
DATA = ROOT / "data"
OUTPUT = ROOT / "artifacts" / "vision"


def model_cache() -> Path:
    """Use the Hugging Face cache if one is configured (e.g. on PACE scratch).

    Otherwise keep weights next to the project, like Part 1 does. Set HF_HOME
    to a scratch directory on ICE so ~7.6 GB of weights stay out of your home quota.
    """
    if "HF_HUB_CACHE" in os.environ:
        return Path(os.environ["HF_HUB_CACHE"])
    if "HF_HOME" in os.environ:
        return Path(os.environ["HF_HOME"]) / "hub"
    return ROOT / ".cache" / "models"
