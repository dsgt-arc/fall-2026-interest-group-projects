"""Download the image subset and all six models once (about 7.6 GB of weights).

    uv run python -m vision_ir.prefetch

Safe to rerun: finished files are skipped. Useful on PACE ICE, where you may
want to download on a login node before a GPU job, and before working offline
(set HF_HUB_OFFLINE=1 afterwards).
"""

from huggingface_hub import snapshot_download

from .data import prepare_gpr1200
from .models import BACKBONES
from .paths import DATA, model_cache


def main() -> None:
    root = prepare_gpr1200(DATA)
    print(f"Images ready: {root}")
    cache = model_cache()
    for key, backbone in BACKBONES.items():
        print(f"Fetching {key} ({backbone.repo}) into {cache} ...")
        snapshot_download(
            backbone.repo, revision=backbone.revision, cache_dir=cache,
            allow_patterns=list(backbone.files),
        )
    print("All models are downloaded.")


if __name__ == "__main__":
    main()
