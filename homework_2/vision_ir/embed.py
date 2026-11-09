"""Embed a list of images once per backbone and cache the vectors on disk."""

import gc
import hashlib
import json
import time
import zipfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .augment import augment
from .models import BACKBONES, encoder_options, load_encoder

# Increase when preprocessing, photo edits, or pooling change, so old caches are ignored.
CACHE_VERSION = 2


@dataclass(frozen=True)
class ImageItem:
    """One image to embed: a file on disk, optionally altered by a named edit."""

    item_id: str
    path: Path
    augmentation: str | None = None

    def load(self) -> Image.Image:
        with Image.open(self.path) as image:
            image = image.convert("RGB")
        return augment(image, self.augmentation) if self.augmentation else image


@dataclass(frozen=True)
class Embeddings:
    ids: list[str]
    final: np.ndarray            # (images, dim) retrieval vectors
    layers: dict[int, np.ndarray]  # block number -> (images, dim_at_that_block)
    seconds: float = float("nan")  # time to embed every image (model already loaded)
    device: str = ""               # where that time was measured

    @cached_property
    def _position(self) -> dict[str, int]:
        return {item_id: i for i, item_id in enumerate(self.ids)}

    def rows(self, ids: list[str]) -> np.ndarray:
        """Return final vectors in the order of `ids`."""
        return self.final[[self._position[i] for i in ids]]

    def layer_rows(self, block: int, ids: list[str]) -> np.ndarray:
        """Return one block's mean-pooled features in the order of `ids`."""
        return self.layers[block][[self._position[i] for i in ids]]


def _computation(backbone) -> list:
    """The Backbone fields that change the vectors (not names or descriptions)."""
    return [backbone.repo, backbone.revision, backbone.image_size, backbone.mean,
            backbone.std, backbone.pooling, backbone.frames]


def _fingerprint(key: str, items: list[ImageItem], options: dict) -> str:
    payload = json.dumps({
        "version": CACHE_VERSION,
        "backbone": _computation(BACKBONES[key]),  # changing how a model computes invalidates
        "options": options,
        "items": [(item.item_id, item.augmentation) for item in items],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _compact(vectors: np.ndarray) -> np.ndarray:
    """Half precision halves the disk space for per-layer features, unless it would overflow."""
    return vectors.astype(np.float16 if np.abs(vectors).max() < 6e4 else np.float32)


def _read_cache(path: Path, fingerprint: str) -> Embeddings | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["fingerprint"]) != fingerprint:
                return None
            layers = {
                int(name.removeprefix("layer_")): stored[name]
                for name in stored.files if name.startswith("layer_")
            }
            return Embeddings(
                stored["ids"].tolist(), stored["final"], layers,
                seconds=float(stored["seconds"]), device=str(stored["device"]),
            )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None  # unreadable cache: recompute it


def embed(
    key: str, items: list[ImageItem], *, cache_dir: Path, output_dir: Path,
    device: torch.device, batch_size: int = 16, **options,
) -> Embeddings:
    """Return cached embeddings when the image list and options match; else compute them.

    `options` are passed to the encoder (e.g. DiT's `block` and `timestep`) and
    become part of the cache name, so different settings never overwrite each other.
    """
    if len({item.item_id for item in items}) != len(items):
        raise ValueError("Image ids must be unique.")
    suffix = "".join(f"_{name}{value}" for name, value in sorted(options.items()))
    path = output_dir / f"{key.replace(' ', '')}{suffix}.npz"
    fingerprint = _fingerprint(key, items, encoder_options(key, options))
    cached = _read_cache(path, fingerprint)
    if cached is not None:
        return cached

    encoder = load_encoder(key, cache_dir, device, **options)
    start_time = time.perf_counter()  # time the embedding only, not reading weights from disk
    finals, layers = [], {}
    for start in tqdm(range(0, len(items), batch_size), desc=f"Embedding with {key}"):
        out = encoder([item.load() for item in items[start : start + batch_size]])
        finals.append(out["final"].float().cpu().numpy())
        for block, vectors in out["layers"].items():
            layers.setdefault(block, []).append(vectors.float().cpu().numpy())
    result = Embeddings(
        ids=[item.item_id for item in items],
        final=np.concatenate(finals).astype(np.float32),
        layers={block: _compact(np.concatenate(parts)) for block, parts in layers.items()},
        seconds=time.perf_counter() - start_time,
        device=device.type,
    )
    if not all(np.isfinite(v).all() for v in (result.final, *result.layers.values())):
        raise RuntimeError(f"{key} produced NaN or infinite embeddings.")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary, fingerprint=np.array(fingerprint), ids=np.array(result.ids),
        seconds=np.array(result.seconds), device=np.array(result.device),
        final=result.final, **{f"layer_{block}": v for block, v in result.layers.items()},
    )
    temporary.replace(path)  # never leave a half-written cache behind
    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return result
