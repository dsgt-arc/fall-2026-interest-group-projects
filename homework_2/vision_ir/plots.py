"""Small matplotlib helpers so the notebook can stay focused on the results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .data import DOMAINS, Photo

# Okabe–Ito colors, distinguishable with common color-vision deficiencies.
MODEL_COLORS = {
    "ViT": "#E69F00", "ConvNeXt": "#D55E00", "DINOv2": "#0072B2",
    "I-JEPA": "#009E73", "V-JEPA 2": "#56B4E9", "DiT": "#CC79A7",
}
MODEL_MARKERS = {"ViT": "o", "ConvNeXt": "s", "DINOv2": "D", "I-JEPA": "^", "V-JEPA 2": "v", "DiT": "P"}


def thumbnail(path, size: int = 160, border: str | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.fit(image.convert("RGB"), (size, size))
    if border:
        image = ImageOps.expand(image.crop((6, 6, size - 6, size - 6)), border=6, fill=border)
    return np.asarray(image)


def show_domains(photos: list[Photo], groups_per_domain: int = 2, per_group: int = 3):
    """One row per domain: a few images from each of its first groups."""
    columns = groups_per_domain * per_group
    figure, axes = plt.subplots(len(DOMAINS), columns, figsize=(1.9 * columns, 2.0 * len(DOMAINS)))
    for row, domain in zip(axes, DOMAINS, strict=True):
        groups = sorted({p.group for p in photos if p.domain == domain.name})[:groups_per_domain]
        chosen = [p for g in groups for p in [q for q in photos if q.group == g][:per_group]]
        for column, (axis, photo) in enumerate(zip(row, chosen, strict=True)):
            axis.imshow(thumbnail(photo.path))
            axis.set_xticks([]), axis.set_yticks([])
            axis.set_title(f"group {photo.group}", fontsize=8)
            if column == 0:
                axis.set_ylabel(f"{domain.name}\n({domain.kind})", fontsize=10)
    figure.suptitle("Each row is one domain; images with the same group number show the same thing")
    figure.tight_layout()
    return figure


def heatmap(table: pd.DataFrame, title: str, *, cmap: str = "viridis", fmt: str = "{:.2f}"):
    """Annotated grid of scores between 0 and 1."""
    figure, axis = plt.subplots(figsize=(1.1 * table.shape[1] + 2.5, 0.55 * table.shape[0] + 1.5))
    image = axis.imshow(table.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(table.shape[1]), table.columns, rotation=30, ha="right")
    axis.set_yticks(range(table.shape[0]), table.index)
    for (row, column), value in np.ndenumerate(table.to_numpy(dtype=float)):
        axis.text(column, row, fmt.format(value), ha="center", va="center",
                  color="black" if value > 0.6 else "white", fontsize=9)
    figure.colorbar(image, ax=axis, fraction=0.03)
    axis.set_title(title)
    figure.tight_layout()
    return figure


def show_rankings(query: Photo, ranked: dict[str, list[Photo]], title: str):
    """Query on the left; each row shows one model's top results (green = same group)."""
    k = len(next(iter(ranked.values())))
    figure, axes = plt.subplots(len(ranked), k + 1, figsize=(1.6 * (k + 1), 1.7 * len(ranked)))
    for row, (model, results) in zip(axes, ranked.items(), strict=True):
        row[0].imshow(thumbnail(query.path, border="#444444"))
        row[0].set_ylabel(model, rotation=0, ha="right", va="center", fontsize=10)
        for axis, photo in zip(row[1:], results, strict=True):
            color = "#009E73" if photo.group == query.group else "#D55E00"
            axis.imshow(thumbnail(photo.path, border=color))
        for axis in row:
            axis.set_xticks([]), axis.set_yticks([])
    axes[0][0].set_title("query", fontsize=9)
    for rank, axis in enumerate(axes[0][1:], start=1):
        axis.set_title(f"#{rank}", fontsize=9)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def layer_curves(table: pd.DataFrame, title: str):
    """mAP of mean-pooled tokens against relative depth (0 = input, 1 = last block)."""
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for model, rows in table.groupby("model", sort=False):
        axis.plot(rows["depth"], rows["mAP"], marker=MODEL_MARKERS[model],
                  color=MODEL_COLORS[model], label=model)
    axis.set(xlabel="relative depth (block ÷ number of blocks)", ylabel="Task 1 mAP",
             xlim=(0, 1.02), ylim=(0, 1), title=title)
    axis.grid(alpha=0.3)
    axis.legend(loc="upper left", fontsize=9)
    figure.tight_layout()
    return figure
