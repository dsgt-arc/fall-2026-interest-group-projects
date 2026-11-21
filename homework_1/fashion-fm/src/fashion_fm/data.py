"""FashionMNIST data loading and short-description parsing."""

import re
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST

FASHION_LABELS: tuple[str, ...] = (
    "t-shirt/top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
)

# This is intentionally a tiny, auditable text interface rather than a large
# language encoder. Longer aliases are checked first to avoid matching "boot"
# before "ankle boot" or "shirt" inside "t-shirt".
_ALIASES: tuple[tuple[str, int], ...] = (
    ("t shirt top", 0),
    ("tee shirt", 0),
    ("t shirt", 0),
    ("tee", 0),
    ("trousers", 1),
    ("trouser", 1),
    ("pants", 1),
    ("pullover", 2),
    ("sweater", 2),
    ("jumper", 2),
    ("dress", 3),
    ("coat", 4),
    ("jacket", 4),
    ("sandal", 5),
    ("sandals", 5),
    ("shirt", 6),
    ("sneaker", 7),
    ("sneakers", 7),
    ("trainer", 7),
    ("trainers", 7),
    ("bag", 8),
    ("handbag", 8),
    ("purse", 8),
    ("ankle boots", 9),
    ("ankle boot", 9),
    ("boot", 9),
    ("boots", 9),
)


def normalize_description(description: str) -> str:
    """Normalize punctuation and whitespace in a short garment description."""

    lowercase = description.lower()
    without_punctuation = re.sub(r"[^a-z0-9]+", " ", lowercase)
    return " ".join(without_punctuation.split())


def description_to_label(description: str) -> int:
    """Map a short description such as ``"a sneaker"`` to a FashionMNIST class."""

    normalized = normalize_description(description)
    if not normalized:
        raise ValueError("description must not be empty")

    # Prefer exact FashionMNIST class names before considering aliases.
    exact_names = {normalize_description(name): label for label, name in enumerate(FASHION_LABELS)}
    if normalized in exact_names:
        return exact_names[normalized]

    # Surround the prompt with spaces so short aliases only match whole words.
    padded = f" {normalized} "
    matches = [(alias, label) for alias, label in _ALIASES if f" {alias} " in padded]
    alias_lengths = (len(alias.split()) for alias, _ in matches)
    longest_match = max(alias_lengths, default=0)
    labels = {label for alias, label in matches if len(alias.split()) == longest_match}

    # A single longest match gives an unambiguous class.
    if len(labels) == 1:
        return labels.pop()
    if len(labels) > 1:
        names = ", ".join(FASHION_LABELS[label] for label in sorted(labels))
        raise ValueError(f"ambiguous description {description!r}; matched: {names}")

    choices = ", ".join(FASHION_LABELS)
    raise ValueError(f"unknown description {description!r}; use one of: {choices}")


def fashion_mnist(
    root: str | Path = "data",
    *,
    train: bool = True,
    download: bool = True,
) -> Dataset:
    """Build the normalized grayscale training or test dataset."""

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    data_root = Path(root).expanduser()
    return FashionMNIST(root=data_root, train=train, transform=transform, download=download)
