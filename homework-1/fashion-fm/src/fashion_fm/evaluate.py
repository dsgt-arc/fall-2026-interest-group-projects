"""Independent semantic and memorization checks for generated samples."""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from fashion_fm.data import FASHION_LABELS, fashion_mnist
from fashion_fm.flow import sample
from fashion_fm.train import load_generator, resolve_device, seed_everything


class FashionClassifier(nn.Module):
    """A compact holdout-tested classifier used only to evaluate generation."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(96 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, len(FASHION_LABELS)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        flattened_features = features.flatten(1)
        return self.classifier(flattened_features)


@torch.inference_mode()
def _classifier_accuracy(classifier: nn.Module, loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    classifier.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = classifier(images)
        predictions = logits.argmax(dim=1)
        correct_predictions = (predictions == labels).sum()
        correct += int(correct_predictions)
        total += labels.numel()
    return correct / total


def train_evaluation_classifier(
    data_dir: str | Path,
    output_path: str | Path,
    *,
    device: torch.device,
    epochs: int = 8,
    batch_size: int = 512,
    workers: int = 4,
    seed: int = 123,
) -> tuple[FashionClassifier, float]:
    """Train and holdout-test the classifier used for semantic evaluation."""

    seed_everything(seed)

    # Keep the official test split separate so it measures classifier quality.
    train_data = fashion_mnist(data_dir, train=True, download=True)
    test_data = fashion_mnist(data_dir, train=False, download=True)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )

    # Train a small classifier that is independent from the generator.
    classifier = FashionClassifier().to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    for epoch in range(epochs):
        classifier.train()
        progress = tqdm(train_loader, desc=f"classifier {epoch + 1}/{epochs}", dynamic_ncols=True)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(classifier(images), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_value = float(loss.detach())
            progress.set_postfix(loss=f"{loss_value:.4f}")

    # Measure holdout accuracy and save the classifier for future evaluations.
    accuracy = _classifier_accuracy(classifier, test_loader, device)
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": classifier.state_dict(),
        "test_accuracy": accuracy,
        "epochs": epochs,
    }
    torch.save(payload, destination)
    return classifier.eval(), accuracy


def load_or_train_classifier(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    workers: int,
) -> tuple[FashionClassifier, float]:
    """Load a prior evaluator or train one when no checkpoint exists."""

    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        # The evaluator is reusable, so only train it when it is missing.
        return train_evaluation_classifier(
            data_dir,
            path,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            workers=workers,
        )

    # Restore an existing evaluator and its recorded test accuracy.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    classifier = FashionClassifier().to(device)
    classifier.load_state_dict(payload["model"])
    return classifier.eval(), float(payload["test_accuracy"])


@torch.inference_mode()
def nearest_training_mse(
    generated: torch.Tensor,
    data_dir: str | Path,
    *,
    device: torch.device,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Return each sample's nearest raw-pixel MSE to all 60,000 train images."""

    dataset = fashion_mnist(data_dir, train=True, download=True)
    training_images = dataset.data

    # Flatten generated images once before comparing them with the training set.
    generated_flat = generated.to(device=device, dtype=torch.float32).flatten(1)
    minimum_squared_distance = torch.full((len(generated),), torch.inf, device=device)

    # Process reference images in chunks so the distance matrix fits in memory.
    for start in tqdm(range(0, len(training_images), chunk_size), desc="nearest training image", leave=False):
        reference = training_images[start : start + chunk_size].to(device=device, dtype=torch.float32)
        reference = (reference / 127.5 - 1.0).flatten(1)
        distance = torch.cdist(generated_flat, reference)
        squared_distance = distance.square()
        chunk_minimum = squared_distance.min(dim=1).values
        minimum_squared_distance = torch.minimum(minimum_squared_distance, chunk_minimum)

    pixel_count = generated_flat.shape[1]
    return (minimum_squared_distance / pixel_count).cpu()


def _within_class_diversity(images: torch.Tensor, labels: torch.Tensor) -> float:
    # Average pairwise pixel distance separately within each clothing class.
    class_means = []
    flat = images.float().flatten(1)
    for label in labels.unique():
        members = flat[labels == label]
        if len(members) > 1:
            squared_distances = torch.pdist(members).square()
            mean_squared_distances = squared_distances / members.shape[1]
            class_means.append(mean_squared_distances.mean())

    if not class_means:
        return 0.0
    return float(torch.stack(class_means).mean())


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    data_dir: str | Path = "data",
    classifier_checkpoint: str | Path | None = None,
    classifier_epochs: int = 8,
    classifier_batch_size: int = 512,
    workers: int = 4,
    num_per_class: int = 16,
    sample_steps: int = 40,
    guidance_scale: float = 2.0,
    seed: int = 2026,
    device: str = "auto",
    use_ema: bool = True,
) -> dict[str, Any]:
    """Generate held-out noise samples and report semantics, style, and novelty."""

    resolved_device = resolve_device(device)
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    # Reuse a requested classifier or keep the default beside the report.
    classifier_path = destination / "fashion-classifier.pt"
    if classifier_checkpoint is not None:
        classifier_path = Path(classifier_checkpoint).expanduser()

    classifier, classifier_test_accuracy = load_or_train_classifier(
        classifier_path,
        data_dir,
        device=resolved_device,
        epochs=classifier_epochs,
        batch_size=classifier_batch_size,
        workers=workers,
    )

    # Generate the same number of fresh samples for every FashionMNIST class.
    model = load_generator(checkpoint_path, device=resolved_device, use_ema=use_ema)
    class_ids = torch.arange(len(FASHION_LABELS), device=resolved_device)
    labels = class_ids.repeat_interleave(num_per_class)
    generated = sample(
        model,
        labels,
        steps=sample_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    # Score how often the independent classifier recognizes the requested class.
    with torch.inference_mode():
        probabilities = classifier(generated).softmax(dim=1)
        predicted = probabilities.argmax(dim=1)
        correct_conditions = predicted == labels
        condition_accuracy = float(correct_conditions.float().mean())
        target_columns = labels.unsqueeze(1)
        target_probabilities = probabilities.gather(1, target_columns)
        target_probability = float(target_probabilities.mean())

    # Compare generated pixels with the training set to detect memorization.
    nearest_mse = nearest_training_mse(generated, data_dir, device=resolved_device)

    # Break condition accuracy down by class to make weak classes visible.
    per_class_accuracy = {}
    for index, name in enumerate(FASHION_LABELS):
        class_predictions = predicted[labels == index]
        class_accuracy = (class_predictions == index).float().mean()
        per_class_accuracy[name] = float(class_accuracy)

    # Gather the remaining summary values before assembling the report.
    checkpoint = str(Path(checkpoint_path).expanduser())
    weight_name = "ema" if use_ema else "raw"
    within_class_mse = _within_class_diversity(generated.cpu(), labels.cpu())
    nearest_mse_min = float(nearest_mse.min())
    nearest_mse_mean = float(nearest_mse.mean())
    exact_copy_threshold = 1e-8
    exact_training_copies = int((nearest_mse < exact_copy_threshold).sum())
    report: dict[str, Any] = {
        "checkpoint": checkpoint,
        "weights": weight_name,
        "seed": seed,
        "samples": len(generated),
        "samples_per_class": num_per_class,
        "sample_steps": sample_steps,
        "guidance_scale": guidance_scale,
        "classifier_test_accuracy": classifier_test_accuracy,
        "condition_accuracy": condition_accuracy,
        "mean_target_probability": target_probability,
        "within_class_pairwise_mse": within_class_mse,
        "nearest_training_mse_min": nearest_mse_min,
        "nearest_training_mse_mean": nearest_mse_mean,
        "exact_training_copies": exact_training_copies,
        "per_class_accuracy": per_class_accuracy,
    }

    # Save both a visual grid and machine-readable metrics.
    display_images = (generated.cpu() + 1) / 2
    grid_path = destination / "validation-grid.png"
    save_image(display_images, grid_path, nrow=num_per_class)

    report_json = json.dumps(report, indent=2) + "\n"
    report_path = destination / "validation-report.json"
    report_path.write_text(report_json)
    return report
