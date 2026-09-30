"""Accelerator/CPU training and checkpoint utilities."""

import random
import warnings
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

from fashion_fm.data import FASHION_LABELS, fashion_mnist
from fashion_fm.flow import make_flow_batch, sample
from fashion_fm.metrics import TrainingMetrics
from fashion_fm.model import TinyConditionalUNet


class TrainConfig(BaseModel):
    """Validated hyperparameters and runtime options for one training run."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    data_dir: str = Field(default="data", min_length=1)
    output_dir: str = Field(default="outputs/fashion-fm", min_length=1)
    epochs: int = Field(default=50, gt=0)
    batch_size: int = Field(default=256, gt=0)
    workers: int = Field(default=4, ge=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    cosine_min_lr: float = Field(default=1e-5, ge=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    cfg_dropout: float = Field(default=0.1, ge=0, le=1)
    ema_decay: float = Field(default=0.99, ge=0, le=1)
    timesteps: int = Field(default=1000, ge=2)
    seed: int = 42
    max_train_samples: int | None = Field(default=None, gt=0)
    sample_steps: int = Field(default=30, gt=0)
    save_every: int = Field(default=5, gt=0)
    mixed_precision: bool = True
    data_parallel: bool = False
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    resume: bool = True

    @model_validator(mode="after")
    def validate_cosine_floor(self) -> Self:
        """Keep the scheduler floor at or below its starting rate."""

        if self.cosine_min_lr > self.learning_rate:
            raise ValueError("cosine_min_lr must not exceed learning_rate")
        return self


class TrainingConfigFile(BaseModel):
    """Typed structure of the checked-in YAML document."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    training: TrainConfig


def load_train_config(path: str | Path) -> TrainConfig:
    """Load training settings from a checked-in YAML file."""

    config_path = Path(path).expanduser()
    with config_path.open(encoding="utf-8") as config_file:
        document = yaml.safe_load(config_file)

    validated_document = TrainingConfigFile.model_validate(document)
    return validated_document.training


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a device, preferring CUDA/ROCm, then MPS, then CPU for auto."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested, but torch.cuda.is_available() is false")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS device requested, but torch.backends.mps.is_available() is false")
    return device


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """Capture random number generators needed for an exact epoch-boundary resume."""

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore random number generators from a training checkpoint."""

    if not state:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _checkpoint_payload(
    model: TinyConditionalUNet,
    ema_model: AveragedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainConfig,
    epoch: int,
    global_step: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "ema_model": ema_model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config.model_dump(mode="python"),
        "architecture": model.metadata(),
        "labels": list(FASHION_LABELS),
        "rng_state": capture_rng_state(),
    }


def _restore_training_state(
    checkpoint: dict[str, Any],
    model: TinyConditionalUNet,
    ema_model: AveragedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, int]:
    """Restore a completed epoch and return the next epoch and global step."""

    model.load_state_dict(checkpoint["model"])
    ema_state = checkpoint.get("ema_model", checkpoint["model"])
    ema_model.module.load_state_dict(ema_state)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng_state(checkpoint.get("rng_state"))

    start_epoch = int(checkpoint["epoch"])
    global_step = int(checkpoint["global_step"])
    return start_epoch, global_step


def save_checkpoint(payload: dict[str, Any], output_dir: Path, name: str) -> Path:
    """Atomically save a training checkpoint."""

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / name
    temporary = output_dir / f".{name}.tmp"

    # Write a temporary file first so an interruption cannot corrupt the checkpoint.
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_generator(
    checkpoint_path: str | Path,
    device: str | torch.device = "auto",
    *,
    use_ema: bool = True,
) -> TinyConditionalUNet:
    """Load EMA or raw weights from a training checkpoint for generation."""

    if isinstance(device, str):
        resolved_device = resolve_device(device)
    else:
        resolved_device = device

    source = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    architecture = checkpoint.get("architecture", {})

    # Recreate the architecture metadata stored with the checkpoint.
    sample_size = int(architecture.get("sample_size", 28))
    timesteps = int(architecture.get("timesteps", 1000))
    model = TinyConditionalUNet(
        sample_size=sample_size,
        timesteps=timesteps,
    )

    # EMA weights are smoother during inference, but older checkpoints may not have them.
    state = checkpoint["model"]
    if use_ema:
        state = checkpoint.get("ema_model", state)

    model.load_state_dict(state)
    model.to(resolved_device).eval()
    return model


def export_generator(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    use_ema: bool = True,
) -> Path:
    """Export selected generator weights without optimizer/training state."""

    source = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)

    # Select the small set of weights needed for inference.
    selected_weights = checkpoint["model"]
    weight_name = "raw"
    if use_ema:
        selected_weights = checkpoint.get("ema_model", selected_weights)
        weight_name = "ema"

    payload = {
        "format_version": 1,
        "model": selected_weights,
        "ema_model": selected_weights,
        "architecture": checkpoint.get("architecture", {}),
        "labels": checkpoint.get("labels", list(FASHION_LABELS)),
        "source": {
            "checkpoint": str(source),
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "weights": weight_name,
        },
    }

    # Use the same temporary-file pattern as full training checkpoints.
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def train(
    config: TrainConfig,
    *,
    device: str | None = None,
    resume: bool | None = None,
) -> Path:
    """Train the conditional flow model and return the final checkpoint path."""

    requested_device = config.device if device is None else device
    should_resume = config.resume if resume is None else resume

    # Prepare reproducible output and record the exact training configuration.
    seed_everything(config.seed)
    resolved_device = resolve_device(requested_device)
    if resolved_device.type == "cpu":
        warnings.warn(
            "Training is using the CPU, which will be considerably slower than CUDA/ROCm or MPS.",
            RuntimeWarning,
            stacklevel=2,
        )
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_json = config.model_dump_json(indent=2) + "\n"
    (output_dir / "config.json").write_text(config_json)

    # Download FashionMNIST and optionally select a reproducible subset.
    dataset = fashion_mnist(config.data_dir, train=True, download=True)
    if config.max_train_samples is not None:
        generator = torch.Generator().manual_seed(config.seed)
        shuffled_indices = torch.randperm(len(dataset), generator=generator)
        indices = shuffled_indices[: config.max_train_samples]
        dataset = Subset(dataset, indices.tolist())

    # Feed shuffled batches efficiently to either the CPU or GPU.
    using_cuda = resolved_device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=using_cuda,
        persistent_workers=config.workers > 0,
        drop_last=False,
    )

    # Build the model, with optional replication across visible GPUs.
    model = TinyConditionalUNet(timesteps=config.timesteps).to(resolved_device)
    training_model: torch.nn.Module = model
    if config.data_parallel:
        visible_gpu_count = torch.cuda.device_count()
        if not using_cuda or visible_gpu_count < 2:
            raise RuntimeError("data_parallel requires at least two visible CUDA/ROCm GPUs")
        training_model = torch.nn.DataParallel(model)

    # Track smoothed weights for generation and decay the learning rate over the run.
    ema_update = get_ema_multi_avg_fn(config.ema_decay)
    ema_model = AveragedModel(model, multi_avg_fn=ema_update, use_buffers=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = config.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps),
        eta_min=config.cosine_min_lr,
    )

    # Continue from the latest complete epoch when a checkpoint is available.
    start_epoch = 0
    global_step = 0
    latest = output_dir / "latest.pt"
    if should_resume and latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        start_epoch, global_step = _restore_training_state(
            checkpoint,
            model,
            ema_model,
            optimizer,
            scheduler,
        )

    # Mixed precision speeds up supported GPUs while CPU training stays full precision.
    autocast_enabled = config.mixed_precision and using_cuda
    autocast_dtype = torch.bfloat16
    metrics = TrainingMetrics(output_dir)

    # Train one epoch at a time and report a running loss in the progress bar.
    for epoch in range(start_epoch, config.epochs):
        completed_epoch = epoch + 1
        metrics.start_epoch(completed_epoch)
        model.train()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"epoch {completed_epoch}/{config.epochs}", dynamic_ncols=True)
        for images, labels in progress:
            images = images.to(resolved_device, non_blocking=True)
            labels = labels.to(resolved_device, non_blocking=True)
            x_t, t, target, conditioned_labels = make_flow_batch(
                images,
                labels,
                cfg_dropout=config.cfg_dropout,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=resolved_device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                prediction = training_model(x_t, t, conditioned_labels)
                loss = F.mse_loss(prediction.float(), target.float())
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            ema_model.update_parameters(model)
            global_step += 1
            loss_value = float(loss.detach())
            running_loss += loss_value

            # Record the same core diagnostics used by the larger GAN project.
            learning_rate = scheduler.get_last_lr()[0]
            metrics.record_step(
                loss=loss_value,
                learning_rate=learning_rate,
                grad_norm=float(grad_norm),
            )
            completed_batches = progress.n + 1
            mean_loss = running_loss / completed_batches
            progress.set_postfix(loss=f"{mean_loss:.4f}")

        # Save one durable CSV row and refresh the training plot.
        metrics.finish_epoch(global_step=global_step)

        # Always refresh latest.pt and periodically retain a numbered checkpoint.
        payload = _checkpoint_payload(model, ema_model, optimizer, scheduler, config, completed_epoch, global_step)
        save_checkpoint(payload, output_dir, "latest.pt")
        is_save_interval = completed_epoch % config.save_every == 0
        is_final_epoch = completed_epoch == config.epochs
        if is_save_interval or is_final_epoch:
            save_checkpoint(payload, output_dir, f"epoch-{completed_epoch:04d}.pt")

        # One fixed sample per class makes regressions visible during training.
        ema_model.module.eval()
        monitor_labels = torch.arange(len(FASHION_LABELS), device=resolved_device)
        monitor = sample(
            ema_model.module,
            monitor_labels,
            steps=config.sample_steps,
            seed=config.seed,
        )
        monitor_images = (monitor.cpu() + 1) / 2
        monitor_path = output_dir / f"samples-{completed_epoch:04d}.png"
        save_image(monitor_images, monitor_path, nrow=5)

    # Rebuild the final plot from the durable CSV, even when no new epochs ran.
    if metrics.csv_path.exists():
        plot_path = metrics.plot()
        print(f"training metrics plot: {plot_path}")

    return latest
