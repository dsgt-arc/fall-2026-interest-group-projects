"""Record compact, resumable training metrics and plots."""

import csv
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

# Training may run without a display, so plots always use a file-only backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics summarized over one completed training epoch."""

    epoch: int
    global_step: int
    loss: float
    loss_std: float
    loss_min: float
    loss_max: float
    learning_rate: float
    grad_norm: float
    epoch_time_sec: float


class TrainingMetrics:
    """Accumulate step metrics and persist one row per completed epoch."""

    COLUMNS = [
        "epoch",
        "timestamp",
        "global_step",
        "loss",
        "loss_std",
        "loss_min",
        "loss_max",
        "learning_rate",
        "grad_norm",
        "epoch_time_sec",
    ]

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.csv_path = self.output_dir / "train_metrics.csv"
        self.plot_path = self.output_dir / "training_metrics.png"

        self.epoch = 0
        self.epoch_started_at = 0.0
        self.step_count = 0
        self.loss_mean = 0.0
        self.loss_m2 = 0.0
        self.loss_min = 0.0
        self.loss_max = 0.0
        self.learning_rate_sum = 0.0
        self.grad_norm_sum = 0.0

    def start_epoch(self, epoch: int) -> None:
        """Reset the running values for a new epoch."""

        self.epoch = epoch
        self.epoch_started_at = time.perf_counter()
        self.step_count = 0
        self.loss_mean = 0.0
        self.loss_m2 = 0.0
        self.loss_min = 0.0
        self.loss_max = 0.0
        self.learning_rate_sum = 0.0
        self.grad_norm_sum = 0.0

    def record_step(self, *, loss: float, learning_rate: float, grad_norm: float) -> None:
        """Add one optimizer step to the current epoch summary."""

        self.step_count += 1

        # Welford's update tracks loss variance without storing every batch loss.
        if self.step_count == 1:
            self.loss_mean = loss
            self.loss_min = loss
            self.loss_max = loss
        else:
            difference = loss - self.loss_mean
            self.loss_mean += difference / self.step_count
            self.loss_m2 += difference * (loss - self.loss_mean)
            self.loss_min = min(self.loss_min, loss)
            self.loss_max = max(self.loss_max, loss)

        self.learning_rate_sum += learning_rate
        self.grad_norm_sum += grad_norm

    def finish_epoch(self, *, global_step: int) -> EpochMetrics:
        """Persist and plot the metrics for the current completed epoch."""

        if self.step_count == 0:
            raise ValueError("cannot finish metrics for an epoch with no steps")

        loss_variance = 0.0
        if self.step_count > 1:
            loss_variance = self.loss_m2 / (self.step_count - 1)

        metrics = EpochMetrics(
            epoch=self.epoch,
            global_step=global_step,
            loss=self.loss_mean,
            loss_std=loss_variance**0.5,
            loss_min=self.loss_min,
            loss_max=self.loss_max,
            learning_rate=self.learning_rate_sum / self.step_count,
            grad_norm=self.grad_norm_sum / self.step_count,
            epoch_time_sec=time.perf_counter() - self.epoch_started_at,
        )

        # Write metrics before checkpointing. Repeated epochs replace their old row.
        self._write_metrics(metrics)
        self.plot()
        return metrics

    def _read_rows(self) -> list[dict[str, str]]:
        """Read all completed epoch rows from disk."""

        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", encoding="utf-8", newline="") as metrics_file:
            return list(csv.DictReader(metrics_file))

    def _write_metrics(self, metrics: EpochMetrics) -> None:
        """Atomically insert or replace one epoch in the metrics CSV."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows = self._read_rows()
        rows = [row for row in rows if int(row["epoch"]) != metrics.epoch]

        row = asdict(metrics)
        row["timestamp"] = datetime.now(timezone.utc).isoformat()
        rows.append({key: str(value) for key, value in row.items()})
        rows.sort(key=lambda item: int(item["epoch"]))

        temporary_path = self.csv_path.with_suffix(".csv.tmp")
        with temporary_path.open("w", encoding="utf-8", newline="") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=self.COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(self.csv_path)

    def plot(self) -> Path:
        """Generate an inspectable training plot directly from the metrics CSV."""

        rows = self._read_rows()
        if not rows:
            raise ValueError(f"cannot plot an empty metrics file: {self.csv_path}")

        epochs = [int(row["epoch"]) for row in rows]
        losses = [float(row["loss"]) for row in rows]
        learning_rates = [float(row["learning_rate"]) for row in rows]
        grad_norms = [float(row["grad_norm"]) for row in rows]

        figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        plot_specs = (
            (axes[0], losses, "Flow-matching loss"),
            (axes[1], learning_rates, "Learning rate"),
            (axes[2], grad_norms, "Gradient norm"),
        )
        for axis, values, label in plot_specs:
            axis.plot(epochs, values, marker="o", linewidth=1.5)
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)

        axes[2].set_xlabel("Completed epoch")
        figure.suptitle("Training metrics")
        figure.tight_layout()

        temporary_path = self.plot_path.with_name(f".{self.plot_path.name}.tmp")
        figure.savefig(temporary_path, format="png", dpi=150)
        plt.close(figure)
        temporary_path.replace(self.plot_path)
        return self.plot_path
