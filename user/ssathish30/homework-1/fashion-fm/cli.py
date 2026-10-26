"""Train a flow-matching model and generate FashionMNIST images."""

import json
from pathlib import Path

import click
import torch
from torchvision.utils import save_image

from fashion_fm.data import FASHION_LABELS, description_to_label
from fashion_fm.evaluate import evaluate_checkpoint
from fashion_fm.flow import sample
from fashion_fm.train import load_generator, load_train_config, train

CHECKPOINT = "outputs/fashion-fm/latest.pt"
DEFAULT_TRAIN_CONFIG = Path("configs/train.yml")
EVAL_DIR = "outputs/evaluation"
GENERATED_GRID = "outputs/generated.png"
IMAGES_PER_PROMPT = 4


@click.group(help=__doc__)
def main() -> None:
    """Run the Fashion FM demo."""


@main.command("train", help="Train on FashionMNIST.")
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_TRAIN_CONFIG,
    show_default=True,
)
def train_command(config: Path) -> None:
    """Train the flow-matching model."""

    training_config = load_train_config(config)
    destination = train(training_config)
    click.echo(f"checkpoint: {destination}")


@main.command("generate", help="Generate images from short descriptions.")
@click.option("--checkpoint", default=CHECKPOINT, show_default=True)
@click.option("--prompt", multiple=True, help="Repeat for multiple descriptions; defaults to all classes.")
def generate_command(checkpoint: str, prompt: tuple[str, ...]) -> None:
    """Generate images with the trained model."""

    # Load the trained model and place new tensors on the same device.
    model = load_generator(checkpoint)
    device = next(model.parameters()).device

    # Use every clothing class when the user does not provide a prompt.
    prompts = prompt or tuple(FASHION_LABELS)
    expanded_prompts = []
    for description in prompts:
        expanded_prompts.extend([description] * IMAGES_PER_PROMPT)

    # Convert the plain-English descriptions into FashionMNIST class IDs.
    label_ids = [description_to_label(description) for description in expanded_prompts]
    labels = torch.tensor(label_ids, device=device)
    generated = sample(model, labels, seed=0)

    # Save one grid that makes the generated classes easy to compare.
    output = Path(GENERATED_GRID)
    output.parent.mkdir(parents=True, exist_ok=True)
    images = (generated.cpu() + 1.0) / 2.0
    save_image(images, output, nrow=IMAGES_PER_PROMPT)

    # Also save each image separately and record which prompt produced it.
    individual_dir = output.with_suffix("")
    individual_dir.mkdir(parents=True, exist_ok=True)
    records = []
    generated_items = zip(images, expanded_prompts, label_ids, strict=True)
    for index, (image, description, label) in enumerate(generated_items):
        class_name = FASHION_LABELS[label]
        safe_class_name = class_name.replace("/", "-")
        image_name = f"{index:04d}-{safe_class_name}.png"
        image_path = individual_dir / image_name
        save_image(image, image_path)
        records.append(
            {
                "index": index,
                "prompt": description,
                "label": label,
                "class": class_name,
                "file": str(image_path),
            }
        )

    # Write the metadata next to the grid so the outputs remain self-describing.
    manifest = json.dumps(records, indent=2) + "\n"
    output.with_suffix(".json").write_text(manifest)
    click.echo(f"generated {len(images)} images: {output}")


@main.command("evaluate", help="Score a checkpoint's samples for class fidelity and novelty.")
@click.option("--checkpoint", default=CHECKPOINT, show_default=True)
@click.option("--output-dir", default=EVAL_DIR, show_default=True)
@click.option("--data-dir", default="data", show_default=True)
@click.option("--classifier", default=None, help="Reuse a judge classifier so runs stay comparable.")
@click.option("--num-per-class", default=16, show_default=True, help="Samples generated per class.")
@click.option("--sample-steps", default=40, show_default=True, help="ODE steps per sample.")
@click.option("--guidance-scale", default=2.0, show_default=True, help="Classifier-free guidance strength.")
@click.option("--seed", default=2026, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--raw-weights", is_flag=True, help="Score raw weights instead of the EMA copy.")
def evaluate_command(
    checkpoint: str,
    output_dir: str,
    data_dir: str,
    classifier: str | None,
    num_per_class: int,
    sample_steps: int,
    guidance_scale: float,
    seed: int,
    device: str,
    raw_weights: bool,
) -> None:
    """Report how well generated samples match their requested class."""

    report = evaluate_checkpoint(
        checkpoint,
        output_dir,
        data_dir=data_dir,
        classifier_checkpoint=classifier,
        num_per_class=num_per_class,
        sample_steps=sample_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        device=device,
        use_ema=not raw_weights,
    )

    # Show the headline numbers; the full report stays on disk as JSON.
    click.echo(f"weights: {report['weights']}  samples: {report['samples']}")
    click.echo(f"classifier test accuracy: {report['classifier_test_accuracy']:.4f}")
    click.echo(f"condition accuracy:       {report['condition_accuracy']:.4f}")
    click.echo(f"mean target probability:  {report['mean_target_probability']:.4f}")
    click.echo(f"within-class pairwise MSE: {report['within_class_pairwise_mse']:.4f}")
    click.echo(f"nearest training MSE:      min {report['nearest_training_mse_min']:.4f}, "
               f"mean {report['nearest_training_mse_mean']:.4f}")
    click.echo(f"exact training copies:     {report['exact_training_copies']}")
    click.echo("\nper-class accuracy:")
    for name, accuracy in report["per_class_accuracy"].items():
        click.echo(f"  {name:<12} {accuracy:.4f}")
    click.echo(f"\nreport: {Path(output_dir) / 'validation-report.json'}")
