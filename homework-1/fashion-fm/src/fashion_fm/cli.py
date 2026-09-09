"""Train a flow-matching model and generate FashionMNIST images."""

import json
from pathlib import Path

import click
import torch
from torchvision.utils import save_image

from fashion_fm.data import FASHION_LABELS, description_to_label
from fashion_fm.flow import sample
from fashion_fm.train import load_generator, load_train_config, train

CHECKPOINT = "outputs/fashion-fm/latest.pt"
DEFAULT_TRAIN_CONFIG = Path("configs/train.yml")
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
