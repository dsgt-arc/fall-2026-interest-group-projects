# fashion-fm

`fashion-fm` is a very small, text-conditioned flow-matching generator for the
28×28 grayscale [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
dataset. It generates new images from short descriptions such as `"a sneaker"`,
`"ankle boot"`, or `"small handbag"`.

The text interface is deliberately small: descriptions are mapped to the ten
FashionMNIST categories and passed through a learned class embedding. This avoids
making a language encoder larger than the image generator itself.

## Architecture

The generator is adapted directly from `unet-tiny` in
`~/imageclef-med-gans-2026/clef_med/models/unet.py`:

| Setting | Reference `unet-tiny` | `fashion-fm` |
| --- | --- | --- |
| U-Net implementation | Diffusers `UNet2DModel` | same |
| Channels | 16, 32, 64, 64 | same |
| Layers per block | 2 | same |
| Down blocks | 3 plain + 1 attention | same |
| Up blocks | 1 attention + 3 plain | same |
| GroupNorm groups | 16 | same |
| Time scale/shift | `scale_shift` | same |
| Input/output | one grayscale channel | same |
| Conditioning | time | time + 11-entry class table |

The extra embedding holds ten garment classes and one null class for
classifier-free guidance. Training uses the same noise-to-image flow-matching
formulation as the reference project, while a fixed-step Heun solver keeps
sampling self-contained. The wrapper pads 28×28 tensors to 32×32 only while
inside the four-level U-Net, then crops back to native FashionMNIST resolution;
this keeps all three downsample/upsample skip shapes exact.

## Install

For CPU, Apple MPS, or a standard PyTorch environment:

```bash
uv sync
```

Platform-specific project files can be selected with the same relative-symlink
workflow as `~/imageclef-med-gans-2026/`. From the repository root, select the
NVIDIA CUDA 12.x configuration with:

```bash
ln -sfn pyproject.cuda.toml pyproject.toml
uv sync
uv run python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'
```

For Linux AMD GPUs, select the ROCm configuration, which uses the official ROCm
7.2 index:

```bash
ln -sfn pyproject.rocm.toml pyproject.toml
uv sync --extra rocm
uv run python -c 'import torch; print(torch.cuda.is_available(), torch.version.hip)'
```

ROCm intentionally appears as a `cuda` device in the PyTorch API. Automatic
device selection prefers CUDA/ROCm, then Apple MPS, then CPU; training emits a
warning when it falls back to CPU. Both variant files contain the complete
project metadata, dependencies, CLI entry point, and development-tool settings,
so either can safely serve as `pyproject.toml`.

## Train

The first run downloads and normalizes the official 60,000-image FashionMNIST
training split to `[-1, 1]`. All hyperparameters and runtime settings live in
[`configs/train.yml`](configs/train.yml), so experiments can be reviewed and
versioned with the code. Pydantic validates field types, ranges, required
sections, and unknown options before training begins:

```bash
uv run fashion-fm train
```

To use a different checked-in experiment configuration:

```bash
uv run fashion-fm train --config configs/train.yml
```

Every completed epoch atomically refreshes `latest.pt`, so an interrupted run
resumes at the next epoch with its model, EMA, optimizer, scheduler, random
number generators, and global step restored. Numbered checkpoints are retained
at the configured `save_every` interval and at the final epoch.

Training also appends one summary per completed epoch to `train_metrics.csv`
and refreshes `training_metrics.png` with loss, learning-rate, and gradient-norm
plots. At the end of every training invocation—including a resumed run—the plot
is regenerated from the CSV and its path is printed for inspection. A
fixed-noise ten-class sample grid is written each epoch so visual progress is
easy to compare. For a quick end-to-end check, set `epochs: 1` and
`max_train_samples: 1024` in the config.

## Generate from descriptions

Each prompt can be repeated to obtain different new images because every output
starts from independent Gaussian noise:

```bash
uv run fashion-fm generate \
  --prompt "a plain t-shirt" \
  --prompt "a sneaker" \
  --prompt "small handbag"
```

The command uses `outputs/fashion-fm/latest.pt` and writes a grid to
`outputs/generated.png`, alongside individual PNGs and a JSON manifest. Pass
`--checkpoint PATH` to use a different model. With no `--prompt`, it generates
four images for each of the ten classes. Aliases include t-shirt/tee,
trousers/pants, pullover/sweater, coat/jacket, sneaker/trainer, bag/handbag, and
ankle boot.

## Test

```bash
uv run pytest
uv run ruff check .
```
