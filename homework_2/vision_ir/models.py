"""Frozen vision backbones that turn images into retrieval vectors.

Every backbone shares the same geometry (shorter side resized, then a centered
square crop), so differences come from the network rather than the cropping.
Each encoder returns:

* `final`:  the vector we use for retrieval (CLS token, mean of patch tokens, or
            the classifier's pooled feature, depending on how the model was trained);
* `layers`: the mean of the tokens leaving several blocks, used to ask *where* in
            the network retrieval-ready information lives.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HALF = (0.5, 0.5, 0.5)
WEIGHTS = ("*.json", "model.safetensors")  # skip TF/Flax copies and duplicate formats


@dataclass(frozen=True)
class Backbone:
    name: str
    repo: str
    revision: str
    params_m: float  # parameters used to embed an image, in millions
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    pooling: str  # "cls", "mean", "pooler", or "dit"
    objective: str
    pretraining_data: str
    frames: int = 0  # > 0 for video models: the photo is repeated as a still clip
    files: tuple[str, ...] = WEIGHTS


BACKBONES: dict[str, Backbone] = {
    "ViT": Backbone(
        name="ViT-B/16", params_m=85.8,
        repo="google/vit-base-patch16-224", revision="3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3",
        image_size=224, mean=HALF, std=HALF, pooling="cls",
        objective="classify images into labeled classes",
        pretraining_data="ImageNet-21k labels, then ImageNet-1k labels",
    ),
    "ConvNeXt": Backbone(
        name="ConvNeXt-B", params_m=87.6,
        repo="facebook/convnext-base-224-22k-1k", revision="8640cfcc2392b9fd32947463552ea18b874de9b2",
        image_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD, pooling="pooler",
        objective="classify images into labeled classes (a CNN, not a Transformer)",
        pretraining_data="ImageNet-22k labels, then ImageNet-1k labels",
    ),
    "DINOv2": Backbone(
        name="DINOv2 ViT-B/14", params_m=86.6,
        repo="facebook/dinov2-base", revision="f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        image_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD, pooling="cls",
        objective="match a frozen ViT-g teacher across cropped, flipped, recolored, blurred views",
        pretraining_data="LVD-142M: ImageNet-22k, Google Landmarks, and similar web images; no labels",
    ),
    # The Hub's preprocessor_config.json lists mean/std 0.5, not the ImageNet statistics
    # I-JEPA was trained with (facebookresearch/ijepa src/transforms.py); we use the latter.
    "I-JEPA": Backbone(
        name="I-JEPA ViT-H/14", params_m=630.8,
        repo="facebook/ijepa_vith14_1k", revision="f157467ea509bc356ff9f61fd3c0d840eec5e04e",
        image_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD, pooling="mean",
        objective="predict the embeddings of hidden image blocks (no hand-made augmentations)",
        pretraining_data="ImageNet-1k images, no labels",
    ),
    # Stand-in for H-JEPA, which has no released image encoder (see the notebook).
    # V-JEPA 2 saw ImageNet photos as 16 repeated frames; 2 frames (one temporal token)
    # scored within 0.02 mAP of 16 frames on this task at about 1/6 of the compute.
    "V-JEPA 2": Backbone(
        name="V-JEPA 2 ViT-L/16", params_m=303.9,
        repo="facebook/vjepa2-vitl-fpc64-256", revision="b3c1679b7c34d3255ef3547f27c7b226aefab26f",
        image_size=256, mean=IMAGENET_MEAN, std=IMAGENET_STD, pooling="mean",
        objective="predict the embeddings of hidden space-time regions of videos",
        pretraining_data="VideoMix22M: ~21M videos + 1M ImageNet images, no labels",
        frames=2,
    ),
    "DiT": Backbone(
        # 675M (as in the DiT paper; diffusers reports 750M because it gives every block its
        # own copy of the shared timestep/class embedders) + the 34M VAE encoder.
        name="DiT-XL/2", params_m=675.1 + 34.2,
        repo="facebook/DiT-XL-2-256", revision="eab87f77abd5aef071a632f08807fbaab0b704d0",
        image_size=256, mean=HALF, std=HALF, pooling="dit",
        objective="remove noise from images (a diffusion model that generates images)",
        pretraining_data="ImageNet-1k images + labels, as Stable Diffusion VAE latents",
        files=("model_index.json", "transformer/*", "vae/*"),
    ),
}

# DiT feature settings (block numbers are 1-based, timesteps run from 0 to 999).
DIT_DEFAULTS = {"block": 14, "timestep": 100, "seed": 0}

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess(images: list[Image.Image], backbone: Backbone) -> torch.Tensor:
    """Resize the shorter side, center-crop a square, scale to [0, 1], normalize."""
    size = backbone.image_size
    arrays = []
    for image in images:
        image = image.convert("RGB")
        scale = size / min(image.size)
        resized = image.resize(
            (max(size, round(image.width * scale)), max(size, round(image.height * scale))),
            Image.Resampling.BICUBIC,
        )
        left, top = (resized.width - size) // 2, (resized.height - size) // 2
        arrays.append(np.asarray(resized.crop((left, top, left + size, top + size))))
    pixels = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float() / 255
    mean = torch.tensor(backbone.mean).view(1, 3, 1, 1)
    std = torch.tensor(backbone.std).view(1, 3, 1, 1)
    return (pixels - mean) / std


def layer_indices(num_layers: int, points: int = 8) -> list[int]:
    """Evenly spaced block numbers (1-based) ending at the last block."""
    return sorted({max(1, round(num_layers * (i + 1) / points)) for i in range(points)})


class HFEncoder:
    """ViT, ConvNeXt, DINOv2, I-JEPA, and V-JEPA 2 through Hugging Face `transformers`."""

    def __init__(self, backbone: Backbone, cache_dir: Path, device: torch.device):
        from transformers import AutoModel

        # google/vit-base-patch16-224 ships no pooler weights; skip the random one.
        options = {"add_pooling_layer": False} if backbone.repo.startswith("google/vit") else {}
        self.backbone, self.device = backbone, device
        self.model = AutoModel.from_pretrained(
            backbone.repo, revision=backbone.revision, cache_dir=cache_dir,
            dtype=torch.float32, use_safetensors=True, **options,
        ).to(device).eval()
        self.skip_tokens = 1 if backbone.pooling == "cls" else 0  # leave out the CLS token

    @torch.inference_mode()
    def __call__(self, images: list[Image.Image]) -> dict:
        pixels = preprocess(images, self.backbone).to(self.device)
        if self.backbone.frames:  # (batch, frames, channels, height, width)
            clip = pixels.unsqueeze(1).repeat(1, self.backbone.frames, 1, 1, 1)
            out = self.model(pixel_values_videos=clip, skip_predictor=True, output_hidden_states=True)
        else:
            out = self.model(pixel_values=pixels, output_hidden_states=True)
        if self.backbone.pooling == "cls":
            final = out.last_hidden_state[:, 0]  # CLS token after the final LayerNorm
        elif self.backbone.pooling == "pooler":
            final = out.pooler_output  # ConvNeXt: global average pool + LayerNorm
        else:
            final = out.last_hidden_state.mean(dim=1)  # JEPA models have no CLS token
        blocks = out.hidden_states[1:]  # drop the patch-embedding output
        layers = {}
        for number in layer_indices(len(blocks)):
            state = blocks[number - 1]
            if state.ndim == 4:  # ConvNeXt stage output: (batch, channels, H, W)
                layers[number] = state.mean(dim=(2, 3))
            else:
                layers[number] = state[:, self.skip_tokens:].mean(dim=1)
        return {"final": final, "layers": layers}


class DiTEncoder:
    """Read features from inside a diffusion transformer while it denoises.

    image → VAE latent → add a little noise (timestep t) → DiT with the "null"
    class label → mean of the tokens leaving block `block`. DiT never trained
    to produce embeddings; we borrow the activations it computes while denoising.
    Defaults follow DDAE (Xiang et al., 2023), whose best DiT-XL/2 linear-probe
    features (CIFAR-10, Tiny-ImageNet) came from blocks 13-14 of 28 at t ≈ 90-120.
    """

    NULL_CLASS = 1000  # DiT reserves label 1000 for classifier-free guidance

    def __init__(self, backbone: Backbone, cache_dir: Path, device: torch.device,
                 *, block: int, timestep: int, seed: int):
        from diffusers import AutoencoderKL, DiTTransformer2DModel

        self.backbone, self.device = backbone, device
        self.block, self.timestep, self.seed = block, timestep, seed
        # DiT's weights are only published as PyTorch .bin files. diffusers reads them with
        # torch.load(weights_only=True), which loads tensors only and cannot run code.
        common = {"revision": backbone.revision, "cache_dir": cache_dir, "use_safetensors": False}
        vae = AutoencoderKL.from_pretrained(backbone.repo, subfolder="vae", **common)
        dit = DiTTransformer2DModel.from_pretrained(backbone.repo, subfolder="transformer", **common)
        self.vae, self.dit = vae.to(device).eval(), dit.to(device).eval()
        self.num_blocks = len(self.dit.transformer_blocks)
        if not 1 <= block <= self.num_blocks:
            raise ValueError(f"block must be between 1 and {self.num_blocks}.")
        if not 0 <= timestep < 1000:
            raise ValueError("timestep must be in [0, 999].")
        # DiT's training schedule: 1,000 linear betas from 1e-4 to 0.02.
        betas = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64)
        self.alphas_cumprod = torch.cumprod(1 - betas, dim=0)
        self._captured: dict[int, torch.Tensor] = {}
        for number, module in enumerate(self.dit.transformer_blocks, start=1):
            module.register_forward_hook(self._capture(number))

    def _capture(self, number: int):
        def hook(_module, _inputs, output):
            self._captured[number] = output.mean(dim=1)  # average the 256 latent patches
        return hook

    @torch.inference_mode()
    def __call__(self, images: list[Image.Image]) -> dict:
        pixels = preprocess(images, self.backbone).to(self.device)
        latents = self.vae.encode(pixels).latent_dist.mean * self.vae.config.scaling_factor
        # One fixed noise sample shared by every image: embeddings are reproducible,
        # independent of batch size, and an edited copy gets the same noise as its original.
        generator = torch.Generator().manual_seed(self.seed)
        noise = torch.randn((1, *latents.shape[1:]), generator=generator).to(self.device)
        alpha = self.alphas_cumprod[self.timestep].item()
        noisy = alpha**0.5 * latents + (1 - alpha) ** 0.5 * noise
        batch = len(images)
        self._captured.clear()
        self.dit(
            noisy,
            timestep=torch.full((batch,), self.timestep, device=self.device),
            class_labels=torch.full((batch,), self.NULL_CLASS, device=self.device),
        )
        layers = {i: self._captured[i] for i in layer_indices(self.num_blocks)}
        return {"final": self._captured[self.block], "layers": layers}


def encoder_options(key: str, options: dict) -> dict:
    """Every setting an encoder will use, with defaults filled in (used to name caches)."""
    if BACKBONES[key].pooling != "dit":
        if options:
            raise ValueError(f"{key} takes no extra options.")
        return {}
    unknown = set(options) - set(DIT_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown DiT options: {sorted(unknown)}.")
    return {**DIT_DEFAULTS, **options}


def load_encoder(key: str, cache_dir: Path, device: torch.device, **options):
    """Create the encoder for one entry of BACKBONES."""
    backbone = BACKBONES[key]
    settings = encoder_options(key, options)
    if backbone.pooling == "dit":
        return DiTEncoder(backbone, cache_dir, device, **settings)
    return HFEncoder(backbone, cache_dir, device)
