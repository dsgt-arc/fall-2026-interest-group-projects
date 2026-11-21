"""The tiny class-conditioned U-Net used as a flow velocity field."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel as DiffusersUNet2DModel

BLOCK_OUT_CHANNELS = (16, 32, 64, 64)
NUM_CLASSES = 10
UNCONDITIONAL_CLASS = NUM_CLASSES
UNET_DOWNSAMPLE_FACTOR = 8


class _DataParallelSafeUNet2DModel(DiffusersUNet2DModel):
    """Avoid Diffusers' broken parameter fallback on Python 3.13 replicas."""

    @property
    def dtype(self) -> torch.dtype:
        return self.conv_in.weight.dtype


class TinyConditionalUNet(nn.Module):
    """A class-conditioned adaptation of ImageCLEF's ``unet-tiny``.

    It preserves the reference channel schedule, block layout, GroupNorm size,
    and scale-shift time conditioning. One small embedding table adds the ten
    FashionMNIST classes plus a classifier-free-guidance null class.
    """

    name = "unet-tiny-conditional"

    def __init__(
        self,
        *,
        sample_size: int = 28,
        timesteps: int = 1000,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")

        self.sample_size = sample_size

        # Four levels contain three 2x downsamplings. Diffusers' plain
        # UNet2DModel needs the working size divisible by 8 for skip shapes to
        # agree, while FashionMNIST itself must remain 28x28.
        remainder = sample_size % UNET_DOWNSAMPLE_FACTOR
        padding = (UNET_DOWNSAMPLE_FACTOR - remainder) % UNET_DOWNSAMPLE_FACTOR
        self.internal_sample_size = sample_size + padding
        self.timesteps = timesteps

        # Add one extra class embedding for unconditional guidance.
        embedding_count = NUM_CLASSES + 1
        self.model = _DataParallelSafeUNet2DModel(
            sample_size=self.internal_sample_size,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=BLOCK_OUT_CHANNELS,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            norm_num_groups=16,
            resnet_time_scale_shift="scale_shift",
            num_class_embeds=embedding_count,
        )
        if gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

    def prepare_timesteps(self, t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        """Map normalized flow time in [0, 1] to Diffusers integer timesteps."""

        if t.ndim == 0:
            t = t.expand(batch_size)
        if t.shape != (batch_size,):
            raise ValueError(f"expected {batch_size} timesteps, got shape {tuple(t.shape)}")

        # Scale normalized flow time to the integer range expected by Diffusers.
        normalized_time = t.to(device=device, dtype=torch.float32)
        maximum_timestep = self.timesteps - 1
        scaled_time = normalized_time * maximum_timestep
        integer_time = scaled_time.long()
        return integer_time.clamp(0, maximum_timestep)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Predict the conditional flow velocity for a noisy image batch."""

        batch_size = x.shape[0]
        expected_label_shape = (batch_size,)
        if labels.shape != expected_label_shape:
            raise ValueError(f"expected {batch_size} labels, got shape {tuple(labels.shape)}")

        # Prepare the conditioning values before running the U-Net.
        t_model = self.prepare_timesteps(t, batch_size, x.device)
        labels = labels.to(device=x.device, dtype=torch.long)

        # Pad 28x28 inputs to 32x32 so all U-Net skip connections line up.
        padding = self.internal_sample_size - self.sample_size
        pad_before = padding // 2
        pad_after = padding - pad_before
        if padding:
            x = F.pad(x, (pad_before, pad_after, pad_before, pad_after))

        output = self.model(x, t_model, class_labels=labels).sample

        # Remove the temporary border and return to FashionMNIST resolution.
        if padding:
            crop_end = pad_before + self.sample_size
            output = output[..., pad_before:crop_end, pad_before:crop_end]
        return output

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable parameters."""

        trainable_parameters = (parameter for parameter in self.parameters() if parameter.requires_grad)
        return sum(parameter.numel() for parameter in trainable_parameters)

    def metadata(self) -> dict[str, Any]:
        """Describe the architecture in a checkpoint-friendly form."""

        return {
            "name": self.name,
            "sample_size": self.sample_size,
            "internal_sample_size": self.internal_sample_size,
            "timesteps": self.timesteps,
            "block_out_channels": list(BLOCK_OUT_CHANNELS),
            "num_classes": NUM_CLASSES,
            "unconditional_class": UNCONDITIONAL_CLASS,
            "parameter_count": self.parameter_count,
        }
