"""Conditional flow-matching training targets and ODE sampling."""

import torch
import torch.nn as nn

from fashion_fm.model import UNCONDITIONAL_CLASS


def make_flow_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    cfg_dropout: float = 0.1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create linear conditional flow-matching inputs and targets.

    The path is ``x_t = (1-t) * noise + t * image`` and its exact velocity is
    ``image - noise``. Some labels are replaced by the null class to train
    classifier-free guidance.
    """

    if not 0.0 <= cfg_dropout <= 1.0:
        raise ValueError("cfg_dropout must be between 0 and 1")

    # Pair each training image with fresh Gaussian noise and a random flow time.
    batch_size = images.shape[0]
    noise = torch.randn(
        images.shape,
        device=images.device,
        dtype=images.dtype,
        generator=generator,
    )
    t = torch.rand(
        batch_size,
        device=images.device,
        dtype=images.dtype,
        generator=generator,
    )

    # Interpolate from noise at time zero to the real image at time one.
    t_image = t[:, None, None, None]
    noise_part = (1.0 - t_image) * noise
    image_part = t_image * images
    x_t = noise_part + image_part
    target_velocity = images - noise

    # Drop some class labels so the same model learns unconditional predictions.
    conditioned_labels = labels.to(device=images.device, dtype=torch.long).clone()
    if cfg_dropout:
        random_values = torch.rand(batch_size, device=images.device, generator=generator)
        dropped_labels = random_values < cfg_dropout
        conditioned_labels[dropped_labels] = UNCONDITIONAL_CLASS
    return x_t, t, target_velocity, conditioned_labels


def _guided_velocity(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    if guidance_scale == 1.0:
        return model(x, t, labels)

    # Evaluate conditional and unconditional batches together for efficiency.
    unconditional = torch.full_like(labels, UNCONDITIONAL_CLASS)
    x_both = torch.cat((x, x), dim=0)
    t_both = torch.cat((t, t), dim=0)
    labels_both = torch.cat((unconditional, labels), dim=0)
    velocities = model(x_both, t_both, labels_both)
    velocity_unconditional, velocity_conditional = velocities.chunk(2)

    # Move away from the unconditional prediction toward the requested class.
    guidance = velocity_conditional - velocity_unconditional
    return velocity_unconditional + guidance_scale * guidance


@torch.inference_mode()
def sample(
    model: nn.Module,
    labels: torch.Tensor,
    *,
    steps: int = 40,
    guidance_scale: float = 2.0,
    seed: int | None = None,
    initial_noise: torch.Tensor | None = None,
    method: str = "heun",
) -> torch.Tensor:
    """Integrate the learned velocity field from Gaussian noise to images."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if method not in {"euler", "heun"}:
        raise ValueError("method must be 'euler' or 'heun'")

    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    labels = labels.to(device=device, dtype=torch.long)

    # Start from provided noise when testing, or sample new noise for generation.
    if initial_noise is None:
        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)
        size = int(getattr(model, "sample_size", 28))
        noise_shape = (labels.shape[0], 1, size, size)
        x = torch.randn(noise_shape, device=device, dtype=dtype, generator=rng)
    else:
        x = initial_noise.to(device=device, dtype=dtype).clone()

    # Follow the learned velocity field from noise toward a generated image.
    dt = 1.0 / steps
    for step in range(steps):
        t_value = step / steps
        time_shape = (labels.shape[0],)
        t = torch.full(time_shape, t_value, device=device, dtype=torch.float32)
        velocity = _guided_velocity(model, x, t, labels, guidance_scale)
        if method == "euler":
            x = x + dt * velocity
            continue

        # Heun's method averages velocities at the start and end of the step.
        prediction = x + dt * velocity
        next_t_value = (step + 1) / steps
        t_next = torch.full(time_shape, next_t_value, device=device, dtype=torch.float32)
        next_velocity = _guided_velocity(model, prediction, t_next, labels, guidance_scale)
        average_velocity = 0.5 * (velocity + next_velocity)
        x = x + dt * average_velocity
    return x.clamp(-1.0, 1.0)
