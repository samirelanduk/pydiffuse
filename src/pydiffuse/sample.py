from itertools import pairwise

import safetensors
import torch

from pydiffuse.unet import unet


def sample(
    positive: torch.Tensor,
    negative: torch.Tensor,
    latent: torch.Tensor,
    model: safetensors.safe_open,
    noise_schedule: list[float],
    cfg: float = 1.0,
) -> torch.Tensor:
    for noise_start, noise_end in pairwise(noise_schedule):
        positive_noise = unet(latent, noise_start, positive, model)
        negative_noise = unet(latent, noise_start, negative, model)
        noise = negative_noise + cfg * (positive_noise - negative_noise)
        scale_noise, scale_image = noise_start**0.5, (1 - noise_start) ** 0.5
        image = (latent - scale_noise * noise) / scale_image
        next_scale_noise, next_scale_image = noise_end**0.5, (1 - noise_end) ** 0.5
        latent = next_scale_image * image + next_scale_noise * noise
    return latent
