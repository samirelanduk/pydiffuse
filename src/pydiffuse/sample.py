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
        negative_noise = unet(latent, noise_end, negative, model)
        latent = positive_noise + cfg * (positive_noise - negative_noise)
    return latent
