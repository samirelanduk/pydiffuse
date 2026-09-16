from itertools import pairwise

import safetensors
import torch

from pydiffuse.unet import unet


def sample_euler(
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


def sample_heun(
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
        trial_latent = next_scale_image * image + next_scale_noise * noise
        if noise_end == 0:
            latent = trial_latent
            continue
        positive_noise = unet(trial_latent, noise_end, positive, model)
        negative_noise = unet(trial_latent, noise_end, negative, model)
        trial_noise = negative_noise + cfg * (positive_noise - negative_noise)
        noise = (noise + trial_noise) / 2
        image = (latent - scale_noise * noise) / scale_image
        latent = next_scale_image * image + next_scale_noise * noise
    return latent
