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
    """Creates a denoised latent image by iteratively running the UNet, using
    the Euler algorithm to step between noise levels."""

    for noise_start, noise_end in pairwise(noise_schedule):
        noise = _predict_noise(latent, noise_start, positive, negative, model, cfg)
        latent = _remove_noise(latent, noise, noise_start, noise_end)
    return latent


def sample_heun(
    positive: torch.Tensor,
    negative: torch.Tensor,
    latent: torch.Tensor,
    model: safetensors.safe_open,
    noise_schedule: list[float],
    cfg: float = 1.0,
) -> torch.Tensor:
    """Creates a denoised latent image by iteratively running the UNet, using
    the Heun algorithm to step between noise levels."""

    for noise_start, noise_end in pairwise(noise_schedule):
        noise = _predict_noise(latent, noise_start, positive, negative, model, cfg)
        trial_latent = _remove_noise(latent, noise, noise_start, noise_end)
        if noise_end == 0:
            latent = trial_latent
            continue
        trial_noise = _predict_noise(
            trial_latent, noise_end, positive, negative, model, cfg
        )
        noise = (noise + trial_noise) / 2
        latent = _remove_noise(latent, noise, noise_start, noise_end)
    return latent


def _predict_noise(
    latent: torch.Tensor,
    noise_level: float,
    positive: torch.Tensor,
    negative: torch.Tensor,
    model: safetensors.safe_open,
    cfg: float,
) -> torch.Tensor:
    """Runs the UNet twice, and returns a final noise prediction that
    prioritises the positive prompt."""

    positive_noise = unet(latent, noise_level, positive, model)
    negative_noise = unet(latent, noise_level, negative, model)
    return negative_noise + cfg * (positive_noise - negative_noise)


def _remove_noise(
    latent: torch.Tensor,
    noise: torch.Tensor,
    noise_level: float,
    next_noise_level: float,
) -> torch.Tensor:
    """Takes a noisy latent and a prediction of its noise, and removes the noise
    proportionally to the noise level we are moving between."""

    scale_noise, scale_image = noise_level**0.5, (1 - noise_level) ** 0.5
    next_scale_noise, next_scale_image = (
        next_noise_level**0.5,
        (1 - next_noise_level) ** 0.5,
    )
    image = (latent - scale_noise * noise) / scale_image
    return next_scale_image * image + next_scale_noise * noise
