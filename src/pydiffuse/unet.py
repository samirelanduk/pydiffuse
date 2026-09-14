import math

import safetensors
import torch


def unet(
    latent: torch.Tensor,
    noise_level: float,
    conditioning: torch.Tensor,
    model: safetensors.safe_open,
) -> torch.Tensor:
    pass


def _noise_to_t(noise_level) -> int:
    if noise_level <= 0:
        return 0
    if noise_level >= 1:
        return 999
    target = math.log(noise_level / (1 - noise_level))
    start, end = 0.00085**0.5, 0.012**0.5
    image_fraction = 1.0
    previous = -math.inf
    for t in range(1000):
        image_fraction *= 1 - (start + (t / 999) * (end - start)) ** 2
        log_ratio = math.log((1 - image_fraction) / image_fraction)
        if log_ratio >= target:
            return t if log_ratio - target <= target - previous else t - 1
        previous = log_ratio
    return 999
