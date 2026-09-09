import math

import torch

# The noise levels at the ends of the schedule Stable Diffusion is trained with.
NOISE_LEVEL_MIN = 0.00085
NOISE_LEVEL_MAX = 0.9953399

KARRAS_RHO = 7.0


def noise_tensor(tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
    """Adds a tensor of random noise to a tensor, with the amount determined by
    a noise level between 0 and 1. The tensor will be added to a tensor of
    noise, after each have been scaled by an amount that preserves a variance of
    1 while applying the given noise level."""

    noise = torch.randn_like(tensor)
    scale_noise, scale_original = _scaling_factors(noise_level)
    original = tensor * scale_original
    noise *= scale_noise
    return original + noise


def karras_schedule(
    steps: int,
    noise_level_min: float = NOISE_LEVEL_MIN,
    noise_level_max: float = NOISE_LEVEL_MAX,
    rho: float = KARRAS_RHO,
) -> list[float]:
    """Creates the noise schedule of Karras et al. (2022) - steps + 1 noise
    levels descending to zero, with the ratio raised to the power 1 / rho
    linearly spaced. Raising rho bunches the levels towards the clean end."""

    ratio_min = _noise_level_to_ratio(noise_level_min)
    ratio_max = _noise_level_to_ratio(noise_level_max)
    min_inv_rho = ratio_min ** (1 / rho)
    max_inv_rho = ratio_max ** (1 / rho)
    ratios = [
        (max_inv_rho + (step / max(steps - 1, 1)) * (min_inv_rho - max_inv_rho)) ** rho
        for step in range(steps)
    ]
    return [_ratio_to_noise_level(ratio) for ratio in ratios] + [0.0]


def exponential_schedule(
    steps: int,
    noise_level_min: float = NOISE_LEVEL_MIN,
    noise_level_max: float = NOISE_LEVEL_MAX,
) -> list[float]:
    """Creates an exponential noise schedule - steps + 1 noise levels descending
    to zero, with the log of the ratio linearly spaced, so each step reduces the
    ratio by the same factor."""

    log_min = math.log(_noise_level_to_ratio(noise_level_min))
    log_max = math.log(_noise_level_to_ratio(noise_level_max))
    ratios = [
        math.exp(log_max + (step / max(steps - 1, 1)) * (log_min - log_max))
        for step in range(steps)
    ]
    return [_ratio_to_noise_level(ratio) for ratio in ratios] + [0.0]


def _scaling_factors(noise_level: float) -> tuple[float, float]:
    """For a given noise level, calculates the two scaling factors that the
    noise tensor and the image tensor should be scaled by to preserve a variance
    of 1."""

    return noise_level**0.5, (1 - noise_level) ** 0.5


def _noise_level_to_ratio(noise_level: float) -> float:
    """Converts a noise level to its ratio of noise scaling to image scaling -
    0 for a clean image, 1 where the two are balanced, and unbounded as the
    noise level approaches 1."""

    scale_noise, scale_original = _scaling_factors(noise_level)
    return scale_noise / scale_original


def _ratio_to_noise_level(ratio: float) -> float:
    """Converts a noise-to-signal ratio to the noise level it represents."""

    return ratio**2 / (1 + ratio**2)
